"""
Backups of her memory: her whole life is one SQLite file, so it is copied every day, checked,
and kept in a rotation.

- Online: SQLite's own backup API copies a consistent snapshot while she keeps living (no stopping,
  no half-written pages). The copy is made next to its final name and only renamed into place once it
  has been checked, so a crash never leaves a backup that looks fine and is not.
- Checked: every backup is opened again - integrity check, her core tables are there, and the row
  counts are what the live database had when the copy was made (tables that only grow cannot have
  fewer rows than before the copy started). `restore_check` goes further: it opens a copy with her own
  memory classes and recalls something, which is what a real restore would depend on.
- Rotated: every backup of the last `keep_daily` days is kept (at least the newest of each day),
  then the newest of each of the last `keep_weekly` weeks. Nothing that is the newest verified
  backup is ever removed.
- Optionally copied to a second place (`extra_dirs`, e.g. another drive): one disk is one failure.

Names carry their time (memory-YYYYmmdd-HHMMSS.sqlite), so the rotation needs no state of its own.
"""
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

NAME = re.compile(r"^memory-(\d{8}-\d{6})\.sqlite$")
# Tables that only ever gain rows (nothing deletes from them), so a copy can never have fewer
# than the live database had before the copy began
GROWS = ("episodes", "interactions", "users", "trait_changes", "state_causes")
CORE = ("episodes", "interactions", "users", "semantic_facts", "persona_traits")

def _stamp(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y%m%d-%H%M%S")

def _parse(name: str) -> Optional[float]:
    m = NAME.match(name)
    return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp() if m else None

class BackupManager:
    def __init__(self, db_path: str, folder: str = None, keep_daily: int = 7, keep_weekly: int = 4,
                 interval: float = 24 * 3600, extra_dirs: Iterable[str] = ()):
        self.db_path = os.path.abspath(db_path)
        self.folder = os.path.abspath(folder or os.path.join(os.path.dirname(self.db_path), "backups"))
        self.keep_daily, self.keep_weekly, self.interval = keep_daily, keep_weekly, interval
        self.extra_dirs = [os.path.abspath(d) for d in extra_dirs if d]
        self.last_result: Optional[Dict[str, Any]] = None

    # --- looking ------------------------------------------------------------------

    def list(self, folder: str = None) -> List[Dict[str, Any]]:
        folder = folder or self.folder
        out = []
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                t = _parse(name)
                if t is not None:
                    path = os.path.join(folder, name)
                    out.append({"name": name, "path": path, "t": t, "bytes": os.path.getsize(path)})
        return sorted(out, key=lambda b: b["t"])

    def due(self, now: float = None) -> bool:
        now = time.time() if now is None else now
        backups = self.list()
        return not backups or now - backups[-1]["t"] >= self.interval

    def status(self, now: float = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        backups = self.list()
        return {"count": len(backups), "bytes": sum(b["bytes"] for b in backups),
                "latest": backups[-1]["name"] if backups else None,
                "age_hours": round((now - backups[-1]["t"]) / 3600, 1) if backups else None,
                "due": self.due(now), "folder": self.folder, "copies": len(self.extra_dirs),
                "last_result": self.last_result}

    # --- checking -----------------------------------------------------------------

    @staticmethod
    def _counts(conn: sqlite3.Connection, tables: Iterable[str]) -> Dict[str, Optional[int]]:
        out = {}
        for t in tables:
            try:
                out[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except sqlite3.OperationalError:
                out[t] = None
        return out

    def verify(self, path: str, before: Dict[str, Optional[int]] = None, after: Dict[str, Optional[int]] = None) -> Dict[str, Any]:
        """Open a backup again and check it. `before`/`after`: live row counts around the copy."""
        problems: List[str] = []
        counts: Dict[str, Optional[int]] = {}
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                check = conn.execute("PRAGMA integrity_check").fetchone()[0]
                if check != "ok":
                    problems.append(f"integrity_check: {check}")
                counts = self._counts(conn, set(CORE) | set(GROWS))
            finally:
                conn.close()
        except sqlite3.Error as e:
            return {"ok": False, "problems": [f"cannot open: {e}"], "counts": {}}
        for t in CORE:
            if counts.get(t) is None and (before or {}).get(t) is not None:
                problems.append(f"table {t} is missing")
        for t in GROWS:
            lo, hi = (before or {}).get(t), (after or {}).get(t)
            n = counts.get(t)
            if n is None:
                continue
            if lo is not None and n < lo:
                problems.append(f"{t}: {n} rows, the live database already had {lo}")
            if hi is not None and n > hi:
                problems.append(f"{t}: {n} rows, more than the live database has now ({hi})")
        return {"ok": not problems, "problems": problems, "counts": counts}

    def restore_check(self, path: str) -> Dict[str, Any]:
        """What a real restore depends on: copy the backup somewhere else and use it as her memory."""
        work = tempfile.mkdtemp(prefix="noa_restore_")
        try:
            copy = os.path.join(work, "memory.sqlite")
            shutil.copy2(path, copy)
            from src.levels.l2_memory.embedding import Embedder
            from src.levels.l2_memory.layers import EpisodicMemory
            from src.levels.l2_memory.storage.database import MemoryDatabase
            db = MemoryDatabase(copy)
            traits = db.get_persona_traits()
            episodes = EpisodicMemory(copy, Embedder())
            sample = (episodes.recent(1) or [{"content": ""}])[0]["content"]
            found = episodes.recall(sample[:40], k=1) if sample.strip() else []
            recall_works = bool(found) or not sample.strip()
            return {"ok": bool(traits) and recall_works, "traits": len(traits), "episodes": episodes.count(),
                    "recall_works": recall_works}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # --- making -------------------------------------------------------------------

    def run(self, now: float = None, reason: str = "daily") -> Dict[str, Any]:
        """One backup, checked, copied elsewhere, and the rotation applied. Never raises."""
        now = time.time() if now is None else now
        result: Dict[str, Any] = {"t": now, "reason": reason, "ok": False}
        tmp = None
        try:
            os.makedirs(self.folder, exist_ok=True)
            free = shutil.disk_usage(self.folder).free
            need = 2 * os.path.getsize(self.db_path)
            if free < need:
                raise OSError(f"not enough free space ({free // 2**20} MB free, {need // 2**20} MB needed)")
            name = f"memory-{_stamp(now)}.sqlite"
            final = os.path.join(self.folder, name)
            tmp = final + ".partial"
            src = sqlite3.connect(self.db_path, timeout=30)
            try:
                before = self._counts(src, GROWS)
                dst = sqlite3.connect(tmp)
                try:
                    src.backup(dst)                        # one consistent snapshot, online
                finally:
                    dst.close()
                after = self._counts(src, GROWS)
            finally:
                src.close()
            check = self.verify(tmp, before, after)
            result.update(name=name, bytes=os.path.getsize(tmp), problems=check["problems"], counts=check["counts"])
            if not check["ok"]:
                raise ValueError("; ".join(check["problems"]))
            os.replace(tmp, final)
            tmp = None
            result["copies"] = self._copy_elsewhere(final)
            result["removed"] = self.rotate(now)
            result["ok"] = True
            logger.info(f"[Backup] {name} ({result['bytes'] // 1024} KB), checked; kept {len(self.list())}"
                        + (f", removed {len(result['removed'])}" if result["removed"] else ""))
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[Backup] failed: {e}")
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        self.last_result = result
        return result

    def run_if_due(self, now: float = None) -> Optional[Dict[str, Any]]:
        return self.run(now) if self.due(now) else None

    def _copy_elsewhere(self, path: str) -> int:
        done = 0
        for d in self.extra_dirs:
            try:
                os.makedirs(d, exist_ok=True)
                target = os.path.join(d, os.path.basename(path))
                shutil.copy2(path, target + ".partial")
                os.replace(target + ".partial", target)
                self.rotate_in(d)
                done += 1
            except OSError as e:
                logger.error(f"[Backup] copy to {d} failed: {e}")
        return done

    # --- rotating -----------------------------------------------------------------

    def _keep(self, backups: List[Dict[str, Any]], now: float) -> set:
        keep = {backups[-1]["name"]} if backups else set()
        by_day, by_week = {}, {}
        for b in backups:                                  # newest of each day / week wins
            d = datetime.fromtimestamp(b["t"])
            by_day[d.strftime("%Y%m%d")] = b
            by_week[d.strftime("%G-%V")] = b
        days = sorted(by_day)[-self.keep_daily:]
        weeks = sorted(by_week)[-self.keep_weekly:]
        keep |= {by_day[k]["name"] for k in days} | {by_week[k]["name"] for k in weeks}
        return keep

    def rotate_in(self, folder: str, now: float = None) -> List[str]:
        now = time.time() if now is None else now
        backups = self.list(folder)
        keep = self._keep(backups, now)
        removed = []
        for b in backups:
            if b["name"] not in keep:
                try:
                    os.remove(b["path"])
                    removed.append(b["name"])
                except OSError as e:
                    logger.error(f"[Backup] could not remove {b['name']}: {e}")
        return removed

    def rotate(self, now: float = None) -> List[str]:
        return self.rotate_in(self.folder, now)
