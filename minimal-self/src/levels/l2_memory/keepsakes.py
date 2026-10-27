"""
Level 2f: her own little room - the things she chose to keep.

Memory holds what happened; this holds *things*: a picture someone sent that she liked, a file she wants to
keep after the inbox forgets it (14 days), a line or a link she wants to come back to. She decides what goes
in (the `keep` tool); nobody else can put things here or take them out.

- The list lives in her database (`keepsakes` table, so it is in every memory backup); the bytes live next to
  it in data/keepsakes/, one file per content hash (the same picture kept twice is stored once).
  BackupManager mirrors that folder into backups/keepsakes (`mirror_files`).
- Pictures are kept as they came (up to MAX_FILE), so she has the original, not the chat-sized copy.
- Forgetting a thing marks it gone; the file stays until no kept thing points at it any more.
- Limits keep it a room and not a warehouse: MAX_ITEMS things, MAX_TOTAL bytes, MAX_PER_DAY new ones a day.
"""
import hashlib
import io
import os
import sqlite3
from typing import Any, Dict, List, Optional

from src.core import clock

KINDS = ("picture", "file", "note", "link")
MIME_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}

class Keepsakes:
    MAX_ITEMS = 2000
    MAX_TOTAL = 2 * 1024 ** 3          # 2 GB of files
    MAX_FILE = 25 * 1024 * 1024
    MAX_PER_DAY = 60

    def __init__(self, db_path: str, folder: str):
        self.db_path = db_path
        self.folder = folder
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS keepsakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT,              -- picture / file / note / link
            title TEXT,             -- what she calls it
            note TEXT,              -- why she kept it / what it is to her
            tags TEXT,              -- comma separated
            favorite INTEGER DEFAULT 0,
            content TEXT,           -- the text of a note, the url of a link
            file TEXT,              -- file name in the folder (sha256 + extension)
            name TEXT,              -- original file name
            mime TEXT,
            size INTEGER DEFAULT 0,
            source TEXT,            -- who / which chat it came from
            created REAL,
            last_seen REAL,
            seen INTEGER DEFAULT 0,
            removed REAL
        )''')
        conn.commit()
        conn.close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- keeping ------------------------------------------------------------------

    def _room_for(self, conn, size: int, now: float) -> Optional[str]:
        """Why this cannot be kept now (None = there is room)."""
        items, total = conn.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM keepsakes WHERE removed IS NULL").fetchone()
        today = conn.execute("SELECT COUNT(*) FROM keepsakes WHERE created > ?", (now - 86400,)).fetchone()[0]
        if items >= self.MAX_ITEMS:
            return f"我的小屋已经放了 {items} 样东西，满了；先扔掉一些不要的（forget）再放。"
        if total + size > self.MAX_TOTAL:
            return "我的小屋放不下这么大的东西了，先扔掉一些不要的再放。"
        if today >= self.MAX_PER_DAY:
            return "今天已经收了很多东西了，明天再收吧。"
        if size > self.MAX_FILE:
            return f"这个太大了（{size / 1024 / 1024:.0f} MB），我最多收 {self.MAX_FILE // 1024 // 1024} MB 的。"
        return None

    def _store(self, data: bytes, ext: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        name = digest[:32] + (ext or "")
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, name)
        if not os.path.exists(path):
            with open(path + ".partial", "wb") as f:
                f.write(data)
            os.replace(path + ".partial", path)
        return name

    def keep(self, kind: str, title: str = "", note: str = "", tags: str = "", data: bytes = None,
             mime: str = "", name: str = "", content: str = "", source: str = "", favorite: bool = False,
             now: float = None) -> Dict[str, Any]:
        """Put a thing in her room. Returns {"ok": True, "id", "again"} or {"ok": False, "note": why not}."""
        now = clock.now() if now is None else now
        kind = kind if kind in KINDS else ("picture" if (mime or "").startswith("image/") else "file" if data else "note")
        title = (title or name or content or kind).strip()[:60]
        if kind in ("picture", "file") and not data:
            return {"ok": False, "note": "没有拿到要收的东西本身。"}
        if kind in ("note", "link") and not (content or "").strip():
            return {"ok": False, "note": "要记下的内容是空的。"}
        conn = self._conn()
        try:
            file_name = None
            if data:
                ext = MIME_EXT.get(mime) or os.path.splitext(name or "")[1][:10].lower()
                digest = hashlib.sha256(data).hexdigest()[:32]
                same = conn.execute("SELECT id, title FROM keepsakes WHERE file LIKE ? AND removed IS NULL",
                                    (digest + "%",)).fetchone()
                if same:
                    return {"ok": True, "id": same["id"], "again": True, "title": same["title"]}
            else:
                same = conn.execute("SELECT id, title FROM keepsakes WHERE content = ? AND removed IS NULL",
                                    (content.strip(),)).fetchone()
                if same:
                    return {"ok": True, "id": same["id"], "again": True, "title": same["title"]}
            why = self._room_for(conn, len(data or b""), now)
            if why:
                return {"ok": False, "note": why}
            if data:
                file_name = self._store(data, ext)
            cur = conn.execute(
                "INSERT INTO keepsakes (kind, title, note, tags, favorite, content, file, name, mime, size, source, "
                "created, last_seen, seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (kind, title, (note or "").strip()[:300], (tags or "").strip()[:120], int(bool(favorite)),
                 (content or "").strip()[:4000], file_name, (name or "")[:100], mime or "", len(data or b""),
                 (source or "")[:80], now, now))
            conn.commit()
            return {"ok": True, "id": cur.lastrowid, "again": False, "title": title}
        finally:
            conn.close()

    # --- looking ------------------------------------------------------------------

    @staticmethod
    def _row(r) -> Dict[str, Any]:
        d = dict(r)
        d["favorite"] = bool(d.get("favorite"))
        return d

    def get(self, item_id: int) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        try:
            r = conn.execute("SELECT * FROM keepsakes WHERE id = ? AND removed IS NULL", (int(item_id),)).fetchone()
            return self._row(r) if r else None
        finally:
            conn.close()

    def list(self, query: str = "", kind: str = "", favorites: bool = False, limit: int = 30,
             offset: int = 0) -> List[Dict[str, Any]]:
        sql, args = "SELECT * FROM keepsakes WHERE removed IS NULL", []
        if kind in KINDS:
            sql += " AND kind = ?"
            args.append(kind)
        if favorites:
            sql += " AND favorite = 1"
        for word in (query or "").split()[:5]:
            sql += " AND (title LIKE ? OR note LIKE ? OR tags LIKE ? OR content LIKE ? OR name LIKE ? OR source LIKE ?)"
            args += [f"%{word}%"] * 6
        sql += " ORDER BY favorite DESC, created DESC LIMIT ? OFFSET ?"
        args += [max(1, min(int(limit), 200)), max(0, int(offset))]
        conn = self._conn()
        try:
            return [self._row(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def path(self, item: Dict[str, Any]) -> Optional[str]:
        if not item or not item.get("file"):
            return None
        p = os.path.join(self.folder, os.path.basename(item["file"]))
        return p if os.path.isfile(p) else None

    def data(self, item: Dict[str, Any]) -> Optional[bytes]:
        p = self.path(item)
        if not p:
            return None
        with open(p, "rb") as f:
            return f.read()

    def picture(self, item: Dict[str, Any], side: int = 1280) -> Optional[Dict[str, Any]]:
        """A kept picture as something a model can look at: {"data": jpeg bytes, "mime"} (None if not a picture)."""
        if not item or item.get("kind") != "picture":
            return None
        raw = self.data(item)
        if not raw:
            return None
        try:
            from PIL import Image
            image = Image.open(io.BytesIO(raw))
            if getattr(image, "is_animated", False):
                image.seek(0)
            image = image.convert("RGB")
            if max(image.size) > side:
                image.thumbnail((side, side))
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=85)
            return {"data": out.getvalue(), "mime": "image/jpeg"}
        except Exception:
            return {"data": raw, "mime": item.get("mime") or "image/jpeg"} if len(raw) < 4 * 1024 * 1024 else None

    def thumbnail(self, item: Dict[str, Any], side: int = 360) -> Optional[bytes]:
        pic = self.picture(item, side)
        return pic["data"] if pic else None

    def touch(self, item_id: int, now: float = None):
        """She looked at it again."""
        conn = self._conn()
        try:
            conn.execute("UPDATE keepsakes SET seen = seen + 1, last_seen = ? WHERE id = ?",
                         (clock.now() if now is None else now, int(item_id)))
            conn.commit()
        finally:
            conn.close()

    # --- changing her mind ------------------------------------------------------------

    def update(self, item_id: int, title: str = None, note: str = None, tags: str = None,
               favorite: bool = None) -> bool:
        fields, args = [], []
        for col, val, n in (("title", title, 60), ("note", note, 300), ("tags", tags, 120)):
            if val is not None:
                fields.append(f"{col} = ?")
                args.append(str(val).strip()[:n])
        if favorite is not None:
            fields.append("favorite = ?")
            args.append(int(bool(favorite)))
        if not fields:
            return False
        conn = self._conn()
        try:
            cur = conn.execute(f"UPDATE keepsakes SET {', '.join(fields)} WHERE id = ? AND removed IS NULL",
                               args + [int(item_id)])
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def forget(self, item_id: int, now: float = None) -> Optional[Dict[str, Any]]:
        """Take a thing out of her room. The file goes too once nothing else kept points at it."""
        item = self.get(item_id)
        if not item:
            return None
        conn = self._conn()
        try:
            conn.execute("UPDATE keepsakes SET removed = ? WHERE id = ?", (clock.now() if now is None else now, item["id"]))
            conn.commit()
            if item.get("file"):
                still = conn.execute("SELECT COUNT(*) FROM keepsakes WHERE file = ? AND removed IS NULL",
                                     (item["file"],)).fetchone()[0]
                p = self.path(item)
                if not still and p:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            return item
        finally:
            conn.close()

    def stats(self) -> Dict[str, Any]:
        conn = self._conn()
        try:
            rows = conn.execute("SELECT kind, COUNT(*), COALESCE(SUM(size), 0) FROM keepsakes WHERE removed IS NULL "
                                "GROUP BY kind").fetchall()
            fav = conn.execute("SELECT COUNT(*) FROM keepsakes WHERE removed IS NULL AND favorite = 1").fetchone()[0]
        finally:
            conn.close()
        kinds = {k: n for k, n, _ in rows}
        return {"items": sum(kinds.values()), "bytes": sum(s for _, _, s in rows), "kinds": kinds, "favorites": fav,
                "max_items": self.MAX_ITEMS, "max_bytes": self.MAX_TOTAL}

    def describe(self, limit: int = 5) -> str:
        """One line for her prompt: what is in her room (empty string when nothing is)."""
        s = self.stats()
        if not s["items"]:
            return ""
        words = {"picture": "张图", "file": "个文件", "note": "条摘记", "link": "个链接"}
        parts = [f"{n} {words.get(k, '样')}" for k, n in s["kinds"].items()]
        fav = [i["title"] for i in self.list(favorites=True, limit=limit)]
        return "、".join(parts) + (f"；最喜欢的有：{'、'.join(fav)}" if fav else "")
