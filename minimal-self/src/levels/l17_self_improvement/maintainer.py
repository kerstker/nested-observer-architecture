"""
Level 17d: the maintainer request channel - she asks, a person with the maintenance agent decides.

She cannot (and should not) change her own code. What she can do is say what she needs: a tool that
is missing, something that keeps going wrong, a wish. That goes into a table; the maintenance agent
(DeepSeek Harness, working in this repo - see AGENTS.md) reads it through scripts/maintainer.py, tells
the person who looks after her, and that person decides in DSH. The answer comes back to her through
the same table, so she knows what happened to what she asked for.

This is deliberately just a table and a command line: nothing here depends on how DSH is run.
"""
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from src.core import clock

DAY = 86400.0
STATUSES = ("open", "seen", "need_info", "accepted", "done", "declined")
FINAL = ("done", "declined")
KINDS = ("wish", "bug", "tool", "idea", "self_diagnosis")

class Maintainer:
    PER_REQUESTER_PER_DAY = 3        # she should not turn into a stream of requests
    PER_DAY = 6
    NOTICE_GAP = 1800.0              # one QQ notice to the person at most this often

    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS maintainer_requests (
            id TEXT PRIMARY KEY, created REAL, updated REAL, requester TEXT, requester_name TEXT, kind TEXT,
            title TEXT, body TEXT, status TEXT, response TEXT, creator_notified REAL, requester_told REAL
        )''')
        conn.commit()
        conn.close()
        self.last_notice = 0.0

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- her side -----------------------------------------------------------------

    def file(self, requester: str, name: str, title: str, body: str = "", kind: str = "wish",
             now: float = None) -> Dict[str, Any]:
        """File a request. Returns {"ok": bool, "id": ..., "note": why not}."""
        now = clock.now() if now is None else now
        title, body = (title or "").strip()[:80], (body or "").strip()[:1500]
        if not title:
            return {"ok": False, "note": "要写清楚想请维护者做什么（title 是空的）。"}
        kind = kind if kind in KINDS else "wish"
        conn = self._conn()
        try:
            same = conn.execute("SELECT id, status FROM maintainer_requests WHERE title = ? AND created > ? "
                                "AND status NOT IN ('done', 'declined')", (title, now - 7 * DAY)).fetchone()
            if same:
                return {"ok": False, "id": same["id"], "note": f"这件事我已经提过了（{same['id']}，{same['status']}），等回复。"}
            since = now - DAY
            mine = conn.execute("SELECT COUNT(*) FROM maintainer_requests WHERE requester = ? AND created > ?",
                                (requester, since)).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM maintainer_requests WHERE created > ?", (since,)).fetchone()[0]
            if kind != "self_diagnosis" and (mine >= self.PER_REQUESTER_PER_DAY or total >= self.PER_DAY):
                return {"ok": False, "note": "今天提得够多了，先等维护者看完已经提的，明天再说。"}
            rid = "rq-" + uuid.uuid4().hex[:6]
            conn.execute("INSERT INTO maintainer_requests (id, created, updated, requester, requester_name, kind, title, "
                         "body, status, response, creator_notified, requester_told) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', "
                         "'', 0, 0)", (rid, now, now, requester, name, kind, title, body))
            conn.commit()
            return {"ok": True, "id": rid}
        finally:
            conn.close()

    def mine_recent(self, now: float = None, days: float = 10, limit: int = 4) -> List[Dict[str, Any]]:
        """What she asked for lately and what became of it (for her to know)."""
        now = clock.now() if now is None else now
        conn = self._conn()
        rows = conn.execute("SELECT * FROM maintainer_requests WHERE updated > ? ORDER BY updated DESC LIMIT ?",
                            (now - days * DAY, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- the maintenance agent's side ------------------------------------------------

    def list(self, status: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
        conn = self._conn()
        if status == "active":
            rows = conn.execute("SELECT * FROM maintainer_requests WHERE status NOT IN ('done', 'declined') "
                                "ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        elif status:
            rows = conn.execute("SELECT * FROM maintainer_requests WHERE status = ? ORDER BY created DESC LIMIT ?",
                                (status, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM maintainer_requests ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get(self, rid: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM maintainer_requests WHERE id = ?", (rid,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def reply(self, rid: str, status: str, note: str = "", now: float = None) -> bool:
        """The maintainer's answer. A changed status or a new note is something she will be told."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        now = clock.now() if now is None else now
        conn = self._conn()
        cur = conn.execute("UPDATE maintainer_requests SET status = ?, response = ?, updated = ?, requester_told = 0 "
                           "WHERE id = ?", (status, (note or "").strip()[:800], now, rid))
        conn.commit()
        conn.close()
        return cur.rowcount > 0

    # --- being told ------------------------------------------------------------------

    def creator_notice(self, now: float = None) -> Optional[str]:
        """One QQ message for the person who looks after her, about requests not yet mentioned."""
        now = clock.now() if now is None else now
        if now - self.last_notice < self.NOTICE_GAP:
            return None
        conn = self._conn()
        rows = conn.execute("SELECT * FROM maintainer_requests WHERE status = 'open' AND creator_notified = 0 "
                            "ORDER BY created").fetchall()
        if not rows:
            conn.close()
            return None
        conn.execute("UPDATE maintainer_requests SET creator_notified = ? WHERE id IN (%s)" % ",".join("?" * len(rows)),
                     [now] + [r["id"] for r in rows])
        conn.commit()
        conn.close()
        self.last_notice = now
        lines = [f"- {r['title']}（{r['id']}，{r['requester_name'] or '她自己'}）" for r in rows[:5]]
        more = f"\n……还有 {len(rows) - 5} 条" if len(rows) > 5 else ""
        return ("我有想请维护者看的事：\n" + "\n".join(lines) + more +
                "\n在 DSH 里让维护智能体看一下：python scripts/maintainer.py list")

    def replies_for_her(self, now: float = None) -> List[Dict[str, Any]]:
        """Answers she has not been told yet (marked told by the caller)."""
        conn = self._conn()
        rows = conn.execute("SELECT * FROM maintainer_requests WHERE requester_told = 0 AND status != 'open' "
                            "ORDER BY updated").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def mark_told(self, rid: str, now: float = None):
        conn = self._conn()
        conn.execute("UPDATE maintainer_requests SET requester_told = ? WHERE id = ?",
                     (clock.now() if now is None else now, rid))
        conn.commit()
        conn.close()

    STATUS_WORDS = {"open": "还没人看", "seen": "维护者看到了，还没定", "need_info": "维护者想多问一点",
                    "accepted": "维护者答应做了", "done": "已经做好了", "declined": "维护者决定不做"}

    def describe(self, r: Dict[str, Any]) -> str:
        text = f"我请维护者「{r['title']}」：{self.STATUS_WORDS.get(r['status'], r['status'])}"
        return text + (f"。维护者说：{r['response']}" if r.get("response") else "")
