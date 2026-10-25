"""
Level 8c: tasks - things she has said she will do, kept as work she actually carries out.

A promise used to be only a goal: a line of text she was reminded of, kept (if at all) when a
proactive chat happened to come round, and never in a group. A task is the executable half of it:
what to do, for whom, where to answer, and when. The heartbeat runs tasks that are due:
- remind: at the time, she reminds the person (an alarm she set, so it rings even if she sleeps);
- do: when it is due and she is awake, she does the work with her tools and tells them the result.
Every task is linked to a goal, so keeping or breaking it is lived like any other goal.
"""
import sqlite3
from typing import Any, Dict, List, Optional

KINDS = ("remind", "do")
MAX_PENDING_PER_PERSON = 5
MAX_PENDING = 30
MAX_AHEAD = 30 * 86400
MAX_ATTEMPTS = 3
RETRY_AFTER = 600

class TaskBook:
    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS agent_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, due REAL, kind TEXT, what TEXT,
            user_id TEXT, nickname TEXT, group_id TEXT, origin TEXT, status TEXT, result TEXT,
            attempts INTEGER DEFAULT 0, next_try REAL, goal_id INTEGER, done_at REAL
        )''')
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, kind: str, what: str, due: float, now: float, user_id: str, nickname: str = "",
            group_id: Optional[str] = None, origin: str = "tool", goal_id: Optional[int] = None) -> int:
        """Raises ValueError with a reason she can repeat when the task cannot be taken on."""
        what = (what or "").strip()
        if kind not in KINDS:
            raise ValueError(f"不认识的任务类型 {kind}（只有 remind / do）")
        if not what:
            raise ValueError("要做什么没说清楚")
        if due - now > MAX_AHEAD:
            raise ValueError("太远了，30 天以内的事我才记得住")
        if len(self.pending(user_id)) >= MAX_PENDING_PER_PERSON:
            raise ValueError(f"已经替 ta 记着 {MAX_PENDING_PER_PERSON} 件事了，先做完几件再说")
        if len(self.pending()) >= MAX_PENDING:
            raise ValueError("手上的事情太多了，记不过来")
        for t in self.pending(user_id):
            if t["kind"] == kind and t["what"] == what[:300]:
                raise ValueError(f"这件事已经记下了（#{t['id']}）")
        conn = self._conn()
        cur = conn.execute(
            "INSERT INTO agent_tasks (created, due, kind, what, user_id, nickname, group_id, origin, status, "
            "attempts, next_try, goal_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)",
            (now, max(now, due), kind, what[:300], user_id, nickname, group_id, origin, max(now, due), goal_id))
        tid = cur.lastrowid
        conn.commit()
        conn.close()
        return tid

    def get(self, tid: int) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        r = conn.execute("SELECT * FROM agent_tasks WHERE id=?", (tid,)).fetchone()
        conn.close()
        return dict(r) if r else None

    def pending(self, user_id: str = None) -> List[Dict[str, Any]]:
        conn = self._conn()
        if user_id is None:
            rows = conn.execute("SELECT * FROM agent_tasks WHERE status IN ('pending','running') ORDER BY due").fetchall()
        else:
            rows = conn.execute("SELECT * FROM agent_tasks WHERE status IN ('pending','running') AND user_id=? "
                                "ORDER BY due", (user_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def due(self, now: float) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM agent_tasks WHERE status='pending' AND next_try<=? ORDER BY due",
                            (now,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM agent_tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _set(self, tid: int, **fields):
        conn = self._conn()
        conn.execute(f"UPDATE agent_tasks SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", (*fields.values(), tid))
        conn.commit()
        conn.close()

    def start(self, tid: int):
        t = self.get(tid)
        self._set(tid, status="running", attempts=(t["attempts"] if t else 0) + 1)

    def finish(self, tid: int, result: str, now: float):
        self._set(tid, status="done", result=(result or "")[:1000], done_at=now)

    def retry_or_fail(self, tid: int, error: str, now: float) -> bool:
        """Returns True when the task has failed for good."""
        t = self.get(tid)
        if not t:
            return True
        if t["attempts"] >= MAX_ATTEMPTS:
            self._set(tid, status="failed", result=error[:1000], done_at=now)
            return True
        self._set(tid, status="pending", result=error[:1000], next_try=now + RETRY_AFTER)
        return False

    def cancel(self, tid: int, by_user: str, may_cancel_any: bool, now: float) -> Optional[Dict[str, Any]]:
        t = self.get(tid)
        if not t or t["status"] not in ("pending", "running"):
            return None
        if t["user_id"] != by_user and not may_cancel_any:
            return None
        self._set(tid, status="cancelled", done_at=now)
        return self.get(tid)
