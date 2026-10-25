import json
import re
import sqlite3
from typing import Any, Dict, List, Optional

def _left(seconds: float) -> str:
    if seconds <= 0:
        return "已经过了时间"
    if seconds < 3600:
        return f"还剩 {seconds / 60:.0f} 分钟"
    if seconds < 86400:
        return f"还剩 {seconds / 3600:.0f} 小时"
    return f"还剩 {seconds / 86400:.0f} 天"

class GoalSystem:
    """
    Level 8b: Long-term goals and plans.

    Drives answer "what do I need right now"; goals are what she has set out to do over days.
    None of them is given to her. They come from her own experience:
    - promise: something she said she would do ("我查查告诉你") - she remembers it and keeps it;
    - relationship: someone she talks to often and likes, but hardly knows - she wants to know them;
    - learn: something she was asked and could not answer, or failed to find out - she tries again;
    - self: what she decides, while reflecting in her sleep, that she wants for herself.
    Each goal has a plan (steps), progress and a reason. Goals add value to the actions that serve
    them (so they enter the same expected-free-energy choice as needs), come to mind on their own,
    and are let go of when they stop making sense - a broken promise is felt, a reached goal too.
    """
    KINDS = ("promise", "relationship", "learn", "self")
    MAX_ACTIVE = 6
    STALE_AFTER = 5 * 86400
    PROMISE = re.compile(r"(等我|我去|我回头|回头|晚点|待会|等会|明天|下次|改天|过两天)[^。！？\n]{0,10}?"
                         r"(查|看看|问问|告诉你|跟你说|找你|发你|回你|研究|想想)"
                         r"|(?:我先|我)(查查|看看|问问|想想)再(回你|告诉你|跟你说|说)")

    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, text TEXT, target TEXT, topic TEXT, why TEXT,
            steps TEXT, importance REAL, progress REAL, status TEXT, created REAL, deadline REAL,
            last_progress REAL, attempts INTEGER, baseline REAL, note TEXT
        )''')
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        d["steps"] = json.loads(d["steps"] or "[]")
        return d

    # --- having goals -------------------------------------------------------------

    def add(self, kind: str, text: str, now: float, target: str = None, topic: str = None, why: str = "",
            steps: List[str] = None, importance: float = 0.5, deadline: float = None,
            baseline: float = 0.0) -> Optional[int]:
        if kind not in self.KINDS or not text:
            return None
        for g in self.active():
            if g["kind"] == kind and g["target"] == target and (g["topic"] == topic or kind == "relationship"):
                return None                                   # she already means to do this
        conn = self._conn()
        cur = conn.execute(
            "INSERT INTO goals (kind, text, target, topic, why, steps, importance, progress, status, created, "
            "deadline, last_progress, attempts, baseline, note) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?, 0, ?, '')",
            (kind, text, target, topic, why, json.dumps([{"text": s, "done": False} for s in steps or []],
                                                         ensure_ascii=False),
             importance, now, deadline, now, baseline))
        gid = cur.lastrowid
        conn.commit()
        conn.close()
        active = self.active()
        if len(active) > self.MAX_ACTIVE:                     # limited capacity: the least important waits
            drop = min((g for g in active if g["kind"] != "promise"), key=lambda g: g["importance"], default=None)
            if drop:
                self._set(drop["id"], status="dropped", note="想做的事太多了，这件先放下")
        return gid

    def detect_promise(self, reply: str, user_id: str, nickname: str, about: str, now: float) -> Optional[int]:
        """Did what she just said commit her to something?"""
        m = self.PROMISE.search(reply or "")
        if not m:
            return None
        when = m.group(1) or ""
        span = 86400 if when in ("明天", "过两天") else 3 * 86400 if when in ("下次", "改天") else 6 * 3600
        needs_lookup = any(v in m.group(0) for v in ("查", "看看", "问问", "研究"))
        return self.add("promise", f"我答应了 {nickname}：「{m.group(0)}」（ta 当时说的是「{about[:30]}」）", now,
                        target=user_id, topic=about[:60] if needs_lookup else None,
                        why="我自己说出口的", importance=0.8, deadline=now + span,
                        steps=(["去查清楚", "告诉 ta"] if needs_lookup else ["告诉 ta"]))

    # --- reading ------------------------------------------------------------------

    def get(self, gid: int) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        r = conn.execute("SELECT * FROM goals WHERE id=?", (gid,)).fetchone()
        conn.close()
        return self._row(r) if r else None

    def active(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM goals WHERE status='active' ORDER BY id").fetchall()
        conn.close()
        return [self._row(r) for r in rows]

    def history(self, limit: int = 8) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM goals WHERE status!='active' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [self._row(r) for r in rows]

    @staticmethod
    def urgency(g: Dict[str, Any], now: float) -> float:
        """1 when nothing presses; up to 2 as a deadline comes close."""
        if not g["deadline"]:
            return 1.0
        span = max(1.0, g["deadline"] - g["created"])
        return 1.0 + min(1.0, max(0.0, 1.0 - (g["deadline"] - now) / span))

    def pull(self, g: Dict[str, Any], now: float) -> float:
        return g["importance"] * (1.0 - g["progress"]) * self.urgency(g, now)

    def serving(self, action: str, target: Optional[str]) -> List[Dict[str, Any]]:
        """The goals an action would move forward."""
        out = []
        for g in self.active():
            if action == "chat" and g["kind"] in ("promise", "relationship") and g["target"] == target:
                out.append(g)
            elif action == "search" and g["kind"] == "learn":
                out.append(g)
        return out

    def value_for(self, action: str, target: Optional[str], now: float) -> float:
        """Extra pragmatic value of an action: how far it would bring her toward what she has set out to do."""
        return 0.6 * sum(self.pull(g, now) for g in self.serving(action, target))

    def most_pressing(self, action: str = None, target: str = None, now: float = 0.0) -> Optional[Dict[str, Any]]:
        goals = self.serving(action, target) if action else self.active()
        return max(goals, key=lambda g: self.pull(g, now), default=None)

    @staticmethod
    def next_step(g: Dict[str, Any]) -> str:
        return next((s["text"] for s in g["steps"] if not s["done"]), "")

    def describe(self, now: float, n: int = 4, about: str = None) -> str:
        goals = sorted(self.active(), key=lambda g: -(self.pull(g, now) + (1 if about and g["target"] == about else 0)))
        lines = []
        for g in goals[:n]:
            line = f"- [{g['id']}] {g['text']}"
            extra = []
            if g["progress"] > 0:
                extra.append(f"进展 {g['progress']:.0%}")
            if g["deadline"]:
                extra.append(_left(g["deadline"] - now))
            step = self.next_step(g)
            if step:
                extra.append(f"下一步：{step}")
            lines.append(line + (f"（{'，'.join(extra)}）" if extra else ""))
        return "\n".join(lines)

    # --- changing -----------------------------------------------------------------

    def _set(self, gid: int, **fields):
        if not fields:
            return
        if "steps" in fields:
            fields["steps"] = json.dumps(fields["steps"], ensure_ascii=False)
        conn = self._conn()
        conn.execute(f"UPDATE goals SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", (*fields.values(), gid))
        conn.commit()
        conn.close()

    def attempt(self, gid: int):
        g = self.get(gid)
        if g:
            self._set(gid, attempts=g["attempts"] + 1)

    def step_done(self, gid: int, now: float):
        g = self.get(gid)
        if not g:
            return
        steps = g["steps"]
        for s in steps:
            if not s["done"]:
                s["done"] = True
                break
        done = sum(s["done"] for s in steps)
        self._set(gid, steps=steps, progress=max(g["progress"], done / max(1, len(steps))), last_progress=now)

    def advance(self, gid: int, progress: float, now: float, note: str = "") -> Optional[Dict[str, Any]]:
        """Set progress; returns the goal if this completed it."""
        g = self.get(gid)
        if not g or g["status"] != "active":
            return None
        progress = max(g["progress"], min(1.0, progress))
        if progress >= 1.0:
            self._set(gid, progress=1.0, status="done", last_progress=now, note=note or "做到了")
            return self.get(gid)
        if progress > g["progress"]:
            self._set(gid, progress=progress, last_progress=now, note=note or g["note"])
        return None

    def complete(self, gid: int, now: float, note: str = "") -> Optional[Dict[str, Any]]:
        return self.advance(gid, 1.0, now, note)

    def abandon(self, gid: int, now: float, reason: str) -> Optional[Dict[str, Any]]:
        g = self.get(gid)
        if not g or g["status"] != "active":
            return None
        self._set(gid, status="abandoned", note=reason, last_progress=now)
        return self.get(gid)

    def review(self, now: float) -> List[Dict[str, Any]]:
        """Goal disengagement: promises past their time, goals that have gone nowhere for days."""
        gone = []
        for g in self.active():
            if g["kind"] == "promise" and g["deadline"] and now > g["deadline"] + 86400:
                gone.append(self.abandon(g["id"], now, "过了说好的时间，没做到"))
            elif g["kind"] != "promise" and now - g["last_progress"] > self.STALE_AFTER and g["attempts"] >= 3:
                gone.append(self.abandon(g["id"], now, "试了好几次都没有进展，算了"))
        return [g for g in gone if g]

    def review_text(self) -> str:
        return "\n".join(f"[{g['id']}] {g['text']}（进展 {g['progress']:.0%}）" for g in self.active()
                         if g["kind"] == "self") or "（没有）"
