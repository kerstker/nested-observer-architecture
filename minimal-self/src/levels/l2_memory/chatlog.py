"""
Level 2e: the conversation as it happened - every message in a chat, hers included, in order.

What she read as "recent conversation" used to come from `interactions`: only messages she had
processed (so in a group, people talking to each other were missing), her own replies stored as
raw output (stickers as "[表情:43]" - a number she could not see), and a query that returned the
*oldest* messages of the week instead of the latest. She lost track of who said what, could not
recognise her own words when quoted, and could not see her own stickers.

The chat log keeps, per chat (a group, or a private chat), every message she perceived or sent:
who, when, what (as she perceived it), whether it was hers, which message it quoted and whom it
@-mentioned. `transcript()` renders the recent part from her own point of view: her lines are
"我（乃爱）", quotes say whose words they were, gaps in time are marked, and it fits a size budget
from the newest message backwards.
"""
import json
import re
import sqlite3
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.core import clock

def chat_key(user_id: Optional[str], group_id: Optional[str]) -> str:
    return f"g:{group_id}" if group_id else f"p:{user_id}"

class ChatLog:
    def __init__(self, db_path: str, keep: int = 400):
        self.db_path = db_path
        self.keep = keep                       # messages kept per chat (older ones are in episodes)
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat TEXT, t REAL, message_id TEXT, sender_id TEXT,
            sender_name TEXT, is_self INTEGER, text TEXT, reply_to TEXT, mentions TEXT, addressed INTEGER
        )''')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_log ON chat_log(chat, t)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_log_msg ON chat_log(message_id)")
        conn.commit()
        conn.close()
        self._writes = 0

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, chat: str, sender_id: str, sender_name: str, text: str, t: float = None,
               message_id: str = None, is_self: bool = False, reply_to: str = None,
               mentions: Iterable[str] = (), addressed: bool = False) -> int:
        text = (text or "").strip()
        if not text:
            return 0
        conn = self._conn()
        if message_id:
            dup = conn.execute("SELECT id FROM chat_log WHERE message_id = ? AND chat = ?", (str(message_id), chat)).fetchone()
            if dup:
                conn.close()
                return dup["id"]
        rid = conn.execute("INSERT INTO chat_log (chat, t, message_id, sender_id, sender_name, is_self, text, reply_to, "
                           "mentions, addressed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (chat, clock.now() if t is None else t, str(message_id) if message_id else None, sender_id,
                            sender_name, int(is_self), text[:1500], str(reply_to) if reply_to else None,
                            json.dumps(list(mentions), ensure_ascii=False), int(addressed))).lastrowid
        conn.commit()
        self._writes += 1
        if self._writes % 200 == 0:            # bounded per chat
            conn.execute("DELETE FROM chat_log WHERE id IN (SELECT id FROM chat_log c WHERE (SELECT COUNT(*) FROM chat_log d "
                         "WHERE d.chat = c.chat AND d.t > c.t) >= ?)", (self.keep,))
            conn.commit()
        conn.close()
        return rid

    def by_message_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        r = conn.execute("SELECT * FROM chat_log WHERE message_id = ? ORDER BY id DESC LIMIT 1", (str(message_id),)).fetchone()
        conn.close()
        return dict(r) if r else None

    def recent(self, chat: str, limit: int = 60, before_id: int = None) -> List[Dict[str, Any]]:
        conn = self._conn()
        if before_id:
            rows = conn.execute("SELECT * FROM chat_log WHERE chat = ? AND id < ? ORDER BY t DESC, id DESC LIMIT ?",
                                (chat, before_id, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chat_log WHERE chat = ? ORDER BY t DESC, id DESC LIMIT ?",
                                (chat, limit)).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]

    def count(self, chat: str) -> int:
        conn = self._conn()
        n = conn.execute("SELECT COUNT(*) FROM chat_log WHERE chat = ?", (chat,)).fetchone()[0]
        conn.close()
        return n

    @staticmethod
    def _when(t: float, now: float) -> str:
        d, n = datetime.fromtimestamp(t), datetime.fromtimestamp(now)
        return d.strftime("%H:%M") if d.date() == n.date() else d.strftime("%m-%d %H:%M")

    def transcript(self, chat: str, now: float = None, max_chars: int = 3500, limit: int = 80,
                   exclude_id: int = None, render_text: Callable[[str], str] = None) -> str:
        """The recent conversation from her point of view, newest last, within a size budget."""
        now = clock.now() if now is None else now
        rows = [r for r in self.recent(chat, limit + 1) if r["id"] != exclude_id][-limit:]
        if not rows:
            return ""
        by_msg = {r["message_id"]: r for r in rows if r["message_id"]}
        lines: List[str] = []
        prev_t = None
        for r in rows:
            who = "我（乃爱）" if r["is_self"] else (r["sender_name"] or r["sender_id"] or "某人")
            text = render_text(r["text"]) if render_text else r["text"]
            if r["reply_to"] and r["reply_to"] in by_msg and "（引用" not in text and "（回复" not in text:
                q = by_msg[r["reply_to"]]
                qwho = "我（乃爱）" if q["is_self"] else (q["sender_name"] or "某人")
                qtext = render_text(q["text"]) if render_text else q["text"]
                text = f"（引用 {qwho} 说的「{qtext[:40]}」）{text}"
            gap = ""
            if prev_t is not None and r["t"] - prev_t > 1800:
                hours = (r["t"] - prev_t) / 3600
                gap = f"（…隔了 {hours:.0f} 小时…）\n" if hours >= 1 else "（…隔了一会儿…）\n"
            prev_t = r["t"]
            lines.append(f"{gap}[{self._when(r['t'], now)}] {who}：{text}")
        # Fit the budget from the newest backwards: what was just said matters most
        out, size = [], 0
        for line in reversed(lines):
            if size + len(line) > max_chars and out:
                out.append("（更早的聊天省略了）")
                break
            out.append(line)
            size += len(line) + 1
        text = "\n".join(reversed(out))
        # "What did you just say/ask?" - name her last message, so older memories cannot stand in for it
        mine = [r for r in rows if r["is_self"]]
        if mine:
            last = render_text(mine[-1]["text"]) if render_text else mine[-1]["text"]
            text += f"\n（我在这个聊天里最后说的是：「{last[:80]}」）"
        return text
