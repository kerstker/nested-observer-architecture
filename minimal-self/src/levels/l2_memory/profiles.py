"""
Level 2c: what she knows about a person, as one short portrait instead of a pile of facts.

Facts about someone accumulate without bound (every consolidation can add some), but a prompt
can only carry a few. Taking "the five strongest" made a long friendship look like five random
details. During sleep, for each person whose facts changed, she rewrites a portrait of bounded
length from all of them; in conversation the portrait goes in, plus the few facts that bear on
what is being said right now (SemanticMemory.relevant).
"""
import sqlite3
from typing import Any, Callable, Dict, List, Optional

from src.core import clock

MAX_CHARS = 160

class PersonProfiles:
    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS person_profiles (
            user_id TEXT PRIMARY KEY, profile TEXT, facts INTEGER, updated REAL
        )''')
        conn.commit()
        conn.close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, user_id: str) -> Optional[str]:
        conn = self._conn()
        r = conn.execute("SELECT profile FROM person_profiles WHERE user_id = ?", (user_id,)).fetchone()
        conn.close()
        return r["profile"] if r and r["profile"] else None

    def stale(self, min_facts: int = 3, limit: int = 5) -> List[Dict[str, Any]]:
        """People with enough facts whose portrait is missing or older than their facts."""
        conn = self._conn()
        rows = conn.execute('''SELECT f.user_id, COUNT(*) n, MAX(f.t) last
            FROM semantic_facts f LEFT JOIN person_profiles p ON p.user_id = f.user_id
            WHERE f.user_id NOT IN ('self', '') AND f.user_id IS NOT NULL
            GROUP BY f.user_id HAVING n >= ? AND (MAX(p.updated) IS NULL OR MAX(f.t) > MAX(p.updated))
            ORDER BY n DESC LIMIT ?''', (min_facts, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def rebuild(self, user_id: str, nickname: str, llm: Callable[..., Dict[str, Any]], persona: str = "") -> Optional[str]:
        conn = self._conn()
        facts = [r["fact"] for r in conn.execute(
            "SELECT fact FROM semantic_facts WHERE user_id = ? ORDER BY confidence DESC, t DESC LIMIT 40", (user_id,))]
        conn.close()
        if not facts:
            return None
        prompt = (f"<|im_start|>system\n{persona}\n下面是我（乃爱）记得的关于 {nickname} 的事（越靠前越确定）：\n"
                  + "\n".join(f"- {f}" for f in facts)
                  + f"\n\n把这些写成一段我心里对 {nickname} 的印象：ta 是什么样的人、平时爱聊什么、我们之间怎么样。"
                  f"只写上面有依据的，不要编；互相矛盾的以靠前的为准。第一人称，{MAX_CHARS} 字以内，一段话，不要列表。"
                  "<|im_end|>\n<|im_start|>assistant\n")
        try:
            text = ((llm(prompt, max_tokens=220, temperature=0.4) or {}).get("reply") or "").strip()
        except Exception:
            return None
        if not text:
            return None
        text = text.replace("\n", " ")[:MAX_CHARS + 40]
        conn = self._conn()
        conn.execute("REPLACE INTO person_profiles (user_id, profile, facts, updated) VALUES (?, ?, ?, ?)",
                     (user_id, text, len(facts), clock.now()))
        conn.commit()
        conn.close()
        return text
