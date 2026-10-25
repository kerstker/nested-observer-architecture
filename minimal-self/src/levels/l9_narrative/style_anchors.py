"""
Her own voice, as examples: what she said when she sounded most like herself.

This is the cheap half of "training her on herself": no weights change, no GPU, no drift.
A handful of her best past replies are put in front of the model as style anchors - the tone it
should imitate - while the facts still come from memory. Retrieval over her own writing is done
by the memory system; this module only answers "what did I say when I was at my best".

"Best" is read off her own records, not picked by hand:
- the exchange felt good to her (episode valence), with no conflict;
- she did not use tool syntax, stage directions, or a stock phrase;
- it is a real reply, not a refusal/silence marker;
- not too short to show anything, not so long that the block eats the prompt;
- spread over different people, so one very positive chat cannot define her whole voice.

Anchors rotate daily (deterministic by date) so she is not shown the same six lines every turn.
"""
import random
import sqlite3
import time
from typing import List

from src.core import clock

MIN_LEN, MAX_LEN = 10, 60
BANNED = ("Action:", "[工具", "```", "（不回）", "(不回)", "[SILENT]", "上面写着", "（发了一个表情）", "Action：",
          "*", "＊")
MAX_PER_USER = 2
CACHE_SECONDS = 6 * 3600


class StyleAnchors:
    def __init__(self, db_path: str, min_valence: float = 0.12, window_days: float = 30.0):
        self.db_path = db_path
        self.min_valence = min_valence
        self.window_days = window_days
        self._cache: List[str] = []
        self._cache_t = 0.0

    def _usable(self, reply: str) -> bool:
        if not (MIN_LEN <= len(reply) <= MAX_LEN):
            return False
        if any(b in reply for b in BANNED):
            return False
        if reply.count("\n") > 1:          # a burst of lines is fine, a wall of text is not
            return False
        if reply.startswith(("（", "(", "[", "#")):
            return False
        return True

    def _candidates(self) -> List[str]:
        cutoff = clock.now() - self.window_days * 86400
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT user_id, content, valence, t FROM episodes "
                "WHERE kind = 'dialogue' AND conflict = 0 AND valence >= ? AND t >= ? "
                "AND user_id IS NOT NULL AND user_id NOT IN ('self', 'console', 'test_user', 'unknown') "
                "AND content LIKE '%我：%' "
                "ORDER BY t DESC LIMIT 800", (self.min_valence, cutoff)).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()

        per_user = {}
        picked: List[str] = []
        for row in rows:                        # newest first: recent voice wins on ties
            reply = row["content"].split("我：", 1)[-1].strip()
            reply = reply.split("\n")[0].strip()      # the first line is her opening line
            if not self._usable(reply):
                continue
            user = str(row["user_id"])
            if per_user.get(user, 0) >= MAX_PER_USER:
                continue
            per_user[user] = per_user.get(user, 0) + 1
            if reply not in picked:
                picked.append(reply)
        return picked

    def examples(self, limit: int = 6) -> List[str]:
        """Her best lines to imitate - same set for the whole day, a different set tomorrow."""
        now = clock.now()
        if not self._cache or now - self._cache_t > CACHE_SECONDS:
            self._cache = self._candidates()
            self._cache_t = now
        if len(self._cache) <= limit:
            return list(self._cache)
        day = int(time.strftime("%Y%m%d"))
        shuffled = list(self._cache)
        random.Random(day).shuffle(shuffled)
        return shuffled[:limit]

