import json
import re
import sqlite3
from collections import deque, defaultdict
from typing import Dict, List, Tuple

from src.core import clock

# Fragments too common to say anything about a group's way of talking
_STOP = set("的了是我你他她它们这那就都也还在有和吗呢吧啊呀哦嗯么个不没要会能说去来到上下好")

class GroupLanguage:
    """
    Level 1: Hearing a group, and picking up how it talks.

    - What people said recently is kept only in memory (a short rolling buffer per group),
      so she can follow the conversation without permanently recording everyone.
    - What persists is statistics: short phrases that several different people keep using.
      During sleep the most promising ones are offered to her as candidate group slang.
    """
    def __init__(self, db_path: str, buffer_size: int = 40):
        self.db_path = db_path
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS group_phrases (
            group_id TEXT, phrase TEXT, count REAL, speakers TEXT, last_seen REAL,
            PRIMARY KEY (group_id, phrase)
        )''')
        conn.commit()
        conn.close()

    @staticmethod
    def _phrases(text: str) -> List[str]:
        out = set()
        for run in re.findall(r"[一-鿿]{2,}", text):
            for n in (2, 3, 4):
                for i in range(len(run) - n + 1):
                    p = run[i:i + n]
                    if not all(ch in _STOP for ch in p) and p[0] not in _STOP and p[-1] not in _STOP:
                        out.add(p)
        for word in re.findall(r"[A-Za-z]{3,12}", text):
            out.add(word.lower())
        return list(out)

    def observe(self, group_id: str, speaker: str, text: str, from_self: bool = False, learn: bool = True):
        now = clock.now()
        self.buffers[group_id].append((now, speaker, text))
        if from_self or not learn:
            return   # she learns the group's language from what others actually typed
        phrases = self._phrases(text)
        if not phrases:
            return
        conn = sqlite3.connect(self.db_path)
        for p in phrases:
            row = conn.execute("SELECT count, speakers FROM group_phrases WHERE group_id=? AND phrase=?",
                               (group_id, p)).fetchone()
            speakers = set(json.loads(row[1])) if row else set()
            if len(speakers) < 10:
                speakers.add(speaker)
            conn.execute("REPLACE INTO group_phrases (group_id, phrase, count, speakers, last_seen) VALUES (?, ?, ?, ?, ?)",
                         (group_id, p, (row[0] if row else 0) + 1, json.dumps(sorted(speakers), ensure_ascii=False), now))
        conn.commit()
        conn.close()

    def recent_lines(self, group_id: str, n: int = 12) -> List[str]:
        return [f"{speaker}：{text}" for _, speaker, text in list(self.buffers.get(group_id, []))[-n:]]

    def candidates(self, group_id: str = None, k: int = 20, min_count: int = 3, min_speakers: int = 2) -> List[str]:
        """Phrases used repeatedly by several people - raw material for group slang."""
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT phrase, count, speakers FROM group_phrases WHERE count >= ?"
        args: Tuple = (min_count,)
        if group_id:
            sql += " AND group_id = ?"
            args += (group_id,)
        rows = conn.execute(sql + " ORDER BY count DESC LIMIT 400", args).fetchall()
        conn.close()
        picked = [p for p, _, s in rows if len(json.loads(s)) >= min_speakers]
        # Prefer longer phrases: a 4-character phrase repeated by many people is more telling than a 2-character one
        picked.sort(key=lambda p: -len(p))
        result = []
        for p in picked:
            if not any(p in longer for longer in result):
                result.append(p)
            if len(result) >= k:
                break
        return result
