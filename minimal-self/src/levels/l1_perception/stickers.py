import base64
import json
import logging
import math
import os
import re
import sqlite3
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core import clock
from src.levels.l2_memory.embedding import Embedder

logger = logging.getLogger(__name__)

class StickerMemory:
    """
    Level 1: Her own sticker collection, gathered the way people gather theirs.

    - Only stickers are collected (QQ marks them: image sub_type 1, market faces, system faces);
      ordinary photos people post are never kept.
    - She does not see pictures. What a sticker *means* to her is when people use it:
      the words around each use are remembered, and that is what she recalls it by.
    - Whether a sticker works is learned from how people react when she uses it.
    """
    MAX_CONTEXTS = 8
    MAX_BYTES = 3 * 1024 * 1024

    def __init__(self, db_path: str, folder: str, embedder: Embedder,
                 fetch: Callable[[str], Optional[bytes]] = None):
        self.db_path = db_path
        self.folder = folder
        self.embedder = embedder
        self.fetch = fetch or self._download
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS stickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            kind TEXT,            -- image / mface / face
            data TEXT,            -- JSON: local path (image) or the segment data needed to send it
            summary TEXT,
            contexts TEXT,        -- JSON list of what was being said when people used it
            uses INTEGER DEFAULT 0,
            users TEXT,
            sent INTEGER DEFAULT 0,
            q REAL DEFAULT 0.5,   -- learned: how well it lands when she uses it
            embedding BLOB,
            first_seen REAL,
            last_seen REAL
        )''')
        conn.commit()
        conn.close()

    # --- what counts as a sticker -------------------------------------------------

    @staticmethod
    def extract(segments: Any) -> List[Dict[str, Any]]:
        if not isinstance(segments, list):
            return []
        found = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            kind, data = seg.get("type"), seg.get("data") or {}
            if kind == "image" and str(data.get("sub_type", "0")) == "1":
                found.append(seg)
            elif kind in ("mface", "face"):
                found.append(seg)
        return found

    @staticmethod
    def _key(seg: Dict[str, Any]) -> Optional[str]:
        kind, data = seg.get("type"), seg.get("data") or {}
        if kind == "image":
            return "image:" + str(data.get("file") or "")
        if kind == "mface":
            return f"mface:{data.get('emoji_package_id')}:{data.get('emoji_id')}"
        if kind == "face":
            return f"face:{data.get('id')}"
        return None

    def _download(self, url: str) -> Optional[bytes]:
        import requests
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) <= self.MAX_BYTES:
                return resp.content
        except Exception as e:
            logger.warning(f"[Stickers] download failed: {e}")
        return None

    # --- learning ------------------------------------------------------------------

    def observe(self, seg: Dict[str, Any], user: str, context: str = "") -> Optional[int]:
        if not self.extract([seg]):
            return None   # never keep ordinary photos, whoever calls this
        key = self._key(seg)
        if not key or key.endswith(":") or key.endswith(":None"):
            return None
        kind, data = seg.get("type"), seg.get("data") or {}
        now = clock.now()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM stickers WHERE key = ?", (key,)).fetchone()
        if row is None:
            if kind == "image":
                content = self.fetch(data.get("url", "")) if data.get("url") else None
                if not content:
                    conn.close()
                    return None
                os.makedirs(self.folder, exist_ok=True)
                name = re.sub(r"[^A-Za-z0-9._-]", "_", str(data.get("file")))[:80] or f"sticker_{int(now)}"
                path = os.path.abspath(os.path.join(self.folder, name))
                with open(path, "wb") as f:
                    f.write(content)
                stored = {"path": path}
            else:
                stored = {k: v for k, v in data.items() if k != "url"}
            summary = data.get("summary") or ("[表情]" if kind != "face" else f"[QQ表情{data.get('id')}]")
            cur = conn.execute("INSERT INTO stickers (key, kind, data, summary, contexts, users, first_seen, last_seen) "
                               "VALUES (?, ?, ?, ?, '[]', '[]', ?, ?)",
                               (key, kind, json.dumps(stored, ensure_ascii=False), summary, now, now))
            sticker_id = cur.lastrowid
            contexts, users, uses = [], [], 0
        else:
            sticker_id = row["id"]
            contexts, users, uses = json.loads(row["contexts"]), json.loads(row["users"]), row["uses"]
        context = (context or "").strip()
        if context and context not in contexts:
            contexts = (contexts + [context[:40]])[-self.MAX_CONTEXTS:]
        if user not in users and len(users) < 20:
            users.append(user)
        emb = self.embedder.embed(" ".join(contexts)) if contexts else None
        conn.execute("UPDATE stickers SET contexts = ?, users = ?, uses = ?, last_seen = ?, embedding = ? WHERE id = ?",
                     (json.dumps(contexts, ensure_ascii=False), json.dumps(users, ensure_ascii=False), uses + 1, now,
                      Embedder.to_bytes(emb) if emb is not None else None, sticker_id))
        conn.commit()
        conn.close()
        return sticker_id

    def relevant(self, text: str, k: int = 6) -> List[Dict[str, Any]]:
        """Stickers whose usual occasions resemble this moment, weighted by how familiar and how well they land."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, summary, contexts, uses, q, embedding FROM stickers WHERE embedding IS NOT NULL").fetchall()
        conn.close()
        if not rows or not text.strip():
            return []
        q = self.embedder.embed(text)
        scored = []
        for r in rows:
            sim = float(np.dot(q, Embedder.from_bytes(r["embedding"])))
            if sim <= 0.05:
                continue
            scored.append((sim * (0.5 + r["q"]) * math.log1p(r["uses"]), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"id": r["id"], "summary": r["summary"], "contexts": json.loads(r["contexts"])} for _, r in scored[:k]]

    def segment(self, sticker_id: int) -> Optional[Dict[str, Any]]:
        """The OneBot message segment that sends this sticker."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT kind, data, summary FROM stickers WHERE id = ?", (sticker_id,)).fetchone()
        conn.close()
        if not row:
            return None
        kind, data, summary = row[0], json.loads(row[1]), row[2]
        if kind == "image":
            try:
                with open(data["path"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
            except OSError:
                return None
            return {"type": "image", "data": {"file": f"base64://{b64}", "sub_type": 1, "summary": summary or "[动画表情]"}}
        return {"type": kind, "data": data}

    def mark_sent(self, sticker_id: int):
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE stickers SET sent = sent + 1 WHERE id = ?", (sticker_id,))
        conn.commit()
        conn.close()

    def reward(self, sticker_id: int, value: float, rate: float = 0.3):
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE stickers SET q = q + ? * (? - q) WHERE id = ?", (rate, value, sticker_id))
        conn.commit()
        conn.close()

    def stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) FROM stickers").fetchone()[0]
        top = conn.execute("SELECT id, summary, contexts, uses, sent, q FROM stickers ORDER BY uses DESC LIMIT 6").fetchall()
        conn.close()
        return {"count": total, "top": [{**dict(r), "contexts": json.loads(r["contexts"])} for r in top]}
