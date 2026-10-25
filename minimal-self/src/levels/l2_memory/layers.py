import json
import sqlite3
from typing import List, Dict, Any, Optional, Iterable
import numpy as np
from src.core import clock
from .embedding import Embedder

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    """SQLite has no ADD COLUMN IF NOT EXISTS; every module creates its own tables."""
    cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
    if column not in cols:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN {column} {ddl}')
        return True
    return False


# Who may recall an episode:
#   private - only the person it happened with (and Noa herself)
#   group   - anyone in that group (what was said in a group is public *in that group*)
#   public  - distilled experience (thoughts, predictions, continuity) that anyone may see
VISIBILITY_PRIVATE, VISIBILITY_GROUP, VISIBILITY_PUBLIC = "private", "group", "public"


def default_visibility(kind: str, group_id: Optional[str]) -> str:
    if kind in ("dialogue", "proactive"):
        return VISIBILITY_GROUP if group_id else VISIBILITY_PRIVATE
    return VISIBILITY_PUBLIC

class EpisodicMemory:
    """
    Level 2a: Episodic memory - what happened, with whom, and how it felt.

    Each episode keeps the affect and prediction error it was experienced with, so that
    emotionally intense or surprising moments are recalled more easily, and recall
    itself slows forgetting (spaced repetition).
    """
    def __init__(self, db_path: str, embedder: Embedder):
        self.db_path = db_path
        self.embedder = embedder
        conn = _connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            t REAL,
            user_id TEXT,
            group_id TEXT,
            kind TEXT,          -- dialogue / continuity / search / proactive / thought / prediction
            content TEXT,
            valence REAL DEFAULT 0,
            arousal REAL DEFAULT 0,
            surprise REAL DEFAULT 0,
            conflict REAL DEFAULT 0,
            embedding BLOB,
            recall_count INTEGER DEFAULT 0,
            last_recalled REAL,
            consolidated INTEGER DEFAULT 0
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_episodes_t ON episodes(t)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)')
        # Memory scope (added later): old rows are backfilled from what they are and where they happened.
        added = _add_column_if_missing(conn, "episodes", "visibility", "TEXT DEFAULT 'private'")
        if added:
            conn.execute("""UPDATE episodes SET visibility = CASE
                    WHEN kind IN ('dialogue', 'proactive') AND group_id IS NOT NULL THEN 'group'
                    WHEN kind IN ('dialogue', 'proactive') THEN 'private'
                    ELSE 'public' END""")
            conn.execute('CREATE INDEX IF NOT EXISTS idx_episodes_visibility ON episodes(visibility)')
        conn.commit()
        conn.close()

    def add(self, content: str, kind: str = "dialogue", user_id: str = None, group_id: str = None,
            valence: float = 0.0, arousal: float = 0.0, surprise: float = 0.0, conflict: float = 0.0,
            t: float = None, consolidated: bool = False, visibility: str = None) -> int:
        t = clock.now() if t is None else t
        visibility = visibility or default_visibility(kind, group_id)
        emb = Embedder.to_bytes(self.embedder.embed(content))
        conn = _connect(self.db_path)
        cur = conn.execute('''INSERT INTO episodes (t, user_id, group_id, kind, content, valence, arousal, surprise, conflict, embedding, consolidated, visibility)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (t, user_id, group_id, kind, content, valence, arousal, surprise, conflict, emb, int(consolidated), visibility))
        conn.commit()
        episode_id = cur.lastrowid
        conn.close()
        return episode_id

    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        d.pop("embedding", None)
        return d

    def recall(self, query: str, user_id: str = None, k: int = 3, min_similarity: float = 0.15,
               exclude_ids: Iterable[int] = (), group_id: str = None) -> List[Dict[str, Any]]:
        """
        Meaning-based recall. Score = similarity x recency x emotional intensity.
        Recency halves every 30 days, and every past recall doubles that half-life.

        Scope: her own experience here + everything that is public + (inside a group) what
        was said in that same group. A private conversation is not recallable by anyone else.
        """
        if not query.strip():
            return []
        q = self.embedder.embed(query)
        conn = _connect(self.db_path)
        if user_id:
            where = "user_id = ? OR user_id = 'self' OR user_id IS NULL OR visibility = 'public'"
            args: List[Any] = [user_id]
            if group_id:
                where += " OR (visibility = 'group' AND group_id = ?)"
                args.append(group_id)
            rows = conn.execute(f"SELECT * FROM episodes WHERE {where}", args).fetchall()
        else:
            rows = conn.execute("SELECT * FROM episodes").fetchall()
        excluded = set(exclude_ids)
        now = clock.now()
        scored = []
        for r in rows:
            if r["id"] in excluded or r["embedding"] is None:
                continue
            sim = float(np.dot(q, Embedder.from_bytes(r["embedding"])))
            if sim < min_similarity:
                continue
            age_days = max(0.0, now - r["t"]) / 86400.0
            recency = 0.5 ** (age_days / (30.0 * (1 + r["recall_count"])))
            intensity = max(abs(r["valence"]), r["arousal"], r["surprise"])
            scored.append((sim * recency * (1 + 0.5 * intensity), sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        if top:
            ids = [r["id"] for _, _, r in top]
            conn.execute(f"UPDATE episodes SET recall_count = recall_count + 1, last_recalled = ? WHERE id IN ({','.join('?' * len(ids))})",
                         [now, *ids])
            conn.commit()
        conn.close()
        return [{**self._row(r), "score": score, "similarity": sim} for score, sim, r in top]

    def novelty(self, text: str) -> float:
        """1 - similarity to the closest thing already experienced."""
        q = self.embedder.embed(text)
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT embedding FROM episodes WHERE embedding IS NOT NULL").fetchall()
        conn.close()
        if not rows:
            return 1.0
        best = max(float(np.dot(q, Embedder.from_bytes(r["embedding"]))) for r in rows)
        return max(0.0, min(1.0, 1.0 - best))

    def recent(self, limit: int = 3, before: float = None, kinds: Iterable[str] = None) -> List[Dict[str, Any]]:
        sql, args = "SELECT * FROM episodes WHERE 1=1", []
        if before is not None:
            sql += " AND t <= ?"
            args.append(before)
        if kinds:
            kinds = list(kinds)
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        sql += " ORDER BY t DESC, id DESC LIMIT ?"
        args.append(limit)
        conn = _connect(self.db_path)
        rows = conn.execute(sql, args).fetchall()
        conn.close()
        return [self._row(r) for r in rows]

    def unconsolidated(self, until: float = None) -> List[Dict[str, Any]]:
        until = clock.now() if until is None else until
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT * FROM episodes WHERE consolidated = 0 AND t <= ? ORDER BY t", (until,)).fetchall()
        conn.close()
        return [self._row(r) for r in rows]

    def mark_consolidated(self, ids: List[int]):
        if not ids:
            return
        conn = _connect(self.db_path)
        conn.execute(f"UPDATE episodes SET consolidated = 1 WHERE id IN ({','.join('?' * len(ids))})", ids)
        conn.commit()
        conn.close()

    def unconsolidated_count(self) -> int:
        conn = _connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM episodes WHERE consolidated = 0").fetchone()[0]
        conn.close()
        return n

    def count(self) -> int:
        conn = _connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()
        return n

    def import_interactions(self) -> int:
        """
        One-time import of the chat history recorded before episodic memory existed,
        so earlier conversations can be recalled by meaning. Imported as already
        consolidated: they shaped nothing new and should not flood the next dream.
        """
        conn = _connect(self.db_path)
        rows = conn.execute('''SELECT i.user_id, i.group_id, i.role, i.content, i.timestamp, u.nickname
            FROM interactions i LEFT JOIN users u ON u.user_id = i.user_id
            WHERE i.content IS NOT NULL ORDER BY i.timestamp''').fetchall()
        conn.close()
        for r in rows:
            speaker = "我" if r["role"] == "assistant" else (r["nickname"] or f"User({str(r['user_id'])[-4:]})")
            self.add(f"{speaker}：{r['content']}", kind="dialogue", user_id=r["user_id"], group_id=r["group_id"],
                     t=r["timestamp"], consolidated=True)
        return len(rows)

    def existing_ids(self, ids: Iterable[int]) -> List[int]:
        ids = [int(i) for i in ids]
        if not ids:
            return []
        conn = _connect(self.db_path)
        rows = conn.execute(f"SELECT id FROM episodes WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
        conn.close()
        return [r["id"] for r in rows]


class SemanticMemory:
    """
    Level 2b: Stable facts about people and the world, distilled from episodes during
    sleep. Every fact keeps the ids of the episodes it came from.
    """
    def __init__(self, db_path: str, embedder: Embedder):
        self.db_path = db_path
        self.embedder = embedder
        conn = _connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS semantic_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            t REAL,
            user_id TEXT,
            fact TEXT,
            confidence REAL,
            sources TEXT,       -- JSON list of episode ids
            embedding BLOB
        )''')
        # A fact is a distilled, shareable statement ("kerst 最近在折腾显卡"), not a verbatim
        # private message. visibility='public' means she may mention it to anyone.
        _add_column_if_missing(conn, "semantic_facts", "visibility", "TEXT DEFAULT 'public'")
        conn.commit()
        conn.close()

    def add(self, user_id: str, fact: str, sources: List[int], confidence: float = 0.7,
            visibility: str = VISIBILITY_PUBLIC):
        """Insert, or reinforce an existing fact that says the same thing."""
        v = self.embedder.embed(fact)
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT * FROM semantic_facts WHERE user_id = ? AND visibility = ?",
                            (user_id, visibility)).fetchall()
        for r in rows:
            if r["embedding"] is not None and float(np.dot(v, Embedder.from_bytes(r["embedding"]))) > 0.9:
                merged = sorted(set(json.loads(r["sources"] or "[]")) | set(sources))
                conn.execute("UPDATE semantic_facts SET confidence = ?, sources = ?, t = ? WHERE id = ?",
                             (min(1.0, r["confidence"] + 0.1), json.dumps(merged), clock.now(), r["id"]))
                conn.commit()
                conn.close()
                return
        conn.execute("INSERT INTO semantic_facts (t, user_id, fact, confidence, sources, embedding, visibility) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (clock.now(), user_id, fact, confidence, json.dumps(sorted(set(sources))),
                      Embedder.to_bytes(v), visibility))
        conn.commit()
        conn.close()

    def for_user(self, user_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT id, t, user_id, fact, confidence, sources FROM semantic_facts WHERE user_id = ? "
                            "ORDER BY confidence DESC, t DESC LIMIT ?", (user_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def public_facts(self, limit: int = 6, exclude_user: str = None) -> List[Dict[str, Any]]:
        """Distilled facts she may say out loud to anyone - what makes her remember a conversation
        she was not the only one in. Private dialogue itself is never exposed this way."""
        conn = _connect(self.db_path)
        if exclude_user:
            rows = conn.execute("SELECT id, t, user_id, fact, confidence FROM semantic_facts "
                                "WHERE visibility = 'public' AND user_id != ? "
                                "ORDER BY confidence DESC, t DESC LIMIT ?", (exclude_user, limit)).fetchall()
        else:
            rows = conn.execute("SELECT id, t, user_id, fact, confidence FROM semantic_facts "
                                "WHERE visibility = 'public' ORDER BY confidence DESC, t DESC LIMIT ?",
                                (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]


class Autobiography:
    """Level 9: The self-narrative 'who I am and what I have been through', rewritten each sleep."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = _connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS autobiography (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            t REAL,
            text TEXT,
            sources TEXT
        )''')
        conn.commit()
        conn.close()

    def latest(self) -> Optional[str]:
        conn = _connect(self.db_path)
        row = conn.execute("SELECT text FROM autobiography ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        return row["text"] if row else None

    def write(self, text: str, sources: List[int]):
        conn = _connect(self.db_path)
        conn.execute("INSERT INTO autobiography (t, text, sources) VALUES (?, ?, ?)",
                     (clock.now(), text, json.dumps(sources)))
        conn.commit()
        conn.close()
