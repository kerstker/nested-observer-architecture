import json
import logging
import sqlite3
import threading
import time
from typing import List, Dict, Any, Optional, Iterable
import numpy as np
from src.core import clock
from .embedding import Embedder

logger = logging.getLogger(__name__)

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


def vacuum(db_path: str):
    """Give back the space of rewritten rows. Never fatal: another connection may be busy."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("VACUUM")
        conn.close()
    except sqlite3.Error as e:
        logger.info(f"[Memory] VACUUM skipped: {e}")


# Who may recall an episode:
#   private - only the person it happened with (and Noa herself)
#   group   - anyone in that group (what was said in a group is public *in that group*)
#   public  - distilled experience (thoughts, predictions, continuity) that anyone may see
VISIBILITY_PRIVATE, VISIBILITY_GROUP, VISIBILITY_PUBLIC = "private", "group", "public"


def default_visibility(kind: str, group_id: Optional[str]) -> str:
    if kind in ("dialogue", "proactive"):
        return VISIBILITY_GROUP if group_id else VISIBILITY_PRIVATE
    return VISIBILITY_PUBLIC

class _EpisodeIndex:
    """The active episodes' vectors and the few columns recall filters and ranks by, in memory.

    Recall used to read every episode from SQLite and compare vectors one by one in Python on each
    call (and the heartbeat calls it for associations): fine at a thousand episodes, seconds per call
    after a year. Here one matrix product scores them all; the index is loaded once and then follows
    the table by id (new rows are picked up on the next recall, whoever wrote them)."""
    COLS = ("id", "t", "user_id", "group_id", "visibility", "valence", "arousal", "surprise", "recall_count")

    def __init__(self, dim: int):
        self.dim = dim
        self.last_id = 0
        self.ids = np.zeros(0, dtype=np.int64)
        self.M = np.zeros((0, dim), dtype=np.float32)
        self.t = np.zeros(0)
        self.user = np.zeros(0, dtype=object)
        self.group = np.zeros(0, dtype=object)
        self.vis = np.zeros(0, dtype=object)
        self.intensity = np.zeros(0)
        self.recalls = np.zeros(0)

    def extend(self, rows: List[sqlite3.Row]):
        rows = [r for r in rows if r["embedding"] is not None]
        if not rows:
            return
        vecs = np.stack([Embedder.from_bytes(r["embedding"]) for r in rows])
        keep = vecs.shape[1] == self.dim
        if not keep:
            return
        self.ids = np.concatenate([self.ids, [r["id"] for r in rows]])
        self.M = np.concatenate([self.M, vecs])
        self.t = np.concatenate([self.t, [r["t"] for r in rows]])
        self.user = np.concatenate([self.user, np.array([r["user_id"] for r in rows], dtype=object)])
        self.group = np.concatenate([self.group, np.array([r["group_id"] for r in rows], dtype=object)])
        self.vis = np.concatenate([self.vis, np.array([r["visibility"] for r in rows], dtype=object)])
        self.intensity = np.concatenate([self.intensity, [max(abs(r["valence"] or 0), r["arousal"] or 0, r["surprise"] or 0)
                                                          for r in rows]])
        self.recalls = np.concatenate([self.recalls, [r["recall_count"] or 0 for r in rows]])

    def drop(self, ids: Iterable[int]):
        keep = ~np.isin(self.ids, list(ids))
        for name in ("ids", "M", "t", "user", "group", "vis", "intensity", "recalls"):
            setattr(self, name, getattr(self, name)[keep])

    def allowed(self, user_id: Optional[str], group_id: Optional[str]) -> np.ndarray:
        """Scope: her own experience + everything public + this person's + (in a group) that group's."""
        if not user_id:
            return np.ones(len(self.ids), dtype=bool)
        mask = (self.user == user_id) | (self.user == "self") | np.equal(self.user, None) | (self.vis == VISIBILITY_PUBLIC)
        if group_id:
            mask |= (self.vis == VISIBILITY_GROUP) & (self.group == group_id)
        return mask


class EpisodicMemory:
    """
    Level 2a: Episodic memory - what happened, with whom, and how it felt.

    Each episode keeps the affect and prediction error it was experienced with, so that
    emotionally intense or surprising moments are recalled more easily, and recall
    itself slows forgetting (spaced repetition).

    Like a person's, her memory does not fill up; it gets harder to search. So:
    - recall runs over an in-memory index (one matrix product, not a scan of the table);
    - what sleep has already distilled (into facts and her autobiography), that was never recalled
      since and was not intense, is archived after a while: it stays in the database and can still
      be found by a deliberate search (`deep=True`), which brings it back into everyday memory -
      the details fade from easy reach, the gist stays where it was distilled to.
    """
    ARCHIVE_AFTER_DAYS = 30

    def __init__(self, db_path: str, embedder: Embedder):
        self.db_path = db_path
        self.embedder = embedder
        self._index: Optional[_EpisodeIndex] = None
        self._lock = threading.RLock()
        self.recall_ms: List[float] = []       # recent recall timings, for the console
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
        # Which vector space the embedding is in, and whether it is still in everyday memory
        _add_column_if_missing(conn, "episodes", "emb_model", "TEXT")
        _add_column_if_missing(conn, "episodes", "archived", "INTEGER DEFAULT 0")
        conn.commit()
        conn.close()
        self.ensure_model()

    # --- vectors in the current model's space ---------------------------------------

    def ensure_model(self, batch: int = 64) -> int:
        """Re-embed episodes whose vectors are from another model (or from before models were recorded)."""
        mid = self.embedder.model_id
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT id, content FROM episodes WHERE emb_model IS NULL OR emb_model != ?", (mid,)).fetchall()
        if rows and len(rows) > 50:
            logger.info(f"[Memory] re-embedding {len(rows)} episodes for {mid} ...")
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            vecs = self.embedder.embed_many([r["content"] or "" for r in chunk])
            conn.executemany("UPDATE episodes SET embedding = ?, emb_model = ? WHERE id = ?",
                             [(Embedder.to_bytes(v), mid, r["id"]) for v, r in zip(vecs, chunk)])
            conn.commit()
        conn.close()
        if len(rows) > 500:
            vacuum(self.db_path)               # the old, larger vectors leave free pages behind
        with self._lock:
            self._index = None
        return len(rows)

    def _refresh(self) -> _EpisodeIndex:
        """The index, brought up to date with any episodes added since it was last read."""
        with self._lock:
            if self._index is None:
                self._index = _EpisodeIndex(self.embedder.dim)
            idx = self._index
            conn = _connect(self.db_path)
            rows = conn.execute(f"SELECT {', '.join(_EpisodeIndex.COLS)}, embedding FROM episodes "
                                "WHERE id > ? AND archived = 0 AND emb_model = ? ORDER BY id",
                                (idx.last_id, self.embedder.model_id)).fetchall()
            top = conn.execute("SELECT MAX(id) FROM episodes").fetchone()[0] or 0
            conn.close()
            idx.extend(rows)
            idx.last_id = max(idx.last_id, top)
            return idx

    def add(self, content: str, kind: str = "dialogue", user_id: str = None, group_id: str = None,
            valence: float = 0.0, arousal: float = 0.0, surprise: float = 0.0, conflict: float = 0.0,
            t: float = None, consolidated: bool = False, visibility: str = None) -> int:
        t = clock.now() if t is None else t
        visibility = visibility or default_visibility(kind, group_id)
        emb = Embedder.to_bytes(self.embedder.embed(content))
        conn = _connect(self.db_path)
        cur = conn.execute('''INSERT INTO episodes (t, user_id, group_id, kind, content, valence, arousal, surprise, conflict, embedding, consolidated, visibility, emb_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (t, user_id, group_id, kind, content, valence, arousal, surprise, conflict, emb, int(consolidated), visibility,
             self.embedder.model_id))
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
               exclude_ids: Iterable[int] = (), group_id: str = None, deep: bool = False) -> List[Dict[str, Any]]:
        """
        Meaning-based recall. Score = similarity x recency x emotional intensity.
        Recency halves every 30 days, and every past recall doubles that half-life.

        Scope: her own experience here + everything that is public + (inside a group) what
        was said in that same group. A private conversation is not recallable by anyone else.

        deep: also search the archive (a deliberate effort to remember); what is found there
        comes back into everyday memory.
        """
        if not query.strip():
            return []
        started = time.perf_counter()
        try:
            return self._recall(query, user_id, k, min_similarity, exclude_ids, group_id, deep)
        finally:
            self.recall_ms = (self.recall_ms + [1000 * (time.perf_counter() - started)])[-200:]

    def _recall(self, query, user_id, k, min_similarity, exclude_ids, group_id, deep) -> List[Dict[str, Any]]:
        q = self.embedder.embed(query)
        now = clock.now()
        with self._lock:
            idx = self._refresh()
            cand: List[tuple] = []           # (score, similarity, id)
            if len(idx.ids):
                sims = self.embedder.scores(idx.M, q)
                ok = idx.allowed(user_id, group_id) & (sims >= min_similarity)
                excluded = list(exclude_ids)
                if excluded:
                    ok &= ~np.isin(idx.ids, excluded)
                if ok.any():
                    age_days = np.maximum(0.0, now - idx.t[ok]) / 86400.0
                    recency = 0.5 ** (age_days / (30.0 * (1 + idx.recalls[ok])))
                    score = sims[ok] * recency * (1 + 0.5 * idx.intensity[ok])
                    top = np.argsort(-score)[:k]
                    cand = [(float(score[i]), float(sims[ok][i]), int(idx.ids[ok][i])) for i in top]
        if deep:
            cand = sorted(cand + self._search_archive(q, user_id, group_id, min_similarity, set(exclude_ids), now, k),
                          key=lambda c: -c[0])[:k]
        if not cand:
            return []
        ids = [c[2] for c in cand]
        conn = _connect(self.db_path)
        rows = {r["id"]: r for r in conn.execute(f"SELECT * FROM episodes WHERE id IN ({','.join('?' * len(ids))})", ids)}
        conn.execute(f"UPDATE episodes SET recall_count = recall_count + 1, last_recalled = ?, archived = 0 "
                     f"WHERE id IN ({','.join('?' * len(ids))})", [now, *ids])
        conn.commit()
        conn.close()
        with self._lock:
            if self._index is not None:
                hit = np.isin(self._index.ids, ids)
                self._index.recalls[hit] += 1
                revived = [i for i in ids if i not in set(self._index.ids[hit].tolist())]
                if revived:                       # back from the archive: into everyday memory again
                    self._index.extend([rows[i] for i in revived if i in rows])
        return [{**self._row(rows[i]), "score": s, "similarity": sim} for s, sim, i in cand if i in rows]

    def _search_archive(self, q, user_id, group_id, min_similarity, excluded, now, k, chunk: int = 5000):
        out = []
        conn = _connect(self.db_path)
        last = 0
        while True:
            rows = conn.execute(f"SELECT {', '.join(_EpisodeIndex.COLS)}, embedding FROM episodes "
                                "WHERE archived = 1 AND emb_model = ? AND id > ? ORDER BY id LIMIT ?",
                                (self.embedder.model_id, last, chunk)).fetchall()
            if not rows:
                break
            last = rows[-1]["id"]
            part = _EpisodeIndex(self.embedder.dim)
            part.extend(rows)
            if not len(part.ids):
                continue
            sims = self.embedder.scores(part.M, q)
            ok = part.allowed(user_id, group_id) & (sims >= min_similarity) & ~np.isin(part.ids, list(excluded))
            for i in np.flatnonzero(ok):
                age = max(0.0, now - part.t[i]) / 86400.0
                score = sims[i] * 0.5 ** (age / (30.0 * (1 + part.recalls[i]))) * (1 + 0.5 * part.intensity[i])
                out.append((float(score), float(sims[i]), int(part.ids[i])))
            out = sorted(out, key=lambda c: -c[0])[:k]
        conn.close()
        return out

    def novelty(self, text: str) -> float:
        """1 - similarity to the closest thing already experienced."""
        q = self.embedder.embed(text)
        with self._lock:
            idx = self._refresh()
            if not len(idx.ids):
                return 1.0
            best = float(self.embedder.scores(idx.M, q).max())
        return max(0.0, min(1.0, 1.0 - best))

    def archive(self, now: float = None, older_than_days: float = None) -> int:
        """Move what sleep has already distilled out of everyday memory (see the class docstring)."""
        now = clock.now() if now is None else now
        cutoff = now - 86400 * (self.ARCHIVE_AFTER_DAYS if older_than_days is None else older_than_days)
        conn = _connect(self.db_path)
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM episodes WHERE archived = 0 AND consolidated = 1 AND t < ? AND recall_count = 0 "
            "AND MAX(ABS(valence), arousal, surprise) < 0.5", (cutoff,)).fetchall()]
        if ids:
            for start in range(0, len(ids), 500):
                part = ids[start:start + 500]
                conn.execute(f"UPDATE episodes SET archived = 1 WHERE id IN ({','.join('?' * len(part))})", part)
            conn.commit()
        conn.close()
        if ids:
            with self._lock:
                if self._index is not None:
                    self._index.drop(ids)
        return len(ids)

    def stats(self) -> Dict[str, Any]:
        conn = _connect(self.db_path)
        total, archived = conn.execute("SELECT COUNT(*), COALESCE(SUM(archived), 0) FROM episodes").fetchone()
        conn.close()
        return {"episodes": total, "archived": archived, "active": total - archived,
                "indexed": len(self._index.ids) if self._index is not None else None, "model": self.embedder.model_id}

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
        _add_column_if_missing(conn, "semantic_facts", "emb_model", "TEXT")
        conn.commit()
        # Facts distilled only from private chats used to be stored as public: rescope them by their sources
        leaked = []
        for r in conn.execute("SELECT id, sources FROM semantic_facts WHERE visibility = 'public'").fetchall():
            try:
                src = [int(i) for i in json.loads(r["sources"] or "[]")]
            except (TypeError, ValueError):
                continue
            if src:
                vis = {v for (v,) in conn.execute(f"SELECT visibility FROM episodes WHERE id IN ({','.join('?' * len(src))})",
                                                  src).fetchall()}
                if vis == {VISIBILITY_PRIVATE}:
                    leaked.append(r["id"])
        if leaked:
            conn.executemany("UPDATE semantic_facts SET visibility = 'private' WHERE id = ?", [(i,) for i in leaked])
            conn.commit()
        rows = conn.execute("SELECT id, fact FROM semantic_facts WHERE emb_model IS NULL OR emb_model != ?",
                            (embedder.model_id,)).fetchall()
        if rows:                                  # vectors from another model: re-embed (facts are few)
            vecs = embedder.embed_many([r["fact"] or "" for r in rows])
            conn.executemany("UPDATE semantic_facts SET embedding = ?, emb_model = ? WHERE id = ?",
                             [(Embedder.to_bytes(v), embedder.model_id, r["id"]) for v, r in zip(vecs, rows)])
            conn.commit()
        conn.close()

    def relevant(self, user_id: str, query: str, k: int = 3, exclude: Iterable[int] = ()) -> List[Dict[str, Any]]:
        """Facts about this person that bear on what is being said now (not just the strongest ones)."""
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT id, t, user_id, fact, confidence, sources, embedding FROM semantic_facts "
                            "WHERE user_id = ? AND emb_model = ?", (user_id, self.embedder.model_id)).fetchall()
        conn.close()
        skip = set(exclude)
        rows = [r for r in rows if r["id"] not in skip and r["embedding"] is not None]
        if not rows or not query.strip():
            return []
        sims = self.embedder.scores(np.stack([Embedder.from_bytes(r["embedding"]) for r in rows]), self.embedder.embed(query))
        order = [i for i in np.argsort(-sims)[:k] if sims[i] >= 0.12]
        return [{**{key: rows[i][key] for key in ("id", "t", "user_id", "fact", "confidence", "sources")},
                 "similarity": float(sims[i])} for i in order]

    def public_relevant(self, query: str, k: int = 4, exclude_user: str = None) -> List[Dict[str, Any]]:
        """Shareable facts (about anyone else) that bear on what is being said now."""
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT id, t, user_id, fact, confidence, embedding FROM semantic_facts "
                            "WHERE visibility = 'public' AND emb_model = ? AND user_id != ?",
                            (self.embedder.model_id, exclude_user or "")).fetchall()
        conn.close()
        rows = [r for r in rows if r["embedding"] is not None]
        if not rows or not query.strip():
            return []
        sims = self.embedder.scores(np.stack([Embedder.from_bytes(r["embedding"]) for r in rows]), self.embedder.embed(query))
        return [{key: rows[i][key] for key in ("id", "t", "user_id", "fact", "confidence")}
                for i in np.argsort(-sims)[:k] if sims[i] >= 0.15]

    def add(self, user_id: str, fact: str, sources: List[int], confidence: float = 0.7,
            visibility: str = VISIBILITY_PUBLIC):
        """Insert, or reinforce an existing fact that says the same thing."""
        v = self.embedder.embed(fact)
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT * FROM semantic_facts WHERE user_id = ? AND visibility = ?",
                            (user_id, visibility)).fetchall()
        for r in rows:
            if r["embedding"] is not None and self.embedder.similarity(v, Embedder.from_bytes(r["embedding"])) > 0.9:
                merged = sorted(set(json.loads(r["sources"] or "[]")) | set(sources))
                conn.execute("UPDATE semantic_facts SET confidence = ?, sources = ?, t = ? WHERE id = ?",
                             (min(1.0, r["confidence"] + 0.1), json.dumps(merged), clock.now(), r["id"]))
                conn.commit()
                conn.close()
                return
        conn.execute("INSERT INTO semantic_facts (t, user_id, fact, confidence, sources, embedding, visibility, emb_model) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (clock.now(), user_id, fact, confidence, json.dumps(sorted(set(sources))),
                      Embedder.to_bytes(v), visibility, self.embedder.model_id))
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
