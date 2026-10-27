"""
Level 8o: her memory of the game, recalled by meaning - where she was, what she did, what she built,
learned, lost and found in Minecraft.

Her QQ life already has a vector memory (l2_memory/embedding.py + layers.py): every episode is a vector,
and "what was it like" is answered by the nearest vectors, not by keywords. The game is her other life, and
it used the same idea only in words: mc_memory.Experience is keyword lessons, mc_places.Places is a map of
coordinates. That is enough to know where home is, not enough to remember by meaning "上次在矿洞里碰到
铁是在哪一层，怎么挖的".

This module is the game's own vector memory:

  kind: event / place / object / lesson / build / quest / death / encounter
  text: one natural Chinese sentence ("我在(-1155,12,280)挖掉了铁矿石")
  tags, pos x/y/z + dim, items, importance 0-1, t, session id
  vector BLOB (float16) + emb_model        <- the same space as her normal memory

remember() writes, recall() reads with a single matrix product (fast over thousands of records), boosted a
little by proximity to `near`, by recency and by importance. record_action() turns a tool call and its result
into the short sentence a player would remember; scene()/brief() renders the compact block that goes into
the play prompt ("我记得的最近的事", "关于这里", "教训").

Vectors from different models never mix: each row carries `emb_model`, and on open every row from another
model is re-embedded (the SemanticMemory pattern in l2_memory/layers.py). With no ONNX model (tests) the
Hash-only Embedder is used and everything still works, by character n-grams.
"""
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.levels.l2_memory.embedding import Embedder, default_model_dir
from src.levels.l8_agency.mc_places import direction

logger = logging.getLogger(__name__)

KINDS = ("event", "place", "object", "lesson", "build", "quest", "death", "encounter")

# how much a memory's rank can be moved by things other than meaning (the vector score spans roughly 0..1)
PROXIMITY_WEIGHT = 0.12          # a memory at the same spot adds at most this much
PROXIMITY_SCALE = 48.0           # blocks; the boost fades with distance over this many blocks
RECENCY_WEIGHT = 0.04            # remembered a minute ago vs a year ago
RECENCY_DAYS = 7.0
IMPORTANCE_WEIGHT = 0.06         # what mattered counts a little more
NEAR_GATE_BLOCKS = 24.0          # a strongly nearby, important memory is admitted even if the wording differs
NEAR_GATE_IMPORTANCE = 0.5

# ids -> the word a person would use. Unknown ids fall back to the id itself (last part).
CN_NAMES = {
    "iron_ore": "铁矿石", "deepslate_iron_ore": "深层铁矿石", "raw_iron": "粗铁", "iron_ingot": "铁锭",
    "coal_ore": "煤矿石", "deepslate_coal_ore": "深层煤矿石", "coal": "煤炭", "charcoal": "木炭",
    "diamond_ore": "钻石矿石", "deepslate_diamond_ore": "深层钻石矿石", "diamond": "钻石",
    "gold_ore": "金矿石", "deepslate_gold_ore": "深层金矿石", "raw_gold": "粗金", "gold_ingot": "金锭",
    "redstone_ore": "红石矿石", "deepslate_redstone_ore": "深层红石矿石", "redstone": "红石",
    "copper_ore": "铜矿石", "deepslate_copper_ore": "深层铜矿石", "raw_copper": "粗铜", "copper_ingot": "铜锭",
    "lapis_ore": "青金石矿石", "lapis_lazuli": "青金石", "emerald_ore": "绿宝石矿石", "emerald": "绿宝石",
    "ancient_debris": "远古残骸", "netherite_scrap": "下界合金碎片",
    "oak_log": "橡木原木", "spruce_log": "云杉原木", "birch_log": "白桦原木", "jungle_log": "丛林原木",
    "acacia_log": "金合欢原木", "dark_oak_log": "深色橡木原木", "mangrove_log": "红树林原木",
    "cherry_log": "樱花原木", "oak_planks": "橡木木板", "spruce_planks": "云杉木板", "stick": "木棍",
    "cobblestone": "圆石", "stone": "石头", "deepslate": "深板岩", "dirt": "泥土", "grass_block": "草方块",
    "sand": "沙子", "gravel": "沙砾", "clay": "黏土", "flint": "燧石",
    "crafting_table": "工作台", "chest": "箱子", "barrel": "木桶", "furnace": "熔炉",
    "blast_furnace": "高炉", "torch": "火把", "ladder": "梯子", "scaffolding": "脚手架",
    "white_bed": "床", "red_bed": "床", "blue_bed": "床", "bed": "床",
    "wooden_pickaxe": "木镐", "stone_pickaxe": "石镐", "iron_pickaxe": "铁镐", "golden_pickaxe": "金镐",
    "diamond_pickaxe": "钻石镐", "netherite_pickaxe": "下界合金镐",
    "wooden_axe": "木斧", "stone_axe": "石斧", "iron_axe": "铁斧", "diamond_axe": "钻石斧",
    "wooden_shovel": "木锹", "stone_shovel": "石锹", "iron_shovel": "铁锹", "diamond_shovel": "钻石锹",
    "wooden_sword": "木剑", "stone_sword": "石剑", "iron_sword": "铁剑", "diamond_sword": "钻石剑",
    "bow": "弓", "arrow": "箭", "shield": "盾牌", "bucket": "桶", "water_bucket": "水桶", "lava_bucket": "熔岩桶",
    "leather_helmet": "皮革帽子", "leather_chestplate": "皮革外套", "leather_leggings": "皮革裤子",
    "leather_boots": "皮革靴子", "iron_helmet": "铁头盔", "iron_chestplate": "铁胸甲",
    "iron_leggings": "铁护腿", "iron_boots": "铁靴子", "diamond_helmet": "钻石头盔",
    "diamond_chestplate": "钻石胸甲", "diamond_leggings": "钻石护腿", "diamond_boots": "钻石靴子",
    "apple": "苹果", "bread": "面包", "wheat": "小麦", "carrot": "胡萝卜", "potato": "土豆",
    "baked_potato": "烤土豆", "raw_beef": "生牛肉", "cooked_beef": "熟牛肉", "raw_porkchop": "生猪排",
    "cooked_porkchop": "熟猪排", "raw_chicken": "生鸡肉", "cooked_chicken": "熟鸡肉",
    "raw_mutton": "生羊肉", "cooked_mutton": "熟羊肉", "egg": "鸡蛋", "sugar": "糖",
    "bone": "骨头", "string": "线", "gunpowder": "火药", "feather": "羽毛", "leather": "皮革",
    "home": "家", "ore": "矿点", "death": "死亡位置", "spot": "地点",
    "campfire": "营火", "boat": "船", "oak_boat": "橡木船", "paper": "纸", "book": "书",
}

# what a tool call is called in a sentence, when the tool itself is the whole story
SKILL_SENTENCES = {
    "chop_tree": "我砍了一棵树，捡了些原木",
    "dig_down": "我挖楼梯往下走",
    "tunnel": "我挖了一条矿道",
    "hunt": "我打猎弄到了吃的",
    "dress": "我穿上了最好的盔甲，盾牌放在副手",
    "declutter": "我把背包里的杂物扔掉了",
    "go_home": "我回家了一趟，把东西放进箱子",
    "equip": "我换上了合适的工具",
    "tidy": "我把工具整理到快捷栏",
    "collect_drops": "我把地上的东西捡了起来",
    "status": "我看了看自己的进度",
    "stop": "我停了手上的活",
}

def _cn(item: Optional[str]) -> str:
    """minecraft:iron_ore -> 铁矿石 (unknown ids keep their own name)."""
    if not item:
        return "东西"
    ident = str(item).strip()
    short = ident.split(":")[-1].split("/")[-1]
    return CN_NAMES.get(short, short)


def _pos_of(p: Any) -> Optional[Tuple[float, float, float]]:
    """(x,y,z) from a tuple, a list or a dict; None if it is not a position."""
    if p is None:
        return None
    if isinstance(p, dict):
        if not all(k in p for k in ("x", "y", "z")):
            return None
        p = (p["x"], p["y"], p["z"])
    try:
        if len(p) >= 3:
            return (float(p[0]), float(p[1]), float(p[2]))
    except (TypeError, ValueError):
        pass
    return None


def _norm(items: Any) -> Any:
    """Items as stored: a dict id->count, a list of ids, or one id; JSON-serializable."""
    if items is None:
        return None
    if isinstance(items, dict):
        return {str(k): int(v) if isinstance(v, (int, float)) else v for k, v in items.items()}
    if isinstance(items, str):
        return [x.strip() for x in items.split(",") if x.strip()]
    if isinstance(items, (list, tuple, set)):
        return [str(x) for x in items]
    return [str(items)]


def _when(t: Optional[float]) -> str:
    """The way a person says when: 刚才 / 12分钟前 / 3小时前 / 2天前."""
    age = max(0.0, time.time() - float(t or time.time()))
    if age < 60:
        return "刚才"
    if age < 3600:
        return f"{int(age // 60)}分钟前"
    if age < 86400:
        return f"{int(age // 3600)}小时前"
    return f"{int(age // 86400)}天前"


def _ids(text: str) -> List[str]:
    return re.findall(r"[a-z_]+:[a-z0-9_/]+", str(text or "").lower())


def _short(text: Any, n: int = 90) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s[:n] + ("…" if len(s) > n else "")


def _failed(result: Any, ok: Optional[bool]) -> bool:
    """Did the action fail? An explicit ok wins; otherwise the wording of the result decides."""
    if ok is not None:
        return not ok
    s = str(result or "")
    return any(mark in s for mark in ("出错", "没成", "没能", "没挖", "没放", "走不到", "挖不动",
                                      "找不到", "没找到", "不够", "够不着", "失败", "是空气", "没开始", "超时", "取消", "停下", "没有我的", "中止",
                                      "还没", "没走", "走不", "不知道", "没法", "还差", "卡住", "用完", "暂停",
                                      "未确认", "未观察", "没有燃料", "没有能", "没拿", "没地方", "没烧", "拿不出"))


class _Index:
    """The records of the current model's space in memory: one matrix for the vectors and the few columns
    recall filters and ranks by. Rebuilt lazily after a write; scoring then is a single matrix product."""

    def __init__(self, dim: int):
        self.dim = dim
        self.ids = np.zeros(0, dtype=np.int64)
        self.M = np.zeros((0, dim), dtype=np.float32)
        self.t = np.zeros(0, dtype=np.float64)
        self.importance = np.zeros(0, dtype=np.float32)
        self.kinds = np.zeros(0, dtype=object)
        self.pos = np.zeros((0, 3), dtype=np.float32)
        self.dims = np.zeros(0, dtype=object)
        self.texts: List[str] = []
        self.tags: List[List[str]] = []
        self.items: List[Any] = []
        self.counts = np.zeros(0, dtype=np.int64)
        self.sessions = np.zeros(0, dtype=object)

    @property
    def n(self) -> int:
        return len(self.ids)


class GameMemory:
    """
    Level 8o: the game as she remembers it, by meaning.

    GameMemory(path, embedder=None, session=None)
        path      a .sqlite file (or a directory -> mc_game_memory.sqlite inside); ":memory:" if None
        embedder  her Embedder; the same instance as gw.episodes.embedder keeps one shared vector space
        session   session id written on new records (start_session() changes it)

    remember(text, kind="event", pos=None, dim=None, items=None, importance=0.5, tags=None, session=None,
             t=None) -> int
    recall(query, k=6, near=None, kinds=None, min_score=0.25) -> [dict]
    recent(n=5, kinds=None) -> [dict]
    nearby(near, radius=48, k=5, kinds=None) -> [dict]
    forget(memory_id) -> bool
    consolidate(...) / sleep() -> dict
    stats() -> dict
    record_action(tool, args=None, result="", ok=None, pos=None) -> Optional[int]
    scene(near=None, query="", k=4, recent_n=4) / brief(...) -> str
    ensure_model() -> int   (re-embeds rows from another model; called on open)
    """

    PROXIMITY = PROXIMITY_WEIGHT
    SCALE = PROXIMITY_SCALE

    def __init__(self, path: Optional[str] = None, embedder: Optional[Embedder] = None, session: Optional[str] = None):
        if embedder is None:
            embedder = Embedder(model_dir=default_model_dir())
        self.embedder = embedder
        self.session = session or time.strftime("%Y%m%d-%H%M%S")
        if path is None:
            db_path = ":memory:"
        elif os.path.isdir(path):
            db_path = os.path.join(path, "mc_game_memory.sqlite")
        else:
            db_path = path
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.RLock()
        self._index_cache: Optional[_Index] = None
        self.recall_ms: List[float] = []          # recent recall timings, for the console / tests
        self.last_recall_ms = 0.0
        self.last_migration = 0                   # rows re-embedded on open because the model changed
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        with self._lock:
            self._conn.execute('''CREATE TABLE IF NOT EXISTS game_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t REAL,
                kind TEXT,
                text TEXT,
                tags TEXT,          -- JSON list
                x REAL, y REAL, z REAL,
                dim TEXT,
                items TEXT,         -- JSON dict or list
                importance REAL DEFAULT 0.5,
                session TEXT,
                embedding BLOB,     -- float16, the embedder's space
                emb_model TEXT,
                n INTEGER DEFAULT 1,
                last_recall REAL
            )''')
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_game_memory_t ON game_memory(t)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_game_memory_kind ON game_memory(kind)")
            self._conn.commit()
        self.last_migration = self.ensure_model()

    # --- vectors in the current model's space ----------------------------------------------

    def ensure_model(self, batch: int = 64) -> int:
        """Re-embed every record whose vector came from another model (or from before models were recorded)."""
        mid = self.embedder.model_id
        with self._lock:
            rows = self._conn.execute("SELECT id, text FROM game_memory WHERE emb_model IS NULL OR emb_model != ?",
                                      (mid,)).fetchall()
            for start in range(0, len(rows), batch):
                chunk = rows[start:start + batch]
                vecs = self.embedder.embed_many([r["text"] or "" for r in chunk])
                self._conn.executemany("UPDATE game_memory SET embedding = ?, emb_model = ? WHERE id = ?",
                                       [(Embedder.to_bytes(v), mid, r["id"]) for v, r in zip(vecs, chunk)])
                self._conn.commit()
            self._index_cache = None
        if rows:
            logger.info(f"[GameMemory] re-embedded {len(rows)} game memories for {mid}")
        return len(rows)

    def _index(self) -> _Index:
        """The in-memory index, rebuilt after writes so recall is one matrix product."""
        with self._lock:
            if self._index_cache is not None:
                return self._index_cache
            idx = _Index(self.embedder.dim)
            rows = self._conn.execute(
                "SELECT id, t, kind, text, tags, items, n, session, importance, x, y, z, dim, embedding "
                "FROM game_memory WHERE embedding IS NOT NULL AND emb_model = ? ORDER BY id",
                (self.embedder.model_id,)).fetchall()
            vecs, ids, ts, imps, kinds, poss, dims = [], [], [], [], [], [], []
            texts, tags, items, counts, sessions = [], [], [], [], []
            for r in rows:
                v = Embedder.from_bytes(r["embedding"])
                if len(v) != self.embedder.dim:
                    continue                      # a stray vector from another dimension: never compare
                vecs.append(v)
                ids.append(r["id"])
                ts.append(r["t"] or 0.0)
                imps.append(float(r["importance"] if r["importance"] is not None else 0.5))
                kinds.append(r["kind"] or "event")
                poss.append((r["x"], r["y"], r["z"]) if r["x"] is not None else (np.nan, np.nan, np.nan))
                dims.append(r["dim"])
                texts.append(r["text"] or "")
                tags.append(json.loads(r["tags"] or "[]"))
                items.append(json.loads(r["items"] or "null"))
                counts.append(int(r["n"] or 1))
                sessions.append(r["session"])
            if vecs:
                idx.M = np.stack(vecs).astype(np.float32)
                idx.ids = np.asarray(ids, dtype=np.int64)
                idx.t = np.asarray(ts, dtype=np.float64)
                idx.importance = np.asarray(imps, dtype=np.float32)
                idx.kinds = np.asarray(kinds, dtype=object)
                idx.pos = np.asarray(poss, dtype=np.float32)
                idx.dims = np.asarray(dims, dtype=object)
                idx.texts = texts
                idx.tags = tags
                idx.items = items
                idx.counts = np.asarray(counts, dtype=np.int64)
                idx.sessions = np.asarray(sessions, dtype=object)
            self._index_cache = idx
            return idx

    def _dirty(self) -> None:
        self._index_cache = None

    # --- writing ---------------------------------------------------------------------------

    def remember(self, text: str, kind: str = "event", pos: Any = None, dim: Optional[str] = None,
                 items: Any = None, importance: float = 0.5, tags: Optional[Iterable[str]] = None,
                 session: Optional[str] = None, t: Optional[float] = None) -> int:
        """Write one memory a person would have; returns its id."""
        text = _short(text, 220)
        if not text:
            raise ValueError("a memory needs text")
        kind = kind if kind in KINDS else "event"
        p = _pos_of(pos)
        x, y, z = p if p else (None, None, None)
        imp = max(0.0, min(1.0, float(importance if importance is not None else 0.5)))
        vec = self.embedder.embed(text)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO game_memory (t, kind, text, tags, x, y, z, dim, items, importance, session, "
                "embedding, emb_model, n) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (time.time() if t is None else float(t), kind, text,
                 json.dumps([str(x) for x in (tags or [])], ensure_ascii=False),
                 x, y, z, dim or (pos.get("dim") or pos.get("dimension") if isinstance(pos, dict) else None),
                 json.dumps(_norm(items), ensure_ascii=False) if items is not None else None,
                 imp, session or self.session, Embedder.to_bytes(vec), self.embedder.model_id))
            self._conn.commit()
            self._dirty()
            return int(cur.lastrowid)

    def forget(self, memory_id: int) -> bool:
        """Drop one record (a place that turned out wrong, a memory she does not want to keep)."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM game_memory WHERE id = ?", (int(memory_id),))
            self._conn.commit()
            self._dirty()
            return cur.rowcount > 0

    def remember_place(self, kind: str, pos: Any, note: str = "", items: Any = None,
                       importance: float = 0.6, dim: Optional[str] = None) -> int:
        """Convenience: a place/object/build memory from coordinates ("home 在(..)", "chest 里有铁")."""
        p = _pos_of(pos)
        if p is None:
            raise ValueError("a place memory needs a position")
        text = note.strip() or f"我把{_cn(kind)}记在({int(p[0])},{int(p[1])},{int(p[2])})"
        return self.remember(text, kind=kind if kind in KINDS else "place", pos=p, dim=dim,
                             items=items, importance=importance, tags=[kind])

    # --- reading ---------------------------------------------------------------------------

    def _threshold(self, min_score: float) -> float:
        """The caller's 0.25 is on the scale of the combined (model + n-gram) vector. The hash-only
        embedder (no ONNX model, tests) scores on a much lower scale, so the same threshold is scaled
        down with it; otherwise recall by wording alone would return nothing."""
        return float(min_score) if self.embedder.backend == "onnx" else float(min_score) * 0.2

    def _record(self, idx: _Index, i: int, score: Optional[float] = None) -> Dict[str, Any]:
        pos = None
        if not math.isnan(float(idx.pos[i][0])):
            pos = {"x": float(idx.pos[i][0]), "y": float(idx.pos[i][1]), "z": float(idx.pos[i][2])}
        rec = {"id": int(idx.ids[i]), "t": float(idx.t[i]), "when": _when(float(idx.t[i])),
               "kind": str(idx.kinds[i]), "text": idx.texts[i], "tags": idx.tags[i], "items": idx.items[i],
               "importance": float(idx.importance[i]), "n": int(idx.counts[i]), "pos": pos,
               "dim": idx.dims[i] if idx.dims[i] not in (None, "") else None,
               "session": idx.sessions[i]}
        if score is not None:
            rec["score"] = round(float(score), 4)
        return rec

    def _boosts(self, idx: _Index, near: Optional[Dict[str, Any]]) -> np.ndarray:
        """Proximity + recency + importance: the small pushes on top of the vector score."""
        n = idx.n
        boost = np.zeros(n, dtype=np.float32)
        if n == 0:
            return boost
        age = np.maximum(0.0, time.time() - idx.t)
        boost += (RECENCY_WEIGHT * np.exp(-age / (RECENCY_DAYS * 86400.0))).astype(np.float32)
        boost += IMPORTANCE_WEIGHT * idx.importance
        if near is not None:
            p = _pos_of(near)
            if p is not None:
                diff = idx.pos - np.asarray(p, dtype=np.float32)
                d = np.sqrt(np.sum(diff * diff, axis=1))       # nan where a record has no position
                near_dim = near.get("dim") if isinstance(near, dict) else None
                good = np.isfinite(d)
                if near_dim:
                    good &= np.asarray([dim in (None, "", near_dim) for dim in idx.dims], dtype=bool)
                d = np.where(good, d, np.inf)
                boost += (PROXIMITY_WEIGHT * np.exp(-d / PROXIMITY_SCALE)).astype(np.float32)
                self._near_dist = d
                return boost
        self._near_dist = None
        return boost

    def recall(self, query: str, k: int = 6, near: Any = None, kinds: Optional[Sequence[str]] = None,
               min_score: float = 0.25) -> List[Dict[str, Any]]:
        """The memories that bear on `query` now, best first, with a small boost for being close to `near`."""
        t0 = time.perf_counter()
        with self._lock:
            idx = self._index()
            if idx.n == 0:
                self.last_recall_ms = (time.perf_counter() - t0) * 1000.0
                return []
            q = self.embedder.embed(query or "")
            sims = np.asarray(self.embedder.scores(idx.M, q), dtype=np.float32).reshape(-1)
            near = near if isinstance(near, dict) else ({"x": near[0], "y": near[1], "z": near[2]}
                                                        if _pos_of(near) else None)
            boost = self._boosts(idx, near)
            rank = sims + boost
            gate = sims >= self._threshold(min_score)
            near_dist = getattr(self, "_near_dist", None)
            if near_dist is not None:
                # what happened right here matters even when the words are different
                gate |= (near_dist <= NEAR_GATE_BLOCKS) & (idx.importance >= NEAR_GATE_IMPORTANCE)
            if kinds:
                gate &= np.isin(idx.kinds, [str(x) for x in kinds])
            out: List[Dict[str, Any]] = []
            if gate.any():
                order = np.argsort(-rank)
                for i in order:
                    if not gate[i]:
                        continue
                    rec = self._record(idx, int(i), rank[i])
                    rec["similarity"] = round(float(sims[i]), 4)
                    if near_dist is not None and np.isfinite(near_dist[i]):
                        rec["distance"] = round(float(near_dist[i]), 1)
                    out.append(rec)
                    if len(out) >= max(0, int(k)):
                        break
        self.last_recall_ms = (time.perf_counter() - t0) * 1000.0
        self.recall_ms.append(self.last_recall_ms)
        del self.recall_ms[:-50]
        return out

    def recent(self, n: int = 5, kinds: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """The last things that happened, newest first."""
        sql = "SELECT id, t, kind, text, tags, items, importance, n, x, y, z, dim FROM game_memory"
        args: List[Any] = []
        if kinds:
            sql += f" WHERE kind IN ({','.join('?' * len(kinds))})"
            args = [str(x) for x in kinds]
        sql += " ORDER BY t DESC, id DESC LIMIT ?"
        args.append(max(0, int(n)))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            out.append({"id": r["id"], "t": r["t"], "when": _when(r["t"]), "kind": r["kind"], "text": r["text"],
                        "tags": json.loads(r["tags"] or "[]"), "importance": r["importance"], "n": r["n"],
                        "pos": {"x": r["x"], "y": r["y"], "z": r["z"]} if r["x"] is not None else None,
                        "dim": r["dim"], "items": json.loads(r["items"] or "null")})
        return out

    def nearby(self, near: Any, radius: float = 48.0, k: int = 5,
               kinds: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """What she remembers about this very place (independent of wording): closest first, then newest."""
        p = _pos_of(near)
        if p is None:
            return []
        with self._lock:
            idx = self._index()
            if idx.n == 0:
                return []
            diff = idx.pos - np.asarray(p, dtype=np.float32)
            d = np.sqrt(np.sum(diff * diff, axis=1))           # nan where a record has no position
            good = np.isfinite(d) & (d <= float(radius))
            near_dim = near.get("dim") if isinstance(near, dict) else None
            if near_dim:
                good &= np.asarray([dim in (None, "", near_dim) for dim in idx.dims], dtype=bool)
            if kinds:
                good &= np.isin(idx.kinds, [str(x) for x in kinds])
            order = np.argsort(np.where(good, d, np.inf))
            out = []
            for i in order:
                if not good[i] or len(out) >= max(0, int(k)):
                    break
                rec = self._record(idx, int(i))
                rec["distance"] = round(float(d[i]), 1)
                rec["direction"] = direction({"x": p[0], "y": p[1], "z": p[2]}, rec["pos"]) if rec["pos"] else ""
                out.append(rec)
        return out

    # --- sleep: decay, merge, prune --------------------------------------------------------

    def consolidate(self, decay: float = 0.97, lesson_decay: float = 0.995, dup_threshold: float = 0.96,
                    min_importance: float = 0.05, keep_importance: float = 0.7, max_records: int = 5000,
                    merge_recent: int = 400) -> Dict[str, int]:
        """What sleep does to a memory: fade what does not matter, merge what says the same thing, and drop
        the faded leftovers - but never the important ones (importance >= keep_importance) nor lessons."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, t, kind, text, tags, importance, n, embedding, emb_model FROM game_memory").fetchall()
            if not rows:
                return {"decayed": 0, "merged": 0, "pruned": 0, "kept": 0}
            # 1. fade: importance slowly approaches zero for what is not important; lessons hardly fade and
            # what really mattered (>= keep_importance) does not fade at all, so sleep can never lose it
            faded = []
            for r in rows:
                imp = float(r["importance"] if r["importance"] is not None else 0.5)
                if r["kind"] == "lesson":
                    new = max(0.01, imp * lesson_decay)
                elif imp >= keep_importance:
                    new = imp
                else:
                    new = max(0.01, imp * decay)
                if abs(new - imp) > 1e-6:
                    faded.append((new, r["id"]))
            if faded:
                self._conn.executemany("UPDATE game_memory SET importance = ? WHERE id = ?", faded)
                self._conn.commit()
            # 2. merge what says the same thing: only among the recent ones and the lessons, so sleep stays cheap
            by_id = {r["id"]: dict(r) for r in rows}
            recent = sorted(by_id.values(), key=lambda r: (r["t"] or 0), reverse=True)[:merge_recent]
            lessons = [r for r in by_id.values() if r["kind"] == "lesson"]
            candidates = {r["id"]: r for r in recent + lessons}
            merged = 0
            for kind in {r["kind"] for r in candidates.values()}:
                group = [r for r in candidates.values() if r["kind"] == kind and r["embedding"] is not None
                         and r["emb_model"] == self.embedder.model_id]
                group = [r for r in group if len(Embedder.from_bytes(r["embedding"])) == self.embedder.dim]
                if len(group) < 2:
                    continue
                M = np.stack([Embedder.from_bytes(r["embedding"]) for r in group]).astype(np.float32)
                sims = M @ M.T - self.embedder.offset                      # Embedder.scores(M, q) == M@q-offset
                sims = np.triu(np.asarray(sims, dtype=np.float32), k=1)
                gone = set()                               # indices already absorbed into a kept row
                totals: Dict[int, Dict[str, Any]] = {}     # keep id -> merged fields, accumulated over rows
                for j in range(1, len(group)):
                    if j in gone:
                        continue
                    # the first earlier row that says the same thing is the one kept
                    hits = [int(i) for i in np.where(sims[:j, j] >= dup_threshold)[0] if int(i) not in gone]
                    if not hits:
                        continue
                    keep = group[hits[0]]
                    kid = int(keep["id"])
                    acc = totals.setdefault(kid, {"n": int(keep["n"] or 1),
                                                  "importance": float(keep["importance"] or 0.5),
                                                  "tags": set(json.loads(keep["tags"] or "[]")),
                                                  "t": float(keep["t"] or 0)})
                    drop = group[j]
                    gone.add(j)
                    acc["n"] += int(drop["n"] or 1)
                    acc["importance"] = max(acc["importance"], float(drop["importance"] or 0.5))
                    acc["tags"] |= set(json.loads(drop["tags"] or "[]"))
                    acc["t"] = max(acc["t"], float(drop["t"] or 0))
                    self._conn.execute("DELETE FROM game_memory WHERE id = ?", (drop["id"],))
                    merged += 1
                for kid, acc in totals.items():
                    self._conn.execute("UPDATE game_memory SET n = ?, importance = ?, tags = ?, t = ? WHERE id = ?",
                                       (acc["n"], min(1.0, acc["importance"] + 0.05),
                                        json.dumps(sorted(acc["tags"]), ensure_ascii=False), acc["t"], kid))
                if merged:
                    self._conn.commit()
            # 3. prune: the faded leftovers, never the important ones; then keep the table bounded
            pruned = 0
            keepers = self._conn.execute(
                "SELECT id, kind, importance, t FROM game_memory").fetchall()
            leftover = [r for r in keepers if float(r["importance"] or 0.5) < min_importance
                        and r["kind"] != "lesson" and float(r["importance"] or 0.5) < keep_importance]
            if leftover:
                self._conn.executemany("DELETE FROM game_memory WHERE id = ?", [(r["id"],) for r in leftover])
                pruned += len(leftover)
            over = len(keepers) - len(leftover) - max_records
            if over > 0:
                deletable = [r for r in keepers if r not in leftover
                             and r["kind"] != "lesson" and float(r["importance"] or 0.5) < keep_importance]
                deletable.sort(key=lambda r: (float(r["importance"] or 0.5), float(r["t"] or 0)))
                drop = deletable[:over]
                self._conn.executemany("DELETE FROM game_memory WHERE id = ?", [(r["id"],) for r in drop])
                pruned += len(drop)
            if faded or merged or pruned:
                self._conn.commit()
            kept = self._conn.execute("SELECT COUNT(*) FROM game_memory").fetchone()[0]
            self._dirty()
        return {"decayed": len(faded), "merged": merged, "pruned": pruned, "kept": int(kept)}

    def sleep(self, **kwargs) -> Dict[str, int]:
        """Her night's compression of the game memory (same as consolidate)."""
        return self.consolidate(**kwargs)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM game_memory").fetchone()[0]
            by_kind = {r["kind"]: r["n"] for r in self._conn.execute(
                "SELECT kind, COUNT(*) AS n FROM game_memory GROUP BY kind ORDER BY n DESC").fetchall()}
            span = self._conn.execute("SELECT MIN(t) AS lo, MAX(t) AS hi, "
                                      "SUM(CASE WHEN x IS NOT NULL THEN 1 ELSE 0 END) AS placed "
                                      "FROM game_memory").fetchone()
            sessions = self._conn.execute("SELECT COUNT(DISTINCT session) FROM game_memory").fetchone()[0]
        return {"records": int(total), "by_kind": by_kind, "with_position": int(span["placed"] or 0),
                "sessions": int(sessions or 0), "oldest": span["lo"], "newest": span["hi"],
                "model_id": self.embedder.model_id, "backend": self.embedder.backend,
                "dim": self.embedder.dim, "session": self.session, "last_recall_ms": round(self.last_recall_ms, 3)}

    # --- what her hands just did -----------------------------------------------------------

    def record_action(self, tool: str, args: Optional[Dict[str, Any]] = None, result: Any = "",
                      ok: Optional[bool] = None, pos: Any = None, dim: Optional[str] = None) -> Optional[int]:
        """Turn one tool call and its result into the short Chinese sentence a player would remember.

        Failed actions become a low-importance event with the reason ("走不到..."), so the next attempt can
        recall why it did not work; successes become event/build/object/quest/lesson memories. Tools that are
        only reading (looking, checking recipes...) are not memorised and return None.
        """
        def remember(text, **kwargs):
            return self.remember(text, dim=dim, **kwargs)

        name = str(tool or "")
        args = args or {}
        text = _short(result, 120)
        # Starting a background skill is not evidence it completed.
        if name == "mc_skill" and any(mark in text for mark in ("开始做", "已经在做", "在后台进行")):
            return None
        dim = dim or (pos.get("dim") or pos.get("dimension") if isinstance(pos, dict) else None)
        failed = _failed(result, ok)
        p = _pos_of(pos) or _pos_of(args)
        where = f"({int(p[0])},{int(p[1])},{int(p[2])})" if p else ""
        block = self._first_id(text, args.get("blocks"), args.get("block"))
        item = self._first_id(args.get("item"), text)

        if name == "death":
            return remember(f"我在{where or '世界里'}死了一次", kind="death", pos=p, importance=0.9,
                                 tags=["death"])
        if name == "mc_mine_block":
            subject = _cn(block) if block else "方块"
            if failed:
                return remember(f"我想挖{where}的{subject}，但没挖成：{text}",
                                     kind="event", pos=p, importance=0.4, tags=["失败", "挖矿"], items=block)
            return remember(f"我在{where}挖掉了{subject}", kind="event", pos=p, importance=0.5,
                                 tags=["挖矿"], items=block)
        if name == "mc_place_block":
            subject = _cn(item) if item else "方块"
            if failed:
                return remember(f"我想在{where}放{subject}，但没放成：{text}",
                                     kind="event", pos=p, importance=0.4, tags=["失败", "放置"], items=item)
            return remember(f"我在{where}放下了{subject}", kind="build", pos=p, importance=0.65,
                                 tags=["建造", subject], items=item)
        if name == "mc_craft":
            if failed:
                return remember(f"我想合成{item and _cn(item) or '东西'}，但没成：{text}",
                                     kind="event", importance=0.4, tags=["失败", "合成"])
            made = item and _cn(item) or _short(text, 60)
            return remember(f"我合成了{made}", kind="object", importance=0.6, tags=["合成"], items=item)
        if name == "mc_skill":
            return self._record_skill(args, text, failed, p, dim)
        if name == "mc_open_block":
            if failed:
                return None
            kind_of = "箱子" if "chest" in text.lower() or "barrel" in text.lower() else "容器"
            return remember(f"我打开了{where}那里的{kind_of}", kind="object", pos=p, importance=0.45,
                                 tags=["箱子"])
        if name in ("mc_container_transfer",):
            direction_ = "take" if str(args.get("direction")) == "take" else "put"
            what = _cn(item) if item else "东西"
            n = args.get("count")
            amount = f"{int(n)}个" if n else "一些"
            if failed:
                return remember(f"我想从{where}那里的箱子里{'拿' if direction_ == 'take' else '放'}{what}，"
                                     f"但没成：{text}", kind="event", pos=p, importance=0.35,
                                     tags=["失败", "箱子"], items=item)
            verb = "从箱子里拿了" if direction_ == "take" else "把一些"
            tail = "" if direction_ == "take" else "放进了箱子"
            return remember(f"我{verb}{amount}{what}{tail}", kind="object", pos=p, importance=0.55,
                                 tags=["箱子", what], items=item)
        if name == "mc_map":
            if args.get("mark"):
                return remember(f"我把这个地方记成「{_short(args['mark'], 30)}」{where}",
                                     kind="place", pos=p, importance=0.75, tags=["标记", "地方"])
            return None
        if name == "mc_remember":
            note = str(args.get("note") or "").strip()
            if not note:
                return None
            about = str(args.get("about") or "").strip()
            tags = ["经验"] + ([about] if about else [])
            return remember(f"我记下了一条经验：{_short(note, 80)}", kind="lesson", importance=0.8,
                                 tags=tags)
        if name == "mc_goal":
            if args.get("item"):
                return remember(f"我把目标换成了{_cn(args['item'])}", kind="quest", importance=0.7,
                                     tags=["目标"], items=str(args["item"]))
            return None
        if name == "mc_follow":
            who = str(args.get("player") or "").strip()
            if not who or who.lower() in ("stop", "停", "不跟了"):
                return remember("我不跟着走了", kind="event", importance=0.3, tags=["玩家"])
            return remember(f"我跟着{who}走", kind="encounter", importance=0.45, tags=["玩家", who])
        if name in ("mc_look", "mc_look_at", "mc_know", "mc_recipes", "mc_container_state",
                    "mc_container_click", "mc_find_blocks", "mc_who_is_near", "mc_rejoin_world"):
            return None                                   # looking and reading are not memories
        if failed and text:
            return remember(f"我试了「{name or '动作'}」，没成：{text}", kind="event", importance=0.35,
                                 tags=["失败"])
        return None

    def _record_skill(self, args: Dict[str, Any], text: str, failed: bool, p: Optional[Tuple], dim: Optional[str] = None) -> Optional[int]:
        def remember(text, **kwargs):
            return self.remember(text, dim=dim, **kwargs)

        name = str(args.get("name") or "").strip()
        if not name:
            return None
        item = self._first_id(args.get("item"), text)
        blocks = self._first_id(args.get("blocks"), text)
        where = f"({int(p[0])},{int(p[1])},{int(p[2])})" if p else ""
        n = self._count(text, r"挖了\s*(\d+)\s*块") or args.get("count")
        if failed:
            what = SKILL_SENTENCES.get(name) or f"做「{name}」"
            return remember(f"我试着{what.replace('我', '')}，但没成：{text}",
                                 kind="event", pos=p, importance=0.4, tags=["失败", name])
        if name == "mine":
            subject = _cn(blocks) if blocks else "矿石"
            amount = f"{int(n)}块" if n else "一些"
            loc = f"在{where}" if where else ""
            return remember(f"我{loc}挖了{amount}{subject}", kind="event", pos=p, importance=0.5,
                                 tags=["挖矿", subject], items=blocks)
        if name == "craft":
            return remember(f"我做成了{_cn(item) if item else '东西'}", kind="object", importance=0.6,
                                 tags=["合成"], items=item)
        if name == "place":
            subject = _cn(item) if item else "方块"
            loc = f"在{where}" if where else ""
            return remember(f"我{loc}放下了{subject}", kind="build", pos=p, importance=0.65,
                                 tags=["建造", subject], items=item)
        if name == "smelt":
            return remember(f"我烧了{_cn(item) if item else '东西'}", kind="object", importance=0.5,
                                 tags=["冶炼"], items=item)
        if name == "dig_down":
            to_y = args.get("y")
            return remember("我挖楼梯往下" + (f"到了 y={to_y}" if to_y is not None else ""),
                                 kind="event", pos=p, importance=0.45, tags=["挖矿"])
        if name == "tunnel":
            amount = f"{int(n)}格" if n else ""
            return remember(f"我挖了一条{amount}矿道", kind="event", pos=p, importance=0.45, tags=["挖矿"])
        if name == "set_home":
            loc = f"在{where}" if where else ""
            return remember(f"我{loc}设置了落脚点：{text}", kind="build", pos=p, importance=0.9,
                                 tags=["家", "建造"])
        if name == "build_hut":
            return remember(f"我在{where or '家址'}搭建起步屋，实际结果：{text}", kind="build", pos=p,
                            importance=0.8, tags=["家", "建造", "起步屋"])
        if name in ("build_portal", "enter_portal", "hunt_resource"):
            return remember(f"游戏行动「{name}」的实测结果：{text}", kind="event", pos=p,
                            importance=0.75, tags=[name])
        if name == "go_to":
            target = _pos_of((args.get("x"), args.get("y"), args.get("z"))) if args.get("x") is not None else p
            t_where = f"({int(target[0])},{int(target[1])},{int(target[2])})" if target else ""
            return remember(f"我走到了{t_where}" if t_where else "我走了一段路",
                                 kind="event", pos=target, importance=0.35, tags=["走路"])
        if name == "store":
            loc = f"{where}的" if where else "附近的"
            return remember(f"我把东西放进{loc}箱子", kind="object", pos=p, importance=0.5,
                                 tags=["箱子"])
        if name == "go_home":
            return remember("我回家了一趟", kind="event", importance=0.45, tags=["家"])
        if name == "chop_tree":
            return remember(SKILL_SENTENCES["chop_tree"], kind="event", pos=p, importance=0.45,
                                 tags=["伐木"])
        if name in SKILL_SENTENCES:
            return remember(SKILL_SENTENCES[name], kind="event", pos=p, importance=0.4, tags=[name])
        return None

    @staticmethod
    def _first_id(*texts: Any) -> Optional[str]:
        for text in texts:
            if text is None:
                continue
            found = _ids(str(text))
            if found:
                return found[0]
            for part in str(text).split(","):
                part = part.strip()
                if part and re.fullmatch(r"[a-z0-9_]+", part):
                    return part
        return None

    @staticmethod
    def _count(text: str, pattern: str) -> Optional[int]:
        m = re.search(pattern, text or "")
        return int(m.group(1)) if m else None

    # --- the block that goes into her play prompt ------------------------------------------

    def scene(self, near: Any = None, query: str = "", k: int = 4, recent_n: int = 4) -> str:
        """A compact block for her play prompt: what she just did, what she remembers about this place,
        about what she is working on, and the lessons that bear on it. "" when there is nothing yet.

        Call it when building the play turn: scene(near={"x":..,"y":..,"z":..}, query="铁矿石 铁镐 目标")
        """
        lines: List[str] = []
        seen = set()
        recent = self.recent(recent_n, kinds=("event", "build", "object", "place", "quest", "death", "encounter"))
        if recent:
            lines.append("我记得的最近的事：")
            for rec in recent:
                lines.append(f"  {rec['when']} {_short(rec['text'], 80)}")
                seen.add(rec["id"])
        here = self.nearby(near, radius=48, k=k) if near is not None else []
        here = [r for r in here if r["id"] not in seen][:k]
        if here:
            lines.append("关于这里：")
            for rec in here:
                arrow = f"，{rec.get('direction')}" if rec.get("direction") else ""
                lines.append(f"  {rec['when']} {_short(rec['text'], 70)}{arrow}")
                seen.add(rec["id"])
        if query and query.strip():
            got = [r for r in self.recall(query, k=k + 2, min_score=0.2)
                   if r["id"] not in seen and r["kind"] != "lesson"][:k]   # lessons get their own line
            if got:
                lines.append(f"关于「{_short(query, 14)}」：")
                for rec in got:
                    lines.append(f"  {rec['when']} {_short(rec['text'], 70)}")
                    seen.add(rec["id"])
            lessons = [r for r in self.recall(query, k=2, kinds=("lesson",), min_score=0.18)
                       if r["id"] not in seen][:2]
        else:
            lessons = self.recent(2, kinds=("lesson",))
        lessons = [r for r in lessons if r["id"] not in seen][:2]
        if lessons:
            lines.append("我记得的教训：")
            for rec in lessons:
                lines.append(f"  {_short(rec['text'], 80)}")
        if not lines or len(lines) == len([x for x in lines if x.endswith("：")]):
            return ""
        return "\n".join(lines[:14])

    def brief(self, near: Any = None, query: str = "", k: int = 4, recent_n: int = 4) -> str:
        """Same as scene(): the block the play prompt gets. Kept under this name too, for the wiring."""
        return self.scene(near=near, query=query, k=k, recent_n=recent_n)

    # --- housekeeping ----------------------------------------------------------------------

    def start_session(self, session: Optional[str] = None) -> str:
        self.session = session or time.strftime("%Y%m%d-%H%M%S")
        return self.session

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error:
                pass
