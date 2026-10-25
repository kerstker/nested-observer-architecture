"""
How well and how fast does she remember? A recall benchmark on her real episodes.

    python scripts/benchmark_memory.py build      # once: DeepSeek writes paraphrased cues (data/, never in git)
    python scripts/benchmark_memory.py run        # hash vs semantic embedder: recall@1/@3, MRR, latency, scaling

For each sampled episode, the cue is a later message that should bring it to mind but says it in
other words. A good memory finds the original among all episodes. The cue file is derived from
private conversations, so it lives in data/ (git-ignored).
"""
import json
import os
import random
import sqlite3
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from src.levels.l2_memory.embedding import Embedder  # noqa: E402

DB = os.path.join(ROOT, "memory.sqlite")
CUES = os.path.join(ROOT, "data", "memory_benchmark.json")

def episodes():
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT id, content FROM episodes WHERE length(content) > 0 ORDER BY id").fetchall()
    conn.close()
    return rows

def build(n: int = 80, seed: int = 7):
    from deepseek_engine import DeepSeekEngine
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        engine = DeepSeekEngine(model=json.load(f).get("DEEPSEEK_MODEL", "deepseek-flash"))
    pool = [(i, c) for i, c in episodes() if 30 <= len(c) <= 400]
    random.Random(seed).shuffle(pool)
    picked, cues = pool[:n], []
    for start in range(0, len(picked), 20):
        batch = picked[start:start + 20]
        listing = "\n".join(f"[{i}] {c[:300]}" for i, c in batch)
        prompt = ("<|im_start|>system\n下面是一个聊天机器人记忆里的若干段经历。为每一段写一句"
                  "「过了几天以后，有人在聊天里说的话」，这句话应该让她想起这段经历，"
                  "但要换一种说法：不要照抄原文里的句子，尽量少用原文的词（人名、作品名可以保留）。"
                  "像真实聊天那样自然、简短（10-30 字）。Output JSON: {\"cues\": [{\"id\": 编号, \"cue\": \"...\"}]}\n"
                  f"{listing}<|im_end|>\n<|im_start|>assistant\n")
        out = engine.generate_json(prompt, max_tokens=2000, temperature=0.7)
        for c in out.get("cues", []):
            if isinstance(c, dict) and c.get("cue") and c.get("id") in {i for i, _ in batch}:
                cues.append({"id": int(c["id"]), "cue": c["cue"]})
        print(f"{len(cues)} cues so far")
    os.makedirs(os.path.dirname(CUES), exist_ok=True)
    with open(CUES, "w", encoding="utf-8") as f:
        json.dump(cues, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(cues)} cues to {CUES}  ({engine.usage_summary()})")

def evaluate(embedder: Embedder, rows, cues):
    ids = np.array([i for i, _ in rows])
    t = time.perf_counter()
    if hasattr(embedder, "embed_many"):
        M = embedder.embed_many([c for _, c in rows])
    else:
        M = np.stack([embedder.embed(c) for _, c in rows])
    build_s = time.perf_counter() - t
    ranks, lat = [], []
    for c in cues:
        t = time.perf_counter()
        sims = M @ embedder.embed(c["cue"])
        lat.append(time.perf_counter() - t)
        target = sims[ids == c["id"]]
        if not len(target):
            continue
        ranks.append(int((sims > target[0]).sum()) + 1)
    r = np.array(ranks)
    return {"recall@1": float((r <= 1).mean()), "recall@3": float((r <= 3).mean()), "recall@10": float((r <= 10).mean()),
            "MRR": float((1 / r).mean()), "median_rank": float(np.median(r)), "cues": len(r),
            "embed_all_s": round(build_s, 1), "query_ms": round(1000 * float(np.mean(lat)), 2), "dim": int(M.shape[1])}

def run():
    with open(CUES, encoding="utf-8") as f:
        cues = json.load(f)
    rows = episodes()
    print(f"{len(rows)} episodes, {len(cues)} cues\n")
    candidates = [("hash (current)", Embedder())]
    semantic = Embedder(model_dir=os.path.join(ROOT, "models", "bge-small-zh-v1.5"))
    if semantic.backend == "onnx":
        candidates.append(("bge-small-zh (semantic)", semantic))
    for name, emb in candidates:
        res = evaluate(emb, rows, cues)
        print(f"{name:26} " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))
    # How the in-memory index scales (vectorised dot product over a whole year of episodes)
    for n in (30_000, 380_000):
        M = np.random.default_rng(0).standard_normal((n, 512)).astype(np.float32)
        q = M[0]
        t = time.perf_counter()
        for _ in range(5):
            np.argpartition(-(M @ q), 10)[:10]
        print(f"index search over {n:>7} vectors: {1000 * (time.perf_counter() - t) / 5:.1f} ms")

if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
