"""Read-only deployment acceptance for game memory, body telemetry and FTB loading.

No tools, chat messages, migrations or writes to either production database are performed.
The live process proves shared object identity; a separate local Embedder verifies the stored vector space.
"""
import argparse
import json
import sqlite3
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimal-self"))

import numpy as np
from src.levels.l2_memory.embedding import Embedder, default_model_dir


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def live_status(base, config_path):
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("console URL must be loopback")
    if parsed.username or parsed.password or parsed.query:
        raise ValueError("console URL must not include credentials or query parameters")
    headers = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as stream:
            token = json.load(stream).get("CONSOLE_TOKEN")
        if token:
            headers["X-Console-Token"] = str(token)
    request = urllib.request.Request(base.rstrip("/") + "/console/api/outside", headers=headers, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=12) as response:
            game = json.load(response).get("game") or {}
    except urllib.error.HTTPError as error:
        return {"available": False, "error": "HTTP " + str(error.code)}
    except (OSError, ValueError):
        return {"available": False, "error": "console unavailable or malformed response"}
    # Never print the console's private journals, trusted users or server details.
    return {"available": bool(game), "connected": game.get("connected"),
            "body": game.get("body"), "game_memory": game.get("game_memory"),
            "memory_queue": game.get("memory_queue"), "quests": game.get("quests")}


def read_vectors(path, embedder, queries, limit=4, include_text=False):
    if not path.is_file():
        return {"available": False, "reason": "game memory database does not exist", "read_only": True}
    # Even SQLite mode=ro can create WAL/SHM sidecars. Copy a stable DB+WAL pair to
    # a disposable folder instead; SQLite never opens the production path at all.
    sources = [path, Path(str(path) + "-wal")]
    def stamps():
        return [(p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in sources]
    with tempfile.TemporaryDirectory(prefix="noa-memory-read-") as folder:
        copied = Path(folder) / "game.sqlite"
        for attempt in range(3):
            try:
                before = stamps()
                for source, stamp in zip(sources, before):
                    destination = copied if source == path else Path(str(copied) + "-wal")
                    if stamp is not None:
                        shutil.copyfile(source, destination)
                    elif destination.exists():
                        destination.unlink()
                if before == stamps():
                    break
            except FileNotFoundError:
                continue
        else:
            return {"available": False, "read_only": True, "reason": "database changed during snapshot; retry later"}
        connection = sqlite3.connect(copied.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)
        try:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute("SELECT id,kind,text,dim,emb_model,embedding FROM game_memory ORDER BY id").fetchall()
        finally:
            connection.close()
    same_model, wrong_model, bad_vectors = [], 0, 0
    vectors = []
    for row in rows:
        if row[4] != embedder.model_id:
            wrong_model += 1
            continue
        try:
            vector = Embedder.from_bytes(row[5])
            if len(vector) != embedder.dim or not np.isfinite(vector).all():
                raise ValueError("invalid vector")
        except (TypeError, ValueError):
            bad_vectors += 1
            continue
        same_model.append(row)
        vectors.append(vector)
    matrix = np.stack(vectors) if vectors else np.zeros((0, embedder.dim), dtype=np.float32)
    recalls = []
    for query in queries:
        started = time.perf_counter()
        scores = embedder.scores(matrix, embedder.embed(query)) if len(vectors) else np.array([])
        hits = []
        for index in np.argsort(scores)[::-1][:max(1, min(8, limit))]:
            if scores[index] < (0.1 if embedder.backend == "hash" else 0.2):
                continue
            row = same_model[int(index)]
            hit = {"id": row[0], "kind": row[1], "dimension": row[3], "score": round(float(scores[index]), 4)}
            if include_text:
                hit["game_text"] = str(row[2])[:180]
            hits.append(hit)
        recalls.append({"query": query, "milliseconds": round((time.perf_counter() - started) * 1000, 2), "hits": hits})
    return {"available": True, "read_only": True, "records": len(rows), "matching_vectors": len(vectors),
            "other_model_vectors": wrong_model, "invalid_vectors": bad_vectors,
            "model_id": embedder.model_id, "backend": embedder.backend, "dim": embedder.dim,
            "snapshot": "stable temporary DB+WAL copy; source SQLite never opened",
            "ranking": "raw vector similarity; no spatial or recency boosts", "queries": recalls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "mc_game_memory.sqlite")
    parser.add_argument("--console-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--include-game-text", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--require-live", action="store_true")
    args = parser.parse_args()
    embedder = Embedder(model_dir=args.model_dir or default_model_dir())
    live = {"available": False, "skipped": True} if args.skip_live else live_status(args.console_url, ROOT / "config.json")
    try:
        vectors = read_vectors(args.db, embedder, args.query or ["家和箱子在哪里", "上次挖矿有什么经验"],
                               include_text=args.include_game_text)
    except sqlite3.Error:
        vectors = {"available": False, "read_only": True, "reason": "game vector schema unavailable"}
    runtime_memory = live.get("game_memory") or {}
    quests = live.get("quests") or {}
    checks = {"runtime_reports_same_embedder_instance": runtime_memory.get("shared_embedder") is True,
              "local_vector_space_matches_runtime": runtime_memory.get("model_id") == embedder.model_id,
              "valid_matching_vectors_present": vectors.get("matching_vectors", 0) > 0 and not vectors.get("invalid_vectors"),
              "quest_sources_loaded": quests.get("source") == "live_pack_snbt" and quests.get("quests", 0) > 0
                                      and not quests.get("errors"),
              "server_quest_progress_synced": quests.get("server_verified") is True
                                              and quests.get("progress_source") == "ftb_client_team_sync"
                                              and quests.get("synced_quests", 0) > 0}
    report = {"t": time.time(), "read_only": True, "live": live, "vector_retrieval": vectors, "checks": checks,
              "note": "A local audit Embedder is a separate instance; shared identity is verified only by the running app."}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not args.require_live or all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
