"""
Read her recent conversations for who-is-related-to-whom (the relation graph), newest first.

    python scripts/backfill_relations.py [--days 7] [--db memory.sqlite]

The running app does this by itself a few batches every half hour; this catches up a backlog in one go.
Safe next to the live app: every write is a short transaction.
"""
import argparse
import json
import os
import sqlite3
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)

from deepseek_engine import DeepSeekEngine  # noqa: E402
from src.levels.l2_memory.graph import RelationGraph  # noqa: E402

def names_in_chat(db: str):
    def names(ep):
        out = {}
        conn = sqlite3.connect(db, timeout=30)
        try:
            if ep.get("group_id"):
                for uid, nick in conn.execute("SELECT user_id, nickname FROM group_members WHERE group_id = ?", (ep["group_id"],)):
                    if nick and nick.strip():
                        out[nick.strip()] = uid
            if ep.get("user_id"):
                r = conn.execute("SELECT nickname FROM users WHERE user_id = ?", (ep["user_id"],)).fetchone()
                if r and r[0] and r[0].strip():
                    out[r[0].strip()] = ep["user_id"]
        finally:
            conn.close()
        out.pop("User", None)
        return out
    return names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=7)
    parser.add_argument("--db", default=os.path.join(ROOT, "memory.sqlite"))
    args = parser.parse_args()
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        engine = DeepSeekEngine(model=json.load(f).get("DEEPSEEK_MODEL", "deepseek-flash"))
    graph = RelationGraph(args.db)
    since = time.time() - args.days * 86400
    added, started = 0, time.time()
    while True:
        batch = graph.pending_episodes(12)
        if not batch or batch[-1]["t"] < since:      # newest unread is older than the window: done
            break
        added += graph.extract(engine.generate_json, names_in_chat(args.db), batch=12, max_batches=1)
        print(f"{added} relations so far; {graph.stats()}", flush=True)
    print(f"done in {time.time() - started:.0f}s: {added} relations added; {graph.stats()}; {engine.usage_summary()}")

if __name__ == "__main__":
    main()
