"""Have the judge grade her past lines for the style examples (also runs on its own while she sleeps).

    python scripts/rate_anchors.py [--db memory.sqlite] [--rounds 5] [--show]
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)

from src.levels.l9_narrative.style_anchors import StyleAnchors  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.path.join(ROOT, "memory.sqlite"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    from deepseek_engine import DeepSeekEngine
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        engine = DeepSeekEngine(model=json.load(f).get("DEEPSEEK_MODEL", "deepseek-flash"))
    anchors = StyleAnchors(args.db)
    for _ in range(args.rounds):
        n = anchors.rate(engine.generate_json)
        print(f"rated {n}")
        if not n:
            break
    good = anchors._candidates()
    print(f"{len(good)} lines pass the judge")
    if args.show:
        for line in anchors.examples(6):
            print("  -", line)
