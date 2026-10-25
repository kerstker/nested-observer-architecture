"""
Look at every collected sticker once and remember what it shows (stickers.looks).

    python scripts/describe_stickers.py [--limit 200]

The running app does this itself, ten at a time every half hour; this catches up in one go.
Without it she only knows *when* people send a sticker, not what it shows - and her own stickers
read back to her as numbers.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)

from deepseek_engine import DeepSeekEngine  # noqa: E402
from src.levels.l1_perception.stickers import StickerMemory  # noqa: E402
from src.levels.l2_memory.embedding import Embedder, default_model_dir  # noqa: E402

PROMPT = ("<|im_start|>user\n这是一张 QQ 表情包。用一句话（30 字以内）说清：画的是什么、表达什么情绪或意思；"
          "图里有字就照写出来。只输出这句话。<|im_end|>\n<|im_start|>assistant\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--db", default=os.path.join(ROOT, "memory.sqlite"))
    args = parser.parse_args()
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        engine = DeepSeekEngine(model=json.load(f).get("DEEPSEEK_MODEL", "deepseek-flash"))
    stickers = StickerMemory(args.db, os.path.join(ROOT, "data", "stickers"), Embedder(model_dir=default_model_dir()))

    def describe(png):
        out = engine.generate_json(PROMPT, max_tokens=80, temperature=0.2, media=[{"data": png, "mime": "image/png"}])
        return (out.get("reply") or "").strip().strip("“”\"") or None

    done = stickers.describe_missing(describe, args.limit)
    print(f"described {done} stickers; {engine.usage_summary()}")

if __name__ == "__main__":
    main()
