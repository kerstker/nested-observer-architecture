"""
Try the console without QQ: boots a Noa in a separate database and serves only the console.

    python scripts/console_demo.py                 # scripted replies, free, port 8090
    python scripts/console_demo.py --real          # talk to DeepSeek (needs DEEPSEEK_API_KEY)
    python scripts/console_demo.py --db memory.sqlite   # look at the real Noa's memory (careful: she lives on in it)

Then open http://127.0.0.1:8090/console
"""
import argparse
import asyncio
import json
import logging
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)
logging.basicConfig(level=logging.WARNING)

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from src.core.event_bus import EventBus  # noqa: E402
from src.evaluation.scenarios import ScriptedLLM, StubSearch  # noqa: E402
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace  # noqa: E402
from console import create_console_router  # noqa: E402

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.path.join(ROOT, "data", "console_demo.sqlite"))
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--tick", type=float, default=10.0, help="heartbeat interval in seconds")
    args = parser.parse_args()

    if args.real:
        from deepseek_engine import DeepSeekEngine
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        engine = DeepSeekEngine(model=config.get("DEEPSEEK_MODEL", "deepseek-flash"))
    else:
        engine = ScriptedLLM("喵～我在听呢，你今天过得怎么样？")

    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    from src.levels.l2_memory.embedding import Embedder, default_model_dir
    gw = GlobalWorkspace(EventBus(), engine, db_path=args.db,
                         embedder=Embedder(model_dir=default_model_dir()) if args.real else None)
    if args.real:
        gw.idle_thought_interval = config.get("IDLE_THOUGHT_INTERVAL", 600)   # paid calls: keep inner thoughts sparse
    else:
        gw.search_tool = StubSearch()

    app = FastAPI()
    app.include_router(create_console_router(gw))

    @app.on_event("startup")
    async def heartbeat():
        async def loop():
            while True:
                await asyncio.sleep(args.tick)
                try:
                    await gw.tick()
                except Exception as e:
                    logging.error(f"tick failed: {e}")
        asyncio.create_task(loop())

    print(f"控制台：http://127.0.0.1:{args.port}/console  （数据库 {os.path.abspath(args.db)}）")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
