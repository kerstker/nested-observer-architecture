"""
Run Noa's behavioural evaluation (design doc, section 7) and print the results.

    python scripts/evaluate_life.py [--seed N] [--only NAME]
    python scripts/evaluate_life.py --real        # talk to DeepSeek (needs DEEPSEEK_API_KEY)

By default a scripted stand-in replaces the language model (free, deterministic).
With --real, the scenarios whose outcome depends on what the model actually says
(continuity, introspection) run against DeepSeek and print her real answers; the
week-long simulations are skipped because they would make hundreds of paid calls.

Everything runs on a simulated clock in temporary databases; memory.sqlite is not touched.
"""
import argparse
import json
import logging
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)
logging.disable(logging.CRITICAL)

from src.evaluation import scenarios  # noqa: E402

REAL_SCENARIOS = {"continuity_of_self", "grounded_introspection"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", help="run a single scenario by function name")
    parser.add_argument("--real", action="store_true", help="use DeepSeek instead of the scripted model")
    args = parser.parse_args()

    engine = None
    if args.real:
        from deepseek_engine import DeepSeekEngine
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        engine = DeepSeekEngine(model=config.get("DEEPSEEK_MODEL", "deepseek-flash"),
                                base_url=config.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        scenarios.RealLLM.engine = engine
        scenarios.LLM_FACTORY = scenarios.RealLLM

    selected = [s for s in scenarios.ALL
                if (not args.only or s.__name__ == args.only)
                and (not args.real or s.__name__ in REAL_SCENARIOS)]
    passed = 0
    for scenario in selected:
        started = time.time()
        r = scenario(seed=args.seed)
        passed += r["passed"]
        print(f"[{'通过' if r['passed'] else '未通过'}] {r['name']}  ({time.time() - started:.1f}s)")
        print(f"    测得：{r['measured']}")
        print(f"    标准：{r['criterion']}")
        if r["note"] and not args.real:
            print(f"    说明：{r['note']}")
        if args.real:
            for question, answer in r.get("dialogue", []):
                print(f"    小明：{question}")
                print("    乃爱：" + answer.replace("\n", "\n          "))
    print(f"\n{passed}/{len(selected)} 项通过")
    if engine:
        print(f"DeepSeek 用量：{engine.usage_summary()}")
    sys.exit(0 if passed == len(selected) else 1)

if __name__ == "__main__":
    main()
