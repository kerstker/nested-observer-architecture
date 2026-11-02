"""Offline synthetic diagnostic, no API, GPU, live DB or production traffic.

Measures exact-repeat reuse, not held-out generalization or LLM intelligence.
Each task has a separately constructed valid witness; output is checked directly.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimal-self"))
from src.levels.l8_agency.verified_reasoning import VerifiedReasoner
from src.levels.l10_consciousness.action_prediction import ActionPrediction
from src.levels.l17_self_improvement.research_gate import Trial, compare_paired


def run():
    cold, warm, elapsed = [], [], []
    for family in range(40):
        for seed in range(3):
            rng = random.Random(10000 * family + seed)
            values = rng.sample(list(range(-8, 9)), 3)
            p = {"variables": {n: list(range(-8, 9)) for n in "abc"},
                 "constraints": [{"op": "sum_eq", "vars": list("abc"), "value": sum(values)},
                                 {"op": "sum_eq", "vars": ["a"], "value": values[0]},
                                 {"op": "sum_eq", "vars": ["b"], "value": values[1]},
                                 {"op": "all_different", "vars": list("abc")}]}
            solver = VerifiedReasoner()
            for rows in (cold, warm):
                result = solver.solve(p)
                answer = result["assignment"]
                correct = result["status"] == "verified" and answer == dict(zip("abc", values))
                rows.append(Trial(str(family), str(family), seed, correct, result["operations"]))
                elapsed.append(result["seconds"])
    gate = compare_paired(cold, warm)
    model = ActionPrediction()
    # Deliberately constructed dependence: verifies learnability and baseline accounting only.
    for i in range(600):
        low = i % 2 == 0
        gw = SimpleNamespace(metabolism=SimpleNamespace(energy=5 if low else 100,
                                                        max_energy=100, is_sleeping=False))
        ctx = SimpleNamespace(origin="reply", group_id=None)
        action = "rest" if low else "work"
        model.observe(model.before_decision(gw, ctx, ["rest", "work"]), action)
        model.observe(model.before_tool(gw, ctx, action), "blocked" if low else "returned")
    return {"scope": "synthetic engineering diagnostic; NOT a SOTA or paper confirmation benchmark",
            "tasks": len(cold), "seeds": [0, 1, 2], "families": 40,
            "cold_accuracy": statistics.mean(r.correct for r in cold),
            "warm_accuracy": statistics.mean(r.correct for r in warm),
            "cold_mean_operations": statistics.mean(r.cost for r in cold),
            "warm_mean_operations": statistics.mean(r.cost for r in warm),
            "max_seconds": max(elapsed), "exact_repeat_gate": gate,
            "synthetic_prediction": model.report(),
            "not_measured": ["human conversational quality", "real tool task completion",
                             "body state causal necessity", "LLM cost reduction", "unseen task generalization"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "synthetic_prediction"}, ensure_ascii=False, indent=2))
