"""
Level 17b: fitness for self-modification.

A change to her own code is only an improvement if her behaviour gets better, not if it merely
compiles. Fitness is her behavioural evaluation (src/evaluation/scenarios.py): every scenario,
on several random seeds, run on the candidate code in a subprocess (never imported into the
live process) and compared with the same run on the unchanged code.

- Hard constraint: no (scenario, seed) that passed before may fail after.
- Graded signal: the numeric metrics scenarios report (METRIC_DIRECTIONS says which way is
  better), so a change can be recognised as better even when everything already passes.
"""
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional

def _strip(env: Dict[str, str]) -> Dict[str, str]:
    out = {k: v for k, v in env.items()
           if not any(s in k.upper() for s in ("DEEPSEEK_API_KEY", "NAPCAT", "CONSOLE_TOKEN", "OPENAI", "ANTHROPIC"))}
    out["PYTHONIOENCODING"] = "utf-8"
    return out

def available(repo_root: str) -> bool:
    return os.path.exists(os.path.join(repo_root, "scripts", "evaluate_life.py"))

def _run_seed(script: str, repo_root: str, seed: int, out_dir: str, timeout: float, only: str = None):
    """One seed in its own process. Returns (results or None, log tail)."""
    path = os.path.join(out_dir, f"seed{seed}.json")
    cmd = [sys.executable, script, "--seed", str(seed), "--json", path] + (["--only", only] if only else [])
    env = _strip(os.environ)
    env["NOA_EVAL_WATCHDOG"] = str(int(timeout))      # the child dumps its stack and exits if it hangs
    # Output goes to a file, never to a pipe nobody is reading (a full pipe blocks the child forever)
    with open(os.path.join(out_dir, f"seed{seed}.log"), "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            proc.wait(timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            proc.kill()
    with open(log.name, encoding="utf-8", errors="replace") as f:
        tail = f.read().strip()[-300:]
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["results"], tail
    except (OSError, ValueError, KeyError):
        return None, tail

def evaluate(repo_root: str, seeds: Iterable[int] = (0, 1, 2), timeout: float = 900.0,
             only: str = None) -> Dict[str, Any]:
    """
    Run the evaluation of the code at `repo_root` on each seed. Never raises.
    Seeds run one after another, not in parallel: concurrent runs on Windows can deadlock in
    asyncio's socketpair fallback (one process accepting another's connection), and a fitness
    signal that sometimes hangs is worse than a slower one. A seed that hangs anyway is retried once.
    """
    script = os.path.join(repo_root, "scripts", "evaluate_life.py")
    out_dir = tempfile.mkdtemp(prefix="noa_fitness_")
    runs: Dict[int, List[Dict[str, Any]]] = {}
    errors: List[str] = []
    for seed in seeds:
        for attempt in range(2):
            results, tail = _run_seed(script, repo_root, seed, out_dir, timeout, only)
            if results is not None:
                runs[seed] = results
                break
        else:
            errors.append(f"seed {seed}: 没有结果（{tail}）")
    return summarise(runs, errors)

def summarise(runs: Dict[int, List[Dict[str, Any]]], errors: List[str] = None) -> Dict[str, Any]:
    by, names, metrics = {}, {}, {}
    for seed, results in runs.items():
        for r in results:
            by[f"{r['id']}@{seed}"] = bool(r["passed"])
            names[r["id"]] = r.get("name", r["id"])
            for k, v in (r.get("metrics") or {}).items():
                metrics.setdefault(k, []).append(float(v))
    return {"passed": sum(by.values()), "total": len(by), "by": by, "names": names,
            "metrics": {k: sum(v) / len(v) for k, v in metrics.items()}, "errors": errors or []}

def compare(before: Dict[str, Any], after: Dict[str, Any], target: Optional[str] = None,
            directions: Dict[str, str] = None, min_gain: float = 0.05) -> Dict[str, Any]:
    """Is the candidate better, the same, or worse than what she is now?"""
    if directions is None:
        try:
            from src.evaluation.scenarios import METRIC_DIRECTIONS as directions
        except Exception:
            directions = {}
    regressions = [k for k, ok in before["by"].items() if ok and not after["by"].get(k, False)]
    fixed = [k for k, ok in after["by"].items() if ok and not before["by"].get(k, False)]
    changes = {}
    for k, b in before["metrics"].items():
        a = after["metrics"].get(k)
        if a is None or k not in directions:
            continue
        gain = (b - a) if directions[k] == "lower" else (a - b)
        changes[k] = {"before": b, "after": a, "gain": gain / max(abs(b), 1e-6)}
    if after["errors"] or regressions or after["total"] < before["total"]:
        verdict = "worse"
    elif fixed:
        verdict = "better"
    elif target and target in changes:
        g = changes[target]["gain"]
        verdict = "better" if g >= min_gain else "worse" if g <= -min_gain else "same"
    else:
        worse = [k for k, c in changes.items() if c["gain"] <= -min_gain]
        better = [k for k, c in changes.items() if c["gain"] >= min_gain]
        verdict = "worse" if worse and not better else "better" if better and not worse else "same"
    parts = [f"评估 {before['passed']}/{before['total']} → {after['passed']}/{after['total']}"]
    if regressions:
        parts.append("变差：" + "、".join(before["names"].get(k.split("@")[0], k) + "@" + k.split("@")[1]
                                        for k in regressions[:5]))
    if fixed:
        parts.append(f"新通过 {len(fixed)} 项")
    for k, c in changes.items():
        if k == target or abs(c["gain"]) >= min_gain:
            parts.append(f"{k} {c['before']:.3g} → {c['after']:.3g}")
    if after["errors"]:
        parts.append("出错：" + "；".join(after["errors"])[:200])
    return {"verdict": verdict, "regressions": regressions, "fixed": fixed, "changes": changes,
            "summary": "，".join(parts)}
