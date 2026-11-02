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
import math
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
    by, names, metrics, metric_samples = {}, {}, {}, {}
    errors = list(errors or [])
    for seed, results in runs.items():
        for r in results:
            sample = f"{r['id']}@{seed}"
            if sample in by:
                errors.append(f"duplicate evaluation sample: {sample}")
                continue
            by[sample] = bool(r["passed"])
            names[r["id"]] = r.get("name", r["id"])
            for k, v in (r.get("metrics") or {}).items():
                try:
                    value = float(v)
                    if not math.isfinite(value):
                        raise ValueError("non-finite metric")
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"invalid metric: {sample}/{k}")
                    continue
                metrics.setdefault(k, []).append(value)
                metric_samples.setdefault(k, []).append(sample)
    return {"passed": sum(by.values()), "total": len(by), "by": by, "names": names,
            "metrics": {k: sum(v) / len(v) for k, v in metrics.items()},
            "metric_samples": {k: sorted(v) for k, v in metric_samples.items()}, "errors": errors}

def compare(before: Dict[str, Any], after: Dict[str, Any], target: Optional[str] = None,
            directions: Dict[str, str] = None, min_gain: float = 0.05,
            quality_tolerances: Dict[str, float] = None) -> Dict[str, Any]:
    """Quality first, then improvement; no statistical significance is implied.

    quality_tolerances are predeclared absolute non-regression limits, e.g.
    {"accuracy": .002, "ce": .01}. Other metrics retain the relative min_gain
    threshold. Do not tune either threshold on the candidate's confirmation set.
    """
    if directions is None:
        try:
            from src.evaluation.scenarios import METRIC_DIRECTIONS as directions
        except Exception:
            directions = {}
    regressions = [k for k, ok in before["by"].items() if ok and not after["by"].get(k, False)]
    fixed = [k for k, ok in after["by"].items() if ok and not before["by"].get(k, False)]
    # E26 lesson: a cheaper or improved target cannot compensate for lost quality.
    # The same scenario/seed and metric cohorts must be measured on both sides.
    evidence_errors = []
    if not before["by"] or set(before["by"]) != set(after["by"]):
        evidence_errors.append("scenario/seed coverage changed or empty")
    if set(before["metrics"]) != set(after["metrics"]):
        evidence_errors.append("metric coverage changed")
    if before.get("metric_samples") != after.get("metric_samples"):
        evidence_errors.append("metric sample coverage changed")
    if before["errors"]:
        evidence_errors.append("baseline evaluation failed")
    if not math.isfinite(min_gain) or min_gain <= 0:
        raise ValueError("min_gain must be finite and positive")
    if target and (target not in directions or target not in before["metrics"]):
        evidence_errors.append("target metric missing or direction unknown")
    quality_tolerances = quality_tolerances or {}
    for k, tolerance in quality_tolerances.items():
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("quality tolerances must be finite and nonnegative")
        if k not in before["metrics"] or k not in directions:
            evidence_errors.append(f"required quality metric missing: {k}")
    changes = {}
    for k, b in before["metrics"].items():
        a = after["metrics"].get(k)
        if not isinstance(b, (int, float)) or not math.isfinite(b) or (
                a is not None and (not isinstance(a, (int, float)) or not math.isfinite(a))):
            evidence_errors.append(f"invalid metric: {k}")
            continue
        if a is None or k not in directions:
            continue
        if directions[k] not in ("lower", "higher"):
            evidence_errors.append(f"invalid metric direction: {k}")
            continue
        gain = (b - a) if directions[k] == "lower" else (a - b)
        changes[k] = {"before": b, "after": a, "gain": gain / max(abs(b), 1e-6)}
    metric_regressions = []
    for k, c in changes.items():
        if k in quality_tolerances:
            loss = c["after"] - c["before"] if directions[k] == "lower" else c["before"] - c["after"]
            if loss > quality_tolerances[k] and not math.isclose(loss, quality_tolerances[k], abs_tol=1e-12):
                metric_regressions.append(k)
        elif c["gain"] <= -min_gain:
            metric_regressions.append(k)
    if after["errors"] or evidence_errors or regressions or metric_regressions:
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
    if evidence_errors:
        parts.append("证据不完整：" + "; ".join(evidence_errors))
    return {"verdict": verdict, "regressions": regressions, "fixed": fixed, "changes": changes,
            "metric_regressions": metric_regressions, "evidence_errors": evidence_errors,
            "summary": "，".join(parts)}
