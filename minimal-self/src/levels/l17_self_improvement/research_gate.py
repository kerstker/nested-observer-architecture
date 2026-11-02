"""Paired, cluster-bootstrap proposal evidence; never applies a code/library change.

Use fresh validation cohorts for selection, frozen test cohorts only for reporting.
This is an empirical gate, not a conformal or repeated-testing guarantee.
"""
from dataclasses import dataclass
import math
import random
from statistics import mean


@dataclass(frozen=True)
class Trial:
    problem: str
    family: str
    seed: int
    correct: bool
    cost: float  # includes unsuccessful attempts, verification and lookup


def compare_paired(before, after, *, selection_families=(), draws=2000, seed=0,
                   minimum_families=20, minimum_seeds=3):
    """No accuracy tradeoff; resample whole families, preserving paired seeds."""
    errors = []
    if type(draws) is not int or not 200 <= draws <= 10000:
        raise ValueError("bootstrap draws must be 200..10000")

    def index(rows):
        out = {}
        for r in rows:
            if (not isinstance(r, Trial) or not isinstance(r.problem, str) or not r.problem
                    or not isinstance(r.family, str) or not r.family or type(r.seed) is not int
                    or type(r.correct) is not bool or type(r.cost) not in (int, float)
                    or not math.isfinite(r.cost) or r.cost < 0):
                errors.append("invalid trial")
                continue
            key = (r.problem, r.seed)
            if key in out:
                errors.append("duplicate problem/seed")
            out[key] = r
        return out

    b, a = index(before), index(after)
    if not b or set(a) != set(b):
        errors.append("paired coverage missing")
    families = {}
    for key in set(b) & set(a):
        x, y = b[key], a[key]
        if x.family != y.family:
            errors.append("family changed")
        families.setdefault(x.family, []).append((y.cost - x.cost, int(y.correct) - int(x.correct), x.seed))
    if set(families) & set(selection_families):
        errors.append("confirmation families overlap selection/training")
    if len(families) < minimum_families:
        errors.append("too few independent families")
    if any(len({r[2] for r in rows}) < minimum_seeds for rows in families.values()):
        errors.append("too few seeds within family")
    if errors:
        return {"accepted": False, "errors": sorted(set(errors))}
    # Equal family weight is declared up front; do not count correlated variants as independent.
    pairs = [(mean(r[0] for r in rows), mean(r[1] for r in rows)) for rows in families.values()]
    rng = random.Random(seed)
    costs, accuracy = [], []
    for _ in range(draws):
        sample = rng.choices(pairs, k=len(pairs))
        costs.append(mean(x[0] for x in sample))
        accuracy.append(mean(x[1] for x in sample))
    costs.sort()
    accuracy.sort()
    lo, hi = int(.025 * (draws - 1)), math.ceil(.975 * (draws - 1))
    cost_ci, accuracy_ci = [costs[lo], costs[hi]], [accuracy[lo], accuracy[hi]]
    regressions = [f"{key[0]}@{key[1]}" for key in b if b[key].correct and not a[key].correct]
    return {"accepted": not regressions and cost_ci[1] < 0 and accuracy_ci[0] >= 0,
            "errors": [], "regressions": regressions, "families": len(families),
            "cost_delta_ci95": cost_ci, "accuracy_delta_ci95": accuracy_ci,
            "mean_cost_delta": mean(x[0] for x in pairs), "bootstrap_seed": seed,
            "interpretation": "Fresh paired family-weighted empirical evidence; not a global optimality guarantee."}
