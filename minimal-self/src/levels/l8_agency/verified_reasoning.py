"""RHS-inspired bounded, verified finite-domain constraint solving.

Actual runtime tiers: exact cache, verified neighboring assignments, greedy,
then backtracking with partial-constraint pruning. No eval, generated code,
network, model training, or implicit side effects. UNSOLVED is not UNSAT.
One operation budget covers lookup, search AND verification. All failed work
is charged. A deadline supplements it; no unbounded search mode is exposed.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import time


class Exhausted(Exception):
    pass


@dataclass
class Budget:
    limit: int
    deadline: float
    spent: int = 0

    def take(self):
        if self.spent >= self.limit or time.perf_counter() >= self.deadline:
            raise Exhausted
        self.spent += 1


def _integer(value):
    return type(value) is int and abs(value) <= 10**9


def normalize(problem):
    if not isinstance(problem, dict) or set(problem) != {"variables", "constraints"}:
        raise ValueError("expected variables and constraints")
    variables, constraints = problem["variables"], problem["constraints"]
    if not isinstance(variables, dict) or not 1 <= len(variables) <= 12:
        raise ValueError("use 1..12 variables")
    normalized = {}
    for name, values in sorted(variables.items()):
        if not isinstance(name, str) or not name.isidentifier() or len(name) > 32:
            raise ValueError("invalid variable name")
        if not isinstance(values, list) or not 1 <= len(values) <= 32 or not all(_integer(v) for v in values):
            raise ValueError("domains need 1..32 bounded integers")
        normalized[name] = sorted(set(values))
    if not isinstance(constraints, list) or len(constraints) > 64:
        raise ValueError("use at most 64 constraints")
    clean = []
    for c in constraints:
        if not isinstance(c, dict) or c.get("op") not in ("sum_eq", "sum_le", "sum_ge", "all_different"):
            raise ValueError("unsupported constraint")
        keys = {"op", "vars"} if c["op"] == "all_different" else {"op", "vars", "value"}
        names = c.get("vars")
        if (set(c) != keys or not isinstance(names, list) or not names or len(names) > 12
                or not all(isinstance(n, str) and n in normalized for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("invalid constraint variables or fields")
        if "value" in keys and not _integer(c["value"]):
            raise ValueError("constraint value must be a bounded integer")
        clean.append({**c, "vars": sorted(names)})
    clean.sort(key=lambda c: json.dumps(c, sort_keys=True))
    return {"variables": normalized, "constraints": clean}


def verify(problem, assignment, budget):
    """Full independent contract check; never trusts a search tier's success flag."""
    budget.take()
    if not isinstance(assignment, dict) or set(assignment) != set(problem["variables"]):
        return False
    for name, domain in problem["variables"].items():
        budget.take()
        if type(assignment[name]) is not int or assignment[name] not in domain:
            return False
    for c in problem["constraints"]:
        budget.take()
        values = [assignment[n] for n in c["vars"]]
        op = c["op"]
        if op == "all_different":
            if len(set(values)) != len(values):
                return False
        elif op == "sum_eq" and sum(values) != c["value"]:
            return False
        elif op == "sum_le" and sum(values) > c["value"]:
            return False
        elif op == "sum_ge" and sum(values) < c["value"]:
            return False
    return True


def _feasible(problem, partial, budget):
    """Necessary partial constraints only; the separate verifier decides outputs."""
    for c in problem["constraints"]:
        budget.take()
        assigned = [partial[n] for n in c["vars"] if n in partial]
        missing = [problem["variables"][n] for n in c["vars"] if n not in partial]
        if c["op"] == "all_different":
            if len(set(assigned)) != len(assigned):
                return False
        else:
            low = sum(assigned) + sum(d[0] for d in missing)
            high = sum(assigned) + sum(d[-1] for d in missing)
            target = c["value"]
            if c["op"] == "sum_eq" and not low <= target <= high:
                return False
            if c["op"] == "sum_le" and low > target:
                return False
            if c["op"] == "sum_ge" and high < target:
                return False
    return True


class VerifiedReasoner:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.negative = OrderedDict()
        self.traces = deque(maxlen=64)
        self.version = "finite-constraints-v1"

    def solve(self, problem, operations=20000, seconds=1.0, scope="local"):
        started = time.perf_counter()
        if type(operations) is not int or not 1 <= operations <= 100000:
            raise ValueError("operation budget must be 1..100000")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 2:
            raise ValueError("deadline must be 0..2 seconds")
        p = normalize(problem)
        identity = json.dumps([self.version, scope, p], sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(identity.encode()).hexdigest()
        scope_key = hashlib.sha256(str(scope).encode()).hexdigest()
        budget = Budget(operations, started + seconds)
        attempts = []
        answer, status = None, "unsolved"

        def check(tier, candidate, primitive):
            before = budget.spent
            passed = verify(p, candidate, budget)
            attempts.append({"tier": tier, "verified": passed, "verification_operations": budget.spent - before})
            if not passed:
                self.negative[(key, primitive)] = True
                self.negative.move_to_end((key, primitive))
                while len(self.negative) > self.capacity:
                    self.negative.popitem(last=False)
            return passed

        try:
            budget.take()  # exact lookup is not free
            if key in self.cache:
                candidate = self.cache[key][1]
                if check("L1_exact", candidate, key):
                    answer = dict(candidate)
                else:
                    del self.cache[key]  # stale/corrupt entries cannot be trusted again
            if answer is None:
                # A near hit supplies only a candidate, not an answer or executable instructions.
                tried = 0
                for primitive, (owner, candidate) in reversed(list(self.cache.items())):
                    budget.take()
                    if owner != scope_key or set(candidate) != set(p["variables"]) or (key, primitive) in self.negative:
                        continue
                    tried += 1
                    if check("L2_candidate", candidate, primitive):
                        answer = dict(candidate)
                        break
                    if tried >= 5:
                        break
            if answer is None:
                greedy = {n: d[0] for n, d in p["variables"].items()}
                if check("L3_greedy", greedy, "greedy"):
                    answer = greedy
            if answer is None:
                order = sorted(p["variables"], key=lambda n: (len(p["variables"][n]), n))
                partial = {}

                def search(index):
                    budget.take()
                    if not _feasible(p, partial, budget):
                        return None
                    if index == len(order):
                        return dict(partial) if check("L4_search", partial, "search") else None
                    name = order[index]
                    for value in p["variables"][name]:
                        partial[name] = value
                        found = search(index + 1)
                        if found is not None:
                            return found
                    partial.pop(name, None)
                    return None

                answer = search(0)
                if answer is None:
                    status = "unsatisfiable"  # only after exhaustive finite search completes
        except Exhausted:
            status = "budget_exhausted"
        if answer is not None:
            status = "verified"
            self.cache[key] = (scope_key, dict(answer))
            self.cache.move_to_end(key)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        trace = {"problem_hash": key, "version": self.version, "status": status,
                 "operations": budget.spent, "budget": operations,
                 "seconds": time.perf_counter() - started, "attempts": attempts}
        self.traces.append(trace)
        return {**trace, "assignment": answer,
                "verification_scope": "Only the supplied finite integer constraints; not unstated requirements."}

    def report(self):
        return {"version": self.version, "cache_entries": len(self.cache),
                "negative_entries": len(self.negative), "recent": list(self.traces),
                "learned_router": False, "trained_adapters": False}
