"""Bounded declarative planning, observed transition learning and verified reuse.

Actions are data, never executable code. Predicted goal attainment is explicitly
separate from observed goal attainment. The caller executes through existing
permission-checked tools and reports observations; this module grants no access.
"""
from collections import OrderedDict
import hashlib
import heapq
import itertools
import json
import math
import threading


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state(value):
    if (not isinstance(value, dict) or not 1 <= len(value) <= 24 or
            any(not isinstance(k, str) or len(k) > 40 or type(v) is not int or abs(v) > 10**6
                for k, v in value.items())):
        raise ValueError("state needs 1..24 named bounded integers")
    return dict(value)


def contract(problem):
    if not isinstance(problem, dict) or set(problem) != {"initial", "goal", "actions"}:
        raise ValueError("expected initial, goal, actions")
    initial, goal = state(problem["initial"]), state(problem["goal"])
    if not set(goal) <= set(initial):
        raise ValueError("goal variables missing from initial state")
    actions = problem["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 32:
        raise ValueError("use 1..32 actions")
    clean = []
    names = set()
    for a in actions:
        if not isinstance(a, dict) or set(a) != {"name", "requires", "effects", "cost"}:
            raise ValueError("action needs name, requires, effects, cost")
        name = a["name"]
        if not isinstance(name, str) or not name.isidentifier() or len(name) > 40 or name in names:
            raise ValueError("action names must be unique identifiers")
        requires, effects = a["requires"], a["effects"]
        # Preconditions are minimum resources; effects are additive, including resource consumption.
        for mapping in (requires, effects):
            if not isinstance(mapping, dict) or not set(mapping) <= set(initial):
                raise ValueError("unknown action state variable")
            if any(type(v) is not int or abs(v) > 10**6 for v in mapping.values()):
                raise ValueError("invalid action values")
        cost = a["cost"]
        if type(cost) not in (int, float) or not math.isfinite(cost) or not .001 <= cost <= 10**6:
            raise ValueError("positive bounded action cost required")
        names.add(name)
        clean.append({"name": name, "requires": dict(requires), "effects": dict(effects), "cost": float(cost)})
    clean.sort(key=lambda a: a["name"])
    return {"initial": initial, "goal": goal, "actions": clean}


def reached(s, goal):
    return all(s[k] >= v for k, v in goal.items())


def transition(s, action):
    if any(s[k] < v for k, v in action["requires"].items()):
        return None
    out = {k: v + action["effects"].get(k, 0) for k, v in s.items()}
    if any(v < 0 or abs(v) > 10**6 for v in out.values()):
        return None
    return out


class Planner:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.skills = OrderedDict()
        self.beliefs = OrderedDict()
        self.effects = OrderedDict()
        self.pending = OrderedDict()
        self.sequence = 0
        self.lock = threading.RLock()

    def _trim(self, mapping):
        while len(mapping) > self.capacity:
            mapping.popitem(last=False)

    def _keys(self, domain, action, own):
        return digest([domain, action]), digest([domain, action, own])

    def _probability(self, domain, action, own):
        broad, specific = self._keys(domain, action, own)
        good, bad = self.beliefs.get(specific, [0, 0]) if own else self.beliefs.get(broad, [0, 0])
        return (good + 1) / (good + bad + 2)

    def _predict(self, s, action, domain, own):
        predicted = transition(s, action)
        if predicted is None:
            return None
        key = self._keys(domain, action["name"], own)[1]
        outcomes = self.effects.get(key, [])
        if outcomes:
            delta = max(outcomes, key=lambda row: row[1])[0]
            if set(delta) <= set(s):
                predicted = {k: v + delta.get(k, 0) for k, v in s.items()}
                if any(v < 0 or abs(v) > 10**6 for v in predicted.values()):
                    return None
        return predicted

    def plan(self, problem, *, scope="local", self_state=None, max_steps=8, max_nodes=4000):
        p = contract(problem)
        own = state(self_state) if self_state is not None else {}
        if type(max_steps) is not int or not 1 <= max_steps <= 16:
            raise ValueError("max_steps must be 1..16")
        if type(max_nodes) is not int or not 1 <= max_nodes <= 20000:
            raise ValueError("max_nodes must be 1..20000")
        with self.lock:
            domain = digest([scope, p["actions"]])
            key = digest([domain, p["goal"]])
            by_name = {a["name"]: a for a in p["actions"]}
            sequence = self.skills.get(key)
            best = None
            expanded = 0
            # Reuse is only a candidate; recheck every transition and the current goal.
            if sequence and len(sequence) <= max_steps:
                s, score, path = p["initial"], 0., []
                for name in sequence:
                    a = by_name.get(name)
                    nxt = self._predict(s, a, domain, own) if a else None
                    if nxt is None:
                        break
                    score += a["cost"] / self._probability(domain, name, own)
                    path.append((name, s, nxt))
                    s = nxt
                else:
                    if reached(s, p["goal"]):
                        best = (score, path)
            # Uniform-cost search with learned expected retry costs; positive costs,
            # depth-aware dominance and finite expansion budget prevent unbounded work.
            counter = itertools.count()
            queue = [(0., next(counter), p["initial"], [])]
            seen = {}
            while queue and expanded < max_nodes:
                cost, _, s, path = heapq.heappop(queue)
                if best is not None and cost >= best[0]:
                    continue
                identity = (tuple(sorted(s.items())), len(path))
                if seen.get(identity, float("inf")) <= cost:
                    continue
                seen[identity] = cost
                expanded += 1
                if reached(s, p["goal"]):
                    best = (cost, path)
                    break
                if len(path) == max_steps:
                    continue
                for a in p["actions"]:
                    if len(queue) + expanded >= max_nodes:
                        break
                    nxt = self._predict(s, a, domain, own)
                    if nxt is None or nxt == s:
                        continue
                    prob = self._probability(domain, a["name"], own)
                    heapq.heappush(queue, (cost + a["cost"] / prob, next(counter), nxt,
                                           path + [(a["name"], s, nxt)]))
            if best is None:
                return {"status": "unsolved", "expanded": expanded, "executed": False}
            self.sequence += 1
            ticket = digest([domain, self.sequence, p["initial"], p["goal"]])
            path = best[1]
            steps = [{"action": n, "before": b, "predicted_after": a,
                      "probability": self._probability(domain, n, own)} for n, b, a in path]
            self.pending[ticket] = {"domain": domain, "skill": key, "problem": p, "own": own,
                                    "path": steps, "index": 0, "observed": p["initial"], "failed": False}
            self._trim(self.pending)
            return {"status": "predicted", "ticket": ticket, "steps": steps, "expanded": expanded,
                    "expected_cost": best[0], "observed_goal": not steps and reached(p["initial"], p["goal"]),
                    "executed": False, "scope": "Predictions from supplied action contracts; report actual states after execution."}

    def observe(self, ticket, action, before, after):
        before, after = state(before), state(after)
        with self.lock:
            episode = self.pending.get(ticket)
            if episode is None:
                raise ValueError("unknown or expired plan ticket")
            index = episode["index"]
            if index >= len(episode["path"]):
                raise ValueError("plan already observed")
            step = episode["path"][index]
            if action != step["action"] or before != episode["observed"] or set(after) != set(before):
                raise ValueError("observation is out of sequence or has mismatched state variables")
            success = after == step["predicted_after"]
            specific = self._keys(episode["domain"], action, episode["own"])[1]
            delta = {k: after[k] - before[k] for k in before}
            outcomes = self.effects.setdefault(specific, [])
            for row in outcomes:
                if row[0] == delta:
                    row[1] += 1
                    break
            else:
                outcomes.append([delta, 1.])
            if sum(row[1] for row in outcomes) >= 128:
                for row in outcomes:
                    row[1] *= .5
            outcomes.sort(key=lambda row: row[1], reverse=True)
            del outcomes[4:]
            self.effects.move_to_end(specific)
            self._trim(self.effects)
            for key in self._keys(episode["domain"], action, episode["own"]):
                counts = self.beliefs.setdefault(key, [0., 0.])
                if sum(counts) >= 128:
                    counts[:] = [v * .5 for v in counts]
                counts[0 if success else 1] += 1
                self.beliefs.move_to_end(key)
            self._trim(self.beliefs)
            done = reached(after, episode["problem"]["goal"])
            episode["observed"] = after
            episode["index"] += 1
            if not success or done or episode["index"] == len(episode["path"]):
                del self.pending[ticket]  # no duplicate feedback or continuing an invalid forecast
                if done:
                    self.skills[episode["skill"]] = [s["action"] for s in episode["path"][:index+1]]
                    self.skills.move_to_end(episode["skill"])
                    self._trim(self.skills)
            return {"transition_matched": success, "observed_goal": done, "replan_required": not success and not done,
                    "prediction_error": {k: after[k] - step["predicted_after"][k] for k in after},
                    "causal_source": "unidentified"}

    def to_dict(self):
        with self.lock:
            # Pending episodes carry private state and are intentionally not restored.
            return {"version": 1, "skills": [[k, list(v)] for k, v in self.skills.items()],
                    "beliefs": [[k, list(v)] for k, v in self.beliefs.items()],
                    "effects": [[k, [[dict(d), n] for d, n in v]] for k, v in self.effects.items()]}

    def load_dict(self, value):
        if not isinstance(value, dict) or value.get("version") != 1:
            return
        with self.lock:
            skills, beliefs = value.get("skills", []), value.get("beliefs", [])
            if not isinstance(skills, list) or not isinstance(beliefs, list):
                return
            for item in skills[-self.capacity:]:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                k, v = item
                if isinstance(k, str) and len(k) == 64 and isinstance(v, list) and len(v) <= 16 and all(isinstance(n, str) and n.isidentifier() and len(n) <= 40 for n in v):
                    self.skills[k] = v[:]
            for item in beliefs[-self.capacity:]:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                k, v = item
                if isinstance(k, str) and len(k) == 64 and isinstance(v, list) and len(v) == 2 and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 128 for n in v):
                    self.beliefs[k] = v[:]
            effects = value.get("effects", [])
            if isinstance(effects, list):
                for item in effects[-self.capacity:]:
                    try:
                        k, rows = item
                        if not isinstance(k, str) or len(k) != 64 or not isinstance(rows, list) or len(rows) > 4:
                            continue
                        clean = []
                        for delta, n in rows:
                            d = state(delta)
                            if type(n) not in (int, float) or not math.isfinite(n) or not 0 < n <= 128:
                                raise ValueError("invalid transition count")
                            clean.append([d, float(n)])
                        self.effects[k] = clean
                    except (ValueError, TypeError):
                        continue
