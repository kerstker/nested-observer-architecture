"""Prequential self/action models. These measure prediction, not consciousness.

Every forecast is frozen before its target is observed; loss is scored before
learning. Paired baselines omit body state or the selected action. Returned tool
calls are transport observations, never proof that a real-world task succeeded.
No message text, tool arguments, results, or user identifiers enter this model.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import math


OUTCOMES = ("returned", "blocked", "timeout", "error")
ANSWER = "__answer__"
OTHER = "__other__"


def _key(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


class Counts:
    """Bounded categorical predictor with a fixed symmetric prior and forgetting."""
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.rows = OrderedDict()

    def predict(self, context, labels):
        row = self.rows.get(context, {})
        total = sum(row.get(k, 0.0) for k in labels) + len(labels)
        return {k: (row.get(k, 0.0) + 1.0) / total for k in labels}

    def learn(self, context, label):
        row = self.rows.setdefault(context, {})
        if sum(row.values()) >= 128:
            row.update({k: v * .5 for k, v in row.items()})
        row[label] = row.get(label, 0.0) + 1
        self.rows.move_to_end(context)
        while len(self.rows) > self.capacity:
            self.rows.popitem(last=False)

    def dump(self):
        return list(self.rows.items())

    def load(self, rows):
        clean = OrderedDict()
        if not isinstance(rows, list):
            return
        for item in rows[-self.capacity:]:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            context, row = item
            if not isinstance(context, str) or not isinstance(row, dict):
                continue
            vals = {k: float(v) for k, v in list(row.items())[:130]
                    if isinstance(k, str) and len(k) <= 80 and type(v) in (int, float)
                    and math.isfinite(v) and 0 <= v <= 128}
            if len(context) <= 128 and sum(vals.values()) <= 256:
                clean[context] = vals
        self.rows = clean


@dataclass(frozen=True)
class Forecast:
    contexts: tuple
    probabilities: tuple
    # Keep predictions as tuples: neither caller nor subsequent learning can change them.
    kind: str


class ActionPrediction:
    def __init__(self):
        self.models = {name: Counts() for name in
                       ("action_world", "action_self", "outcome_world", "outcome_action", "outcome_self")}
        self.metrics = {name: [0, 0.0, 0.0] for name in self.models}  # n, log loss, Brier sum
        self.recent = deque(maxlen=32)

    @staticmethod
    def context(gw, ctx):
        # Coarse finite features keep the sample complexity and persistent state bounded.
        metabolism = getattr(gw, "metabolism", None)
        energy = float(getattr(metabolism, "energy", 0))
        maximum = max(1., float(getattr(metabolism, "max_energy", 100)))
        band = min(2, max(0, int(3 * energy / maximum))) if math.isfinite(energy) else 0
        origin = str(getattr(ctx, "origin", "reply"))[:20]
        world = _key((origin, bool(getattr(ctx, "group_id", None))))
        own = _key((world, band, bool(getattr(metabolism, "is_sleeping", False))))
        return world, own

    def _forecast(self, kind, contexts, labels):
        return Forecast(tuple(contexts), tuple(tuple(self.models[name].predict(key, labels).items())
                                               for name, key in contexts), kind)

    def before_decision(self, gw, ctx, tools):
        world, own = self.context(gw, ctx)
        labels = tuple(sorted(set(str(n)[:80] for n in tools))[:128]) + (ANSWER, OTHER)
        return self._forecast("action", (("action_world", world), ("action_self", own)), labels)

    def before_tool(self, gw, ctx, name):
        world, own = self.context(gw, ctx)
        return self._forecast("outcome", (("outcome_world", world),
                              ("outcome_action", _key((world, name))),
                              ("outcome_self", _key((own, name)))), OUTCOMES)

    def observe(self, forecast, actual):
        losses = {}
        for (name, context), frozen in zip(forecast.contexts, forecast.probabilities):
            probabilities = dict(frozen)
            label = actual if actual in probabilities else OTHER
            if label not in probabilities:
                raise ValueError("unknown observation")
            loss = -math.log(probabilities[label])
            brier = sum((p - (k == label)) ** 2 for k, p in probabilities.items())
            metric = self.metrics[name]
            metric[0] += 1
            metric[1] += loss
            metric[2] += brier
            losses[name] = loss
            self.models[name].learn(context, label)
        self.recent.append({"kind": forecast.kind, "actual": str(actual)[:80], "loss": losses})

    def report(self):
        metrics = {k: {"n": n, "log_loss": total / n if n else None,
                       "brier": brier / n if n else None}
                   for k, (n, total, brier) in self.metrics.items()}
        gains = {}
        for name, base, full in (("body_to_action", "action_world", "action_self"),
                                 ("action_to_outcome", "outcome_world", "outcome_action"),
                                 ("body_to_outcome", "outcome_action", "outcome_self")):
            b, f = metrics[base], metrics[full]
            gains[name] = ((b["log_loss"] - f["log_loss"]) / math.log(2)
                           if b["n"] and b["n"] == f["n"] else None)
        return {"mode": "observational", "metrics": metrics, "predictive_gain_bits": gains,
                "recent": list(self.recent), "causal_evidence": False,
                "meaning": "First attempted tool per response; tool return status is not task success."}

    def to_dict(self):
        return {"version": 1, "models": {k: v.dump() for k, v in self.models.items()},
                "metrics": self.metrics}

    def load_dict(self, data):
        if not isinstance(data, dict) or data.get("version") != 1:
            return
        models, metrics = data.get("models", {}), data.get("metrics", {})
        if not isinstance(models, dict) or not isinstance(metrics, dict):
            return
        for name, model in self.models.items():
            model.load(models.get(name, []))
            v = metrics.get(name)
            if (isinstance(v, list) and len(v) == 3 and type(v[0]) is int and 0 <= v[0] <= 10**9
                    and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 10**12 for x in v)):
                self.metrics[name] = list(v)
