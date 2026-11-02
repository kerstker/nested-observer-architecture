"""Stage 1 continuous body predictions and bounded, reversible processing controls."""
import asyncio
from collections import deque
from dataclasses import asdict, dataclass, field
import logging
import math
from typing import Dict, List

from src.core import clock
from .welfare import WelfareGuard

logger = logging.getLogger(__name__)
CHANNELS = ("energy", "hunger", "social", "curiosity", "hygiene")
WEIGHTS = {"energy": .4, "hunger": .3, "social": .4, "curiosity": .15, "hygiene": .05}


def bound(value, lower, upper):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("core state must be finite")
    return min(upper, max(lower, value))


@dataclass
class CoreState:
    t: float
    body: Dict[str, float]
    expect: Dict[str, float]
    error: Dict[str, float]
    precision: Dict[str, float]
    valence: float = 0.
    arousal: float = 0.
    focus: List[str] = field(default_factory=list)
    open_predictions: List[dict] = field(default_factory=list)


class CoreLoop:
    INTERVAL = .5
    TAU = 600.
    EPSILON = .01
    BODY_WEIGHT = .35
    MAX_OBSERVATION_GAP = 60.
    LABELS = {"joy": (.15, .15), "anger": (-.15, .2), "sorrow": (-.15, .05),
              "fear": (-.15, .2), "surprise": (0., .2), "disgust": (-.1, .1), "neutral": (0., 0.)}

    def __init__(self, gw):
        self.gw = gw
        self.closed = False
        body = self.read_body()
        # Trend observations follow the body's cadence, not this loop's polling rate.
        # They deliberately start fresh after a restart; EMA error is not velocity.
        self.need_observed_at = gw.homeostasis.last_update
        self.need_observed = {k: body[k] for k in ("hunger", "social")}
        self.need_rates = dict.fromkeys(self.need_observed, 0.)
        self.state = CoreState(clock.now(), body, dict(body), dict.fromkeys(CHANNELS, 0.),
                               dict.fromkeys(CHANNELS, 20.), .65*gw.emotion.valence(), .65*gw.emotion.arousal())
        self.variance = dict.fromkeys(CHANNELS, .04)
        self.body_valence = self.body_arousal = 0.
        self.body_weight = self.BODY_WEIGHT       # scoped experimental ablation; never disables the guard
        self.event_valence = self.event_arousal = 0.
        self.guard = WelfareGuard(gw)
        self.history = deque(maxlen=2880)
        self.last_history = -math.inf
        self.original = {"steps": gw.agent_max_steps, "threshold": gw.workspace.ignition_threshold,
                         "mood": getattr(gw.episodes, "mood_valence", 0.),
                         "decay": getattr(gw.emotion, "negative_decay_multiplier", 1.)}
        saved = gw.state_store.latest_core()
        if saved:
            try:
                self.restore(saved)
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring invalid saved core state")
        self.state.body = self.read_body()
        self.state.error = {k: self.state.body[k]-self.state.expect[k] for k in CHANNELS}
        self.sync_controls()

    def read_body(self):
        h, m = self.gw.homeostasis, self.gw.metabolism
        return {k: bound(v, 0., 1.) for k, v in {
            "energy": m.energy/m.max_energy, "hunger": h.hunger, "social": h.social_need,
            "curiosity": h.curiosity_need, "hygiene": h.hygiene_need}.items()}

    def step(self, dt):
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("invalid core timestep")
        s = self.state
        s.t = clock.now()
        s.body = self.read_body()
        self._observe_need_trends(s.body)
        follow = -math.expm1(-dt/self.TAU)
        for key in CHANNELS:
            s.expect[key] += follow*(s.body[key]-s.expect[key])
            s.error[key] = s.body[key]-s.expect[key]
            self.variance[key] += follow*(s.error[key]**2-self.variance[key])
            s.precision[key] = bound(1/(self.variance[key]+self.EPSILON), .25, 20.)
        pleasant = sum(WEIGHTS[k]*s.precision[k]*s.error[k]*(1 if k == "energy" else -1) for k in CHANNELS)
        surprise = sum(WEIGHTS[k]*s.precision[k]*abs(s.error[k]) for k in CHANNELS)
        self.body_valence, self.body_arousal = math.tanh(pleasant), math.tanh(surprise)
        event_v = bound(self.gw.emotion.valence()+self.event_valence, -1., 1.)
        event_a = bound(self.gw.emotion.arousal()+self.event_arousal, 0., 1.)
        # Ablating body input does not increase the event weight.
        target_v = self.body_weight*self.body_valence + .65*event_v
        target_a = self.body_weight*self.body_arousal + .65*event_a
        s.valence = bound(s.valence + follow*(target_v-s.valence), -1., 1.)
        s.arousal = bound(s.arousal + follow*(target_a-s.arousal), 0., 1.)
        self.event_valence *= math.exp(-dt/self.TAU)
        self.event_arousal *= math.exp(-dt/self.TAU)
        self.guard.observe(s.t, s.valence, self.gw.existential_anxiety, self.diagnostics())
        self.sync_controls()
        if s.t-self.last_history >= 30:
            self.history.append({"t": s.t, "valence": s.valence, "arousal": s.arousal,
                                 "error": dict(s.error), "recovering": self.guard.active})
            self.last_history = s.t
        if s.t-self.gw.last_snapshot_time >= 300:
            self.gw._save_state("core")
        return s

    def sync_controls(self):
        s = self.state
        self.gw.agent_max_steps = min(6, max(2, 2+round(4*s.body["energy"]*(1-.4*s.arousal))))
        self.gw.workspace.ignition_threshold = .3-.1*(s.arousal-.5)
        self.gw.episodes.mood_valence = s.valence

    def generation_params(self):
        s = self.state
        negative = max(-s.valence, 0.)
        energy = s.body["energy"]
        return {"max_tokens": max(48, int((60+240*energy)*(1-.6*negative*s.arousal))),
                "temperature": round(bound(.5+.5*s.arousal-1.1*negative*s.arousal, .1, 1.1), 3),
                "delay": 0. if energy >= .5 else min(30., (.5-energy)*60)}

    def _observe_need_trends(self, body):
        stamp = self.gw.homeostasis.last_update
        elapsed = stamp-self.need_observed_at
        age = clock.now()-stamp
        missing = elapsed < 0 or elapsed > self.MAX_OBSERVATION_GAP or not 0 <= age <= self.MAX_OBSERVATION_GAP
        for need in self.need_rates:
            if missing:
                self.need_rates[need] = 0.
            elif elapsed > 0:
                change = body[need]-self.need_observed[need]
                # A relief action is a level shift, not a sustained negative drift.
                # The next nonnegative body update replaces the retained rate.
                if change >= 0:
                    self.need_rates[need] = change/elapsed
            self.need_observed[need] = body[need]
        self.need_observed_at = stamp

    def anticipated_needs(self, actual, horizon):
        horizon = bound(horizon, 0., float("inf"))
        # The heartbeat can decide immediately after ticking the body, before us.
        self._observe_need_trends(self.read_body())
        predicted = dict(actual)
        for need in ("hunger", "social"):
            extra = min(.3, self.need_rates[need]*horizon)
            predicted[need] = min(1., actual.get(need, 0.)+extra)
        return predicted

    def record_reply_label(self, label):
        # A category is an event; arbitrary model-supplied numbers never become state.
        if label in self.LABELS:
            self.event_valence, self.event_arousal = self.LABELS[label]
            return True
        return False

    def diagnostics(self):
        return {"body_valence": self.body_valence, "body_arousal": self.body_arousal,
                "body_contribution": self.body_weight*self.body_valence,
                "errors": dict(self.state.error), "body_weight": self.body_weight,
                "need_rates": dict(self.need_rates), "need_observed_at": self.need_observed_at}

    def payload(self):
        return {"version": 1, "state": asdict(self.state), "variance": dict(self.variance),
                "diagnostics": self.diagnostics(), "welfare": self.guard.payload()}

    def restore(self, payload):
        raw = payload["state"]
        if payload.get("version") != 1:
            raise ValueError("unsupported core version")
        # Validate everything before replacing live state.
        body = {k: bound(raw["body"][k], 0., 1.) for k in CHANNELS}
        expect = {k: bound(raw["expect"][k], 0., 1.) for k in CHANNELS}
        precision = {k: max(.25, bound(raw["precision"][k], .25, 20.)*.5) for k in CHANNELS}
        valence, arousal = bound(raw["valence"], -1., 1.), bound(raw["arousal"], 0., 1.)
        self.state = CoreState(clock.now(), body, expect, {k: body[k]-expect[k] for k in CHANNELS},
                               precision, valence, arousal)
        self.variance = {k: 1/p-self.EPSILON for k, p in precision.items()}
        self.guard.restore(payload.get("welfare") or {})

    async def run(self):
        previous = clock.now()
        while not self.closed:
            await asyncio.sleep(self.INTERVAL)
            if self.closed:
                break
            now = clock.now()
            try:
                self.step(max(0., now-previous))
            except Exception:
                logger.exception("Core loop step failed")
            previous = now

    def close(self):
        self.closed = True
        self.gw._save_state("core:disable")
        self.gw.agent_max_steps = self.original["steps"]
        self.gw.workspace.ignition_threshold = self.original["threshold"]
        self.gw.episodes.mood_valence = self.original["mood"]
        self.gw.emotion.negative_decay_multiplier = self.original["decay"]
