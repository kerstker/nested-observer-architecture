from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple
from src.core import clock

@dataclass
class EmotionState:
    joy: float = 0.0
    anger: float = 0.0
    sorrow: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0

NEGATIVE = ("anger", "sorrow", "fear", "disgust")

class EmotionEngine:
    """
    Level 4: Emotion System.

    Built to behave the way a person's feelings do, not as six independent sliders:

    - Each emotion fades at its own pace (`HALF_LIFE_SCALE` x the base half-life, joy's): a startle is
      gone in about a minute, fear and disgust in minutes, anger in a quarter of an hour, sorrow lingers.
    - A push has diminishing returns: it closes a share of the *remaining* distance to 1, so a steady
      trickle of small pushes settles at a level instead of pinning the emotion at its ceiling.
    - Opposites push each other down (`INHIBITION`): good news takes the edge off sorrow and anger, bad news
      dims joy. A little of both at once is real (bittersweet); both near the top at once is not.
    - The same kind of thing happening again and again matters less each time (habituation, per `source`):
      the tenth unanswered message does not hurt like the first.
    - Sleep soothes: negative feelings fade about twice as fast while she sleeps.
    - Under the emotions there is a mood: the slow average of how she has been feeling over hours. A
      moment's feeling is the weather, the mood is the climate; she is described by both.
    """
    HALF_LIFE_SCALE = {"joy": 1.0, "anger": 0.75, "sorrow": 2.25, "fear": 0.5, "surprise": 0.075, "disgust": 0.5}
    INHIBITION = 0.4
    HABITUATION_HALF_LIFE = 3600.0
    HABITUATION = 0.5
    SLEEP_SOOTHING = 2.0
    MOOD_HALF_LIFE = 4 * 3600.0

    def __init__(self, half_life_seconds: float = 20 * 60):
        self.state = EmotionState()
        self.mood_history = []
        # Emotions return to neutral with time: each one halves every `half_life_seconds`.
        self.half_life_seconds = half_life_seconds
        self.last_decay = clock.now()
        # Temperament: the resting point emotions return to (set from personality; zero = neutral)
        self.baseline: Dict[str, float] = {}
        self.mood_valence = 0.0
        self.mood_arousal = 0.0
        self._habit: Dict[str, Tuple[float, float]] = {}          # source -> (how many lately, when)

    # --- change ------------------------------------------------------------------

    def _habituate(self, source: Optional[str], now: float) -> float:
        if not source:
            return 1.0
        count, t = self._habit.get(source, (0.0, now))
        count *= 0.5 ** (max(0.0, now - t) / self.HABITUATION_HALF_LIFE)
        self._habit[source] = (count + 1.0, now)
        return 1.0 / (1.0 + self.HABITUATION * count)

    def update(self, delta: EmotionState, source: str = None):
        """Apply a push. `source` names the kind of event, so repeats of it lose their sting (or sweetness)."""
        scale = self._habituate(source, clock.now())
        s = self.state
        # What is already felt gives way to its opposite - but one event that brings both (bittersweet news)
        # does not cancel itself, so this is worked out before the pushes are added
        joy_gain = max(0.0, delta.joy) * scale * (1.0 - s.joy)
        negative_gain = sum(max(0.0, getattr(delta, n)) * scale * (1.0 - getattr(s, n)) for n in NEGATIVE)
        if joy_gain > 0:
            damp = 1.0 - min(0.6, self.INHIBITION * 2 * joy_gain)
            for name in NEGATIVE:
                setattr(s, name, getattr(s, name) * damp)
        if negative_gain > 0:
            s.joy *= 1.0 - min(0.6, self.INHIBITION * 2 * negative_gain)
        for f in fields(EmotionState):
            current, push = getattr(s, f.name), getattr(delta, f.name)
            value = current + push * scale * (1.0 - current) if push > 0 else current + push
            setattr(s, f.name, max(0.0, min(1.0, value)))

    def decay(self, now: float = None, asleep: bool = False):
        """Exponential return toward her temperament (baseline), proportional to time since the last call."""
        now = clock.now() if now is None else now
        elapsed = max(0.0, now - self.last_decay)
        self.last_decay = now
        for f in fields(EmotionState):
            pace = self.half_life_seconds * self.HALF_LIFE_SCALE.get(f.name, 1.0)
            if asleep and f.name in NEGATIVE:
                pace /= self.SLEEP_SOOTHING
            factor = 0.5 ** (elapsed / pace)
            base = self.baseline.get(f.name, 0.0)
            setattr(self.state, f.name, base + (getattr(self.state, f.name) - base) * factor)
        # the mood follows how she has been feeling, slowly
        keep = 0.5 ** (elapsed / self.MOOD_HALF_LIFE)
        self.mood_valence = self.mood_valence * keep + self.valence() * (1.0 - keep)
        self.mood_arousal = self.mood_arousal * keep + self.arousal() * (1.0 - keep)

    # --- reading -----------------------------------------------------------------

    def valence(self) -> float:
        """Pleasantness in [-1, 1]: joy against the strongest negative emotion."""
        s = self.state
        return max(-1.0, min(1.0, s.joy - max(s.anger, s.sorrow, s.fear, s.disgust)))

    def arousal(self) -> float:
        """Activation in [0, 1]: driven by the most activating emotion."""
        s = self.state
        return max(0.0, min(1.0, max(0.7 * s.joy, s.anger, s.fear, 0.8 * s.surprise, 0.4 * s.sorrow)))

    def to_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self.state, f.name) for f in fields(EmotionState)}

    def load_dict(self, values: Dict[str, float]):
        for f in fields(EmotionState):
            setattr(self.state, f.name, float(values.get(f.name, 0.0)))

    def mood_dict(self) -> Dict[str, float]:
        return {"valence": self.mood_valence, "arousal": self.mood_arousal}

    def load_mood(self, values: Dict[str, float]):
        self.mood_valence = float((values or {}).get("valence", 0.0))
        self.mood_arousal = float((values or {}).get("arousal", 0.0))

    def get_dominant_emotion(self) -> str:
        emotions = {
            "joy": self.state.joy,
            "anger": self.state.anger,
            "sorrow": self.state.sorrow,
            "fear": self.state.fear,
            "surprise": self.state.surprise,
            "disgust": self.state.disgust
        }
        # Filter emotions below threshold
        active_emotions = {k: v for k, v in emotions.items() if v > 0.1}

        if not active_emotions:
            return "neutral"

        return max(active_emotions, key=active_emotions.get)

    def get_compound_state(self) -> str:
        """Return complex state description e.g. 'Anxious' (Fear + Surprise)"""
        dom = self.get_dominant_emotion()
        s = self.state
        if s.joy > 0.5 and s.surprise > 0.3:
            return "Excited"
        if s.joy > 0.5 and s.fear > 0.3:
            return "Nervous Laughter"
        if s.anger > 0.5 and s.sorrow > 0.3:
            return "Betrayed"
        if s.joy > 0.25 and s.sorrow > 0.25:
            return "Bittersweet"
        return dom.capitalize()

    # --- in words ----------------------------------------------------------------

    SINGLE = {"joy": ("有点开心", "挺开心", "很开心"), "sorrow": ("有点失落", "难过", "很难过"),
              "anger": ("有点不爽", "生气", "很生气"), "fear": ("有点不安", "害怕", "很害怕"),
              "surprise": ("有点意外", "很惊讶", "吓了一跳"), "disgust": ("有点反感", "很反感", "恶心")}
    MIXED = {frozenset(("joy", "sorrow")): "开心里带着点失落，说不清是好是坏",
             frozenset(("anger", "sorrow")): "又气又委屈",
             frozenset(("fear", "sorrow")): "不安，还有点难过",
             frozenset(("joy", "surprise")): "又惊又喜",
             frozenset(("joy", "fear")): "又紧张又期待",
             frozenset(("anger", "disgust")): "又烦又反感",
             frozenset(("fear", "surprise")): "被吓了一跳",
             frozenset(("anger", "fear")): "又气又怕"}

    def _level(self, name: str) -> str:
        v = getattr(self.state, name)
        return self.SINGLE[name][0 if v < 0.3 else 1 if v < 0.6 else 2]

    def describe(self) -> str:
        """How she feels, as she would say it: the moment (one feeling, or a mixed one) and the mood under it."""
        ranked = sorted(((getattr(self.state, f.name), f.name) for f in fields(EmotionState)), reverse=True)
        (v1, n1), (v2, n2) = ranked[0], ranked[1]
        if v1 < 0.12:
            now = "心里挺平静的"
        elif v2 >= 0.2 and v2 >= 0.55 * v1 and frozenset((n1, n2)) in self.MIXED:
            now = self.MIXED[frozenset((n1, n2))]
        else:
            now = self._level(n1)
            if v2 >= 0.25 and n2 != n1:
                now += f"，也有点{self._level(n2).replace('有点', '')}"
        m = self.mood_valence
        if m < -0.3:
            under = "这阵子心里一直不太舒服"
        elif m < -0.12:
            under = "这半天整体有点闷"
        elif m > 0.3:
            under = "这半天整体心情很好"
        elif m > 0.12:
            under = "这半天整体心情不错"
        else:
            under = ""
        return now + (f"（{under}）" if under else "")
