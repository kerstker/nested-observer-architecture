from dataclasses import dataclass, fields
from typing import Dict
from src.core import clock

@dataclass
class EmotionState:
    joy: float = 0.0
    anger: float = 0.0
    sorrow: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0

class EmotionEngine:
    """
    Level 4: Emotion System (Enhanced)
    Manages 6 basic emotions + compound states.

    Two things make the numbers mean something:
    - Each emotion fades at its own pace (`HALF_LIFE_SCALE` x the base half-life, joy's): a startle is
      gone in about a minute, fear and disgust in minutes, anger in a quarter of an hour, sorrow lingers.
      With one shared half-life, surprise stayed near its maximum for as long as joy did.
    - A push has diminishing returns: it closes a share of the *remaining* distance to 1 instead of adding
      a fixed amount, so a steady trickle of small pushes settles at a level rather than pinning the
      emotion at its ceiling (where it says nothing). From rest a push is worth exactly its size.
    """
    HALF_LIFE_SCALE = {"joy": 1.0, "anger": 0.75, "sorrow": 2.25, "fear": 0.5, "surprise": 0.075, "disgust": 0.5}

    def __init__(self, half_life_seconds: float = 20 * 60):
        self.state = EmotionState()
        self.mood_history = []
        # Emotions return to neutral with time: each one halves every `half_life_seconds`.
        self.half_life_seconds = half_life_seconds
        self.last_decay = clock.now()
        # Temperament: the resting point emotions return to (set from personality; zero = neutral)
        self.baseline: Dict[str, float] = {}

    def update(self, delta: EmotionState):
        for f in fields(EmotionState):
            current, push = getattr(self.state, f.name), getattr(delta, f.name)
            value = current + push * (1.0 - current) if push > 0 else current + push
            setattr(self.state, f.name, max(0.0, min(1.0, value)))

    def decay(self, now: float = None):
        """Exponential return toward her temperament (baseline), proportional to time since the last call."""
        now = clock.now() if now is None else now
        elapsed = max(0.0, now - self.last_decay)
        self.last_decay = now
        for f in fields(EmotionState):
            factor = 0.5 ** (elapsed / (self.half_life_seconds * self.HALF_LIFE_SCALE.get(f.name, 1.0)))
            base = self.baseline.get(f.name, 0.0)
            setattr(self.state, f.name, base + (getattr(self.state, f.name) - base) * factor)

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
        # Simple compound logic
        if self.state.joy > 0.5 and self.state.surprise > 0.3:
            return "Excited"
        if self.state.joy > 0.5 and self.state.fear > 0.3:
            return "Nervous Laughter"
        if self.state.anger > 0.5 and self.state.sorrow > 0.3:
            return "Betrayed"
        return dom.capitalize()
