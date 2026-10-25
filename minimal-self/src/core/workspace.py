import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from src.core import clock

@dataclass
class Candidate:
    """Something competing for conscious access."""
    source: str        # perception / needs / memory / prediction / continuity / inner_speech / self_model
    kind: str          # user_message / hunger / social / curiosity / fatigue / pain / ignored / ...
    content: str       # first-person description, as it would appear in her mind
    salience: float    # computed by the proposing module
    payload: Dict[str, Any] = field(default_factory=dict)

class Workspace:
    """
    Level 10: Global Workspace (Baars / Dehaene).

    Modules propose candidates; only those whose salience crosses the ignition
    threshold compete, and at most `capacity` of them win. Winners are broadcast to
    every module (via the EventBus), and the sequence of winners is the stream of
    consciousness. Limited capacity is what makes attention - and distraction -
    possible: when she is very hungry, hunger can take a slot from a conversation.
    """
    def __init__(self, capacity: int = 2, ignition_threshold: float = 0.3, noise: float = 0.05,
                 rng: random.Random = None, trace_len: int = 50):
        self.capacity = capacity
        self.ignition_threshold = ignition_threshold
        self.noise = noise
        self.rng = rng or random.Random()
        self.trace = deque(maxlen=trace_len)   # (t, [Candidate])
        self.attention: Optional[str] = None

    def compete(self, candidates: List[Candidate], bonus: Callable[[Candidate], float] = None) -> List[Candidate]:
        """`bonus` adds context-dependent relevance (e.g. things about the person she is talking to)."""
        scored = []
        for c in candidates:
            # Neural noise: near-equal candidates do not always resolve the same way
            effective = c.salience + (bonus(c) if bonus else 0.0) + self.rng.gauss(0.0, self.noise)
            if effective >= self.ignition_threshold:
                scored.append((effective, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        winners = [c for _, c in scored[:self.capacity]]
        if winners:
            self.trace.append((clock.now(), winners))
            self.attention = winners[0].kind
        return winners

    def recent(self, n: int = 5) -> List[Candidate]:
        out = []
        for _, winners in list(self.trace)[-n:]:
            out.extend(winners)
        return out
