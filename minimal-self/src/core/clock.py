import time

class Clock:
    """Wall-clock time. Every module reads time through `clock.now()` so that the
    whole system can be run on a simulated clock (a week of life in a few minutes)."""
    def now(self) -> float:
        return time.time()

class SimClock(Clock):
    """A manually advanced clock for simulations and tests."""
    def __init__(self, start: float = None):
        self.t = time.time() if start is None else start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float):
        self.t += seconds

_clock: Clock = Clock()

def now() -> float:
    return _clock.now()

def set_clock(c: Clock):
    global _clock
    _clock = c

def get_clock() -> Clock:
    return _clock
