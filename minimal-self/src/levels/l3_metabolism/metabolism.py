from typing import Dict, Any
from src.core import clock

class Metabolism:
    """
    Level 3: Biological State Simulation
    Manages energy, fatigue, and circadian rhythm.
    """
    def __init__(self, max_energy: float = 100.0):
        self.max_energy = max_energy
        self.energy = max_energy
        self.is_sleeping = False
        # Per reply (more for long ones). Calibrated to what she really does: a typical day is ~40 replies, a busy
        # one 300-600 (several QQ groups). Awake she loses 5/h and sleeping restores 15/h, so a day's replies
        # must fit in what is left: at 0.12 a typical day is ~7 h of sleep, 300 replies ~9 h, 600 replies ~12 h.
        # (At the old 0.25 a 600-reply day meant 19 h asleep - it only worked because every message with the
        # word 吃 in it counted as a meal worth +20 energy.)
        self.cognitive_cost = 0.12
        # Rates are per hour of wall-clock time, so behaviour does not depend on tick frequency.
        # Awake: 100 -> 25 takes ~15h; asleep: 25 -> 80 takes ~3.7h.
        self.awake_drain_per_hour = 5.0
        self.sleep_recovery_per_hour = 15.0
        self.last_update = clock.now()

    def tick(self) -> Dict[str, Any]:
        """
        Periodic update. Drains energy while awake and restores it while asleep,
        proportional to the time elapsed since the previous tick. Returns current state.
        """
        now = clock.now()
        elapsed_hours = max(0.0, now - self.last_update) / 3600.0
        self.last_update = now

        if self.is_sleeping:
            self.recover(self.sleep_recovery_per_hour * elapsed_hours)
        else:
            self.consume(self.awake_drain_per_hour * elapsed_hours)

        return self.get_status()

    def consume(self, amount: float):
        self.energy = max(0.0, self.energy - amount)

    def recover(self, amount: float):
        self.energy = min(self.max_energy, self.energy + amount)

    def get_status(self) -> Dict[str, Any]:
        return {
            "energy": self.energy,
            "is_sleeping": self.is_sleeping,
            "status_desc": self.get_status_desc()
        }

    def get_status_desc(self) -> str:
        if self.is_sleeping:
            return "Sleeping (Zzz...)"
        if self.energy > 80:
            return "Energetic"
        if self.energy > 50:
            return "Normal"
        if self.energy > 20:
            return "Tired"
        return "Exhausted"
