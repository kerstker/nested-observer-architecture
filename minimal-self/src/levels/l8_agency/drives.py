import math
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Option:
    need: str      # social / hunger / curiosity
    action: str    # chat / ask_food / search / wait
    target: Optional[str]
    score: float

class DriveSystem:
    """
    Level 8: Closed-loop drives.

    Each need has actions Noa can take on her own. For every (need, action, target)
    she keeps a learned value Q = how much that action has actually relieved that
    need before. Each decision scores options as urgency x Q - cost and samples one
    with a softmax, so she keeps some exploration but increasingly does what works:
    after a week she should seek out the people who answer her, not the ones who
    ignore her.
    """
    COSTS = {"chat": 0.1, "ask_food": 0.1, "search": 0.15}

    def __init__(self, db_path: str, temperature: float = 0.1, initial_q: float = 0.5, rng: random.Random = None):
        self.db_path = db_path
        self.temperature = temperature
        self.initial_q = initial_q
        self.rng = rng or random.Random()
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS q_values (
            need TEXT, action TEXT, target TEXT, q REAL, n INTEGER,
            PRIMARY KEY (need, action, target)
        )''')
        conn.commit()
        conn.close()

    def _get(self, need: str, action: str, target: Optional[str]):
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT q, n FROM q_values WHERE need=? AND action=? AND target=?",
                           (need, action, target or "")).fetchone()
        conn.close()
        return row if row else (self.initial_q, 0)

    def q(self, need: str, action: str, target: Optional[str] = None) -> float:
        return self._get(need, action, target)[0]

    def options(self, needs: Dict[str, float], energy: float, users: List[str]) -> List[Option]:
        # Acting costs more when tired
        fatigue = max(0.0, (40.0 - energy) / 100.0)
        opts = [Option("none", "wait", None, 0.0)]
        for user in users:
            opts.append(Option("social", "chat", user,
                               needs.get("social", 0) * self.q("social", "chat", user) - self.COSTS["chat"] - fatigue))
            opts.append(Option("hunger", "ask_food", user,
                               needs.get("hunger", 0) * self.q("hunger", "ask_food", user) - self.COSTS["ask_food"] - fatigue))
        opts.append(Option("curiosity", "search", None,
                           needs.get("curiosity", 0) * self.q("curiosity", "search") - self.COSTS["search"] - fatigue))
        return opts

    def choose(self, needs: Dict[str, float], energy: float, users: List[str]) -> Option:
        opts = self.options(needs, energy, users)
        top = max(o.score for o in opts)
        weights = [math.exp((o.score - top) / self.temperature) for o in opts]
        return self.rng.choices(opts, weights=weights, k=1)[0]

    def reward(self, need: str, action: str, target: Optional[str], reward: float, learning_gain: float = 1.0) -> float:
        """Move Q toward the observed relief. Returns the new value."""
        q, n = self._get(need, action, target)
        n += 1
        rate = min(1.0, max(0.1, 1.0 / n) * learning_gain)
        q += rate * (reward - q)
        conn = sqlite3.connect(self.db_path)
        conn.execute("REPLACE INTO q_values (need, action, target, q, n) VALUES (?, ?, ?, ?, ?)",
                     (need, action, target or "", q, n))
        conn.commit()
        conn.close()
        return q
