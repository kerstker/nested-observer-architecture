import math
import random
import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

def digamma(x: float) -> float:
    """psi(x) for x > 0: recurrence up to x >= 6, then the asymptotic series."""
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv, inv2 = 1.0 / x, 1.0 / (x * x)
    return result + math.log(x) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252))

def expected_information_gain(a: float, b: float) -> float:
    """
    How much one more outcome would teach her about a success probability p ~ Beta(a, b):
    the mutual information between the next Bernoulli outcome and p,
    I = H(Bern(E[p])) - E_p[H(Bern(p))]. Large when she has little experience, ~0 when she is sure.
    """
    n = a + b
    mu = a / n
    h = -(mu * math.log(mu) + (1 - mu) * math.log(1 - mu)) if 0 < mu < 1 else 0.0
    expected_h = -(mu * (digamma(a + 1) - digamma(n + 1)) + (1 - mu) * (digamma(b + 1) - digamma(n + 1)))
    return max(0.0, h - expected_h)

@dataclass
class Option:
    need: str      # social / hunger / curiosity / none
    action: str    # chat / ask_food / search / wait
    target: Optional[str]
    score: float   # = -G, the negative expected free energy
    pragmatic: float = 0.0
    epistemic: float = 0.0
    cost: float = 0.0
    belief: float = 0.5     # what she expects (probability it works)

class DriveSystem:
    """
    Level 8: Drives as active inference.

    Every self-initiated action is chosen by one principle: minimise expected free energy,
        G(action) = cost - pragmatic value - epistemic value
    - pragmatic value: how much the action is expected to bring her toward the states she prefers
      (not lonely, not hungry, not bored) - expected outcome x how far it would move each need;
    - epistemic value: how much the outcome would teach her about the world - the expected
      information gain of her belief about whether the action works. Curiosity is not a separate
      rule: uncertain things (a new person, a group she has never tried) are worth finding out.
    Beliefs are Beta distributions over "does this work" (not single numbers), updated by Bayes
    with slow forgetting, so she also knows how sure she is. Choice is a softmax over -G.
    """
    COSTS = {"chat": 0.1, "ask_food": 0.1, "search": 0.15, "interject": 0.0}
    GAINS = {"chat": 0.4, "ask_food": 0.6, "search": 0.5}          # how far one success moves the need
    PREF_SCALE = 1.5
    FORGETTING = 0.98                                              # old evidence slowly counts less

    def __init__(self, db_path: str, temperature: float = 0.03, initial_q: float = 0.5, rng: random.Random = None):
        self.db_path = db_path
        self.temperature = temperature
        self.prior: Tuple[float, float] = (1.0 + initial_q - 0.5, 1.0 + 0.5 - initial_q)   # (1, 1) by default
        self.rng = rng or random.Random()
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS beliefs (
            need TEXT, action TEXT, target TEXT, alpha REAL, beta REAL, n INTEGER,
            PRIMARY KEY (need, action, target)
        )''')
        # Migration: learned values from the earlier Q-learning become equivalent evidence
        if conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0] == 0:
            try:
                rows = conn.execute("SELECT need, action, target, q, n FROM q_values").fetchall()
            except sqlite3.OperationalError:
                rows = []
            for need, action, target, q, n in rows:
                conn.execute("INSERT OR IGNORE INTO beliefs VALUES (?, ?, ?, ?, ?, ?)",
                             (need, action, target, self.prior[0] + q * n, self.prior[1] + (1 - q) * n, n))
        conn.commit()
        conn.close()

    # --- beliefs ------------------------------------------------------------------

    def belief(self, need: str, action: str, target: Optional[str] = None) -> Tuple[float, float, int]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT alpha, beta, n FROM beliefs WHERE need=? AND action=? AND target=?",
                           (need, action, target or "")).fetchone()
        conn.close()
        return row if row else (self.prior[0], self.prior[1], 0)

    def q(self, need: str, action: str, target: Optional[str] = None) -> float:
        """Expected probability that this action works (posterior mean)."""
        a, b, _ = self.belief(need, action, target)
        return a / (a + b)

    def uncertainty(self, need: str, action: str, target: Optional[str] = None) -> float:
        a, b, _ = self.belief(need, action, target)
        return expected_information_gain(a, b)

    def reward(self, need: str, action: str, target: Optional[str], reward: float, learning_gain: float = 1.0) -> float:
        """Bayesian update with an observed outcome in [0, 1]. Returns the new expected value."""
        a, b, n = self.belief(need, action, target)
        a0, b0 = self.prior
        a = a0 + (a - a0) * self.FORGETTING + learning_gain * reward
        b = b0 + (b - b0) * self.FORGETTING + learning_gain * (1 - reward)
        conn = sqlite3.connect(self.db_path)
        conn.execute("REPLACE INTO beliefs (need, action, target, alpha, beta, n) VALUES (?, ?, ?, ?, ?, ?)",
                     (need, action, target or "", a, b, n + 1))
        conn.commit()
        conn.close()
        return a / (a + b)

    # --- choosing -----------------------------------------------------------------

    def _evaluate(self, need: str, action: str, target: Optional[str], urgency: float,
                  curiosity: float, cost: float, goal: float = 0.0) -> Option:
        a, b, _ = self.belief(need, action, target)
        mu = a / (a + b)
        gain = self.GAINS.get(action, 0.4)
        # Preferences are quadratic in how unmet a need is: relief from a pressing need counts most
        relief = urgency ** 2 - max(0.0, urgency - gain) ** 2
        # Long-term goals are preferred states too: an action that would move one forward is worth more
        pragmatic = self.PREF_SCALE * mu * relief + mu * goal
        epistemic = (0.3 + 0.7 * curiosity) * expected_information_gain(a, b)
        return Option(need, action, target, pragmatic + epistemic - cost, pragmatic, epistemic, cost, mu)

    def options(self, needs: Dict[str, float], energy: float, users: List[str],
                goal_value: Callable[[str, Optional[str]], float] = None) -> List[Option]:
        fatigue = max(0.0, (40.0 - energy) / 100.0)     # acting costs more when tired
        curiosity = needs.get("curiosity", 0.0)
        gv = goal_value or (lambda action, target: 0.0)
        opts = [Option("none", "wait", None, 0.0)]
        for user in users:
            opts.append(self._evaluate("social", "chat", user, needs.get("social", 0.0), curiosity,
                                       self.COSTS["chat"] + fatigue, gv("chat", user)))
            opts.append(self._evaluate("hunger", "ask_food", user, needs.get("hunger", 0.0), curiosity,
                                       self.COSTS["ask_food"] + fatigue))
        opts.append(self._evaluate("curiosity", "search", None, curiosity, curiosity,
                                   self.COSTS["search"] + fatigue, gv("search", None)))
        return opts

    def choose(self, needs: Dict[str, float], energy: float, users: List[str],
               goal_value: Callable[[str, Optional[str]], float] = None) -> Option:
        opts = self.options(needs, energy, users, goal_value)
        top = max(o.score for o in opts)
        weights = [math.exp((o.score - top) / self.temperature) for o in opts]
        return self.rng.choices(opts, weights=weights, k=1)[0]

    @staticmethod
    def describe(o: Option) -> str:
        if o.action == "wait":
            return "什么都不做也挺好"
        return (f"预期满足 {o.pragmatic:.2f} + 想弄清楚 {o.epistemic:.2f} − 代价 {o.cost:.2f} "
                f"（我估计成功的可能 {o.belief:.0%}）")
