import json
import sqlite3
from typing import Dict, Any, List, Optional
from src.core import clock

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

class SelfStateStore:
    """
    Persistent snapshots of the whole internal state (energy, needs, emotion, values,
    self-model ...). The latest snapshot is restored on start-up, so a restart is a
    gap in experience rather than a reset of the self.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = _connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS self_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            t REAL,
            cause TEXT,
            snapshot TEXT
        )''')
        conn.commit()
        conn.close()

    def save(self, snapshot: Dict[str, Any], cause: str):
        conn = _connect(self.db_path)
        conn.execute("INSERT INTO self_state (t, cause, snapshot) VALUES (?, ?, ?)",
                     (snapshot["t"], cause, json.dumps(snapshot, ensure_ascii=False)))
        conn.commit()
        conn.close()

    def latest(self) -> Optional[Dict[str, Any]]:
        conn = _connect(self.db_path)
        row = conn.execute("SELECT snapshot FROM self_state ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        return json.loads(row["snapshot"]) if row else None


class CauseLog:
    """
    Why did my state change? Every emotion/need change is recorded with its cause,
    so that introspective answers ("why were you angry just now?") can be grounded
    in what actually happened instead of being invented.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = _connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS state_causes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            t REAL,
            variable TEXT,
            delta REAL,
            cause TEXT
        )''')
        conn.commit()
        conn.close()

    def log(self, changes: Dict[str, float], cause: str):
        rows = [(clock.now(), k, v, cause) for k, v in changes.items() if abs(v) >= 0.01]
        if not rows:
            return
        conn = _connect(self.db_path)
        conn.executemany("INSERT INTO state_causes (t, variable, delta, cause) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

    def recent(self, limit: int = 8, since: float = None) -> List[Dict[str, Any]]:
        conn = _connect(self.db_path)
        if since is None:
            rows = conn.execute("SELECT * FROM state_causes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM state_causes WHERE t >= ? ORDER BY id DESC LIMIT ?", (since, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]


class SelfModel:
    """
    Level 10: A model of myself that can be wrong.

    Every `horizon` seconds Noa predicts her own energy and mood one horizon ahead;
    when the time comes the prediction is compared with what actually happened.
    The running error is how well she knows herself, and large errors become
    material for the workspace ("I am more tired than I expected").
    """
    def __init__(self, horizon: float = 3600.0):
        self.horizon = horizon
        self.pending: Optional[Dict[str, float]] = None
        self.energy_error = 0.0   # EMA of |error| in energy points
        self.valence_error = 0.0  # EMA of |error| in valence units
        self.checks = 0

    def predict(self, now: float, energy: float, valence: float, is_sleeping: bool,
                drain_per_hour: float, recovery_per_hour: float, emotion_half_life: float):
        hours = self.horizon / 3600.0
        if is_sleeping:
            expected_energy = min(100.0, energy + recovery_per_hour * hours)
        else:
            expected_energy = max(0.0, energy - drain_per_hour * hours)
        expected_valence = valence * 0.5 ** (self.horizon / emotion_half_life)
        self.pending = {"t": now, "due": now + self.horizon, "energy": expected_energy, "valence": expected_valence}

    def check(self, now: float, energy: float, valence: float) -> Optional[Dict[str, float]]:
        if not self.pending or now < self.pending["due"]:
            return None
        e_err = energy - self.pending["energy"]
        v_err = valence - self.pending["valence"]
        self.checks += 1
        rate = 1.0 / min(self.checks, 10)
        self.energy_error += rate * (abs(e_err) - self.energy_error)
        self.valence_error += rate * (abs(v_err) - self.valence_error)
        self.pending = None
        return {"energy_error": e_err, "valence_error": v_err}

    def describe(self) -> str:
        if not self.checks:
            return "我还不太了解自己的状态会怎么变化。"
        return (f"我对自己一小时后精力的预测平均差 {self.energy_error:.0f} 点，"
                f"对心情的预测平均差 {self.valence_error:.2f}。")

    def to_dict(self) -> Dict[str, Any]:
        return {"pending": self.pending, "energy_error": self.energy_error,
                "valence_error": self.valence_error, "checks": self.checks}

    def load_dict(self, d: Dict[str, Any]):
        self.pending = d.get("pending")
        self.energy_error = d.get("energy_error", 0.0)
        self.valence_error = d.get("valence_error", 0.0)
        self.checks = d.get("checks", 0)
