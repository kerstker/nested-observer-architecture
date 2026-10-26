import json
import math
import sqlite3
from typing import Dict, Any, List, Optional

# ToM's coarse emotion guess mapped onto a sentiment axis
SENTIMENT = {"sad": -0.6, "angry": -0.8, "happy": 0.7, "tired": -0.2, "neutral": 0.0}

class UserModelStore:
    """
    Level 7: A persistent, predictive model of each person.

    Whenever Noa says something that invites an answer, she predicts how long the
    person will take and in what mood they will answer. When the answer comes (or
    does not), the prediction error is what makes it feel surprising, and it is also
    what updates the model - she learns who answers quickly, who is usually happy,
    and who tends to ignore her.
    """
    DEFAULT_LATENCY = 600.0   # seconds, before anything is known
    DEFAULT_SPREAD = 1.5      # how widely a person's answer times vary (std of log-seconds) before anything is known
    MIN_SPREAD = 0.6          # ... and never trusted to be narrower than this

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.models: Dict[str, Dict[str, Any]] = {}
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS user_models (user_id TEXT PRIMARY KEY, data TEXT)")
        for user_id, data in conn.execute("SELECT user_id, data FROM user_models").fetchall():
            self.models[user_id] = json.loads(data)
        conn.commit()
        conn.close()

    def get(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.models:
            self.models[user_id] = {
                "log_latency": math.log(self.DEFAULT_LATENCY),
                "log_latency_var": self.DEFAULT_SPREAD ** 2,
                "sentiment": 0.0,
                "invitations": 0,     # times Noa said something expecting an answer
                "answered": 0,        # ... and got one within the expected window
                "private": False,
                "pending": None,
            }
        return self.models[user_id]

    def save(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("REPLACE INTO user_models (user_id, data) VALUES (?, ?)",
                     (user_id, json.dumps(self.get(user_id), ensure_ascii=False)))
        conn.commit()
        conn.close()

    def expected_latency(self, user_id: str) -> float:
        return math.exp(self.get(user_id)["log_latency"])

    def spread(self, user_id: str) -> float:
        """Std of this person's log answer time: people who answer in 5 s or in 2 h are not surprising either way."""
        return max(self.MIN_SPREAD, math.sqrt(self.get(user_id).get("log_latency_var", self.DEFAULT_SPREAD ** 2)))

    def response_probability(self, user_id: str) -> float:
        m = self.get(user_id)
        return (m["answered"] + 1) / (m["invitations"] + 2)   # Laplace-smoothed

    def private_users(self) -> List[str]:
        return [u for u, m in self.models.items() if m.get("private")]

    def on_noa_message(self, user_id: str, now: float, kind: str, expects_reply: bool, need: str = None):
        """Noa just spoke to `user_id`; if it invites an answer, predict it."""
        m = self.get(user_id)
        if not expects_reply:
            m["pending"] = None
        else:
            m["invitations"] += 1
            m["pending"] = {"t": now, "kind": kind, "need": need,
                            "expected_latency": self.expected_latency(user_id),
                            "expected_sentiment": m["sentiment"],
                            "p_reply": self.response_probability(user_id),
                            "ignored": False}
        self.save(user_id)

    def on_user_message(self, user_id: str, now: float, sentiment: float, private: bool,
                        learning_rate: float = 0.2) -> Optional[Dict[str, Any]]:
        """The person spoke. Returns the prediction error if an answer was expected."""
        m = self.get(user_id)
        m["private"] = m["private"] or private
        m["messages"] = m.get("messages", 0) + 1
        pending = m["pending"]
        error = None
        if pending:
            latency = max(1.0, now - pending["t"])
            latency_error = math.log(latency / pending["expected_latency"])   # > 0: slower than expected
            # ... in units of how much this person's answer times usually vary: a "faster than expected"
            # that happens a third of the time (anyone who answers in bursts and then disappears) is no surprise
            spread = self.spread(user_id)
            latency_z = latency_error / spread
            sentiment_error = sentiment - pending["expected_sentiment"]
            on_time = not pending["ignored"]
            if on_time:
                m["answered"] += 1
            deviation = math.log(latency) - m["log_latency"]
            m["log_latency"] += learning_rate * deviation
            m["log_latency_var"] = m.get("log_latency_var", self.DEFAULT_SPREAD ** 2) * (1 - learning_rate)                 + learning_rate * deviation ** 2
            error = {"user_id": user_id, "kind": pending["kind"], "need": pending["need"],
                     "on_time": on_time, "latency": latency,
                     "expected_latency": pending["expected_latency"],
                     "latency_error": latency_error, "latency_z": latency_z, "sentiment_error": sentiment_error,
                     # A late answer after being given up on is itself a (pleasant) surprise; a normal-sized
                     # deviation (|z| <= 1) is no surprise at all
                     "surprise": min(1.0, max(0.0, abs(latency_z) - 1.0) / 4 + abs(sentiment_error) / 2
                                     + (0.0 if on_time else 0.4))}
            m["pending"] = None
        m["sentiment"] += learning_rate * (sentiment - m["sentiment"])
        self.save(user_id)
        return error

    def check_timeouts(self, now: float) -> List[Dict[str, Any]]:
        """Answers that should have arrived by now but did not."""
        errors = []
        for user_id, m in self.models.items():
            p = m["pending"]
            if not p or p["ignored"]:
                continue
            window = max(3 * p["expected_latency"], 1800.0)
            if now - p["t"] > window:
                p["ignored"] = True
                errors.append({"user_id": user_id, "kind": p["kind"], "need": p["need"],
                               "elapsed": now - p["t"], "expected_latency": p["expected_latency"],
                               # Being ignored by someone who always answers is the most surprising
                               "surprise": p["p_reply"]})
                self.save(user_id)
        return errors

    def describe(self, user_id: str) -> str:
        m = self.get(user_id)
        minutes = self.expected_latency(user_id) / 60
        mood = "偏开心" if m["sentiment"] > 0.2 else "偏低落" if m["sentiment"] < -0.2 else "平稳"
        if m["invitations"]:
            return (f"ta 通常约 {minutes:.0f} 分钟内回我，我找 ta 时 ta 回应的概率约 "
                    f"{self.response_probability(user_id):.0%}，最近情绪{mood}。")
        return f"我对 ta 还不太了解，最近情绪{mood}。"
