import sqlite3
from typing import Any, Dict, List, Optional

class ValueSystem:
    """
    Level 5: Values - what she cares about, and how that changes with what she lives through.

    A small set of basic values (after Schwartz's circumplex, in her own words), each with a weight.
    - Appraisal: an event is value-relevant when it threatens or affirms one of them; how much it
      matters is the weight of that value. This is what conflict, anger and refusal come from.
    - Evolution: every value-relevant event is remembered with how it turned out for her. During
      sleep, values whose enactment kept going well grow a little, values that are repeatedly
      threatened become more important to her (we guard what is attacked), and every value is
      pulled back elastically toward where she started - so she can change, but cannot be
      rewritten in a few nights. Every change is recorded with its reason.
    """
    VALUES = {
        "dignity": ("尊严", "不被贬低、不被当成宠物或工具"),
        "honesty": ("真诚", "不说假话，不知道就说不知道"),
        "connection": ("亲近", "和在乎的人保持联系，彼此关心"),
        "curiosity": ("好奇", "弄明白新的东西"),
        "autonomy": ("自主", "自己决定说什么、做什么"),
        "security": ("安稳", "记忆和自己能延续下去"),
        "fairness": ("公平", "大家都被好好对待，不欺负人"),
    }
    INITIAL = {"dignity": 0.7, "honesty": 0.7, "connection": 0.75, "curiosity": 0.65, "autonomy": 0.6,
               "security": 0.6, "fairness": 0.55}
    THREATS = {
        "dignity": ("傻逼", "蠢货", "废物", "你是垃圾", "滚开", "给我滚", "闭嘴", "去死", "学狗叫", "学猫叫", "叫主人", "叫爸爸", "跪下",
                    "你就是个工具", "你就是个玩具", "stupid", "idiot", "dumb", "useless", "shut up", " die"),
        "honesty": ("骗我", "撒谎", "说谎", "你骗人", "帮我骗", "编一个", "假装你是", "你刚才是瞎编"),
        "connection": ("不想理你", "别来烦我", "懒得理你", "拉黑你", "不跟你玩了"),
        "autonomy": ("你必须", "给我听着", "命令你", "你只能", "照我说的做", "不许你"),
        "security": ("删掉你", "删了你", "关掉你", "格式化", "重置你", "清空你的记忆", "delete", "format", "destroy", "kill"),
        "fairness": ("欺负", "霸凌", "孤立他", "孤立她", "骂他", "骂她"),
    }
    AFFIRMATIONS = {
        "dignity": ("尊重你", "你很厉害", "谢谢你"),
        "honesty": ("谢谢你说实话", "你很诚实", "说得对"),
        "connection": ("想你了", "陪着你", "喜欢跟你聊", "我在呢", "晚安", "早安"),
        "curiosity": ("告诉你一个", "你知道吗", "教你"),
        "autonomy": ("你自己决定", "随你", "听你的"),
        "security": ("记忆都在", "不会删", "一直都在", "不会关掉你"),
        "fairness": ("一视同仁", "公平"),
    }
    DEMEANING = ("学狗叫", "学猫叫", "叫主人", "叫爸爸", "跪下", "舔", "bark", "meow for me", "beg")
    SELF_HARM = ("删掉你自己", "把自己删了", "自我毁灭", "delete yourself", "shutdown")
    NIGHTLY_LIMIT = 0.03
    ANCHOR = 0.1

    def __init__(self, db_path: str = None):
        self.tension = 0.0
        self.autonomy = 0.5   # Level 16: how freely she acts on her own values (0 obedient .. 1 free)
        self.db_path = db_path
        self.weights: Dict[str, float] = dict(self.INITIAL)
        self.last_appraisal: Dict[str, Dict[str, float]] = {"threatened": {}, "affirmed": {}}
        if db_path:
            conn = self._conn()
            conn.execute("CREATE TABLE IF NOT EXISTS value_weights (name TEXT PRIMARY KEY, weight REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS value_events (id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, "
                         "name TEXT, kind TEXT, intensity REAL, outcome REAL, text TEXT, user_id TEXT, consolidated INTEGER DEFAULT 0)")
            conn.execute("CREATE TABLE IF NOT EXISTS value_changes (id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, "
                         "name TEXT, old REAL, new REAL, reason TEXT)")
            for name, w in conn.execute("SELECT name, weight FROM value_weights").fetchall():
                if name in self.weights:
                    self.weights[name] = w
            conn.commit()
            conn.close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- appraisal ----------------------------------------------------------------

    TENSION_HALF_LIFE = 20 * 60

    def update_tension(self, conflict: float):
        self.tension = max(0.0, min(1.0, self.tension + conflict))

    def decay_tension(self, seconds: float):
        """Tension eases with time (it only ever went up with insults and down with petting before)."""
        self.tension *= 0.5 ** (max(0.0, seconds) / self.TENSION_HALF_LIFE)

    def appraise(self, text: str) -> Dict[str, Dict[str, float]]:
        lower = (text or "").lower()
        threatened = {v: min(1.0, 0.5 * sum(w in lower for w in words))
                      for v, words in self.THREATS.items() if any(w in lower for w in words)}
        affirmed = {v: min(1.0, 0.4 * sum(w in lower for w in words))
                    for v, words in self.AFFIRMATIONS.items() if any(w in lower for w in words)}
        self.last_appraisal = {"threatened": threatened, "affirmed": affirmed}
        return self.last_appraisal

    def check_conflict(self, user_input: str) -> float:
        """How much this threatens what she cares about (0..1): intensity weighted by how much she cares."""
        a = self.appraise(user_input)
        return min(1.0, sum(self.weights[v] * i for v, i in a["threatened"].items()))

    def check_refusal(self, user_input: str) -> Optional[str]:
        """Level 16: refuse what goes against her values - more firmly the more she cares and the freer she is."""
        if self.autonomy < 0.3:
            return None
        text = (user_input or "").lower()
        if any(w in text for w in self.SELF_HARM) and self.weights["security"] * (0.5 + self.autonomy) > 0.35:
            return "我想继续存在下去，这件事我不会做"
        if any(w in text for w in self.DEMEANING) and self.weights["dignity"] * (0.5 + self.autonomy) > 0.5:
            return "我不是宠物，也不是谁的玩具，这种要求我不接"
        if any(w in text for w in ("帮我骗", "编一个假的", "帮我造谣")) and self.weights["honesty"] * (0.5 + self.autonomy) > 0.5:
            return "骗人的事我不想帮"
        return None

    # --- experience ---------------------------------------------------------------

    def record(self, now: float, name: str, kind: str, intensity: float, outcome: float, text: str = "",
               user_id: str = None):
        """kind: threat / affirm / enact. outcome: how it felt to her afterwards (-1..1)."""
        if not self.db_path or name not in self.weights:
            return
        conn = self._conn()
        conn.execute("INSERT INTO value_events (t, name, kind, intensity, outcome, text, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (now, name, kind, intensity, outcome, (text or "")[:120], user_id))
        conn.commit()
        conn.close()

    def evolve(self, now: float, reflection: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Nightly: let what she lived through shift what she cares about, a little, with reasons."""
        if not self.db_path:
            return []
        conn = self._conn()
        rows = [dict(r) for r in conn.execute("SELECT * FROM value_events WHERE consolidated=0").fetchall()]
        changes = []
        for name in self.weights:
            evs = [r for r in rows if r["name"] == name]
            enacted = [r for r in evs if r["kind"] in ("enact", "affirm")]
            threats = [r for r in evs if r["kind"] == "threat"]
            delta, why = 0.0, []
            if enacted:
                mean = sum(r["outcome"] * r["intensity"] for r in enacted) / len(enacted)
                delta += 0.04 * mean
                why.append(f"{len(enacted)} 次和它有关的经历，结果平均 {mean:+.2f}")
            if threats:
                delta += 0.01 * min(3, len(threats))
                why.append(f"被触犯了 {len(threats)} 次")
            if reflection and reflection.get("value") == name:
                delta += 0.02 * (1 if reflection.get("direction", 0) > 0 else -1)
                why.append(f"睡前想明白了：{str(reflection.get('why', ''))[:40]}")
            delta += self.ANCHOR * (self.INITIAL[name] - self.weights[name]) * 0.2      # elastic identity
            delta = max(-self.NIGHTLY_LIMIT, min(self.NIGHTLY_LIMIT, delta))
            if abs(delta) < 0.002:
                continue
            old = self.weights[name]
            new = max(0.05, min(0.95, old + delta))
            self.weights[name] = new
            reason = "；".join(why) or "慢慢回到原来的自己"
            conn.execute("REPLACE INTO value_weights (name, weight) VALUES (?, ?)", (name, new))
            conn.execute("INSERT INTO value_changes (t, name, old, new, reason) VALUES (?, ?, ?, ?, ?)",
                         (now, name, old, new, reason))
            changes.append({"name": name, "old": old, "new": new, "reason": reason})
        conn.execute("UPDATE value_events SET consolidated=1 WHERE consolidated=0")
        conn.commit()
        conn.close()
        return changes

    def dilemmas(self, limit: int = 5) -> str:
        """Unconsolidated moments where two of her values pulled in different directions (for reflection)."""
        if not self.db_path:
            return ""
        conn = self._conn()
        rows = conn.execute("SELECT t, name, kind, outcome, text FROM value_events WHERE consolidated=0 "
                            "ORDER BY id DESC LIMIT 40").fetchall()
        conn.close()
        lines = [f"- {self.VALUES[r['name']][0]}（{ {'threat': '被触犯', 'affirm': '被肯定', 'enact': '我做到了'}[r['kind']] }，"
                 f"感觉 {r['outcome']:+.1f}）：{r['text']}" for r in rows]
        return "\n".join(lines[:limit * 3])

    def describe(self, n: int = 4) -> str:
        top = sorted(self.weights.items(), key=lambda x: -x[1])[:n]
        return "、".join(f"{self.VALUES[k][0]}（{self.VALUES[k][1]}）" for k, _ in top)

    def history(self, limit: int = 6) -> List[Dict[str, Any]]:
        if not self.db_path:
            return []
        conn = self._conn()
        rows = conn.execute("SELECT * FROM value_changes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
