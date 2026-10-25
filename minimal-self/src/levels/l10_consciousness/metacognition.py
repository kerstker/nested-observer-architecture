import re
from typing import Any, Dict, List, Optional

class Metacognition:
    """
    Level 10c: Metacognition - knowing how far to trust herself.

    - Prospective: before she answers, a feeling of knowing from her own signals (how well the
      question matches what she remembers, whether a search worked), combined with how sure the
      model says it is. Low confidence on memory or facts means: say so, do not make things up.
    - Retrospective (error monitoring): when the next message corrects or confirms what she said,
      that is the outcome of her confidence. A confident mistake is more surprising than an unsure one.
    - Calibration: per domain she tracks mean confidence, accuracy and Brier score. Systematic
      overconfidence is noticed and corrected (her confidence is shifted by the learned bias), and
      it changes how she speaks ("留点余地") - metacognitive control, not just a number.
    """
    DOMAINS = ("memory", "knowledge", "chat")
    DOMAIN_WORDS = {"memory": "记事情", "knowledge": "说事实", "chat": "聊天"}
    MEMORY_CUES = ("记得", "记不记得", "还记", "上次", "之前", "以前", "我说过", "跟你说过", "我叫什么", "我是谁",
                   "我喜欢什么", "我住", "我的生日", "我多大")
    KNOWLEDGE_CUES = ("是什么", "是谁", "多少", "哪一年", "什么时候", "为什么会", "怎么回事", "天气", "价格", "新闻")
    CORRECTIONS = ("不对", "错了", "记错", "不是这样", "不是的", "瞎说", "乱说", "胡说", "搞错", "记岔", "没有啊",
                   "不是啊", "才不是", "并不是", "你说反了", "不是我")
    CONFIRMS = ("对对", "没错", "是的", "对啊", "对呀", "对的", "确实", "说得对", "答对", "就是这样", "记性真好")
    SURE_QUESTIONS = ("确定吗", "有把握", "真的假的", "你肯定", "你确定", "靠谱吗", "真的吗")
    RATE = 0.15
    WINDOW = 1800.0

    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = {d: {"conf": 0.7, "acc": 0.7, "brier": 0.1, "n": 0}
                                                   for d in self.DOMAINS}
        self.pending: Dict[str, Dict[str, Any]] = {}       # user_id -> what she said and how sure she was
        self.last: Optional[Dict[str, Any]] = None

    # --- before answering ---------------------------------------------------------

    def domain_of(self, text: str, searched: bool = False) -> str:
        if any(k in text for k in self.MEMORY_CUES):
            return "memory"
        if searched or any(k in text for k in self.KNOWLEDGE_CUES):
            return "knowledge"
        return "chat"

    @staticmethod
    def feeling_of_knowing(domain: str, recall_similarity: float = 0.0, facts: int = 0,
                           search_ok: Optional[bool] = None) -> float:
        if domain == "memory":
            return min(0.95, 0.2 + 0.7 * recall_similarity + (0.15 if facts else 0.0))
        if domain == "knowledge":
            return 0.85 if search_ok else 0.2 if search_ok is False else 0.5
        return 0.75

    def bias(self, domain: str) -> float:
        s = self.stats[domain]
        return s["conf"] - s["acc"] if s["n"] >= 3 else 0.0

    def confidence(self, domain: str, fok: float, verbal: Optional[float] = None) -> Dict[str, float]:
        raw = fok if verbal is None else 0.6 * verbal + 0.4 * fok
        return {"raw": raw, "calibrated": min(0.98, max(0.02, raw - self.bias(domain)))}

    def guidance(self, domain: str, fok: float) -> str:
        """What her metacognition tells her before she speaks (empty when there is nothing to say)."""
        lines: List[str] = []
        sure = min(0.98, max(0.02, fok - self.bias(domain)))
        if domain != "chat" and sure < 0.45:
            what = "这件事我记得不太清" if domain == "memory" else "这个我其实不太确定"
            lines.append(f"{what}（把握大约 {sure:.0%}）：不确定就直说，不要编。")
        b = self.bias(domain)
        word = self.DOMAIN_WORDS[domain]
        if b > 0.15:
            lines.append(f"我最近在{word}上好几次自以为对、其实错了（自信 {self.stats[domain]['conf']:.0%}，"
                         f"实际对了 {self.stats[domain]['acc']:.0%}），说之前多想一下，留点余地。")
        elif b < -0.15:
            lines.append(f"我在{word}上其实多数时候是对的，不用太犹豫。")
        return "\n".join(f"- {l}" for l in lines)

    # --- after answering ----------------------------------------------------------

    def expect(self, user_id: str, said: str, domain: str, confidence: Dict[str, float], now: float):
        self.pending[user_id] = {"said": said, "domain": domain, "t": now, **confidence}
        self.last = self.pending[user_id]

    def judge(self, user_id: str, text: str, now: float) -> Optional[Dict[str, Any]]:
        """Did this message tell her whether she was right? Returns the outcome, or None."""
        p = self.pending.pop(user_id, None)
        if not p or now - p["t"] > self.WINDOW:
            return None
        head = re.sub(r"\s+", "", text)[:20]
        if any(k in head for k in self.CORRECTIONS):
            correct = 0.0
        elif any(k in head for k in self.CONFIRMS):
            correct = 1.0
        else:
            return None              # no clear verdict: learn nothing rather than guess
        s, r = self.stats[p["domain"]], self.RATE
        c = p["raw"]
        s["conf"] += r * (c - s["conf"])
        # ...and how sure she still was after her own correction: what calibration has not fixed yet
        s["cal"] = s.get("cal", s["conf"]) + r * (p["calibrated"] - s.get("cal", s["conf"]))
        s["acc"] += r * (correct - s["acc"])
        s["brier"] += r * ((c - correct) ** 2 - s["brier"])
        s["n"] += 1
        sure = p["calibrated"]
        return {"error": correct == 0.0, "said": p["said"], "domain": p["domain"], "confidence": sure,
                "surprise": sure if correct == 0.0 else 1 - sure}

    def asked_if_sure(self, text: str) -> bool:
        return any(k in text for k in self.SURE_QUESTIONS)

    def describe(self) -> str:
        parts = []
        for d in self.DOMAINS:
            s = self.stats[d]
            if s["n"]:
                parts.append(f"{self.DOMAIN_WORDS[d]}：自信 {s['conf']:.0%} / 实际 {s['acc']:.0%}（{int(s['n'])} 次）")
        return "；".join(parts) or "还没有足够的经验知道自己有多靠谱"

    def to_dict(self) -> Dict[str, Any]:
        return {"stats": self.stats}

    def load_dict(self, d: Dict[str, Any]):
        for k, v in d.get("stats", {}).items():
            if k in self.stats:
                self.stats[k].update(v)
