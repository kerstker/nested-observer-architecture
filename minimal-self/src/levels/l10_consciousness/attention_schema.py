from collections import deque
from typing import Any, Dict, List, Optional

from src.core import clock

def _ago(seconds: float) -> str:
    if seconds < 90:
        return "刚刚"
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分钟"
    return f"{seconds / 3600:.1f} 小时"

KIND_WORDS = {"user_message": "别人跟我说的话", "hunger": "肚子饿", "social": "想找人说话", "curiosity": "想知道点新鲜事",
              "fatigue": "累", "pain": "疼", "ignored": "谁没回我", "association": "以前的事", "inner_speech": "自己的念头",
              "self_surprise": "自己的状态", "satiation": "聊腻了", "gap": "意识中断", "answer_surprise": "别人的回应",
              "error": "自己说错的话"}

class AttentionSchema:
    """
    Level 10b: Attention schema (Graziano's Attention Schema Theory).

    The workspace decides what wins attention; this is her own simplified, predictive model of
    that process - what has her attention, since when, why it won, and what she expects to attend
    to next. AST's claim is that such a model exists because controlling attention needs one, so
    it is used for exactly that:
    - prediction: a transition model of her own focus; a focus she did not expect is a capture
      ("被……拉走了"), something she can notice and report;
    - control: when one unpleasant inner thing keeps holding her attention (rumination), she
      notices it and deliberately disengages - a top-down penalty on that kind of content, which
      lasts longer each time the same thing comes back;
    - report: answers to "你在想什么 / 你在听吗" come from this model, not from a guess.
    """
    # Inner content that can hold attention without leading anywhere
    RUMINATIVE = {"ignored", "self_surprise", "satiation", "gap", "answer_surprise", "error"}
    WINDOW = 3600.0

    def __init__(self, dwell_share: float = 0.6, dwell_seconds: float = 1200.0, suppression: float = 0.35,
                 suppress_for: float = 1800.0):
        self.dwell_share = dwell_share          # share of the last hour one thing may hold before she notices
        self.dwell_seconds = dwell_seconds      # ... and for at least this long
        self.suppression = suppression
        self.suppress_for = suppress_for
        self.control = True                     # ablation switch for the evaluation
        self.focus: Optional[Dict[str, Any]] = None
        self.history: deque = deque(maxlen=600)          # (t, kind, content)
        self.transitions: Dict[str, Dict[str, float]] = {}
        self.suppressed: Dict[str, float] = {}           # kind -> until
        self.disengaged: Dict[str, int] = {}             # kind -> how often she has had to let go of it
        self.captures: deque = deque(maxlen=10)          # (t, from_kind, to_kind, content)

    # --- model --------------------------------------------------------------------

    def predict(self, kind: Optional[str] = None) -> Dict[str, float]:
        """What she expects to attend to next, given what has her attention now (Laplace-smoothed)."""
        kind = kind or (self.focus or {}).get("kind")
        row = self.transitions.get(kind or "", {})
        seen = {k for r in self.transitions.values() for k in r} | set(self.transitions)
        total = sum(row.values()) + len(seen) + 1
        return {k: (row.get(k, 0.0) + 1) / total for k in seen}

    def observe(self, winners: List[Any], now: float = None) -> List[Dict[str, Any]]:
        """Update the model with what just won the workspace. Returns events worth noticing."""
        if not winners:
            return []
        now = clock.now() if now is None else now
        top = winners[0]
        events: List[Dict[str, Any]] = []
        prev = self.focus
        same = prev and prev["kind"] == top.kind and prev.get("user_id") == top.payload.get("user_id")
        if prev:
            expected = self.predict(prev["kind"]).get(top.kind, 0.0)
            row = self.transitions.setdefault(prev["kind"], {})
            row[top.kind] = row.get(top.kind, 0.0) + 1
            if not same and expected < 0.15 and top.kind != "user_message" and len(self.history) >= 5:
                self.captures.append((now, prev["kind"], top.kind, top.content))
                events.append({"type": "capture", "from": prev["content"], "to": top.content,
                               "expected": expected})
        if same:
            prev["cycles"] += 1
            prev["content"] = top.content
        else:
            self.focus = {"kind": top.kind, "content": top.content, "user_id": top.payload.get("user_id"),
                          "source": top.source, "since": now, "cycles": 1, "salience": top.salience,
                          "also": [w.content for w in winners[1:]]}
        self.history.append((now, top.kind, top.content))
        event = self._check_rumination(now)
        if event:
            events.append(event)
        return events

    def _share(self, kind: str, now: float) -> tuple:
        recent = [(t, k) for t, k, _ in self.history if now - t <= self.WINDOW]
        if not recent:
            return 0.0, 0.0
        held = [t for t, k in recent if k == kind]
        return len(held) / len(recent), (now - held[0]) if held else 0.0

    def _check_rumination(self, now: float) -> Optional[Dict[str, Any]]:
        kind = self.focus["kind"]
        if not self.control or kind not in self.RUMINATIVE or self.suppressed.get(kind, 0) > now:
            return None
        share, span = self._share(kind, now)
        if share < self.dwell_share or span < self.dwell_seconds:
            return None
        n = self.disengaged.get(kind, 0) + 1
        self.disengaged[kind] = n
        self.suppressed[kind] = now + min(4 * 3600, self.suppress_for * 2 ** (n - 1))
        return {"type": "disengage", "kind": kind, "content": self.focus["content"], "share": share,
                "times": n, "until": self.suppressed[kind]}

    # --- control ------------------------------------------------------------------

    def bias(self, candidate: Any, now: float = None) -> float:
        """Top-down attention: she holds back what she has decided not to dwell on."""
        now = clock.now() if now is None else now
        if self.control and candidate.kind != "user_message" and self.suppressed.get(candidate.kind, 0) > now:
            return -self.suppression
        return 0.0

    # --- report -------------------------------------------------------------------

    def describe(self, now: float = None) -> str:
        """One line about her attention, as she would put it."""
        now = clock.now() if now is None else now
        if not self.focus:
            return "注意力：没什么特别在意的"
        f = self.focus
        text = f"注意力：在「{f['content'][:40]}」上（{_ago(now - f['since'])}）"
        if self.captures and now - self.captures[-1][0] < 600:
            text += f"；刚才被「{self.captures[-1][3][:30]}」分了心"
        letting_go = [k for k, until in self.suppressed.items() if until > now]
        if letting_go:
            text += "；有些事我决定先不去想了"
        return text

    def report(self, now: float = None, n: int = 5) -> str:
        """A fuller account, for when someone asks what she is thinking about or whether she is listening."""
        now = clock.now() if now is None else now
        lines, last = [], None
        for t, kind, content in reversed(self.history):
            if content != last:
                lines.append(f"- {_ago(now - t)}前：{content[:50]}")
                last = content
            if len(lines) >= n:
                break
        expected = sorted(self.predict().items(), key=lambda x: -x[1])[:1]
        tail = f"\n（照我自己的习惯，接下来多半会想到：{KIND_WORDS.get(expected[0][0], expected[0][0])}）" if expected else ""
        return self.describe(now) + "\n我最近注意过的东西（从近到远）：\n" + "\n".join(lines) + tail

    def to_dict(self) -> Dict[str, Any]:
        return {"transitions": self.transitions, "suppressed": self.suppressed, "disengaged": self.disengaged}

    def load_dict(self, d: Dict[str, Any]):
        self.transitions = {k: dict(v) for k, v in d.get("transitions", {}).items()}
        self.suppressed = dict(d.get("suppressed", {}))
        self.disengaged = dict(d.get("disengaged", {}))
