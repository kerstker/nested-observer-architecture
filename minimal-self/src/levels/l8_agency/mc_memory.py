"""
Level 8m: what her hands have done, and what she learned from it.

A script forgets; a player remembers. Two kinds of memory for her game body:
  - the last things her hands did, step by step with how it went - so her mind knows what "she" just did;
  - experience: what worked and what did not ("y=-58 has almost no iron", "in a narrow tunnel a table goes
    against the wall"), kept across sessions, counted, and brought back when it is relevant again.
Lessons come from what happened (a step that failed and why, a step that worked after failing, a goal done)
and from her own words (mc_remember).
"""
import json
import logging
import os
import re
import time
import hashlib
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


# arguments that size an action rather than say what it is (see Experience.action_key)
VOLATILE = {"mine": ("count", "radius"), "tunnel": ("length", "branch", "spacing", "torch_every", "y"),
            "explore_cave": ("max_seconds", "max_nodes"), "chop_tree": ("max_logs", "radius"),
            "collect_drops": ("radius",), "hunt": ("count",)}


class Experience:
    def __init__(self, path: Optional[str] = None, keep: int = 300):
        self.path = path
        self.keep = keep
        self.on_learn = None
        self.lessons: List[Dict[str, Any]] = []
        self.recent: Deque[Tuple[float, str]] = deque(maxlen=12)
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    self.lessons = json.load(f)
            except (OSError, ValueError):
                pass

    # what the hands just did ---------------------------------------------------------------------
    def did(self, text: str):
        self.recent.append((time.time(), text))

    def last(self, n: int = 5) -> List[str]:
        now = time.time()
        return [f"{int((now - t) // 60)}分钟前 {x}" if now - t >= 60 else f"刚才 {x}" for t, x in list(self.recent)[-n:]]

    # what she learned -----------------------------------------------------------------------------
    def learn(self, text: str, about: str = "", source: str = "body") -> None:
        text = str(text).strip()[:160]
        if not text:
            return
        for l in self.lessons:
            if l["text"] == text:
                l["n"] += 1
                l["t"] = time.time()
                break
        else:
            self.lessons.append({"text": text, "about": about, "source": source, "n": 1, "t": time.time()})
        # her own notes and what was confirmed often stay; one-off body notes go first
        self.lessons.sort(key=lambda l: (l["source"] == "self", l["n"], l["t"]), reverse=True)
        del self.lessons[self.keep:]
        self._save()
        if self.on_learn:
            try:
                self.on_learn(text, about, source)
            except Exception:
                logging.getLogger(__name__).warning("[Game] lesson observer failed", exc_info=True)

    def relevant(self, words: str, n: int = 4) -> List[str]:
        keys = [w for w in re.split(r"[\s，。、：:（）()×,]+", words or "") if len(w) >= 2]
        scored = []
        for l in self.lessons:
            hay = l["text"] + " " + l.get("about", "")
            s = sum(2 for k in keys if k in hay)
            if s:
                s += 1 if l["source"] == "self" else 0          # her own words count a little more
                scored.append((s, l["n"], l["t"], l["text"]))
        return [t for *_, t in sorted(scored, reverse=True)[:n]]

    @staticmethod
    def action_key(name, args):
        # What the action is, not how much of it: the planner sizes mine/tunnel by what it saw this time, and the
        # skill call drops planning-only fields, so a key over every argument never matched its own last failure.
        args = {k: v for k, v in (args or {}).items() if k not in VOLATILE.get(name, ())}
        encoded = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode()).hexdigest()[:20]

    def feedback(self, name, args, success, result, pos=None, dim=None, now=None, status=None):
        """Measured outcomes teach retries, scoped to the action and local place/dimension."""
        now = time.time() if now is None else now
        key = self.action_key(name, args)
        zone = [int(float(v) // 24) for v in pos] if pos else None
        old = next((x for x in self.lessons if x.get("action_key") == key and x.get("zone") == zone
                    and x.get("dim") == dim), None)
        failures = 0 if success else int((old or {}).get("failures", 0)) + 1
        row = {"text": f"{name}：{str(result)[:110]}", "about": name, "source": "body", "n": (old or {}).get("n", 0) + 1,
               "t": now, "action_key": key, "zone": zone, "dim": dim, "failures": failures,
               "retry_after": now + min(600, 30 * 2 ** min(failures - 1, 5)) if failures else 0}
        if status in ("completed", "partial", "failed", "interrupted", "unverified"):
            row.update(status=status, completed=status == "completed")
        if old:
            self.lessons.remove(old)
        self.lessons.insert(0, row)
        del self.lessons[self.keep:]
        self._save()
        return row

    def advice(self, name, args, pos=None, dim=None, now=None):
        now = time.time() if now is None else now
        zone = [int(float(v) // 24) for v in pos] if pos else None
        key = self.action_key(name, args)
        return next((dict(x) for x in self.lessons if x.get("action_key") == key and x.get("zone") == zone
                     and x.get("dim") == dim and x.get("retry_after", 0) > now), None)

    def recent_failure(self, name, pos=None, dim=None, now=None):
        """Any backed-off failure of this action here, whatever its arguments (the model only names the skill)."""
        now = time.time() if now is None else now
        zone = [int(float(v) // 24) for v in pos] if pos else None
        return next((dict(x) for x in self.lessons if x.get("action_key") and x.get("about") == name
                     and x.get("zone") == zone and x.get("dim") == dim and x.get("retry_after", 0) > now), None)

    def _save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.lessons, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass
