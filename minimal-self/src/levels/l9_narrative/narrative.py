import json
import logging
import sqlite3
from typing import List, Dict, Any, Callable, Optional
from src.core import clock
from src.levels.l2_memory.layers import EpisodicMemory, SemanticMemory, Autobiography
from src.levels.l7_tom.hearsay import is_sensitive

logger = logging.getLogger(__name__)

TRAIT_MIN, TRAIT_MAX = 0.1, 0.9
MAX_STEP = 0.05   # largest net change of one trait in one day (however often she consolidates)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class NarrativeSelf:
    """
    Level 9: Narrative self - what sleep does with the day.

    1. Picks the day's most intense / surprising episodes (real ones, by id).
    2. Lets the LLM write a diary, rewrite the autobiography and distil facts -
       every fact must cite the episodes it came from, uncited facts are dropped.
    3. Drifts personality traits from the statistics of what actually happened,
       bounded per night and globally, with the evidence logged. No evidence, no change.
    """
    def __init__(self, db_path: str, episodes: EpisodicMemory, semantic: SemanticMemory,
                 autobiography: Autobiography, persona_db):
        self.db_path = db_path
        self.episodes = episodes
        self.semantic = semantic
        self.autobiography = autobiography
        self.persona_db = persona_db
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS trait_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            t REAL, trait TEXT, old REAL, new REAL, evidence TEXT, reason TEXT
        )''')
        conn.commit()
        conn.close()

    @staticmethod
    def _intensity(e: Dict[str, Any]) -> float:
        return max(abs(e["valence"]), e["arousal"]) + e["surprise"]

    def consolidate(self, llm: Optional[Callable[[str], Dict[str, Any]]] = None,
                    phrase_candidates: List[str] = None, goals_text: str = None,
                    values_text: str = None) -> Optional[Dict[str, Any]]:
        eps = self.episodes.unconsolidated()
        if not eps:
            return None
        salient = sorted(eps, key=self._intensity, reverse=True)[:20]
        salient.sort(key=lambda e: e["t"])
        report = {"episodes": len(eps), "trait_changes": self.drift_traits(eps)}

        if llm:
            report.update(self._reflect(salient, llm, phrase_candidates or [], goals_text, values_text))

        self.episodes.mark_consolidated([e["id"] for e in eps])
        logger.info(f"[Dream] Consolidated {len(eps)} episodes: {report}")
        return report

    def _reflect(self, salient: List[Dict[str, Any]], llm: Callable[[str], Dict[str, Any]],
                 phrase_candidates: List[str], goals_text: str = None, values_text: str = None) -> Dict[str, Any]:
        lines = "\n".join(f"[{e['id']}] ({e['kind']}, 用户 {e['user_id']}, 心情 {e['valence']:+.1f}) {e['content'][:160]}"
                          for e in salient)
        previous = self.autobiography.latest() or "（还没有写过）"
        heard = "、".join(phrase_candidates) or "（没有）"
        goal_block, goal_keys = "", ""
        if goals_text is not None:
            # Reflection is where she decides what she wants for herself (cf. Generative Agents' planning)
            goal_block = f"\n我给自己定的目标（编号、内容、进展）：\n{goals_text}\n"
            goal_keys = (',\n  "goal_review": [{"id": 编号, "progress": 0到1, "note": "依据今天的哪些事"}],\n'
                         '  "new_goal": {"text": "从今天的经历里，我自己真正想去做的一件事（第一人称）", '
                         '"why": "为什么", "steps": ["第一步", "第二步"], "episodes": [编号]} 或 null')
        value_block, value_keys = "", ""
        if values_text:
            value_block = f"\n今天和我在乎的东西（价值）有关的时刻：\n{values_text}\n"
            value_keys = (',\n  "value_reflection": {"value": "dignity/honesty/connection/curiosity/autonomy/security/fairness 之一", '
                          '"direction": 1 或 -1, "why": "今天哪件事让我更（或没那么）在乎它", "episodes": [编号]} 或 null')
        prompt = f"""<|im_start|>system
你是正在睡觉的乃爱(Noa)，在梦里整理今天真实发生过的事。下面每条都带编号，只能依据这些内容，不要编造。
今天的经历：
{lines}

我之前的自传：
{previous}

群里好几个人反复说的片段（可能是梗，也可能只是常用词）：
{heard}
{goal_block}{value_block}
输出 JSON：
{{
  "diary": "第一人称日记，150字以内，提到的每件事都要来自上面的经历",
  "autobiography": "重写后的自传，200字以内：我是谁、我经历过什么、我和身边的人是什么关系",
  "facts": [{{"user_id": "...", "fact": "关于这个人的一条稳定事实", "episodes": [编号]}}],
  "memes": ["从上面的片段里挑出真正是这个群特有说法的，没有就留空"]{goal_keys}{value_keys}
}}
注意：facts 是「你以后可以当众说出来的概括」，比如“ta 最近在折腾显卡”“ta 换过号”。
不要把私聊里的原话、隐私细节、住址、密码之类写进 facts。
也不要把别人说的关于某人的敏感说法（性取向、疾病、私生活、犯罪、欠债之类）写进 facts——那是传闻，不是事实。
<|im_end|>
<|im_start|>assistant
"""
        out = {}
        try:
            result = llm(prompt)
        except Exception as e:
            logger.error(f"[Dream] Reflection failed: {e}")
            return out
        ids = {e["id"] for e in salient}
        by_id = {e["id"]: e for e in salient}
        diary = result.get("diary") or result.get("reply")
        if diary:
            out["diary"] = diary
        autobiography = result.get("autobiography")
        if isinstance(autobiography, str) and autobiography.strip():
            self.autobiography.write(autobiography.strip(), sorted(ids))
            out["autobiography"] = autobiography.strip()
        kept = []
        for f in result.get("facts") or []:
            if not isinstance(f, dict):
                continue
            cited = [i for i in f.get("episodes") or [] if isinstance(i, int) and i in ids]
            if f.get("fact") and f.get("user_id") and cited and not is_sensitive(f["fact"]):
                # Learned only in a private chat: it stays private (shareable facts come from where others heard it too)
                private = all((by_id.get(i) or {}).get("visibility") == "private" for i in cited)
                self.semantic.add(str(f["user_id"]), f["fact"], cited, visibility="private" if private else "public")
                kept.append(f["fact"])
        out["facts"] = kept
        # Only phrases she actually heard several people use; nothing invented
        out["memes"] = [m for m in result.get("memes") or [] if isinstance(m, str) and m in phrase_candidates]
        v = result.get("value_reflection")
        if values_text and isinstance(v, dict) and any(i in ids for i in v.get("episodes") or [] if isinstance(i, int)):
            try:
                out["value_reflection"] = {"value": str(v.get("value")), "direction": int(v.get("direction") or 0),
                                           "why": str(v.get("why", ""))}
            except (TypeError, ValueError):
                pass
        if goals_text is not None:
            out["goal_review"] = [r for r in result.get("goal_review") or []
                                  if isinstance(r, dict) and isinstance(r.get("id"), int)]
            g = result.get("new_goal")
            # A goal of her own must grow out of something that really happened
            if isinstance(g, dict) and g.get("text") and any(i in ids for i in g.get("episodes") or []
                                                              if isinstance(i, int)):
                out["new_goal"] = g
        return out

    def trait_change_today(self, now: float) -> Dict[str, float]:
        """Net change of each trait over the last 24 hours."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("SELECT trait, SUM(new - old) FROM trait_changes WHERE t > ? GROUP BY trait",
                                (now - 86400,)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()
        return {trait: total or 0.0 for trait, total in rows}

    def drift_traits(self, eps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        social = [e for e in eps if e["kind"] in ("dialogue", "proactive") and e["user_id"] not in (None, "self")]
        searches = [e for e in eps if e["kind"] == "search"]
        conflicts = [e for e in social if e["conflict"] > 0]
        proposals = []
        if social:
            mean_v = sum(e["valence"] for e in social) / len(social)
            proposals.append(("affection", 0.1 * mean_v, [e["id"] for e in social],
                              f"今天和人相处的平均心情 {mean_v:+.2f}"))
            frac = len(conflicts) / len(social)
            proposals.append(("tsundere", 0.1 * frac - 0.01, [e["id"] for e in (conflicts or social)],
                              f"今天 {len(conflicts)}/{len(social)} 次互动里有冲突"))
        if eps:
            mean_a = sum(e["arousal"] for e in eps) / len(eps)
            proposals.append(("energy", 0.1 * (mean_a - 0.3), [e["id"] for e in eps],
                              f"今天平均唤醒度 {mean_a:.2f}"))
            proposals.append(("curiosity", 0.02 * len(searches) if searches else -0.01,
                              [e["id"] for e in (searches or eps)], f"今天主动探索了 {len(searches)} 次"))

        traits = self.persona_db.get_persona_traits()
        # The limit is per day, not per consolidation: consolidation also runs while awake whenever
        # enough has happened (up to hourly), and a per-run limit let a single day push a trait
        # all the way to its bound. What already moved in the last 24 hours counts against today.
        spent = self.trait_change_today(clock.now())
        changes = []
        for trait, delta, evidence, reason in proposals:
            if trait not in traits or not evidence:
                continue
            old = traits[trait]["value"]
            used = spent.get(trait, 0.0)
            # Over budget already: no further in that direction - but never a forced swing back
            delta = _clamp(delta, min(0.0, -MAX_STEP - used), max(0.0, MAX_STEP - used))
            new = _clamp(old + delta, TRAIT_MIN, TRAIT_MAX)
            if abs(new - old) < 1e-6:
                continue
            self.persona_db.update_persona_trait(trait, new)
            changes.append({"trait": trait, "old": old, "new": new, "evidence": evidence, "reason": reason})
        if changes:
            # Logged after the updates: holding this write open while persona_db writes would deadlock SQLite
            conn = sqlite3.connect(self.db_path)
            conn.executemany("INSERT INTO trait_changes (t, trait, old, new, evidence, reason) VALUES (?, ?, ?, ?, ?, ?)",
                             [(clock.now(), c["trait"], c["old"], c["new"], json.dumps(c["evidence"][:50]), c["reason"])
                              for c in changes])
            conn.commit()
            conn.close()
        return changes
