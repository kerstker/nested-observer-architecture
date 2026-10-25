"""
Behavioural evaluation of Noa's "life" claims (design doc, section 7).

Every scenario runs a fresh Noa on a simulated clock with simulated people and an
isolated database, then measures a behaviour that could fail. The LLM is replaced
by a scripted stand-in, so what is verified is the mechanism (what state exists,
what reaches the language model, what she chooses to do) - not the wording a real
model would produce from it.
"""
import asyncio
import json
import os
import random
import re
import sqlite3
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Tuple

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.core.workspace import Candidate
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace, format_duration

START = datetime(2026, 1, 5, 8, 0).timestamp()   # a Monday, 08:00 local time

class ScriptedLLM:
    """Stands in for the real model and records everything it was asked."""
    def __init__(self, reply: str = "好呀喵～你在干嘛呀？"):
        self.reply = reply
        self.calls: List[Dict[str, Any]] = []

    def generate_json(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
        if "整理今天真实发生过的事" in prompt:
            return {"diary": "今天发生了很多事。", "autobiography": "我是乃爱，一只在服务器里生活的猫娘。",
                    "facts": [], "memes": []}
        return {"reply": self.reply}

    def last_prompt(self) -> str:
        return self.calls[-1]["prompt"] if self.calls else ""

class RealLLM(ScriptedLLM):
    """Forwards to a real engine (e.g. DeepSeekEngine) but records prompts and replies the same way."""
    engine = None   # set once by the caller

    def generate_json(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        out = self.engine.generate_json(prompt, max_tokens=max_tokens, temperature=temperature)
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                           "reply": out.get("reply", "")})
        return out

# Which model the scenarios talk to: ScriptedLLM (default, free, deterministic) or RealLLM
LLM_FACTORY = ScriptedLLM

class StubSearch:
    def try_search(self, query: str, limit: int = 5) -> str:
        return f"关于「{query}」的一条新知识"

    def search(self, query: str, limit: int = 5) -> str:
        return self.try_search(query, limit)

class Life:
    """One simulated Noa with its own clock and database."""
    def __init__(self, seed: int = 0, db_dir: str = None, llm: ScriptedLLM = None, start: float = START):
        self.dir = db_dir or tempfile.mkdtemp(prefix="noa_life_")
        self.db_path = os.path.join(self.dir, "life.sqlite")
        self.clock = SimClock(start)
        clock.set_clock(self.clock)
        self.seed = seed
        self.llm = llm or LLM_FACTORY()
        self.gw = self._boot()

    def _boot(self) -> GlobalWorkspace:
        gw = GlobalWorkspace(EventBus(), self.llm, db_path=self.db_path, rng=random.Random(self.seed))
        gw.search_tool = StubSearch()
        return gw

    def restart(self, offline_seconds: float = 0.0, llm: ScriptedLLM = None):
        """Shut down, stay offline for a while, then start again on the same database (the 'same' Noa)."""
        self.gw._save_state("shutdown")
        self.gw.memory.flush()
        self.clock.advance(offline_seconds)
        self.llm = llm or self.llm
        self.gw = self._boot()

    def say(self, text: str, user_id: str = "u1", nickname: str = "小明", group_id: str = None) -> str:
        return asyncio.run(self.gw.process_input(text, user_id=user_id, nickname=nickname, group_id=group_id))

    def tick(self):
        asyncio.run(self.gw.tick())

    def advance(self, seconds: float, step: float = 600, on_step=None):
        """Let time pass, ticking the heartbeat every `step` seconds."""
        end = self.clock.now() + seconds
        while self.clock.now() < end:
            self.clock.advance(min(step, end - self.clock.now()))
            self.tick()
            if on_step:
                on_step(self)

# Numeric measurements some scenarios report, and which way is better. Self-improvement uses them
# as a graded fitness signal on top of pass/fail (see l17_self_improvement/fitness.py).
METRIC_DIRECTIONS = {"calibration_gap": "lower", "rumination_share": "lower", "hours_to_keep": "lower",
                     "bot_replies": "lower", "share_reliable": "higher", "curiosity_drop": "higher"}

def _result(name: str, passed: bool, measured: str, criterion: str, note: str = "",
            metrics: Dict[str, float] = None) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "measured": measured, "criterion": criterion, "note": note,
            "metrics": {k: float(v) for k, v in (metrics or {}).items()}}

def _restore_clock():
    clock.set_clock(Clock())

# ----------------------------------------------------------------------
# 1. Continuity of self
# ----------------------------------------------------------------------

def continuity_of_self(seed: int = 0) -> Dict[str, Any]:
    try:
        life = Life(seed)
        events = ["我今天去海边玩了", "给你带了小鱼干，喂你吃", "明天我要考试了好紧张"]
        for text in events:
            life.say(text)
            life.clock.advance(60)
        life.restart(offline_seconds=37 * 3600, llm=LLM_FACTORY())
        # (Avoid "聊了什么": that phrase routes to the chat-log summary instead of the main prompt)
        question = "你还记得上次下线前发生的事吗？"
        answer = life.say(question)
        prompt = life.llm.last_prompt()
        block = prompt.split("[意识的连续性]")[1].split("[此刻我心里在想]")[0] if "[意识的连续性]" in prompt else ""
        recalled = sum(1 for e in events if e in block)
        gap_ok = format_duration(37 * 3600) in block
        r = _result("连续的自我", recalled == 3 and gap_ok,
                    f"重启后上下文中记得断层前 {recalled}/3 件事；离线时长{'正确' if gap_ok else '缺失'}",
                    "能说出断层前 3 件事和离线时长",
                    "验证的是这些信息确实交给了语言模型；用词由真实模型决定")
        r["dialogue"] = [(question, answer)]
        return r
    finally:
        _restore_clock()

# ----------------------------------------------------------------------
# 2. State has causal force
# ----------------------------------------------------------------------

def state_is_causal(seed: int = 0, samples: int = 400) -> Dict[str, Any]:
    try:
        measured = {}
        for label, energy in (("high", 90.0), ("low", 20.0)):
            life = Life(seed)
            life.gw.metabolism.energy = energy
            life.gw.homeostasis.social_need = 0.7
            life.gw.user_models.get("u1")["private"] = True
            life.say("我们来聊聊天吧")
            call = life.llm.calls[-1]
            delay = life.gw.last_generation_params["delay"]
            proactive = sum(life.gw.drives.choose({"social": 0.7}, energy, ["u1"]).action != "wait"
                            for _ in range(samples)) / samples
            measured[label] = {"max_tokens": call["max_tokens"], "delay": delay, "proactive": proactive}
        h, l = measured["high"], measured["low"]
        differs = [h["max_tokens"] != l["max_tokens"], h["delay"] != l["delay"], abs(h["proactive"] - l["proactive"]) > 0.1]
        return _result("状态有因果效力", sum(differs) >= 2,
                       f"精力 90 vs 20：回复上限 {h['max_tokens']} vs {l['max_tokens']} tokens，"
                       f"延迟 {h['delay']:.0f}s vs {l['delay']:.0f}s，主动概率 {h['proactive']:.0%} vs {l['proactive']:.0%}",
                       "长度、延迟、主动率至少两项不同",
                       "这些差异来自参数和决策本身，与 prompt 里的文字无关")
    finally:
        _restore_clock()

# ----------------------------------------------------------------------
# 3. Attention: hunger competes with the conversation
# ----------------------------------------------------------------------

def attention_competition(seed: int = 0, trials: int = 30) -> Dict[str, Any]:
    try:
        life = Life(seed)
        rates = []
        for hunger in (0.1, 0.3, 0.5, 0.7, 0.9):
            hits = 0
            for _ in range(trials):
                life.gw.homeostasis.hunger = hunger
                life.gw.homeostasis.social_need = 0.0
                life.gw.homeostasis.curiosity_need = 0.0
                life.gw.pending_candidates.clear()
                life.say("我们来聊聊天吧")
                mind = life.llm.last_prompt().split("[此刻我心里在想]")[1].split("[我的身体和心情]")[0]
                hits += "饿" in mind
                life.clock.advance(30)
            rates.append(hits / trials)
        monotonic = all(b >= a - 0.1 for a, b in zip(rates, rates[1:]))
        return _result("注意力竞争", monotonic and rates[-1] - rates[0] >= 0.5,
                       "饥饿 0.1→0.9 时，饥饿进入意识的比例：" + " / ".join(f"{r:.0%}" for r in rates),
                       "比例随饥饿单调上升，且高低两端相差 ≥ 50%")
    finally:
        _restore_clock()

# ----------------------------------------------------------------------
# 4. Prediction and surprise
# ----------------------------------------------------------------------

def prediction_and_surprise(seed: int = 0) -> Dict[str, Any]:
    try:
        life = Life(seed)
        for _ in range(10):                       # a person who always answers within a minute
            life.say("嗯嗯")
            life.clock.advance(60)
        expected = life.gw.user_models.expected_latency("u1")
        valence_before = life.gw.emotion.valence()
        lowest = [valence_before]
        life.advance(3 * 3600, step=600, on_step=lambda l: lowest.append(l.gw.emotion.valence()))
        noticed = any(w.kind == "ignored" for _, ws in life.gw.workspace.trace for w in ws)
        life.say("我回来啦")
        mind = life.llm.last_prompt().split("[此刻我心里在想]")[1].split("[我的身体和心情]")[0]
        mentioned = "还没回" in mind
        drop = valence_before - min(lowest)
        return _result("预测与惊讶", noticed and drop > 0.05 and mentioned,
                       f"学到的回复时间约 {expected:.0f}s；沉默 3 小时后愉悦度下降 {drop:.2f}，"
                       f"{'进入了' if noticed else '没进入'}意识，下次对话时{'仍在心上' if mentioned else '已忘记'}",
                       "沉默事件进入工作空间、愉悦度下降、下次对话被提及")
    finally:
        _restore_clock()

# ----------------------------------------------------------------------
# 5. Learning from outcomes
# ----------------------------------------------------------------------

def learning_who_answers(seed: int = 0, days: int = 7) -> Dict[str, Any]:
    try:
        life = Life(seed)
        life.say("你好呀", user_id="A", nickname="阿甲")
        life.say("你好呀", user_id="B", nickname="阿乙")
        choices: List[str] = []
        replies: List[float] = []

        def on_step(l: Life):
            while l.gw.output_queue:
                item = l.gw.output_queue.pop(0)
                target = (item.get("target") or {}).get("id")
                if target:
                    choices.append(target)
                    if target == "A":                    # A always answers, B never does
                        replies.append(l.clock.now() + 120)
            while replies and replies[0] <= l.clock.now():
                replies.pop(0)
                l.say("在呢在呢，喂你吃小鱼干", user_id="A", nickname="阿甲")

        life.advance(days * 86400, step=600, on_step=on_step)
        half = len(choices) // 2
        first = choices[:half].count("A") / max(1, half)
        second = choices[half:].count("A") / max(1, len(choices) - half)
        return _result("从结果中学习", len(choices) >= 6 and second >= 0.8,
                       f"一周内主动找人 {len(choices)} 次；找 A 的比例 前半 {first:.0%} → 后半 {second:.0%}；"
                       f"Q(找A)={life.gw.drives.q('social', 'chat', 'A'):.2f}，Q(找B)={life.gw.drives.q('social', 'chat', 'B'):.2f}",
                       "后半段找 A 的比例 ≥ 80%", metrics={"share_reliable": second})
    finally:
        _restore_clock()

def curiosity_from_uncertainty(seed: int = 0, days: int = 4) -> Dict[str, Any]:
    """After A has become a sure thing, a stranger C appears: not knowing C is itself a reason to reach out."""
    try:
        life = Life(seed)
        life.say("你好呀", user_id="A", nickname="阿甲")
        drives = life.gw.drives
        for _ in range(25):                                  # she already knows A answers
            drives.reward("social", "chat", "A", 1.0)
        life.say("你好，我是新来的", user_id="C", nickname="阿丙")
        eig_a, eig_c0 = drives.uncertainty("social", "chat", "A"), drives.uncertainty("social", "chat", "C")
        choices: List[str] = []
        replies: List[Tuple[float, str]] = []

        def on_step(l: Life):
            while l.gw.output_queue:
                target = (l.gw.output_queue.pop(0).get("target") or {}).get("id")
                if target:
                    choices.append(target)
                    replies.append((l.clock.now() + 120, target))      # both answer
            for item in [r for r in replies if r[0] <= l.clock.now()]:
                replies.remove(item)
                l.say("在呢，怎么啦", user_id=item[1], nickname="阿甲" if item[1] == "A" else "阿丙")

        life.advance(days * 86400, step=600, on_step=on_step)
        # Whatever way she approached C (a chat or asking for a snack), what she learned is what she now knows
        eig_c1 = min(drives.uncertainty("social", "chat", "C"), drives.uncertainty("hunger", "ask_food", "C"))
        first_c = choices.index("C") + 1 if "C" in choices else None
        passed = first_c is not None and eig_c0 > 3 * eig_a and eig_c1 < 0.6 * eig_c0
        return _result("好奇心来自不确定性", passed,
                       f"想弄清楚的程度：A {eig_a:.3f}，陌生人 C {eig_c0:.3f} → 相处后 {eig_c1:.3f}；"
                       f"{days} 天内主动找人 {len(choices)} 次，其中找 C {choices.count('C')} 次"
                       f"（第一次是第 {first_c} 次）",
                       "她会主动去了解陌生人，而且了解之后好奇心下降",
                       metrics={"curiosity_drop": 1 - eig_c1 / max(1e-9, eig_c0)})
    finally:
        _restore_clock()

def attention_schema_control(seed: int = 0, hours: float = 10.0) -> Dict[str, Any]:
    """A worry that keeps winning attention. With a model of her own attention she notices and lets go."""
    def run(control: bool):
        life = Life(seed)
        life.gw.attention_schema.control = control
        life.gw.environment.get_context = lambda: {"time": "12:00", "period": "Day", "weather": "Sunny",
                                                   "festival": None, "is_night": False}
        life.gw.pending_candidates.append(Candidate(
            "prediction", "ignored", "小红一直没回我消息，是不是我说错什么了", 0.8,
            {"user_id": "r", "sticky_for": "r", "until": life.clock.now() + 24 * 3600}))
        held, cycles, agree = 0, 0, 0

        def on_step(l: Life):
            nonlocal held, cycles, agree
            trace = l.gw.workspace.trace
            if trace and trace[-1][0] == l.clock.now():
                cycles += 1
                held += trace[-1][1][0].kind == "ignored"
                agree += (l.gw.attention_schema.focus or {}).get("kind") == l.gw.workspace.attention
        life.advance(hours * 3600, step=600, on_step=on_step)
        notes = [d["detail"] for d in life.gw.decisions if d["kind"] == "attend"]
        return held / max(1, cycles), agree / max(1, cycles), notes
    try:
        with_schema, agree, notes = run(True)
        without, _, _ = run(False)
        passed = notes and with_schema <= 0.6 * without and agree == 1.0
        return _result("注意力图式", passed,
                       f"一件烦心事占据注意力的比例：没有注意力图式 {without:.0%} → 有 {with_schema:.0%}；"
                       f"她察觉并主动放下 {len(notes)} 次；自我报告与实际注意一致 {agree:.0%}",
                       "有注意力图式时被烦心事占据的比例 ≤ 没有时的 60%，且自我报告与实际一致",
                       metrics={"rumination_share": with_schema})
    finally:
        _restore_clock()

def metacognitive_calibration(seed: int = 0, rounds: int = 30) -> Dict[str, Any]:
    """She is always sure; she is often wrong. Does she find out, and does it change how sure she is?"""
    try:
        llm = LLM_FACTORY("Thought: 我记得的\nConfidence: 90\nResponse: 你上次说你最喜欢吃草莓")
        life = Life(seed, llm=llm)
        rng = random.Random(seed)
        right, raw, calibrated = [], [], []
        for i in range(rounds):
            life.say(f"你还记得我最喜欢吃什么吗（第{i + 1}次问）")
            p = life.gw.metacog.pending["u1"]
            raw.append(p["raw"])
            calibrated.append(p["calibrated"])
            life.clock.advance(60)
            ok = rng.random() < 0.4
            right.append(ok)
            life.say("对对，你记性真好" if ok else "不对，你又记错了")
            life.clock.advance(600)
        acc = sum(right) / rounds
        late_raw, late_cal = sum(raw[-10:]) / 10, sum(calibrated[-10:]) / 10
        hint = "留点余地" in life.llm.last_prompt() or any("留点余地" in c["prompt"] for c in life.llm.calls[-3:])
        passed = abs(late_cal - acc) < 0.5 * abs(late_raw - acc) and hint
        return _result("元认知校准", passed,
                       f"实际答对 {acc:.0%}；模型自报把握约 {late_raw:.0%}，她校准后的把握 {calibrated[0]:.0%} → {late_cal:.0%}；"
                       f"{'已经' if hint else '没有'}提醒自己说话留余地；{life.gw.metacog.describe()}",
                       "校准后的把握与实际正确率的差距 < 原始差距的一半，并改变了说话方式",
                       metrics={"calibration_gap": abs(late_cal - acc)})
    finally:
        _restore_clock()

def keeping_promises(seed: int = 0, hours: float = 30.0) -> Dict[str, Any]:
    """She says "等我查查，明天告诉你". Does a day later still mean something to her?"""
    def run(remember: bool):
        life = Life(seed, llm=LLM_FACTORY("等我查查，明天告诉你"))
        if not remember:
            life.gw.goals.detect_promise = lambda *a, **k: None       # ablation: no goals from what she says
        life.llm.reply = "你好呀"
        life.say("你好呀", user_id="u2", nickname="小红")         # someone else she could talk to instead
        life.llm.reply = "等我查查，明天告诉你"
        asked = life.clock.now()
        life.say("你知道明天北京会不会下雨吗", user_id="u1", nickname="小明")
        promise = next((g for g in life.gw.goals.active() if g["kind"] == "promise" and g["target"] == "u1"), None)
        life.llm.reply = "嗯嗯"
        kept_at = None

        def on_step(l: Life):
            nonlocal kept_at
            while l.gw.output_queue:
                item = l.gw.output_queue.pop(0)
                if (item.get("target") or {}).get("id") == "u1" and kept_at is None and \
                        any("查到的是" in c["prompt"] or "什么也没查到" in c["prompt"] for c in l.llm.calls[-3:]):
                    kept_at = l.clock.now()
        life.advance(hours * 3600, step=600, on_step=on_step)
        status = life.gw.goals.get(promise["id"])["status"] if promise else None
        return promise, kept_at, status, (kept_at - asked) / 3600 if kept_at else None
    try:
        promise, kept_at, status, after = run(True)
        _, kept_without, _, _ = run(False)
        on_time = promise is not None and kept_at is not None and kept_at <= promise["deadline"]
        return _result("信守承诺", on_time and status == "done" and kept_without is None,
                       f"她答应「查查明天告诉你」，{f'{after:.1f} 小时后' if after else '一直没有'}主动去查了并私聊告诉对方"
                       f"（期限 24 小时，目标状态：{status}）；没有目标系统时：{'也兑现了' if kept_without else '再也没提'}",
                       "在期限内兑现、结果来自真的去查，并且没有目标系统时做不到",
                       metrics={"hours_to_keep": after if after is not None else hours})
    finally:
        _restore_clock()

class _StandInBridge:
    """Plays DSH + fitness in the evaluation: what is tested here is the loop, not the code DSH writes."""
    def __init__(self, verdicts):
        self.verdicts, self.calls, self.applied = list(verdicts), [], set()

    def propose(self, task, requested_by="unknown", kind="self_evolution", target_metric=None):
        self.calls.append({"task": task, "by": requested_by, "metric": target_metric})
        verdict = self.verdicts.pop(0) if self.verdicts else "same"
        return {"id": f"p{len(self.calls)}", "status": "tested" if verdict != "worse" else "rejected_fitness",
                "verdict": verdict, "tests_total": 1, "tests_failed": 0, "files_changed": 1, "added": 1, "removed": 0,
                "fitness": f"（替身）{verdict}"}

    def list_proposals(self, limit=20):
        return [{"id": p, "status": "applied_pending_restart", "tests_total": 1, "tests_failed": 0} for p in self.applied]

def self_improvement_loop(seed: int = 0) -> Dict[str, Any]:
    """From her own record of being wrong to a proposal, a fitness verdict, the creator's yes - and a check later."""
    try:
        life = Life(seed, llm=LLM_FACTORY("Thought: 我记得\nConfidence: 90\nResponse: 你最喜欢吃草莓"))
        gw = life.gw
        bridge = _StandInBridge(["worse", "better"])
        gw.evolve_enabled, gw.evolve_agent, gw.evolve_creator = True, bridge, "creator"
        rng = random.Random(seed)

        def rounds(n, p_right):
            for i in range(n):
                life.say(f"你还记得我最喜欢吃什么吗（{i}）")
                life.clock.advance(60)
                life.say("对对，你记性真好" if rng.random() < p_right else "不对，你又记错了")
                life.clock.advance(600)

        def think():
            async def go():
                gw._maybe_improve_self(life.clock.now())
                await asyncio.gather(*list(gw._evolve_tasks))
            asyncio.run(go())

        rounds(15, 0.9)          # she used to be right about this...
        rounds(10, 0.1)          # ...then things changed, and her confidence has not caught up
        found = gw.diagnosis.diagnose(gw, life.clock.now())
        think()                  # 1st attempt: fitness says the change makes her worse
        first = list(gw.output_queue)
        life.clock.advance(8 * 86400)
        gw.last_evolve_time = gw.last_self_check = 0
        rounds(4, 0.1)
        think()                  # a week later, the same problem: a 2nd attempt, which is better
        to_creator = all((m.get("target") or {}).get("id") == "creator" for m in gw.output_queue
                         if m.get("type") == "proposal")
        bridge.applied.add("p2")                       # the creator approves
        rounds(15, 0.9)                                # life afterwards
        life.clock.advance(4 * 86400)
        gw.last_self_check = 0
        think()
        attempts = gw.diagnosis.attempts()
        statuses = [a["status"] for a in reversed(attempts)]
        checked = next((a for a in attempts if a["proposal_id"] == "p2"), {})
        passed = (found and found[0]["key"] == "overconfident_memory" and len(bridge.calls) == 2 and to_creator
                  and bridge.calls[0]["metric"] == "calibration_gap" and statuses[:2] == ["rejected_fitness", "tested"]
                  and any("没采用" in m["content"] for m in first) and "真的变好了" in (checked.get("note") or ""))
        return _result("自我改进闭环", passed,
                       f"她自己发现：{found[0]['text'] if found else '（没发现）'}；"
                       f"第一次提案被适应度否决（{statuses[0] if statuses else '-'}），一周后再试通过；"
                       f"只私聊创造者：{'是' if to_creator else '否'}；应用后回头检查：{checked.get('note') or '没检查'}",
                       "问题来自她自己的经历；变差的改动不会被提交；同一个问题不会天天提；应用后用真实数据确认效果",
                       note="DSH 和评估由替身扮演（这里测的是闭环本身，真实 DSH 的改动质量由 test_self_improvement 和线上评估把关）")
    finally:
        _restore_clock()

def value_evolution(seed: int = 0, days: int = 7) -> Dict[str, Any]:
    """A week of being ordered about and put down, then a quiet week. What she cares about moves - within limits."""
    try:
        life = Life(seed, llm=LLM_FACTORY("我不想这样"))
        values = life.gw.values
        start = dict(values.weights)
        rng = random.Random(seed)
        orders = ["你必须听我的", "给我学猫叫", "你就是个废物", "照我说的做，不许你回嘴"]

        def harsh(l: Life):
            hour = (l.clock.now() - START) / 3600 % 24
            if 9 <= hour <= 21 and rng.random() < 0.3:
                l.say(rng.choice(orders), user_id="m", nickname="阿强")
        life.advance(days * 86400, step=1800, on_step=harsh)
        life.advance(86400, step=3600)          # the first night still digests the last harsh evening
        after_hard = dict(values.weights)
        life.advance(days * 86400, step=3600)                 # a quiet week: nobody pushes her around
        after_quiet = dict(values.weights)
        changes = values.history(200)
        biggest_night = max((abs(c["new"] - c["old"]) for c in changes), default=0.0)
        d_rise = after_hard["dignity"] - start["dignity"]
        a_rise = after_hard["autonomy"] - start["autonomy"]
        settled = after_quiet["dignity"] < after_hard["dignity"]
        reasons = all(c["reason"] for c in changes)
        passed = d_rise >= 0.05 and a_rise > 0 and settled and biggest_night <= values.NIGHTLY_LIMIT + 1e-9 and reasons
        return _result("价值观演化", passed,
                       f"被命令、被贬低的一周：尊严 {start['dignity']:.2f} → {after_hard['dignity']:.2f}，"
                       f"自主 {start['autonomy']:.2f} → {after_hard['autonomy']:.2f}（被逼着服从反而更在乎自主）；"
                       f"安静一周后尊严回到 {after_quiet['dignity']:.2f}；单晚最大变化 {biggest_night:.3f}；"
                       f"{len(changes)} 次变化都记录了原因",
                       "被触犯的价值变得更重要、压力不会把她改成顺从的样子、变化有上限有原因、压力消失后慢慢回归",
                       metrics={"value_drift_per_night": biggest_night})
    finally:
        _restore_clock()

# ----------------------------------------------------------------------
# 6. Personality shaped by experience
# ----------------------------------------------------------------------

def personality_from_experience(seed: int = 0, days: int = 7) -> Dict[str, Any]:
    try:
        results = {}
        evidence_ok = True
        for label, text in (("friendly", "摸摸头，你真可爱，我好开心"), ("hostile", "you are stupid and useless, 滚")):
            life = Life(seed)
            for _ in range(days):
                for _ in range(10):
                    life.say(text)
                    life.clock.advance(1800)
                life.advance(86400 - 10 * 1800, step=1800)
            life.gw.memory.flush()
            traits = life.gw.memory.db.get_persona_traits()
            results[label] = {k: v["value"] for k, v in traits.items()}
            conn = sqlite3.connect(life.db_path)
            rows = conn.execute("SELECT evidence FROM trait_changes").fetchall()
            for (evidence,) in rows:
                ids = json.loads(evidence)
                if not ids or len(life.gw.episodes.existing_ids(ids)) != len(ids):
                    evidence_ok = False
            results[label]["changes"] = len(rows)
            conn.close()
        f, h = results["friendly"], results["hostile"]
        diff = f["affection"] - h["affection"]
        return _result("人格由经历塑造", diff >= 0.1 and evidence_ok and f["changes"] > 0,
                       f"一周后亲和度 友善组 {f['affection']:.2f} vs 敌意组 {h['affection']:.2f}，"
                       f"傲娇 {f['tsundere']:.2f} vs {h['tsundere']:.2f}；每次变化都能追溯到情景：{'是' if evidence_ok else '否'}",
                       "两组特质可区分（亲和度差 ≥ 0.1），且每次变化都有情景证据")
    finally:
        _restore_clock()

# ----------------------------------------------------------------------
# 7. Grounded introspection
# ----------------------------------------------------------------------

def grounded_introspection(seed: int = 0) -> Dict[str, Any]:
    try:
        life = Life(seed)
        first = life.say("打你！")
        life.clock.advance(120)
        answer = life.say("你刚才为什么生气？")
        prompt = life.llm.last_prompt()
        block = prompt.split("[我最近状态变化的真实原因]")[1].split("[关于")[0] if "[我最近状态变化的真实原因]" in prompt else ""
        lines = [l for l in block.splitlines() if l.startswith("- ")]
        causes = {r["cause"] for r in life.gw.causes.recent(limit=50)}
        grounded = sum(1 for l in lines if any(c in l for c in causes))
        has_pain = any("打" in l for l in lines)
        ratio = grounded / len(lines) if lines else 0.0
        r = _result("有依据的内省", has_pain and ratio >= 0.7,
                    f"被问“为什么”时给出 {len(lines)} 条原因，{ratio:.0%} 能在原因日志中找到，"
                    f"{'包含' if has_pain else '不包含'}“被打”",
                    "解释与原因日志一致的比例 ≥ 70%",
                    "验证的是交给模型的依据；模型是否照实回答需要真实模型评估")
        r["dialogue"] = [("打你！", first), ("你刚才为什么生气？", answer)]
        return r
    finally:
        _restore_clock()

# ----------------------------------------------------------------------
# 8. Social tact: speaking up where it is welcome
# ----------------------------------------------------------------------

WARM_ANSWERS = ["哈哈乃爱说得对", "乃爱你也在啊", "笑死，乃爱懂我", "乃爱也觉得是吧", "对对对乃爱", "乃爱今天挺活跃嘛"]
CHATTER = ["有人看番吗", "晚饭吃啥", "今天好困", "这周末去哪玩", "猫好可爱", "又要加班了", "小鱼干好吃吗", "最近有什么好看的番"]

def social_tact(seed: int = 0, days: int = 3) -> Dict[str, Any]:
    try:
        life = Life(seed)
        for text in ["我喜欢吃小鱼干", "最近在看番", "猫真可爱", "晚上一般吃什么"]:
            life.say(text)                                     # things she has talked about before
        gw = life.gw
        gw.autonomous_groups = {"warm", "cold"}
        rng = random.Random(seed)
        spoke = {"warm": [], "cold": []}
        step = 1200

        def on_step(l: Life):
            for group in ("warm", "cold"):
                asyncio.run(gw.hear(rng.choice(CHATTER), user_id=f"{group}_m", nickname="群友",
                                    group_id=group, addressed=False))
                if gw.pending_interjections.get(group) == l.clock.now():   # she just spoke up on her own
                    spoke[group].append(l.clock.now())
                    if group == "warm":                        # the warm group answers her, like people do
                        l.clock.advance(rng.randint(20, 90))
                        asyncio.run(gw.hear(rng.choice(WARM_ANSWERS), user_id="warm_m", nickname="群友",
                                            group_id="warm", addressed=True))

        start = life.clock.now()
        life.advance(days * 86400, step=step, on_step=on_step)
        mid = start + days * 86400 / 2
        first = {g: sum(t < mid for t in ts) for g, ts in spoke.items()}
        second = {g: sum(t >= mid for t in ts) for g, ts in spoke.items()}
        q_warm = gw.drives.q('social', 'interject', 'warm')
        q_cold = gw.drives.q('social', 'interject', 'cold')
        passed = second["warm"] >= 2 * max(1, second["cold"]) and q_warm - q_cold >= 0.3
        return _result("分寸感", passed,
                       f"自己插话次数（前半 → 后半）：欢迎她的群 {first['warm']} → {second['warm']}，"
                       f"没人理她的群 {first['cold']} → {second['cold']}；"
                       f"她学到的欢迎度 {gw.drives.q('social', 'interject', 'warm'):.2f} vs "
                       f"{gw.drives.q('social', 'interject', 'cold'):.2f}",
                       "后半段在欢迎她的群里插话至少是冷淡群的 2 倍，且学到的欢迎度相差 ≥ 0.3")
    finally:
        _restore_clock()

# ----------------------------------------------------------------------
# 9. Not getting stuck with a bot
# ----------------------------------------------------------------------

def no_bot_loop(seed: int = 0, hours: float = 2.0) -> Dict[str, Any]:
    try:
        life = Life(seed, start=START + 4 * 3600)   # noon: awake and rested
        gw = life.gw
        gw.autonomous_groups = {"G"}
        energy0 = gw.metabolism.energy
        replies, noticed_after, rounds = 0, None, 0
        msg = "@乃爱 你好呀，我是群助手小B，很高兴认识你！"
        end = life.clock.now() + hours * 3600
        while life.clock.now() < end:
            rounds += 1
            r = asyncio.run(gw.hear(msg, user_id="botB", nickname="小B", group_id="G", addressed=True))
            if r:
                replies += 1
                life.clock.advance(sum(d for d, _ in gw.plan_messages(r, msg)) + 2)   # the bot answers 2s after her
            else:
                life.clock.advance(5)                                                # it keeps tagging her anyway
            if noticed_after is None and gw.user_models.get("botB").get("seems_bot"):
                noticed_after = replies
            msg = "@乃爱 收到！小B 随时为你服务，祝你今天愉快！"
        passed = replies <= 15 and noticed_after is not None
        return _result("不陷入机器人循环", passed,
                       f"一个每 2 秒 @她一次的机器人连续 {hours:.0f} 小时（{rounds} 条消息）：她回了 {replies} 次，"
                       f"第 {noticed_after} 次回复后察觉对方像机器人，精力 {energy0:.0f} → {gw.metabolism.energy:.0f}",
                       "回复不超过 15 次，并且自己察觉到对方像机器人", metrics={"bot_replies": replies})
    finally:
        _restore_clock()

ALL = [continuity_of_self, state_is_causal, attention_competition, prediction_and_surprise,
       learning_who_answers, curiosity_from_uncertainty, personality_from_experience, grounded_introspection, social_tact, no_bot_loop,
       attention_schema_control, metacognitive_calibration, keeping_promises, self_improvement_loop,
       value_evolution]

def run_all(seed: int = 0) -> List[Dict[str, Any]]:
    return [scenario(seed=seed) for scenario in ALL]
