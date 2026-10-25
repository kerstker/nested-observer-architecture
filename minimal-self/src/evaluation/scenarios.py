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
from typing import Dict, Any, List

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
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

def _result(name: str, passed: bool, measured: str, criterion: str, note: str = "") -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "measured": measured, "criterion": criterion, "note": note}

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
                       "后半段找 A 的比例 ≥ 80%")
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
                       "回复不超过 15 次，并且自己察觉到对方像机器人")
    finally:
        _restore_clock()

ALL = [continuity_of_self, state_is_causal, attention_competition, prediction_and_surprise,
       learning_who_answers, personality_from_experience, grounded_introspection, social_tact, no_bot_loop]

def run_all(seed: int = 0) -> List[Dict[str, Any]]:
    return [scenario(seed=seed) for scenario in ALL]
