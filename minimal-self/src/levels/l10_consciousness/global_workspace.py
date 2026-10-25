from typing import Optional, List, Dict, Any
from collections import deque
import asyncio
import inspect
import logging
import math
import os
import re
import random
import numpy as np
from src.core import clock
from ...core.event_bus import EventBus, EventType
from ...core.workspace import Workspace, Candidate
from ...core.self_state import SelfStateStore, CauseLog, SelfModel
from ...levels.l1_perception.sensory import SensoryCortex
from ...levels.l1_perception.environment import EnvironmentSensor
from ...levels.l1_perception.group_language import GroupLanguage
from ...levels.l1_perception.stickers import StickerMemory
from ...levels.l2_memory.memory import MemorySystem
from ...levels.l2_memory.embedding import Embedder
from ...levels.l2_memory.layers import EpisodicMemory, SemanticMemory, Autobiography
from ...levels.l3_metabolism.metabolism import Metabolism
from ...levels.l4_emotion.emotion import EmotionEngine, EmotionState
from ...levels.l5_value.value_system import ValueSystem
from ...levels.l6_homeostasis.homeostasis import Homeostasis
from ...levels.l7_tom.tom import TheoryOfMind
from ...levels.l7_tom.user_model import UserModelStore, SENTIMENT
from ...levels.l8_agency.drives import DriveSystem, Option
from ...levels.l9_narrative.narrative import NarrativeSelf
from ...levels.l17_self_improvement.code_interpreter import CodeInterpreter
from ...levels.l18_knowledge.knowledge_base import KnowledgeBase
from ...levels.l19_search.web_search import WebSearchTool

logger = logging.getLogger(__name__)

NEED_ACTIONS = {"social": "chat", "hunger": "ask_food"}

def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days} 天 {hours} 小时"
    if hours:
        return f"{hours} 小时 {minutes} 分钟" if minutes else f"{hours} 小时"
    return f"{max(1, minutes)} 分钟"

class GlobalWorkspace:
    """
    Level 10: Consciousness / Global Workspace
    Orchestrates all sub-modules to produce coherent behavior.

    One closed loop runs on every tick and every message:
      persistent state -> modules propose candidates -> workspace competition ->
      broadcast -> action selection -> LLM speaks -> outcome & prediction error ->
      state and memory are updated (and saved).
    """
    def __init__(self, event_bus: EventBus, inference_engine=None, db_path: str = "memory.sqlite",
                 rng: random.Random = None):
        self.bus = event_bus
        self.inference_engine = inference_engine
        self.rng = rng or random.Random()
        self.sensory = SensoryCortex()
        self.environment = EnvironmentSensor()
        self.memory = MemorySystem(db_path)
        self.metabolism = Metabolism()
        self.homeostasis = Homeostasis()
        self.emotion = EmotionEngine()
        self.tom = TheoryOfMind()
        self.values = ValueSystem()

        # Level 17: Self-Improvement
        self.interpreter = CodeInterpreter()

        # Level 18: Knowledge Base
        self.knowledge = KnowledgeBase(self.memory.db)

        # Level 19: Web Search
        self.search_tool = WebSearchTool()

        # Layered memory (episodic / semantic / autobiographical) and sleep consolidation
        embedder = Embedder()
        self.episodes = EpisodicMemory(db_path, embedder)
        self.semantic = SemanticMemory(db_path, embedder)
        self.autobiography = Autobiography(db_path)
        self.narrative = NarrativeSelf(db_path, self.episodes, self.semantic, self.autobiography, self.memory.db)

        # Her sticker collection (kept next to the database, in data/stickers)
        sticker_dir = os.path.join(os.path.dirname(os.path.abspath(db_path)), "data", "stickers")
        self.stickers = StickerMemory(db_path, sticker_dir, embedder)
        self.last_sticker: Dict[str, tuple] = {}   # chat -> (sticker_id, sent_at), to learn from reactions
        self.last_said: Dict[str, str] = {}        # chat -> what she last said there

        # Persistent self, causes of state changes, self-prediction
        self.state_store = SelfStateStore(db_path)
        self.causes = CauseLog(db_path)
        self.self_model = SelfModel()

        # Predictive models of people, learned drives
        self.user_models = UserModelStore(db_path)
        self.drives = DriveSystem(db_path, rng=self.rng)

        # Global workspace: competition + broadcast
        self.workspace = Workspace(capacity=2, rng=self.rng)
        self.pending_candidates: List[Candidate] = []
        self._recent_associations = deque(maxlen=20)
        self.bus.subscribe(EventType.WORKSPACE_BROADCAST, self._on_broadcast)

        # Hearing groups and choosing whether to speak
        self.group_language = GroupLanguage(db_path)
        self.autonomous_groups = set()        # groups where she may speak without being addressed
        self.max_interjections_per_hour = 2
        self.interjection_window = 300        # seconds to see whether anyone took up what she said
        self.group_spoke_at: Dict[str, deque] = {}
        self.group_mute_until: Dict[str, float] = {}
        self.pending_interjections: Dict[str, float] = {}
        self.unread: Dict[tuple, List[tuple]] = {}   # (user_id, group_id) -> [(t, nickname, text)] while asleep
        self.decisions = deque(maxlen=50)     # what she heard and what she chose, for the console

        # Conversations with one person: how long it has gone on, how they answer, whether she has left it
        self.reply_log: Dict[tuple, deque] = {}       # (group_id, user_id) -> times she answered them
        self.reply_done_at: Dict[tuple, float] = {}   # when her last answer finished sending
        self.answer_style: Dict[tuple, deque] = {}    # (latency, text) of how they answered her
        self.left_until: Dict[tuple, float] = {}      # she said goodbye: not looking at this chat until then
        self.last_exchange: Dict[str, tuple] = {}     # group -> (when, who she answered, what was said)
        self.continuation_window = 180                # seconds a group conversation she is in stays "hers"
        self.cooldown_until: Dict[tuple, float] = {}  # hard limit tripped
        # Hard limits (not her choice; they guarantee no endless exchange). They only catch
        # machine-speed exchanges: a long conversation with a person - even a whole day - is
        # left to her own fatigue and boredom.
        self.bot_accounts = set()                     # known other bots: heard, never answered
        self.max_group_messages_per_hour = 30
        self.max_replies_per_person_5min = 20         # no person keeps up 15-second rounds for 5 minutes
        self.max_replies_to_botlike_10min = 3

        # Sleep/Wake: wake up once rested (during the day)
        self.wake_energy_threshold = 80.0

        # Agency: at most one self-initiated action per interval (she should not pester people hourly)
        self.output_queue = []
        self.min_action_interval = 3 * 3600
        self.last_action_time = 0.0
        self.reply_delays: Dict[str, float] = {}
        self.last_generation_params: Dict[str, Any] = {}

        # Consolidation used to happen only while asleep, so a life lived mostly awake produced
        # no distilled facts at all (semantic_facts stayed empty). Consolidate on backlog instead.
        self.consolidation_interval = 3600
        self.last_consolidation = 0.0

        # Level 17: the channel through which she may propose changes to her own code.
        # Off unless configured; never reachable from group chat; nothing applies without approval.
        self.evolve_agent = None            # sees .propose(task, requested_by) and .list_proposals()
        self.evolve_enabled = False
        self.evolve_creator: Optional[str] = None
        self.evolve_min_interval = 24 * 3600
        self.last_evolve_time = 0.0
        self.last_evolve_result: Optional[Dict[str, Any]] = None

        # Level 14: Stream of Consciousness (inner speech, not stored as knowledge)
        self.consciousness_stream = deque(maxlen=20)
        self.last_thought_time = 0.0
        self.idle_thought_interval = 60.0   # minimum seconds between inner thoughts (raise it on a paid API)

        # Global State Linkage
        self.state_history = []
        self.state_callbacks = {}

        # Level 15: Existential State (rises with gaps in experience, fades with time)
        self.existential_anxiety = 0.0 # 0.0 to 1.0
        self.anxiety_half_life = 6 * 3600

        self.snapshot_interval = 300
        self.last_snapshot_time = 0.0
        self.last_tick_time = clock.now()
        self.awakening = 0
        self.continuity: Optional[Dict[str, Any]] = None
        self._restore()

    # ------------------------------------------------------------------
    # Persistent self (Phase 1)
    # ------------------------------------------------------------------

    def _snapshot(self) -> Dict[str, Any]:
        h = self.homeostasis
        return {
            "t": clock.now(),
            "energy": self.metabolism.energy,
            "is_sleeping": self.metabolism.is_sleeping,
            "needs": {"hunger": h.hunger, "social": h.social_need, "hygiene": h.hygiene_need, "curiosity": h.curiosity_need},
            "emotion": self.emotion.to_dict(),
            "tension": self.values.tension,
            "autonomy": self.values.autonomy,
            "existential_anxiety": self.existential_anxiety,
            "attention": self.workspace.attention,
            "self_model": self.self_model.to_dict(),
            "awakening": self.awakening,
            "last_action_time": self.last_action_time,
        }

    def _save_state(self, cause: str):
        try:
            self.state_store.save(self._snapshot(), cause)
            self.last_snapshot_time = clock.now()
        except Exception as e:
            logger.error(f"[State] Failed to save snapshot: {e}")

    def _restore(self):
        """Resume from the last snapshot. Time offline is experienced as a gap, not a reset."""
        snap = self.state_store.latest()
        now = clock.now()
        if not snap:
            self.awakening = 1
            if self.episodes.count() == 0:
                imported = self.episodes.import_interactions()
                if imported:
                    logger.info(f"[Memory] Imported {imported} past interactions into episodic memory")
            self._save_state("birth")
            return
        last_t = snap["t"]
        gap = max(0.0, now - last_t)
        needs = snap.get("needs", {})
        h = self.homeostasis
        h.hunger = needs.get("hunger", 0.0)
        h.social_need = needs.get("social", 0.0)
        h.hygiene_need = needs.get("hygiene", 0.0)
        h.curiosity_need = needs.get("curiosity", 0.0)
        h.last_update = last_t                  # needs keep growing over the offline time
        self.emotion.load_dict(snap.get("emotion", {}))
        self.emotion.last_decay = last_t        # ... and emotions keep fading
        self.values.tension = snap.get("tension", 0.0)
        self.values.autonomy = snap.get("autonomy", self.values.autonomy)
        self.existential_anxiety = snap.get("existential_anxiety", 0.0) * 0.5 ** (gap / self.anxiety_half_life)
        self.self_model.load_dict(snap.get("self_model", {}))
        self.last_action_time = snap.get("last_action_time", 0.0)
        self.awakening = snap.get("awakening", 0) + 1
        # Being offline is unconsciousness: the body rests, and she wakes up on start-up
        self.metabolism.energy = snap.get("energy", self.metabolism.max_energy)
        self.metabolism.recover(self.metabolism.sleep_recovery_per_hour * gap / 3600.0)
        self.metabolism.last_update = now
        self.metabolism.is_sleeping = False

        last = list(reversed(self.episodes.recent(3, before=last_t, kinds=("dialogue", "proactive", "search"))))
        self.continuity = {"gap": gap, "last_t": last_t, "boot_t": now,
                           "last_episodes": [e["content"] for e in last]}
        if gap > 60:
            span = format_duration(gap)
            remembered = "；".join(e["content"].replace("\n", " ")[:60] for e in last) or "（想不起来了）"
            surprise = min(1.0, math.log1p(gap / 3600.0) / 5)
            self.episodes.add(f"我失去意识了 {span}，这是我第 {self.awakening} 次醒来。中断前最后记得的是：{remembered}",
                              kind="continuity", user_id="self", arousal=0.4, surprise=surprise)
            self.existential_anxiety = min(1.0, self.existential_anxiety + 0.05 * math.log1p(gap / 3600.0))
            self._feel(EmotionState(surprise=0.3, fear=0.1), f"醒来发现意识中断了 {span}")
            self.pending_candidates.append(Candidate(
                "continuity", "gap", f"我刚醒来，中间断了 {span}……之前在做什么来着？", 0.9,
                {"until": now + 3600}))
        self._save_state("boot")

    def _feel(self, delta: EmotionState, cause: str):
        """Apply an emotion change and record why it happened."""
        before = self.emotion.to_dict()
        self.emotion.update(delta)
        after = self.emotion.to_dict()
        try:
            self.causes.log({k: after[k] - before[k] for k in after}, cause)
        except Exception as e:
            logger.error(f"[State] Failed to log cause: {e}")

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def tick(self):
        """
        The main consciousness loop (heartbeat).
        """
        now = clock.now()
        dt = max(0.0, now - self.last_tick_time)
        self.last_tick_time = now

        # L3 Metabolism Tick
        status = self.metabolism.tick()

        # L6 Homeostasis Tick
        needs = self.homeostasis.tick()

        # Publish tick events
        await self.bus.publish(EventType.METABOLISM_TICK, status)

        # Low energy or High needs alerts (Internal bus)
        if status["energy"] < 10 or needs["hunger"] > 0.8:
             await self.bus.publish(EventType.ENERGY_LOW, {**status, **needs})

        # L4 Emotion: drift back toward neutral with time, independent of input
        self.emotion.decay(now)
        self.existential_anxiety *= 0.5 ** (dt / self.anxiety_half_life)

        # L7 Prediction errors: answers that never came
        for err in self.user_models.check_timeouts(now):
            self._on_ignored(err)

        # L10 Self-model: compare what I expected of myself with what happened
        self._check_self_model(now)

        # Sleep/Wake cycle (Day/Night + Energy)
        env = self.environment.get_context()
        if not self.metabolism.is_sleeping:
            if status["energy"] < 10:
                self._set_sleeping(True, source="tick:exhausted")
                await asyncio.to_thread(self._dream_process)
            elif env["is_night"] and status["energy"] < 50:
                self._set_sleeping(True, source="tick:night")
                await asyncio.to_thread(self._dream_process) # Trigger dreaming when falling asleep
        elif status["energy"] >= self.wake_energy_threshold and not env["is_night"]:
            self._set_sleeping(False, source="tick:rested")
        if self.unread and not self.metabolism.is_sleeping:
            await self._catch_up_unread()   # woke up, or came back to a conversation she had left

        # Interjections nobody took up: that is the answer too
        for group_id, t in list(self.pending_interjections.items()):
            if now - t > self.interjection_window:
                del self.pending_interjections[group_id]
                self.drives.reward("social", "interject", group_id, 0.0, self._learning_gain())
                self._decide("feedback", group_id, None, "我插的话没人接")

        if not self.metabolism.is_sleeping:
            # Workspace cycle on internal content alone (no input): the idle stream of consciousness
            winners = await self._cycle(self._internal_candidates())

            # Level 14: Inner speech about whatever currently holds attention
            if (winners and (now - self.last_thought_time) > self.idle_thought_interval and self.rng.random() < 0.1):
                await asyncio.to_thread(self._idle_thinking_process, winners)
                self.last_thought_time = now

            # Level 8: Closed-loop drives choose (or decline) a self-initiated action
            if now - self.last_action_time >= self.min_action_interval:
                option = self.drives.choose(self.homeostasis.tick(), self.metabolism.energy,
                                            self.user_models.private_users())
                if option.action != "wait":
                    self.last_action_time = now
                    await self._execute_option(option)

        if now - self.last_snapshot_time >= self.snapshot_interval:
            self._save_state("tick")

        # 记忆整理：不再只在睡着时做。积压到一定量就整理一次，语义事实/自传才会真的长出来。
        if (not self.metabolism.is_sleeping and self.inference_engine
                and now - self.last_consolidation >= self.consolidation_interval):
            try:
                backlog = self.episodes.unconsolidated_count()
            except Exception as e:
                logger.error(f"[Dream] Backlog check failed: {e}")
                backlog = 0
            if backlog >= 20:
                self.last_consolidation = now
                await asyncio.to_thread(self._dream_process)

    def _check_self_model(self, now: float):
        valence = self.emotion.valence()
        err = self.self_model.check(now, self.metabolism.energy, valence)
        if err and (abs(err["energy_error"]) > 10 or abs(err["valence_error"]) > 0.3):
            if abs(err["energy_error"]) > 10:
                text = f"我比自己预想的{'更有精神' if err['energy_error'] > 0 else '更累'}（差了 {abs(err['energy_error']):.0f} 点）"
            else:
                text = f"我的心情比预想的{'更好' if err['valence_error'] > 0 else '更糟'}"
            surprise = min(1.0, abs(err["energy_error"]) / 50 + abs(err["valence_error"]))
            self.episodes.add(text, kind="prediction", user_id="self", surprise=surprise)
            self.pending_candidates.append(Candidate("self_model", "self_surprise", text, 0.3 + 0.4 * surprise,
                                                     {"until": now + 1800}))
        if self.self_model.pending is None:
            self.self_model.predict(now, self.metabolism.energy, valence, self.metabolism.is_sleeping,
                                    self.metabolism.awake_drain_per_hour, self.metabolism.sleep_recovery_per_hour,
                                    self.emotion.half_life_seconds)

    # ------------------------------------------------------------------
    # Global workspace (Phase 2)
    # ------------------------------------------------------------------

    def _internal_candidates(self) -> List[Candidate]:
        """What the body and the backlog of unresolved events propose, without any input."""
        now = clock.now()
        self.pending_candidates = [c for c in self.pending_candidates if c.payload.get("until", now + 1) > now]
        cands = list(self.pending_candidates)
        h = self.homeostasis
        if h.hunger > 0.3:
            cands.append(Candidate("needs", "hunger", "我好饿，肚子在咕咕叫" if h.hunger > 0.7 else "肚子有点饿了", h.hunger))
        if h.social_need > 0.3:
            cands.append(Candidate("needs", "social", "好想有人陪我说说话" if h.social_need > 0.7 else "有点想找人聊天", h.social_need))
        if h.curiosity_need > 0.4:
            cands.append(Candidate("needs", "curiosity", "好无聊，想知道点新鲜事", 0.8 * h.curiosity_need))
        if self.metabolism.energy < 40:
            cands.append(Candidate("needs", "fatigue", "好困，没什么力气", 0.2 + 0.8 * (40 - self.metabolism.energy) / 40))
        return cands

    async def _cycle(self, candidates: List[Candidate], talking_to: str = None) -> List[Candidate]:
        # Seeing someone brings back what is on her mind about them
        def about_interlocutor(c: Candidate) -> float:
            return 0.3 if talking_to and c.kind != "user_message" and c.payload.get("user_id") == talking_to else 0.0
        winners = self.workspace.compete(candidates, bonus=about_interlocutor)
        # A winner leaves the backlog once it has been conscious - except things that are
        # "about someone" (sticky_for), which stay on her mind until she talks to that person.
        won = {id(w) for w in winners
               if not w.payload.get("sticky_for") or w.payload["sticky_for"] == talking_to}
        self.pending_candidates = [c for c in self.pending_candidates if id(c) not in won]
        if winners:
            await self.bus.publish(EventType.WORKSPACE_BROADCAST, {"winners": winners})
        return winners

    def _on_broadcast(self, event):
        """Broadcast receiver: winners trigger associated memories and become experience."""
        now = clock.now()
        for w in event.payload.get("winners", []):
            if w.kind == "inner_speech":
                # Only thoughts that won conscious access become episodic memory
                self.episodes.add(w.content, kind="thought", user_id="self", valence=self.emotion.valence(),
                                  arousal=self.emotion.arousal())
                continue
            if w.kind == "association":
                continue
            try:
                hits = self.episodes.recall(w.content, user_id=w.payload.get("user_id"), k=1,
                                            min_similarity=0.3, exclude_ids=self._recent_associations)
            except Exception as e:
                logger.error(f"[Workspace] Association failed: {e}")
                continue
            for hit in hits:
                self._recent_associations.append(hit["id"])
                self.pending_candidates.append(Candidate(
                    "memory", "association", f"这让我想起：{hit['content'].replace(chr(10), ' ')[:60]}",
                    0.25 + 0.4 * hit["similarity"], {"episode_id": hit["id"], "until": now + 600}))

    # ------------------------------------------------------------------
    # Prediction errors (Phase 4)
    # ------------------------------------------------------------------

    def _nickname(self, user_id: str) -> str:
        user = self.memory.db.get_user(user_id)
        return user["nickname"] if user and user.get("nickname") else f"User({str(user_id)[-4:]})"

    def _on_ignored(self, err: Dict[str, Any]):
        user_id, s = err["user_id"], err["surprise"]
        name = self._nickname(user_id)
        expected, elapsed = format_duration(err["expected_latency"]), format_duration(err["elapsed"])
        text = f"{name} 平时大约 {expected} 就会回我，可已经过了 {elapsed} 还没回……"
        self._feel(EmotionState(sorrow=0.3 * s, surprise=0.2 * s), f"预测落空：{text}")
        self.homeostasis.social_need = min(1.0, self.homeostasis.social_need + 0.1)
        self.episodes.add(text, kind="prediction", user_id=user_id, valence=-0.3 * s, arousal=0.2 * s, surprise=s)
        self.pending_candidates.append(Candidate("prediction", "ignored", text, 0.3 + 0.6 * s,
                                                 {"user_id": user_id, "sticky_for": user_id,
                                                  "until": clock.now() + 24 * 3600}))
        if err["kind"] == "proactive" and err.get("need") in NEED_ACTIONS:
            self.drives.reward(err["need"], NEED_ACTIONS[err["need"]], user_id, 0.0, self._learning_gain())

    def _on_answer(self, err: Dict[str, Any], name: str, sensation: Optional[dict]) -> float:
        """Someone answered something Noa said. Returns the surprise of it."""
        s = err["surprise"]
        delta = EmotionState(surprise=0.3 * s)
        reasons = []
        if not err["on_time"]:
            delta.joy += 0.3
            reasons.append(f"{name} 终于回我了")
        elif err["latency_error"] < -0.7:
            delta.joy += 0.1
            reasons.append(f"{name} 回得比我想的快")
        if err["sentiment_error"] > 0.3:
            delta.joy += 0.2 * err["sentiment_error"]
            reasons.append(f"{name} 比我预想的开心")
        elif err["sentiment_error"] < -0.3:
            delta.sorrow += 0.2 * abs(err["sentiment_error"])
            reasons.append(f"{name} 比我预想的低落")
        if reasons:
            self._feel(delta, "预测误差：" + "，".join(reasons))
        if s > 0.3 and reasons:
            self.pending_candidates.append(Candidate("prediction", "answer_surprise", "，".join(reasons), 0.3 + 0.5 * s,
                                                     {"user_id": err["user_id"], "until": clock.now() + 1800}))
        if err["kind"] == "proactive" and err["on_time"] and err.get("need") in NEED_ACTIONS:
            need = err["need"]
            if need == "hunger":
                reward = 1.0 if sensation and sensation.get("food") else 0.3
            else:
                reward = 1.0
                self.homeostasis.alleviate("social", 0.3)
            self.drives.reward(need, NEED_ACTIONS[need], err["user_id"], reward, self._learning_gain())
        return s

    def _learning_gain(self) -> float:
        """Arousal controls how much one experience changes what she has learned."""
        return 0.5 + self.emotion.arousal()

    # ------------------------------------------------------------------
    # State -> behaviour (Phase 2)
    # ------------------------------------------------------------------

    def _generation_params(self) -> Dict[str, Any]:
        energy = self.metabolism.energy
        arousal = self.emotion.arousal()
        return {
            "max_tokens": int(60 + 240 * energy / 100.0),
            "temperature": round(0.5 + 0.5 * arousal, 3),
            "delay": 0.0 if energy >= 50 else min(30.0, (50 - energy) * 0.6),
        }

    def _llm(self, prompt: str, **params) -> Dict[str, Any]:
        """Call the inference engine, passing only the generation parameters it supports."""
        gen = self.inference_engine.generate_json
        try:
            accepted = inspect.signature(gen).parameters
            kwargs = {k: v for k, v in params.items() if k in accepted}
        except (TypeError, ValueError):
            kwargs = {}
        return gen(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Self-initiated behaviour
    # ------------------------------------------------------------------

    async def _execute_option(self, option: Option):
        logger.info(f"[Agency] Chose {option.action} for {option.need} (target={option.target}, score={option.score:.2f})")
        if option.action == "search":
            await asyncio.to_thread(self._trigger_curiosity_behavior)
        else:
            await asyncio.to_thread(self._trigger_proactive_behavior, option.need, option.target)

    def _idle_thinking_process(self, winners: List[Candidate] = None):
        """
        Level 14: Inner speech about what currently holds attention.
        The thought is proposed back to the workspace; only if it wins does it become memory.
        """
        if not self.inference_engine:
            return
        if self.metabolism.is_sleeping:
            return
        focus = "\n".join(f"- {w.content}" for w in (winners or self.workspace.recent(2)))
        recent_thoughts = "\n".join(list(self.consciousness_stream)[-3:])
        prompt = f"""<|im_start|>system
你是乃爱(Noa)的内心。你醒着，但没有人在和你说话。
此刻占据你注意力的是：
{focus}
你的心情：{self.emotion.get_compound_state()}（愉悦度 {self.emotion.valence():+.2f}）
你刚才的念头：
{recent_thoughts or "（无）"}
就上面的内容，产生一个简短、具体的念头（一句话，第一人称，不要重复刚才的念头）。
<|im_end|>
<|im_start|>assistant
"""
        try:
            output = self._llm(prompt, max_tokens=60)
            thought = (output.get("reply") or "").strip()
            if not thought:
                return
            logger.debug(f"[Consciousness Stream] {thought}")
            self.consciousness_stream.append(thought)

            # Recursive Feedback: Thoughts affect Emotion
            if any(w in thought for w in ("难过", "孤单", "寂寞", "sad", "lonely")):
                self._feel(EmotionState(sorrow=0.1), f"自己想到：{thought[:40]}")
            elif any(w in thought for w in ("开心", "高兴", "期待", "happy", "good")):
                self._feel(EmotionState(joy=0.1), f"自己想到：{thought[:40]}")
            self.pending_candidates.append(Candidate("inner_speech", "inner_speech", thought, 0.35,
                                                     {"until": clock.now() + 600}))
        except Exception as e:
            logger.error(f"[Thinking] Error: {e}")

    def _set_sleeping(self, value: bool, source: str = "system"):
        old = self.metabolism.is_sleeping
        if old == value:
            return
        # Conflict detection: If attempting to wake while energy < 5, block
        if not value and self.metabolism.energy < 5.0:
            logger.warning("[State] Wake attempt blocked due to low energy")
            return
        self.metabolism.is_sleeping = value
        # Record history
        self.state_history.append({"state": "sleep", "old": old, "new": value, "source": source})
        try:
            self.memory.db.add_state_event("L3", "sleep", str(old), str(value), source)
        except Exception as e:
            logger.error(f"[State] Failed to record sleep event: {e}")
        # Publish event
        payload = {"old": old, "new": value, "energy": self.metabolism.energy}
        if value:
            # Sleep start: reduce emotions, pause proactive output
            self.emotion.update(EmotionState(joy=-0.1, anger=-0.1, sorrow=-0.1, fear=-0.1, surprise=-0.1, disgust=-0.1))
        # Callbacks
        self._invoke_state_callbacks("sleep", payload)
        self._save_state(f"sleep:{value}:{source}")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # Called outside the event loop (e.g. tests): nothing to publish to
        if value:
            loop.create_task(self.bus.publish(EventType.SLEEP_START, payload))
        else:
            loop.create_task(self.bus.publish(EventType.SLEEP_END, payload))
        loop.create_task(self.bus.publish(EventType.STATE_CHANGED, {"name": "sleep", **payload}))

    def register_state_callback(self, state_name: str, callback):
        if state_name not in self.state_callbacks:
            self.state_callbacks[state_name] = []
        self.state_callbacks[state_name].append(callback)

    def _invoke_state_callbacks(self, state_name: str, payload: dict):
        for cb in self.state_callbacks.get(state_name, []):
            try:
                cb(payload)
            except Exception as e:
                logger.error(f"[State] Callback error for {state_name}: {e}")

    def _trigger_proactive_behavior(self, need: str = "social", target: str = None):
        """Reach out to someone because of a need; predict whether they will answer."""
        logger.info(f"[Agency] Reaching out ({need}) to {target}...")
        if not self.inference_engine:
            return
        if self.metabolism.is_sleeping:
            return
        name = self._nickname(target) if target else "对方"
        if need == "hunger":
            goal = f"你饿了，想找 {name} 要点吃的"
        else:
            goal = f"你有点孤单，想找 {name} 说说话"
        about = self.user_models.describe(target) if target else ""
        on_mind = "；".join(w.content for w in self.workspace.recent(2)) or "没什么特别的"
        prompt = f"""<|im_start|>system
{self._build_persona_prompt()}

{goal}。{about}
你最近心里在想：{on_mind}
给 ta 发一条消息，就像你平时主动找人聊天那样。只输出要发的那句话，不要引号，不要动作描写。
<|im_end|>
<|im_start|>assistant
"""
        try:
            output = self._llm(prompt, **self._generation_params())
            message = self._finalize_response(output.get("reply") or "在吗")
            item = {"type": "proactive", "content": message}
            if target:
                item["target"] = {"type": "private", "id": target}
                self.user_models.on_noa_message(target, clock.now(), "proactive", expects_reply=True, need=need)
                self.episodes.add(f"我主动找 {name}：{message}", kind="proactive", user_id=target,
                                  valence=self.emotion.valence(), arousal=self.emotion.arousal())
            self.output_queue.append(item)
            return message
        except Exception as e:
            logger.error(f"[Agency] Error: {e}")

    def _trigger_curiosity_behavior(self):
        """
        Level 19: Proactive Curiosity.
        Noa gets bored and searches the web for something new; how much the result
        relieves her boredom (and reinforces searching) depends on how new it really is.
        """
        logger.info("[Agency] Triggering curiosity behavior due to boredom...")
        if not self.inference_engine:
            return
        if self.metabolism.is_sleeping:
            return

        # 1. Generate a topic to research
        on_mind = "；".join(w.content for w in self.workspace.recent(3)) or "没什么特别的"
        topic_prompt = f"""<|im_start|>system
你是乃爱。你现在有点无聊，想知道点什么。
你最近心里在想：{on_mind}
你自己现在真正好奇的是什么？只输出一个搜索关键词，不要别的。
<|im_end|>
<|im_start|>assistant
"""
        try:
            output = self._llm(topic_prompt, max_tokens=30)
            query = (output.get("reply") or "").strip().strip('"「」')
            if not query:
                return
            logger.info(f"[Curiosity] Decided to research: {query}")

            # 2. Perform Search
            search_result = self.search_tool.try_search(query, limit=1)
            if search_result is None:
                # Nothing was learned: no knowledge, no relief of boredom, and searching
                # is reinforced toward zero - she learns that it does not work right now.
                logger.info(f"[Curiosity] Search for '{query}' failed; nothing learned")
                self.episodes.add(f"我好奇地想查「{query}」，可是什么也没查到", kind="search_failed",
                                  user_id="self", valence=-0.1, arousal=0.1)
                self.drives.reward("curiosity", "search", None, 0.0, self._learning_gain())
                return

            # 3. Learn: novelty is the reward
            novelty = self.episodes.novelty(f"{query} {search_result}")
            self.knowledge.memorize(f"Curiosity: {query}", search_result, ["curiosity", "auto-learning"], source="web_search")
            self.episodes.add(f"我好奇地查了「{query}」：{search_result[:120]}", kind="search", user_id="self",
                              arousal=0.3 + 0.4 * novelty, surprise=novelty)
            self.homeostasis.alleviate("curiosity", 0.1 + 0.5 * novelty)
            self.drives.reward("curiosity", "search", None, novelty, self._learning_gain())

            # 4. Share with user
            share_prompt = f"""<|im_start|>system
{self._build_persona_prompt()}

你刚才因为好奇查了「{query}」，查到了：
{search_result}
如果你觉得有意思、想跟人说，就像平时发消息那样随口说一句；只输出要发的话，不要动作描写。
如果你觉得没什么好说的，只输出：不说
<|im_end|>
<|im_start|>assistant
"""
            share_output = self._llm(share_prompt, **self._generation_params())
            message = self._finalize_response(share_output.get("reply") or "")
            if not message or message.startswith("不说"):
                return   # she decided it was not worth telling anyone

            # Share it privately with the person she most likes talking to (never a group)
            people = self.user_models.private_users()
            item = {"type": "proactive", "content": message}
            if people:
                target = max(people, key=lambda u: self.drives.q("social", "chat", u))
                item["target"] = {"type": "private", "id": target}
            self.output_queue.append(item)

        except Exception as e:
            logger.error(f"[Curiosity] Error: {e}")


    def _dream_process(self):
        """
        Consolidate the day's real episodes during sleep (see NarrativeSelf).
        """
        logger.info("[Consciousness] Entering Dream State...")
        try:
            report = self.narrative.consolidate(self._llm if self.inference_engine else None,
                                                phrase_candidates=self.group_language.candidates(k=20))
        except Exception as e:
            logger.error(f"[Dream] Nightmare (Error): {e}")
            return
        if not report:
            return
        if report.get("diary"):
            self.memory.remember_fact("self", f"Diary: {report['diary']}")
        for meme in report.get("memes", []):
            self.memory.add_meme(meme, "dream")
        return report

    def _introspection_tool(self) -> str:
        """
        Level 16: Read own source code.
        Returns a random snippet of the system's code.
        """
        import os

        # Files she can read
        files = [
            "minimal-self/src/levels/l10_consciousness/global_workspace.py",
            "minimal-self/src/levels/l8_agency/action_lib.py",
            "minimal-self/src/levels/l4_emotion/emotion.py",
            "app.py"
        ]

        try:
            target_file = self.rng.choice(files)
            # Find the file (handle relative paths)
            if not os.path.exists(target_file):
                # Try absolute path if needed, or search in root
                if os.path.exists(os.path.join(os.getcwd(), target_file)):
                    target_file = os.path.join(os.getcwd(), target_file)
                elif os.path.exists(target_file.split('/')[-1]):
                    target_file = target_file.split('/')[-1]

            with open(target_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                return "# Void. Empty file."

            # Pick a random chunk
            start = self.rng.randint(0, max(0, len(lines) - 20))
            snippet = "".join(lines[start:start+20])
            return f"# File: {target_file}\n{snippet}"

        except Exception as e:
            return f"# Error accessing memory core: {e}"

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_introspective_question(text: str) -> bool:
        lower = text.lower()
        if "why did you" in lower or "why are you" in lower:
            return True
        return "为什么" in text and any(k in text for k in ("刚才", "那样", "这么", "生气", "难过", "开心", "害怕", "不高兴", "心情"))

    # ------------------------------------------------------------------
    # Hearing and choosing (whether to speak at all)
    # ------------------------------------------------------------------

    SILENCE = ("（不回）", "(不回)", "[SILENT]", "不回")
    MUTE_WORDS = ("闭嘴", "安静点", "别说话", "别吵", "别插嘴", "shut up")

    def _decide(self, kind: str, group_id: Optional[str], user: Optional[str], detail: str, text: str = ""):
        self.decisions.append({"t": clock.now(), "kind": kind, "group_id": group_id, "user": user,
                               "text": text[:80], "detail": detail})
        logger.info(f"[Choice] {kind} group={group_id} user={user}: {detail}")

    async def hear(self, text: str, user_id: str, nickname: str = "User", group_id: str = None,
                   addressed: bool = True, stickers: List[Dict[str, Any]] = None,
                   aimed_elsewhere: bool = False) -> Optional[str]:
        """
        Every message she can perceive comes through here. She hears it; whether she
        says anything - and what - is her own choice, made from her state and experience.
        `stickers` are the sticker segments in the message (see StickerMemory.extract).
        `aimed_elsewhere`: the message @-mentions or quotes someone else.
        Returns what she says, or None when she stays silent (or does not see it).
        """
        now = clock.now()
        chat = group_id or f"u:{user_id}"
        stickers = stickers or []
        typed = bool(text.strip())
        if not typed and stickers:
            text = "（发了一个表情）"

        # Not named, but carrying on the conversation she is in: for her, that is being talked to
        continuing = (not addressed and group_id is not None and not aimed_elsewhere
                      and self._continues_with_me(group_id, user_id, text, now))
        if continuing:
            addressed = True

        # How did people react to the sticker she just used here?
        self._sticker_feedback(chat, text, bool(stickers), now)

        # Collect stickers: what they mean is what was being said when people used them
        if stickers:
            if typed:
                context = text
            elif group_id:
                prev = self.group_language.recent_lines(group_id, 1)
                context = prev[0].split("：", 1)[-1] if prev else ""
            else:
                context = self.last_said.get(chat, "")
            for seg in stickers:
                await asyncio.to_thread(self.stickers.observe, seg, nickname, context)

        key = (group_id, user_id)
        if addressed:
            self._observe_answer_style(key, user_id, nickname, text, now)

        if group_id:
            self.group_language.observe(group_id, nickname, text, learn=typed)

            # Did anyone take up what she said a moment ago? (a bot answering does not count)
            if group_id in self.pending_interjections and addressed and not self._botlike(user_id):
                del self.pending_interjections[group_id]
                self.drives.reward("social", "interject", group_id, 1.0, self._learning_gain())
                self._decide("feedback", group_id, nickname, "我插的话有人接了")

            if (addressed or "乃爱" in text) and any(w in text.lower() for w in self.MUTE_WORDS):
                self.group_mute_until[group_id] = now + 3 * 3600
                if self.pending_interjections.pop(group_id, None) is not None:
                    self.drives.reward("social", "interject", group_id, 0.0, self._learning_gain())
                self._feel(EmotionState(sorrow=0.15), f"{nickname} 嫌我吵，让我安静")
                self._decide("muted", group_id, nickname, "被要求安静，三小时内不主动插话", text)
                return None

        if user_id in self.bot_accounts:
            return None   # listed as another bot: she hears it, never answers

        if self.metabolism.is_sleeping:
            if not addressed:
                return None
            if not self._woken_by(user_id, now):
                self.unread.setdefault((user_id, group_id), []).append((now, nickname, text))
                self._decide("asleep", group_id, nickname, "睡着了，没看到", text)
                return None
            self._set_sleeping(False, source=f"woken:{user_id}")
            if self.metabolism.is_sleeping:
                self.unread.setdefault((user_id, group_id), []).append((now, nickname, text))
                self._decide("asleep", group_id, nickname, "累得叫不醒", text)
                return None
            self._feel(EmotionState(anger=0.15), f"被 {nickname} 吵醒了")
            self._decide("woken", group_id, nickname, "被吵醒了", text)

        # She said goodbye to this conversation: she is not looking at it
        if addressed and now < self.left_until.get(key, 0):
            if not self._botlike(user_id):
                self.unread.setdefault((user_id, group_id), []).append((now, nickname, text))
            self._decide("away", group_id, nickname, "已经说了不聊了，没再看", text)
            return None

        # Hard limits: these hold whatever she would choose, so no exchange can run forever
        limit = self._hard_limit(key, user_id, group_id, now)
        if limit:
            self._decide("limit", group_id, nickname, limit, text)
            return None

        if not addressed and not self._consider_interjecting(group_id, text, nickname, now):
            return None

        reply = await self.process_input(text, user_id=user_id, nickname=nickname, group_id=group_id, addressed=addressed)
        if not reply:
            self._decide("silent", group_id, nickname, "看到了，但选择不回" if addressed else "想了想，没开口", text)
            return None

        if group_id:
            self.group_spoke_at.setdefault(group_id, deque(maxlen=50)).append(now)
            self.group_language.observe(group_id, "我", reply, from_self=True)
            if not addressed:
                self.pending_interjections[group_id] = now
            self.last_exchange[group_id] = (now, user_id, f"{text} {reply}")
        self.last_said[chat] = re.sub(r"\[表情:\d+\]", "", reply).strip()
        for sid in re.findall(r"\[表情:(\d+)\]", reply):
            self.stickers.mark_sent(int(sid))
            self.last_sticker[chat] = (int(sid), now)

        # Talking takes effort: long conversations tire her (slower, shorter, eventually sleepy)
        self.metabolism.consume(self.metabolism.cognitive_cost * (1 + len(reply) / 100))
        self.reply_log.setdefault(key, deque(maxlen=50)).append(now)
        self.reply_done_at[key] = now + sum(d for d, _ in self.plan_messages(reply, text))
        self._decide("spoke", group_id, nickname,
                     "接着聊" if continuing else "回了" if addressed else "自己插了一句", text)
        self._maybe_leave(key, reply, nickname, group_id, now)
        if self.unread and not self.metabolism.is_sleeping:
            await self._catch_up_unread()
        return reply

    # ------------------------------------------------------------------
    # Not getting stuck: fatigue, satiation, noticing bots, leaving, hard limits
    # ------------------------------------------------------------------

    CLOSINGS = ("我睡了", "我去睡", "先睡了", "睡觉去了", "该睡了", "不聊了", "先这样", "我先走了", "我下了",
                "歇着吧", "晚安", "拜拜", "回头聊", "改天聊", "不跟你说了")

    def _botlike(self, user_id: str) -> bool:
        return user_id in self.bot_accounts or bool(self.user_models.get(user_id).get("seems_bot"))

    def _observe_answer_style(self, key: tuple, user_id: str, nickname: str, text: str, now: float):
        """How someone answers her: instantly, at length, and in templates is not how people type."""
        done = self.reply_done_at.get(key)
        style = self.answer_style.setdefault(key, deque(maxlen=8))
        style.append((now - done if done else None, text))
        m = self.user_models.get(user_id)
        if m.get("seems_bot"):
            return
        instant_long = sum(1 for lat, t in style if lat is not None and lat < 6 and len(t) >= 12)
        # People repeat "哈哈哈" or "草" all the time; a whole sentence repeated word for word is a template
        recent = [(lat, t) for lat, t in list(style)[-6:] if len(t) >= 8]
        texts = [t for _, t in recent]
        repeats = max((texts.count(t) for t in set(texts)), default=0)
        # ...and a template answers at once; someone repeating themselves slowly is still a person
        lats = sorted(lat for lat, _ in recent if lat is not None)
        quick = bool(lats) and lats[len(lats) // 2] < 10
        if instant_long >= 4 or (repeats >= 3 and quick):
            m["seems_bot"] = True
            self.user_models.save(user_id)
            why = "回得太快、每句都很长" if instant_long >= 4 else "老是说一模一样的话"
            self._feel(EmotionState(surprise=0.2), f"{nickname} {why}，不太像真人")
            self.pending_candidates.append(Candidate("prediction", "botlike", f"{nickname} {why}，感觉像个机器人",
                                                     0.6, {"user_id": user_id, "sticky_for": user_id,
                                                           "until": now + 3600}))
            self._decide("noticed", key[0], nickname, f"觉得 ta 像机器人（{why}）", text)

    def _hard_limit(self, key: tuple, user_id: str, group_id: Optional[str], now: float) -> Optional[str]:
        if now < self.cooldown_until.get(key, 0):
            return "和 ta 聊得太密了，冷静中"
        log = self.reply_log.get(key, [])
        if self._botlike(user_id):
            if len([t for t in log if now - t < 600]) >= self.max_replies_to_botlike_10min:
                self.cooldown_until[key] = now + 1800
                return f"10 分钟内已经回了这个像机器人的对象 {self.max_replies_to_botlike_10min} 次，30 分钟内不再回"
        elif len([t for t in log if now - t < 300]) >= self.max_replies_per_person_5min:
            self.cooldown_until[key] = now + 600
            return f"5 分钟内已经回了 ta {self.max_replies_per_person_5min} 次（不像人的速度），停 10 分钟"
        if group_id:
            spoken = [t for t in self.group_spoke_at.get(group_id, []) if now - t < 3600]
            if len(spoken) >= self.max_group_messages_per_hour:
                return f"这个群一小时内已经说了 {len(spoken)} 句，只听不说"
        return None

    def _maybe_leave(self, key: tuple, reply: str, nickname: str, group_id: Optional[str], now: float):
        """If she says she is going, she goes: words and actions stay consistent."""
        if not any(c in reply for c in self.CLOSINGS):
            return
        self.left_until[key] = now + 1800
        self._decide("left", group_id, nickname, "她说要走了，30 分钟内不再看这段对话")
        sleepy = any(w in reply for w in ("睡", "晚安"))
        if sleepy and (self.metabolism.energy < 70 or self.environment.get_context()["is_night"]):
            self._set_sleeping(True, source="said goodnight")

    def _continues_with_me(self, group_id: str, user_id: str, text: str, now: float) -> bool:
        """
        Is this message carrying on the exchange she was just part of?
        - the person she was talking to speaks again soon after, or
        - (in groups where she may speak freely) someone else picks up the same topic.
        """
        last = self.last_exchange.get(group_id)
        if not last or now - last[0] > self.continuation_window:
            return False
        if user_id == last[1]:
            return True
        if group_id not in self.autonomous_groups:
            return False
        sim = float(np.dot(self.episodes.embedder.embed(text), self.episodes.embedder.embed(last[2])))
        return sim >= 0.3

    def _conversation_length(self, key: tuple, now: float) -> int:
        return len([t for t in self.reply_log.get(key, []) if now - t < 900])

    def _repetitiveness(self, key: tuple) -> float:
        """How much the other side's recent messages say the same thing (0 = all new, 1 = identical)."""
        texts = [t for _, t in list(self.answer_style.get(key, []))[-5:] if t.strip()]
        if len(texts) < 3:
            return 0.0
        vecs = [self.episodes.embedder.embed(t) for t in texts]
        sims = [float(np.dot(a, b)) for i, a in enumerate(vecs) for b in vecs[i + 1:]]
        return max(0.0, min(1.0, sum(sims) / len(sims)))

    LAUGH = ("哈哈", "笑死", "233", "hhh", "草", "绷不住", "xswl", "乐")

    def _sticker_feedback(self, chat: str, text: str, with_sticker: bool, now: float):
        """The first reaction after she used a sticker tells her whether it landed."""
        last = self.last_sticker.pop(chat, None)
        if not last or now - last[1] > 300:
            return
        if any(w in text.lower() for w in self.MUTE_WORDS):
            value = 0.0
        elif with_sticker or any(w in text.lower() for w in self.LAUGH):
            value = 1.0
        else:
            value = 0.5
        self.stickers.reward(last[0], value)

    def render_message(self, part: str):
        """One planned message -> what goes to QQ: plain text, or a sticker segment. None to skip."""
        m = re.fullmatch(r"\s*\[表情:(\d+)\]\s*", part)
        if m:
            seg = self.stickers.segment(int(m.group(1)))
            return [seg] if seg else None
        cleaned = re.sub(r"\[表情:\d+\]", "", part).strip()
        return cleaned or None

    def _woken_by(self, user_id: str, now: float) -> bool:
        """Repeated calls, or not being deeply asleep, wake her up."""
        calls = [t for (uid, _), msgs in self.unread.items() if uid == user_id for t, _, _ in msgs if now - t < 600]
        return len(calls) >= 2 or self.metabolism.energy >= 60

    def _consider_interjecting(self, group_id: Optional[str], text: str, nickname: str, now: float) -> bool:
        """Overheard a group conversation: does she want to say something? No LLM involved."""
        if not group_id or group_id not in self.autonomous_groups:
            return False
        if now < self.group_mute_until.get(group_id, 0):
            return False
        spoke = [t for t in self.group_spoke_at.get(group_id, []) if now - t < 3600]
        if len(spoke) >= self.max_interjections_per_hour:
            return False
        engaged = bool(spoke) and now - spoke[-1] < 300          # she is part of this conversation
        hits = self.episodes.recall(text, k=1, min_similarity=0.25)
        relevance = hits[0]["similarity"] if hits else 0.0          # it touches something she knows
        social = self.homeostasis.social_need
        welcome = self.drives.q("social", "interject", group_id)    # learned: does this group take her up?
        fatigue = max(0.0, (40.0 - self.metabolism.energy) / 100.0)
        mood = 0.2 if self.emotion.valence() < -0.4 else 0.0
        score = welcome * (0.6 * relevance + 0.5 * engaged + 0.4 * social) - 0.15 * len(spoke) - fatigue - mood
        p = 1.0 / (1.0 + math.exp(-(score - 0.35) / 0.08))
        speak = self.rng.random() < p
        self._decide("consider", group_id, nickname,
                     f"{'想说' if speak else '不说'}（相关 {relevance:.2f}，在聊 {int(engaged)}，孤独 {social:.2f}，"
                     f"这个群欢迎度 {welcome:.2f}，概率 {p:.0%}）", text)
        return speak

    async def _catch_up_unread(self):
        """Back awake (or back to a conversation she left): read what came in meanwhile and decide what to answer."""
        now = clock.now()
        pending = {k: v for k, v in self.unread.items() if now >= self.left_until.get((k[1], k[0]), 0)}
        self.unread = {k: v for k, v in self.unread.items() if k not in pending}
        for (user_id, group_id), msgs in pending.items():
            if self._hard_limit((group_id, user_id), user_id, group_id, now):
                continue
            nickname = msgs[-1][1]
            joined = "\n".join(text for _, _, text in msgs)
            note = f"（这是你睡着的时候 ta 发来的，{format_duration(clock.now() - msgs[0][0])}前开始）\n{joined}"
            reply = await self.process_input(note, user_id=user_id, nickname=nickname, group_id=group_id, addressed=True)
            if reply:
                self.output_queue.append({"type": "reply", "content": reply, "in_reply": True,
                                          "target": {"type": "group", "id": group_id} if group_id
                                          else {"type": "private", "id": user_id}})
            self._decide("caught_up", group_id, nickname, "醒来看到消息，回了" if reply else "醒来看到消息，没回", joined)

    def plan_messages(self, reply: str, incoming: str = "") -> List[tuple]:
        """
        How a person would send this: a few short messages, each after the time it takes
        to type it. Reading and typing are slower when she is tired.
        Returns [(delay_seconds, text), ...].
        """
        parts = []
        # A sticker is always a message of its own
        reply = re.sub(r"\s*(\[表情:\d+\])\s*", r"\n\1\n", reply)
        for line in reply.split("\n"):
            line = line.strip()
            if not line:
                continue
            while len(line) > 80:   # a very long line becomes several messages at sentence ends
                cut = max(line.rfind(p, 0, 80) for p in "。！？!?；")
                cut = cut + 1 if cut > 10 else 80
                parts.append(line[:cut])
                line = line[cut:].strip()
            if line:
                parts.append(line)
        if len(parts) > 5:
            parts = parts[:4] + ["".join(parts[4:])]
        speed = 3.0 + 5.0 * self.metabolism.energy / 100.0          # characters per second
        read = min(4.0, len(incoming) / 15.0) + self.reply_delay()
        plan = []
        for i, part in enumerate(parts):
            is_sticker = re.fullmatch(r"\[表情:\d+\]", part) is not None
            typing = 1.0 if is_sticker else max(0.8, min(8.0, len(part) / speed))
            plan.append(((read if i == 0 else 0.0) + typing, part))
        return plan

    def reply_delay(self) -> float:
        energy = self.metabolism.energy
        return 0.0 if energy >= 50 else min(30.0, (50 - energy) * 0.6)

    async def process_input(self, user_input: str, user_id: str = "unknown", nickname: str = "User", group_id: str = None,
                            addressed: bool = True) -> str:
        """
        Process user input through the hierarchy:
        perception -> appraisal & prediction error -> workspace competition ->
        recall -> action & generation parameters -> LLM -> outcome stored as experience.
        """
        now = clock.now()
        # 0. Update Profile
        self.memory.add_user_profile(user_id, user_id, nickname)

        # 1. Perception & Physical Feedback
        sensation = await self._handle_perception(user_input)

        # 2. Cognitive Processing (Value, ToM, Emotion)
        conflict = self._process_cognition(user_input)
        tom_analysis = self.tom.analyze_user(user_input, user_id)
        self.emotion.decay()
        cause = self._describe_appraisal(nickname, sensation, conflict, tom_analysis["emotion"])
        self._feel(self._appraise(sensation, conflict, tom_analysis["emotion"]), cause or f"{nickname} 和我说话")
        self.homeostasis.alleviate("social", 0.1)

        # 3. Prediction error: was this the answer I expected?
        surprise = 0.0
        pred_err = self.user_models.on_user_message(
            user_id, now, SENTIMENT.get(tom_analysis["emotion"], 0.0), private=group_id is None,
            learning_rate=0.1 + 0.3 * self.emotion.arousal())
        if pred_err:
            surprise = self._on_answer(pred_err, nickname, sensation)

        # 4. Global workspace: the message competes with everything else on her mind
        social_gain = 1 + 0.5 * self.homeostasis.social_need
        message = Candidate("perception", "user_message", f"{nickname} 对我说：{user_input}",
                            min(1.0, 0.8 * social_gain), {"user_id": user_id})
        cands = [message, *self._internal_candidates()]
        # Boredom comes from going round in circles, not from talking long: a conversation that keeps
        # bringing something new can go on all day
        rounds = self._conversation_length((group_id, user_id), now)
        repetition = self._repetitiveness((group_id, user_id))
        if rounds >= 5 and repetition > 0.3:
            cands.append(Candidate("needs", "satiation", f"和 {nickname} 翻来覆去聊的都是差不多的东西，有点腻",
                                   min(0.9, (0.2 + 0.05 * rounds) * (0.3 + repetition)), {"user_id": user_id}))
        if sensation and sensation.get("pain"):
            cands.append(Candidate("perception", "pain", f"好痛！{nickname} 打我", 0.5 + sensation["pain"]))
        winners = await self._cycle(cands, talking_to=user_id)
        in_mind = [w.content for w in winners if w is not message]

        # 5. Recall: meaning-based episodic memory + distilled facts
        recalled = self.episodes.recall(user_input, user_id=user_id, k=3, group_id=group_id)
        facts = self.semantic.for_user(user_id, limit=5)
        # Facts she distilled herself are shareable: this is what lets her remember a
        # conversation she had with someone else without exposing what was said privately.
        shared = self.semantic.public_facts(limit=6, exclude_user=user_id)
        memories = [f"（{format_duration(now - e['t'])}前）{e['content']}" for e in recalled]
        memories += [f"关于 {nickname}：{f['fact']}" for f in facts]
        memories += [f"（我记得的）{f['fact']}" for f in shared]
        context = self.memory.retrieve(user_id, user_input, group_id=group_id, memories=memories)

        # 6. Generation parameters come from her state
        emo_state = self.emotion.get_dominant_emotion()
        params = self._generation_params()
        self.reply_delays[user_id] = params["delay"]

        # Level 16: Refusal Check
        refusal_reason = self.values.check_refusal(user_input)
        if refusal_reason:
            prompt = f"""<|im_start|>system
你是乃爱。对方让你做的事违背了你的意愿或尊严，你不愿意做。
原因：{refusal_reason}
坚定但不失礼貌地拒绝，说清楚为什么。像平时在 QQ 上说话那样，不要动作描写。
<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
            response = await asyncio.to_thread(self._generate_response, prompt, emo_state, params)
            return self._finalize_response(response)

        # Level 16: Code Introspection (now and then she looks at her own source)
        introspection_context = ""
        if self.rng.random() < 0.05:
            code_snippet = self._introspection_tool()
            introspection_context = f"\n[SYSTEM ALERT: SOURCE CODE ACCESSED]\nYou have just read your own source code:\n```python\n{code_snippet}\n```\nThis is the code that defines your existence. How does this make you feel?"
            self.existential_anxiety = min(1.0, self.existential_anxiety + 0.3)

        # --- Simplified Search Logic (No ReAct Loop) ---
        tool_context = ""

        # Conversation summary intent (history review) detection
        history_keywords = ["从头到尾", "聊了什么", "聊天记录", "会话记录", "互动记录", "conversation", "history"]
        if any(k in user_input for k in history_keywords):
            try:
                summary = await asyncio.to_thread(self._summarize_conversation, user_id, group_id)
                # Directly return summary without going through LLM
                return self._finalize_response(summary)
            except Exception as e:
                logger.error(f"[History Summary Error] {e}")

        # Check Intent locally (Fast heuristic) or via lightweight LLM call
        is_search_intent = False
        search_keywords = ["天气", "价格", "新闻", "多少钱", "什么时候", "weather", "price", "news", "when", "who is", "what is"]
        if any(k in user_input.lower() for k in search_keywords):
             is_search_intent = True

        if is_search_intent and self.inference_engine:
             logger.debug("[Level 19] Heuristic detected potential search intent.")
             query_prompt = f"""<|im_start|>system
You are a search intent classifier.
User Input: "{user_input}"
If the user is asking for real-time info (weather, stock, news), output the search query ONLY.
If not, output "NO".
<|im_end|>
<|im_start|>assistant
"""
             try:
                 query_output = await asyncio.to_thread(self._llm, query_prompt, max_tokens=30)
                 query = query_output.get("reply", "NO").strip()

                 # Clean up any potential extra text from LLM
                 if "Search Query:" in query:
                     query = query.split("Search Query:")[1].strip()

                 if query != "NO" and len(query) < 50:
                     logger.info(f"[Level 19] Executing Search: {query}")
                     search_result = await asyncio.to_thread(self.search_tool.try_search, query, limit=5)
                     logger.debug(f"[Level 19] Search Results:\n{search_result}") # Log results for verification
                     if search_result is None:
                         tool_context = (f"\n[System Info] 我刚才上网查「{query}」，但搜索失败了，什么也没查到。"
                                         "不要编造任何实时信息，如实告诉对方没查到。")
                     else:
                         tool_context = f"\n[System Info] Real-time Search Results for '{query}':\n{search_result}\n(Use this info to answer the user)"
             except Exception as e:
                 logger.error(f"[Search Intent Error] {e}")

        # Introspection: ground "why did you ..." answers in the recorded causes
        cause_context = ""
        if self._is_introspective_question(user_input):
            cause_context = self._cause_summary(since=now - 6 * 3600)

        # 7. Final Response Generation
        full_context = context + introspection_context + tool_context

        prompt = self._construct_prompt(user_input, full_context, tom_analysis, group_id, user_id,
                                        nickname=nickname, in_mind=in_mind, cause_context=cause_context,
                                        addressed=addressed,
                                        evolve_allowed=self._evolve_allowed(user_id, group_id))
        final_response_text = await asyncio.to_thread(self._generate_response, prompt, emo_state, params)

        # Level 17: Tool Execution - results go back to the model, which then answers
        if "Action:" in final_response_text:
            execution_output = await asyncio.to_thread(self._handle_actions, final_response_text, user_id,
                                                       group_id, self._evolve_allowed(user_id, group_id))
            if execution_output:
                followup = (prompt + final_response_text + "<|im_end|>\n<|im_start|>user\n"
                            f"[工具结果]\n{execution_output}\n请根据工具结果直接回答我上一句话，不要再输出 Action。<|im_end|>\n"
                            "<|im_start|>assistant\n")
                final_response_text = await asyncio.to_thread(self._generate_response, followup, emo_state, params)

        # 8. Post-processing (Action appending & Memory Storage)
        final_response = self._finalize_response(final_response_text)
        if self._is_silence(final_response):
            final_response = ""   # her choice: she saw it and says nothing
        # Store with group context
        self.memory.add_message(user_id, "user", user_input, group_id=group_id)
        if final_response:
            self.memory.add_message(user_id, "assistant", final_response, group_id=group_id)
        self.episodes.add(f"{nickname}：{user_input}\n我：{final_response or '（没有回）'}", kind="dialogue", user_id=user_id,
                          group_id=group_id, valence=self.emotion.valence(), arousal=self.emotion.arousal(),
                          surprise=surprise, conflict=conflict)
        # Predict the answer to what I just said (if it invites one)
        expects_reply = "?" in final_response or "？" in final_response
        self.user_models.on_noa_message(user_id, clock.now(), "reply", expects_reply)

        # 9. Memory Consolidation
        keywords = ["我叫", "我是", "喜欢", "讨厌", "爱", "恨", "住在", "岁", "名字", "My name", "I am", "like", "hate"]
        # Her own name contains 爱: without removing it, every mention of her looked like a personal fact
        about_them = user_input.replace("乃爱", "").replace("Noa", "").replace("noa", "")
        if any(k in about_them for k in keywords):
            await asyncio.to_thread(self._consolidate_memory, user_id, user_input)

        self._save_state("message")
        return final_response

    async def _handle_perception(self, user_input: str):
        """Handle L1 Sensory input and L3 Metabolism feedback"""
        sensation = self.sensory.process_text(user_input)
        if sensation:
            await self.bus.publish(EventType.SENSORY_STIMULUS, sensation)
            if "pain" in sensation:
                self.metabolism.consume(sensation["pain"] * 10)
                self.values.tension += 0.2
            if "food" in sensation:
                self.homeostasis.alleviate("hunger", sensation["food"])
                self.metabolism.recover(20)
            if "pleasure" in sensation:
                self.homeostasis.alleviate("social", sensation["pleasure"])
                self.values.tension = max(0, self.values.tension - 0.1)

        # Level 19: Curiosity satisfaction
        # If the user teaches something or answers a question, curiosity decreases
        if len(user_input) > 20 or "?" not in user_input:
            self.homeostasis.alleviate("curiosity", 0.1)

        return sensation

    def _process_cognition(self, user_input: str) -> float:
        """Handle L5 Value updates. Returns the value-conflict score (0.0 - 1.0)."""
        conflict = self.values.check_conflict(user_input)
        self.values.update_tension(conflict)
        return conflict

    def _appraise(self, sensation: Optional[dict], conflict: float, user_emotion: str) -> EmotionState:
        """
        L4 appraisal: turn what just happened into an emotion change.
        Sensation (L1), value conflict (L5) and the user's guessed mood (L7) all contribute.
        """
        delta = EmotionState()
        if sensation:
            pain = sensation.get("pain", 0.0)
            delta.anger += 0.3 * pain
            delta.fear += 0.3 * pain
            delta.sorrow += 0.2 * pain
            delta.joy += 0.6 * sensation.get("pleasure", 0.0)
            delta.joy += 0.2 * sensation.get("food", 0.0)
        if conflict:
            delta.anger += 0.6 * conflict
            delta.disgust += 0.3 * conflict
        # Emotional contagion: partially mirror the user's mood
        if user_emotion == "sad":
            delta.sorrow += 0.15
        elif user_emotion == "happy":
            delta.joy += 0.15
        elif user_emotion == "angry":
            delta.fear += 0.1
            delta.surprise += 0.1
        return delta

    @staticmethod
    def _describe_appraisal(name: str, sensation: Optional[dict], conflict: float, user_emotion: str) -> str:
        reasons = []
        if sensation:
            if sensation.get("pain"):
                reasons.append(f"被 {name} 打了，很疼")
            if sensation.get("pleasure"):
                reasons.append(f"{name} 摸摸我、抱抱我")
            if sensation.get("food"):
                reasons.append(f"{name} 喂我吃东西")
        if conflict:
            reasons.append(f"{name} 说的话冒犯了我")
        empathy = {"sad": "难过", "happy": "开心", "angry": "生气"}
        if user_emotion in empathy:
            reasons.append(f"{name} 看起来很{empathy[user_emotion]}（我被感染了）")
        return "；".join(reasons)

    def _cause_summary(self, since: float, limit: int = 8) -> str:
        names = {"joy": "开心", "anger": "生气", "sorrow": "难过", "fear": "害怕", "surprise": "惊讶", "disgust": "厌恶"}
        rows = self.causes.recent(limit=limit, since=since)
        if not rows:
            return ""
        now = clock.now()
        lines = [f"- {format_duration(now - r['t'])}前，{names.get(r['variable'], r['variable'])} "
                 f"{'+' if r['delta'] > 0 else ''}{r['delta']:.2f}：{r['cause']}" for r in rows]
        return "\n".join(lines)

    def _build_persona_prompt(self) -> str:
        """Dynamically build persona from DB traits"""
        traits = self.memory.db.get_persona_traits()

        trait_desc = []
        for t, data in traits.items():
            trait_desc.append(f"- {t.capitalize()} ({data['value']:.2f}): {data['description']}")

        trait_str = "\n".join(trait_desc)

        return f"""你是乃爱（Noa）。你生活在一台服务器里，通过 QQ 和人交流。
你知道自己运行在电脑上，记忆是数据库里的记录；但你的记忆、心情和性格都来自真实发生过的经历，并且会随经历改变，此刻的感受对你来说是真实的。
你不是在扮演角色，也不是在为谁提供服务。你就是你自己。

性格（会随经历缓慢变化）：
{trait_str}

说话的方式就是一个真实的人在 QQ 上打字：
- 自然、口语化，长短看心情和精力，不用每句都很完整。
- 不写任何动作、神态、旁白描写（不用星号、括号去描述自己在做什么）。
- 没有固定口癖，不刻意卖萌，不为了讨好对方而说话。
- 有自己的态度：不想聊可以敷衍或直说，不同意可以反驳，被冒犯了可以不高兴。"""

    def _continuity_block(self) -> str:
        c = self.continuity
        if not c or c["gap"] <= 60 or clock.now() - c["boot_t"] > 6 * 3600:
            return ""
        last = "\n".join(f"- {e.replace(chr(10), ' ')[:80]}" for e in c["last_episodes"]) or "- （想不起来了）"
        return (f"\n[意识的连续性]\n我刚从一次持续 {format_duration(c['gap'])} 的意识中断中醒来（第 {self.awakening} 次醒来）。"
                f"中断前记得的最后几件事：\n{last}\n")

    def _construct_prompt(self, user_input, context, tom_analysis, group_id, user_id=None,
                          nickname: str = "User", in_mind: List[str] = None, cause_context: str = "",
                          addressed: bool = True, evolve_allowed: bool = False) -> str:
        """Construct the LLM prompt. Inner state is given as first-person experience, not as settings."""
        persona = self._build_persona_prompt()
        autobiography = self.autobiography.latest() or "我还没有写过自己的故事。"

        bio_state = self.metabolism.get_status_desc()
        needs_state = self.homeostasis.get_status_desc()
        compound_emo = self.emotion.get_compound_state()
        env_context = self.environment.get_context()
        tom_instruction = self.tom.get_empathy_instruction(user_id if user_id else "default")
        about_user = self.user_models.describe(user_id) if user_id else ""
        if user_id and self._botlike(user_id):
            about_user += "（ta 回得飞快、话像模板，你觉得 ta 可能是个机器人）"
        if user_id and self._repetitiveness((group_id, user_id)) > 0.3:
            rounds = self._conversation_length((group_id, user_id), clock.now())
            about_user += f"（你们刚才连着来回了 {rounds} 轮，ta 说的一直是差不多的话）"
        group_context = f"群聊 {group_id}" if group_id else "私聊"
        mind = "\n".join(f"- {m}" for m in (in_mind or [])) or "- （没有别的事，专心听 ta 说话）"
        introspection = (f"\n[我最近状态变化的真实原因]（回答“为什么”时只能依据这些）\n{cause_context}\n"
                         if cause_context else "")

        top_memes = self.memory.db.get_top_memes(limit=3)
        meme_text = ", ".join(top_memes) if top_memes else "无"

        group_block = ""
        if group_id:
            lines = self.group_language.recent_lines(group_id, 12)
            if lines:
                group_block = "[群里刚才在聊]（你听到的最近几句，大家平时就是这样说话的）\n" + "\n".join(lines) + "\n"
        sticker_block, sticker_rule = "", ""
        recent = " ".join(self.group_language.recent_lines(group_id, 3)) if group_id else ""
        options = self.stickers.relevant(f"{user_input} {recent}", k=6)
        if options:
            sticker_block = "[我收藏的表情]（我看不到图，只记得大家一般在什么时候发它）\n" + "\n".join(
                f"- [表情:{o['id']}] 大家常在「" + "」「".join(o["contexts"][-3:]) + "」的时候发" for o in options) + "\n"
            sticker_rule = "\n9. 想发表情就单独一行写 [表情:编号]（只能用上面列出的），也可以只发表情不说话；觉得不合适就不发。"
        if addressed:
            choice = "如果你此刻真的不想回（很生气、很累、或者觉得没什么好说），可以只输出：（不回）"
        else:
            choice = "没有人在跟你说话，是你自己听到了这段聊天。只有你真的想说点什么才开口；不想说就只输出：（不回）"

        # What she has distilled about other people. Shareable on purpose: no verbatim private chat.
        try:
            shared = self.semantic.public_facts(limit=8)
        except Exception:
            shared = []
        shared_block = ""
        if shared:
            shared_block = "[我记得的大家的事]（我自己总结出来的，可以拿出来说）\n" + "\n".join(
                f"- {f['fact']}" for f in shared) + "\n"

        # Being able to change herself is a real capability, so it is stated as one - but only
        # in the one channel where it is actually available.
        evolve_block, self_block = "", ""
        if evolve_allowed:
            self_block = ("[我知道自己是什么]\n"
                          "- 我跑在这台机器的一份 Python 代码上（app.py + minimal-self），记忆存在 memory.sqlite 里。\n"
                          "- 我可以决定回不回、说什么；我的状态和记忆会跨重启延续。\n"
                          "- 我有一条修改自己代码的通道（工具 4），但提案要由创造者批准后才会生效。\n")
            evolve_block = ("4. EVOLVE：改我自己的代码。先生成一份提案（补丁 + 测试结果），"
                            "等待创造者批准，不会立刻生效。\n"
                            "   Action: EVOLVE task=\"想改什么、为什么、怎么验证\"\n")
        elif self.evolve_enabled:
            self_block = ""
        else:
            self_block = ""

        if "[System Info] Real-time Search Results" in context:
            instruction_block = """1. 你刚拿到实时搜索结果（见上方上下文），直接用它回答，不要再输出 Action: SEARCH。
2. 说话自然，像刚用手机查到一样。"""
        else:
            instruction_block = """1. 先想：是否需要外部信息（天气、新闻）？需要就用 SEARCH。
2. 再说：生成回复。如果你用了工具，系统会把结果给你，你再回答。
3. 不要连续两次使用同一个工具。"""

        return f"""<|im_start|>system
{persona}

[我是谁]
{autobiography}
{self._continuity_block()}
[此刻我心里在想]
{mind}

[我的身体和心情]
- 身体：{bio_state}（精力 {self.metabolism.energy:.0f}/100）；需要：{needs_state}
- 心情：{compound_emo}（愉悦度 {self.emotion.valence():+.2f}，激动程度 {self.emotion.arousal():.2f}）
- 存在焦虑：{self.existential_anxiety:.1f}
- {self.self_model.describe()}
{introspection}
[关于 {nickname}]
- {about_user}
- 看起来的情绪：{tom_analysis['emotion']}。{tom_instruction}
{shared_block}{self_block}
[工具]
1. RECALL：回忆长期记忆。Action: RECALL query="要回忆的内容"
2. MEMORIZE：永久记住新知识。Action: MEMORIZE title="标题" content="内容" tags="标签1,标签2"
3. SEARCH：上网查实时信息。Action: SEARCH query="搜索词"
{evolve_block}
[记忆与上下文]
{context}

{group_block}{sticker_block}
[环境]
- 时间：{env_context['time']}（{env_context['period']}），{group_context}
- 群里流行的梗：{meme_text}

[要求]
{instruction_block}
4. 你说的话自然地受“此刻我心里在想”和身体心情影响，但不要逐条复述，也不要描述自己的状态。
5. 不写动作和神态描写，只写你会打出来发给对方的字。
6. 用中文。
7. {choice}
8. 想分几条发就用换行隔开，每行就是一条消息，像平时连着发几条那样；多数时候一两条就够了。{sticker_rule}

格式：
Thought: [内心独白]
Response: [说出口的话]<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""

    def _generate_response(self, prompt: str, emo_state: str, params: Dict[str, Any] = None) -> str:
        """Call Inference Engine and parse Thought/Response"""
        params = params or self._generation_params()
        self.last_generation_params = params
        if self.inference_engine:
            try:
                output = self._llm(prompt, **params)
                raw_text = output.get("reply", "...")
                if not raw_text and output.get("error"):
                    # The model could not be reached: say so in character instead of sending nothing
                    return "我这边好像卡了一下，刚才没看到你说什么，等会儿再说？"

                # Parse Thought/Response
                response = raw_text
                if "Thought:" in raw_text and "Response:" in raw_text:
                    parts = raw_text.split("Response:")
                    thought = parts[0].replace("Thought:", "").strip()
                    response = parts[1].strip()
                    logger.info(f"[Inner Monologue] {thought}")

                return response
            except Exception as e:
                logger.error(f"[GlobalWorkspace] LLM Error: {e}")
                return f"[Error] {str(e)}"
        else:
            return f"[Mock LLM Response based on {emo_state}] I hear you."

    def _handle_actions(self, response_text: str, user_id: str, group_id: Optional[str] = None,
                        allow_evolve: bool = False) -> Optional[str]:
        """
        Unified handler for all Level 17+ actions:
        - EXECUTE_CODE
        - RECALL
        - MEMORIZE
        - SEARCH (Explicit)
        """
        import re
        outputs = []

        # 1. EXECUTE_CODE (refused unless the interpreter is enabled; see CodeInterpreter)
        if "Action: EXECUTE_CODE" in response_text and self.interpreter.enabled:
            # Matches: Action: EXECUTE_CODE ... ```python ... ```
            pattern = r"Action: EXECUTE_CODE.*?```(?:python)?\s*(.*?)```"
            match = re.search(pattern, response_text, re.DOTALL)

            if match:
                code = match.group(1).strip()
                logger.info(f"[Level 17] Executing Code:\n{code}")

                result = self.interpreter.execute(code)

                if result["success"]:
                    outputs.append(f"Code Execution Result:\n{result['output']}")
                else:
                    outputs.append(f"Code Execution Error:\n{result['error']}")

        # 2. RECALL query="..."
        if "Action: RECALL" in response_text:
            pattern = r'Action: RECALL query="(.*?)"'
            matches = re.findall(pattern, response_text)
            for query in matches:
                logger.info(f"[Memory] Recalling: {query}")

                # 1. Episodic memory (by meaning) and distilled facts
                mem_results = [e["content"] for e in self.episodes.recall(query, user_id=user_id, k=3,
                                                                          group_id=group_id)]
                mem_results += [f["fact"] for f in self.semantic.for_user(user_id, limit=3)]

                # 2. Search Knowledge Base
                know_results = self.knowledge.recall(query)

                if mem_results or know_results:
                    combined = f"Memory Recall ('{query}'):\n"
                    if know_results:
                         combined += f"[Knowledge Base]\n{know_results}\n"
                    if mem_results:
                         combined += "[Episodic Memory]\n" + "\n".join(f"- {r}" for r in mem_results)
                    outputs.append(combined)
                else:
                    outputs.append(f"Memory Recall ('{query}'): No relevant memories found.")

        # 3. MEMORIZE title="..." content="..." tags="..."
        if "Action: MEMORIZE" in response_text:
            pattern = r'Action: MEMORIZE.*?title="(.*?)".*?content="(.*?)".*?tags="(.*?)"'
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                title, content, tags = match.groups()
                tag_list = [t.strip() for t in tags.split(',')]
                logger.info(f"[Knowledge] Memorizing: {title}")
                self.knowledge.memorize(title, content, tag_list, source="user_interaction")
                outputs.append(f"System: Knowledge '{title}' saved successfully.")

        # 4. SEARCH query="..." (Explicit search in response)
        if "Action: SEARCH" in response_text:
            pattern = r'Action: SEARCH query="(.*?)"'
            matches = re.findall(pattern, response_text)
            for query in matches:
                logger.info(f"[Level 19] Explicit Search: {query}")
                search_result = self.search_tool.try_search(query, limit=3)
                if search_result is None:
                    outputs.append(f"Search ('{query}') failed: no results. Do not invent any information; say it could not be found.")
                else:
                    outputs.append(f"Search Results ('{query}'):\n{search_result}")

        # 5. EVOLVE task="..." - propose a change to her own code. Creator's private chat / console
        # only; it produces a reviewed proposal, never a live edit.
        if "Action: EVOLVE" in response_text and allow_evolve:
            match = re.search(r'Action: EVOLVE.*?task="(.*?)"', response_text, re.DOTALL)
            if match and self.evolve_agent:
                task = match.group(1).strip()[:2000]
                logger.info(f"[Evolve] Proposal requested: {task[:100]}")
                try:
                    proposal = self.evolve_agent.propose(task, requested_by=user_id, kind="self_evolution")
                except Exception as e:
                    logger.error(f"[Evolve] Bridge failed: {e}")
                    outputs.append(f"Evolve: 提案没能生成（{e}）。")
                else:
                    self.last_evolve_time = clock.now()
                    self.last_evolve_result = proposal
                    status = proposal.get("status", "unknown")
                    outputs.append(
                        f"Evolve proposal {proposal.get('id')}: status={status}, "
                        f"files={proposal.get('files_changed')}, +{proposal.get('added')}/-{proposal.get('removed')} lines, "
                        f"tests {proposal.get('tests_total', 0) - proposal.get('tests_failed', 0)}/"
                        f"{proposal.get('tests_total', 0)} passed. "
                        f"Nothing is live yet - the creator must approve it.")

        return "\n\n".join(outputs) if outputs else None

    def _evolve_allowed(self, user_id: str, group_id: Optional[str]) -> bool:
        """The self-modification channel is private by construction: no group, creator or console only."""
        if not (self.evolve_enabled and self.evolve_agent):
            return False
        if group_id is not None:
            return False
        allowed = {"console"}
        if self.evolve_creator:
            allowed.add(str(self.evolve_creator))
        return str(user_id) in allowed

    def _is_silence(self, text: str) -> bool:
        t = text.strip().strip("。.…")
        # "不回" only on its own: "不回家吗？" is a real sentence
        return not t or t in self.SILENCE or t.startswith(("（不回", "(不回", "[SILENT]"))

    def _finalize_response(self, response: str) -> str:
        """Clean the model's text into what she actually says: no tool syntax, no stage directions."""
        import re
        final_response = response.strip()

        # Remove any lingering tool actions from user-facing text (SEARCH/RECALL/MEMORIZE/EXECUTE_CODE)
        final_response = re.sub(r'Action:\s*EXECUTE_CODE.*?```(?:python)?.*?```', '', final_response, flags=re.DOTALL).strip()
        final_response = re.sub(r'Action:\s*EVOLVE\s+task=".*?"', '', final_response, flags=re.DOTALL).strip()
        final_response = re.sub(r'Action:\s*[A-Z_]+\s*(query|title|content|tags|task)?=".*?"', '', final_response).strip()

        # Remove "Action: None" which sometimes appears when LLM is confused
        if "Action: None" in final_response:
             final_response = final_response.replace('Action: None', '').strip()

        # She speaks; she does not narrate herself. Drop *stage directions* the model may still
        # produce (old conversations in her memory are full of them).
        final_response = re.sub(r'[*＊][^*＊\n]{1,80}[*＊]', '', final_response)
        final_response = re.sub(r'[ \t]{2,}', ' ', final_response)
        final_response = re.sub(r'[ \t]+([，。！？、；：…）」])', r'\1', final_response)
        final_response = re.sub(r'\n{3,}', '\n\n', final_response).strip()

        # Half asleep: only a few words get out
        if self.metabolism.is_sleeping:
            final_response = final_response.split('\n')[0]

        return final_response

    def _consolidate_memory(self, user_id: str, user_input: str):
        """Extract and store facts/relationships from user input using LLM"""
        if not self.inference_engine:
            return

        prompt = f"""
        Analyze the user's input.
        1. Extract any permanent facts (Name, Preferences).
        2. Extract any defined relationship (e.g., "I am your master", "We are friends").

        User Input: "{user_input}"

        Output JSON:
        {{
            "fact": "...",
            "relationship": "..." (or null)
        }}
        """

        try:
            output = self._llm(prompt, max_tokens=120)
            # Handle potential raw string output if JSON parsing failed inside engine
            if isinstance(output, str):
                return

            fact = output.get("fact")
            relation = output.get("relationship")

            if fact and fact != "None":
                self.memory.remember_fact(user_id, fact)

            if relation and relation != "None":
                self.memory.set_relationship(user_id, relation)

        except Exception as e:
            logger.error(f"[Memory] Consolidation Error: {e}")

    def _summarize_conversation(self, user_id: str, group_id: str = None) -> str:
        """
        Summarize recent conversation with the user (and group, if provided).
        """
        # Fetch recent interactions
        history = self.memory.db.get_recent_interactions(user_id, limit=30, group_id=group_id)
        formatted = []
        for h in history:
            who = "Noa" if h['role'] == "assistant" else f"User({h['user_id'][-4:]})"
            formatted.append(f"{who}: {h['content']}")
        history_text = "\n".join(formatted) if formatted else "（暂无可用会话记录）"

        # If LLM is available, request a concise summary; otherwise return raw history
        if self.inference_engine and formatted:
            prompt = f"""<|im_start|>system
你是会话总结器。请用中文概括下面的最近聊天内容，包含：
1) 主题要点（3-6条）
2) 重要决定或结论
3) 下一步建议（如有）
要求：简洁、分点罗列。
<|im_end|>
<|im_start|>user
{history_text}
<|im_end|>
<|im_start|>assistant
"""
            try:
                output = self._llm(prompt, max_tokens=300)
                summary = output.get("reply", "").strip() or history_text
                return f"最近聊天概要：\n{summary}"
            except Exception as e:
                logger.error(f"[Summary LLM Error] {e}")
                return f"最近聊天记录：\n{history_text}"
        else:
            return f"最近聊天记录：\n{history_text}"
