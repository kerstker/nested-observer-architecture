import asyncio
import logging
import os
import random
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l4_emotion.emotion import EmotionState
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None, "is_night": False}

def make_gw(reply="嗯，我也这么觉得", db=None, seed=0):
    clock.set_clock(SimClock(START + 4 * 3600))
    db = db or os.path.join(tempfile.mkdtemp(prefix="noa_self_"), "s.sqlite")
    gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=db, rng=random.Random(seed))
    gw.search_tool = StubSearch()
    gw.environment.get_context = lambda: DAY
    return gw

def hear(gw, text, user="u1", nick="小明", group=None, addressed=True):
    return asyncio.run(gw.hear(text, user_id=user, nickname=nick, group_id=group, addressed=addressed))

class TestSelectiveReplies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_questions_get_answers_small_talk_often_does_not(self):
        gw = make_gw()
        c = clock.get_clock()
        questions = ["你周末想去哪玩？", "最近在看什么番？", "晚饭吃了啥？", "你喜欢猫还是狗？", "明天会下雨吗？",
                     "你觉得我该不该换工作？", "最喜欢哪个季节？", "你会唱歌吗？", "昨天睡得好吗？", "要不要一起打游戏？"]
        asked = []
        for q in questions:
            asked.append(bool(hear(gw, q)))
            c.advance(120)
        laughs = []
        for _ in range(20):
            laughs.append(bool(hear(gw, "哈哈哈")))
            c.advance(120)
        self.assertEqual(sum(1 for a in asked if a), 10)
        self.assertLess(sum(laughs), 12)
        self.assertGreater(sum(laughs), 0)
        self.assertIn("只是随口一句", " ".join(d["detail"] for d in gw.decisions))

    def test_continuations_are_answered_selectively(self):
        replies = 0
        for seed in range(12):
            gw = make_gw(seed=seed)
            hear(gw, "乃爱你觉得显卡贵不贵", group="G")
            clock.get_clock().advance(30)
            replies += bool(hear(gw, "我前两天刚买了一张二手的", group="G", addressed=False))
        self.assertTrue(2 <= replies <= 10, replies)

    def test_a_fast_human_typist_is_not_a_bot(self):
        gw = make_gw("哈哈好")
        c = clock.get_clock()
        for i in range(8):
            r = hear(gw, f"真的笑死我了今天第{i}次被老板点名", group="G")
            c.advance(sum(d for d, _ in gw.plan_messages(r, "x")) + 4 if r else 4)   # answers 4s later
        self.assertFalse(gw.user_models.get("u1").get("seems_bot"))

    def test_old_bot_flags_are_cleared_once(self):
        db = os.path.join(tempfile.mkdtemp(prefix="noa_self_"), "s.sqlite")
        gw = make_gw(db=db)
        gw.user_models.get("human")["seems_bot"] = True
        gw.user_models.save("human")
        snap = gw._snapshot()
        snap.pop("bot_model")
        gw.state_store.save(snap, "old")
        gw2 = make_gw(db=db)
        self.assertFalse(gw2.user_models.get("human").get("seems_bot"))

class TestPersonalityHomeostasis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_mood_returns_to_her_temperament(self):
        gw = make_gw()
        gw.emotion.update(EmotionState(anger=0.9, sorrow=0.6))
        gw.emotion.decay(gw.emotion.last_decay + 4 * 3600)
        self.assertLess(gw.emotion.state.anger, 0.05)
        self.assertGreater(gw.emotion.state.joy, 0.1)            # settles at her warm baseline, not at zero
        self.assertGreater(gw.emotion.valence(), 0)

    def test_she_notices_when_she_is_not_herself(self):
        gw = make_gw("你倒是轻松，你到底想说什么意思？？")
        c = clock.get_clock()
        for i in range(4):
            hear(gw, f"今天天气不错第{i}次", user=f"p{i}")
            c.advance(60)
        self.assertIn("说话有点冲", gw._feeling_words())
        self.assertIn("self_check", [d["kind"] for d in gw.decisions])

    def test_identity_anchor_and_dsh_self_knowledge(self):
        gw = make_gw()
        persona = gw._build_persona_prompt()
        self.assertIn("你都还是这样的你", persona)
        self.assertNotIn("DSH", persona)
        gw.evolve_enabled, gw.evolve_creator = True, "1000000001"
        self.assertIn("DSH", gw._build_persona_prompt())

if __name__ == "__main__":
    unittest.main()
