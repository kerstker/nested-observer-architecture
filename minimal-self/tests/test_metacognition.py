import asyncio
import logging
import os
import random
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.core.workspace import Candidate
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l10_consciousness.attention_schema import AttentionSchema
from src.levels.l10_consciousness.metacognition import Metacognition
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None, "is_night": False}

def make_gw(reply="嗯，我记得"):
    clock.set_clock(SimClock(START + 4 * 3600))
    db = os.path.join(tempfile.mkdtemp(prefix="noa_meta_"), "m.sqlite")
    gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=db, rng=random.Random(0))
    gw.search_tool = StubSearch()
    gw.environment.get_context = lambda: DAY
    return gw

def hear(gw, text, user="u1", nick="小明"):
    return asyncio.run(gw.hear(text, user_id=user, nickname=nick, group_id=None, addressed=True))

def cand(kind, content="x", salience=0.7, **payload):
    return Candidate("self_model", kind, content, salience, payload)

class TestAttentionSchema(unittest.TestCase):
    def test_learns_what_follows_what(self):
        a = AttentionSchema()
        for i in range(10):
            a.observe([cand("hunger")], now=i * 60.0)
            a.observe([cand("social")], now=i * 60.0 + 30)
        self.assertGreater(a.predict("hunger")["social"], 0.6)

    def test_capture_is_noticed(self):
        a = AttentionSchema()
        for i in range(10):
            a.observe([cand("hunger")], now=i * 60.0)
            a.observe([cand("social")], now=i * 60.0 + 30)
        events = a.observe([cand("pain", "好痛")], now=700)
        self.assertEqual(events[0]["type"], "capture")
        self.assertIn("分了心", a.describe(now=720))

    def test_rumination_is_noticed_and_let_go(self):
        a = AttentionSchema()
        events = []
        for i in range(8):
            events += a.observe([cand("ignored", "小红一直没回我", user_id="r")], now=i * 300.0)
        self.assertIn("disengage", [e["type"] for e in events])
        self.assertLess(a.bias(cand("ignored"), now=2400), 0)
        self.assertEqual(a.bias(cand("user_message"), now=2400), 0)      # never deaf to people
        self.assertEqual(a.bias(cand("ignored"), now=2400 + 5 * 3600), 0)  # and not forever

    def test_letting_go_takes_longer_each_time(self):
        a = AttentionSchema()
        untils, t = [], 0.0
        while len(untils) < 2:
            for e in a.observe([cand("ignored", "又没回")], now=t):
                if e["type"] == "disengage":
                    untils.append(e["until"] - t)
            t += 300
        self.assertGreater(untils[1], untils[0])

    def test_no_control_when_ablated(self):
        a = AttentionSchema()
        a.control = False
        for i in range(10):
            self.assertEqual([e for e in a.observe([cand("ignored")], now=i * 300.0) if e["type"] == "disengage"], [])

    def test_persistence(self):
        a = AttentionSchema()
        a.observe([cand("hunger")], now=0)
        a.observe([cand("social")], now=10)
        b = AttentionSchema()
        b.load_dict(a.to_dict())
        self.assertEqual(b.transitions, a.transitions)

class TestMetacognition(unittest.TestCase):
    def test_domains(self):
        m = Metacognition()
        self.assertEqual(m.domain_of("你还记得我喜欢什么吗"), "memory")
        self.assertEqual(m.domain_of("珠穆朗玛峰有多少米"), "knowledge")
        self.assertEqual(m.domain_of("哈哈哈今天好开心"), "chat")

    def test_overconfidence_is_learned_and_corrected(self):
        m = Metacognition()
        for i in range(20):
            m.expect("u", "你喜欢草莓", "memory", m.confidence("memory", 0.9, 0.9), i * 60.0)
            m.judge("u", "不对，你记错了" if i % 5 < 3 else "对对", i * 60.0 + 30)
        self.assertGreater(m.bias("memory"), 0.3)
        self.assertLess(m.confidence("memory", 0.9, 0.9)["calibrated"], 0.6)
        self.assertIn("留点余地", m.guidance("memory", 0.9))
        self.assertEqual(m.bias("chat"), 0.0)                     # other domains are their own matter

    def test_no_verdict_no_learning(self):
        m = Metacognition()
        m.expect("u", "嗯", "chat", m.confidence("chat", 0.75), 0)
        self.assertIsNone(m.judge("u", "今天下雨了", 10))
        self.assertEqual(m.stats["chat"]["n"], 0)

    def test_low_feeling_of_knowing_says_so(self):
        m = Metacognition()
        fok = m.feeling_of_knowing("memory", recall_similarity=0.1)
        self.assertIn("不确定就直说", m.guidance("memory", fok))
        self.assertEqual(m.guidance("chat", 0.75), "")

class TestInTheWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_confidence_is_read_and_never_said(self):
        gw = make_gw("Thought: 我记得\nConfidence: 85\nResponse: 你喜欢草莓")
        reply = hear(gw, "你还记得我喜欢什么吗")
        self.assertEqual(reply, "你喜欢草莓")
        self.assertEqual(gw.metacog.pending["u1"]["domain"], "memory")
        self.assertTrue(any("Confidence:" in c["prompt"] for c in gw.inference_engine.calls))

    def test_a_confident_mistake_is_felt(self):
        gw = make_gw("Thought: 我记得\nConfidence: 95\nResponse: 你喜欢草莓")
        hear(gw, "你还记得我喜欢什么吗")
        clock.get_clock().advance(60)
        hear(gw, "不对，我喜欢的是芒果")
        self.assertIn("metacog", [d["kind"] for d in gw.decisions])
        # The mistake is on her mind while she answers the correction
        self.assertIn("error", [w.kind for _, ws in gw.workspace.trace for w in ws])
        self.assertTrue(any("说我说错了" in r["cause"] for r in gw.causes.recent(10)))

    def test_attention_questions_are_answered_from_the_schema(self):
        gw = make_gw("在呢")
        gw.homeostasis.hunger = 0.9
        asyncio.run(gw.tick())
        hear(gw, "你在想什么呢")
        prompt = gw.inference_engine.last_prompt()
        self.assertIn("[我的注意力]", prompt)
        self.assertIn("注意力：", prompt)

    def test_state_survives_restart(self):
        gw = make_gw("Thought: x\nConfidence: 90\nResponse: 你喜欢草莓")
        for i in range(4):
            hear(gw, f"你还记得我喜欢什么吗{i}")
            clock.get_clock().advance(60)
            hear(gw, "不对")
            clock.get_clock().advance(60)
        gw._save_state("test")
        gw2 = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=gw.state_store.db_path)
        self.assertEqual(gw2.metacog.stats["memory"]["n"], gw.metacog.stats["memory"]["n"])

if __name__ == "__main__":
    unittest.main()
