import asyncio
import logging
import os
import random
import re
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l8_agency.goals import GoalSystem
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None, "is_night": False}

def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_goal_"), "g.sqlite")

def make_gw(reply="嗯"):
    clock.set_clock(SimClock(START + 2 * 3600))
    gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=db_path(), rng=random.Random(0))
    gw.search_tool = StubSearch()
    gw.environment.get_context = lambda: DAY
    return gw

def hear(gw, text, user="u1", nick="小明", group=None):
    return asyncio.run(gw.hear(text, user_id=user, nickname=nick, group_id=group, addressed=True))

class FailingSearch:
    def try_search(self, query, limit=5):
        return None

class TestGoalSystem(unittest.TestCase):
    def test_promises_are_recognised(self):
        g = GoalSystem(db_path())
        gid = g.detect_promise("等我查查，明天告诉你", "u1", "小明", "北京明天下雨吗", 0.0)
        goal = g.get(gid)
        self.assertEqual((goal["kind"], goal["target"]), ("promise", "u1"))
        self.assertEqual(goal["topic"], "北京明天下雨吗")
        self.assertEqual([s["text"] for s in goal["steps"]], ["去查清楚", "告诉 ta"])
        self.assertIsNone(g.detect_promise("哈哈明天见", "u1", "小明", "拜拜", 0.0))
        self.assertIsNone(g.detect_promise("我查了，但没查到", "u1", "小明", "上海天气", 0.0))   # done, not promised
        later = g.get(g.detect_promise("今年才过一半呢\n我查查再回你，别急", "u1", "小明", "诺奖给了谁", 0.0))
        self.assertEqual((later["topic"], later["deadline"]), ("诺奖给了谁", 6 * 3600))
        self.assertIsNone(g.detect_promise("等我查查告诉你", "u1", "小明", "北京明天下雨吗", 10.0))  # same promise

    def test_deadlines_make_goals_press(self):
        g = GoalSystem(db_path())
        gid = g.detect_promise("明天告诉你", "u1", "小明", "那部电影好看吗", 0.0)
        goal = g.get(gid)
        self.assertLess(g.pull(goal, 0.0), g.pull(goal, 80000))
        self.assertGreater(g.value_for("chat", "u1", 80000), g.value_for("chat", "u1", 0.0))
        self.assertEqual(g.value_for("chat", "u2", 0.0), 0.0)

    def test_broken_promises_and_stale_goals_are_let_go(self):
        g = GoalSystem(db_path())
        p = g.detect_promise("明天告诉你", "u1", "小明", "那部电影好看吗", 0.0)
        l = g.add("learn", "弄明白黑洞", 0.0, topic="黑洞")
        for _ in range(3):
            g.attempt(l)
        gone = {x["id"]: x for x in g.review(3 * 86400)}
        self.assertIn(p, gone)
        self.assertNotIn(l, gone)                         # not stale yet
        gone = {x["id"] for x in g.review(6 * 86400)}
        self.assertIn(l, gone)

    def test_capacity(self):
        g = GoalSystem(db_path())
        for i in range(8):
            g.add("learn", f"弄明白{i}", 0.0, topic=str(i), importance=0.1 * i)
        self.assertEqual(len(g.active()), GoalSystem.MAX_ACTIVE)
        self.assertNotIn("0", [x["topic"] for x in g.active()])   # the least important waits

class TestGoalsInHerLife(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_she_keeps_a_promise(self):
        gw = make_gw("等我查查，明天告诉你")
        hear(gw, "你知道北京明天会不会下雨吗")
        self.assertEqual(gw.goals.active()[0]["kind"], "promise")
        gw.inference_engine.reply = "我查了，北京明天小雨，记得带伞"
        c = clock.get_clock()
        for _ in range(24 * 6):                    # a day, ticking every 10 minutes
            c.advance(600)
            asyncio.run(gw.tick())
            if not gw.goals.active():
                break
        self.assertEqual(gw.goals.active(), [])
        self.assertEqual(gw.goals.history(1)[0]["status"], "done")
        sent = [m for m in gw.output_queue if (m.get("target") or {}).get("id") == "u1"]
        self.assertTrue(sent)
        self.assertTrue(any("查到的是" in call["prompt"] for call in gw.inference_engine.calls))

    def test_what_stumped_her_becomes_something_to_find_out(self):
        gw = make_gw("Thought: 不知道\nConfidence: 20\nResponse: 这个我还真不知道")
        gw.search_tool = FailingSearch()
        hear(gw, "珠穆朗玛峰到底有多少米")
        goal = gw.goals.active()[0]
        self.assertEqual((goal["kind"], goal["target"]), ("learn", "u1"))
        gw.search_tool = StubSearch()
        gw.user_models.get("u1")["private"] = True
        gw.inference_engine.reply = "珠峰 8848.86 米，你上次问我的"
        gw._trigger_curiosity_behavior()
        self.assertEqual(gw.goals.history(1)[0]["status"], "done")
        self.assertEqual(gw.output_queue[-1]["target"]["id"], "u1")     # told the person who asked

    def test_she_marks_what_a_conversation_fulfilled(self):
        gw = make_gw("嗯")
        gid = gw.goals.add("relationship", "多了解一下小明", clock.now(), target="u1")
        gw.inference_engine.reply = f"原来你喜欢爬山啊\n[完成:{gid}]"
        reply = hear(gw, "我周末一般去爬山")
        self.assertNotIn("完成", reply)
        self.assertEqual(gw.goals.get(gid)["status"], "done")
        self.assertIn("[我心里记着的事]", gw.inference_engine.calls[0]["prompt"])

    def test_someone_she_likes_but_hardly_knows(self):
        gw = make_gw("哈哈是吗")
        c = clock.get_clock()
        for i in range(30):                      # she answers most of them; ~15 conversations' worth
            hear(gw, f"哈哈今天好开心第{i}次，太棒了")
            c.advance(200)
        self.assertIn("relationship", [g["kind"] for g in gw.goals.active()])

    def test_goals_of_her_own_come_from_reflection(self):
        class DreamLLM(ScriptedLLM):
            def generate_json(self, prompt, max_tokens=200, temperature=0.7):
                if "整理今天真实发生过的事" in prompt:
                    ids = [int(x) for x in re.findall(r"^\[(\d+)\]", prompt, re.M)]
                    return {"diary": "今天学了吉他。", "autobiography": "我是乃爱。", "facts": [], "memes": [],
                            "goal_review": [],
                            "new_goal": {"text": "学会弹一首完整的曲子", "why": "今天小明弹给我听了",
                                         "steps": ["查一查怎么入门", "问小明"], "episodes": ids[:1]}}
                return super().generate_json(prompt, max_tokens, temperature)
        gw = make_gw()
        gw.inference_engine = DreamLLM("嗯")
        hear(gw, "我今天弹吉他给你听")
        gw._dream_process()
        goal = gw.goals.active()[0]
        self.assertEqual((goal["kind"], goal["text"]), ("self", "学会弹一首完整的曲子"))

    def test_invented_goals_are_rejected(self):
        class DreamLLM(ScriptedLLM):
            def generate_json(self, prompt, max_tokens=200, temperature=0.7):
                if "整理今天真实发生过的事" in prompt:
                    return {"diary": "x", "facts": [], "memes": [], "new_goal": {"text": "征服世界", "episodes": [99999]}}
                return super().generate_json(prompt, max_tokens, temperature)
        gw = make_gw()
        gw.inference_engine = DreamLLM("嗯")
        hear(gw, "在吗")
        gw._dream_process()
        self.assertEqual(gw.goals.active(), [])

if __name__ == "__main__":
    unittest.main()
