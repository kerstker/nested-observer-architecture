import asyncio
import logging
import os
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l1_perception.group_language import GroupLanguage
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None, "is_night": False}

def make_gw(reply="在呢，怎么了"):
    clock.set_clock(SimClock(START))
    db = os.path.join(tempfile.mkdtemp(prefix="noa_choice_"), "c.sqlite")
    gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=db)
    gw.search_tool = StubSearch()
    gw.environment.get_context = lambda: DAY
    return gw

def hear(gw, text, user="u1", nick="小明", group=None, addressed=True):
    return asyncio.run(gw.hear(text, user_id=user, nickname=nick, group_id=group, addressed=addressed))

class TestChoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_overheard_outside_allowed_groups_never_speaks(self):
        gw = make_gw()
        self.assertIsNone(hear(gw, "今晚吃啥", group="G", addressed=False))
        self.assertEqual(gw.inference_engine.calls, [])                   # no model call just for hearing
        self.assertIn("小明：今晚吃啥", gw.group_language.recent_lines("G"))  # but she did hear it

    def test_she_can_choose_silence(self):
        gw = make_gw("Thought: 不想理他\nResponse: （不回）")
        self.assertIsNone(hear(gw, "在吗"))
        self.assertEqual(gw.decisions[-1]["kind"], "silent")
        self.assertIn("（没有回）", gw.episodes.recent(1)[0]["content"])

    def test_sleeping_she_does_not_see_it_then_catches_up(self):
        gw = make_gw("刚睡醒，看到你找我了")
        gw._set_sleeping(True, source="test")
        gw.metabolism.energy = 30
        self.assertIsNone(hear(gw, "在吗"))
        self.assertEqual(gw.decisions[-1]["kind"], "asleep")
        gw.metabolism.energy = 85
        asyncio.run(gw.tick())
        self.assertFalse(gw.metabolism.is_sleeping)
        item = gw.output_queue[-1]
        self.assertEqual(item["target"], {"type": "private", "id": "u1"})
        self.assertTrue(any("睡着的时候" in c["prompt"] for c in gw.inference_engine.calls))

    def test_repeated_calls_wake_her_up(self):
        gw = make_gw()
        gw._set_sleeping(True, source="test")
        gw.metabolism.energy = 30
        self.assertIsNone(hear(gw, "在吗"))
        self.assertIsNone(hear(gw, "在吗？"))
        self.assertTrue(hear(gw, "醒醒！"))
        self.assertFalse(gw.metabolism.is_sleeping)
        self.assertGreater(gw.emotion.state.anger, 0)

    def test_asked_to_be_quiet(self):
        gw = make_gw()
        gw.autonomous_groups = {"G"}
        self.assertIsNone(hear(gw, "乃爱别吵了", group="G", addressed=True))
        self.assertGreater(gw.group_mute_until["G"], clock.now())
        gw.homeostasis.social_need = 1.0
        hear(gw, "我们继续聊", group="G", addressed=False)
        self.assertNotEqual(gw.decisions[-1]["kind"], "consider")       # did not even consider speaking

    def test_typing_rhythm(self):
        gw = make_gw()
        plan = gw.plan_messages("在呢\n怎么啦，找我有事吗？", incoming="在吗")
        self.assertEqual([t for _, t in plan], ["在呢", "怎么啦，找我有事吗？"])
        self.assertTrue(all(d > 0 for d, _ in plan))
        gw.metabolism.energy = 10
        tired = gw.plan_messages("在呢\n怎么啦，找我有事吗？", incoming="在吗")
        self.assertGreater(sum(d for d, _ in tired), sum(d for d, _ in plan))   # slower when tired
        long_line = "我觉得这件事情其实挺复杂的。" * 10
        self.assertGreater(len(gw.plan_messages(long_line)), 1)

class TestGroupLanguage(unittest.TestCase):
    def test_phrases_used_by_several_people(self):
        db = os.path.join(tempfile.mkdtemp(prefix="noa_lang_"), "l.sqlite")
        lang = GroupLanguage(db)
        for who, text in [("甲", "这也太绝绝子了"), ("乙", "绝绝子"), ("甲", "真的绝绝子"), ("丙", "今天下雨")]:
            lang.observe("G", who, text)
        self.assertIn("绝绝子", lang.candidates("G"))
        lang.observe("G", "我", "绝绝子绝绝子绝绝子", from_self=True)    # her own words do not count
        self.assertNotIn("下雨", lang.candidates("G"))

if __name__ == "__main__":
    unittest.main()
