import asyncio
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from napcat import napcat_server  # noqa: E402
from src.core import clock  # noqa: E402
from src.core.clock import SimClock, Clock  # noqa: E402
from src.core.event_bus import EventBus  # noqa: E402
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START  # noqa: E402
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace  # noqa: E402

DAY = {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None, "is_night": False}

def make_gw():
    clock.set_clock(SimClock(START + 4 * 3600))
    db = os.path.join(tempfile.mkdtemp(prefix="noa_cont_"), "c.sqlite")
    gw = GlobalWorkspace(EventBus(), ScriptedLLM("对啊，显卡现在贵得离谱"), db_path=db)
    gw.search_tool = StubSearch()
    gw.environment.get_context = lambda: DAY
    return gw

def hear(gw, text, user="u1", nick="小明", group="G", addressed=False, elsewhere=False):
    return asyncio.run(gw.hear(text, user_id=user, nickname=nick, group_id=group,
                               addressed=addressed, aimed_elsewhere=elsewhere))

class TestContinuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_the_person_she_talks_with_keeps_going(self):
        gw = make_gw()                                     # G is not even an autonomous group
        self.assertTrue(hear(gw, "乃爱你觉得显卡贵不贵", addressed=True))
        clock.get_clock().advance(40)
        self.assertTrue(hear(gw, "我觉得还会再涨"))          # no name, no @: still talking to her
        self.assertEqual(gw.decisions[-1]["detail"], "接着聊")

    def test_conversation_goes_cold(self):
        gw = make_gw()
        hear(gw, "乃爱你觉得显卡贵不贵", addressed=True)
        clock.get_clock().advance(600)
        self.assertIsNone(hear(gw, "我觉得还会再涨"))

    def test_talking_to_someone_else(self):
        gw = make_gw()
        hear(gw, "乃爱你觉得显卡贵不贵", addressed=True)
        clock.get_clock().advance(30)
        self.assertIsNone(hear(gw, "你那张卡多少钱买的", elsewhere=True))   # @/quote someone else

    def test_others_picking_up_her_topic(self):
        gw = make_gw()
        gw.autonomous_groups = {"G"}
        hear(gw, "乃爱你觉得显卡贵不贵", addressed=True)
        clock.get_clock().advance(30)
        self.assertTrue(hear(gw, "显卡贵得离谱，我都不敢买了", user="u2", nick="小红"))
        self.assertEqual(gw.decisions[-1]["detail"], "接着聊")
        # ...but in a group where she does not speak freely, only her interlocutor carries it on
        gw2 = make_gw()
        hear(gw2, "乃爱你觉得显卡贵不贵", addressed=True)
        clock.get_clock().advance(30)
        self.assertIsNone(hear(gw2, "显卡贵得离谱，我都不敢买了", user="u2", nick="小红"))

    def test_quote_reply_to_her_is_recognised(self):
        napcat_server._remember_sent({"status": "ok", "retcode": 0, "data": {"message_id": 1637783067}})
        self.assertTrue(napcat_server.is_own_message(1637783067))
        self.assertTrue(napcat_server.is_own_message("1637783067"))
        self.assertFalse(napcat_server.is_own_message(42))

if __name__ == "__main__":
    unittest.main()
