import asyncio
import logging
import os
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None, "is_night": False}

def make_gw(reply="哈哈是吗，那你呢？"):
    clock.set_clock(SimClock(START + 4 * 3600))
    db = os.path.join(tempfile.mkdtemp(prefix="noa_loop_"), "l.sqlite")
    gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=db)
    gw.search_tool = StubSearch()
    gw.environment.get_context = lambda: DAY
    return gw

def hear(gw, text, user="u1", nick="小明", group=None, addressed=True):
    return asyncio.run(gw.hear(text, user_id=user, nickname=nick, group_id=group, addressed=addressed))

class TestNoEndlessLoops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_bot_that_keeps_tagging_her_is_left_alone(self):
        gw = make_gw()
        c = clock.get_clock()
        replies, start, msg = 0, c.now(), "@乃爱 你好呀，我是机器人小B，很高兴认识你"
        energy = gw.metabolism.energy
        while c.now() - start < 1800:
            r = hear(gw, msg, user="botB", nick="小B", group="G")
            replies += bool(r)
            c.advance(sum(d for d, _ in gw.plan_messages(r, msg)) + 2 if r else 4)
            msg = "@乃爱 收到！小B 随时为你服务，祝你今天愉快！"
        self.assertLessEqual(replies, 10)                                   # was 450 before
        self.assertTrue(gw.user_models.get("botB").get("seems_bot"))
        self.assertLess(gw.metabolism.energy, energy)                       # talking tired her
        self.assertIn("limit", [d["kind"] for d in gw.decisions])

    def test_a_normal_person_is_not_cut_off(self):
        gw = make_gw("嗯嗯")
        c = clock.get_clock()
        replied = 0
        for i in range(10):
            replied += bool(hear(gw, f"今天第{i}件事", group="G"))
            c.advance(60)
        self.assertEqual(replied, 10)
        self.assertFalse(gw.user_models.get("u1").get("seems_bot"))

    def test_a_lively_fast_conversation_is_not_cut_off(self):
        gw = make_gw("哈哈真的假的，然后呢")
        c = clock.get_clock()
        topics = ["我今天去爬山了", "山顶上居然有卖烤肠的", "十块钱一根你敢信", "不过味道还行", "下山的时候差点摔了",
                  "还好扶住了一棵树", "那棵树上有个松鼠", "松鼠还冲我叫", "我拍了张照片", "可惜糊了",
                  "明天打算再去一次", "这次带个好点的相机"]
        replied = 0
        for t in topics:                       # a person answering every 25 seconds
            replied += bool(hear(gw, t, group="G"))
            c.advance(25)
        self.assertEqual(replied, len(topics))
        self.assertNotIn("satiation", [w.kind for _, ws in gw.workspace.trace for w in ws])

    def test_a_day_long_conversation_keeps_going(self):
        gw = make_gw("嗯嗯，这个我也想过")
        c = clock.get_clock()
        replied = 0
        lines = ["早上起来发现猫把花盆打翻了", "楼下新开了家咖啡店", "下午突然下了场大雨", "晚上想看部老电影",
                 "老板又让周末加班", "在计划国庆去哪里玩", "最近在学吉他手指好疼", "今天地铁上遇到个怪人",
                 "外卖小哥送错单了", "我妈打电话问我找对象没", "隔壁装修吵了一整天", "刷到一个很好笑的视频"]
        for i in range(120):                   # eight hours, a message every 4 minutes, always something new
            replied += bool(hear(gw, f"{lines[i % len(lines)]}，第{i // len(lines) + 1}次想起来还是觉得离谱"))
            c.advance(240)
            asyncio.run(gw.tick())
        self.assertGreaterEqual(replied, 110)
        self.assertLess(gw.metabolism.energy, 70)   # she is tired by the end of it, as anyone would be

    def test_going_round_in_circles_bores_her(self):
        gw = make_gw("嗯")
        c = clock.get_clock()
        # A person, not a template: different words, the same thing over and over
        for t in ["你在干嘛呀", "在干嘛呢你", "你干嘛呢", "干嘛呢在干嘛", "你在干啥呀", "在干嘛啊你", "你到底在干嘛"]:
            hear(gw, t)
            c.advance(30)
        self.assertFalse(gw.user_models.get("u1").get("seems_bot"))
        self.assertIn("satiation", [w.kind for _, ws in gw.workspace.trace for w in ws])

    def test_goodbye_means_goodbye(self):
        gw = make_gw("好困，我先睡了，回头聊")
        gw.metabolism.energy = 90
        self.assertTrue(hear(gw, "在吗"))
        self.assertIsNone(hear(gw, "别睡啊"))
        self.assertEqual(gw.decisions[-1]["kind"], "away")
        clock.get_clock().advance(1900)
        gw.inference_engine.reply = "我回来了，刚才说什么？"
        asyncio.run(gw.tick())
        self.assertIn("刚才说什么", gw.output_queue[-1]["content"])

    def test_listed_bots_are_never_answered(self):
        gw = make_gw()
        gw.bot_accounts = {"botX"}
        self.assertIsNone(hear(gw, "@乃爱 你好", user="botX", group="G"))
        self.assertEqual(gw.inference_engine.calls, [])

    def test_group_hourly_cap(self):
        gw = make_gw("嗯")
        gw.max_group_messages_per_hour = 3
        c = clock.get_clock()
        got = []
        for i in range(5):
            got.append(bool(hear(gw, f"乃爱 {i}", user=f"p{i}", group="G")))
            c.advance(60)
        self.assertEqual(got, [True, True, True, False, False])

    def test_her_name_is_not_a_personal_fact(self):
        gw = make_gw("在呢")
        hear(gw, "@乃爱 在吗")
        self.assertEqual(len(gw.inference_engine.calls), 1)     # no extra memory-extraction call

if __name__ == "__main__":
    unittest.main()
