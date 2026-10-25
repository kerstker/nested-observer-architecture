import asyncio
import json
import logging
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from napcat.qq import QQ  # noqa: E402
from src.core import clock  # noqa: E402
from src.core.clock import SimClock, Clock  # noqa: E402
from src.core.event_bus import EventBus  # noqa: E402
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START  # noqa: E402
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace  # noqa: E402

class FakeManager:
    """Stands in for NapCat: answers API calls like NapCat would, from a table."""
    def __init__(self, qq, answers):
        self.qq, self.answers, self.sent = qq, answers, []
        self.robot_connections = {"noa": object()}

    async def send_to_robot(self, robot_id, frame):
        self.sent.append(frame)
        data = self.answers.get(frame["action"])
        if data is not None:
            asyncio.get_running_loop().call_soon(self.qq.resolve, {"status": "ok", "retcode": 0, "data": data,
                                                                    "echo": frame["echo"]})
        return True

def run(coro):
    return asyncio.run(coro)

class TestPerceiving(unittest.TestCase):
    def test_what_a_person_would_see(self):
        q = QQ()
        card = json.dumps({"prompt": "[分享]", "meta": {"news": {"title": "显卡又涨价了", "desc": "5090 一卡难求"}}})
        segs = [{"type": "text", "data": {"text": "你看这个"}}, {"type": "json", "data": {"data": card}},
                {"type": "record", "data": {}}, {"type": "dice", "data": {"result": "6"}},
                {"type": "image", "data": {"sub_type": 0, "file": "a.jpg"}}]
        text = run(q.perceive(segs))
        for part in ("你看这个", "显卡又涨价了", "5090 一卡难求", "语音", "掷骰子：6", "图片"):
            self.assertIn(part, text)

    def test_quotes_and_forwards_are_opened(self):
        q = QQ()
        q.self_id = "999"
        q.manager = FakeManager(q, {
            "get_msg": {"sender": {"user_id": 999}, "message": [{"type": "text", "data": {"text": "我觉得会涨"}}]},
            "get_forward_msg": {"messages": [{"sender": {"nickname": "阿甲"}, "message": [{"type": "text", "data": {"text": "第一句"}}]},
                                             {"sender": {"nickname": "阿乙"}, "message": "第二句"}]}})
        quoted = run(q.perceive([{"type": "reply", "data": {"id": "1"}}, {"type": "text", "data": {"text": "真的吗"}}]))
        # Her own words quoted by someone else must read as hers, never as a bare "我"
        self.assertTrue(quoted.startswith("（引用了我（乃爱）说的：「我觉得会涨」）"), quoted)
        at = run(q.perceive([{"type": "at", "data": {"qq": "999"}}, {"type": "at", "data": {"qq": "5", "name": "小明"}},
                             {"type": "text", "data": {"text": "你们看"}}]))
        self.assertEqual(at, "@我（乃爱） @小明 你们看")
        fwd = run(q.perceive([{"type": "forward", "data": {"id": "f1"}}]))
        self.assertIn("阿甲：第一句", fwd)
        self.assertIn("阿乙：第二句", fwd)

    def test_pictures_are_handed_over_whole(self):
        class FakeSenses:
            def fetch_image(self, d):
                return {"data": b"raw-" + d["url"].encode(), "mime": "image/jpeg"}

            def can_describe(self):
                return True

            def see(self, d):
                raise AssertionError("described although the picture itself rides along")

        q = QQ()
        q.senses = FakeSenses()
        segs = [{"type": "text", "data": {"text": "你看"}},
                {"type": "image", "data": {"url": "a"}}, {"type": "image", "data": {"url": "b"}},
                {"type": "image", "data": {"url": "c"}}]
        media = run(q.media(segs, limit=2))
        self.assertEqual([m["data"] for m in media], [b"raw-a", b"raw-b"])
        self.assertEqual(media[0]["mime"], "image/jpeg")
        # caption=False (the picture itself is attached): she is not given a description
        self.assertEqual(run(q.perceive(segs, look=True, caption=False)),
                         "你看[发了一张图片][发了一张图片][发了一张图片]")
        self.assertEqual(run(QQ().media(segs)), [])            # no senses: no pictures, no crash

    def test_quoted_pictures_are_looked_at_when_paying_attention(self):
        class FakeSenses:
            def __init__(self):
                self.calls = 0

            def can_describe(self):
                return True

            def see(self, d):
                self.calls += 1
                return "一张小猫的照片"

        q = QQ()
        q.self_id = "999"
        q.read_images = False          # QQ OCR is not what this test is about
        q.senses = FakeSenses()
        q.manager = FakeManager(q, {"get_msg": {
            "sender": {"user_id": 5, "nickname": "小明"},
            "message": [{"type": "image", "data": {"sub_type": 0, "url": "u"}}]}})
        text = run(q.perceive([{"type": "reply", "data": {"id": "1"}},
                               {"type": "text", "data": {"text": "这是啥"}}], look=True))
        self.assertIn("引用了 小明 说的", text)
        self.assertIn("一张小猫的照片", text)                 # a quoted picture cannot ride along: API describes it
        calls = q.senses.calls
        run(q.perceive([{"type": "reply", "data": {"id": "1"}}]))
        self.assertEqual(q.senses.calls, calls)              # overheard: no API call for a quoted picture

    def test_words_in_pictures_are_read(self):
        q = QQ()
        q.manager = FakeManager(q, {"ocr_image": {"texts": [{"text": "我太难了"}]}})
        text = run(q.perceive([{"type": "image", "data": {"sub_type": 1, "url": "https://x/1"}}]))
        self.assertEqual(text, "[表情包：上面写着：我太难了]")
        run(q.perceive([{"type": "image", "data": {"sub_type": 1, "url": "https://x/1"}}]))
        self.assertEqual(len(q.manager.sent), 1)                        # read once, remembered

    def test_looking_and_listening_when_paying_attention(self):
        class FakeSenses:
            async def hear(self, qq, d):
                return "明天几点集合"

            def can_describe(self):
                return True

            def see(self, d):
                return "一只橘猫趴在键盘上"

        q = QQ()
        q.senses = FakeSenses()
        segs = [{"type": "record", "data": {"file": "a.amr"}}, {"type": "image", "data": {"sub_type": 0, "url": "u"}}]
        self.assertEqual(run(q.perceive(segs, look=True)), "[语音：明天几点集合][图片：一只橘猫趴在键盘上]")
        self.assertEqual(run(q.perceive(segs)), "[发了一段语音][发了一张图片]")       # overheard, not looked at

    def test_calls_without_napcat_fail_soft(self):
        self.assertIsNone(run(QQ().call("get_msg", {"message_id": 1})))
        self.assertEqual(run(QQ().perceive([{"type": "reply", "data": {"id": "1"}}])), "（回复了一条消息）")

class TestNotices(unittest.TestCase):
    def test_qq_events_become_hers(self):
        q = QQ()
        q.self_id = "999"
        self.assertEqual(q.notice({"notice_type": "notify", "sub_type": "poke", "user_id": 5, "target_id": 999,
                                   "group_id": 7})["kind"], "poke")
        self.assertIsNone(q.notice({"notice_type": "notify", "sub_type": "poke", "user_id": 5, "target_id": 6}))
        self.assertEqual(q.notice({"notice_type": "group_ban", "sub_type": "ban", "user_id": 999, "operator_id": 5,
                                   "group_id": 7, "duration": 600})["kind"], "muted")
        self.assertEqual(q.notice({"notice_type": "group_increase", "user_id": 8, "group_id": 7})["kind"], "joined")
        q.notice({"notice_type": "notify", "sub_type": "input_status", "user_id": 5, "event_type": 1})
        self.assertTrue(q.is_typing("5"))

    def test_gestures_go_out_as_napcat_actions(self):
        q = QQ()
        q.manager = FakeManager(q, {"send_poke": {}, "set_msg_emoji_like": {}})
        self.assertTrue(run(q.poke("5", "7")))
        self.assertTrue(run(q.react("123", "笑")))
        self.assertEqual([f["action"] for f in q.manager.sent], ["send_poke", "set_msg_emoji_like"])
        self.assertEqual(q.manager.sent[1]["params"]["emoji_id"], "178")

class TestHerOnQQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def _gw(self, reply="哈哈"):
        clock.set_clock(SimClock(START + 4 * 3600))
        gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=os.path.join(tempfile.mkdtemp(prefix="noa_qq_"), "q.sqlite"),
                             rng=random.Random(0))
        gw.search_tool = StubSearch()
        return gw

    def test_pokes_are_play_but_not_a_loop(self):
        gw = self._gw()
        for _ in range(3):
            gw.drives.reward("social", "chat", "u1", 1.0)
        pokes = 0
        for _ in range(10):
            run(gw.on_qq_event({"kind": "poke", "user_id": "u1", "group_id": None}, "小明"))
            pokes += sum(1 for g in gw.take_gestures() if g["kind"] == "poke")
            clock.get_clock().advance(20)
        self.assertGreaterEqual(pokes, 1)
        self.assertLessEqual(pokes, 2)                     # a poke war ends

    def test_she_can_only_listen_while_muted(self):
        gw = self._gw()
        run(gw.on_qq_event({"kind": "muted", "user_id": "admin", "group_id": "G", "duration": 600}, "管理"))
        self.assertIsNone(run(gw.hear("乃爱你在吗", user_id="u1", nickname="小明", group_id="G", addressed=True)))
        self.assertIn("禁言", gw.decisions[-1]["detail"])
        clock.get_clock().advance(700)
        self.assertTrue(run(gw.hear("乃爱你在吗", user_id="u1", nickname="小明", group_id="G", addressed=True)))

    def test_gestures_in_her_reply_are_done_not_said(self):
        gw = self._gw("[回应:笑]\n笑死\n[戳一戳]")
        reply = run(gw.hear("你看我刚才摔了个狗吃屎", user_id="u1", nickname="小明"))
        self.assertEqual(reply, "笑死")
        self.assertEqual(sorted(g["kind"] for g in gw.take_gestures()), ["poke", "react"])

    def test_a_reaction_instead_of_words(self):
        reacted = 0
        for seed in range(10):
            gw = self._gw()
            gw.rng = random.Random(seed)
            gw.emotion.state.joy = 0.8
            for i in range(6):
                run(gw.hear("哈哈哈", user_id="u1", nickname="小明"))
                clock.get_clock().advance(60)
            reacted += any(g["kind"] == "react" for g in gw.take_gestures())
        self.assertGreater(reacted, 3)

if __name__ == "__main__":
    unittest.main()
