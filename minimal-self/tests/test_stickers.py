import asyncio
import base64
import logging
import os
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l1_perception.stickers import StickerMemory
from src.levels.l2_memory.embedding import Embedder
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

PNG = b"\x89PNG fake sticker bytes"

def sticker(name="laugh.gif", url="https://example.invalid/laugh"):
    return {"type": "image", "data": {"file": name, "sub_type": 1, "summary": "[动画表情]", "url": url}}

PHOTO = {"type": "image", "data": {"file": "selfie.jpg", "sub_type": 0, "url": "https://example.invalid/selfie"}}

def make_gw(reply):
    clock.set_clock(SimClock(START))
    db = os.path.join(tempfile.mkdtemp(prefix="noa_sticker_"), "s.sqlite")
    gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=db)
    gw.search_tool = StubSearch()
    gw.stickers.fetch = lambda url: PNG
    return gw

def hear(gw, text, stickers=None, user="u1", nick="小明", group=None, addressed=True):
    return asyncio.run(gw.hear(text, user_id=user, nickname=nick, group_id=group, addressed=addressed, stickers=stickers))

class TestStickers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_only_stickers_are_collected_never_photos(self):
        segs = [PHOTO, sticker(), {"type": "face", "data": {"id": "178"}}, {"type": "text", "data": {"text": "hi"}}]
        kinds = [(s["type"], s["data"].get("sub_type")) for s in StickerMemory.extract(segs)]
        self.assertEqual(kinds, [("image", 1), ("face", None)])

    def test_meaning_is_the_occasion(self):
        db = os.path.join(tempfile.mkdtemp(prefix="noa_sticker_"), "s.sqlite")
        mem = StickerMemory(db, os.path.join(os.path.dirname(db), "st"), Embedder(), fetch=lambda u: PNG)
        sid = mem.observe(sticker(), "甲", "笑死我了")
        mem.observe(sticker(), "乙", "绷不住了")
        self.assertIsNone(mem.observe(PHOTO, "丙", "自拍"))                 # photos are never kept
        found = mem.relevant("哈哈笑死我了这也太好笑", k=3)
        self.assertEqual(found[0]["id"], sid)
        self.assertEqual(found[0]["contexts"], ["笑死我了", "绷不住了"])
        seg = mem.segment(sid)
        self.assertEqual(seg["type"], "image")
        self.assertEqual(base64.b64decode(seg["data"]["file"][len("base64://"):]), PNG)

    def test_she_learns_stickers_and_can_use_them(self):
        gw = make_gw("哈哈哈\n[表情:1]")
        # A sticker-only message: she perceives it, and learns it from what was just said
        gw.last_said["u:u1"] = "我今天把蛋糕烤糊了"
        hear(gw, "", stickers=[sticker()])          # she may or may not answer a bare sticker; she keeps it
        self.assertEqual(gw.stickers.stats()["count"], 1)
        self.assertEqual(gw.stickers.stats()["top"][0]["contexts"], ["我今天把蛋糕烤糊了"])
        # Next time a similar moment comes, it is offered to her and she can send it
        reply = hear(gw, "我也把蛋糕烤糊了哈哈")
        self.assertIn("[表情:1]", gw.inference_engine.last_prompt())
        plan = gw.plan_messages(reply)
        self.assertEqual([p for _, p in plan], ["哈哈哈", "[表情:1]"])
        self.assertEqual(gw.render_message("[表情:1]")[0]["type"], "image")
        self.assertIsNone(gw.render_message("[表情:99]"))          # a sticker she does not have is skipped
        self.assertEqual(gw.render_message("好呀[表情:99]"), "好呀")

    def test_reaction_teaches_her(self):
        gw = make_gw("[表情:1]")
        gw.stickers.observe(sticker(), "甲", "笑死")
        gw.last_sticker["u:u1"] = (1, clock.now())
        hear(gw, "哈哈哈哈笑死")
        self.assertGreater(gw.stickers.stats()["top"][0]["q"], 0.5)

    def test_no_stickers_no_offer(self):
        gw = make_gw("嗯")
        hear(gw, "在吗")
        self.assertNotIn("[我收藏的表情]", gw.inference_engine.last_prompt())

if __name__ == "__main__":
    unittest.main()
