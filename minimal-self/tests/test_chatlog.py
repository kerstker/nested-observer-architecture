"""Following a conversation: the chat log she reads is the latest part, from her side, with everyone in it."""
import asyncio
import os
import random
import sqlite3
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, START
from src.levels.l2_memory.chatlog import ChatLog, chat_key
from src.levels.l2_memory.storage.database import MemoryDatabase
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_chat_"), "c.sqlite")

def make_gw(reply="嗯嗯"):
    clock.set_clock(SimClock(START + 2 * 3600))
    gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=db_path(), rng=random.Random(0))
    gw.environment.get_context = lambda: {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None,
                                          "is_night": False}
    return gw

def last_reply_prompt(gw):
    return next(c["prompt"] for c in reversed(gw.inference_engine.calls) if "[要求]" in c["prompt"])

class TestTranscript(unittest.TestCase):
    def test_latest_messages_from_her_side_within_budget(self):
        clock.set_clock(SimClock(START))
        log = ChatLog(db_path())
        chat = chat_key("u1", None)
        for i in range(30):
            log.record(chat, "u1", "老白", f"第{i}句", t=clock.now() + i)
        log.record(chat, "self", "乃爱", "我在听", t=clock.now() + 31, is_self=True)
        text = log.transcript(chat, max_chars=200)
        self.assertIn("我（乃爱）：我在听", text)
        self.assertIn("第29句", text)
        self.assertNotIn("第0句", text)                       # the oldest are the ones left out
        self.assertIn("更早的聊天省略了", text)

    def test_quotes_resolve_to_whose_words_they_were(self):
        clock.set_clock(SimClock(START))
        log = ChatLog(db_path())
        chat = chat_key(None, "g1")
        log.record(chat, "self", "乃爱", "我觉得早起跑步挺好", message_id="m1", is_self=True)
        log.record(chat, "u2", "阿禾", "我不同意", reply_to="m1")
        self.assertIn("阿禾：（引用 我（乃爱） 说的「我觉得早起跑步挺好」）我不同意", log.transcript(chat))

    def test_gaps_in_time_are_marked(self):
        clock.set_clock(SimClock(START))
        log = ChatLog(db_path())
        chat = chat_key("u1", None)
        log.record(chat, "u1", "老白", "晚安", t=START - 8 * 3600)
        log.record(chat, "u1", "老白", "早", t=START)
        self.assertIn("隔了 8 小时", log.transcript(chat))

class TestInTheConversation(unittest.TestCase):
    def test_group_chatter_she_was_not_part_of_is_in_what_she_reads(self):
        gw = make_gw()
        asyncio.run(gw.hear("明天几点集合啊？", user_id="u4", nickname="大刘", group_id="g1", addressed=False))
        asyncio.run(gw.hear("九点，学校东门", user_id="u5", nickname="阿珊", group_id="g1", addressed=False))
        asyncio.run(gw.hear("@我（乃爱） 他们说几点集合？", user_id="u6", nickname="小周", group_id="g1", addressed=True))
        prompt = last_reply_prompt(gw)
        self.assertIn("大刘：明天几点集合啊？", prompt)
        self.assertIn("阿珊：九点，学校东门", prompt)
        transcript = prompt.split("[Recent Conversation]")[1].split("[")[0]
        self.assertNotIn("他们说几点集合？", transcript)          # the message she answers is the user turn, not history

    def test_her_own_stickers_read_as_what_they_show(self):
        gw = make_gw()
        conn = sqlite3.connect(gw.memory.db.db_path)
        conn.execute("INSERT INTO stickers (id, key, kind, data, contexts, looks) VALUES "
                     "(43, 'image:x', 'image', '{}', '[]', '流泪小猫，配文“那种感觉又上来了”')")
        conn.commit()
        conn.close()
        gw.record_own("[表情:43]", user_id="u1")
        asyncio.run(gw.hear("你刚才发的表情什么意思", user_id="u1", nickname="小柚", addressed=True))
        prompt = last_reply_prompt(gw)
        self.assertIn("我（乃爱）：[发了表情：流泪小猫，配文“那种感觉又上来了”]", prompt)
        self.assertNotIn("[表情:43]", prompt.split("[我收藏的表情]")[0])

    def test_a_known_sticker_someone_sends_reads_as_what_it_shows(self):
        gw = make_gw()
        conn = sqlite3.connect(gw.memory.db.db_path)
        conn.execute("INSERT INTO stickers (id, key, kind, data, contexts, users, uses, looks) VALUES "
                     "(6, 'image:abc.gif', 'image', '{}', '[]', '[]', 3, '仓鼠被锤子砸扁喊“老吴！”')")
        conn.commit()
        conn.close()
        seg = {"type": "image", "data": {"sub_type": "1", "file": "abc.gif"}}
        asyncio.run(gw.hear("", user_id="u1", nickname="小柚", addressed=True, stickers=[seg]))
        rows = gw.chatlog.recent(chat_key("u1", None))
        self.assertEqual(rows[0]["text"], "[表情包：仓鼠被锤子砸扁喊“老吴！”]")

    def test_her_replies_are_recorded_as_hers(self):
        gw = make_gw("我觉得都挺好")
        asyncio.run(gw.hear("猫好还是狗好", user_id="u1", nickname="阿宁", addressed=True))
        rows = gw.chatlog.recent(chat_key("u1", None))
        self.assertEqual([(r["sender_name"], bool(r["is_self"])) for r in rows], [("阿宁", False), ("乃爱", True)])

class TestHistoryQuery(unittest.TestCase):
    def test_recent_interactions_are_the_latest_not_the_oldest(self):
        clock.set_clock(SimClock(START))
        db = MemoryDatabase(db_path())
        for i in range(30):
            db.add_interaction("u1", "user", f"m{i}", "default", None)
            clock.get_clock().advance(10)
        got = [r["content"] for r in db.get_recent_interactions("u1", limit=5)]
        self.assertEqual(got, ["m25", "m26", "m27", "m28", "m29"])

if __name__ == "__main__":
    unittest.main()
