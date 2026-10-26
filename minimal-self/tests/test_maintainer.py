"""The maintainer channel: she asks, the maintainer is told, the answer comes back to her."""
import asyncio
import os
import subprocess
import sys
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import START, ScriptedLLM
from src.levels.l17_self_improvement.maintainer import Maintainer
from src.levels.l8_agency.toolbox import ToolContext, Toolbox, parse_text_actions
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_mt_"), "m.sqlite")


class TestChannel(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START))
        self.m = Maintainer(db_path())

    def test_a_request_is_filed_and_the_same_one_is_not_filed_twice(self):
        first = self.m.file("self", "乃爱", "想要一个查天气的工具", "总有人问天气", "tool")
        self.assertTrue(first["ok"])
        again = self.m.file("self", "乃爱", "想要一个查天气的工具")
        self.assertFalse(again["ok"])
        self.assertEqual(again["id"], first["id"])
        self.assertFalse(self.m.file("self", "乃爱", "")["ok"])

    def test_she_cannot_flood_the_maintainer(self):
        results = [self.m.file("self", "乃爱", f"第{i}件事") for i in range(5)]
        self.assertEqual([r["ok"] for r in results], [True, True, True, False, False])
        clock.get_clock().advance(86400 + 1)
        self.assertTrue(self.m.file("self", "乃爱", "第二天的事")["ok"])

    def test_the_creator_is_told_once_and_not_too_often(self):
        self.m.file("self", "乃爱（和小明聊天时）", "想要一个查天气的工具")
        notice = self.m.creator_notice()
        self.assertIn("想要一个查天气的工具", notice)
        self.assertIn("scripts/maintainer.py", notice)
        self.assertIsNone(self.m.creator_notice())                       # nothing new
        self.m.file("self", "乃爱", "另一件事")
        self.assertIsNone(self.m.creator_notice())                       # too soon after the last notice
        clock.get_clock().advance(self.m.NOTICE_GAP + 1)
        self.assertIn("另一件事", self.m.creator_notice())

    def test_the_answer_comes_back_to_her_once(self):
        rid = self.m.file("self", "乃爱", "想要一个查天气的工具")["id"]
        self.assertEqual(self.m.replies_for_her(), [])
        self.assertTrue(self.m.reply(rid, "accepted", "可以，这周做"))
        (answer,) = self.m.replies_for_her()
        self.assertIn("可以，这周做", self.m.describe(answer))
        self.m.mark_told(rid)
        self.assertEqual(self.m.replies_for_her(), [])
        self.m.reply(rid, "done", "好了，现在能查了")
        self.assertEqual(len(self.m.replies_for_her()), 1)               # a new answer is new again
        self.assertFalse(self.m.reply("rq-nope", "done"))
        with self.assertRaises(ValueError):
            self.m.reply(rid, "whatever")


class TestInHerLife(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START))
        self.gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path())
        self.gw.evolve_creator = "10001"

    def test_the_tool_files_a_request(self):
        box = Toolbox(self.gw)
        ctx = ToolContext("u1", "小明")
        self.assertIn("ask_maintainer", [t["function"]["name"] for t in box.specs(ctx)])
        result = asyncio.run(box.run("ask_maintainer", {"title": "想要一个查天气的工具", "detail": "总有人问", "kind": "tool"}, ctx))
        self.assertIn("记下", result)
        (row,) = self.gw.maintainer.list()
        self.assertEqual(row["kind"], "tool")
        self.assertIn("小明", row["requester_name"])
        self.assertEqual(parse_text_actions('Action: ask_maintainer title="x"')[0][0], "ask_maintainer")

    def test_tick_tells_the_creator_and_then_tells_her_the_answer(self):
        rid = self.gw.maintainer.file("self", "乃爱", "想要一个查天气的工具")["id"]
        self.gw._maintainer_tick(clock.now())
        notice = [o for o in self.gw.output_queue if o["target"]["id"] == "10001"]
        self.assertEqual(len(notice), 1)
        self.assertIn("想要一个查天气的工具", notice[0]["content"])

        self.gw.maintainer.reply(rid, "accepted", "可以，这周做")
        before = self.gw.episodes.unconsolidated_count()
        self.gw._maintainer_tick(clock.now())
        self.assertGreater(self.gw.episodes.unconsolidated_count(), before)       # she experienced the answer
        self.assertEqual(self.gw.maintainer.replies_for_her(), [])
        prompt = self.gw._construct_prompt("在吗", "", {"emotion": "neutral"}, None, user_id="u1", nickname="小明")
        self.assertIn("[我请维护者做的事]", prompt)
        self.assertIn("可以，这周做", prompt)


class TestCommandLine(unittest.TestCase):
    def run_cli(self, db, *args):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "maintainer.py"), "--db", db, *args],
                             capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return out.returncode, out.stdout

    def test_list_show_reply(self):
        clock.set_clock(SimClock(START))
        db = db_path()
        rid = Maintainer(db).file("self", "乃爱", "想要一个查天气的工具", "总有人问", "tool")["id"]
        code, out = self.run_cli(db, "list")
        self.assertEqual(code, 0)
        self.assertIn(rid, out)
        self.assertIn("总有人问", self.run_cli(db, "show", rid)[1])
        code, out = self.run_cli(db, "reply", rid, "--status", "declined", "--note", "这个暂时不做")
        self.assertEqual(code, 0)
        self.assertEqual(Maintainer(db).get(rid)["status"], "declined")
        self.assertIn("没有请求", self.run_cli(db, "list")[1])               # closed ones are not "active"
        self.assertEqual(self.run_cli(db, "reply", "rq-nope", "--status", "done")[0], 1)


if __name__ == "__main__":
    unittest.main()
