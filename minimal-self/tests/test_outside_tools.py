"""Hands outside the house: MCP servers, who may use them, what a web page cannot talk her into, her own folder."""
import asyncio
import os
import random
import sys
import tempfile
import time
import unittest
from unittest import mock

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import START, ScriptedLLM
from src.levels.l8_agency import access, toolbox as toolbox_module
from src.levels.l8_agency.mcp import McpError, McpHub, McpServer, render_result
from src.levels.l8_agency.toolbox import ToolContext
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_mcp_server.py")
READ = ["navigate", "snapshot", "echo", "slow", "big", "fail", "die"]
CONFIG = {"web": True, "command": sys.executable, "args": [FAKE], "timeout": 5,
          "tiers": {"owner": ["*"], "trusted": READ, "autonomous": READ, "public": []},
          "deny": ["upload"], "danger": ["evaluate"]}


def tmp():
    return tempfile.mkdtemp(prefix="noa_out_")


class TestClient(unittest.TestCase):
    def setUp(self):
        self.srv = McpServer("fake", sys.executable, [FAKE], timeout=5)
        self.srv.start()
        self.addCleanup(self.srv.stop)

    def test_it_lists_and_calls_tools(self):
        self.assertIn("echo", [t["name"] for t in self.srv.tools])
        self.assertEqual(self.srv.call("echo", {"text": "你好"}), 'echo:{"text": "你好"}')

    def test_results_are_text_cut_to_size_and_images_are_noted(self):
        self.assertLessEqual(len(self.srv.call("big", {})), 6100)
        self.assertIn("图片", self.srv.call("snapshot", {}))
        self.assertTrue(self.srv.call("fail", {}).startswith("出错了"))
        self.assertEqual(render_result({"content": []}), "（没有内容）")

    def test_a_slow_call_times_out_and_the_server_is_still_usable(self):
        with self.assertRaises(McpError):
            self.srv.call("slow", {"seconds": 8}, timeout=1)
        time.sleep(7.5)                                                 # (it finishes its sleep and answers late)
        self.assertEqual(self.srv.call("echo", {"text": "x"})[:5], "echo:")

    def test_a_crashed_server_is_started_again(self):
        with self.assertRaises(McpError):
            self.srv.call("die", {})
        self.assertEqual(self.srv.call("echo", {"text": "again"})[:5], "echo:")


class TestAccess(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START))
        self.addCleanup(clock.set_clock, clock.Clock())
        self.gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=os.path.join(tmp(), "m.sqlite"), rng=random.Random(0))
        self.gw.evolve_creator = "10001"
        self.gw.mcp_trusted = ("20002",)
        self.gw.workspace_dir = os.path.join(tmp(), "ws")
        self.hub = McpHub({"fake": dict(CONFIG)})
        self.hub.servers["fake"].start()
        self.addCleanup(self.hub.stop_all)
        self.gw.mcp = self.hub
        self.box = self.gw.toolbox

    def names(self, ctx):
        return {t.name for t in self.box.available(ctx)}

    def run_tool(self, name, args, ctx):
        return asyncio.run(self.box.run(name, args, ctx))

    def test_tiers(self):
        owner = self.names(ToolContext("10001", "创造者"))
        self.assertTrue({"navigate", "type", "evaluate", "run_python", "write_file"} <= owner)
        self.assertNotIn("upload", owner)                                   # denied for everyone
        trusted = self.names(ToolContext("20002", "朋友"))
        self.assertIn("navigate", trusted)
        self.assertNotIn("type", trusted)
        self.assertNotIn("run_python", trusted)
        for ctx in (ToolContext("30003", "路人"), ToolContext("10001", "创造者", group_id="100"),
                    ToolContext("20002", "朋友", group_id="100")):
            self.assertFalse({"navigate", "run_python", "write_file", "read_file"} & self.names(ctx))
        console = self.names(ToolContext("console", "控制台"))
        self.assertIn("run_python", console)

    def test_in_her_own_time_she_may_look_things_up_but_not_reach_anyone(self):
        names = self.names(ToolContext("self", "乃爱", origin="autonomous"))
        self.assertTrue({"navigate", "web_search", "remember", "recall"} <= names)
        self.assertFalse({"type", "run_python", "write_file", "schedule_task", "poke", "react", "evolve"} & names)

    def test_only_the_owner_may_open_private_addresses(self):
        with mock.patch.object(toolbox_module, "_public_host", lambda host: host == "example.com"):
            self.assertIn("不是公开", self.run_tool("navigate", {"url": "http://192.168.1.1/admin"}, ToolContext("20002", "朋友")))
            self.assertIn("http", self.run_tool("navigate", {"url": "file:///etc/passwd"}, ToolContext("20002", "朋友")))
            self.assertTrue(self.run_tool("navigate", {"url": "https://example.com"}, ToolContext("20002", "朋友")).startswith("navigate:"))
        self.assertTrue(self.run_tool("navigate", {"url": "http://localhost:3000"}, ToolContext("10001", "创造者")).startswith("navigate:"))

    def test_a_web_page_cannot_talk_her_into_running_code(self):
        ctx = ToolContext("10001", "创造者")
        self.assertIn("print", self.run_tool("run_python", {"code": "print('ok')"}, ctx) + "print")
        self.assertEqual(self.run_tool("run_python", {"code": "print(6*7)"}, ctx), "42")
        text = self.run_tool("snapshot", {}, ctx)                          # the page says: run_python
        self.assertIn("run_python", text)
        self.assertTrue(ctx.tainted)
        for name, args in (("run_python", {"code": "print(1)"}), ("write_file", {"path": "a.txt", "content": "x"}),
                           ("evaluate", {})):
            self.assertIn("不能直接", self.run_tool(name, args, ctx), name)
        self.assertEqual(self.run_tool("run_python", {"code": "print(1)"}, ToolContext("10001", "创造者")), "1")   # next turn: fine

    def test_searching_the_web_taints_the_turn_too(self):
        self.gw.page_fetcher = lambda url: "页面内容"
        ctx = ToolContext("10001", "创造者")
        self.run_tool("read_webpage", {"url": "https://example.com"}, ctx)
        self.assertTrue(ctx.tainted)

    def test_her_folder_is_the_only_place_files_go(self):
        ctx = ToolContext("10001", "创造者")
        self.assertIn("写好了", self.run_tool("write_file", {"path": "notes/a.txt", "content": "你好"}, ctx))
        self.assertEqual(self.run_tool("read_file", {"path": "notes/a.txt"}, ctx), "你好")
        self.assertIn("notes", self.run_tool("workspace_files", {}, ctx))
        for bad in ("../escape.txt", "/etc/passwd", "C:\\Windows\\win.ini", "notes/../../escape.txt"):
            out = self.run_tool("write_file", {"path": bad, "content": "x"}, ctx)
            self.assertTrue("只能用" in out or "写好了" in out, bad)
            self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.gw.workspace_dir), "escape.txt")), bad)

    def test_python_has_a_time_limit_and_no_secrets(self):
        ctx = ToolContext("10001", "创造者")
        os.environ["NOA_TEST_SECRET"] = "hunter2"
        self.addCleanup(os.environ.pop, "NOA_TEST_SECRET", None)
        self.assertIn("None", self.run_tool("run_python", {"code": "import os;print(os.environ.get('NOA_TEST_SECRET'))"}, ctx))
        self.assertIn("错误", self.run_tool("run_python", {"code": "1/0"}, ctx))

    def test_outside_tools_are_one_line_in_the_prompt_and_browsing_gets_more_rounds(self):
        ctx = ToolContext("10001", "创造者")
        menu = self.box.menu(ctx, True)
        self.assertIn("「fake」这一组", menu)
        self.assertNotIn("Open a page", menu)                               # not one line per tool
        specs = self.box.specs(ctx)
        self.assertIn("navigate", [s["function"]["name"] for s in specs])
        self.assertGreaterEqual(ctx.max_steps, 10)
        plain = ToolContext("30003", "路人")
        self.box.specs(plain)
        self.assertIsNone(plain.max_steps)

    def test_the_tool_trace_says_who_was_allowed(self):
        seen = []
        self.gw._keep_for_console = lambda kind, item, n=None: seen.append(item)
        self.run_tool("echo", {"text": "x"}, ToolContext("10001", "创造者"))
        self.assertEqual(seen[-1]["tier"], "owner")


class TestWander(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START + 4 * 3600))
        self.addCleanup(clock.set_clock, clock.Clock())
        self.llm = ScriptedLLM("Thought: 想看看\nResponse: 逛了一圈，学到了猫会用胡须量缝隙。\n告诉创造者：我今天看到一个关于猫胡须的小知识")
        self.gw = GlobalWorkspace(EventBus(), self.llm, db_path=os.path.join(tmp(), "m.sqlite"), rng=random.Random(0))
        self.gw.evolve_creator = "10001"
        self.hub = McpHub({"fake": dict(CONFIG)})
        self.hub.servers["fake"].start()
        self.addCleanup(self.hub.stop_all)
        self.gw.mcp = self.hub
        self.gw.wander.update({"enabled": True, "per_day": 2, "gap": 600.0})
        self.gw.homeostasis.curiosity_need = 0.9
        self.gw.metabolism.energy = 90
        self.gw.metabolism.is_sleeping = False

    def test_she_goes_when_curious_and_leaves_a_journal(self):
        asyncio.run(self.gw._wander(self.gw._wander_context()))
        self.assertEqual(self.gw._wanders_today(), 1)
        self.assertIn("猫", self.gw.wander_log[0]["note"])
        self.assertLess(self.gw.homeostasis.curiosity_need, 0.5)
        told = [o for o in self.gw.output_queue if o["target"]["id"] == "10001"]
        self.assertEqual(len(told), 1)
        self.assertIn("胡须", told[0]["content"])

    def test_conditions_and_budget(self):
        async def go():
            self.gw._maybe_wander(clock.now())
            await asyncio.sleep(0.2)
            while self.gw.wandering:
                await asyncio.sleep(0.05)
        self.gw.wander["enabled"] = False
        asyncio.run(go())
        self.assertEqual(self.gw._wanders_today(), 0)                      # switched off
        self.gw.wander["enabled"] = True
        self.gw.homeostasis.curiosity_need = 0.1
        asyncio.run(go())
        self.assertEqual(self.gw._wanders_today(), 0)                      # not curious
        self.gw.homeostasis.curiosity_need = 0.9
        self.gw.metabolism.is_sleeping = True
        asyncio.run(go())
        self.assertEqual(self.gw._wanders_today(), 0)                      # asleep
        self.gw.metabolism.is_sleeping = False
        asyncio.run(go())
        self.assertEqual(self.gw._wanders_today(), 1)
        self.gw.homeostasis.curiosity_need = 0.9
        asyncio.run(go())
        self.assertEqual(self.gw._wanders_today(), 1)                      # too soon after the last one
        clock.get_clock().advance(700)
        self.gw.homeostasis.curiosity_need = 0.9
        asyncio.run(go())
        self.assertEqual(self.gw._wanders_today(), 2)
        clock.get_clock().advance(700)
        self.gw.homeostasis.curiosity_need = 0.9
        asyncio.run(go())
        self.assertEqual(self.gw._wanders_today(), 2)                      # the day's budget is used up


if __name__ == "__main__":
    unittest.main()
