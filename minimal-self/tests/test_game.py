"""Her body in Minecraft: game chat is heard like any message, answered in game, and she plays on her own."""
import asyncio
import json
import os
import random
import sys
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import START, ScriptedLLM
from src.levels.l8_agency import access
from src.levels.l8_agency.game import GameBridge, mcpfabric_server, minecraft_server, parse_chat, parse_events, split_for_chat
from src.levels.l8_agency.mcp import McpHub
from src.levels.l8_agency.toolbox import ToolContext
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_mc_server.py")


class TestPieces(unittest.TestCase):
    def test_chat_text_is_parsed(self):
        text = ("Found 3 chat message(s):\n\n1. 2026-10-26T10:00:00.000Z - Steve: hello there\n"
                "2. 2026-10-26T10:00:05.000Z - Alex_2: 乃爱：来挖矿吗？\n3. 2026-10-26T10:00:09.000Z - Steve: a: b\n")
        self.assertEqual(parse_chat(text), [("2026-10-26T10:00:00.000Z", "Steve", "hello there"),
                                            ("2026-10-26T10:00:05.000Z", "Alex_2", "乃爱：来挖矿吗？"),
                                            ("2026-10-26T10:00:09.000Z", "Steve", "a: b")])
        self.assertEqual(parse_chat("出错了：Cannot connect to Minecraft server"), [])

    def test_what_she_says_fits_game_chat(self):
        self.assertEqual(split_for_chat("好呀"), ["好呀"])
        lines = split_for_chat("这是一句很长很长的话，" * 12 + "\n第二行")
        self.assertLessEqual(len(lines), 3)
        self.assertTrue(all(len(x) <= 90 for x in lines))
        self.assertEqual(split_for_chat(""), [])
        self.assertEqual(split_for_chat("（已经发过了，就不再多说）"), [])            # seen in the friends' server
        self.assertEqual(split_for_chat("好的\n(nods)\n走吧"), ["好的", "走吧"])

    def test_the_hub_entry(self):
        cfg = minecraft_server("mc.example.com", 25570, "Noa")
        self.assertIn("--host", cfg["args"])
        self.assertIn("25570", cfg["args"])
        self.assertTrue(access.allowed(cfg, access.PLAYER, "dig-block"))
        self.assertTrue(access.allowed(cfg, access.AUTONOMOUS, "dig-block"))
        self.assertFalse(access.allowed(cfg, access.PUBLIC, "dig-block"))


class TestGame(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START + 4 * 3600))
        self.addCleanup(clock.set_clock, clock.Clock())
        self.dir = tempfile.mkdtemp(prefix="noa_mc_")
        os.environ["NOA_FAKE_MC"] = self.dir
        self.addCleanup(os.environ.pop, "NOA_FAKE_MC", None)
        self.llm = ScriptedLLM("好呀，我来了")
        self.gw = GlobalWorkspace(EventBus(), self.llm, db_path=os.path.join(self.dir, "m.sqlite"), rng=random.Random(0))
        cfg = minecraft_server("localhost", 25565, "Noa")
        cfg.update({"command": sys.executable, "args": [FAKE]})
        self.hub = McpHub({"minecraft": cfg})
        self.hub.servers["minecraft"].start()
        self.addCleanup(self.hub.stop_all)
        self.gw.mcp = self.hub
        self.gw.mc_players = ("Steve",)
        self.game = GameBridge(self.gw, "Noa", names=("乃爱",), poll_seconds=0, play_gap=60, play_per_hour=2)
        self.gw.game = self.game
        self.gw.metabolism.energy = 90
        self.gw.metabolism.is_sleeping = False

    def say_in_game(self, line):
        with open(os.path.join(self.dir, "chat.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def sent(self):
        path = os.path.join(self.dir, "sent.txt")
        return open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []

    def poll(self):
        async def go():
            await self.game.poll(clock.now())
            while self.game.busy:
                await asyncio.sleep(0.05)
        asyncio.run(go())
        clock.get_clock().advance(5)

    def test_old_chat_is_history_and_new_chat_is_answered_in_game(self):
        self.say_in_game("2026-10-26T10:00:00.000Z - Steve: 昨天的话")
        self.poll()
        self.assertEqual(self.sent(), [])                                   # what was said before she came is not answered
        self.say_in_game("2026-10-26T10:01:00.000Z - Steve: 乃爱 你在吗")
        self.poll()
        self.assertEqual(self.sent(), ["好呀，我来了"])
        self.poll()
        self.assertEqual(len(self.sent()), 1)                               # and nothing is answered twice
        self.assertTrue(self.game.connected)

    def test_she_does_not_answer_herself(self):
        self.poll()
        self.say_in_game("2026-10-26T10:01:00.000Z - Noa: 乃爱 我来了")
        self.poll()
        self.assertEqual(self.sent(), [])

    def test_players_are_people_to_her_and_trust_decides_what_they_can_make_her_do(self):
        self.poll()
        self.say_in_game("2026-10-26T10:01:00.000Z - Steve: 乃爱 你好呀")
        self.poll()
        self.assertIsNotNone(self.gw.memory.db.get_user("mc:Steve"))
        steve, stranger = ToolContext("mc:Steve", "Steve", "mc"), ToolContext("mc:Herobrine", "Herobrine", "mc")
        names = lambda ctx: {t.name for t in self.gw.toolbox.available(ctx)}
        self.assertTrue({"dig-block", "send-chat", "move-to-position"} <= names(steve))
        self.assertFalse({"dig-block", "send-chat", "run_python", "write_file", "navigate"} & names(stranger))
        self.assertFalse({"run_python", "write_file", "browser_navigate"} & names(steve))     # the game is all they get
        self.gw.mc_players = ("*",)
        self.assertIn("dig-block", names(stranger))

    def test_nothing_happens_while_locked(self):
        self.poll()
        self.gw.firewall.set_locked(True)
        self.say_in_game("2026-10-26T10:01:00.000Z - Steve: 乃爱 你在吗")
        self.poll()
        self.assertEqual(self.sent(), [])

    def test_she_plays_on_her_own_within_a_budget(self):
        self.poll()
        t = clock.now()
        self.assertTrue(self.game.play_due(t))
        asyncio.run(self.game.play(t))
        self.assertEqual(self.game.journal[0]["tools"], [])                 # (the scripted model used no tools)
        self.assertEqual(self.sent(), ["好呀，我来了"])
        self.assertTrue(any("Minecraft" in e["content"] for e in self.gw.episodes.recent(5)))
        self.assertFalse(self.game.play_due(t + 5))                          # too soon
        self.assertTrue(self.game.play_due(t + 100))
        asyncio.run(self.game.play(t + 100))
        self.assertFalse(self.game.play_due(t + 400))                        # the hour's budget (2) is used up
        self.assertTrue(self.game.play_due(t + 3700))

    def test_she_does_not_go_playing_while_someone_is_talking_or_she_is_asleep(self):
        self.poll()
        t = clock.now()
        self.game.last_chat = t - 5
        self.assertFalse(self.game.play_due(t))
        self.game.last_chat = 0
        self.gw.metabolism.is_sleeping = True
        self.assertFalse(self.game.play_due(t))

    def test_the_silent_answer_is_not_said(self):
        self.llm.reply = "（不说）"
        self.poll()
        asyncio.run(self.game.play(clock.now()))
        self.assertEqual(self.sent(), [])

    def test_the_tool_menu_says_what_the_game_body_is(self):
        menu = self.gw.toolbox.menu(ToolContext("mc:Steve", "Steve", "mc"), True)
        self.assertIn("Minecraft", menu)
        self.assertIn("dig-block", menu)


class TestMcpfabric(unittest.TestCase):
    """The same life in a real modded client: the mcpfabric bridge."""

    def setUp(self):
        clock.set_clock(SimClock(START + 4 * 3600))
        self.addCleanup(clock.set_clock, clock.Clock())
        self.dir = tempfile.mkdtemp(prefix="noa_mcf_")
        os.environ["NOA_FAKE_MC"] = self.dir
        os.environ["NOA_FAKE_MC_MODE"] = "mcpfabric"
        self.addCleanup(os.environ.pop, "NOA_FAKE_MC", None)
        self.addCleanup(os.environ.pop, "NOA_FAKE_MC_MODE", None)
        self.gw = GlobalWorkspace(EventBus(), ScriptedLLM("我在呢"), db_path=os.path.join(self.dir, "m.sqlite"),
                                  rng=random.Random(0))
        cfg = mcpfabric_server("unused.js", "http://127.0.0.1:25599", "t")
        cfg.update({"command": sys.executable, "args": [FAKE]})
        self.hub = McpHub({"minecraft": cfg})
        self.hub.servers["minecraft"].start()
        self.addCleanup(self.hub.stop_all)
        self.gw.mcp = self.hub
        self.gw.mc_players = ("*",)
        self.game = GameBridge(self.gw, "Noa", names=("乃爱",), poll_seconds=0)

    def test_chat_events_are_parsed(self):
        text = json.dumps({"messages": [
            {"id": 7, "type": "chat", "data": {"text": "<Steve> 乃爱 来", "sender": "Steve"}},
            {"id": 8, "type": "system_message", "data": {"text": "Alex joined the game"}},
            {"id": 9, "type": "chat", "data": {"text": "<Alex> hi"}},
            {"id": 10, "type": "chat", "data": {"player": "Bob", "text": "server side"}}]})
        self.assertEqual(parse_events(text), [("7", "Steve", "乃爱 来"), ("9", "Alex", "hi"), ("10", "Bob", "server side")])
        self.assertEqual(parse_events("出错了：bridge not reachable"), [])
        # what a "no chat reports" server really sends (seen on the friends' server)
        real = json.dumps({"messages": [
            {"id": 3, "type": "system_message", "data": {"text": "<kerst_ker> noa你在吗？", "overlay": False}},
            {"id": 4, "type": "system_message", "data": {"text": "你的游戏模式已被更新为创造模式", "overlay": False}},
            {"id": 5, "type": "system_message", "data": {"text": "JusticeedGen取得了进度[怪物猎人]", "overlay": False}},
            {"id": 6, "type": "system_message", "data": {"text": "<x> action bar", "overlay": True}}]})
        self.assertEqual(parse_events(real), [("3", "kerst_ker", "noa你在吗？")])

    def test_she_hears_and_answers_through_the_mod(self):
        def poll():
            async def go():
                await self.game.poll(clock.now())
                while self.game.busy:
                    await asyncio.sleep(0.05)
            asyncio.run(go())
        poll()                                                          # history
        with open(os.path.join(self.dir, "chat.txt"), "a", encoding="utf-8") as f:
            f.write("t - Steve: 乃爱 你好呀\n")
        poll()
        sent = open(os.path.join(self.dir, "sent.txt"), encoding="utf-8").read().splitlines()
        self.assertEqual(sent, ["我在呢"])

    def test_a_pause_menu_or_open_inventory_in_her_way_is_closed(self):
        for screen in ("pause", "container", "death"):
            with open(os.path.join(self.dir, "screen.txt"), "w", encoding="utf-8") as f:
                f.write(screen)
            self.assertEqual(self.game.clear_screen(), screen)
            self.assertEqual(open(os.path.join(self.dir, "screen.txt"), encoding="utf-8").read(), "none")
        self.assertIsNone(self.game.clear_screen())                      # nothing open: nothing to do

    def test_chests_crafting_and_recipes_through_the_bridge(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from src.levels.l8_agency import mc_extra
        seen = []

        class Bridge(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.headers.get("Authorization"), body["method"], body["params"]))
                out = json.dumps({"ok": True, "result": {"method": body["method"]}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass
        server = HTTPServer(("127.0.0.1", 0), Bridge)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.gw.mc_bridge = (f"http://127.0.0.1:{server.server_port}", "secret")
        run = lambda name, args, ctx: asyncio.run(self.gw.toolbox.run(name, args, ctx))
        steve = ToolContext("mc:Steve", "Steve", "mc")
        self.assertIn("container.open", run("mc_open_block", {"x": 1, "y": 64, "z": 2}, steve))
        self.assertIn("container.transfer", run("mc_container_transfer", {"direction": "take", "item": "minecraft:oak_log"}, steve))
        self.assertIn("recipes.query", run("mc_recipes", {"items": "minecraft:crafting_table, minecraft:stick"}, steve))
        self.assertEqual(seen[0], ("Bearer secret", "container.open", {"x": 1, "y": 64, "z": 2}))
        self.assertEqual(seen[2][2], {"items": ["minecraft:crafting_table", "minecraft:stick"]})
        stranger = ToolContext("30003", "路人", "100")
        self.assertNotIn("mc_open_block", {t.name for t in self.gw.toolbox.available(stranger)})
        self.gw.firewall.set_locked(True)
        self.assertIn("紧急封锁", run("mc_open_block", {"x": 1, "y": 64, "z": 2}, steve))
        self.gw.firewall.set_locked(False)
        self.assertEqual(len(seen), 3)

    def test_admin_tools_are_never_hers(self):
        names = {t.name for t in self.gw.toolbox.available(ToolContext("10001", "创造者"))} | \
                {t.name for t in self.gw.toolbox.available(ToolContext("mc:Steve", "Steve", "mc"))}
        self.assertIn("navigate_to", names)
        self.assertIn("screenshot", names)
        self.assertFalse({"run_command", "give_item", "set_block", "supervised_command"} & names)


if __name__ == "__main__":
    unittest.main()
