"""Her body in Minecraft: game chat is heard like any message, answered in game, and she plays on her own."""
import asyncio
import json
import os
import random
import struct
import sys
import tempfile
import unittest
import zlib

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import START, ScriptedLLM
from src.levels.l8_agency import access
from src.levels.l8_agency.game import (GameBridge, mcpfabric_server, minecraft_server, parse_chat, parse_events,
                                       shrink_png, split_for_chat, strip_tool_markup)
from src.levels.l8_agency.mcp import McpHub
from src.levels.l8_agency.toolbox import ToolContext
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_mc_server.py")


def tiny_png(width=80, height=40, rgb=(200, 30, 30)):
    """A real little PNG, without needing Pillow to make the fixture."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class ToolLLM:
    """A stand-in engine that makes native tool calls, so the tool loop can be tested without a model."""
    supports_tools = True

    def __init__(self, steps):
        self.steps = list(steps)
        self.seen = []

    def chatml_to_messages(self, prompt):
        import re
        messages = []
        for role, body in re.findall(r"<\|im_start\|>(system|user|assistant)\n?(.*?)(?=<\|im_start\|>|\Z)", prompt, re.S):
            content = body.split("<|im_end|>")[0].strip()
            if content:
                messages.append({"role": role, "content": content})
        return messages

    def with_media(self, messages, media, limit=3):
        import base64
        import copy
        pics = [m for m in (media or []) if m.get("data")][:limit]
        messages = copy.deepcopy(messages)
        if not pics:
            return messages
        target = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if target is None:
            return messages
        parts = [{"type": "text", "text": target.get("content") if isinstance(target.get("content"), str) else ""}]
        for picture in pics:
            b64 = base64.b64encode(picture["data"]).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{picture.get('mime') or 'image/png'};base64,{b64}"}})
        target["content"] = parts
        return messages

    def chat_tools(self, messages, tools, max_tokens=300, temperature=0.7, tool_choice=None):
        import copy
        self.seen.append(copy.deepcopy(messages))
        return self.steps.pop(0) if self.steps else {"content": "Response: 好", "tool_calls": [], "raw": None}


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
        self.assertEqual(split_for_chat("（对游戏里说）这片枫树林挺好看的"), ["这片枫树林挺好看的"])   # seen in game

    def test_a_tool_call_written_out_as_text_is_not_said(self):
        # what her model really emitted on the friends' server instead of making the call:
        # "枫木板我这就加工，稍等哈～\n\n<｜｜DSML｜｜ calls>..." and it went out in game chat
        raw = ('枫木板我这就加工，稍等哈～\n\n<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="mc_craft">\n'
               '<｜｜DSML｜｜ parameter name="item" string="true">biomeswevegone:maple_planks</｜｜DSML｜｜ parameter>\n'
               '</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>')
        self.assertEqual(split_for_chat(raw), ["枫木板我这就加工，稍等哈～"])
        self.assertEqual(strip_tool_markup(raw), "枫木板我这就加工，稍等哈～")
        self.assertEqual(split_for_chat(raw.split("\n", 1)[1]), [])       # only the markup: nothing to say
        self.assertEqual(split_for_chat('<|im_start|>assistant\n好呀'), ["好呀"])

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
        self.game.last_chat = t - 2                       # someone just spoke to her: a short beat first
        self.assertFalse(self.game.play_due(t))
        self.game.last_chat = t - 10                      # the conversation has a pause: back to her own business
        self.assertTrue(self.game.play_due(t))
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
        self.gw.game = self.game

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

    def _bridge(self, entities):
        """A stand-in for the mod's HTTP bridge: answers the reflex methods and records what was asked."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        calls = []

        class Bridge(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((body["method"], body["params"]))
                result = {"perception.entities": {"entities": entities},
                          "nav.status": {"active": False}}.get(body["method"], {"ok": True})
                out = json.dumps({"ok": True, "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass
        server = HTTPServer(("127.0.0.1", 0), Bridge)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.gw.mc_bridge = (f"http://127.0.0.1:{server.server_port}", "t")
        return calls

    def test_she_sees_who_is_near_by_name(self):
        self._bridge([{"uuid": GameBridge.offline_uuid("Steve"), "type": "minecraft:player", "x": 1, "y": 64, "z": 1, "distance": 3.2},
                      {"uuid": GameBridge.offline_uuid("Stranger"), "type": "minecraft:player", "x": 9, "y": 64, "z": 1, "distance": 9},
                      {"uuid": GameBridge.offline_uuid("Noa"), "type": "minecraft:player", "x": 0, "y": 64, "z": 0, "distance": 0},
                      {"uuid": "x", "type": "minecraft:pig", "distance": 5}, {"uuid": "y", "type": "minecraft:pig", "distance": 6}])
        self.game.known_players.add("Steve")
        text = asyncio.run(self.gw.toolbox.run("mc_who_is_near", {}, ToolContext("mc:Steve", "Steve", "mc")))
        self.assertIn("Steve（3 格）", text)
        self.assertIn("一个不认识的玩家", text)
        self.assertNotIn("Noa", text)
        self.assertIn("pig×2", text)

    def test_her_reply_is_the_only_way_her_words_go_out(self):
        names = {t.name for t in self.gw.toolbox.available(ToolContext("self", "乃爱", origin="game"))}
        self.assertNotIn("send_chat", names)                                 # (it said everything twice)
        self.assertFalse({"list_players", "get_status"} & names)             # (they made her think she was offline)
        self.assertEqual(split_for_chat("好呀[表情:117]"), ["好呀"])
        self.assertEqual(split_for_chat("[表情:117]"), [])

    def test_offline_uuids_match_what_the_server_gives(self):
        # JusticeedGen's player entity as seen on the friends' offline-mode server
        self.assertEqual(GameBridge.offline_uuid("JusticeedGen"), "fc72b067-3595-319a-8f41-36e32a4eb47b")

    def test_she_follows_someone_and_looks_at_whoever_is_near(self):
        steve = GameBridge.offline_uuid("Steve")
        calls = self._bridge([{"uuid": steve, "type": "minecraft:player", "x": 20, "y": 64, "z": 5, "distance": 15},
                              {"uuid": GameBridge.offline_uuid("Noa"), "type": "minecraft:player", "x": 0, "y": 64, "z": 0,
                               "distance": 0}])
        self.game.set_follow("Steve")
        self.game.reflex(clock.now())
        path = [p for m, p in calls if m == "nav.pathTo"]
        self.assertEqual(path[0]["x"], 20)
        self.assertTrue(path[0]["sprint"])                                   # far away: she runs
        self.assertEqual(self.game.people_near, 1)                           # herself is not company
        self.assertFalse(self.game.play_due(clock.now()))                    # walking with someone: no turn of her own
        self.game.set_follow(None)
        calls.clear()
        self.game.reflex(clock.now())
        self.assertFalse([m for m, _ in calls if m == "nav.pathTo"])
        self.assertFalse([m for m, _ in calls if m == "control.lookAt"])     # 15 blocks: too far to look at

    def test_people_around_make_her_turns_come_sooner(self):
        self.game.connected = True
        self.game.primed = True
        self.gw.metabolism.energy = 90
        self.gw.metabolism.is_sleeping = False
        t = clock.now()
        self.game.last_play = t - 60
        self.assertFalse(self.game.play_due(t))                              # alone: every few minutes
        self.game.people_near = 2
        self.assertTrue(self.game.play_due(t))                               # with company: much more often

    def test_someone_talking_interrupts_her_own_turn(self):
        self.game.interrupt = True
        out = asyncio.run(self.gw.toolbox.run("navigate_to", {}, ToolContext("self", "乃爱", origin="game")))
        self.assertIn("先停下", out)
        self.game.interrupt = False
        out = asyncio.run(self.gw.toolbox.run("navigate_to", {}, ToolContext("self", "乃爱", origin="game")))
        self.assertIn("navigate_to ok", out)

    def test_an_answer_to_her_or_someone_beside_her_is_talking_to_her(self):
        t = clock.now()
        self.assertTrue(self.game.spoken_to("乃爱 你好", None, t))
        self.assertFalse(self.game.spoken_to("我就是kerst", None, t))
        self.assertTrue(self.game.spoken_to("你面前", 3.0, t))               # standing right in front of her
        self.assertFalse(self.game.spoken_to("你面前", 40.0, t))
        self.game.last_said = t - 20                                         # she just asked "anyone around?"
        self.assertTrue(self.game.spoken_to("我就是kerst", None, t))
        self.assertFalse(self.game.spoken_to("我就是kerst", None, t + 200))

    def test_no_turn_of_her_own_while_her_client_is_not_in_the_world(self):
        self.game.connected = True
        self.game.people_near = 2
        self.gw.metabolism.energy = 90
        self.gw.metabolism.is_sleeping = False
        self.game.in_world = False
        self.assertFalse(self.game.play_due(clock.now()))
        self.game.in_world = True
        self.assertTrue(self.game.play_due(clock.now()))

    def test_admin_tools_are_never_hers(self):
        names = {t.name for t in self.gw.toolbox.available(ToolContext("10001", "创造者"))} | \
                {t.name for t in self.gw.toolbox.available(ToolContext("mc:Steve", "Steve", "mc"))}
        self.assertIn("navigate_to", names)
        self.assertIn("screenshot", names)
        self.assertFalse({"run_command", "give_item", "set_block", "supervised_command"} & names)

    def test_the_mods_background_jobs_are_hers_and_its_own_mind_is_not(self):
        cfg = mcpfabric_server("unused.js", "http://127.0.0.1:25599", "t")
        self.assertEqual(cfg["env"]["MCPFABRIC_AGENT"], "1")             # the runtime that has the jobs is on
        for job in ("collect_blocks", "craft_item", "travel_to", "plan_craft", "job_status", "job_cancel"):
            self.assertTrue(access.allowed(cfg, access.AUTONOMOUS, job), job)
            self.assertTrue(access.allowed(cfg, access.PLAYER, job), job)
        # the runtime's own memory, goal tree, map and container tools would be a second mind: not hers
        for second_mind in ("agent_brief", "observe", "map_view", "remember", "goal_add", "open_container"):
            self.assertFalse(access.allowed(cfg, access.OWNER, second_mind), second_mind)
        self.assertIn("job_status", GameBridge.PLAY_PROMPT)

    def test_what_she_just_did_is_in_the_scene_of_her_next_turn(self):
        self.game.journal = [{"t": clock.now(), "note": "我在 Minecraft 里自己玩了一会儿（2 步：mc_skill、get_self）",
                              "tools": ["mc_skill"]}]
        scene = self.game._scene()
        self.assertIn("刚才", scene)
        self.assertIn("mc_skill", scene)

    def test_tool_markup_never_reaches_what_she_says(self):
        raw = '好呀\n<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="mc_craft">\n</｜｜DSML｜｜ calls>'
        self.assertEqual(self.gw._finalize_response(raw), "好呀")

    def _rejoin_bridge(self):
        """A stand-in for the mod bridge: disconnected to start with, joined again on session.join."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        state = {"screen": {"inWorld": False, "screen": "disconnected", "canJoin": True,
                            "lastWorldId": "mp:4494f8c4934660df"}, "calls": []}

        class Bridge(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state["calls"].append((body["method"], body["params"]))
                if body["method"] == "screen.get":
                    result = state["screen"]
                elif body["method"] == "session.join":
                    result = {"joining": body["params"].get("worldId")}
                    state["screen"] = {"inWorld": True, "screen": "none", "canJoin": False,
                                       "lastWorldId": body["params"].get("worldId")}
                else:
                    result = {"ok": True}
                out = json.dumps({"ok": True, "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass
        server = HTTPServer(("127.0.0.1", 0), Bridge)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.gw.mc_bridge = (f"http://127.0.0.1:{server.server_port}", "t")
        return state

    def test_she_gets_back_into_the_world_after_a_disconnect(self):
        state = self._rejoin_bridge()
        t = clock.now()
        self.assertEqual(self.game.rejoin(t), "joining")
        joined = [p for m, p in state["calls"] if m == "session.join"]
        self.assertEqual(joined[0], {"worldId": "mp:4494f8c4934660df"})     # the world she was just in
        dropped = {"inWorld": False, "screen": "disconnected", "canJoin": True,
                   "lastWorldId": "mp:4494f8c4934660df"}
        self.assertEqual(self.game.rejoin(t + 1, dropped), "too_soon")      # not a reconnect loop
        self.assertEqual(self.game.rejoin(t + 30), "in_world")              # the join worked
        self.assertTrue(self.game.in_world)

    def test_the_rejoin_tool_is_hers(self):
        state = self._rejoin_bridge()
        with open(os.path.join(self.dir, "inworld.txt"), "w", encoding="utf-8") as f:
            f.write("0")
        with open(os.path.join(self.dir, "screen.txt"), "w", encoding="utf-8") as f:
            f.write("disconnected")
        names = {t.name for t in self.gw.toolbox.available(ToolContext("mc:Steve", "Steve", "mc"))}
        self.assertIn("mc_rejoin_world", names)
        out = asyncio.run(self.gw.toolbox.run("mc_rejoin_world", {}, ToolContext("mc:Steve", "Steve", "mc")))
        self.assertIn("重新进游戏", out)
        self.assertEqual([p for m, p in state["calls"] if m == "session.join"],
                         [{"worldId": "mp:4494f8c4934660df"}])

    def _image_bridge(self, png=None):
        """A stand-in for the mod bridge that can take a screenshot."""
        import base64
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        png = png or tiny_png()
        calls = []

        class Bridge(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((body["method"], body["params"]))
                if body["method"] == "vision.screenshot":
                    result = {"format": "png", "width": 80, "height": 40, "bytes": len(png),
                              "base64": base64.b64encode(png).decode("ascii")}
                elif body["method"] == "screen.get":
                    result = {"inWorld": True, "screen": "none", "canJoin": False, "lastWorldId": "mp:x"}
                else:
                    result = {"ok": True}
                out = json.dumps({"ok": True, "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass
        server = HTTPServer(("127.0.0.1", 0), Bridge)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.gw.mc_bridge = (f"http://127.0.0.1:{server.server_port}", "t")
        return calls

    def test_a_screenshot_becomes_a_small_picture_for_the_model(self):
        calls = self._image_bridge()
        shot = self.game.look(force=True)
        self.assertIsNotNone(shot)
        self.assertEqual(shot["mime"], "image/png")
        self.assertTrue(shot["data"].startswith(b"\x89PNG"))
        self.assertEqual(self.game.last_shot, clock.now())
        self.assertIn("vision.screenshot", [m for m, _ in calls])
        from PIL import Image
        import io
        shrunk = shrink_png(tiny_png(1600, 900), 640)       # a 4K game frame does not go to the model at 4K
        self.assertEqual(Image.open(io.BytesIO(shrunk)).width, 640)

    def test_her_own_turns_come_with_eyes(self):
        self._image_bridge()
        seen = {}

        async def fake_respond(prompt, emo, params, media, ctx):
            seen["media"] = media
            return "（不说）"
        self.gw._respond = fake_respond
        asyncio.run(self.game.play(clock.now()))
        self.assertTrue(seen.get("media"))
        self.assertTrue(seen["media"][0]["data"].startswith(b"\x89PNG"))

    def test_looking_on_demand_puts_the_frame_into_the_same_turn(self):
        self._image_bridge()
        engine = ToolLLM([
            {"content": "", "tool_calls": [{"id": "c1", "name": "mc_look", "arguments": {}}], "raw": None},
            {"content": "Response: 看到了", "tool_calls": [], "raw": None},
        ])
        self.gw.inference_engine = engine
        prompt = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
        out = asyncio.run(self.gw._respond_native(prompt, "", {"max_tokens": 40}, None,
                                                  ToolContext("self", "乃爱", origin="game")))
        self.assertIn("看到了", out)
        self.assertTrue(any(isinstance(m.get("content"), list)
                            and any(b.get("type") == "image_url" for b in m["content"])
                            for m in engine.seen[-1]))
        self.assertIsNotNone(self.game.last_shot)

    def test_in_the_game_she_gets_the_game_body_not_the_browser(self):
        from src.levels.l8_agency.toolbox import Tool
        self._image_bridge()
        self.gw.toolbox.tools["browser_navigate"] = Tool("browser_navigate", "打开网页", {"type": "object", "properties": {}},
                                                         lambda gw, ctx, a: "x", group="browser")
        game_names = {s["function"]["name"] for s in self.gw.toolbox.specs(ToolContext("self", "乃爱", origin="game"))}
        owner_names = {s["function"]["name"] for s in self.gw.toolbox.specs(ToolContext("console", "创造者"))}
        self.assertNotIn("browser_navigate", game_names)    # a game turn is the game, not the browser
        self.assertIn("browser_navigate", owner_names)
        self.assertIn("mc_look", game_names)
        self.assertIn("mc_skill", game_names)

    def test_in_danger_she_looks_much_more_often(self):
        self._image_bridge()
        now = clock.now()
        self.game.last_shot = now - 10
        self.game.danger, self.game.health = 99.0, 20.0
        self.assertIsNone(self.game.look())                 # 10s < 20s: normal cadence
        self.game.danger = 6.0                              # a hostile is right there
        self.assertIsNotNone(self.game.look())              # 10s > 4s: she looks again

    def test_body_state_notices_moving_and_danger(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        state = {"x": 10.0, "y": 64.0, "z": -3.0, "health": 18.0, "food": 17, "onGround": True}
        calls = []

        class Bridge(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((body["method"], body["params"]))
                if body["method"] == "player.getState":
                    result = dict(state, motion={"x": 0, "y": 0, "z": 0})
                elif body["method"] == "perception.entities":
                    result = {"entities": [{"kind": "hostile", "distance": 7.0, "type": "minecraft:zombie"},
                                           {"kind": "hostile", "distance": 3.0, "type": "minecraft:skeleton"}]}
                else:
                    result = {"ok": True}
                out = json.dumps({"ok": True, "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass
        server = HTTPServer(("127.0.0.1", 0), Bridge)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.gw.mc_bridge = (f"http://127.0.0.1:{server.server_port}", "t")
        self.game.body_state()
        self.assertEqual((self.game.health, self.game.food, self.game.danger), (18.0, 17.0, 3.0))
        self.assertEqual([h["distance"] for h in self.game.hostiles], [3.0, 7.0])   # nearest first
        self.game.last_moved = 0
        self.game.body_state()                              # same place: not moving
        self.assertEqual(self.game.last_moved, 0)
        state["x"] = 11.0
        self.game.body_state()                              # moved a block: the body is busy
        self.assertEqual(self.game.last_moved, clock.now())
        self.assertTrue(self.game.on_ground)

    def _vitals_bridge(self, state, entities=None, inventory=None):
        """A stand-in bridge for the survival reflexes: vitals, entity list, inventory."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        calls = []
        inventory = inventory or {"hotbar": [], "main": [], "armor": [], "selectedSlot": 0}

        class Bridge(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((body["method"], body["params"]))
                m = body["method"]
                if m == "player.getState":
                    result = dict(state, motion={"x": 0, "y": 0, "z": 0})
                elif m == "perception.entities":
                    result = {"entities": entities or []}
                elif m == "player.getInventory":
                    result = inventory
                else:
                    result = {"ok": True}
                out = json.dumps({"ok": True, "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass
        server = HTTPServer(("127.0.0.1", 0), Bridge)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.gw.mc_bridge = (f"http://127.0.0.1:{server.server_port}", "t")
        return calls

    def test_she_eats_when_hurt_and_hungry(self):
        calls = self._vitals_bridge({"health": 12.0, "food": 8, "x": 10.0, "y": 64.0, "z": -3.0},
                                    inventory={"hotbar": [{"slot": 5, "id": "minecraft:apple", "count": 1}]})
        self.game.connected, self.game.in_world = True, True
        self.game.health, self.game.food = 12.0, 8.0
        self.game.last_eat = 0
        now = clock.now()
        self.assertEqual(self.game.survive(now), "eat")
        methods = [m for m, _ in calls]
        self.assertIn("inventory.selectHotbar", methods)
        self.assertIn("control.startUsing", methods)
        self.assertEqual([p for m, p in calls if m == "inventory.selectHotbar"], [{"slot": 5}])
        self.assertEqual(self.game.survive(now + 2.2), "eating")     # still holding the food
        self.assertIn("control.stopUsing", [m for m, _ in calls])
        self.assertEqual(self.game.last_eat, now + 2.2)

    def test_she_hits_a_hostile_that_gets_close(self):
        calls = self._vitals_bridge({"health": 18.0, "food": 18, "x": 10.0, "y": 64.0, "z": -3.0, "onGround": True})
        self.game.connected = True
        self.game.health, self.game.food = 18.0, 18.0
        self.game.last_eat = clock.now()
        self.game.hostiles = [{"uuid": "m1", "kind": "hostile", "type": "minecraft:zombie", "distance": 2.5,
                               "x": 11.0, "y": 64.0, "z": -3.0}]
        self.game.on_ground = True
        self.assertEqual(self.game.survive(clock.now()), "fight")
        methods = [m for m, _ in calls]
        self.assertGreaterEqual(methods.count("control.look"), 2)   # she turns smoothly, not in one snap
        self.assertIn("interact.attackEntity", methods)
        self.assertEqual([p for m, p in calls if m == "interact.attackEntity"], [{"uuid": "m1"}])

    def test_she_runs_away_when_she_is_low_and_does_not_attack(self):
        calls = self._vitals_bridge({"health": 6.0, "food": 15, "x": 10.0, "y": 64.0, "z": -3.0})
        self.game.connected, self.game.in_world = True, True
        self.game.health, self.game.food = 6.0, 15.0
        self.game.last_eat = clock.now()                     # not eating right now
        self.game.last_flee = 0
        self.game.hostiles = [{"uuid": "m1", "kind": "hostile", "type": "minecraft:zombie", "distance": 3.0,
                               "x": 11.0, "y": 64.0, "z": -3.0}]
        self.game.last_pos = (10.0, 64.0, -3.0)
        self.assertEqual(self.game.survive(clock.now()), "flee")
        methods = [m for m, _ in calls]
        self.assertIn("nav.pathTo", methods)
        self.assertNotIn("interact.attackEntity", methods)
        self.assertNotIn("control.jumpOnce", methods)

    def test_a_creeper_is_run_from_not_hit(self):
        calls = self._vitals_bridge({"health": 20.0, "food": 20, "x": 10.0, "y": 64.0, "z": -3.0})
        self.game.connected = True
        self.game.health, self.game.food = 20.0, 20.0
        self.game.last_eat = clock.now()
        self.game.last_flee = 0
        self.game.hostiles = [{"uuid": "c1", "kind": "hostile", "type": "minecraft:creeper", "distance": 2.0,
                               "x": 11.0, "y": 64.0, "z": -3.0}]
        self.assertEqual(self.game.survive(clock.now()), "flee")
        self.assertNotIn("interact.attackEntity", [m for m, _ in calls])

    def test_only_hostiles_are_fought(self):
        calls = self._vitals_bridge({"health": 18.0, "food": 18, "x": 10.0, "y": 64.0, "z": -3.0})
        self.game.connected = True
        self.game.health, self.game.food = 18.0, 18.0
        self.game.last_eat = clock.now()
        self.game.hostiles = []                              # players and items are not hostiles
        self.assertEqual(self.game.survive(clock.now()), "")
        self.assertNotIn("interact.attackEntity", [m for m, _ in calls])

    def _action_bridge(self, blocks=None, inventory=None):
        """A stand-in bridge with a tiny world: for the walk-turn-mine and walk-turn-place rituals."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        calls = []
        world = dict(blocks or {})
        inv = inventory or {"hotbar": [], "main": [], "armor": [], "selectedSlot": 0}

        class Bridge(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                m, p = body["method"], body["params"] or {}
                calls.append((m, p))
                if m == "player.getState":
                    result = {"x": 0.0, "y": 64.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "health": 20.0, "food": 20,
                              "onGround": True}
                elif m == "perception.blocks":
                    pos = p["positions"][0]
                    result = {"blocks": [dict(world.get((pos["x"], pos["y"], pos["z"]),
                                                        {"air": True, "loaded": True}))]}
                elif m == "player.getInventory":
                    result = inv
                elif m == "nav.pathTo":
                    result = {"state": "navigating"}
                elif m == "nav.status":
                    result = {"active": False, "state": "reached", "distance": 0}
                elif m == "interact.breakBlock":
                    world[(p["x"], p["y"], p["z"])] = {"air": True, "loaded": True}
                    result = {"started": True}
                elif m == "interact.placeBlock":
                    vec = {"up": (0, 1, 0), "down": (0, -1, 0), "north": (0, 0, -1),
                           "south": (0, 0, 1), "east": (1, 0, 0), "west": (-1, 0, 0)}[p["face"]]
                    spot = (p["x"] + vec[0], p["y"] + vec[1], p["z"] + vec[2])
                    world[spot] = {"air": False, "loaded": True, "id": "minecraft:cobblestone"}
                    result = {"result": "SUCCESS", "placed": True}
                else:
                    result = {"ok": True}
                out = json.dumps({"ok": True, "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass
        server = HTTPServer(("127.0.0.1", 0), Bridge)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.gw.mc_bridge = (f"http://127.0.0.1:{server.server_port}", "t")
        return calls

    def test_she_turns_her_head_over_a_moment_not_in_a_teleport(self):
        calls = self._action_bridge()
        self.game.aim(10.0, 65.62, 0.0, 0.2)                # eye height: level view
        looks = [p for m, p in calls if m == "control.look"]
        self.assertGreaterEqual(len(looks), 3)              # a hand turns the mouse over several frames
        self.assertAlmostEqual(looks[-1]["yaw"], -90.0, delta=2.0)   # facing +x (east)
        self.assertAlmostEqual(looks[-1]["pitch"], 0.0, delta=2.0)

    def test_she_walks_to_a_block_turns_and_mines_it(self):
        calls = self._action_bridge(blocks={(4, 64, 0): {"air": False, "loaded": True, "id": "minecraft:oak_log",
                                                         "bestSlot": 2, "canHarvest": True, "requiresTool": False}})
        out = self.game.mine_block(4, 64, 0)
        self.assertIn("挖掉了", out)
        methods = [m for m, _ in calls]
        self.assertIn("nav.pathTo", methods)                # she walks there first
        self.assertGreaterEqual(methods.count("control.look"), 3)   # and turns the camera smoothly
        self.assertEqual([p for m, p in calls if m == "inventory.selectHotbar"], [{"slot": 2}])
        self.assertIn("interact.breakBlock", methods)

    def test_she_places_a_block_like_a_player(self):
        calls = self._action_bridge(
            blocks={(2, 63, 0): {"air": False, "loaded": True, "id": "minecraft:dirt"}},
            inventory={"hotbar": [{"slot": 3, "id": "minecraft:cobblestone", "count": 16}],
                       "main": [], "armor": [], "selectedSlot": 3})
        out = self.game.place_block(2, 64, 0)
        self.assertIn("放好了", out)
        placed = [p for m, p in calls if m == "interact.placeBlock"]
        self.assertEqual(placed[0]["face"], "up")           # clicked the block below, face up
        self.assertGreaterEqual([m for m, _ in calls].count("control.look"), 3)

    def test_the_raw_teleport_tools_are_not_hers_but_the_player_like_ones_are(self):
        cfg = mcpfabric_server("unused.js", "http://127.0.0.1:25599", "t")
        for raw in ("break_block", "place_block", "look", "look_at", "use_item", "swap_slots"):
            self.assertFalse(access.allowed(cfg, access.OWNER, raw), raw)
        self._image_bridge()
        names = {t.name for t in self.gw.toolbox.available(ToolContext("self", "乃爱", origin="game"))}
        self.assertIn("mc_mine_block", names)
        self.assertIn("mc_place_block", names)
        self.assertIn("mc_look_at", names)

    def test_when_nothing_moves_her_she_finds_work_herself(self):
        self._bridge([])                                    # nobody around
        self.game.connected = True
        self.game.last_moved = clock.now() - 60
        self.game.last_sched = 0
        self.game.play_busy = True                          # a model turn is running: the body still works
        out = asyncio.run(self.game.auto_work(clock.now()))
        self.assertEqual(out, "collect_blocks")
        log = open(os.path.join(self.dir, "calls.log"), encoding="utf-8").read()
        self.assertIn("collect_blocks", log)
        self.assertIn('"blocks"', log)                      # the tool takes a list of block ids
        self.assertIn('"minecraft:oak_log"', log)
        self.assertEqual(len(self.game.auto_starts), 1)

    def test_a_job_that_stops_moving_her_is_cancelled_and_replaced(self):
        self._bridge([])
        self.game.connected = True
        self.game.last_moved = clock.now() - 60
        self.game.last_sched = 0
        self.game.job_running = lambda: True                # a job claims to be running...
        out = asyncio.run(self.game.auto_work(clock.now()))  # ...but she has not moved for a minute
        self.assertEqual(out, "collect_blocks")
        log = open(os.path.join(self.dir, "calls.log"), encoding="utf-8").read()
        self.assertIn("job_cancel", log)

    def test_a_second_look_right_after_the_first_is_not_taken(self):
        self._image_bridge()
        self.assertEqual(self.game.look_now(), "看到了，画面已经拿到。")
        self.assertIn("刚看过", self.game.look_now())

    def test_mc_skill_does_not_start_another_action_while_a_job_runs(self):
        self._bridge([])
        self.game.job_running = lambda: True
        started = []
        self.game.skills.start = lambda name, **kw: started.append(name) or "开始"
        out = asyncio.run(self.gw.toolbox.run("mc_skill", {"name": "chop_tree"},
                                              ToolContext("mc:Steve", "Steve", "mc")))
        self.assertIn("后台任务", out)
        self.assertEqual(started, [])

    def test_she_stays_with_a_friend_who_is_talking_instead_of_wandering_off(self):
        steve = GameBridge.offline_uuid("Steve")
        self._bridge([{"uuid": steve, "type": "minecraft:player", "kind": "player",
                       "x": 20, "y": 64, "z": 5, "distance": 12}])
        started = []
        self.game.skills.start = lambda name, **kw: started.append((name, kw)) or "开始走了"
        self.game.connected = True
        self.game.known_players.add("Steve")
        self.game.last_chat = clock.now() - 10
        self.game.last_moved = clock.now() - 60
        self.game.last_sched = 0
        self.assertEqual(asyncio.run(self.game.auto_work(clock.now())), "friend")
        self.assertEqual(started, [("go_to_player", {"player": "Steve"})])

    def test_auto_work_waits_while_she_is_moving_or_just_started_something(self):
        self._bridge([])
        self.game.connected = True
        self.game.last_moved = clock.now()                  # moving right now
        self.assertEqual(asyncio.run(self.game.auto_work(clock.now())), "")
        self.game.last_moved = clock.now() - 60
        self.game.last_sched = clock.now()                  # just started something
        self.assertEqual(asyncio.run(self.game.auto_work(clock.now())), "")


if __name__ == "__main__":
    unittest.main()
