"""Her own room (keepsakes), the read-only window into herself, and the senses that feed them:
long pictures she can read, originals she can keep, hearing that starts with her."""
import asyncio
import io
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core import clock  # noqa: E402
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import START
from src.levels.l2_memory.keepsakes import Keepsakes
from src.levels.l8_agency import access
from src.levels.l8_agency.toolbox import ToolContext
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

def tmp_db():
    return os.path.join(tempfile.mkdtemp(prefix="noa_room_"), "m.sqlite")

def png(w=64, h=48, color=(200, 80, 80)) -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (w, h), color).save(out, format="PNG")
    return out.getvalue()

def make_gw(llm=None):
    clock.set_clock(SimClock(START + 2 * 3600))
    gw = GlobalWorkspace(EventBus(), llm, db_path=tmp_db(), rng=random.Random(0))
    gw.evolve_creator = "owner"
    return gw

def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)

class TestKeepsakes(unittest.TestCase):
    def setUp(self):
        self.db = tmp_db()
        self.room = Keepsakes(self.db, os.path.join(os.path.dirname(self.db), "data", "keepsakes"))

    def test_keep_a_picture_once_and_find_it(self):
        pic = png()
        a = self.room.keep("picture", "小猫", "好可爱", "猫,可爱", data=pic, mime="image/png", source="白羽")
        self.assertTrue(a["ok"])
        again = self.room.keep("picture", "又是小猫", data=pic, mime="image/png")
        self.assertTrue(again["again"])                       # the same bytes are not kept twice
        self.assertEqual(again["id"], a["id"])
        found = self.room.list("可爱")
        self.assertEqual([i["id"] for i in found], [a["id"]])
        item = self.room.get(a["id"])
        self.assertEqual(self.room.data(item), pic)           # the original, untouched
        self.assertEqual(self.room.picture(item)["mime"], "image/jpeg")

    def test_notes_links_favorites_and_forgetting(self):
        n = self.room.keep("note", content="今天的晚霞是橘子味的")
        l = self.room.keep("link", content="https://example.com/song")
        self.room.update(l["id"], favorite=True)
        self.assertEqual(self.room.list()[0]["id"], l["id"])  # favorites first
        self.assertIn("1 条摘记", self.room.describe())
        f = self.room.keep("file", "笔记", data=b"hello", name="a.txt")
        path = self.room.path(self.room.get(f["id"]))
        self.assertTrue(os.path.exists(path))
        self.room.forget(f["id"])
        self.assertIsNone(self.room.get(f["id"]))
        self.assertFalse(os.path.exists(path))                # nothing else pointed at the file
        self.assertTrue(self.room.get(n["id"]))

    def test_limits(self):
        self.room.MAX_PER_DAY = 1
        self.assertTrue(self.room.keep("note", content="一")["ok"])
        self.assertIn("明天", self.room.keep("note", content="二")["note"])
        self.room.MAX_PER_DAY = 99
        self.room.MAX_FILE = 3
        self.assertIn("太大", self.room.keep("file", data=b"12345", name="x.bin")["note"])
        self.assertFalse(self.room.keep("note", content="  ")["ok"])

class TestRoomTools(unittest.TestCase):
    def test_she_keeps_a_picture_she_was_shown_and_looks_at_it_again(self):
        gw = make_gw()
        gw.inference_engine = type("E", (), {"supports_images": True})()
        pic = png(80, 60)
        gw._shelve_pictures("g1", [{"data": b"small", "original": pic, "original_mime": "image/png",
                                     "mime": "image/jpeg", "name": "abc.png"}], "白羽", "u1", clock.now())
        ctx = ToolContext("u1", "白羽", "g1")
        out = run(gw.toolbox.run("keep", {"what": "picture", "title": "晚霞", "favorite": True}, ctx))
        self.assertIn("收好了", out)
        item = gw.keepsakes.list()[0]
        self.assertEqual(gw.keepsakes.data(item), pic)        # the original, not the chat-sized copy
        self.assertTrue(item["favorite"])
        self.assertIn("白羽", item["source"])
        listed = run(gw.toolbox.run("my_keepsakes", {}, ctx))
        self.assertIn("晚霞", listed)
        looked = run(gw.toolbox.run("look_at_keepsake", {"id": item["id"]}, ctx))
        self.assertIn(f"[收藏:{item['id']}]", looked)
        self.assertEqual(gw.pending_mc_media["mime"], "image/jpeg")   # the picture rides with the next step
        seg = gw.render_message(f"[收藏:{item['id']}]")
        self.assertEqual(seg[0]["type"], "image")
        self.assertTrue(seg[0]["data"]["file"].startswith("base64://"))
        self.assertIn("晚霞", gw.render_own(f"[收藏:{item['id']}]"))
        plan = gw.plan_messages(f"给你看[收藏:{item['id']}]")
        self.assertEqual([p for _, p in plan], ["给你看", f"[收藏:{item['id']}]"])

    def test_no_picture_at_hand_and_old_pictures_are_let_go(self):
        gw = make_gw()
        ctx = ToolContext("u1", "白羽", None)
        self.assertIn("没有可以收的图", run(gw.toolbox.run("keep", {"what": "picture"}, ctx)))
        gw._shelve_pictures("u:u1", [{"data": b"x", "mime": "image/jpeg"}], "白羽", "u1", clock.now() - 3600)
        self.assertEqual(gw.pictures_at_hand("u:u1"), [])

    def test_keep_a_file_from_the_chat_and_read_it_later(self):
        gw = make_gw()
        folder = os.path.join(gw.inbox_dir, "u1")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "123-读书笔记.txt"), "w", encoding="utf-8") as f:
            f.write("第一章：小王子和玫瑰")
        ctx = ToolContext("u1", "白羽", None)
        self.assertIn("收好了", run(gw.toolbox.run("keep", {"what": "file", "name": "读书笔记.txt"}, ctx)))
        shutil.rmtree(folder)                                  # the inbox forgets; her room does not
        item = gw.keepsakes.list()[0]
        self.assertIn("小王子", run(gw.toolbox.run("look_at_keepsake", {"id": item["id"]}, ctx)))
        self.assertIn("扔掉", run(gw.toolbox.run("change_keepsake", {"id": item["id"], "forget": True}, ctx)))

    def test_room_tools_in_her_own_time_but_not_in_the_game(self):
        gw = make_gw()
        free = {t.name for t in gw.toolbox.available(ToolContext("self", "乃爱", None, origin="autonomous"))}
        self.assertTrue({"keep", "my_keepsakes", "look_inside"} <= free)
        game = {t.name for t in gw.toolbox.available(ToolContext("self", "乃爱", None, origin="game"))}
        self.assertFalse({"keep", "look_inside"} & game)

class TestLookInside(unittest.TestCase):
    def setUp(self):
        self.gw = make_gw()
        self.gw.episodes.add("白羽说她养了一只叫团子的猫", kind="chat", user_id="u1")

    def look(self, args, user="owner", group=None, origin="reply"):
        return run(self.gw.toolbox.run("look_inside", args, ToolContext(user, "x", group, origin=origin)))

    def test_code_is_readable_but_secrets_and_outside_are_not(self):
        self.assertIn("minimal-self/", self.look({"what": "code"}))
        page = self.look({"what": "code", "path": "minimal-self/src/levels/l2_memory/keepsakes.py", "start": 1})
        self.assertIn("class Keepsakes", page)
        self.assertIn("往下看用 start=", page)
        self.assertIn("看不了", self.look({"what": "code", "path": "config.json"}))
        self.assertIn("看不了", self.look({"what": "code", "path": "data/workspace"}))
        self.assertIn("只能看", self.look({"what": "code", "path": "../../.."}))
        self.assertIn("keepsakes.py", self.look({"what": "code", "search": "class Keepsakes"}))
        listing = self.look({"what": "code"})
        self.assertNotIn("config.json", listing)
        self.assertNotIn("memory.sqlite", listing)

    def test_data_is_read_only_and_rows_are_private(self):
        overview = self.look({"what": "data"}, user="stranger", group="g1")
        self.assertIn("episodes", overview)
        self.assertNotIn("团子", self.look({"what": "data", "table": "episodes"}, user="stranger", group="g1"))
        self.assertIn("团子", self.look({"what": "data", "table": "episodes"}))
        self.assertIn("团子", self.look({"what": "data", "table": "episodes"}, user="self", origin="autonomous"))
        self.assertIn("创造者", self.look({"what": "data", "sql": "SELECT 1"}, user="self", origin="autonomous"))
        self.assertIn("团子", self.look({"what": "data", "sql": "SELECT content FROM episodes"}))
        before = sqlite3.connect(self.gw.memory.db.db_path).execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        self.assertIn("不能改", self.look({"what": "data", "sql": "DELETE FROM episodes"}))
        self.assertIn("不能改", self.look({"what": "data", "sql": "SELECT 1; DELETE FROM episodes"}))
        # even a statement that slips past the check cannot write: the connection itself is read-only
        out = self.look({"what": "data", "sql": "WITH x AS (SELECT 1) DELETE FROM episodes"})
        self.assertIn("没查成", out)
        after = sqlite3.connect(self.gw.memory.db.db_path).execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        self.assertEqual(before, after)

    def test_process_and_logs(self):
        out = self.look({"what": "process"}, user="stranger", group="g1")
        self.assertIn("pid", out)
        self.assertIn("QQ 连接", out)
        self.assertIn("只在我自己的时间", self.look({"what": "logs"}, user="stranger", group="g1"))
        from console import LogBuffer
        import logging
        self.gw.log_buffer = LogBuffer()
        self.gw.log_buffer.emit(logging.LogRecord("napcat.qq", logging.WARNING, "", 0, "ocr 超时", None, None))
        self.assertIn("ocr 超时", self.look({"what": "logs", "level": "warning"}))

    def test_she_knows_she_has_them(self):
        persona = self.gw._build_persona_prompt()
        self.assertIn("look_inside", persona)
        self.assertIn("小屋", persona)

class TestPictures(unittest.TestCase):
    def test_a_long_screenshot_is_cut_into_readable_pieces(self):
        from napcat.senses import Senses
        s = Senses()
        tall = png(1080, 6000)
        pic = s.prepare_image(tall)
        self.assertGreaterEqual(len(pic["parts"]), 3)
        self.assertLessEqual(len(pic["parts"]), Senses.MAX_TILES)
        from PIL import Image
        first = Image.open(io.BytesIO(pic["parts"][0]))
        self.assertEqual(first.size[0], 1080)                  # not squashed to 1024 on the long side
        self.assertEqual(pic["original"], tall)

    def test_small_and_wide_pictures_go_as_they_are(self):
        from napcat.senses import Senses
        s = Senses()
        small = png(300, 200)
        self.assertEqual(s.prepare_image(small), {"data": small, "mime": "image/png"})
        self.assertNotIn("parts", s.prepare_image(png(2400, 1000)))

    def test_an_expired_url_falls_back_to_the_local_file(self):
        from napcat.senses import Senses
        s = Senses()
        path = os.path.join(tempfile.mkdtemp(), "a.png")
        with open(path, "wb") as f:
            f.write(png())
        self.assertIsNotNone(s.fetch_image({"url": "http://127.0.0.1:9/gone.png", "path": path}))

    def test_pieces_go_to_the_model_in_order(self):
        from deepseek_engine import DeepSeekEngine
        messages = [{"role": "user", "content": "看"}]
        out = DeepSeekEngine.with_media(messages, [{"data": b"a", "mime": "image/png", "parts": [b"1", b"2", b"3"]},
                                                   {"data": b"b", "mime": "image/png"}])
        blocks = [p for p in out[0]["content"] if p["type"] == "image_url"]
        self.assertEqual(len(blocks), 4)

    def test_she_is_told_a_long_picture_was_cut(self):
        gw = make_gw()
        prompt = gw._construct_prompt("看这个", "", {"emotion": "neutral"}, None, "u1",
                                      media=[{"data": b"x", "parts": [b"1", b"2"]}])
        self.assertIn("切成了 2 段", prompt)

class TestHearing(unittest.TestCase):
    def test_missing_install_is_reported_not_crashed(self):
        from napcat.senses import Senses
        s = Senses(hear_url="http://127.0.0.1:9")
        s.configure_hearing(tempfile.mkdtemp())
        self.assertFalse(s.ensure_hearing())
        self.assertIn("缺少", s.hear_problem)

    def test_it_starts_the_server_once_and_waits_for_it(self):
        from napcat import senses as mod
        root = tempfile.mkdtemp()
        for p in ("scripts/senses/hear_server.py", "models/whisper-large-v3-turbo/x"):
            os.makedirs(os.path.dirname(os.path.join(root, p)), exist_ok=True)
            open(os.path.join(root, p), "w").close()
        s = mod.Senses(hear_url="http://127.0.0.1:9")
        import sys
        s.configure_hearing(root, python=sys.executable)
        started = []

        class Proc:
            pid, returncode = 1, None
            def poll(self): return None
            def terminate(self): started.append("stopped")
            def wait(self, timeout=None): return 0

        real = mod.subprocess.Popen
        mod.subprocess.Popen = lambda *a, **k: started.append(a[0]) or Proc()
        try:
            self.assertFalse(s.ensure_hearing(now=1000))
            self.assertFalse(s.ensure_hearing(now=1010))      # still loading: not started twice
            self.assertEqual(len([x for x in started if x != "stopped"]), 1)
            self.assertTrue(any("hear_server.py" in str(a) for a in started[0]))
            s.stop_hearing()
            self.assertIn("stopped", started)
        finally:
            mod.subprocess.Popen = real

        s.configure_hearing(root, python=sys.executable, enabled=False)
        self.assertFalse(s.ensure_hearing())

class TestBackupMirrorsTheRoom(unittest.TestCase):
    def test_files_are_mirrored_and_never_deleted(self):
        from src.core.backup import BackupManager
        gw = make_gw()
        gw.keepsakes.keep("file", "a", data=b"hello", name="a.txt")
        bm = BackupManager(gw.memory.db.db_path, file_dirs=[gw.keepsakes.folder])
        r = bm.run(clock.now())
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["files"], 1)
        mirrored = os.path.join(bm.folder, "keepsakes")
        self.assertEqual(len(os.listdir(mirrored)), 1)
        self.assertEqual(bm.mirror_files(), 0)                  # already there

class TestConsoleRoom(unittest.TestCase):
    def test_room_and_body_endpoints(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from console import create_console_router
        gw = make_gw()
        k = gw.keepsakes.keep("picture", "晚霞", data=png(), mime="image/png")
        app = FastAPI()
        app.include_router(create_console_router(gw))
        c = TestClient(app)
        room = c.get("/console/api/room").json()
        self.assertEqual(room["items"][0]["title"], "晚霞")
        self.assertNotIn("file", room["items"][0])
        thumb = c.get(f"/console/api/room/{k['id']}/file?thumb=1")
        self.assertEqual(thumb.headers["content-type"], "image/jpeg")
        self.assertEqual(c.get(f"/console/api/room/{k['id']}/file").content, png())
        body = c.get("/console/api/body").json()
        self.assertIn("uptime", body)
        self.assertEqual(c.post("/console/api/room/forget", json={"id": k["id"]}).status_code, 200)
        self.assertEqual(c.get(f"/console/api/room/{k['id']}/file").status_code, 404)
        self.assertIs(gw.log_buffer.__class__.__name__, "LogBuffer")

if __name__ == "__main__":
    unittest.main()
