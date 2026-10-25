import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from console import create_console_router  # noqa: E402
from src.core.event_bus import EventBus  # noqa: E402
from src.evaluation.scenarios import ScriptedLLM, StubSearch  # noqa: E402
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace  # noqa: E402

def make_client(token=None, host="testclient"):
    db = os.path.join(tempfile.mkdtemp(prefix="noa_console_"), "c.sqlite")
    gw = GlobalWorkspace(EventBus(), ScriptedLLM("喵？"), db_path=db)
    gw.search_tool = StubSearch()
    app = FastAPI()
    app.include_router(create_console_router(gw, token))
    return TestClient(app, client=(host, 50000)), gw

class TestConsole(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_page_and_state(self):
        client, _ = make_client()
        self.assertIn("乃爱 · 控制台", client.get("/console").text)
        s = client.get("/console/api/state").json()
        for key in ("energy", "emotions", "needs", "stream", "traits", "people", "episodes", "causes"):
            self.assertIn(key, s)

    def test_say_becomes_experience(self):
        client, _ = make_client()
        r = client.post("/console/api/say", json={"text": "摸摸头", "nickname": "小明"}).json()
        self.assertIn("喵？", r["reply"])
        self.assertGreater(r["state"]["emotions"]["joy"], 0)
        self.assertTrue(any("小明" in e["content"] for e in r["state"]["episodes"]))
        self.assertTrue(r["state"]["stream"])

    def test_actions(self):
        client, gw = make_client()
        self.assertTrue(client.post("/console/api/act", json={"action": "sleep"}).json()["state"]["sleeping"])
        self.assertFalse(client.post("/console/api/act", json={"action": "wake"}).json()["state"]["sleeping"])
        fed = client.post("/console/api/act", json={"action": "feed"}).json()
        self.assertEqual(fed["state"]["needs"]["hunger"], 0)
        self.assertEqual(client.post("/console/api/act", json={"action": "fly"}).status_code, 400)

    def test_remote_access_needs_token(self):
        client, _ = make_client(host="192.168.1.20")
        self.assertEqual(client.get("/console/api/state").status_code, 403)

    def test_token(self):
        client, _ = make_client(token="s3cret", host="192.168.1.20")
        self.assertEqual(client.get("/console/api/state").status_code, 401)
        self.assertEqual(client.get("/console/api/state", headers={"X-Console-Token": "s3cret"}).status_code, 200)
        self.assertEqual(client.get("/console?token=s3cret").status_code, 200)

if __name__ == "__main__":
    unittest.main()
