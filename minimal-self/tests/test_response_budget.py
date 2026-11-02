import asyncio
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.evaluation.probes import Replica
from src.levels.l8_agency.toolbox import ToolContext
from src.levels.l10_consciousness.response_budget import reply_request_params


class Recorder:
    supports_tools = True

    def __init__(self):
        self.requests = []

    def chatml_to_messages(self, prompt):
        return [{"role": "user", "content": prompt}]

    def chat_tools(self, messages, tools, max_tokens=300, temperature=.7, tool_choice=None):
        self.requests.append(("native", max_tokens, temperature))
        return {"content": "Thought: 简短思考\nConfidence: 80\nEmotion: neutral\nResponse: 已收到", "tool_calls": []}

    def generate_json(self, prompt, max_tokens=300, temperature=.7, media=None):
        self.requests.append(("text", max_tokens, temperature))
        return {"reply": "Thought: 简短思考\nConfidence: 80\nResponse: 已收到"}


class BudgetTests(unittest.TestCase):
    def test_real_protocols_share_one_reserve_only_when_opted_in(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"source.sqlite"
            with closing(sqlite3.connect(source)) as db:
                db.execute("CREATE TABLE sentinel(x)")
            recorder = Recorder()
            with Replica(source, recorder) as rep:
                gw = rep.gw
                ctx = ToolContext("budget_test", "测试")
                params = {"max_tokens":91, "temperature":.46, "delay":15.}
                for enabled in (False, True, False):
                    gw.configure_core(enabled=enabled, welfare=enabled)
                    self.assertEqual(gw._generate_response("test", "neutral", params), "已收到")
                    self.assertEqual(recorder.requests[-1], ("text", 241 if enabled else 91, .46))
                    self.assertEqual(asyncio.run(gw._respond_native("test", "neutral", params, None, ctx)), "已收到")
                    self.assertEqual(recorder.requests[-1], ("native", 241, .46))
                    self.assertEqual(params, {"max_tokens":91, "temperature":.46, "delay":15.})

    def test_helper_preserves_legacy_text_fields_and_native_default(self):
        params = {"temperature":.7, "delay":0.}
        self.assertEqual(reply_request_params(params), params)
        self.assertEqual(reply_request_params(params, native=True)["max_tokens"], 350)
        self.assertNotIn("max_tokens", params)

    def test_reference_budget_is_not_expanded_twice(self):
        params = {"max_tokens":276, "temperature":.65}
        self.assertEqual(reply_request_params(params, native=True, core_enabled=True)["max_tokens"], 426)
        self.assertEqual(reply_request_params(params, core_enabled=True)["max_tokens"], 426)
