import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from deepseek_engine import DeepSeekEngine  # noqa: E402

class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        return self._payload

def ok(content, usage=None):
    return FakeResponse(200, {"choices": [{"message": {"content": content}}],
                              "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 80}})

class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "body": json})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

def engine(responses):
    session = FakeSession(responses)
    e = DeepSeekEngine(api_key="test-key", session=session, max_retries=2)
    return e, session

class TestChatML(unittest.TestCase):
    def test_conversation_prompt(self):
        prompt = "<|im_start|>system\n你是乃爱<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(DeepSeekEngine.chatml_to_messages(prompt),
                         [{"role": "system", "content": "你是乃爱"}, {"role": "user", "content": "你好"}])

    def test_system_only_prompt_gets_a_user_turn(self):
        msgs = DeepSeekEngine.chatml_to_messages("<|im_start|>system\n产生一个念头\n<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")

    def test_tool_followup_keeps_assistant_turn(self):
        prompt = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n天气<|im_end|>\n<|im_start|>assistant\n"
                  "Action: SEARCH query=\"天气\"<|im_end|>\n<|im_start|>user\n[工具结果]\n晴<|im_end|>\n<|im_start|>assistant\n")
        roles = [m["role"] for m in DeepSeekEngine.chatml_to_messages(prompt)]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_plain_prompt(self):
        self.assertEqual(DeepSeekEngine.chatml_to_messages("Analyze this"), [{"role": "user", "content": "Analyze this"}])

class TestRequests(unittest.TestCase):
    def test_request_body(self):
        e, s = engine([ok("喵～")])
        out = e.generate_json("<|im_start|>user\n你好<|im_end|>", max_tokens=108, temperature=0.65)
        self.assertEqual(out, {"reply": "喵～"})
        body = s.requests[0]["body"]
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["max_tokens"], 108)
        self.assertEqual(body["temperature"], 0.65)
        self.assertNotIn("response_format", body)
        self.assertEqual(s.requests[0]["headers"]["Authorization"], "Bearer test-key")
        self.assertTrue(s.requests[0]["url"].endswith("/chat/completions"))

    def test_json_mode(self):
        e, s = engine([ok('{"diary": "今天很开心", "facts": []}')])
        out = e.generate_json("输出 JSON：{...}", max_tokens=200)
        self.assertEqual(out["diary"], "今天很开心")
        self.assertEqual(s.requests[0]["body"]["response_format"], {"type": "json_object"})
        self.assertGreaterEqual(s.requests[0]["body"]["max_tokens"], 800)

    def test_retries_server_errors(self):
        e, s = engine([FakeResponse(500, text="busy"), ConnectionError("reset"), ok("好了")])
        import deepseek_engine
        original = deepseek_engine.time.sleep
        deepseek_engine.time.sleep = lambda s: None
        try:
            out = e.generate_json("<|im_start|>user\nhi<|im_end|>")
        finally:
            deepseek_engine.time.sleep = original
        self.assertEqual(out["reply"], "好了")
        self.assertEqual(len(s.requests), 3)

    def test_client_error_is_not_retried(self):
        e, s = engine([FakeResponse(401, text="invalid key")])
        out = e.generate_json("<|im_start|>user\nhi<|im_end|>")
        self.assertEqual(out["reply"], "")
        self.assertIn("401", out["error"])
        self.assertEqual(len(s.requests), 1)
        self.assertEqual(e.usage["failures"], 1)

    def test_usage_is_tracked(self):
        e, _ = engine([ok("a"), ok("b")])
        e.generate_json("x")
        e.generate_json("y")
        self.assertEqual(e.usage["calls"], 2)
        self.assertEqual(e.usage["prompt_tokens"], 200)
        self.assertIn("缓存命中 80%", e.usage_summary())

    def test_missing_key(self):
        original = DeepSeekEngine._lookup_key
        DeepSeekEngine._lookup_key = staticmethod(lambda: None)
        try:
            with self.assertRaises(ValueError):
                DeepSeekEngine()
        finally:
            DeepSeekEngine._lookup_key = original

if __name__ == "__main__":
    unittest.main()
