import json
import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "lora"))
import build_dataset  # noqa: E402
from local_engine import HybridEngine  # noqa: E402

class Fake:
    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def generate_json(self, prompt, max_tokens=200, temperature=0.7):
        self.calls += 1
        return {"reply": self.reply, **({"error": self.error} if self.error else {})}

CHAT = "<|im_start|>system\n你是乃爱<|im_end|>\n<|im_start|>user\n在吗<|im_end|>\n<|im_start|>assistant\n"
DREAM = "<|im_start|>system\n整理今天真实发生过的事。\n输出 JSON：{}<|im_end|>\n<|im_start|>assistant\n"

class TestHybrid(unittest.TestCase):
    def test_her_model_talks_structured_work_goes_remote(self):
        local, remote = Fake("在呢"), Fake("{}")
        h = HybridEngine(local, remote)
        self.assertEqual(h.generate_json(CHAT)["reply"], "在呢")
        h.generate_json(DREAM)
        self.assertEqual((local.calls, remote.calls), (1, 1))

    def test_falls_back_when_her_model_is_down(self):
        h = HybridEngine(Fake("", error="connection refused"), Fake("在的"))
        self.assertEqual(h.generate_json(CHAT)["reply"], "在的")

class TestDataset(unittest.TestCase):
    def _db(self, rows):
        path = os.path.join(tempfile.mkdtemp(prefix="noa_ds_"), "m.sqlite")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY, user_id TEXT, role TEXT, content TEXT, "
                     "timestamp REAL, session_id TEXT, context_json TEXT, group_id TEXT)")
        conn.execute("CREATE TABLE users (user_id TEXT, nickname TEXT)")
        conn.execute("INSERT INTO users VALUES ('u1', '张小明')")
        for i, (role, text) in enumerate(rows):
            conn.execute("INSERT INTO interactions (user_id, role, content, timestamp, group_id) VALUES (?, ?, ?, ?, NULL)",
                         ("u1", role, text, 1000.0 + i * 30))
        conn.commit()
        conn.close()
        return path

    def test_only_her_real_voice_with_others_masked(self):
        db = self._db([("user", "我是张小明，QQ 123456789"), ("assistant", "哦，记住了"),
                       ("user", "摸摸头"), ("assistant", "*蹭蹭你的手* 喵～"),
                       ("user", "你懂什么"), ("assistant", "你到底想说什么意思？？"),
                       ("user", "今天好累"), ("assistant", "那早点睡吧\n别熬了")])
        out = tempfile.mkdtemp(prefix="noa_ds_out_")
        report = build_dataset.build(db, out, val_share=0.0)
        self.assertEqual(report["kept"], 2)
        self.assertEqual(report["dropped"], {"动作描写": 1, "语气太冲": 1})
        with open(os.path.join(out, "train.jsonl"), encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("张小明", text)
        self.assertNotIn("123456789", text)
        self.assertIn("[某人]", text)
        targets = [json.loads(l)["messages"][-1]["content"] for l in text.splitlines()]
        self.assertEqual(sorted(targets), ["哦，记住了", "那早点睡吧\n别熬了"])

if __name__ == "__main__":
    unittest.main()
