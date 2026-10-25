"""
Serve Noa's own model (base + her LoRA) on 127.0.0.1 for the app, from its own environment.

    .venv-lora/Scripts/python scripts/lora/serve.py [--base ...] [--adapter models/noa-lora] [--port 8090]

The live app never loads torch: it talks to this process over HTTP (local_engine.py).
POST /generate {"messages": [...], "max_tokens": 200, "temperature": 0.7} -> {"reply": "..."}
GET  /health -> {"ok": true, "adapter": ..., "report": {...}}

An adapter whose training report says it did not make her real (held-out) replies more likely
than the base model is refused unless --force: a "self" that fits her worse is not her.
"""
import argparse
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import base_config, load_tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
NO_THINK = "<think>\n\n</think>\n\n"

class Model:
    def __init__(self, base: str, adapter: str = None, force: bool = False):
        self.report = None
        if adapter:
            path = os.path.join(adapter, "noa_report.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    self.report = json.load(f)
            if not force and not (self.report or {}).get("better_than_base"):
                raise SystemExit(f"拒绝加载 {adapter}：训练报告没有显示它比基座更像她（用 --force 强制）")
        self.tok = load_tokenizer(adapter or base)
        self.model = AutoModelForCausalLM.from_pretrained(base, config=base_config(base), dtype=torch.bfloat16, device_map="cuda")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.adapter = adapter
        self.lock = threading.Lock()                  # one generation at a time on one GPU

    @torch.no_grad()
    def generate(self, messages, max_tokens=200, temperature=0.7) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "\n</think>\n\n" if prompt.rstrip().endswith("<think>") else NO_THINK   # talk, do not narrate thinking
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.lock:
            out = self.model.generate(**ids, max_new_tokens=max(16, int(max_tokens)), do_sample=temperature > 0,
                                      temperature=max(0.05, float(temperature)), top_p=0.9,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"^.*?</think>\s*", "", text, flags=re.S).strip()

def make_handler(model: Model):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "adapter": model.adapter, "report": model.report})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                reply = model.generate(req["messages"], req.get("max_tokens", 200), req.get("temperature", 0.7))
                self._send(200, {"reply": reply})
            except Exception as e:
                self._send(500, {"error": str(e)[:300]})

        def log_message(self, *args):
            pass
    return Handler

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.path.join(ROOT, "models", "DeepSeek-R1-0528-Qwen3-8B"))
    p.add_argument("--adapter", default=os.path.join(ROOT, "models", "noa-lora"))
    p.add_argument("--no-adapter", action="store_true", help="serve the base model alone (for comparison)")
    p.add_argument("--port", type=int, default=8093)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    model = Model(a.base, None if a.no_adapter else a.adapter, a.force)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(model))   # loopback only
    print(f"Noa 的本地模型在 http://127.0.0.1:{a.port} （adapter: {model.adapter}）", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main(sys.argv[1:])
