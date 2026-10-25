"""
Her hearing: QQ voice messages to text, with Whisper large-v3-turbo, on 127.0.0.1 only.

    .venv-lora/Scripts/python scripts/senses/hear_server.py [--port 8091]

POST /hear {"path": "<wav or mp3 file NapCat converted>"} -> {"text": "..."}
The audio is read and forgotten; nothing is stored here.
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

def load(path: str):
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:                                   # simple linear resample to what Whisper expects
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(ROOT, "models", "whisper-large-v3-turbo"))
    p.add_argument("--port", type=int, default=8091)
    a = p.parse_args(argv)
    asr = pipeline("automatic-speech-recognition", model=a.model, dtype=torch.float16, device="cuda")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with lock:
                    out = asr(load(req["path"]), generate_kwargs={"language": "zh", "task": "transcribe"})
                self._send(200, {"text": (out.get("text") or "").strip()})
            except Exception as e:
                self._send(500, {"error": str(e)[:300]})

        def log_message(self, *args):
            pass

    print(f"她的听觉在 http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()

if __name__ == "__main__":
    main(sys.argv[1:])
