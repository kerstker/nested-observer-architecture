"""
Her own model as an inference engine (same interface as DeepSeekEngine.generate_json).

The model itself runs in its own process (scripts/lora/serve.py, in .venv-lora); this client only
speaks HTTP to 127.0.0.1, so the live app needs no torch. HybridEngine lets her own model do the
talking and leaves structured work (anything that must come back as JSON: sleep consolidation,
memory extraction) to the larger remote model, falling back to it whenever her model is not up.
"""
import json
import logging
import urllib.request
from typing import Any, Dict

from deepseek_engine import DeepSeekEngine

logger = logging.getLogger(__name__)
_LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 127.0.0.1 never goes via a proxy

class LocalEngine:
    # Her own model is text-only: pictures still go through the local vision server's description.
    supports_images = False

    def __init__(self, url: str = "http://127.0.0.1:8093", timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.usage = {"calls": 0, "failures": 0}

    def available(self) -> bool:
        try:
            with _LOOPBACK.open(self.url + "/health", timeout=3) as r:
                return json.load(r).get("ok", False)
        except Exception:
            return False

    def generate_json(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        body = json.dumps({"messages": DeepSeekEngine.chatml_to_messages(prompt), "max_tokens": max_tokens,
                           "temperature": temperature}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url + "/generate", data=body, headers={"Content-Type": "application/json"})
        self.usage["calls"] += 1
        try:
            with _LOOPBACK.open(req, timeout=self.timeout) as r:
                return {"reply": json.load(r).get("reply", "")}
        except Exception as e:
            self.usage["failures"] += 1
            logger.error(f"[Local] 调用失败：{e}")
            return {"reply": "", "error": str(e)}

    def usage_summary(self) -> str:
        return f"本地模型 {self.usage['calls']} 次调用（失败 {self.usage['failures']}）"

class HybridEngine:
    """Her own model talks; JSON-structured work goes to the remote model; any local failure falls back."""
    # The talking is done by the text-only local model, so pictures take the described path.
    supports_images = False

    def __init__(self, local: LocalEngine, remote):
        self.local, self.remote = local, remote

    def generate_json(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        if self.remote is not None and DeepSeekEngine._wants_json(prompt):
            return self.remote.generate_json(prompt, max_tokens, temperature)
        out = self.local.generate_json(prompt, max_tokens, temperature)
        if (out.get("error") or not out.get("reply")) and self.remote is not None:
            return self.remote.generate_json(prompt, max_tokens, temperature)
        return out

    def usage_summary(self) -> str:
        remote = self.remote.usage_summary() if self.remote is not None and hasattr(self.remote, "usage_summary") else ""
        return f"{self.local.usage_summary()}；DeepSeek：{remote}"
