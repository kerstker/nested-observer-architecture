import json
import logging
import os
import re
import sys
import threading
import time
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger(__name__)

_CHATML_TURN = re.compile(r"<\|im_start\|>(system|user|assistant)\n?(.*?)(?=<\|im_start\|>|\Z)", re.S)


class DeepSeekEngine:
    """DeepSeek API 推理引擎，与 InferenceEngine 接口相同：generate_json(prompt, max_tokens, temperature)

    - 现有 prompt 都是 ChatML 字符串（<|im_start|>...），这里转换成 messages，提示词本身不用改
    - 关闭思考模式：思考模式下 temperature 会被忽略（Noa 的"激动程度 -> 温度"就失效了），
      而且推理过程会占掉本来就很短的 max_tokens
    - API Key 只从环境变量 DEEPSEEK_API_KEY 读取，不写进 config.json
    """

    def __init__(self, api_key: str = None, model: str = "deepseek-flash",
                 base_url: str = "https://api.deepseek.com", timeout: float = 60.0,
                 max_retries: int = 2, session: requests.Session = None):
        self.api_key = api_key or self._lookup_key()
        if not self.api_key:
            raise ValueError("缺少 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY")
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._usage_lock = threading.Lock()
        self.usage = {"calls": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0}

    @staticmethod
    def _lookup_key() -> Optional[str]:
        """进程环境变量优先；Windows 上再读注册表里的用户环境变量
        （SetEnvironmentVariable(..., 'User') 设的值，不会传给设置之前就已启动的程序）"""
        key = os.environ.get("DEEPSEEK_API_KEY")
        if key or sys.platform != "win32":
            return key
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg:
                return winreg.QueryValueEx(reg, "DEEPSEEK_API_KEY")[0] or None
        except OSError:
            return None

    @staticmethod
    def chatml_to_messages(prompt: str) -> List[Dict[str, str]]:
        """把 ChatML 字符串拆成 API 的 messages 列表。结尾空的 assistant 轮（生成起点）会被去掉。"""
        messages = []
        for role, body in _CHATML_TURN.findall(prompt):
            content = body.split("<|im_end|>")[0].strip()
            if not content:
                continue
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": role, "content": content})
        if not messages:
            return [{"role": "user", "content": prompt.strip()}]
        # 只有 system 的 prompt（内心念头、主动开场白）需要一个 user 轮来触发生成
        if not any(m["role"] == "user" for m in messages):
            messages.append({"role": "user", "content": "（请按上面的要求直接输出）"})
        return messages

    @staticmethod
    def _wants_json(prompt: str) -> bool:
        return "Output JSON" in prompt or "输出 JSON" in prompt

    def generate_json(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        wants_json = self._wants_json(prompt)
        body = {
            "model": self.model,
            "messages": self.chatml_to_messages(prompt),
            "max_tokens": max(max_tokens, 800) if wants_json else max_tokens,   # JSON 被截断就无法解析
            "temperature": temperature,
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        if wants_json:
            body["response_format"] = {"type": "json_object"}

        try:
            text = self._post(body)
        except Exception as e:
            with self._usage_lock:
                self.usage["failures"] += 1
            logger.error(f"[DeepSeek] 调用失败：{e}")
            return {"reply": "", "error": str(e)}

        if wants_json:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                logger.warning("[DeepSeek] JSON 模式返回的内容无法解析，按普通文本处理")
        return {"reply": text}

    def _post(self, body: Dict[str, Any]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(self.url, headers=headers, json=body, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    # 4xx（Key 错误、参数错误、余额不足）重试也没用
                    raise ValueError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                self._record_usage(data.get("usage") or {})
                return (data["choices"][0]["message"].get("content") or "").strip()
            except ValueError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise last_error

    def _record_usage(self, usage: Dict[str, Any]):
        with self._usage_lock:
            u = self.usage
            u["calls"] += 1
            u["prompt_tokens"] += usage.get("prompt_tokens", 0)
            u["completion_tokens"] += usage.get("completion_tokens", 0)
            u["cache_hit_tokens"] += usage.get("prompt_cache_hit_tokens", 0)
            if u["calls"] % 50 == 0:
                logger.info(f"[DeepSeek] 用量：{self.usage_summary()}")

    def usage_summary(self) -> str:
        u = self.usage
        hit = u["cache_hit_tokens"] / u["prompt_tokens"] if u["prompt_tokens"] else 0.0
        return (f"{u['calls']} 次调用（失败 {u['failures']}），输入 {u['prompt_tokens']} tokens"
                f"（缓存命中 {hit:.0%}），输出 {u['completion_tokens']} tokens")
