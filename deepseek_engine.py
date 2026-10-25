import base64
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
    - media=...：图片跟着这轮对话一起交给模型看（不是先转成一句文字描述），
      图片作为最后一个 user 轮里的 image_url 内容块；模型上下文里只有文字的部分保持不变
    """

    # 模型能直接看图（app 据此决定"把图交给她"；不能看图的引擎只会知道"来了张图"）
    supports_images = True
    # 原生 function calling：GlobalWorkspace 据此走多步工具循环，而不是从文本里抠 "Action: ..."
    supports_tools = True

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

    @staticmethod
    def with_media(messages: List[Dict[str, Any]], media: Optional[List[Dict[str, Any]]],
                   limit: int = 3) -> List[Dict[str, Any]]:
        """把图片放进最后一个 user 轮：她读这条消息时，图就在眼前。

        media: [{"data": bytes, "mime": "image/jpeg"}, ...]
        没有可用的图片时原样返回，消息仍是纯文本（旧行为不变）。
        """
        pictures = [m for m in (media or []) if m.get("data")][:limit]
        if not pictures:
            return messages
        target = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if target is None:
            return messages
        parts: List[Dict[str, Any]] = [{"type": "text", "text": target["content"]}]
        for picture in pictures:
            b64 = base64.b64encode(picture["data"]).decode("ascii")
            mime = picture.get("mime") or "image/jpeg"
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        target["content"] = parts
        return messages

    def generate_json(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7,
                      media: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        wants_json = self._wants_json(prompt)
        messages = self.with_media(self.chatml_to_messages(prompt), media)
        body = {
            "model": self.model,
            "messages": messages,
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

    def chat_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], max_tokens: int = 300,
                   temperature: float = 0.7) -> Dict[str, Any]:
        """一轮带工具的对话。返回 {"content": str, "tool_calls": [{"id", "name", "arguments"(dict)}], "raw": message}；
        失败时 {"content": "", "tool_calls": [], "error": str}。messages 原样发出（调用方负责拼好 tool 轮）。"""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            message = self._post_message(body)
        except Exception as e:
            with self._usage_lock:
                self.usage["failures"] += 1
            logger.error(f"[DeepSeek] 工具调用失败：{e}")
            return {"content": "", "tool_calls": [], "error": str(e)}
        calls = []
        for c in message.get("tool_calls") or []:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {"value": args}
            except json.JSONDecodeError:
                args = {"_unparsed": fn.get("arguments")}
            calls.append({"id": c.get("id") or f"call_{len(calls)}", "name": fn.get("name") or "", "arguments": args})
        return {"content": (message.get("content") or "").strip(), "tool_calls": calls,
                "raw": {k: message[k] for k in ("role", "content", "tool_calls") if k in message}}

    def _post(self, body: Dict[str, Any]) -> str:
        return (self._post_message(body).get("content") or "").strip()

    def _post_message(self, body: Dict[str, Any]) -> Dict[str, Any]:
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
                return data["choices"][0]["message"]
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
