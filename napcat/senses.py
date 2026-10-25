"""
Her senses beyond text: what she hears, and the pictures she can look at.

hear(): a QQ voice message -> what was said (NapCat converts it to wav, Whisper transcribes).
fetch_image(): an image segment -> the bytes, for handing the picture to the model itself
        (see qq.media() / DeepSeekEngine.with_media()). That is the normal path: she looks at the
        picture in the conversation, no local vision model needed.
see():  when the picture cannot ride along with the conversation (the model that does the talking is
        a local text-only one, or the picture is inside a quoted message), the same DeepSeek API
        describes it in a short sentence. `caption_engine` is wired by app.py (the main engine, or
        the remote half of HybridEngine) - without it she only knows "a picture came".

All fail soft. Nothing heard or seen is stored here; only what she makes of it can become a memory.
"""
import asyncio
import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

CAPTION_PROMPT = """<|im_start|>system
你看到一张图片（QQ 聊天里别人发来的）。用一两句中文说图里有什么：
如果是表情包或梗图，说它表达的情绪或意思；如果是截图，说是什么的截图；图里有字就顺带读出来。
不要猜图里人的身份。只输出描述本身，不要解释。
<|im_end|>
<|im_start|>user
（请看这张图）<|im_end|>
<|im_start|>assistant
"""

class Senses:
    def __init__(self, hear_url: str = "http://127.0.0.1:8091",
                 max_image_bytes: int = 8 * 1024 * 1024):
        self.hear_url = hear_url.rstrip("/")
        self.max_image_bytes = max_image_bytes
        self.caption_engine = None          # set by app.py: a DeepSeek engine that can look at pictures
        self._seen: dict = {}

    @staticmethod
    def _post(url: str, payload: dict, timeout: float) -> Optional[dict]:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))     # loopback: never via a proxy
        try:
            with opener.open(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            logger.info(f"[Senses] {url} 没有结果：{type(e).__name__}")
            return None

    def health(self) -> dict:
        """Which senses are up right now (sight is the model's own, not a local service)."""
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(self.hear_url + "/health", timeout=2) as r:
                return {"hear": bool(json.load(r).get("ok"))}
        except Exception:
            return {"hear": False}

    async def hear(self, qq, record: dict) -> str:
        """A voice segment's data -> transcript ('' if she cannot hear right now)."""
        data = await qq.call("get_record", {"file": record.get("file") or record.get("file_id"), "out_format": "wav"},
                             timeout=10)
        path = (data or {}).get("file")
        if not path:
            return ""
        out = await asyncio.to_thread(self._post, self.hear_url + "/hear", {"path": path}, 30)
        return ((out or {}).get("text") or "").strip()

    @staticmethod
    def _fetch(url: str, limit: int) -> Optional[bytes]:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                data = r.read(limit + 1)
            return data if len(data) <= limit else None
        except Exception:
            return None

    @staticmethod
    def _mime(raw: bytes) -> str:
        """What the bytes are, so the model gets a correct data URL (NapCat does not always say)."""
        if raw.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if raw.startswith(b"\x89PNG"):
            return "image/png"
        if raw.startswith(b"GIF8"):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    @staticmethod
    def _to_jpeg(raw: bytes) -> Optional[bytes]:
        """First frame, RGB, reasonable size - what vision models digest well (and much smaller)."""
        try:
            import io
            from PIL import Image

            image = Image.open(io.BytesIO(raw))
            if getattr(image, "is_animated", False):
                image.seek(0)
            image = image.convert("RGB")
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024))
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=85)
            return out.getvalue()
        except Exception:
            return None

    def fetch_image(self, image: dict) -> Optional[dict]:
        """An image segment's data -> {"data": bytes, "mime": ...} (None when it cannot be fetched).

        Animated stickers (GIF/WebP) are sent as their first frame in JPEG: the model sees the meme,
        and the payload stays small enough for a chat-sized request.
        """
        url = image.get("url") or (image.get("file") if str(image.get("file", "")).startswith("http") else None)
        raw = self._fetch(url, self.max_image_bytes) if url else None
        if not raw:
            return None
        mime = self._mime(raw)
        if mime in ("image/gif", "image/webp") or len(raw) > 2 * 1024 * 1024:
            converted = self._to_jpeg(raw)
            if converted:
                return {"data": converted, "mime": "image/jpeg"}
        return {"data": raw, "mime": mime}

    def can_describe(self) -> bool:
        """Whether a DeepSeek engine that can look at pictures is wired up (see app.py)."""
        return self.caption_engine is not None

    def see(self, image: dict) -> str:
        """An image segment -> a short description from the API ('' when she cannot look right now).

        Only used when the picture cannot go into the conversation itself; the same API is used,
        so it is still the model that looks - just one sentence instead of the original image.
        """
        key = image.get("file") or image.get("url") or ""
        if key in self._seen:
            return self._seen[key]
        picture = self.fetch_image(image)
        if not picture or not self.caption_engine:
            return ""
        text = (self._caption(picture) or "").strip()[:120]
        if len(self._seen) > 300:
            self._seen.clear()
        self._seen[key] = text
        return text

    def _caption(self, picture: dict) -> str:
        try:
            out = self.caption_engine.generate_json(CAPTION_PROMPT, max_tokens=80, temperature=0.2,
                                                    media=[picture])
        except Exception as e:
            logger.warning(f"[Senses] 看图失败：{type(e).__name__}: {e}")
            return ""
        reply = (out or {}).get("reply") or ""
        # Some engines answer in the Thought/Response shape; only the description itself is wanted
        if "Response:" in reply or "Response：" in reply:
            reply = reply.split("Response:", 1)[-1].split("Response：", 1)[-1]
        return reply.strip()

senses = Senses()
