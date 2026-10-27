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
import os
import subprocess
import sys
import time
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
                 max_image_bytes: int = 30 * 1024 * 1024):
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

    # --- keeping her hearing running -------------------------------------------------------
    # The Whisper server (scripts/senses/hear_server.py) used to need a person to start it by hand, so most of
    # the time she simply could not hear. The app now starts it with itself and restarts it when it has died
    # (at most every RESTART_GAP seconds, so a broken install does not spin).

    RESTART_GAP = 600

    def configure_hearing(self, root: str, python: str = "", enabled: bool = True):
        """Where the hearing server lives. `python` defaults to the project's .venv-lora (where Whisper is)."""
        self._hear_root = root
        self._hear_python = python or os.path.join(root, ".venv-lora", "Scripts" if os.name == "nt" else "bin",
                                                   "python.exe" if os.name == "nt" else "python")
        self._hear_script = os.path.join(root, "scripts", "senses", "hear_server.py")
        self._hear_model = os.path.join(root, "models", "whisper-large-v3-turbo")
        self._hear_enabled = enabled
        self._hear_proc = None
        self._hear_started = 0.0
        self.hear_problem = ""

    def hearing_installed(self) -> str:
        """'' when the hearing server can be started here, otherwise what is missing."""
        if not getattr(self, "_hear_root", None):
            return "没有配置"
        for what, path in (("Python 环境", self._hear_python), ("听觉程序", self._hear_script),
                           ("Whisper 模型", self._hear_model)):
            if not os.path.exists(path):
                return f"缺少{what}（{os.path.basename(path)}）"
        return ""

    def ensure_hearing(self, now: float = None) -> bool:
        """Start the hearing server if it is not up (and allowed, installed, not just started). True = it is up."""
        now = time.time() if now is None else now
        if self.health().get("hear"):
            self.hear_problem = ""
            return True
        if not getattr(self, "_hear_enabled", False):
            return False
        missing = self.hearing_installed()
        if missing:
            self.hear_problem = missing
            return False
        proc = self._hear_proc
        if proc is not None and proc.poll() is None and now - self._hear_started < 180:
            self.hear_problem = "正在启动（加载 Whisper 要一会儿）"
            return False                                  # still loading the model
        if proc is not None and proc.poll() is not None:
            self.hear_problem = f"听觉服务退出了（退出码 {proc.returncode}，见 data/hear_server.err.log）"
        if self._hear_started and now - self._hear_started < self.RESTART_GAP:
            return False
        self.stop_hearing()                               # a server that never came up is not left behind
        port = self.hear_url.rsplit(":", 1)[-1].split("/")[0]
        log_dir = os.path.join(self._hear_root, "data")
        os.makedirs(log_dir, exist_ok=True)
        try:
            out = open(os.path.join(log_dir, "hear_server.log"), "ab")
            err = open(os.path.join(log_dir, "hear_server.err.log"), "ab")
            self._hear_proc = subprocess.Popen(
                [self._hear_python, "-X", "utf8", self._hear_script, "--port", port],
                cwd=self._hear_root, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._hear_started = now
            self.hear_problem = "正在启动（加载 Whisper 要一会儿）"
            logger.info(f"[Senses] 听觉服务启动中（pid {self._hear_proc.pid}，端口 {port}）")
        except OSError as e:
            self.hear_problem = f"启动失败：{e}"
            logger.warning(f"[Senses] 听觉服务启动失败：{e}")
        return False

    def stop_hearing(self):
        """Only the server this app started is stopped (one started by hand is left alone)."""
        proc = getattr(self, "_hear_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._hear_proc = None

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
                             timeout=15)
        path = (data or {}).get("file")
        if not path:
            logger.info("[Senses] get_record 没有给出音频文件")
            return ""
        out = await asyncio.to_thread(self._post, self.hear_url + "/hear", {"path": path}, 60)
        if out is None and getattr(self, "_hear_enabled", False):
            await asyncio.to_thread(self.ensure_hearing)     # it may have died: bring it back for the next one
        return ((out or {}).get("text") or "").strip()

    @staticmethod
    def _fetch(url: str, limit: int) -> Optional[bytes]:
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
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
    def _to_jpeg(raw: bytes, side: int = 1600) -> Optional[bytes]:
        """First frame, RGB, reasonable size - what vision models digest well (and much smaller)."""
        try:
            import io
            from PIL import Image

            image = Image.open(io.BytesIO(raw))
            if getattr(image, "is_animated", False):
                image.seek(0)
            image = image.convert("RGB")
            if max(image.size) > side:
                image.thumbnail((side, side))
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=85)
            return out.getvalue()
        except Exception:
            return None

    # A long screenshot (a chat log, a web page) squashed to one ~1600px picture has unreadable text: it is cut
    # into pieces of about this shape instead, top to bottom, each still sharp. The API itself scales anything
    # above ~2 MP down, so a piece is kept near that.
    TILE_WIDTH = 1280
    TILE_RATIO = 1.4                 # a piece is at most this many times as tall as it is wide
    MAX_TILES = 6

    @classmethod
    def _tiles(cls, raw: bytes) -> Optional[list]:
        """A tall picture -> its pieces as JPEG bytes (None when it is not tall enough to need cutting)."""
        try:
            import io
            from PIL import Image

            image = Image.open(io.BytesIO(raw))
            if getattr(image, "is_animated", False):
                return None
            w, h = image.size
            if h <= w * cls.TILE_RATIO * 1.3 or w < 200:
                return None
            image = image.convert("RGB")
            # narrow enough that each piece is readable, wide enough that MAX_TILES pieces cover the whole picture
            width = min(w, cls.TILE_WIDTH)
            scaled_h = h * width / w
            per = width * cls.TILE_RATIO
            if scaled_h > per * cls.MAX_TILES:        # very long: taller pieces rather than more of them
                per = scaled_h / cls.MAX_TILES
            if width != w:
                image = image.resize((width, int(scaled_h)), Image.LANCZOS)
            total = image.size[1]
            count = min(cls.MAX_TILES, int(-(-total // per)))
            step = total / count
            overlap = int(min(60, step * 0.05))              # a line of text cut at the edge shows in both pieces
            pieces = []
            for i in range(count):
                top = max(0, int(i * step) - overlap)
                bottom = min(total, int((i + 1) * step) + overlap)
                out = io.BytesIO()
                image.crop((0, top, width, bottom)).save(out, format="JPEG", quality=85)
                pieces.append(out.getvalue())
            return pieces
        except Exception:
            return None

    KEEP_ORIGINAL = 12 * 1024 * 1024  # originals up to this size ride along, so she can keep the real thing

    def prepare_image(self, raw: bytes) -> Optional[dict]:
        """Picture bytes -> what she looks at: {"data", "mime"[, "parts"][, "original"]}.

        - a small still picture goes as it is;
        - an animated sticker (GIF/WebP) goes as its first frame;
        - a big photo is scaled to 1600px; a tall screenshot is cut into sharp pieces (`parts`, in order) so its
          text stays readable - `data` is then the first piece;
        - `original` is the untouched bytes when they differ from what is sent (what `keep` stores).
        """
        if not raw:
            return None
        mime = self._mime(raw)
        picture = {"data": raw, "mime": mime}
        if mime not in ("image/gif", "image/webp"):
            parts = self._tiles(raw)
            if parts:
                picture = {"data": parts[0], "mime": "image/jpeg", "parts": parts}
        if "parts" not in picture and (mime in ("image/gif", "image/webp") or len(raw) > 2 * 1024 * 1024):
            converted = self._to_jpeg(raw)
            if converted:
                picture = {"data": converted, "mime": "image/jpeg"}
        if picture["data"] is not raw and len(raw) <= self.KEEP_ORIGINAL:
            picture["original"] = raw
            picture["original_mime"] = mime
        return picture

    def fetch_image(self, image: dict) -> Optional[dict]:
        """An image segment's data -> what she looks at (see prepare_image), None when it cannot be fetched.

        The URL is tried first; when it has expired (old QQ links do) the caller may hand a local file instead
        (qq.media asks NapCat's get_image for it).
        """
        url = image.get("url") or (image.get("file") if str(image.get("file", "")).startswith("http") else None)
        raw = self._fetch(url, self.max_image_bytes) if url else None
        path = image.get("path") or ""
        if not raw and path and os.path.isfile(path) and os.path.getsize(path) <= self.max_image_bytes:
            with open(path, "rb") as f:
                raw = f.read()
        return self.prepare_image(raw) if raw else None

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
