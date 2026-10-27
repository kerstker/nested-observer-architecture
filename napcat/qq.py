"""
Everything she can perceive and do on QQ through NapCat (OneBot v11 + NapCat's extensions).

- call(): a real request/response API call (echo -> future), so she can ask QQ things -
  what a quoted message said, what is inside a forwarded chat, who someone is in a group,
  what was said in a group before she came in.
- perceive(): turns a message's segments into what a person would see: quotes, forwards,
  link/mini-program cards, voice, video, files, dice - not just the plain text.
- media(): the pictures in the message as bytes, for handing to a model that can see for itself.
- notice(): pokes, recalls, new members, being muted, emoji reactions to her messages,
  "someone is typing" - QQ events that are not messages.
- act(): her non-verbal QQ gestures: poke back, react with an emoji, recall her own message,
  show she is typing.

Requests (friend / group invitations) are never accepted by her on her own: they are reported.
"""
import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# QQ system face ids NapCat accepts for set_msg_emoji_like (a small, safe set)
REACTIONS = {"赞": 76, "爱心": 66, "笑": 178, "抱抱": 49, "OK": 124, "微笑": 14, "鼓掌": 99, "流泪": 5}

class QQ:
    def __init__(self, manager=None):
        self.manager = manager                  # napcat_server.ws_manager (robot_connections, send_to_robot)
        self.pending: Dict[str, asyncio.Future] = {}
        self.typing: Dict[str, float] = {}      # user_id -> when they were last seen typing to her
        self.recalled: deque = deque(maxlen=200)
        self.self_id: Optional[str] = None
        self.login: Dict[str, Any] = {}
        self.version: Dict[str, Any] = {}
        self.read_images = True                  # QQ's own OCR (NapCat ocr_image, Windows): words in pictures
        self._ocr_cache: Dict[str, str] = {}
        self.senses = None                       # napcat.senses.Senses: hearing and sight, when running

    # --- API ----------------------------------------------------------------------

    def connected(self) -> bool:
        return bool(self.manager and self.manager.robot_connections)

    async def call(self, action: str, params: Dict[str, Any] = None, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
        """Send an API request and wait for its answer. Returns the response's data, or None."""
        if not self.connected():
            return None
        robot_id = next(iter(self.manager.robot_connections))
        echo = f"noa_{uuid.uuid4().hex[:12]}"
        fut = asyncio.get_running_loop().create_future()
        self.pending[echo] = fut
        try:
            await self.manager.send_to_robot(robot_id, {"action": action, "params": params or {}, "echo": echo})
            resp = await asyncio.wait_for(fut, timeout)
        except (asyncio.TimeoutError, Exception) as e:
            logger.info(f"[QQ] {action} 没有结果：{type(e).__name__}")
            return None
        finally:
            self.pending.pop(echo, None)
        if resp.get("retcode") != 0 and resp.get("status") != "ok":
            logger.info(f"[QQ] {action} 失败：{resp.get('message') or resp.get('wording')}")
            return None
        return resp.get("data") if resp.get("data") is not None else {}

    def resolve(self, data: Dict[str, Any]) -> bool:
        """Called for every API response NapCat sends back. True if it answered one of our calls."""
        fut = self.pending.get(str(data.get("echo", "")))
        if fut and not fut.done():
            fut.set_result(data)
            return True
        return False

    async def self_check(self) -> str:
        """On connecting: who am I on QQ, and which NapCat is this? Also proves request/response works."""
        self.login = await self.call("get_login_info", timeout=5) or {}
        self.version = await self.call("get_version_info", timeout=5) or {}
        if self.login.get("user_id"):
            self.self_id = str(self.login["user_id"])
        return (f"QQ 已连上：我是 {self.login.get('nickname', '?')}（{self.login.get('user_id', '?')}），"
                f"{self.version.get('app_name', 'NapCat')} {self.version.get('app_version', '?')}")

    async def ocr(self, image: str, timeout: float = 10.0) -> str:
        """Words written in a picture (memes, screenshots), via QQ's own OCR. '' when none or unavailable.

        NapCat's OCR downloads the picture and runs Windows OCR: fast for screenshots (1-3s), but it
        never returns for large animated stickers, so the timeout is caller-controlled (short for
        stickers, longer for screenshots) and a failure is remembered - retrying the same picture
        would only stall the conversation again.
        """
        if not self.read_images or not image:
            return ""
        if image in self._ocr_cache:
            return self._ocr_cache[image]
        data = await self.call("ocr_image", {"image": image}, timeout=timeout)
        texts = (data or {}).get("texts") if isinstance(data, dict) else data
        words = "".join(t.get("text", "") for t in texts or [] if isinstance(t, dict)).strip()
        if len(self._ocr_cache) > 500:
            self._ocr_cache.clear()
        self._ocr_cache[image] = words[:120]
        return self._ocr_cache[image]

    # --- perceiving messages --------------------------------------------------------

    @staticmethod
    def _card_text(raw: str) -> str:
        try:
            card = json.loads(raw)
        except (TypeError, ValueError):
            return "[卡片]"
        meta = card.get("meta") or {}
        for v in meta.values():
            if isinstance(v, dict):
                title = v.get("title") or v.get("desc") or ""
                desc = v.get("desc") if v.get("title") else ""
                if title:
                    return f"[分享：{title}{'｜' + desc[:40] if desc and desc != title else ''}]"
        return f"[分享：{card.get('prompt', '卡片')}]"

    @staticmethod
    def quote_prefix(is_self: bool, who: str, text: str) -> str:
        """How a quoted message reads. Never a bare "我": inside someone else's message she read it as them."""
        return f"（引用了{'我（乃爱）' if is_self else ' ' + who + ' '}说的：「{text[:60]}」）"

    @staticmethod
    def at_text(name: str, is_self: bool) -> str:
        return "@我（乃爱）" if is_self else f"@{name}"

    async def _quoted(self, message_id: str, names: Callable[[str], str], look: bool = False) -> str:
        data = await self.call("get_msg", {"message_id": message_id}, timeout=4)
        if not data:
            return "（回复了一条消息）"
        sender = data.get("sender") or {}
        mine = bool(self.self_id and str(sender.get("user_id")) == str(self.self_id))
        who = sender.get("card") or sender.get("nickname") or names(str(sender.get("user_id")))
        # A quoted picture/voice cannot ride along with the conversation, so it is looked at here
        text = await self.perceive(data.get("message") or [], names, depth=1, look=look, caption=True)
        return self.quote_prefix(mine, who, text)

    async def _forward(self, fid: str, names, depth: int) -> str:
        data = await self.call("get_forward_msg", {"id": fid}, timeout=6)
        nodes = (data or {}).get("messages") or []
        if not nodes:
            return "[一段聊天记录]"
        lines = []
        for n in nodes[:8]:
            s = n.get("sender") or {}
            body = await self.perceive(n.get("message") or n.get("content") or [], names, depth + 1,
                                       caption=False)     # a forwarded chat may hold many pictures
            lines.append(f"{s.get('card') or s.get('nickname') or '某人'}：{body[:50]}")
        more = f"……（共 {len(nodes)} 条）" if len(nodes) > 8 else ""
        return "[聊天记录]\n" + "\n".join(lines) + more

    async def perceive(self, segments, names: Callable[[str], str] = lambda u: "某人", depth: int = 0,
                       look: bool = False, caption: bool = True, file_notes: Dict[str, str] = None) -> str:
        """What a person would take in from this message.

        `look`: really listen to voice / look at pictures (slower; for messages she is paying attention to).
        `caption`: when the talking model can see for itself (deepseek), the picture rides along with the
        message (qq.media()) and no description is made; a text-only model gets an API description instead.
        """
        if isinstance(segments, str):
            return segments.strip()
        parts: List[str] = []
        for s in segments or []:
            if isinstance(s, str):
                parts.append(s)
                continue
            t, d = s.get("type"), s.get("data") or {}
            if t == "text":
                parts.append(d.get("text", ""))
            elif t == "at":
                qq_id = str(d.get("qq") or "")
                if qq_id == "all":
                    parts.append("@全体成员 ")
                else:
                    mine = bool(self.self_id and qq_id == str(self.self_id))
                    parts.append(self.at_text(d.get("name") or names(qq_id), mine) + " ")
            elif t == "reply" and depth == 0:
                parts.insert(0, await self._quoted(str(d.get("id")), names, look=look))
            elif t == "forward" and depth < 2:
                parts.append(await self._forward(str(d.get("id")), names, depth))
            elif t == "json":
                parts.append(self._card_text(d.get("data")))
            elif t == "xml":
                parts.append("[分享了一个卡片]")
            elif t == "record":
                said = await self.senses.hear(self, d) if (look and self.senses) else ""
                parts.append(f"[语音：{said}]" if said else "[发了一段语音]")
            elif t == "video":
                parts.append("[发了一段视频]")
            elif t == "file":
                parts.append((file_notes or {}).get(str(d.get("file_id") or d.get("file")))
                             or f"[发了一个文件：{d.get('file') or d.get('name') or ''}]")
            elif t == "image":
                photo = str(d.get("sub_type", "0")) == "0"
                # Stickers are often animated GIFs: QQ's OCR never returns for those, so wait less.
                words = await self.ocr(d.get("url") or d.get("file") or "", timeout=10.0 if photo else 4.0)
                seen = await asyncio.to_thread(self.senses.see, d) \
                    if (look and self.senses and caption and self.senses.can_describe()) else ""
                what = "图片" if photo else "表情包"
                details = "；".join(x for x in (seen, f"上面写着：{words}" if words else "") if x)
                if details:
                    parts.append(f"[{what}：{details}]")
                elif photo:
                    parts.append("[发了一张图片]")
            elif t == "dice":
                parts.append(f"[掷骰子：{d.get('result', '?')}]")
            elif t == "rps":
                parts.append(f"[猜拳：{ {'1': '石头', '2': '剪刀', '3': '布'}.get(str(d.get('result')), '?') }]")
            elif t == "location":
                parts.append(f"[位置：{d.get('title') or ''}]")
            elif t == "music":
                parts.append(f"[分享音乐：{d.get('title') or ''}]")
            elif t == "contact":
                parts.append("[推荐了一个联系人]" if d.get("type") != "group" else "[推荐了一个群]")
            elif t == "poke":
                parts.append("[戳了一下]")
        return "".join(parts).strip()

    # --- files people send her ----------------------------------------------------------

    MAX_FILE = 25 * 1024 * 1024

    @staticmethod
    def safe_name(name: str) -> str:
        """A file name that cannot point anywhere but into its own folder."""
        name = re.split(r"[\\/]", str(name or "file"))[-1]
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(". ") or "file"
        return name[:100]

    async def fetch_file(self, d: Dict[str, Any], group_id: Optional[str], user_id: str):
        """The bytes of a file in a message. Returns (bytes, None) or (None, why not)."""
        file_id = d.get("file_id") or d.get("file")
        try:
            size = int(d.get("file_size") or 0)
        except (TypeError, ValueError):
            size = 0
        if size > self.MAX_FILE:
            return None, f"文件太大了（{size / 1024 / 1024:.0f} MB，我最多收 {self.MAX_FILE // 1024 // 1024} MB）"
        info = await self.call("get_file", {"file_id": file_id}, timeout=40) or {}
        path = info.get("file")
        if path and os.path.isfile(path):                          # NapCat runs on this machine: read it directly
            if os.path.getsize(path) > self.MAX_FILE:
                return None, "文件太大了"
            return await asyncio.to_thread(lambda: open(path, "rb").read()), None
        if info.get("base64"):
            try:
                return base64.b64decode(info["base64"]), None
            except ValueError:
                pass
        url = info.get("url") or ""
        if not url.startswith("http"):
            url = ""
            if group_id:
                got = await self.call("get_group_file_url", {"group_id": int(group_id), "file_id": file_id}, timeout=15)
            else:
                got = await self.call("get_private_file_url", {"user_id": int(user_id), "file_id": file_id}, timeout=15)
            url = (got or {}).get("url") or ""
        if not url.startswith("http"):
            return None, "没能从 QQ 取到这个文件（可能过期了或者没权限）"
        return await asyncio.to_thread(self._download, url)

    def _download(self, url: str):
        import requests
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                if r.status_code != 200:
                    return None, f"下载失败（HTTP {r.status_code}）"
                data = b""
                for chunk in r.iter_content(65536):
                    data += chunk
                    if len(data) > self.MAX_FILE:
                        return None, "文件太大了"
                return data, None
        except Exception as e:
            return None, f"下载失败：{type(e).__name__}"

    async def file_notes(self, segments, group_id: Optional[str], user_id: str, inbox: str, message_id: str = "") -> Dict[str, str]:
        """Fetch the files of this message into `inbox`, read them, and say what is in them.
        Returns {file_id: text for the message}. A file that cannot be had is still reported honestly."""
        from src.levels.l1_perception import documents
        notes: Dict[str, str] = {}
        for seg in segments or []:
            if not isinstance(seg, dict) or seg.get("type") != "file":
                continue
            d = seg.get("data") or {}
            key = str(d.get("file_id") or d.get("file"))
            name = self.safe_name(d.get("file") or d.get("name") or "file")
            data, why = await self.fetch_file(d, group_id, user_id)
            if data is None:
                notes[key] = f"[发了一个文件：{name}（我没能打开：{why}）]"
                continue
            folder = os.path.join(inbox, str(group_id or user_id))
            os.makedirs(folder, exist_ok=True)
            stored = f"{message_id or int(time.time())}-{name}"
            await asyncio.to_thread(lambda: open(os.path.join(folder, stored), "wb").write(data))
            result = await asyncio.to_thread(documents.extract, data, name)
            if result["kind"] == "image":
                words = await self.ocr(os.path.join(folder, stored), timeout=10.0)
                if words:
                    result = {"kind": "image", "text": words, "note": "图片里识别出的文字"}
            notes[key] = (f"[发了一个文件：{name}（{documents.preview(result, len(data))}）"
                          f"\n要看完整内容用 read_attachment 工具，文件名 {name}]")
        return notes

    async def media(self, segments, limit: int = 2) -> List[Dict[str, Any]]:
        """The pictures in this message, as bytes: what she looks at when the model can see for itself.

        Each item is {"data": bytes, "mime": "image/jpeg"}; frames that cannot be fetched are skipped.
        Only the message itself is looked at (not quoted or forwarded pictures), and at most `limit`.
        """
        if not self.senses or limit <= 0:
            return []
        pictures: List[Dict[str, Any]] = []
        for seg in segments or []:
            if not isinstance(seg, dict) or seg.get("type") != "image":
                continue
            d = dict(seg.get("data") or {})
            picture = await asyncio.to_thread(self.senses.fetch_image, d)
            if not picture and d.get("file"):
                # The URL has expired or will not download: NapCat can still give the picture as a local file
                got = await self.call("get_image", {"file": d.get("file")}, timeout=20) or {}
                if got.get("file"):
                    picture = await asyncio.to_thread(self.senses.fetch_image, {"path": got["file"]})
            if picture:
                picture["name"] = str(d.get("file") or "")[:80]
                picture["sticker"] = str(d.get("sub_type", "0")) != "0"
                pictures.append(picture)
            if len(pictures) >= limit:
                break
        return pictures

    async def group_history(self, group_id: str, count: int = 20) -> List[Dict[str, Any]]:
        data = await self.call("get_group_msg_history", {"group_id": int(group_id), "count": count}, timeout=6)
        return (data or {}).get("messages") or []

    async def member(self, group_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        return await self.call("get_group_member_info", {"group_id": int(group_id), "user_id": int(user_id)}, timeout=4)

    # --- notices --------------------------------------------------------------------

    def notice(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Translate a QQ notice into an event she can live through (or None when it is nothing to her)."""
        nt, st = data.get("notice_type"), data.get("sub_type")
        me = str(data.get("self_id") or self.self_id or "")
        uid = str(data.get("user_id") or data.get("operator_id") or "")
        gid = str(data["group_id"]) if data.get("group_id") else None
        if nt == "notify" and st == "poke":
            if str(data.get("target_id")) == me and uid != me:
                return {"kind": "poke", "user_id": uid, "group_id": gid}
            return None
        if nt == "notify" and st == "input_status":
            if data.get("event_type", 1) in (1, "1"):
                self.typing[uid] = time.time()
            return None
        if nt in ("group_recall", "friend_recall"):
            self.recalled.append(str(data.get("message_id")))
            if uid == me:
                return None
            return {"kind": "recall", "user_id": uid, "group_id": gid, "operator_id": str(data.get("operator_id") or uid)}
        if nt == "group_increase":
            return {"kind": "joined" if uid != me else "i_joined", "user_id": uid, "group_id": gid}
        if nt == "group_decrease":
            return {"kind": "left" if uid != me else "i_was_removed", "user_id": uid, "group_id": gid,
                    "kicked": st in ("kick", "kick_me")}
        if nt == "group_ban" and uid == me:
            return {"kind": "muted" if st == "ban" else "unmuted", "user_id": str(data.get("operator_id") or ""),
                    "group_id": gid, "duration": int(data.get("duration") or 0)}
        if nt == "group_admin" and uid == me:
            return {"kind": "made_admin" if st == "set" else "admin_removed", "group_id": gid, "user_id": uid}
        if nt == "essence" and str(data.get("sender_id")) == me:
            return {"kind": "essence", "group_id": gid, "user_id": str(data.get("operator_id") or "")}
        if nt == "group_msg_emoji_like":
            return {"kind": "reaction", "group_id": gid, "user_id": uid, "message_id": str(data.get("message_id")),
                    "likes": data.get("likes") or []}
        return None

    def is_typing(self, user_id: str, within: float = 6.0) -> bool:
        return time.time() - self.typing.get(str(user_id), 0) < within

    # --- gestures -------------------------------------------------------------------

    async def poke(self, user_id: str, group_id: str = None) -> bool:
        params = {"user_id": int(user_id), **({"group_id": int(group_id)} if group_id else {})}
        if await self.call("send_poke", params) is not None:
            return True
        return await self.call("group_poke" if group_id else "friend_poke", params) is not None

    async def react(self, message_id: str, emoji: str) -> bool:
        eid = REACTIONS.get(emoji, REACTIONS["赞"])
        return await self.call("set_msg_emoji_like", {"message_id": int(message_id), "emoji_id": str(eid)}) is not None

    async def recall(self, message_id: str) -> bool:
        return await self.call("delete_msg", {"message_id": int(message_id)}) is not None

    async def show_typing(self, user_id: str) -> bool:
        return await self.call("set_input_status", {"user_id": str(user_id), "event_type": 1}, timeout=3) is not None

    async def mark_read(self, user_id: str = None, group_id: str = None) -> bool:
        if group_id:
            return await self.call("mark_group_msg_as_read", {"group_id": int(group_id)}, timeout=3) is not None
        return await self.call("mark_private_msg_as_read", {"user_id": int(user_id)}, timeout=3) is not None

    def describe_request(self, data: Dict[str, Any]) -> str:
        if data.get("request_type") == "friend":
            return f"{data.get('user_id')} 想加我好友：{data.get('comment') or '（没写理由）'}"
        if data.get("request_type") == "group":
            what = "邀请我进群" if data.get("sub_type") == "invite" else "申请进群"
            return f"{data.get('user_id')} {what} {data.get('group_id')}：{data.get('comment') or '（没写理由）'}"
        return "收到一个请求"

qq = QQ()
