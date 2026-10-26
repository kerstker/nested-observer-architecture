"""
Noa 控制台：在浏览器里实时看她的内心状态、和她说话、做一些操作。

挂在 app.py 的 FastAPI 上，地址 http://<主机>:8080/console
- 没设 CONSOLE_TOKEN 时只允许本机访问
- 设了 CONSOLE_TOKEN 后可以远程访问，打开 /console?token=<token>
"""
import asyncio
import hmac
import logging
import os
from collections import deque
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from src.core import clock
from .state import snapshot, mind, trace, trace_call, social, memory, history, metrics, to_json

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
PAGE = os.path.join(os.path.dirname(__file__), "index.html")

class SayRequest(BaseModel):
    text: str
    user_id: str = "console"
    nickname: str = "控制台"
    group_id: Optional[str] = None

class ActRequest(BaseModel):
    action: str
    user_id: str = "console"
    nickname: str = "控制台"

class EvolveRequest(BaseModel):
    task: str = ""
    id: str = ""

class MaintainerReply(BaseModel):
    id: str
    status: str
    note: str = ""

class OutsideRequest(BaseModel):
    wander_enabled: Optional[bool] = None
    per_day: Optional[int] = None

class SettingsRequest(BaseModel):
    idle_thought_interval: Optional[float] = None
    autonomous_groups: Optional[list] = None
    unmute_group: Optional[str] = None
    mute_group: Optional[str] = None
    mute_minutes: float = 180
    clear_left: bool = False
    clear_cooldown: bool = False
    clear_unread: bool = False

class TaskRequest(BaseModel):
    id: int = 0
    kind: str = "remind"
    what: str = ""
    minutes: float = 10

class QqRequest(BaseModel):
    action: str
    params: dict = {}
    timeout: float = 15.0

# Feeding and petting go through the normal perception path, so they are experienced
# (appraisal, cause log, episode) exactly like the same words from QQ would be.
SPOKEN_ACTIONS = {"feed": "喂你吃小鱼干", "pet": "摸摸头"}

class LogBuffer(logging.Handler):
    """The last few hundred log lines, for the console (the files are on disk; this is what is happening now)."""
    def __init__(self, size: int = 500):
        super().__init__(logging.INFO)
        self.lines = deque(maxlen=size)
        self.seq = 0

    def emit(self, record: logging.LogRecord):
        try:
            self.seq += 1
            self.lines.append({"seq": self.seq, "t": record.created, "level": record.levelname,
                               "name": record.name, "msg": record.getMessage()})
        except Exception:
            pass

def _install_log_buffer() -> LogBuffer:
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, LogBuffer):
            return h
    buf = LogBuffer()
    root.addHandler(buf)
    return buf

def create_console_router(gw, token: str = None) -> APIRouter:
    router = APIRouter(prefix="/console")
    logs = _install_log_buffer()

    def authorize(request: Request):
        if token:
            given = request.headers.get("X-Console-Token") or request.query_params.get("token") or ""
            if not hmac.compare_digest(given, token):
                raise HTTPException(401, "需要正确的控制台口令（CONSOLE_TOKEN）")
        elif (request.client.host if request.client else "") not in LOCAL_HOSTS:
            raise HTTPException(403, "控制台只允许本机访问；要远程访问请在 config.json 里设置 CONSOLE_TOKEN")

    def json_response(data) -> Response:
        return Response(to_json(data), media_type="application/json")

    @router.get("", response_class=HTMLResponse)
    async def page(request: Request):
        authorize(request)
        with open(PAGE, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @router.get("/api/state")
    async def state(request: Request):
        authorize(request)
        return json_response(await asyncio.to_thread(snapshot, gw))

    @router.post("/api/say")
    async def say(body: SayRequest, request: Request):
        authorize(request)
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "说点什么吧")
        # Same path as QQ: she may choose not to answer, or not see it because she is asleep
        reply = await gw.hear(text, user_id=body.user_id, nickname=body.nickname, group_id=body.group_id, addressed=True)
        return json_response({"reply": reply or "", "state": await asyncio.to_thread(snapshot, gw)})

    @router.post("/api/act")
    async def act(body: ActRequest, request: Request):
        authorize(request)
        a = body.action
        result = None
        if a in SPOKEN_ACTIONS:
            result = await gw.hear(SPOKEN_ACTIONS[a], user_id=body.user_id, nickname=body.nickname) or "（她没有回）"
        elif a == "sleep":
            gw._set_sleeping(True, source="console")
            result = "她睡着了"
        elif a == "wake":
            gw._set_sleeping(False, source="console")
            result = "她醒了" if not gw.metabolism.is_sleeping else "精力太低，叫不醒"
        elif a == "dream":
            report = await asyncio.to_thread(gw._dream_process)
            result = report or "今天没有新的经历可以整理"
        elif a == "think":
            await asyncio.to_thread(gw._idle_thinking_process)
            result = gw.consciousness_stream[-1] if gw.consciousness_stream else "她什么也没想"
        elif a == "tick":
            await gw.tick()
            result = "心跳了一次"
        elif a == "backup":
            if getattr(gw, "backup", None) is None:
                raise HTTPException(400, "备份没有开启")
            r = await asyncio.to_thread(gw.backup.run, clock.now(), "console")
            result = (f"备份好了：{r['name']}（{r['bytes'] // 1024} KB，已检查，另存 {r.get('copies', 0)} 处）" if r["ok"]
                      else f"备份失败：{r.get('error')}")
        elif a == "snapshot":
            await asyncio.to_thread(gw._save_state, "console")
            result = "存了一份状态快照"
        elif a == "selfcheck":
            found = await asyncio.to_thread(gw.diagnosis.diagnose, gw, clock.now())
            result = ("自检发现：" + "；".join(f"{f['text']}（{f['evidence']}）" for f in found)) if found                 else "自检：没有超过阈值的问题"
        else:
            raise HTTPException(400, f"未知操作：{a}")
        return json_response({"result": result, "state": await asyncio.to_thread(snapshot, gw)})

    # ---- 更深的视图：只在对应标签页打开时才拉 ----
    @router.get("/api/mind")
    async def mind_view(request: Request):
        authorize(request)
        return json_response(await asyncio.to_thread(mind, gw))

    @router.get("/api/trace")
    async def trace_view(request: Request, full: bool = False):
        authorize(request)
        return json_response(trace(gw, full))

    @router.get("/api/trace/{i}")
    async def trace_one(i: int, request: Request):
        authorize(request)
        return json_response(trace_call(gw, i))

    @router.get("/api/social")
    async def social_view(request: Request):
        authorize(request)
        return json_response(social(gw))

    @router.get("/api/memory")
    async def memory_view(request: Request, q: str = ""):
        authorize(request)
        return json_response(await asyncio.to_thread(memory, gw, q))

    @router.get("/api/history")
    async def history_view(request: Request, hours: float = 24.0):
        authorize(request)
        return json_response(await asyncio.to_thread(history, gw, max(0.1, min(hours, 24 * 30))))

    @router.get("/api/metrics")
    async def metrics_view(request: Request):
        authorize(request)
        return json_response(await asyncio.to_thread(metrics, gw))

    @router.get("/api/logs")
    async def logs_view(request: Request, after: int = 0, level: str = ""):
        authorize(request)
        rows = [r for r in logs.lines if r["seq"] > after and (not level or r["level"] == level)]
        return json_response({"seq": logs.seq, "lines": rows[-300:]})

    @router.post("/api/tasks")
    async def task_add(body: TaskRequest, request: Request):
        authorize(request)
        now = clock.now()
        try:
            tid = gw.tasks.add(body.kind, body.what, now + max(0.0, body.minutes) * 60, now, "console", "控制台",
                               origin="console")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return json_response({"id": tid})

    @router.post("/api/tasks/cancel")
    async def task_cancel(body: TaskRequest, request: Request):
        authorize(request)
        t = gw.tasks.cancel(body.id, "console", True, clock.now())
        if not t:
            raise HTTPException(400, "取消不了（不存在或已经结束）")
        if t.get("goal_id"):
            gw.goals.abandon(t["goal_id"], clock.now(), "在控制台被取消了")
        return json_response(t)

    @router.post("/api/settings")
    async def settings(body: SettingsRequest, request: Request):
        authorize(request)
        done = []
        now = clock.now()
        if body.idle_thought_interval is not None:
            gw.idle_thought_interval = max(10.0, float(body.idle_thought_interval))
            done.append(f"念头最短间隔 = {gw.idle_thought_interval:.0f} 秒")
        if body.autonomous_groups is not None:
            gw.autonomous_groups = {str(g).strip() for g in body.autonomous_groups if str(g).strip()}
            done.append(f"可自己开口的群 = {'、'.join(sorted(gw.autonomous_groups)) or '无'}")
        if body.mute_group:
            gw.group_mute_until[body.mute_group] = now + max(1.0, body.mute_minutes) * 60
            done.append(f"群 {body.mute_group} 安静 {body.mute_minutes:.0f} 分钟")
        if body.unmute_group:
            gw.group_mute_until.pop(body.unmute_group, None)
            done.append(f"群 {body.unmute_group} 解除安静")
        if body.clear_left:
            gw.left_until.clear()
            done.append("回到所有她离开的对话")
        if body.clear_cooldown:
            gw.cooldown_until.clear()
            done.append("清除硬上限冷却")
        if body.clear_unread:
            gw.unread.clear()
            done.append("清空未读")
        return json_response({"result": "；".join(done) or "没有改动", "state": await asyncio.to_thread(snapshot, gw)})

    # ---- Level 17: proposals to change her own code -------------------------------------
    # Creating one runs an agent in a throwaway copy of the repo (minutes). Applying one is
    # explicit and needs a restart; nothing here ever edits the live tree by itself.
    @router.get("/api/evolve")
    async def evolve_list(request: Request):
        authorize(request)
        bridge = getattr(gw, "evolve_agent", None)
        if bridge is None:
            return json_response({"enabled": False, "proposals": [],
                                  "hint": "config.json 里设置 EVOLVE_ENABLED=true 后重启才会开启"})
        return json_response({"enabled": bool(gw.evolve_enabled),
                              "creator": gw.evolve_creator,
                              "proposals": await asyncio.to_thread(bridge.list_proposals)})

    # What she has asked of the person who maintains her, and the answers (see l17_self_improvement/maintainer.py)
    @router.get("/api/maintainer")
    async def maintainer_list(request: Request):
        authorize(request)
        rows = await asyncio.to_thread(gw.maintainer.list, None, 40)
        return json_response({"requests": rows})

    @router.post("/api/maintainer")
    async def maintainer_reply(body: MaintainerReply, request: Request):
        authorize(request)
        try:
            ok = await asyncio.to_thread(gw.maintainer.reply, body.id, body.status, body.note)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not ok:
            raise HTTPException(404, "没有这个请求")
        return json_response({"ok": True})

    # Hands outside the house: the MCP servers she can use (a browser, ...), and her own time on the web.
    # Who may use which tool is decided in l8_agency/access.py; this only shows it and holds the switch.
    @router.get("/api/outside")
    async def outside_state(request: Request):
        authorize(request)
        hub = getattr(gw, "mcp", None)
        if hub is None:
            return json_response({"enabled": False, "hint": "config.json 里设置 MCP_ENABLED=true 后重启才会开启"})
        from src.levels.l8_agency import access
        tools = hub.tools()
        return json_response({
            "enabled": True, "servers": hub.status(), "tools": [
                {"server": t["server"], "name": t["name"], "description": t["description"][:100],
                 "tiers": [tier for tier in (access.OWNER, access.TRUSTED, access.AUTONOMOUS)
                           if access.allowed(hub.config.get(t["server"], {}), tier, t["name"])]} for t in tools],
            "trusted": list(gw.mcp_trusted), "wander": dict(gw.wander), "wandering": gw.wandering,
            "journal": gw.wander_log[:10]})

    @router.post("/api/outside")
    async def outside_set(body: OutsideRequest, request: Request):
        authorize(request)
        if body.wander_enabled is not None:
            gw.wander["enabled"] = bool(body.wander_enabled)
        if body.per_day is not None:
            gw.wander["per_day"] = max(0, min(12, int(body.per_day)))
        return json_response({"wander": dict(gw.wander)})

    @router.post("/api/evolve")
    async def evolve_submit(body: EvolveRequest, request: Request):
        authorize(request)
        bridge = getattr(gw, "evolve_agent", None)
        if bridge is None or not gw.evolve_enabled:
            raise HTTPException(400, "改造通道没有开启（config.json: EVOLVE_ENABLED）")
        task = body.task.strip()
        if not task:
            raise HTTPException(400, "要改什么？task 不能为空")
        result = await asyncio.to_thread(bridge.propose, task, "console", "self_evolution")
        return json_response(result)

    @router.post("/api/evolve/apply")
    async def evolve_apply(body: EvolveRequest, request: Request):
        authorize(request)
        bridge = getattr(gw, "evolve_agent", None)
        if bridge is None or not gw.evolve_enabled:
            raise HTTPException(400, "改造通道没有开启")
        if not body.id:
            raise HTTPException(400, "缺少提案 id")
        result = await asyncio.to_thread(bridge.apply, body.id, "console")
        if not result.get("ok"):
            raise HTTPException(400, result.get("error") or "应用失败")
        return json_response(result)

    # ---- QQ 能力（调试 / 手动触发）：OCR、取图、语音转写、合并转发、群历史 ----
    @router.post("/api/qq")
    async def qq_action(body: QqRequest, request: Request):
        authorize(request)
        from napcat.qq import qq as _qq
        if not _qq.connected():
            raise HTTPException(400, "NapCat 未连接")
        data = await _qq.call(body.action, body.params or {}, timeout=float(body.timeout))
        return json_response({"action": body.action, "ok": data is not None, "data": data})

    return router
