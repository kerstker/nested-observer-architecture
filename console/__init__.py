"""
Noa 控制台：在浏览器里实时看她的内心状态、和她说话、做一些操作。

挂在 app.py 的 FastAPI 上，地址 http://<主机>:8080/console
- 没设 CONSOLE_TOKEN 时只允许本机访问
- 设了 CONSOLE_TOKEN 后可以远程访问，打开 /console?token=<token>
"""
import asyncio
import hmac
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from .state import snapshot, to_json

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

# Feeding and petting go through the normal perception path, so they are experienced
# (appraisal, cause log, episode) exactly like the same words from QQ would be.
SPOKEN_ACTIONS = {"feed": "喂你吃小鱼干", "pet": "摸摸头"}

def create_console_router(gw, token: str = None) -> APIRouter:
    router = APIRouter(prefix="/console")

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
        else:
            raise HTTPException(400, f"未知操作：{a}")
        return json_response({"result": result, "state": await asyncio.to_thread(snapshot, gw)})

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

    return router
