from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, Set
from collections import deque
import logging
import json
import asyncio
import hmac
from .event_models import MessageEvent, VersionInfo
from .security import verify_token, verify_signature
from .message_utils import extract_plain_text
from .qq import qq

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI()

# 存储机器人实例信息
robots: Dict[str, Dict[str, Any]] = {}

# 消息处理器回调
message_handler = None


# WebSocket 连接管理器
class WebSocketManager:
    def __init__(self):
        # 活动连接的 WebSocket 客户端
        self.active_connections: Set[WebSocket] = set()
        # 机器人 ID 到 WebSocket 的映射
        self.robot_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, robot_id: str):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.robot_connections[robot_id] = websocket
        logger.info(f"WebSocket connection established for robot {robot_id}")
    
    def disconnect(self, websocket: WebSocket, robot_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if robot_id in self.robot_connections:
            del self.robot_connections[robot_id]
        logger.info(f"WebSocket connection closed for robot {robot_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
    
    async def send_to_robot(self, robot_id: str, message: Dict[str, Any]):
        if robot_id in self.robot_connections:
            websocket = self.robot_connections[robot_id]
            await websocket.send_json(message)
            return True
        return False


# 创建 WebSocket 管理器实例
ws_manager = WebSocketManager()
qq.manager = ws_manager

# QQ 里不是消息的事件（戳一戳、撤回、进群、禁言、表情回应……）和请求（加好友、拉群）
event_handler = None

def register_event_handler(handler):
    global event_handler
    event_handler = handler

def register_message_handler(handler):
    """注册消息处理器"""
    global message_handler
    message_handler = handler

# NapCat 里配置的访问令牌（反向 WebSocket / HTTP 上报都要带上）。为空时不校验。
access_token: str = ""

def set_access_token(token: Optional[str]):
    global access_token
    access_token = token or ""

def _token_ok(headers, query_params) -> bool:
    if not access_token:
        return True
    given = headers.get("authorization", "")
    if given.lower().startswith("bearer "):
        given = given[7:]
    given = given or query_params.get("access_token", "")
    return hmac.compare_digest(given, access_token)

# 她自己发出去的消息的 message_id：别人"引用回复"这些消息，就是在回她
_sent_message_ids: deque = deque(maxlen=1000)

def _remember_sent(response: Dict[str, Any]):
    data = response.get("data")
    if response.get("retcode") == 0 and isinstance(data, dict) and data.get("message_id") is not None:
        _sent_message_ids.append(str(data["message_id"]))

def is_own_message(message_id: Any) -> bool:
    return str(message_id) in _sent_message_ids

# 每条消息在自己的任务里处理：生成回复要好几秒，不能卡住 WebSocket 的接收循环
_handler_tasks: Set[asyncio.Task] = set()

def _dispatch_event(robot_id: str, data: Dict[str, Any]):
    if not event_handler:
        return
    task = asyncio.create_task(event_handler(robot_id, data))
    _handler_tasks.add(task)
    task.add_done_callback(_handler_tasks.discard)

def _handle_frame(robot_id: str, data: Dict[str, Any]):
    """One frame from NapCat: an event to live through, or the answer to one of her API calls."""
    if data.get("self_id"):
        qq.self_id = str(data["self_id"])
    if "post_type" in data:
        if data["post_type"] == "message":
            _dispatch(robot_id, MessageEvent(**data))
        elif data["post_type"] in ("notice", "request"):
            _dispatch_event(robot_id, data)
        elif data.get("meta_event_type") == "lifecycle" and data.get("sub_type") == "connect":
            _dispatch_event(robot_id, data)
    elif "retcode" in data:
        if not qq.resolve(data):
            logger.info(f"Received API response: {data}")
        _remember_sent(data)
    else:
        logger.warning(f"Received unknown data format: {data}")

def _dispatch(robot_id: str, message_event: MessageEvent):
    if not message_handler:
        logger.info(f"No message handler registered for {robot_id}")
        return
    task = asyncio.create_task(message_handler(robot_id, message_event))
    _handler_tasks.add(task)
    task.add_done_callback(_handler_tasks.discard)

# OneBot v11 Webhook 上报端点
@app.post("/onebot/v11/webhook/{robot_id}")
async def webhook(robot_id: str, request: Request):
    try:
        # 获取请求头
        timestamp = request.headers.get("X-Timestamp", "")
        signature = request.headers.get("X-Signature", "")
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        
        # 获取请求体
        body_bytes = await request.body()
        data = json.loads(body_bytes)
        
        # 校验 token
        if not _token_ok(request.headers, request.query_params) or not verify_token(robot_id, token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # 校验签名
        if not verify_signature(robot_id, timestamp, signature, body_bytes):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        logger.info(f"Received webhook from {robot_id}: {json.dumps(data, ensure_ascii=False)}")
        
        # 存储机器人信息
        if robot_id not in robots:
            robots[robot_id] = {
                "last_message": data,
                "status": "online"
            }
        else:
            robots[robot_id]["last_message"] = data
        
        # 处理消息事件
        if data["post_type"] == "message":
            # 转换为 MessageEvent 对象
            _dispatch(robot_id, MessageEvent(**data))

        return {"status": "ok"}
    except HTTPException:
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from {robot_id}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# OneBot v11 HTTP API 端点 - 获取版本信息
@app.get("/onebot/v11/{robot_id}/get_version_info")
async def get_version_info(robot_id: str):
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail="Robot not found")
    
    return {
        "status": "ok",
        "data": VersionInfo(
            app_name="NapCat",
            app_version="1.0.0",
            protocol_version="v11",
            onebot_version="11"
        ).model_dump(),
        "retcode": 0,
        "message": ""
    }

# OneBot v11 HTTP API 端点 - 发送私聊消息
@app.post("/onebot/v11/{robot_id}/send_private_msg")
async def send_private_msg(robot_id: str, request: Request):
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail="Robot not found")
    
    try:
        data = await request.json()
        logger.info(f"Sending private message from {robot_id}: {json.dumps(data, ensure_ascii=False)}")
        
        # 这里可以添加实际发送消息的逻辑
        # 目前只是模拟返回
        return {
            "status": "ok",
            "data": {
                "message_id": 123456
            },
            "retcode": 0,
            "message": ""
        }
    except Exception as e:
        logger.error(f"Error sending private message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# OneBot v11 HTTP API 端点 - 发送群消息
@app.post("/onebot/v11/{robot_id}/send_group_msg")
async def send_group_msg(robot_id: str, request: Request):
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail="Robot not found")
    
    try:
        data = await request.json()
        logger.info(f"Sending group message from {robot_id}: {json.dumps(data, ensure_ascii=False)}")
        
        # 这里可以添加实际发送消息的逻辑
        # 目前只是模拟返回
        return {
            "status": "ok",
            "data": {
                "message_id": 123456
            },
            "retcode": 0,
            "message": ""
        }
    except Exception as e:
        logger.error(f"Error sending group message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 发送消息的辅助函数
def send_message(robot_id: str, message_type: str, target_id: int, message: str):
    """发送消息的辅助函数
    
    Args:
        robot_id: 机器人 ID
        message_type: 消息类型，private 或 group
        target_id: 目标 ID，私聊为 user_id，群聊为 group_id
        message: 要发送的消息内容
        
    Returns:
        发送结果
    """
    # 这里可以添加实际发送消息的逻辑
    # 目前只是记录日志
    logger.info(f"Sending {message_type} message to {target_id} from {robot_id}: {message}")
    return {
        "status": "ok",
        "message_id": 123456
    }

# 获取机器人状态
@app.get("/robots/{robot_id}")
async def get_robot_status(robot_id: str):
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail="Robot not found")
    
    return robots[robot_id]

# 获取所有机器人
@app.get("/robots")
async def get_all_robots():
    return robots

# WebSocket 端点 - 用于 NapCat 客户端连接
@app.websocket("/onebot/v11/ws/{robot_id}")
async def websocket_endpoint(websocket: WebSocket, robot_id: str):
    if not _token_ok(websocket.headers, websocket.query_params):
        logger.warning(f"Rejected WebSocket connection for {robot_id}: bad access token")
        await websocket.close(code=1008)
        return
    await ws_manager.connect(websocket, robot_id)
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket message from {robot_id}: {json.dumps(data, ensure_ascii=False)}")
            
            # 存储机器人信息
            if robot_id not in robots:
                robots[robot_id] = {
                    "last_message": data,
                    "status": "online"
                }
            else:
                robots[robot_id]["last_message"] = data
            
            _handle_frame(robot_id, data)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, robot_id)
    except Exception as e:
        logger.error(f"WebSocket error for robot {robot_id}: {e}")
        ws_manager.disconnect(websocket, robot_id)

# 兼容的 WebSocket 端点 - 处理 /webhook/{robot_id} 的 WebSocket 连接
@app.websocket("/onebot/v11/webhook/{robot_id}")
async def websocket_webhook_endpoint(websocket: WebSocket, robot_id: str):
    if not _token_ok(websocket.headers, websocket.query_params):
        logger.warning(f"Rejected WebSocket connection for {robot_id}: bad access token")
        await websocket.close(code=1008)
        return
    await ws_manager.connect(websocket, robot_id)
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket message from {robot_id} via webhook endpoint: {json.dumps(data, ensure_ascii=False)}")
            
            # 存储机器人信息
            if robot_id not in robots:
                robots[robot_id] = {
                    "last_message": data,
                    "status": "online"
                }
            else:
                robots[robot_id]["last_message"] = data
            
            _handle_frame(robot_id, data)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, robot_id)
    except Exception as e:
        logger.error(f"WebSocket error for robot {robot_id} via webhook endpoint: {e}")
        ws_manager.disconnect(websocket, robot_id)
