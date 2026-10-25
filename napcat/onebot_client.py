import httpx
from typing import Dict, Any, Optional
from .event_models import SendMessageResponse

class OneBotClient:
    def __init__(self, robot_id: str, base_url: str, token: Optional[str] = None):
        self.robot_id = robot_id
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.client = httpx.AsyncClient(timeout=30.0)
    
    # 添加请求头
    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    # 发送私聊消息
    async def send_private_msg(self, user_id: int, message: Any, auto_escape: bool = False) -> SendMessageResponse:
        url = f"{self.base_url}/send_private_msg"
        data = {
            "user_id": user_id,
            "message": message,
            "auto_escape": auto_escape
        }
        
        response = await self.client.post(url, json=data, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()
        
        if result.get("retcode") != 0:
            raise Exception(f"Failed to send private message: {result.get('message', 'Unknown error')}")
        
        return SendMessageResponse(**result.get("data", {}))
    
    # 发送群消息
    async def send_group_msg(self, group_id: int, message: Any, auto_escape: bool = False) -> SendMessageResponse:
        url = f"{self.base_url}/send_group_msg"
        data = {
            "group_id": group_id,
            "message": message,
            "auto_escape": auto_escape
        }
        
        response = await self.client.post(url, json=data, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()
        
        if result.get("retcode") != 0:
            raise Exception(f"Failed to send group message: {result.get('message', 'Unknown error')}")
        
        return SendMessageResponse(**result.get("data", {}))
    
    # 获取版本信息
    async def get_version_info(self) -> Dict[str, Any]:
        url = f"{self.base_url}/get_version_info"
        
        response = await self.client.get(url, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()
        
        if result.get("retcode") != 0:
            raise Exception(f"Failed to get version info: {result.get('message', 'Unknown error')}")
        
        return result.get("data", {})
    
    # 获取机器人信息
    async def get_login_info(self) -> Dict[str, Any]:
        url = f"{self.base_url}/get_login_info"
        
        response = await self.client.get(url, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()
        
        if result.get("retcode") != 0:
            raise Exception(f"Failed to get login info: {result.get('message', 'Unknown error')}")
        
        return result.get("data", {})
    
    # 关闭客户端连接
    async def close(self):
        await self.client.aclose()

# 全局客户端存储
clients: Dict[str, OneBotClient] = {}

# 获取或创建客户端
def get_or_create_client(robot_id: str, base_url: str, token: Optional[str] = None) -> OneBotClient:
    if robot_id not in clients:
        clients[robot_id] = OneBotClient(robot_id, base_url, token)
    return clients[robot_id]

# 获取客户端
def get_client(robot_id: str) -> Optional[OneBotClient]:
    return clients.get(robot_id)

# 关闭所有客户端
async def close_all_clients():
    for client in clients.values():
        await client.close()
    clients.clear()
