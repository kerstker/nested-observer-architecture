import asyncio
import websockets
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, Union
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WSClient:
    """WebSocket 客户端，支持长期运行、心跳、重连和多种 token 注入方式"""
    
    def __init__(self, 
                 ws_url: str, 
                 token: str = "",
                 token_header: str = "Authorization",
                 token_query_param: str = "access_token",
                 token_in_header: bool = True,
                 token_in_query: bool = False,
                 heartbeat_interval: int = 30,
                 max_reconnect_time: int = 30,
                 message_handler: Optional[Callable] = None):
        """初始化 WebSocket 客户端
        
        Args:
            ws_url: WebSocket URL
            token: 认证 token
            token_header: token 头部名称
            token_query_param: token 查询参数名称
            token_in_header: 是否在 header 中注入 token
            token_in_query: 是否在 URL query 中注入 token
            heartbeat_interval: 心跳间隔（秒）
            max_reconnect_time: 最大重连时间（秒）
            message_handler: 消息处理函数
        """
        self.ws_url = ws_url
        self.token = token
        self.token_header = token_header
        self.token_query_param = token_query_param
        self.token_in_header = token_in_header
        self.token_in_query = token_in_query
        self.heartbeat_interval = heartbeat_interval
        self.max_reconnect_time = max_reconnect_time
        self.message_handler = message_handler
        
        # 连接状态
        self.ws = None
        self.connected = False
        self.reconnect_count = 0
        self.is_closing = False
        
        # 任务
        self.heartbeat_task = None
        self.reconnect_task = None
    
    def _inject_token(self, url: str) -> str:
        """注入 token 到 URL 或 header
        
        Args:
            url: 原始 URL
            
        Returns:
            str: 注入 token 后的 URL
        """
        if self.token and self.token_in_query:
            # 在 URL query 中注入 token
            parsed_url = urlparse(url)
            query = parse_qs(parsed_url.query)
            query[self.token_query_param] = [self.token]
            new_query = urlencode(query, doseq=True)
            new_url = urlunparse(parsed_url._replace(query=new_query))
            return new_url
        return url
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头
        
        Returns:
            Dict[str, str]: 请求头
        """
        headers = {}
        if self.token and self.token_in_header:
            # 在 header 中注入 token
            if self.token_header.lower() == "authorization":
                headers[self.token_header] = f"Bearer {self.token}"
            else:
                headers[self.token_header] = self.token
        return headers
    
    async def _heartbeat(self):
        """心跳任务"""
        while self.connected and not self.is_closing:
            try:
                # 发送心跳
                if self.ws and self.connected:
                    # 尝试发送 OneBot get_status（如果支持）
                    heartbeat_data = {
                        "action": "get_status",
                        "params": {},
                        "echo": f"heartbeat_{time.time()}"
                    }
                    await self.ws.send(json.dumps(heartbeat_data))
                    logger.debug("Sent heartbeat: get_status")
                
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                # 心跳失败，触发重连
                await self._handle_disconnect()
                break
    
    async def _handle_disconnect(self):
        """处理断开连接"""
        if self.connected:
            self.connected = False
            logger.info("WebSocket disconnected, starting reconnection...")
            
            # 取消心跳任务
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                self.heartbeat_task = None
            
            # 关闭 WebSocket
            if self.ws:
                try:
                    await self.ws.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                self.ws = None
            
            # 启动重连
            await self._reconnect()
    
    async def _reconnect(self):
        """重连逻辑，使用指数退避策略"""
        while not self.connected and not self.is_closing:
            try:
                # 计算重连延迟（指数退避）
                delay = min(2 ** self.reconnect_count, self.max_reconnect_time)
                logger.info(f"Reconnecting in {delay} seconds...")
                
                await asyncio.sleep(delay)
                
                # 注入 token 到 URL
                url_with_token = self._inject_token(self.ws_url)
                
                # 获取请求头
                headers = self._get_headers()
                
                # 连接 WebSocket
                # 使用正确的方式传递 headers
                self.ws = await websockets.connect(url_with_token, additional_headers=headers)
                self.connected = True
                self.reconnect_count = 0
                logger.info("WebSocket reconnected successfully")
                
                # 启动心跳任务
                self.heartbeat_task = asyncio.create_task(self._heartbeat())
                
                # 启动消息接收循环
                asyncio.create_task(self._receive_messages())
                
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                self.reconnect_count += 1
                
                if self.reconnect_count > 10:
                    logger.error("Too many reconnection attempts, will continue trying...")
    
    async def _receive_messages(self):
        """接收消息循环"""
        while self.connected and not self.is_closing:
            try:
                if self.ws:
                    message = await self.ws.recv()
                    logger.debug(f"Received message: {message}")
                    
                    # 解析消息
                    data = json.loads(message)
                    
                    # 调用消息处理器
                    if self.message_handler:
                        await self.message_handler(data)
            except websockets.exceptions.ConnectionClosedOK:
                logger.info("WebSocket connection closed cleanly")
                await self._handle_disconnect()
                break
            except websockets.exceptions.ConnectionClosedError:
                logger.error("WebSocket connection closed with error")
                await self._handle_disconnect()
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {message}, error: {e}")
            except Exception as e:
                logger.error(f"Error in message receiver: {e}")
                # 继续运行，不崩溃
                await asyncio.sleep(1)
    
    async def start(self):
        """启动 WebSocket 客户端"""
        logger.info(f"Starting WebSocket client to {self.ws_url}")
        
        # 注入 token 到 URL
        url_with_token = self._inject_token(self.ws_url)
        
        # 获取请求头
        headers = self._get_headers()
        
        try:
            # 连接 WebSocket
            # 使用正确的方式传递 headers
            self.ws = await websockets.connect(url_with_token, additional_headers=headers)
            self.connected = True
            self.reconnect_count = 0
            logger.info("WebSocket connected successfully")
            
            # 启动心跳任务
            self.heartbeat_task = asyncio.create_task(self._heartbeat())
            
            # 启动消息接收循环
            await self._receive_messages()
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            await self._handle_disconnect()
    
    async def send(self, data: Dict[str, Any]) -> bool:
        """发送消息
        
        Args:
            data: 要发送的数据
            
        Returns:
            bool: 是否发送成功
        """
        try:
            if self.connected and self.ws:
                await self.ws.send(json.dumps(data))
                logger.debug(f"Sent message: {data}")
                return True
            else:
                logger.error("Cannot send message: WebSocket not connected")
                return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def close(self):
        """关闭 WebSocket 客户端"""
        logger.info("Closing WebSocket client...")
        
        self.is_closing = True
        
        # 取消心跳任务
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            self.heartbeat_task = None
        
        # 关闭 WebSocket
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            self.ws = None
        
        self.connected = False
        logger.info("WebSocket client closed")
    
    def register_message_handler(self, handler: Callable):
        """注册消息处理器
        
        Args:
            handler: 消息处理函数
        """
        self.message_handler = handler
