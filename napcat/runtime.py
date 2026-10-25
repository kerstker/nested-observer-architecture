from collections import OrderedDict
from typing import Dict, Any, Callable, Optional
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def exception_protected(func: Callable) -> Callable:
    """异常保护装饰器，确保任何 handler 异常不应导致 WS 主循环崩溃
    
    Args:
        func: 要保护的函数
        
    Returns:
        Callable: 保护后的函数
    """
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in handler {func.__name__}: {e}", exc_info=True)
            # 继续运行，不崩溃
    return wrapper


def sync_exception_protected(func: Callable) -> Callable:
    """同步函数的异常保护装饰器
    
    Args:
        func: 要保护的同步函数
        
    Returns:
        Callable: 保护后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in sync handler {func.__name__}: {e}", exc_info=True)
            # 继续运行，不崩溃
    return wrapper


class RuntimeManager:
    """运行时管理器，负责 LRU 去重和限流"""
    
    def __init__(self):
        # LRU 去重配置
        self.message_id_cache = OrderedDict()  # 存储最近的 message_id
        self.max_message_cache = 500  # 默认缓存 500 个 message_id
        
        # 限流配置
        self.rate_limit_interval = 1  # 限流间隔（秒）
        self.user_key_last_processed = {}  # 存储每个 user_key 最后处理的时间
    
    def should_process_message(self, message_id: str) -> bool:
        """检查消息是否应该被处理（去重）
        
        Args:
            message_id: 消息 ID
            
        Returns:
            bool: 是否应该处理该消息
        """
        if message_id in self.message_id_cache:
            # 见过该消息，忽略
            return False
        
        # 添加到缓存
        self.message_id_cache[message_id] = time.time()
        
        # 保持缓存大小
        if len(self.message_id_cache) > self.max_message_cache:
            # 移除最旧的消息
            self.message_id_cache.popitem(last=False)
        
        return True
    
    def should_process_user_key(self, user_key: str) -> bool:
        """检查用户是否应该被处理（限流）
        
        Args:
            user_key: 用户唯一标识
            
        Returns:
            bool: 是否应该处理该用户的消息
        """
        current_time = time.time()
        
        if user_key in self.user_key_last_processed:
            last_time = self.user_key_last_processed[user_key]
            if current_time - last_time < self.rate_limit_interval:
                # 1秒内处理过，更新时间戳但忽略消息
                self.user_key_last_processed[user_key] = current_time
                return False
        
        # 更新最后处理时间
        self.user_key_last_processed[user_key] = current_time
        return True
    
    def generate_user_key(self, event: Dict[str, Any]) -> str:
        """生成用户唯一标识
        
        Args:
            event: 消息事件
            
        Returns:
            str: 用户唯一标识
        """
        message_type = event.get('message_type', '')
        user_id = event.get('user_id', '')
        group_id = event.get('group_id', '')
        
        if message_type == 'private':
            # 私聊：user_key="qq:{user_id}"
            return f"qq:{user_id}"
        elif message_type == 'group':
            # 群聊：user_key="qq:{user_id}@group:{group_id}"
            return f"qq:{user_id}@group:{group_id}"
        else:
            # 其他类型，默认使用用户ID
            return f"qq:{user_id}"
