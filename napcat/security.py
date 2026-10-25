import hashlib
import hmac
from typing import Optional

# 存储机器人的 token 信息
token_store: dict[str, str] = {}

# 设置机器人 token
def set_robot_token(robot_id: str, token: str):
    token_store[robot_id] = token

# 获取机器人 token
def get_robot_token(robot_id: str) -> Optional[str]:
    return token_store.get(robot_id)

# 校验请求签名
def verify_signature(robot_id: str, timestamp: str, signature: str, body: bytes) -> bool:
    token = get_robot_token(robot_id)
    if not token:
        return True  # 未设置 token 时跳过校验
    
    # 计算签名
    secret = f"{token}".encode()
    message = f"{timestamp}".encode() + b"\n" + body
    expected_signature = hmac.new(secret, message, digestmod=hashlib.sha256).hexdigest()
    
    return hmac.compare_digest(expected_signature, signature)

# 校验 token
def verify_token(robot_id: str, token: Optional[str]) -> bool:
    if not token:
        return True  # 未提供 token 时跳过校验
    
    stored_token = get_robot_token(robot_id)
    if not stored_token:
        return True  # 未设置 token 时跳过校验
    
    return token == stored_token
