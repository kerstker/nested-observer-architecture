from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# OneBot 事件基础模型
class OneBotEvent(BaseModel):
    post_type: str
    self_id: Optional[int] = None
    time: Optional[int] = None

# 消息事件模型
class MessageEvent(OneBotEvent):
    post_type: str = "message"
    message_type: str  # private/group
    sub_type: Optional[str] = None
    message_id: Optional[int] = None
    user_id: int
    message: Any  # 可以是字符串或消息段列表
    raw_message: Optional[str] = None
    font: Optional[int] = None
    sender: Optional[Dict[str, Any]] = None
    group_id: Optional[int] = None  # 群消息特有

# 私聊消息事件
class PrivateMessageEvent(MessageEvent):
    message_type: str = "private"

# 群消息事件
class GroupMessageEvent(MessageEvent):
    message_type: str = "group"
    group_id: int
    anonymous: Optional[Dict[str, Any]] = None

# 通知事件模型
class NoticeEvent(OneBotEvent):
    post_type: str = "notice"
    notice_type: str

# 请求事件模型
class RequestEvent(OneBotEvent):
    post_type: str = "request"
    request_type: str

# 消息段模型
class MessageSegment(BaseModel):
    type: str
    data: Dict[str, Any]

# 版本信息响应
class VersionInfo(BaseModel):
    app_name: str
    app_version: str
    protocol_version: str
    onebot_version: str

# 发送消息响应
class SendMessageResponse(BaseModel):
    message_id: int
