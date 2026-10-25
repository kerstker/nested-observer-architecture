from typing import List, Dict, Any, Union
from .event_models import MessageEvent

# 典型样例注释
"""
样例1：私聊
{'post_type': 'message', 'message_type': 'private', 'user_id': 12345, 'message': '你好', 'raw_message': '你好'}

样例2：群聊@
{'post_type': 'message', 'message_type': 'group', 'group_id': 67890, 'user_id': 12345, 'message': [{"type": "at", "data": {"qq": "98765"}}, " 你好"], 'raw_message': '@机器人 你好'}

样例3：群聊前缀
{'post_type': 'message', 'message_type': 'group', 'group_id': 67890, 'user_id': 12345, 'message': '/noa 你好', 'raw_message': '/noa 你好'}

样例4：混合图片
{'post_type': 'message', 'message_type': 'group', 'group_id': 67890, 'user_id': 12345, 'message': [{"type": "image", "data": {"file": "image.jpg"}}, " 文字内容"], 'raw_message': '[图片] 文字内容'}

样例5：超长文本
{'post_type': 'message', 'message_type': 'private', 'user_id': 12345, 'message': '这是一段非常长的文本，用于测试文本拆分功能。' * 20, 'raw_message': '这是一段非常长的文本，用于测试文本拆分功能。' * 20}

样例6：纯表情
{'post_type': 'message', 'message_type': 'group', 'group_id': 67890, 'user_id': 12345, 'message': [{"type": "face", "data": {"id": "1"}}], 'raw_message': '[表情]'}
"""

def extract_plain_text(event: Union[MessageEvent, Dict[str, Any]]) -> str:
    """从 message 数组或 raw_message 提取纯文本（忽略图片/表情/json/at 的非文本部分）"""
    # 获取消息和原始消息
    if isinstance(event, MessageEvent):
        message = event.message
        raw_message = event.raw_message or ''
    else:
        message = event.get('message', '')
        raw_message = event.get('raw_message', '')
    
    if isinstance(message, str):
        # 如果 message 是字符串，直接返回
        return message.strip()
    elif isinstance(message, list):
        # 如果 message 是消息段列表，提取文本部分
        plain_text = []
        for segment in message:
            if isinstance(segment, dict):
                if segment.get('type') == 'text':
                    # 文本段
                    plain_text.append(segment.get('data', {}).get('text', ''))
                # 忽略图片、表情、json、at 等非文本段
            elif isinstance(segment, str):
                # 兼容可能的字符串元素
                plain_text.append(segment)
        return ''.join(plain_text).strip()
    else:
        # 其他情况，尝试使用 raw_message
        return raw_message.strip() if raw_message else ''

def should_trigger(event: Union[MessageEvent, Dict[str, Any]], config: Dict[str, Any], self_id: int = None) -> bool:
    """判断是否应该触发消息处理
    * 私聊：默认触发
    * 群聊：仅当 @机器人（使用 self_id 或 at 段）或以前缀触发（config BOT_TRIGGER_PREFIX="/noa"）
    """
    # 获取消息类型
    if isinstance(event, MessageEvent):
        message_type = event.message_type
        message = event.message
        raw_message = event.raw_message or ''
    else:
        message_type = event.get('message_type', '')
        message = event.get('message', [])
        raw_message = event.get('raw_message', '')
    
    # 私聊：默认触发
    if message_type == 'private':
        return True
    
    # 群聊：需要满足 @机器人 或 前缀触发
    elif message_type == 'group':
        trigger_prefix = config.get('BOT_TRIGGER_PREFIX', '/noa')
        
        # 检查是否有前缀
        if raw_message.startswith(trigger_prefix):
            return True
        
        # 检查是否 @机器人
        has_at = False
        if isinstance(message, list):
            for segment in message:
                if isinstance(segment, dict):
                    if segment.get('type') == 'at':
                        at_qq = segment.get('data', {}).get('qq', '')
                        if str(at_qq) == str(self_id):
                            has_at = True
                            break
        elif isinstance(raw_message, str):
            # 检查 raw_message 中是否包含 @机器人（CQ码形式）
            if f'[CQ:at,qq={self_id}]' in raw_message or f'@{self_id}' in raw_message:
                has_at = True
        
        if has_at:
            return True
        
        # 检查提取的纯文本是否以前缀触发
        plain_text = extract_plain_text(event)
        if plain_text.startswith(trigger_prefix):
            return True
            
        # 增强触发逻辑：检查是否包含名字（如 "乃爱", "Noa"）
        # 这样即使没有 At 也能回应
        keywords = ["乃爱", "Noa", "noa", "@乃爱", "@Noa"]
        if any(k in plain_text for k in keywords):
            return True
    
    # 其他情况：不触发
    return False

def split_reply(text: str, max_chars: int = 500) -> List[str]:
    """按段落/句号优先切分文本，必要时硬切
    - 优先按段落（\n\n）切分
    - 其次按句号（。）切分
    - 最后硬切分
    """
    if not text:
        return []
    
    # 1. 按段落切分
    paragraphs = []
    for para in text.split('\n\n'):
        if len(para) <= max_chars:
            paragraphs.append(para)
        else:
            # 2. 段落过长，按句号切分
            sentences = []
            current = ''
            for sentence in para.split('。'):
                sentence = sentence + '。' if sentence else ''
                if len(current) + len(sentence) <= max_chars:
                    current += sentence
                else:
                    if current:
                        sentences.append(current)
                    current = sentence
            if current:
                sentences.append(current)
            
            # 3. 句子过长，硬切分
            for sentence in sentences:
                if len(sentence) <= max_chars:
                    paragraphs.append(sentence)
                else:
                    # 硬切分
                    for i in range(0, len(sentence), max_chars):
                        paragraphs.append(sentence[i:i+max_chars])
    
    # 去除末尾空段落和多余空格
    return [para.strip() for para in paragraphs if para.strip()]