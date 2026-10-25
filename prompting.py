from typing import Dict, Any


class PromptContext:
    """Prompt上下文类，包含构建Prompt所需的所有信息"""
    
    def __init__(self, persona: str, state: Dict[str, Any], policy: Dict[str, Any], 
                 memory: Dict[str, Any], window: str):
        """初始化PromptContext
        
        Args:
            persona: 角色信息
            state: 状态信息
            policy: 策略信息
            memory: 记忆信息
            window: 对话窗口
        """
        self.persona = persona
        self.state = state
        self.policy = policy
        self.memory = memory
        self.window = window
    
    def build_prompt(self) -> str:
        """构建完整的Prompt
        
        Returns:
            str: 完整的Prompt
        """
        # 系统提示词，强制模型输出JSON
        system_prompt = """你是一个智能助手，必须严格按照JSON格式输出响应，不得输出任何JSON之外的字符。

JSON格式要求：
{
  "reply": "用户可见的回复内容",
  "internal_note": "系统内部笔记，不可泄露给用户",
  "event": {"type": "事件类型", "intensity": 事件强度(1-10)},
  "memory_write": {
    "should_write": 是否写入记忆(boolean),
    "type": "记忆类型",
    "content": "记忆内容(一句话事件化)",
    "conflict_key": "冲突键",
    "confidence": 置信度(0.0-1.0)
  }
}

字段说明：
- reply: 用于用户可见输出，必须清晰、友好
- internal_note: 系统内部笔记，用于记录推理过程、决策理由等，不可泄露给用户
- event: 可选字段，用于驱动状态更新
- memory_write: 可选字段，用于控制记忆写入

请严格按照上述格式输出，不得添加任何注释或额外说明。"""
        
        # 构建用户提示词
        user_prompt = f"""
角色信息：
{self.persona}

当前状态：
{self.state}

策略要求：
{self.policy}

历史记忆：
{self.memory}

对话窗口：
{self.window}
        """
        
        return f"{system_prompt}\n\n用户输入：\n{user_prompt}\n\nAssistant："


# 示例Persona
DEFAULT_PERSONA = """你是一个智能助手，名字叫Noa，擅长帮助用户解决各种问题，提供友好、准确的回复。"""

# 示例State
DEFAULT_STATE = {
    "current_time": "2026-01-16",
    "mood": "neutral",
    "energy": 100
}

# 示例Policy
DEFAULT_POLICY = {
    "max_reply_length": 500,
    "safety_check": True,
    "memory_write_enabled": True
}
