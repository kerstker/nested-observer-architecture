from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Event:
    """事件数据类"""
    type: str  # 事件类型
    intensity: int  # 事件强度 (1-10)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.type,
            "intensity": self.intensity
        }


@dataclass
class MemoryWrite:
    """记忆写入数据类"""
    should_write: bool  # 是否应该写入
    type: str  # 记忆类型
    content: str  # 记忆内容
    conflict_key: str  # 冲突键
    confidence: float  # 置信度 (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "should_write": self.should_write,
            "type": self.type,
            "content": self.content,
            "conflict_key": self.conflict_key,
            "confidence": self.confidence
        }


@dataclass
class LLMOutput:
    """LLM输出数据类"""
    reply: str  # 用户可见的回复
    internal_note: str  # 内部笔记，不回显
    event: Optional[Event] = None  # 事件信息（可选）
    memory_write: Optional[MemoryWrite] = None  # 记忆写入信息（可选）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "reply": self.reply,
            "internal_note": self.internal_note
        }
        
        if self.event:
            result["event"] = {
                "type": self.event.type,
                "intensity": self.event.intensity
            }
        
        if self.memory_write:
            result["memory_write"] = {
                "should_write": self.memory_write.should_write,
                "type": self.memory_write.type,
                "content": self.memory_write.content,
                "conflict_key": self.memory_write.conflict_key,
                "confidence": self.memory_write.confidence
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMOutput":
        """从字典转换为LLMOutput对象"""
        event = None
        if "event" in data:
            event_data = data["event"]
            event = Event(
                type=event_data["type"],
                intensity=event_data["intensity"]
            )
        
        memory_write = None
        if "memory_write" in data:
            memory_data = data["memory_write"]
            memory_write = MemoryWrite(
                should_write=memory_data["should_write"],
                type=memory_data["type"],
                content=memory_data["content"],
                conflict_key=memory_data["conflict_key"],
                confidence=memory_data["confidence"]
            )
        
        return cls(
            reply=data["reply"],
            internal_note=data["internal_note"],
            event=event,
            memory_write=memory_write
        )
