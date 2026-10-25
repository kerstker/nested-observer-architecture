import time
from typing import Dict, Any, List

from src.levels.l2_memory.storage.database import MemoryDatabase

class RealTimeGroupRelationAnalyzer:
    """
    实时群组关系分析器
    - 解析 Napcat 事件结构，识别 @ 提及、同线程交互
    - 更新群组关系图（权重、情感）
    - 提供快照与邻居查询用于上下文理解
    """
    POSITIVE_WORDS = {"谢谢", "喜欢", "不错", "赞", "好", "棒", "爱", "支持"}
    NEGATIVE_WORDS = {"讨厌", "差", "坏", "不行", "生气", "骂", "滚", "黑"}

    def __init__(self, db: MemoryDatabase):
        self.db = db

    def _sentiment_delta(self, text: str) -> float:
        delta = 0.0
        for w in self.POSITIVE_WORDS:
            if w in text:
                delta += 0.2
        for w in self.NEGATIVE_WORDS:
            if w in text:
                delta -= 0.3
        return max(-1.0, min(1.0, delta))

    def on_message(self, event: Dict[str, Any]):
        """
        解析群消息并更新关系图
        """
        if event.get("message_type") != "group":
            return
        group_id = str(event.get("group_id"))
        sender = event.get("sender", {})
        user_id = str(sender.get("user_id"))
        nickname = sender.get("nickname", "")
        message_id = str(event.get("message_id", ""))
        segments = event.get("message", [])
        raw_message = event.get("raw_message", "")

        # 成员登记
        self.db.upsert_group_member(group_id, user_id, nickname)

        # 查找 @ 提及
        mentioned_ids: List[str] = []
        if isinstance(segments, list):
            for seg in segments:
                if isinstance(seg, dict) and seg.get("type") == "at":
                    qq = seg.get("data", {}).get("qq")
                    if qq:
                        mentioned_ids.append(str(qq))

        # 计算情感
        text = ""
        if isinstance(segments, list):
            for seg in segments:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    text += seg.get("data", {}).get("text", "")
                elif isinstance(seg, str):
                    text += seg
        elif isinstance(segments, str):
            text = segments
        if not text and raw_message:
            text = raw_message

        senti = self._sentiment_delta(text)

        # 更新关系图：sender -> mentioned
        if mentioned_ids:
            for mid in mentioned_ids:
                self.db.upsert_group_member(group_id, mid, "")  # 记录被提及者
                self.db.update_group_relation(group_id, user_id, mid, weight_delta=1.0, sentiment_delta=senti, message_id=message_id, event_type="mention")
        else:
            # 未显式 @，仍按同线程交互加弱边（对群内所有成员弱提升需避免爆炸，这里跳过）
            pass

    def snapshot_text(self, group_id: str, target_user_id: str, limit: int = 5) -> str:
        """
        返回面向 Prompt 的关系快照文本
        """
        neighbors = self.db.get_user_neighbors(group_id, target_user_id, limit=limit)
        if not neighbors:
            return "Group Relations: None"
        lines = []
        for e in neighbors:
            other = e["user_id_b"] if e["user_id_a"] == target_user_id else e["user_id_a"]
            lines.append(f"- {target_user_id} <-> {other} (w={e['weight']:.1f}, s={e['sentiment']:.2f})")
        return "Group Relations:\n" + "\n".join(lines)
