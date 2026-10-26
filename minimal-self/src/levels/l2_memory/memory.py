import threading
import queue
import time
from typing import List, Dict, Any
from functools import lru_cache
from .storage.database import MemoryDatabase

class MemorySystem:
    """
    Level 2: Memory System
    Handles short-term context and long-term retrieval using SQLite persistence.
    Now with Async Write Queue for performance.
    """
    def __init__(self, db_path: str = "memory.sqlite"):
        self.db = MemoryDatabase(db_path)
        self.self_ids = set()        # her own QQ number(s): shown as herself, not as a stranger
        
        # Async Write Queue
        self.write_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_write_queue, daemon=True)
        self.worker_thread.start()

    def _process_write_queue(self):
        """Background thread to process DB writes"""
        while True:
            task = self.write_queue.get()
            try:
                if task is None: break

                func, args = task
                func(*args)
            except Exception as e:
                print(f"[Memory] Write Error: {e}")
            finally:
                # Always mark done, otherwise flush() would hang after a failed write
                self.write_queue.task_done()

    def flush(self):
        """Block until every queued write has reached the database."""
        self.write_queue.join()

    def add_message(self, user_id: str, role: str, content: str, group_id: str = None):
        """Add interaction to history (Async)"""
        # Note: We pass group_id to add_interaction
        self.write_queue.put((self.db.add_interaction, (user_id, role, content, "default", group_id)))

    def add_user_profile(self, user_id: str, username: str, nickname: str):
        """Update user profile (synchronous, so the nickname is resolvable for this very message)"""
        self.db.upsert_user(user_id, username, nickname)
        self._get_user_cached.cache_clear()

    def set_relationship(self, user_id: str, relation: str):
        """Set relationship (Async)"""
        self.write_queue.put((self.db.set_relationship, (user_id, relation)))
        self._get_relationship_cached.cache_clear()
        
    def add_meme(self, content: str, source_user_id: str):
        """Store meme (Async)"""
        self.write_queue.put((self.db.add_meme, (content, source_user_id)))

    @lru_cache(maxsize=100)
    def _get_user_cached(self, user_id: str):
        return self.db.get_user(user_id)

    @lru_cache(maxsize=100)
    def _get_relationship_cached(self, user_id: str):
        return self.db.get_relationship(user_id)

    def retrieve(self, user_id: str, query: str = "", k: int = 3, group_id: str = None,
                 memories: List[str] = None, transcript: str = None) -> str:
        """
        Retrieve relevant context for the conversation.
        Combines recent history and relevant long-term memories.
        Level 16.5: Adds explicit nickname resolution for identity clarity.
        Level G: Adds group relation snapshot for multi-user context.
        """
        # 1. Get recent history (Short-term)
        # Now supports group_id for unified context
        history = self.db.get_recent_interactions(user_id, limit=10, group_id=group_id)
        formatted_history = []
        
        participants = set()
        
        for h in history:
            role_name = "User" 
            content = h['content']
            
            if h['role'] == "assistant":
                formatted_history.append(f"Noa: {content}")
                continue
                
            # Resolve User Identity
            speaker_id = h['user_id']
            speaker_user = self._get_user_cached(speaker_id)
            speaker_name = speaker_user['nickname'] if speaker_user else f"User({speaker_id[-4:]})"
            
            participants.add(speaker_name)
            
            # Label
            if speaker_id == user_id:
                label = f"Target User ({speaker_name})"
            else:
                label = f"{speaker_name}"
                
            formatted_history.append(f"{label}: {content}")
            
        history_text = "\n".join(formatted_history)
        if transcript:
            # The chat log, from her side: everyone in the chat, her own lines, quotes, times
            history_text = ("（这是这个聊天里最近的消息，按时间顺序；“我（乃爱）”是我自己说的，"
                            "“引用 X 说的「…」”是在回复 X 的那句话）\n" + transcript)
        
        # 2. Get relevant long-term memories
        # Recall by meaning lives in the episodic/semantic layers: the caller passes what it found
        memories = memories or []

        memory_text = "\n".join([f"- {m}" for m in memories])
        
        # 3. Get User Profile & Relationship (Cached)
        user = self._get_user_cached(user_id)
        rel = self._get_relationship_cached(user_id)
        
        user_name = user['nickname'] if user else "Unknown"
        rel_text = f"Relationship: {rel['relation_type']} (Strength: {rel['strength']:.1f})"
        
        # 4. Who is in this chat: from the chat log when there is one (it has everyone, not just
        # the people she answered - "nobody else is here" made her ignore what she had just heard)
        if transcript:
            import re as _re
            participants |= set(_re.findall(r"\] ([^：\n]+?)：", transcript))
        other_participants = sorted(participants - {user_name, "Noa", "我（乃爱）"})
        others_text = "、".join(other_participants) if other_participants else "没有别人"

        # 5. Group Relations Snapshot
        relations_text = "（没有）"
        if group_id:
            snapshot = self.db.get_user_neighbors(group_id, user_id, limit=5)
            if snapshot:
                relations_lines = []
                for e in snapshot:
                    other = e["user_id_b"] if e["user_id_a"] == user_id else e["user_id_a"]
                    if other in self.self_ids:
                        other_name = "我（乃爱）"
                    else:
                        u = self._get_user_cached(other)
                        other_name = (u or {}).get("nickname") or f"QQ {other}"
                    relations_lines.append(f"- {user_name} 和 {other_name}：互动 {e['weight']:.0f} 次，气氛 {e['sentiment']:+.2f}")
                relations_text = "\n".join(relations_lines)
        
        return f"""
[这个聊天里有谁]
- 我（乃爱）；现在跟我说话的是 {user_name}
- 刚才也在说话的：{others_text}
- 我和 {user_name}：{rel['relation_type']}（亲近程度 {rel['strength']:.1f}）

[群里的人际关系]
{relations_text}

[以前的事]（从记忆里想起来的，不是刚才发生的）
{memory_text if memory_text else "（没想起什么）"}

[Recent Conversation]（刚才的聊天，最后一条是最新的）
{history_text}
"""
