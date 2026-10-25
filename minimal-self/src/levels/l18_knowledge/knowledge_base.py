from typing import List, Dict, Any
from src.levels.l2_memory.storage.database import MemoryDatabase
import logging

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Level 18: Knowledge Base System
    Provides long-term storage and retrieval of factual knowledge (RAG).
    """
    def __init__(self, db: MemoryDatabase):
        self.db = db

    def memorize(self, title: str, content: str, tags: List[str], source: str = "manual"):
        """Save new knowledge"""
        try:
            self.db.add_knowledge(title, content, tags, source)
            logger.info(f"Knowledge memorized: {title}")
        except Exception as e:
            logger.error(f"Failed to memorize knowledge: {e}")

    def recall(self, query: str, limit: int = 3) -> str:
        """
        Search knowledge base and return formatted string.
        Returns empty string if nothing found.
        """
        try:
            results = self.db.search_knowledge(query, limit)
            if not results:
                return ""
            
            formatted = "Recall Results:\n"
            for r in results:
                formatted += f"- [{r['title']}] (Tags: {r['tags']}): {r['content']}\n"
            return formatted
        except Exception as e:
            logger.error(f"Failed to recall knowledge: {e}")
            return ""
