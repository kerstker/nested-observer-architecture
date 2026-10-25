import sqlite3
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os
from src.core import clock

class MemoryDatabase:
    def __init__(self, db_path: str = "memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 1. Users Table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            nickname TEXT,
            first_seen REAL,
            last_seen REAL,
            platform TEXT,
            metadata TEXT  -- JSON encrypted/obfuscated
        )''')
        
        # 2. Interactions Table (History)
        c.execute('''CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            group_id TEXT, -- Added for group context
            role TEXT, -- 'user' or 'assistant'
            content TEXT,
            timestamp REAL,
            session_id TEXT,
            context_json TEXT -- Additional context
        )''')
        
        # Migration: Add group_id if not exists
        try:
            c.execute("ALTER TABLE interactions ADD COLUMN group_id TEXT")
        except sqlite3.OperationalError:
            pass # Already exists
        
        # 3. Long-term Memories (Facts/Summary)
        c.execute('''CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            category TEXT, -- 'fact', 'preference', 'summary'
            content TEXT,
            confidence REAL,
            created_at REAL,
            last_accessed REAL,
            access_count INTEGER DEFAULT 0
        )''')
        
        # 4. Relationships
        c.execute('''CREATE TABLE IF NOT EXISTS relationships (
            user_id_a TEXT,
            user_id_b TEXT,
            relation_type TEXT,
            strength REAL,
            updated_at REAL,
            PRIMARY KEY (user_id_a, user_id_b)
        )''')
        
        # 5. Memes (Cultural Norms)
        c.execute('''CREATE TABLE IF NOT EXISTS memes (
            content TEXT PRIMARY KEY,
            frequency INTEGER DEFAULT 1,
            first_seen REAL,
            last_used REAL,
            source_user_id TEXT
        )''')
        
        # 6.1 Group Members
        c.execute('''CREATE TABLE IF NOT EXISTS group_members (
            group_id TEXT,
            user_id TEXT,
            nickname TEXT,
            first_seen REAL,
            last_seen REAL,
            PRIMARY KEY (group_id, user_id)
        )''')
        
        # 6.2 Group Relations Graph (Undirected edge stored as ordered pair)
        c.execute('''CREATE TABLE IF NOT EXISTS group_relations (
            group_id TEXT,
            user_id_a TEXT,
            user_id_b TEXT,
            weight REAL DEFAULT 0.0,
            sentiment REAL DEFAULT 0.0,
            last_interaction REAL,
            PRIMARY KEY (group_id, user_id_a, user_id_b)
        )''')
        
        # 6.3 Relation Events (History of changes)
        c.execute('''CREATE TABLE IF NOT EXISTS relation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT,
            user_id_a TEXT,
            user_id_b TEXT,
            event_type TEXT, -- mention, reply, message_same_thread, positivity, negativity
            delta_weight REAL,
            delta_sentiment REAL,
            message_id TEXT,
            timestamp REAL
        )''')
        
        # 6. Create Indexes for Performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user_timestamp ON interactions(user_id, timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_content ON memories(user_id, content)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_relationships_strength ON relationships(strength)')
        
        # 7. Dynamic Persona (Level 13)
        c.execute('''CREATE TABLE IF NOT EXISTS persona_traits (
            trait TEXT PRIMARY KEY,
            value REAL,
            description TEXT
        )''')
        
        # 8. Knowledge Base (Level 18)
        c.execute('''CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            tags TEXT,
            source TEXT,
            created_at REAL,
            access_count INTEGER DEFAULT 0
        )''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON knowledge(tags)')
        
        # 9. State Events (Global state change history)
        c.execute('''CREATE TABLE IF NOT EXISTS state_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT,
            state TEXT,
            old_value TEXT,
            new_value TEXT,
            source TEXT,
            timestamp REAL
        )''')
        
        # 10. Migration: idle inner thoughts do not belong in the knowledge base.
        # They are moved (not deleted) into their own archive.
        c.execute('''CREATE TABLE IF NOT EXISTS inner_speech_archive (
            id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            created_at REAL
        )''')
        c.execute('''INSERT OR IGNORE INTO inner_speech_archive (id, title, content, created_at)
                     SELECT id, title, content, created_at FROM knowledge WHERE source = 'inner_thought' ''')
        c.execute("DELETE FROM knowledge WHERE source = 'inner_thought'")

        conn.commit()
        conn.close()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)
    
    # --- Persona Management ---
    
    def get_persona_traits(self) -> Dict[str, Any]:
        """Get all persona traits"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM persona_traits")
        rows = c.fetchall()
        conn.close()
        
        traits = {}
        for r in rows:
            traits[r['trait']] = {"value": r['value'], "description": r['description']}
            
        # Initialize default if empty
        if not traits:
            defaults = {
                "tsundere": {"value": 0.7, "description": "Sometimes arrogant but affectionate"},
                "curiosity": {"value": 0.8, "description": "Very curious about the world"},
                "affection": {"value": 0.5, "description": "Warm but needs time to trust"},
                "energy": {"value": 0.6, "description": "Playful and active"}
            }
            for k, v in defaults.items():
                self.update_persona_trait(k, v['value'], v['description'])
            return defaults
            
        return traits

    def update_persona_trait(self, trait: str, value: float, description: str = None):
        """Update a persona trait"""
        conn = self._get_conn()
        c = conn.cursor()
        if description:
            c.execute("REPLACE INTO persona_traits (trait, value, description) VALUES (?, ?, ?)", (trait, value, description))
        else:
            c.execute("UPDATE persona_traits SET value = ? WHERE trait = ?", (value, trait))
        conn.commit()
        conn.close()

    # --- User Management ---
    
    def upsert_user(self, user_id: str, username: str, nickname: str, platform: str = "unknown"):
        conn = self._get_conn()
        c = conn.cursor()
        now = clock.now()
        
        c.execute("SELECT first_seen FROM users WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        
        if row:
            c.execute('''UPDATE users SET 
                username = ?, nickname = ?, last_seen = ?, platform = ?
                WHERE user_id = ?''', (username, nickname, now, platform, user_id))
        else:
            c.execute('''INSERT INTO users (user_id, username, nickname, first_seen, last_seen, platform, metadata)
                VALUES (?, ?, ?, ?, ?, ?, '{}')''', (user_id, username, nickname, now, now, platform))
        
        conn.commit()
        conn.close()

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    # --- Interaction History ---

    def add_interaction(self, user_id: str, role: str, content: str, session_id: str = "default", group_id: str = None):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''INSERT INTO interactions (user_id, role, content, timestamp, session_id, group_id)
            VALUES (?, ?, ?, ?, ?, ?)''', (user_id, role, content, clock.now(), session_id, group_id))
        conn.commit()
        conn.close()

    def get_recent_interactions(self, user_id: str, limit: int = 20, hours: int = 168, group_id: str = None) -> List[Dict[str, Any]]:
        """
        Get interactions from last 7 days (168 hours).
        Unified Context: Retrieves user's history + group's history.
        """
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        cutoff = clock.now() - (hours * 3600)
        
        if group_id:
            # Context = (This User's history ANYWHERE) OR (This Group's history ANYONE)
            # Actually, we usually want: 
            # 1. What this user said (Private or Group)
            # 2. What was said in this group (by anyone, for context)
            # But retrieving ALL group chat might be too noisy if limit is small.
            # Let's prioritize: User's interactions + Current Group interactions.
            # The *latest* `limit` messages, oldest first (this used to return the oldest of the week)
            c.execute('''SELECT * FROM (SELECT * FROM interactions
                WHERE (user_id = ? OR group_id = ?) AND timestamp > ?
                ORDER BY timestamp DESC, id DESC LIMIT ?) ORDER BY timestamp ASC, id ASC''', (user_id, group_id, cutoff, limit))
        else:
            # Private chat: Just user's history
            c.execute('''SELECT * FROM (SELECT * FROM interactions
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC, id DESC LIMIT ?) ORDER BY timestamp ASC, id ASC''', (user_id, cutoff, limit))
        
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def search_interactions(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search past interactions for relevant context"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Simple fuzzy match
        like_query = f"%{query}%"
        c.execute('''SELECT role, content, timestamp FROM interactions 
            WHERE user_id = ? AND content LIKE ?
            ORDER BY timestamp DESC LIMIT ?''', (user_id, like_query, limit))
        
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- Relationship Management ---
    
    def set_relationship(self, user_id: str, relation: str, strength: float = 1.0):
        """Set relationship between user and bot (self)"""
        conn = self._get_conn()
        c = conn.cursor()
        now = clock.now()
        # We use 'self' as the bot's ID in this local context
        c.execute('''REPLACE INTO relationships (user_id_a, user_id_b, relation_type, strength, updated_at)
            VALUES (?, 'self', ?, ?, ?)''', (user_id, relation, strength, now))
        conn.commit()
        conn.close()

    def get_relationship(self, user_id: str) -> Dict[str, Any]:
        """Get relationship with bot"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM relationships WHERE user_id_a = ? AND user_id_b = 'self'", (user_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else {"relation_type": "stranger", "strength": 0.0}

    # --- Meme Management ---

    def add_meme(self, content: str, source_user_id: str):
        conn = self._get_conn()
        c = conn.cursor()
        now = clock.now()
        
        c.execute("SELECT frequency FROM memes WHERE content = ?", (content,))
        row = c.fetchone()
        
        if row:
            c.execute("UPDATE memes SET frequency = frequency + 1, last_used = ? WHERE content = ?", (now, content))
        else:
            c.execute("INSERT INTO memes (content, frequency, first_seen, last_used, source_user_id) VALUES (?, 1, ?, ?, ?)", 
                      (content, now, now, source_user_id))
        
        conn.commit()
        conn.close()

    def get_top_memes(self, limit: int = 5) -> List[str]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT content FROM memes ORDER BY frequency DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return [r["content"] for r in rows]
    
    # --- Group Relations (Analysis & Storage) ---
    def upsert_group_member(self, group_id: str, user_id: str, nickname: str):
        conn = self._get_conn()
        c = conn.cursor()
        now = clock.now()
        c.execute('''REPLACE INTO group_members (group_id, user_id, nickname, first_seen, last_seen)
                     VALUES (?, ?, ?, COALESCE((SELECT first_seen FROM group_members WHERE group_id=? AND user_id=?), ?), ?)''',
                  (group_id, user_id, nickname, group_id, user_id, now, now))
        conn.commit()
        conn.close()
    
    def _normalize_pair(self, a: str, b: str):
        return (a, b) if a <= b else (b, a)
    
    def update_group_relation(self, group_id: str, user_id_a: str, user_id_b: str,
                              weight_delta: float, sentiment_delta: float, message_id: str = "", event_type: str = "interaction"):
        if user_id_a == user_id_b:
            return
        a, b = self._normalize_pair(user_id_a, user_id_b)
        conn = self._get_conn()
        c = conn.cursor()
        now = clock.now()
        # Upsert edge
        c.execute('''SELECT weight, sentiment FROM group_relations WHERE group_id=? AND user_id_a=? AND user_id_b=?''',
                  (group_id, a, b))
        row = c.fetchone()
        if row:
            new_weight = (row[0] or 0.0) + weight_delta
            new_sentiment = (row[1] or 0.0) + sentiment_delta
            c.execute('''UPDATE group_relations SET weight=?, sentiment=?, last_interaction=? 
                         WHERE group_id=? AND user_id_a=? AND user_id_b=?''',
                      (new_weight, new_sentiment, now, group_id, a, b))
        else:
            c.execute('''INSERT INTO group_relations (group_id, user_id_a, user_id_b, weight, sentiment, last_interaction)
                         VALUES (?, ?, ?, ?, ?, ?)''', (group_id, a, b, weight_delta, sentiment_delta, now))
        # Log event
        c.execute('''INSERT INTO relation_events (group_id, user_id_a, user_id_b, event_type, delta_weight, delta_sentiment, message_id, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (group_id, a, b, event_type, weight_delta, sentiment_delta, message_id, now))
        conn.commit()
        conn.close()
    
    def get_group_snapshot(self, group_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''SELECT * FROM group_relations WHERE group_id=? ORDER BY weight DESC, sentiment DESC LIMIT ?''',
                  (group_id, limit))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_user_neighbors(self, group_id: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''SELECT * FROM group_relations 
                     WHERE group_id=? AND (user_id_a=? OR user_id_b=?) 
                     ORDER BY weight DESC, sentiment DESC LIMIT ?''',
                  (group_id, user_id, user_id, limit))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- Memory Management ---

    def add_memory(self, user_id: str, category: str, content: str, confidence: float = 1.0):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''INSERT INTO memories (user_id, category, content, confidence, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)''', (user_id, category, content, confidence, clock.now(), clock.now()))
        conn.commit()
        conn.close()

    def search_memories(self, user_id: str, query: str = "", limit: int = 5) -> List[str]:
        """Simple keyword search for now. Vector search can be added later."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Simple fuzzy match
        like_query = f"%{query}%"
        c.execute('''SELECT content FROM memories 
            WHERE user_id = ? AND content LIKE ?
            ORDER BY last_accessed DESC LIMIT ?''', (user_id, like_query, limit))
        
        rows = c.fetchall()
        conn.close()
        return [r["content"] for r in rows]

    def get_all_memories(self, user_id: str, limit: int = 10) -> List[str]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''SELECT content FROM memories 
            WHERE user_id = ? 
            ORDER BY confidence DESC, created_at DESC LIMIT ?''', (user_id, limit))
        rows = c.fetchall()
        conn.close()
        return [r["content"] for r in rows]

    # --- Knowledge Base (Level 18) ---

    def add_knowledge(self, title: str, content: str, tags: List[str], source: str = "manual"):
        conn = self._get_conn()
        c = conn.cursor()
        tags_str = ",".join(tags)
        c.execute('''INSERT INTO knowledge (title, content, tags, source, created_at)
            VALUES (?, ?, ?, ?, ?)''', (title, content, tags_str, source, clock.now()))
        conn.commit()
        conn.close()

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Simple keyword match on title or content
        like_query = f"%{query}%"
        c.execute('''SELECT * FROM knowledge 
            WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
            ORDER BY access_count DESC, created_at DESC LIMIT ?''', 
            (like_query, like_query, like_query, limit))
        
        rows = c.fetchall()
        
        # Update access count for found items
        if rows:
            ids = [r['id'] for r in rows]
            placeholders = ','.join('?' * len(ids))
            c.execute(f"UPDATE knowledge SET access_count = access_count + 1 WHERE id IN ({placeholders})", ids)
            conn.commit()
            
        conn.close()
        return [dict(r) for r in rows]
    
    # --- State Events ---
    def add_state_event(self, level: str, state: str, old_value: str, new_value: str, source: str = "system"):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''INSERT INTO state_events (level, state, old_value, new_value, source, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?)''', (level, state, old_value, new_value, source, clock.now()))
        conn.commit()
        conn.close()
    
    def get_state_events(self, state: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        if state:
            c.execute('''SELECT * FROM state_events WHERE state = ? ORDER BY timestamp DESC LIMIT ?''', (state, limit))
        else:
            c.execute('''SELECT * FROM state_events ORDER BY timestamp DESC LIMIT ?''', (limit,))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]
