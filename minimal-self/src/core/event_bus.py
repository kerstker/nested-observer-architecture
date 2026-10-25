from enum import Enum
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import time

class EventType(Enum):
    # L1
    USER_INPUT = "user_input"
    SENSORY_STIMULUS = "sensory_stimulus"
    # L3
    METABOLISM_TICK = "metabolism_tick"
    ENERGY_LOW = "energy_low"
    SLEEP_START = "sleep_start"
    SLEEP_END = "sleep_end"
    # L4
    EMOTION_UPDATE = "emotion_update"
    # L5
    PAIN_SPIKE = "pain_spike"
    # L10
    REFLECTION = "reflection"
    DIALOGUE_COMPLETED = "dialogue_completed"
    WORKSPACE_BROADCAST = "workspace_broadcast"   # winners of a workspace cycle
    PREDICTION_ERROR = "prediction_error"
    # Global State
    STATE_CHANGED = "state_changed"

@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any]
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

import logging
import asyncio
from inspect import iscoroutinefunction

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], Any]]] = {}

    def subscribe(self, event_type: EventType, callback: Callable[[Event], Any]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def publish(self, event_type: EventType, payload: Dict[str, Any]):
        event = Event(event_type, payload)
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    if iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"[EventBus] Error in subscriber for {event_type}: {e}")
