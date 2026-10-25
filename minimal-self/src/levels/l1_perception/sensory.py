from typing import Dict, Any, Optional
import re

class SensoryCortex:
    """
    Level 1: Perception System
    Responsible for parsing raw input into sensory signals.
    """
    def __init__(self):
        self.pain_keywords = ["打", "杀", "笨", "滚", "去死", "废物"]
        self.pleasure_keywords = ["摸摸", "抱抱", "亲亲", "好", "乖", "棒", "喂", "吃"]
        self.touch_keywords = ["摸", "揉", "戳", "抱"]
        self.thermal_keywords = {"冷": "cold", "热": "hot", "暖": "warm"}

    def process_text(self, text: str) -> Optional[dict]:
        """
        Extract sensory information from text.
        Returns a dict of stimuli, e.g. {"pain": 0.5, "pleasure": 0.8, "food": 1.0}
        """
        stimuli = {}
        text_lower = text.lower()
        
        # 1. Pain Detection
        # Bare "打" is avoided on purpose: it also matches 打开 / 打算 / 打游戏
        pain_triggers = ["hit", "slap", "kick", "punch", "hurt",
                         "打你", "打我", "打她", "打人", "揍", "踢", "踹", "掐", "扇你", "扇我",
                         "咬你", "咬我"]
        # Not "好疼"/"好痛": that is the speaker's own pain (手指好疼), not something done to her
        if any(t in text_lower for t in pain_triggers):
            stimuli["pain"] = 0.5
            
        # 2. Pleasure Detection (Petting)
        pet_triggers = ["pet", "pat", "stroke", "rub", "hug", "摸摸", "抱抱", "贴贴", "摸头"]
        if any(t in text_lower for t in pet_triggers):
            stimuli["pleasure"] = 0.5
            
        # 3. Food Detection (Feeding)
        food_triggers = ["feed", "eat", "cookie", "fish", "treat", "喂", "吃", "鱼干", "罐头"]
        if any(t in text_lower for t in food_triggers):
            stimuli["food"] = 1.0
            
        if not stimuli:
            return None
            
        return stimuli
