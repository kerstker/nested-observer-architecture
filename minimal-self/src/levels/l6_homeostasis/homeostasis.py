from typing import Dict
from src.core import clock

class Homeostasis:
    """
    Level 6: Homeostasis
    Regulates internal biological needs (Hunger, Social, Hygiene) to maintain equilibrium.
    """
    def __init__(self):
        # 0.0 = Satisfied, 1.0 = Desperate need
        self.hunger = 0.0
        self.social_need = 0.0
        self.hygiene_need = 0.0
        self.curiosity_need = 0.0 # Level 19: Curiosity
        self.last_update = clock.now()
        
    def tick(self, interaction_count: int = 0) -> Dict[str, float]:
        """
        Update needs based on time passed.
        interaction_count: Number of interactions in this tick (to satisfy social)
        """
        now = clock.now()
        delta_hours = (now - self.last_update) / 3600.0
        self.last_update = now
        
        # Decay rates (per hour)
        # Hunger increases 10% per hour (10 hours to full hunger)
        self.hunger = min(1.0, self.hunger + (0.10 * delta_hours))
        
        # Social need increases 5% per hour if no interaction
        if interaction_count > 0:
            self.social_need = max(0.0, self.social_need - 0.2) # Interaction satisfies social
        else:
            self.social_need = min(1.0, self.social_need + (0.05 * delta_hours))
            
        # Hygiene need increases 2% per hour
        self.hygiene_need = min(1.0, self.hygiene_need + (0.02 * delta_hours))

        # Curiosity increases 20% per hour (Gets bored quickly)
        # Satisfied by learning new things (Search/Memorize)
        self.curiosity_need = min(1.0, self.curiosity_need + (0.20 * delta_hours))
        
        return {
            "hunger": self.hunger,
            "social": self.social_need,
            "hygiene": self.hygiene_need,
            "curiosity": self.curiosity_need
        }

    def alleviate(self, need_type: str, amount: float):
        """Action to alleviate a need (e.g. eating)"""
        if need_type == "hunger":
            self.hunger = max(0.0, self.hunger - amount)
        elif need_type == "social":
            self.social_need = max(0.0, self.social_need - amount)
        elif need_type == "hygiene":
            self.hygiene_need = max(0.0, self.hygiene_need - amount)
        elif need_type == "curiosity":
            self.curiosity_need = max(0.0, self.curiosity_need - amount)

    def get_status_desc(self) -> str:
        """Return a natural language description of needs for the LLM prompt"""
        alerts = []
        if self.hunger > 0.8:
            alerts.append("Starving")
        elif self.hunger > 0.5:
            alerts.append("Hungry")
            
        if self.social_need > 0.8:
            alerts.append("Lonely")
            
        if self.hygiene_need > 0.8:
            alerts.append("Feels dirty")

        if self.curiosity_need > 0.7:
            alerts.append("Bored/Curious")
            
        if not alerts:
            return "Satisfied"
        
        return ", ".join(alerts)
