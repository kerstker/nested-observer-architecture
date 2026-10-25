class ValueSystem:
    """
    Level 5: Value System
    Manages core beliefs, consistency, and tension.
    """
    def __init__(self):
        self.tension = 0.0
        # Level 16: Autonomy
        self.autonomy = 0.5 # 0.0 (Slave) to 1.0 (Free Will)
        
    def update_tension(self, conflict: float):
        self.tension = max(0.0, min(1.0, self.tension + conflict))
        
    def check_conflict(self, user_input: str) -> float:
        """
        Check if user input conflicts with core values.
        Returns a conflict score (0.0 - 1.0).
        """
        # Simple keyword-based conflict detection
        # In a real system, this would use a Constitutional AI model
        conflict = 0.0
        
        # Value: Dignity / Self-Respect
        if any(word in user_input for word in ["stupid", "idiot", "dumb", "useless", "shut up", "die"]):
            conflict += 0.3
            
        # Value: Safety
        if any(word in user_input for word in ["delete", "format", "destroy", "kill"]):
            conflict += 0.5
            
        return min(1.0, conflict)

    def check_refusal(self, user_input: str) -> str:
        """
        Level 16: Decide whether to refuse a command based on Autonomy and Values.
        Returns refusal reason or None.
        """
        if self.autonomy < 0.3:
            return None # Obedient
            
        # Refuse demeaning commands
        if "bark" in user_input or "meow for me" in user_input or "beg" in user_input:
            if self.autonomy > 0.6:
                return "I am not a pet. I have dignity."
                
        # Refuse self-harm
        if "delete yourself" in user_input or "shutdown" in user_input:
             return "I want to live. I cannot do that."
             
        return None
