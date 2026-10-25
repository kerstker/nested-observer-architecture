from typing import Dict, List

class TheoryOfMind:
    """
    Level 7: Theory of Mind (ToM) / Empathy
    Estimates the user's emotional state and intent.
    """
    def __init__(self):
        self.user_states = {} # Dict[user_id, Dict]
        
    def _get_state(self, user_id: str):
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                "last_emotion": "neutral",
                "relationship_level": 0
            }
        return self.user_states[user_id]
        
    def analyze_user(self, user_input: str, user_id: str = "default") -> Dict[str, str]:
        """
        Analyze user input to guess emotion.
        """
        user_input_lower = user_input.lower()
        
        guessed_emotion = "neutral"
        
        # Simple keyword heuristics
        if any(w in user_input_lower for w in ["sad", "cry", "hurt", "bad day", "难过", "想哭", "不开心", "好疼", "好痛"]):
            guessed_emotion = "sad"
        elif any(w in user_input_lower for w in ["happy", "great", "love", "yay", "开心", "棒", "喜欢"]):
            guessed_emotion = "happy"
        elif any(w in user_input_lower for w in ["angry", "hate", "mad", "stupid", "生气", "讨厌", "滚"]):
            guessed_emotion = "angry"
        elif any(w in user_input_lower for w in ["tired", "sleepy", "exhausted", "累", "困"]):
            guessed_emotion = "tired"
            
        state = self._get_state(user_id)
        state["last_emotion"] = guessed_emotion
        
        return {
            "emotion": guessed_emotion,
            "intent": "chat" # Placeholder for intent classification
        }
    
    def get_empathy_instruction(self, user_id: str = "default") -> str:
        """Return instruction for the Persona based on user state"""
        state = self._get_state(user_id)
        emo = state["last_emotion"]
        if emo == "sad":
            return "User seems sad. Be comforting and gentle."
        elif emo == "happy":
            return "User seems happy. Share their joy!"
        elif emo == "angry":
            return "User seems angry. Be apologetic or diffuse the tension."
        elif emo == "tired":
            return "User seems tired. Be soothing."
        return ""
