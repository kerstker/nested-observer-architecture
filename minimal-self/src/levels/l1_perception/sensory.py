from typing import Dict, Any, Optional
import re

class SensoryCortex:
    """
    Level 1: Perception System
    Responsible for parsing raw input into sensory signals.

    A sensation is something done *to her*: being fed, petted, hurt, washed. Words alone are not that -
    "你喜欢吃什么" is a question about food, "喂，在吗" is a greeting, "摸摸欧派" is about someone else,
    "我不会打你" is a promise. The old keyword list (any message containing 吃 / 喂 / 摸摸 ...) turned
    one message in twenty into a meal for her (hunger gone, +20 energy) or a caress, and ordinary
    English words ("great", "weather" contain "eat") into food. So each sensation needs the act *and*
    her as its object, and a negation just before it cancels it.
    """
    # act, then her (or her head / ears / tail ...) as the object
    PAIN = re.compile(r"(?:打|揍|踢|踹|掐|扇|咬|捶|锤|抽)(?:死|了|着)?(?:你|乃爱|她)|"
                      r"\b(?:hit|slap|kick|punch|hurt|bite)s? (?:you|noa)\b", re.I)
    PETTING = re.compile(r"(?:摸摸|摸|撸|揉揉|揉|抱抱|抱|贴贴|亲亲|亲|蹭蹭)(?:你|乃爱|头|耳朵|尾巴|脸|下巴)|"
                         r"^\s*(?:抱抱|贴贴|亲亲|摸摸|rua)\s*[~～!！。.…]*\s*$|"
                         r"\b(?:pet|pat|hug|stroke|cuddle|kiss)s? (?:you|noa)\b|\bheadpats?\b", re.I)
    FOOD = re.compile(r"(?:喂|投喂)(?:你|乃爱)|(?:给|请|送|赏)(?:你|乃爱)(?:吃|喝)|"
                      r"(?:请你|给你|送你|喂你|带给你).{0,8}(?:吃|小鱼干|鱼干|罐头|猫粮|零食|小鱼|蛋糕|牛奶)|"
                      r"(?:小鱼干|鱼干|罐头|猫粮|零食).{0,6}(?:给你|喂你|送你|请你)|"
                      r"\bfeed(?:s|ing)? (?:you|noa)\b|\b(?:here'?s|have) (?:a |some )?(?:treat|fish|cookie|snack)s?\b", re.I)
    CLEANING = re.compile(r"(?:给你|帮你|替你|带你|给乃爱|帮乃爱).{0,4}(?:洗澡|洗头|洗干净|洗脸|洗|梳毛|梳头|刷毛|擦干净|吹干)", re.I)
    NEGATION = re.compile(r"(?:不|别|没|勿|未|莫|才不|不会|不敢|舍不得|没有|don'?t|never|not)\s*(?:会|想|要|能|敢|去|准|许)?\s*$", re.I)

    def __init__(self):
        self.thermal_keywords = {"冷": "cold", "热": "hot", "暖": "warm"}

    def _happens(self, pattern: "re.Pattern", text: str) -> bool:
        """The act is meant (and not denied: 不会打你 / 别摸你 / don't hit you)."""
        return any(not self.NEGATION.search(text[max(0, m.start() - 6):m.start()]) for m in pattern.finditer(text))

    def process_text(self, text: str) -> Optional[dict]:
        """
        Extract sensory information from text.
        Returns a dict of stimuli, e.g. {"pain": 0.5, "pleasure": 0.8, "food": 1.0, "clean": 1.0}
        """
        stimuli = {}
        text = text or ""
        # (Not "好疼"/"好痛": that is the speaker's own pain, not something done to her.)
        if self._happens(self.PAIN, text):
            stimuli["pain"] = 0.5
        if self._happens(self.PETTING, text):
            stimuli["pleasure"] = 0.5
        if self._happens(self.FOOD, text):
            stimuli["food"] = 1.0
        if self._happens(self.CLEANING, text):
            stimuli["clean"] = 1.0
        return stimuli or None
