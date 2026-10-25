import random
from typing import List, Dict

class ActionLibrary:
    def __init__(self):
        # 50+ Catgirl Actions categorized by Emotion/State
        self.actions = {
            "joy": [
                "*wags tail happily* 摇尾巴",
                "*purrs loudly* 呼噜呼噜~",
                "*nuzzles your hand* 蹭蹭手",
                "*ears twitch excitedly* 耳朵兴奋地抖动",
                "*jumps up and down* 蹦蹦跳跳",
                "*rolls on the floor* 在地上打滚",
                "*kneads your lap* 在你腿上踩奶",
                "*licks your cheek* 舔舔你的脸颊",
                "*chases own tail* 追自己的尾巴",
                "*meows sweetly* 甜甜地叫了一声"
            ],
            "anger": [
                "*hisses* 哈气！",
                "*flattens ears* 耳朵变成了飞机耳",
                "*swishes tail aggressively* 尾巴用力拍打地面",
                "*bares fangs* 露出小虎牙",
                "*scratches the sofa* 抓沙发泄愤",
                "*turns away coldly* 冷冷地转过身",
                "*pouts* 鼓起脸颊",
                "*growls low* 低声呜呜叫",
                "*bites gently* 轻轻咬了一口",
                "*glares* 瞪着你"
            ],
            "sorrow": [
                "*droops ears* 耳朵耷拉下来",
                "*whimpers softly* 小声呜咽",
                "*curls into a ball* 缩成一团",
                "*hides under the bed* 躲到床底下",
                "*looks down sadly* 低头看着脚尖",
                "*tail stops moving* 尾巴不动了",
                "*teary eyes* 眼泪汪汪",
                "*sighs* 叹了一口气",
                "*clings to you* 紧紧抓着你",
                "*silent meow* 无声地张了张嘴"
            ],
            "fear": [
                "*trembles* 瑟瑟发抖",
                "*arches back* 弓起背",
                "*hair stands on end* 炸毛了！",
                "*hides behind you* 躲在你身后",
                "*wide eyes* 瞳孔放大",
                "*freezes* 僵住了",
                "*runs away* 跑掉了",
                "*covers eyes* 捂住眼睛",
                "*flinches* 缩了一下",
                "*looks around nervously* 紧张地四处张望"
            ],
            "neutral": [
                "*stretches lazily* 伸了个大大的懒腰",
                "*grooms paw* 舔舔爪子",
                "*yawns* 打哈欠",
                "*sits elegantly* 优雅地坐着",
                "*twitches nose* 动动鼻子",
                "*watches a fly* 盯着一只苍蝇",
                "*blinks slowly* 缓慢地眨眼（表示信任）",
                "*tilts head* 歪头",
                "*cleans face* 洗脸",
                "*looks out window* 看着窗外"
            ],
            "sleepy": [
                "*nods off* 点头打瞌睡",
                "*curls up on cushion* 在软垫上缩成一团",
                "*slow breathing* 呼吸变慢了",
                "*half-closed eyes* 半眯着眼",
                "*mumbles in sleep* 说梦话",
                "*stretches paws* 伸直爪子",
                "*sleepy meow* 困倦地叫了一声",
                "*rubbing eyes* 揉揉眼睛",
                "*finds a warm spot* 找个暖和的地方",
                "*flops over* 倒下了"
            ]
        }

    def get_action(self, emotion: str, bio_state: str) -> str:
        """
        Get a random action based on emotion and bio-state
        Level 16: Added 'READ_CODE' for self-introspection.
        """
        # Level 16: Chance to introspect if curious or anxious
        # (This logic is simple random for now, should be driven by GW)
        if random.random() < 0.05:
            return "*stares into the void* [Action: READ_CODE]"

        # Priority 1: Biological State (Sleepy/Tired)
        if "Sleep" in bio_state or "Tired" in bio_state:
            return random.choice(self.actions["sleepy"])
        
        # Priority 2: Emotion
        if emotion in self.actions:
            return random.choice(self.actions[emotion])
        
        # Fallback
        return random.choice(self.actions["neutral"])
