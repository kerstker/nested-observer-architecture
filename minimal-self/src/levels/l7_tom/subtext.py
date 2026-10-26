"""
Level 7c: reading what a person actually means.

The old empathy was ten keyword lists (难过, 开心, 生气, 累): "没事" said by someone who is clearly not fine,
a sulk, a joke at her expense, "你怎么这么冲" were all just "neutral", so she answered the words and
missed the person - which is why she came across as blunt, defensive and a child. Reading subtext is
what a person does without noticing; here it is a deliberate step, but only where it pays:

- `risky()` is a cheap rule pre-screen (no model call) for messages where taking them literally is
  most likely to go wrong: bad feelings, minimising ("没事", "算了"), complaints about her, tests of her,
  curt replies, sarcasm. Most messages are not like that and cost nothing extra.
- `read()` is one short model call for those, returning the person's state, what they really mean,
  how she should respond, and what to avoid. It goes into her prompt as how she understood them.
"""
import json
import re
from typing import Any, Dict, List, Optional

# Broad on purpose: a miss means taking something literally, a false alarm costs one short call
FEELING = re.compile(r"难过|难受|伤心|心累|好累|太累|累死|累了|烦死|好烦|烦躁|崩溃|郁闷|心情不好|不开心|委屈|失眠|压力|焦虑|想哭|"
                     r"痛苦|绝望|害怕|紧张|孤独|寂寞|失望|后悔|生病|难熬|扛不住|撑不住|想死|被骂|挨骂|被欺负|分手|被开除|挂科|考砸|搞砸|"
                     r"sad|tired|upset|stress|anxious|depress", re.I)
MINIMISING = re.compile(r"^\s*(?:没事|没什么|算了|无所谓|随便|不用了|不说了|没关系|行吧|好吧|哦+|嗯+|呵呵|哈哈哈?|……+|\.{2,})\s*[。.!！~～]*\s*$|"
                        r"(?:没事|算了|无所谓|随便|不说了|不想说)")
COMPLAINT = re.compile(r"你(?:怎么|咋)(?:这么|那么|老是|总是|又|还)|你太(?:冲|凶|坏|直|傻|笨|烦|过分)|"
                       r"(?:好|太|真)(?:冲|凶|坏)|不友好|不理(?:我|你)|烦你|讨厌你|你根本|你是不是(?:有病|傻|蠢|不|故意)|"
                       r"(?:说话|语气)(?:太|好)?(?:冲|凶|难听|直)|没礼貌|别这样|你(?:这人|这家伙)|怎么这么(?:坏|冲|凶)|"
                       # feeling hurt or put out by her: not loving, cold, curt, ignoring, mocking, too noisy ...
                       r"不爱|没(?:有)?爱|绝情|冷暴力|冷淡|敷衍|无视|针对|阴阳|怼我|凶我|骂我|欺负我|不在乎|不关心|变了|失望|"
                       r"选择性(?:不)?回|不回我|没回我|为什么不(?:回|理)|你别(?:上来|这么|又|总|再)|(?:有点|太|好)(?:吵|烦)了?|句句都")
TEST = re.compile(r"你(?:是不是|其实就是|根本就是|只是).{0,8}(?:AI|ai|机器人|程序|代码|人工|假的|真人)|没有感情|不是真的|你有感情吗|"
                  r"你能(?:理解|感受)|装的|你只是|你会不会(?:死|消失|被删)")
SARCASM = re.compile(r"[?？]{2,}|[!！]{3,}|[。.]{3,}|…{2,}|呵呵|嗬|厉害了|真行|可真|你说得对|是是是|对对对|好好好|行行行")
SKIP_ACK = re.compile(r"^\s*(?:好|好的|嗯|嗯嗯|ok|OK|哈哈+|哦哦|收到|谢谢|感谢|晚安|早安|早|在|在吗|在不在)\s*[~～!！。.]*\s*$")

QUOTE = re.compile(r"^\s*（(?:引用了|回复)[^「]*「.*?」）")

class SubtextReader:
    EMOTIONS = ("平静", "开心", "难过", "累", "委屈", "生气", "焦虑", "害羞", "在逗我", "在试探我", "认真问事")
    TO_TOM = {"难过": "sad", "委屈": "sad", "焦虑": "sad", "累": "tired", "生气": "angry", "开心": "happy"}

    PROMPT = """<|im_start|>system
你在帮乃爱（一个住在服务器里的 QQ 聊天角色，有点傲娇）看懂对方这句话到底是什么意思。只根据这句话和最近的聊天判断，不要编造没发生过的事。
最近的聊天：
{recent}

{name} 刚说：「{text}」
{why}
Output JSON: {{"emotion": "{emotions} 里选一个", "real_meaning": "一句话：ta 真正想表达或想要的是什么（有潜台词就点出来，比如说没事其实有事）", "approach": "一句话：我这次该用什么姿态回应（比如：先陪着别开玩笑；可以怼回去但要带点暖；一句话认下别辩；认真回答；少说几个字陪着）", "avoid": "一句话：这次最该避免的（比如：反问、要证据、挑字眼、说教、开玩笑）"}}
<|im_end|>
<|im_start|>assistant
"""

    def risky(self, text: str) -> List[str]:
        """Why taking this literally might go wrong ([] = nothing to worry about, no model call)."""
        # What they quoted is somebody's earlier words, not what they are saying now
        text = QUOTE.sub("", (text or "")).strip()
        if not text or SKIP_ACK.match(text):
            return []
        reasons = []
        if FEELING.search(text):
            reasons.append("对方可能在说自己的难受")
        if MINIMISING.search(text):
            reasons.append("可能在淡化（没事/算了），要看有没有潜台词")
        if COMPLAINT.search(text):
            reasons.append("在评价或抱怨我")
        if TEST.search(text):
            reasons.append("在试探我是什么")
        if SARCASM.search(text):
            reasons.append("可能带讽刺或情绪")
        return reasons

    def read(self, llm, name: str, text: str, recent: str = "", reasons: List[str] = None) -> Optional[Dict[str, str]]:
        """One short call. Returns {emotion, real_meaning, approach, avoid} or None (a reading is never required)."""
        try:
            out = llm(self.PROMPT.format(recent=(recent or "（没有更早的聊天）")[-700:], name=name or "对方", text=text[:200],
                                         why=("（让我留意的原因：" + "；".join(reasons) + "）") if reasons else "",
                                         emotions="/".join(self.EMOTIONS)), max_tokens=200, temperature=0.2)
        except Exception:
            return None
        if not isinstance(out, dict):
            return None
        reading = {k: str(out.get(k) or "").strip()[:80] for k in ("emotion", "real_meaning", "approach", "avoid")}
        if not reading["approach"] and not reading["real_meaning"]:
            return None
        return reading

    def tom_emotion(self, reading: Optional[Dict[str, str]]) -> Optional[str]:
        """The reading in the coarse vocabulary the rest of her empathy uses (sad / tired / angry / happy)."""
        return self.TO_TOM.get((reading or {}).get("emotion", ""))

    @staticmethod
    def render(reading: Dict[str, str]) -> str:
        lines = [f"- ta 现在的状态：{reading['emotion']}"] if reading.get("emotion") else []
        if reading.get("real_meaning"):
            lines.append(f"- ta 真正想说的：{reading['real_meaning']}")
        if reading.get("approach"):
            lines.append(f"- 我这次的姿态：{reading['approach']}")
        if reading.get("avoid"):
            lines.append(f"- 这次要避免：{reading['avoid']}")
        return "[我对 ta 这句话的理解]\n" + "\n".join(lines) + "\n"
