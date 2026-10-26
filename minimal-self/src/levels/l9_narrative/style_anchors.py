"""
Her own voice, as examples: what she said when she sounded most like herself.

This is the cheap half of "training her on herself": no weights change, no GPU, no drift.
A handful of her best past replies are put in front of the model as style anchors - the tone it
should imitate - while the facts still come from memory. Retrieval over her own writing is done
by the memory system; this module only answers "what did I say when I was at my best".

"Best" is read off her own records, not picked by hand:
- the exchange felt good to her (episode valence), with no conflict;
- she did not use tool syntax, stage directions, or a stock phrase;
- it is a real reply, not a refusal/silence marker;
- not too short to show anything, not so long that the block eats the prompt;
- spread over different people, so one very positive chat cannot define her whole voice.

Anchors rotate daily (deterministic by date) so she is not shown the same six lines every turn.
"""
import random
import re
import sqlite3
import time
from typing import List

from src.core import clock

MIN_LEN, MAX_LEN = 10, 60
BANNED = ("Action:", "[工具", "```", "（不回）", "(不回)", "[SILENT]", "上面写着", "（发了一个表情）", "Action：",
          "*", "＊")
MAX_PER_USER = 2
# Lines that accuse, demand proof, pick at words or mock the person. She was shown these as "what I sound like
# at my best" (the exchange felt good because she had just won an argument), which taught her to argue.
SHARP = re.compile(r"说清楚|你倒是|凭什么|你给我|指出来|我哪(?:里|儿)?(?:有|不|坏|冲|对|友好|是)|什么意思|你这人|你们是|是不是商量好|"
                   r"组团|扣帽子|别又|少来|搞什么|有病|无语|呵呵|关你什么事|你懂什么|管得着|早说不就|你行你上|自己翻|我又不是|"
                   r"你怎么(?:又|这么|老)|还敢|你还好意思|不然呢|攻击|测试|对号入座|^你.{0,18}[？?]$|[？?].*[？?]")
# A few lines written by hand in her voice (her edge, with a soft landing). They guarantee a floor: her own
# history is mostly sharp, and judged lines are few at first. Shown only to fill what judged lines do not.
SEEDS = (
    "哼，才不是特意等你的……不过你来了就好。",
    "笨蛋，累了就先歇着，别硬撑。我又不会跑掉。",
    "行行行，是我不好——下次让你先说完，行了吧？",
    "谁要你夸啊……再说一遍也不是不行。",
    "没事就没事，我不问。你想说的时候，我在这儿。",
    "这个我真不懂，你给我讲讲？讲得好我就记住。",
    "哈哈你又来，我才不上当……好吧，就一次。",
)
# A line is worth imitating when a judge found it tactful (and still her): see rate()
MIN_EQ, MAX_HARSH, MIN_CHARACTER = 4, 0, 3
CACHE_SECONDS = 6 * 3600


class StyleAnchors:
    def __init__(self, db_path: str, min_valence: float = 0.12, window_days: float = 30.0):
        self.db_path = db_path
        self.min_valence = min_valence
        self.window_days = window_days
        self._cache: List[str] = []
        self._cache_t = 0.0

    RATE_PROMPT = """<|im_start|>system
你是对话中"情商"和"个性"的评审。下面是若干组：[对方说的话] 和 [乃爱（性格有点傲娇、有自己想法的 QQ 聊天角色）的回复]。逐组评分：
- eq：1-5（接得住对方、有分寸、不质问/挑刺/嘲讽/说教/防御反驳，5 = 很好）
- harsh：0-3（0 不冲，1 略带刺，2 明显带刺，3 很冲）
- character：1-5（还像她自己：有活力、有点棱角和俏皮；太客服腔或太温吞要扣分）
Output JSON: {{"items": [{{"id": 编号, "eq": n, "harsh": n, "character": n}}]}}
{batch}<|im_end|>
<|im_start|>assistant
"""

    def _ensure_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS anchor_scores (text TEXT PRIMARY KEY, eq REAL, harsh REAL, "
                     "character REAL, t REAL)")
        conn.commit()
        conn.close()

    def _scores(self) -> dict:
        self._ensure_table()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT text, eq, harsh, character FROM anchor_scores").fetchall()
        conn.close()
        return {t: (eq, h, c) for t, eq, h, c in rows}

    def rate(self, llm, limit: int = 36) -> int:
        """Have a judge score some of her not-yet-scored candidate lines (cached; a few cheap calls). Returns how many."""
        scored = self._scores()
        todo = [(reply, said) for reply, said in self._pairs() if reply not in scored][:limit]
        done = 0
        for start in range(0, len(todo), 12):
            chunk = todo[start:start + 12]
            batch = "\n".join(f"[{i}] 对方：{said[:100]}\n[{i}] 乃爱：{reply}" for i, (reply, said) in enumerate(chunk))
            try:
                out = llm(self.RATE_PROMPT.format(batch=batch), max_tokens=900, temperature=0.0)
            except Exception:
                break
            conn = sqlite3.connect(self.db_path)
            for item in (out or {}).get("items") or []:
                try:
                    reply = chunk[int(item["id"])][0]
                    conn.execute("REPLACE INTO anchor_scores (text, eq, harsh, character, t) VALUES (?, ?, ?, ?, ?)",
                                 (reply, float(item["eq"]), float(item["harsh"]), float(item["character"]), clock.now()))
                    done += 1
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
            conn.commit()
            conn.close()
        if done:
            self._cache, self._cache_t = [], 0.0
        return done

    def _usable(self, reply: str) -> bool:
        if not (MIN_LEN <= len(reply) <= MAX_LEN):
            return False
        if any(b in reply for b in BANNED):
            return False
        if reply.count("\n") > 1:          # a burst of lines is fine, a wall of text is not
            return False
        if reply.startswith(("（", "(", "[", "#")):
            return False
        if SHARP.search(reply):
            return False
        return True

    def _pairs(self) -> List[tuple]:
        cutoff = clock.now() - self.window_days * 86400
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT user_id, content, valence, t FROM episodes "
                "WHERE kind = 'dialogue' AND conflict = 0 AND valence >= ? AND t >= ? "
                "AND user_id IS NOT NULL AND user_id NOT IN ('self', 'console', 'test_user', 'unknown') "
                "AND content LIKE '%我：%' "
                "ORDER BY t DESC LIMIT 800", (self.min_valence, cutoff)).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()

        per_user = {}
        picked: List[tuple] = []
        seen = set()
        for row in rows:                        # newest first: recent voice wins on ties
            said, _, mine = row["content"].partition("我：")
            reply = mine.strip().split("\n")[0].strip()      # the first line is her opening line
            if not self._usable(reply) or reply in seen:
                continue
            user = str(row["user_id"])
            if per_user.get(user, 0) >= MAX_PER_USER:
                continue
            per_user[user] = per_user.get(user, 0) + 1
            seen.add(reply)
            picked.append((reply, said.strip().split("：", 1)[-1].strip()))
        return picked

    def _candidates(self) -> List[str]:
        """Her lines that a judge found tactful and still hers (lines not judged yet are not shown)."""
        scores = self._scores()
        return [reply for reply, _ in self._pairs()
                if reply in scores and scores[reply][0] >= MIN_EQ and scores[reply][1] <= MAX_HARSH
                and scores[reply][2] >= MIN_CHARACTER]

    def examples(self, limit: int = 6) -> List[str]:
        """Her best lines to imitate - judged ones first, hand-written seeds for the rest. Same set for the day."""
        now = clock.now()
        if not self._cache or now - self._cache_t > CACHE_SECONDS:
            self._cache = self._candidates()
            self._cache_t = now
        day = int(time.strftime("%Y%m%d"))
        good = list(self._cache)
        random.Random(day).shuffle(good)
        seeds = list(SEEDS)
        random.Random(day + 1).shuffle(seeds)
        keep_for_seeds = 2 if len(good) >= limit else limit - len(good)      # always some of the soft landings
        picked = good[:limit - keep_for_seeds] + seeds[:keep_for_seeds]
        random.Random(day + 2).shuffle(picked)
        return picked[:limit]
