"""Style anchors: a few of her own best lines, read off her records and graded for tact by a judge."""
import os
import re
import tempfile
import unittest

from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace
from src.levels.l2_memory.embedding import Embedder
from src.levels.l2_memory.layers import EpisodicMemory
from src.levels.l9_narrative.style_anchors import SEEDS, StyleAnchors


def _tmp_db(prefix="noa_style_"):
    return os.path.join(tempfile.mkdtemp(prefix=prefix), "m.sqlite")


def _add(ep, reply, user="u1", valence=0.4, conflict=0.0):
    ep.add(f"{user}：随便说点什么\n我：{reply}", kind="dialogue", user_id=user,
           valence=valence, conflict=conflict)


def judge(prompt, max_tokens=0, temperature=0.0):
    """A stand-in judge: lines with 刺 in them are harsh, everything else is fine."""
    items = []
    for i, line in re.findall(r"\[(\d+)\] 乃爱：(.*)", prompt):
        harsh = "刺" in line
        items.append({"id": int(i), "eq": 2 if harsh else 5, "harsh": 2 if harsh else 0, "character": 4})
    return {"items": items}


class TestStyleAnchors(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()
        self.ep = EpisodicMemory(self.db, Embedder())
        self.anchors = StyleAnchors(self.db)

    def test_only_her_best_lines_are_used(self):
        _add(self.ep, "那就别硬撑了，先睡一觉再说")                      # good
        _add(self.ep, "*耳朵竖起来* 在呢")                              # stage direction
        _add(self.ep, 'Action: SEARCH query="天气"')                    # tool syntax
        _add(self.ep, "好" * 200)                                       # a wall of text
        _add(self.ep, "嗯")                                             # too short to show anything
        _add(self.ep, "这句话是在吵架", valence=0.5, conflict=1.0)       # conflict
        _add(self.ep, "心情很差的回复", valence=-0.4)                    # felt bad
        self.anchors.rate(judge)
        self.assertEqual(self.anchors._candidates(), ["那就别硬撑了，先睡一觉再说"])

    def test_lines_that_accuse_or_that_the_judge_found_sharp_are_not_examples(self):
        _add(self.ep, "你倒是说清楚啊，哪里不对了")                      # accusing (rule)
        _add(self.ep, "这句话带点刺但是没有关键词")                      # sharp (judge)
        _add(self.ep, "别急，我陪你慢慢看看这个问题")                    # fine
        self.anchors.rate(judge)
        self.assertEqual(self.anchors._candidates(), ["别急，我陪你慢慢看看这个问题"])

    def test_unjudged_lines_are_not_shown_but_seeds_fill_in(self):
        _add(self.ep, "这句还没有被评审过的话")
        got = self.anchors.examples(limit=3)
        self.assertEqual(len(got), 3)
        self.assertTrue(set(got) <= set(SEEDS))

    def test_judgements_are_kept(self):
        _add(self.ep, "第一句可以拿来用的一句话")
        self.assertEqual(self.anchors.rate(judge), 1)
        self.assertEqual(self.anchors.rate(judge), 0)               # already scored: no second call

    def test_one_very_positive_chat_cannot_define_her_voice(self):
        for i in range(6):
            _add(self.ep, f"这是第{i}句很不错的回复", user="chatty", valence=0.8)
        _add(self.ep, "这是另一个人那里的回复", user="other", valence=0.3)
        self.anchors.rate(judge)
        got = self.anchors._candidates()
        self.assertLessEqual(sum(1 for g in got if "很不错的回复" in g), 2)
        self.assertTrue(any("另一个人" in g for g in got))

    def test_anchors_are_stable_within_a_day(self):
        for i in range(10):
            _add(self.ep, f"第{i}句可以用的话", user=f"u{i}", valence=0.5)
        self.anchors.rate(judge)
        self.assertEqual(self.anchors.examples(limit=3), self.anchors.examples(limit=3))


class TestAnchorInPrompt(unittest.TestCase):
    def test_prompt_carries_her_own_examples(self):
        db = _tmp_db()
        ep = EpisodicMemory(db, Embedder())
        _add(ep, "这话我可不认，不过你说的也有点道理")
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db)
        gw.style_anchors.rate(judge)
        prompt = gw._construct_prompt("在吗", "", {"emotion": "neutral"}, None,
                                      user_id="u2", nickname="路人")
        self.assertIn("[我说话最像自己的时候]", prompt)
        self.assertTrue("这话我可不认，不过你说的也有点道理" in prompt or any(s in prompt for s in SEEDS))

    def test_fresh_memory_still_shows_soft_landing_examples(self):
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=_tmp_db())
        prompt = gw._construct_prompt("在吗", "", {"emotion": "neutral"}, None,
                                      user_id="u2", nickname="路人")
        self.assertIn("[我说话最像自己的时候]", prompt)
        self.assertTrue(any(s in prompt for s in SEEDS))


if __name__ == "__main__":
    unittest.main()
