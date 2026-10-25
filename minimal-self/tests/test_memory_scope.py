"""Memory scope: what she may remember in front of whom.

The rule this file locks in:
  private - only the person it happened with (plus Noa herself) can recall it
  group   - anyone in that same group can recall it
  public  - distilled experience (thoughts, facts she wrote down) is shareable
A private conversation must never become recallable by a third person.
"""
import os
import tempfile
import unittest

from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM
from src.levels.l2_memory.embedding import Embedder
from src.levels.l2_memory.layers import (EpisodicMemory, SemanticMemory,
                                         VISIBILITY_PRIVATE, VISIBILITY_PUBLIC)
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace


def _db():
    return os.path.join(tempfile.mkdtemp(prefix="noa_scope_"), "m.sqlite")


class TestEpisodeScope(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.ep = EpisodicMemory(self.db, Embedder())

    def test_private_dialogue_is_not_recallable_by_anyone_else(self):
        text = "kerst：我跟你说个秘密，我搬家了"
        self.ep.add(text, kind="dialogue", user_id="kerst", group_id=None)
        self.assertEqual(self.ep.recall(text, user_id="stranger", k=3), [])
        self.assertEqual(self.ep.recall(text, user_id="stranger", k=3, group_id="G1"), [])
        self.assertTrue(self.ep.recall(text, user_id="kerst", k=3))

    def test_group_dialogue_is_recallable_inside_that_group_only(self):
        text = "群里聊：50 块配电脑"
        self.ep.add(text, kind="dialogue", user_id="alice", group_id="G1")
        self.assertTrue(self.ep.recall(text, user_id="bob", k=3, group_id="G1"))
        self.assertEqual(self.ep.recall(text, user_id="bob", k=3, group_id="G2"), [])
        self.assertEqual(self.ep.recall(text, user_id="bob", k=3), [])

    def test_inner_life_is_public(self):
        text = "我想他们是不是只是把我当成会说话的搜索框"
        self.ep.add(text, kind="thought", user_id="self")
        self.assertTrue(self.ep.recall(text, user_id="stranger", k=3))

    def test_visibility_is_derived_from_where_it_happened(self):
        self.ep.add("私聊一句", kind="dialogue", user_id="u", group_id=None)
        self.ep.add("群聊一句", kind="dialogue", user_id="u", group_id="G")
        self.ep.add("一个念头", kind="thought", user_id="self")
        conn_rows = self.ep.recall("私聊一句", user_id="u", k=5)
        self.assertEqual(conn_rows[0]["visibility"], VISIBILITY_PRIVATE)
        pub = self.ep.recall("一个念头", user_id="someone_else", k=5)
        self.assertEqual(pub[0]["visibility"], VISIBILITY_PUBLIC)


class TestDistilledFacts(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.sem = SemanticMemory(self.db, Embedder())

    def test_public_facts_are_shareable_private_ones_are_not(self):
        self.sem.add("kerst", "kerst 最近在折腾显卡和换号", [1], visibility=VISIBILITY_PUBLIC)
        self.sem.add("sec", "私下说他其实很怕黑", [2], visibility=VISIBILITY_PRIVATE)
        facts = [f["fact"] for f in self.sem.public_facts(limit=10)]
        self.assertIn("kerst 最近在折腾显卡和换号", facts)
        self.assertNotIn("私下说他其实很怕黑", facts)
        self.assertEqual([f["fact"] for f in self.sem.public_facts(exclude_user="kerst")], [])


class TestScopeInPrompt(unittest.TestCase):
    def _gw(self):
        db = _db()
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("在呢"), db_path=db)
        return gw

    def test_shared_facts_reach_the_prompt_but_evolve_does_not_by_default(self):
        gw = self._gw()
        gw.semantic.add("kerst", "kerst 最近在折腾显卡和换号", [1], visibility=VISIBILITY_PUBLIC)
        prompt = gw._construct_prompt("你好", "", {"emotion": "neutral"}, None,
                                      user_id="stranger", nickname="路人")
        self.assertIn("kerst 最近在折腾显卡和换号", prompt)
        self.assertNotIn("EVOLVE", prompt)

    def test_creator_private_chat_sees_the_real_capability(self):
        gw = self._gw()
        gw.evolve_enabled = True
        gw.evolve_agent = object()
        gw.evolve_creator = "creator"
        prompt = gw._construct_prompt("你能改自己吗", "", {"emotion": "neutral"}, None,
                                      user_id="creator", nickname="菜菜", evolve_allowed=True)
        self.assertIn("EVOLVE", prompt)
        self.assertIn("[我知道自己是什么]", prompt)

    def test_group_never_sees_it(self):
        gw = self._gw()
        gw.evolve_enabled = True
        gw.evolve_agent = object()
        gw.evolve_creator = "creator"
        prompt = gw._construct_prompt("你能改自己吗", "", {"emotion": "neutral"}, "G1",
                                      user_id="creator", nickname="菜菜",
                                      evolve_allowed=gw._evolve_allowed("creator", "G1"))
        self.assertNotIn("EVOLVE", prompt)


if __name__ == "__main__":
    unittest.main()
