"""Memory that does not fill up: the in-memory index, archiving and recall by effort, re-embedding
when the model changes, person portraits, the combined embedder, and the daily personality limit."""
import os
import random
import sqlite3
import tempfile
import unittest

import numpy as np

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, START
from src.levels.l2_memory.embedding import Embedder, default_model_dir
from src.levels.l2_memory.layers import EpisodicMemory, SemanticMemory
from src.levels.l2_memory.profiles import PersonProfiles
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = 86400

def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_mem_"), "m.sqlite")

class TestIndex(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START))
        self.db = db_path()
        self.ep = EpisodicMemory(self.db, Embedder())

    def test_new_episodes_are_found_after_the_index_was_built(self):
        self.ep.add("我们聊了猫咪和小鱼干", user_id="u1")
        self.assertEqual(len(self.ep.recall("猫咪小鱼干", user_id="u1")), 1)        # builds the index
        self.ep.add("后来又聊到了显卡和电脑", user_id="u1")
        hits = self.ep.recall("显卡电脑", user_id="u1")
        self.assertTrue(hits and "显卡" in hits[0]["content"])

    def test_scope_is_the_same_as_before(self):
        self.ep.add("私聊：我的生日是三月五号", user_id="u1")
        self.ep.add("群里：大家在聊生日蛋糕", user_id="u2", group_id="g1")
        self.assertEqual(self.ep.recall("生日", user_id="u3"), [])
        self.assertEqual(len(self.ep.recall("生日", user_id="u3", group_id="g1")), 1)
        self.assertEqual(len(self.ep.recall("生日", user_id="u1")), 1)

    def test_recall_counts_are_kept_in_memory_and_on_disk(self):
        eid = self.ep.add("一起看了烟花", user_id="u1")
        self.ep.recall("烟花", user_id="u1")
        self.ep.recall("烟花", user_id="u1")
        conn = sqlite3.connect(self.db)
        self.assertEqual(conn.execute("SELECT recall_count FROM episodes WHERE id=?", (eid,)).fetchone()[0], 2)
        conn.close()
        self.assertEqual(self.ep._index.recalls[self.ep._index.ids == eid][0], 2)

    def test_distilled_old_details_fade_but_effort_brings_them_back(self):
        old = self.ep.add("很久以前聊过一次天文望远镜", user_id="u1", consolidated=True)
        vivid = self.ep.add("很久以前被吓坏了的望远镜事故", user_id="u1", consolidated=True, arousal=0.9)
        fresh = self.ep.add("望远镜的事还没整理", user_id="u1")
        clock.get_clock().advance(40 * DAY)
        self.assertEqual(self.ep.archive(), 1)                              # only the calm, distilled one
        everyday = {e["id"] for e in self.ep.recall("天文望远镜", user_id="u1", k=5)}
        self.assertNotIn(old, everyday)
        self.assertTrue({vivid, fresh} <= everyday)
        deep = {e["id"] for e in self.ep.recall("天文望远镜", user_id="u1", k=5, deep=True)}
        self.assertIn(old, deep)
        self.assertIn(old, {e["id"] for e in self.ep.recall("天文望远镜", user_id="u1", k=5)})   # back in reach
        self.assertEqual(self.ep.stats()["archived"], 0)

    def test_vectors_from_another_model_are_re_embedded(self):
        eid = self.ep.add("下雨天适合睡觉", user_id="u1")
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE episodes SET emb_model='some-old-model', embedding=? WHERE id=?", (b"\x00" * 64, eid))
        conn.commit()
        conn.close()
        fresh = EpisodicMemory(self.db, Embedder())
        self.assertEqual(len(fresh.recall("下雨睡觉", user_id="u1")), 1)

    def test_novelty_uses_the_index(self):
        self.assertEqual(self.ep.novelty("什么都没有"), 1.0)
        self.ep.add("今天吃了拉面")
        self.assertLess(self.ep.novelty("今天吃了拉面"), 0.05)

class TestEmbedder(unittest.TestCase):
    def test_ngram_scale_is_unchanged(self):
        e = Embedder()
        v = e.embed("一模一样的一句话")
        self.assertAlmostEqual(e.similarity(v, v), 1.0, places=3)
        self.assertEqual(e.offset, 0.0)
        self.assertEqual(e.similarity(v, np.zeros(10)), 0.0)          # another space: not comparable

    @unittest.skipUnless(default_model_dir(), "local model not downloaded")
    def test_combined_vectors_recall_paraphrases(self):
        e = Embedder(model_dir=default_model_dir())
        self.assertEqual(e.backend, "onnx")
        a, b, c = e.embed_many(["我最近在玩原神", "这阵子一直在打一款开放世界的游戏", "今天北京下雨了"])
        self.assertAlmostEqual(e.similarity(a, a), 1.0, places=2)
        self.assertGreater(e.similarity(a, b), e.similarity(a, c) + 0.05)
        self.assertLess(abs(e.similarity(a, c)), 0.2)                       # unrelated sits near 0

class TestPeople(unittest.TestCase):
    def test_portrait_and_relevant_facts(self):
        clock.set_clock(SimClock(START))
        db = db_path()
        sem = SemanticMemory(db, Embedder())
        for f in ("喜欢看科幻电影", "养了一只橘猫", "在学吉他", "下个月要考驾照"):
            sem.add("u1", f, [])
        profiles = PersonProfiles(db)
        self.assertEqual([p["user_id"] for p in profiles.stale()], ["u1"])
        llm = ScriptedLLM("小明喜欢科幻和他的橘猫，最近在忙学吉他和考驾照。")
        self.assertIn("橘猫", profiles.rebuild("u1", "小明", llm.generate_json))
        self.assertEqual(profiles.stale(), [])
        self.assertIn("橘猫", sem.relevant("u1", "你家那只橘猫怎么样了", k=1)[0]["fact"])
        clock.get_clock().advance(60)
        sem.add("u1", "换了新工作", [])
        self.assertEqual([p["user_id"] for p in profiles.stale()], ["u1"])     # facts changed: redraw

    def test_prompt_carries_the_portrait(self):
        clock.set_clock(SimClock(START + 2 * 3600))
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯嗯"), db_path=db_path(), rng=random.Random(0))
        gw.environment.get_context = lambda: {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None,
                                              "is_night": False}
        for f in ("养了一只橘猫", "在学吉他", "喜欢科幻"):
            gw.semantic.add("u1", f, [])
        conn = sqlite3.connect(gw.memory.db.db_path)
        conn.execute("INSERT INTO person_profiles VALUES ('u1', '小明是个爱猫的吉他新手', 3, ?)", (clock.now(),))
        conn.commit()
        conn.close()
        import asyncio
        asyncio.run(gw.hear("我的猫今天又闯祸了", user_id="u1", nickname="小明", addressed=True))
        prompt = next(c["prompt"] for c in reversed(gw.inference_engine.calls) if "[要求]" in c["prompt"])
        self.assertIn("我对 小明 的印象：小明是个爱猫的吉他新手", prompt)
        self.assertIn("橘猫", prompt)

class TestPersonalityLimit(unittest.TestCase):
    def test_the_limit_is_per_day_however_often_she_consolidates(self):
        clock.set_clock(SimClock(START))
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path(), rng=random.Random(0))
        before = gw.memory.db.get_persona_traits()["affection"]["value"]
        happy = [{"id": i, "kind": "dialogue", "user_id": "u1", "valence": 0.9, "arousal": 0.3, "conflict": 0}
                 for i in range(5)]
        for _ in range(8):                                  # hourly consolidation all day
            gw.narrative.drift_traits(happy)
            clock.get_clock().advance(3600)
        after = gw.memory.db.get_persona_traits()["affection"]["value"]
        self.assertLessEqual(after - before, 0.05 + 1e-9)
        clock.get_clock().advance(DAY)                       # a new day: she may move again
        gw.narrative.drift_traits(happy)
        self.assertGreater(gw.memory.db.get_persona_traits()["affection"]["value"], after)

    def test_over_budget_stops_but_does_not_swing_back(self):
        clock.set_clock(SimClock(START))
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path(), rng=random.Random(0))
        conn = sqlite3.connect(gw.memory.db.db_path)            # the old bug: +0.2 in a few hours
        conn.execute("INSERT INTO trait_changes (t, trait, old, new, evidence, reason) VALUES (?, 'affection', 0.5, 0.7, '[]', 'x')",
                     (clock.now(),))
        conn.commit()
        conn.close()
        gw.memory.db.update_persona_trait("affection", 0.7)
        calm = [{"id": 1, "kind": "dialogue", "user_id": "u1", "valence": 0.05, "arousal": 0.3, "conflict": 0}]
        gw.narrative.drift_traits(calm)
        self.assertAlmostEqual(gw.memory.db.get_persona_traits()["affection"]["value"], 0.7, places=6)
        sad = [{**calm[0], "valence": -0.9}]
        gw.narrative.drift_traits(sad)                           # moving back is still allowed
        self.assertLess(gw.memory.db.get_persona_traits()["affection"]["value"], 0.7)

if __name__ == "__main__":
    unittest.main()
