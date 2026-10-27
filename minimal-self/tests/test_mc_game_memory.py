"""Her game memory by meaning: writing what she did, recalling it by meaning, and what sleep does to it.

The tests run with the hash-only Embedder (no ONNX model), so they exercise the same code path as the
real one but score by character n-grams; `recall` scales its threshold down for that backend accordingly.
"""
import os
import sqlite3
import tempfile
import time
import unittest

from src.levels.l2_memory.embedding import Embedder
from src.levels.l8_agency.mc_game_memory import GameMemory


def _mem(name: str = "gm.sqlite") -> GameMemory:
    return GameMemory(os.path.join(tempfile.mkdtemp(), name), Embedder())


class TestRecallByMeaning(unittest.TestCase):
    def setUp(self):
        self.m = _mem()
        self.iron = self.m.remember("我在矿洞里挖到了铁矿石，拿到了两个粗铁", kind="event", pos=(-1155, 12, 280),
                                    importance=0.6, tags=["挖矿"])
        self.home = self.m.remember("我在(100,70,-50)安了家：箱子、工作台、熔炉", kind="build", pos=(100, 70, -50),
                                    importance=0.9, tags=["家"])
        self.chest = self.m.remember("我把十二个铁锭放进了家旁边的箱子里", kind="object", pos=(101, 70, -50),
                                     importance=0.7, tags=["箱子"])
        self.lesson = self.m.remember("我记得一条经验：y=-58 深层几乎没有铁，要回到 y=16 附近挖", kind="lesson",
                                      importance=0.85, tags=["经验", "铁"])
        self.death = self.m.remember("我在(-200,30,600)死了一次", kind="death", pos=(-200, 30, 600), importance=0.9)

    def tearDown(self):
        self.m.close()

    def test_recall_by_paraphrase_returns_the_right_memory_first(self):
        got = self.m.recall("怎么挖到铁矿石")
        self.assertTrue(got)
        self.assertEqual(got[0]["id"], self.iron)
        self.assertIn("铁矿石", got[0]["text"])
        self.assertIn("score", got[0])
        self.assertGreater(got[0]["score"], got[-1]["score"])
        got = self.m.recall("铁锭放在哪个箱子")
        self.assertEqual(got[0]["id"], self.chest)
        got = self.m.recall("我在哪里死过")
        self.assertEqual(got[0]["id"], self.death)
        got = self.m.recall("挖矿经验", k=3, kinds=["lesson"])
        self.assertEqual(got[0]["id"], self.lesson)
        self.assertEqual(self.m.recall("怎么种田"), [])          # nothing she remembers is about farming

    def test_recall_shapes_and_limits(self):
        got = self.m.recall("铁矿石 箱子 家", k=2)
        self.assertLessEqual(len(got), 2)
        for rec in got:
            for key in ("id", "text", "kind", "score", "similarity", "t", "when", "importance", "pos"):
                self.assertIn(key, rec)
        self.assertEqual(self.m.recall("铁", min_score=1.5), [])   # the gate holds

    def test_memories_survive_reopening(self):
        again = GameMemory(self.m.db_path, self.m.embedder)
        try:
            self.assertEqual(again.stats()["records"], 5)
            self.assertEqual(again.recall("怎么挖到铁矿石")[0]["id"], self.iron)
        finally:
            again.close()


class TestBoostAndFilters(unittest.TestCase):
    def test_proximity_puts_what_happened_here_first(self):
        m = _mem()
        here = m.remember("我在那里放下了箱子", kind="build", pos=(0, 64, 0), importance=0.5)
        far = m.remember("我在那里放下了箱子", kind="build", pos=(400, 64, 400), importance=0.5)
        got = m.recall("箱子放哪了", near=(0, 64, 0))
        self.assertEqual(got[0]["id"], here)                      # same words, but this is the place we are at
        self.assertIn("distance", got[0])
        self.assertLess(got[0]["distance"], 1.0)
        ids = [r["id"] for r in m.recall("箱子放哪了", near=(400, 64, 400))]
        self.assertEqual(ids[0], far)
        here_first = m.nearby((0, 64, 0), radius=64, k=2)
        self.assertEqual(here_first[0]["id"], here)
        self.assertEqual(here_first[0]["direction"], "就在这")
        m.close()

    def test_importance_and_recency_order_equal_words(self):
        m = _mem()
        now = time.time()
        old = m.remember("我挖到了煤", kind="event", importance=0.5, t=now - 10 * 86400)
        new = m.remember("我挖到了煤", kind="event", importance=0.5, t=now - 60)
        m2 = _mem("gm2.sqlite")                                  # importance with everything else equal
        low = m2.remember("我挖到了煤", kind="event", importance=0.2, t=now)
        high = m2.remember("我挖到了煤", kind="event", importance=0.9, t=now)
        got = m.recall("我挖到了煤", k=2)
        self.assertEqual(got[0]["id"], new)
        self.assertEqual(got[1]["id"], old)
        got2 = m2.recall("我挖到了煤", k=2)
        self.assertEqual(got2[0]["id"], high)
        self.assertEqual(got2[1]["id"], low)
        m.close()
        m2.close()

    def test_kind_filter_recent_and_forget(self):
        m = _mem()
        lesson = m.remember("我记得一条经验：夜里别在矿洞里乱走", kind="lesson", t=500.0, importance=0.8)
        first = m.remember("我砍了一棵树", kind="event", t=1000.0, importance=0.4)
        second = m.remember("我放下了工作台", kind="build", t=2000.0, importance=0.5)
        self.assertEqual([r["id"] for r in m.recent(2)], [second, first])
        self.assertEqual([r["id"] for r in m.recent(3, kinds=["lesson"])], [lesson])
        self.assertTrue(m.forget(first))
        self.assertFalse(m.forget(999999))
        self.assertNotIn(first, [r["id"] for r in m.recent(10)])
        self.assertEqual([r["id"] for r in m.recent(3)], [second, lesson][:2])
        m.close()


class TestSleep(unittest.TestCase):
    def test_duplicates_merge_and_the_important_never_fade(self):
        m = _mem()
        for _ in range(6):
            m.remember("我在(5,60,5)挖掉了泥土", kind="event", pos=(5, 60, 5), importance=0.3)
        home = m.remember("我在(1,2,3)安了家：箱子和工作台", kind="build", pos=(1, 2, 3), importance=0.95)
        sand = m.remember("我在(9,9,9)挖掉了沙子", kind="event", pos=(9, 9, 9), importance=0.1)
        out = m.consolidate(decay=1.0)
        self.assertEqual(out["merged"], 5)
        self.assertEqual(m.stats()["records"], 3)
        merged = [r for r in m.recent(10) if "泥土" in r["text"]][0]
        self.assertGreaterEqual(merged["n"], 6)
        for _ in range(8):
            m.consolidate(decay=0.5)
        ids = [r["id"] for r in m.recent(20)]
        self.assertIn(home, ids)                                 # a thing that mattered is not lost
        self.assertNotIn(sand, ids)                              # the faded leftover is gone
        found = m.recall("安家 箱子 工作台", k=3)
        self.assertIn(home, [r["id"] for r in found])
        m.close()

    def test_table_stays_bounded_but_keeps_lessons(self):
        m = _mem()
        lesson = m.remember("我记得一条经验：先做木镐再挖石头", kind="lesson", importance=0.6)
        for i in range(12):
            m.remember(f"我在({i},64,{i})挖掉了泥土", kind="event", pos=(i, 64, i), importance=0.2)
        out = m.consolidate(decay=1.0, max_records=4)
        self.assertGreater(out["pruned"], 0)
        self.assertLessEqual(m.stats()["records"], 6)            # bounded (protected rows may exceed the cap)
        self.assertIn(lesson, [r["id"] for r in m.recent(20)])
        m.close()


class TestModelMigration(unittest.TestCase):
    def test_vectors_from_another_model_are_re_embedded(self):
        path = os.path.join(tempfile.mkdtemp(), "mig.sqlite")
        old = GameMemory(path, Embedder(dim=1024))
        for i in range(4):
            old.remember(f"我在({i},64,0)挖掉了铁矿石")
        old.close()
        new = GameMemory(path, Embedder(dim=512))
        try:
            self.assertEqual(new.last_migration, 4)
            self.assertEqual(new.stats()["model_id"], "hash-512")
            rows = sqlite3.connect(path).execute("SELECT emb_model FROM game_memory").fetchall()
            self.assertEqual({r[0] for r in rows}, {"hash-512"})
            self.assertTrue(new.recall("铁矿石在哪"))
        finally:
            new.close()


class TestRecordAction(unittest.TestCase):
    def setUp(self):
        self.m = _mem()

    def tearDown(self):
        self.m.close()

    def test_mining_action_becomes_a_sentence(self):
        rid = self.m.record_action("mc_mine_block", {"x": -1155, "y": 12, "z": 280},
                                   "挖掉了 minecraft:iron_ore（-1155,12,280）。", ok=True,
                                   pos=(-1155, 12, 280))
        self.assertIsInstance(rid, int)
        rec = self.m.recent(1)[0]
        self.assertIn("(-1155,12,280)", rec["text"])
        self.assertIn("铁矿石", rec["text"])
        self.assertEqual(rec["kind"], "event")
        self.assertEqual(rec["pos"], {"x": -1155.0, "y": 12.0, "z": 280.0})
        self.assertIn("minecraft:iron_ore", rec["items"])

    def test_failure_is_remembered_as_an_attempt_with_reason(self):
        self.m.record_action("mc_mine_block", {"x": 1, "y": 2, "z": 3},
                             "走不到那个方块旁边（被挡住或者没路）。", ok=False, pos=(1, 2, 3))
        rec = self.m.recent(1)[0]
        self.assertIn("没挖成", rec["text"])
        self.assertIn("走不到", rec["text"])
        self.assertLess(rec["importance"], 0.5)
        self.assertIn("失败", rec["tags"])
        # ok omitted: the wording of the result decides
        self.m.record_action("mc_place_block", {"x": 4, "y": 5, "z": 6}, "走不到要放的位置旁边。")
        self.assertIn("没放成", self.m.recent(1)[0]["text"])

    def test_place_skill_chest_and_goal(self):
        rid = self.m.record_action("mc_place_block", {"x": 10, "y": 65, "z": 10, "item": "minecraft:crafting_table"},
                                   "放好了 minecraft:crafting_table（10,65,10）。", ok=True)
        rec = self.m.recent(1)[0]
        self.assertEqual(rec["kind"], "build")
        self.assertIn("放下了工作台", rec["text"])
        self.assertEqual(rec["id"], rid)
        self.m.record_action("mc_skill", {"name": "mine", "blocks": "minecraft:iron_ore", "count": 3},
                             "挖了 3 块。捡了 2 次东西。", ok=True)
        rec = self.m.recent(1)[0]
        self.assertIn("3块铁矿石", rec["text"])
        self.assertEqual(rec["kind"], "event")
        self.m.record_action("mc_container_transfer", {"direction": "take", "item": "minecraft:raw_iron", "count": 9},
                             "拿了 9 个粗铁。", ok=True, pos=(101, 70, -50))
        rec = self.m.recent(1)[0]
        self.assertEqual(rec["kind"], "object")
        self.assertIn("从箱子里拿了9个粗铁", rec["text"])
        self.m.record_action("mc_skill", {"name": "set_home"}, "安好家了。", ok=True, pos=(100, 70, -50))
        self.assertEqual(self.m.recent(1)[0]["kind"], "build")
        self.m.record_action("mc_goal", {"item": "minecraft:iron_pickaxe"}, "目标换成了铁镐", ok=True)
        rec = self.m.recent(1)[0]
        self.assertEqual(rec["kind"], "quest")
        self.assertIn("铁镐", rec["text"])

    def test_lessons_and_reading_tools(self):
        self.m.record_action("mc_remember", {"note": "深板岩层铁很少，要回到 y=16 挖", "about": "铁"},
                             "记下了。", ok=True)
        rec = self.m.recent(1)[0]
        self.assertEqual(rec["kind"], "lesson")
        self.assertIn("深板岩层铁很少", rec["text"])
        self.assertGreaterEqual(rec["importance"], 0.75)
        for tool in ("mc_look", "mc_look_at", "mc_recipes", "mc_know", "mc_who_is_near", "mc_find_blocks"):
            self.assertIsNone(self.m.record_action(tool, {}, "看过了。", ok=True))


class TestScene(unittest.TestCase):
    def test_scene_is_a_compact_block_or_nothing(self):
        m = _mem()
        self.assertEqual(m.scene(near=(0, 64, 0), query="铁矿石"), "")
        self.assertEqual(m.brief(), "")
        m.record_action("mc_mine_block", {"x": -1155, "y": 12, "z": 280},
                        "挖掉了 minecraft:iron_ore（-1155,12,280）。", ok=True, pos=(-1155, 12, 280))
        m.record_action("mc_place_block", {"x": 100, "y": 70, "z": -50, "item": "minecraft:chest"},
                        "放好了 minecraft:chest（100,70,-50）。", ok=True, pos=(100, 70, -50))
        m.remember("我记得一条经验：铁矿石要往下挖到 y=16 附近", kind="lesson", importance=0.8)
        m.remember("这个箱子里我放过铁锭", kind="object", pos=(101, 70, -50), t=time.time() - 1800, importance=0.6)
        m.remember("我记得铁矿石一般在 y=16 的矿洞里", kind="event", pos=(0, 16, 0), t=time.time() - 3600,
                   importance=0.5)
        block = m.scene(near=(100, 70, -50), query="铁矿石 铁镐", recent_n=2)
        self.assertIn("我记得的最近的事：", block)
        self.assertIn("关于这里：", block)
        self.assertIn("关于「", block)
        self.assertIn("我记得的教训：", block)
        self.assertLessEqual(len(block.splitlines()), 14)
        self.assertEqual(m.brief(near=(100, 70, -50), query="铁矿石 铁镐", recent_n=2), block)
        m.close()


class TestSpeed(unittest.TestCase):
    def test_recall_over_two_thousand_records_stays_under_20ms(self):
        m = _mem("speed.sqlite")
        for i in range(2000):
            m.remember(f"我在({i},{60 + i % 10},{i * 2})挖掉了铁矿石，拿到了粗铁", kind="event",
                       pos=(i, 60 + i % 10, i * 2), importance=0.3)
        self.assertEqual(m.stats()["records"], 2000)
        m.recall("铁矿石在哪挖的", k=6)                            # warm: build the index
        times = []
        for _ in range(25):
            t0 = time.perf_counter()
            got = m.recall("铁矿石在哪挖的", k=6, near=(1000, 64, 2000))
            times.append((time.perf_counter() - t0) * 1000.0)
            self.assertEqual(len(got), 6)
        times.sort()
        median = times[len(times) // 2]
        self.assertLess(median, 20.0, f"recall too slow: median {median:.1f}ms, max {times[-1]:.1f}ms")
        m.close()


if __name__ == "__main__":
    unittest.main()
