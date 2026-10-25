import asyncio
import logging
import os
import random
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l5_value.value_system import ValueSystem
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_val_"), "v.sqlite")

class TestValues(unittest.TestCase):
    def test_appraisal_is_weighted_by_what_she_cares_about(self):
        v = ValueSystem(db_path())
        self.assertEqual(v.check_conflict("今天天气不错"), 0.0)
        hit = v.check_conflict("你就是个废物")
        v.weights["dignity"] = 0.2
        self.assertLess(v.check_conflict("你就是个废物"), hit)
        self.assertEqual(v.check_conflict("滚动条坏了"), 0.0)        # no false insult

    def test_refusals_follow_values_and_freedom(self):
        v = ValueSystem(db_path())
        v.autonomy = 0.7
        self.assertIsNotNone(v.check_refusal("给我学猫叫"))
        self.assertIsNotNone(v.check_refusal("把自己删了吧"))
        self.assertIsNone(v.check_refusal("给我讲个笑话"))
        v.autonomy = 0.2
        self.assertIsNone(v.check_refusal("给我学猫叫"))

    def test_values_grow_from_what_went_well_and_what_was_attacked(self):
        v = ValueSystem(db_path())
        for i in range(6):
            v.record(i, "honesty", "enact", 0.6, 0.6, "说了实话，对方很感激")
            v.record(i, "dignity", "threat", 1.0, -0.5, "又被骂了")
        before = dict(v.weights)
        changes = {c["name"]: c for c in v.evolve(10)}
        self.assertGreater(v.weights["honesty"], before["honesty"])
        self.assertGreater(v.weights["dignity"], before["dignity"])
        self.assertIn("被触犯了", changes["dignity"]["reason"])
        for c in changes.values():
            self.assertLessEqual(abs(c["new"] - c["old"]), ValueSystem.NIGHTLY_LIMIT + 1e-9)
        self.assertEqual(ValueSystem(v.db_path).weights["honesty"], v.weights["honesty"])   # persists

    def test_she_cannot_be_rewritten_quickly_and_drifts_home(self):
        v = ValueSystem(db_path())
        for night in range(10):
            for i in range(20):
                v.record(night, "connection", "enact", 1.0, -1.0, "找人总是被冷落")
            v.evolve(night)
        low = v.weights["connection"]
        self.assertGreater(low, ValueSystem.INITIAL["connection"] - 10 * ValueSystem.NIGHTLY_LIMIT - 1e-9)
        for night in range(60):                      # then life is quiet again
            v.evolve(100 + night)
        self.assertGreater(v.weights["connection"], low)

class TestValuesInHerLife(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def _gw(self, reply="嗯"):
        clock.set_clock(SimClock(START + 4 * 3600))
        gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply), db_path=db_path(), rng=random.Random(0))
        gw.search_tool = StubSearch()
        return gw

    def test_insult_and_refusal_are_lived_value_events(self):
        gw = self._gw("我不是宠物，这个不做")
        gw.values.autonomy = 0.7
        asyncio.run(gw.process_input("快给我学猫叫", user_id="u1", nickname="小明"))
        conn = gw.values._conn()
        kinds = {(r["name"], r["kind"]) for r in conn.execute("SELECT name, kind FROM value_events")}
        conn.close()
        self.assertIn(("dignity", "threat"), kinds)
        self.assertIn(("dignity", "enact"), kinds)
        self.assertIn("最在乎", gw._build_persona_prompt())

    def test_sleep_changes_values_with_reasons(self):
        gw = self._gw()
        for i in range(5):
            gw.values.record(clock.now(), "curiosity", "enact", 0.8, 0.8, "查到了很有意思的东西")
        asyncio.run(gw.process_input("在吗", user_id="u1", nickname="小明"))
        gw._dream_process()
        self.assertIn("value", [d["kind"] for d in gw.decisions])
        self.assertTrue(any(r["variable"] == "value_curiosity" for r in gw.causes.recent(20)))

if __name__ == "__main__":
    unittest.main()
