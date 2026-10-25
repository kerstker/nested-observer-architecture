import unittest
import asyncio
import os
import random
import tempfile
import time
from src.core.event_bus import EventBus
from src.levels.l1_perception.environment import EnvironmentSensor
from src.levels.l3_metabolism.metabolism import Metabolism
from src.levels.l4_emotion.emotion import EmotionEngine, EmotionState
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

class DummyLLM:
    def generate_json(self, prompt):
        return {"reply": "测试回复"}

class SlowLLM:
    def generate_json(self, prompt):
        time.sleep(0.5)  # Simulates a blocking llama.cpp call
        return {"reply": "慢回复"}

DAY = {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None, "is_night": False}
NIGHT = {**DAY, "time": "23:00", "period": "Night", "is_night": True}

def make_gw(llm=None, env=DAY):
    # Always a throwaway database: constructing a workspace with the default db_path writes
    # into the live memory.sqlite ("在仓库根跑测试 = 往她的记忆里塞测试数据").
    db = os.path.join(tempfile.mkdtemp(prefix="noa_phase0_"), "p.sqlite")
    gw = GlobalWorkspace(EventBus(), llm or DummyLLM(), db_path=db)
    gw.environment.get_context = lambda: env
    return gw

class TestCodeExecutionDisabled(unittest.TestCase):
    def test_interpreter_refuses(self):
        gw = make_gw()
        result = gw.interpreter.execute("print(1)")
        self.assertFalse(result["success"])

    def test_action_not_executed_or_echoed(self):
        gw = make_gw()
        text = "Response: 好的\nAction: EXECUTE_CODE\n```python\nprint(().__class__)\n```"
        self.assertIsNone(gw._handle_actions(text, "u1"))
        self.assertNotIn("EXECUTE_CODE", gw._finalize_response(text))

class TestSleepWakeCycle(unittest.TestCase):
    def test_energy_recovers_while_sleeping(self):
        meta = Metabolism(100)
        meta.energy = 20
        meta.is_sleeping = True
        meta.last_update = time.time() - 3600
        meta.tick()
        self.assertAlmostEqual(meta.energy, 35, delta=0.1)

    def test_energy_drains_by_time_not_ticks(self):
        meta = Metabolism(100)
        for _ in range(100):
            meta.tick()
        self.assertGreater(meta.energy, 99.9)

    def test_wakes_when_rested_during_day(self):
        gw = make_gw(env=DAY)
        gw._set_sleeping(True, source="test")
        gw.metabolism.energy = 85
        asyncio.run(gw.tick())
        self.assertFalse(gw.metabolism.is_sleeping)

    def test_keeps_sleeping_at_night(self):
        gw = make_gw(env=NIGHT)
        gw._set_sleeping(True, source="test")
        gw.metabolism.energy = 85
        asyncio.run(gw.tick())
        self.assertTrue(gw.metabolism.is_sleeping)

    def test_falls_asleep_when_exhausted(self):
        gw = make_gw(env=DAY)
        gw.metabolism.energy = 5
        asyncio.run(gw.tick())
        self.assertTrue(gw.metabolism.is_sleeping)

class TestGlobalRandomUntouched(unittest.TestCase):
    def test_environment_does_not_reseed_global_random(self):
        random.seed(42)
        expected = random.random()
        random.seed(42)
        EnvironmentSensor().get_context()
        self.assertEqual(random.random(), expected)

class TestEmotionResponds(unittest.TestCase):
    def test_petting_brings_joy(self):
        gw = make_gw()
        asyncio.run(gw.process_input("摸摸头", user_id="u1", nickname="ker"))
        self.assertEqual(gw.emotion.get_dominant_emotion(), "joy")

    def test_insult_brings_anger(self):
        gw = make_gw()
        asyncio.run(gw.process_input("you are stupid and useless", user_id="u1", nickname="ker"))
        self.assertEqual(gw.emotion.get_dominant_emotion(), "anger")

    def test_decay_halves_after_half_life(self):
        eng = EmotionEngine(half_life_seconds=1200)
        eng.update(EmotionState(joy=0.8))
        eng.decay(eng.last_decay + 1200)
        self.assertAlmostEqual(eng.state.joy, 0.4, places=3)

class TestLLMDoesNotBlockLoop(unittest.TestCase):
    def test_event_loop_keeps_running_during_llm_call(self):
        gw = make_gw(llm=SlowLLM())
        beats = []

        async def heartbeat():
            for _ in range(20):
                beats.append(time.time())
                await asyncio.sleep(0.05)

        async def main():
            await asyncio.gather(gw.process_input("你好", user_id="u1", nickname="ker"), heartbeat())

        asyncio.run(main())
        gaps = [b - a for a, b in zip(beats, beats[1:])]
        self.assertLess(max(gaps), 0.3)

if __name__ == "__main__":
    unittest.main()
