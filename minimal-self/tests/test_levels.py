import unittest
from src.levels.l1_perception.sensory import SensoryCortex
from src.levels.l3_metabolism.metabolism import Metabolism
from src.levels.l4_emotion.emotion import EmotionEngine, EmotionState

class TestFramework(unittest.TestCase):
    def test_sensory(self):
        cortex = SensoryCortex()
        res = cortex.process_text("再不听话我就打你")
        self.assertIsNotNone(res)
        self.assertTrue("pain" in res)
        # the speaker begging not to be hit is not a blow to her
        self.assertIsNone(cortex.process_text("好痛别打我"))
        
        res2 = cortex.process_text("摸摸头")
        self.assertTrue("pleasure" in res2)

    def test_metabolism(self):
        meta = Metabolism(100)
        meta.consume(10)
        self.assertEqual(meta.energy, 90)
        meta.recover(50)
        self.assertEqual(meta.energy, 100)

    def test_emotion(self):
        eng = EmotionEngine()
        eng.update(EmotionState(joy=0.5))
        self.assertEqual(eng.state.joy, 0.5)
        self.assertEqual(eng.get_dominant_emotion(), "joy")

if __name__ == '__main__':
    unittest.main()
