"""Her game memory: what her hands did, and lessons that come back when they matter."""
import os
import tempfile
import unittest

from src.levels.l8_agency.mc_memory import Experience


class TestExperience(unittest.TestCase):
    def test_lessons_are_kept_counted_and_recalled_by_topic(self):
        path = os.path.join(tempfile.mkdtemp(), "exp.json")
        e = Experience(path)
        e.learn("挖 铁矿石 拿 粗铁 没成（在 y=-58）：矿道新挖了 20 格，挖到矿 0 块", about="粗铁 铁镐")
        e.learn("挖 铁矿石 拿 粗铁 没成（在 y=-58）：矿道新挖了 20 格，挖到矿 0 块", about="粗铁 铁镐")
        e.learn("窄矿道里工作台贴墙放", about="工作台", source="self")
        again = Experience(path)
        self.assertEqual(again.lessons[0]["text"], "窄矿道里工作台贴墙放")       # her own notes first
        self.assertEqual([l["n"] for l in again.lessons if "y=-58" in l["text"]], [2])
        self.assertIn("y=-58", again.relevant("铁镐 挖 铁矿石")[0])
        self.assertEqual(again.relevant("钓鱼"), [])

    def test_what_the_hands_just_did(self):
        e = Experience()
        for i in range(8):
            e.did(f"step {i}")
        self.assertEqual(len(e.last(5)), 5)
        self.assertTrue(e.last(1)[0].endswith("step 7"))


if __name__ == "__main__":
    unittest.main()
