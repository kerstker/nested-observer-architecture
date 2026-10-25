import os
import random
import sqlite3
import tempfile
import unittest

from src.levels.l8_agency.drives import DriveSystem, digamma, expected_information_gain

def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_ai_"), "d.sqlite")

class TestInformationGain(unittest.TestCase):
    def test_digamma(self):
        self.assertAlmostEqual(digamma(1.0), -0.5772156649, places=8)
        self.assertAlmostEqual(digamma(0.5), -1.9635100260, places=8)
        self.assertAlmostEqual(digamma(10.0) - digamma(9.0), 1 / 9, places=9)

    def test_gain_shrinks_with_experience(self):
        gains = [expected_information_gain(1 + n / 2, 1 + n / 2) for n in (0, 4, 20, 100)]
        self.assertEqual(gains, sorted(gains, reverse=True))
        self.assertGreater(gains[0], 0.15)
        self.assertLess(gains[-1], 0.01)

    def test_gain_is_bounded_and_symmetric(self):
        self.assertAlmostEqual(expected_information_gain(2, 5), expected_information_gain(5, 2), places=12)
        self.assertLessEqual(expected_information_gain(1, 1), 0.7)   # never more than one Bernoulli entropy
        self.assertGreaterEqual(expected_information_gain(50, 0.5), 0.0)

class TestActiveInferenceDrives(unittest.TestCase):
    def test_bayesian_update_and_uncertainty(self):
        d = DriveSystem(db_path())
        self.assertEqual(d.q("social", "chat", "A"), 0.5)
        u0 = d.uncertainty("social", "chat", "A")
        for _ in range(10):
            d.reward("social", "chat", "A", 1.0)
        self.assertGreater(d.q("social", "chat", "A"), 0.85)
        self.assertLess(d.uncertainty("social", "chat", "A"), u0 / 3)
        for _ in range(10):
            d.reward("social", "chat", "B", 0.0)
        self.assertLess(d.q("social", "chat", "B"), 0.15)

    def test_forgetting_keeps_her_open_to_change(self):
        d = DriveSystem(db_path())
        for _ in range(200):
            d.reward("social", "chat", "A", 1.0)
        a, b, n = d.belief("social", "chat", "A")
        self.assertLess(a + b, 60)                     # evidence saturates, it does not grow forever
        self.assertEqual(n, 200)

    def test_new_person_is_worth_finding_out(self):
        d = DriveSystem(db_path(), rng=random.Random(0))
        for _ in range(30):
            d.reward("social", "chat", "A", 0.8)
        opts = {o.target: o for o in d.options({"social": 0.5, "curiosity": 0.6}, 80, ["A", "C"])
                if o.action == "chat"}
        self.assertGreater(opts["C"].epistemic, 3 * opts["A"].epistemic)   # the unknown person is interesting
        self.assertGreater(opts["A"].pragmatic, opts["C"].pragmatic)       # the known one is the safe bet
        self.assertIn("想弄清楚", d.describe(opts["C"]))

    def test_nothing_needed_means_waiting(self):
        d = DriveSystem(db_path(), rng=random.Random(0))
        for u in ("A", "B"):
            for _ in range(30):
                d.reward("social", "chat", u, 0.9)
                d.reward("hunger", "ask_food", u, 0.9)
        for _ in range(30):
            d.reward("curiosity", "search", None, 0.9)
        picks = [d.choose({"social": 0.0, "hunger": 0.0, "curiosity": 0.0}, 90, ["A", "B"]).action
                 for _ in range(200)]
        self.assertGreater(picks.count("wait"), 150)

    def test_pressing_need_drives_action(self):
        d = DriveSystem(db_path(), rng=random.Random(0))
        for _ in range(30):
            d.reward("social", "chat", "A", 0.9)
        picks = [d.choose({"social": 0.9, "hunger": 0.0, "curiosity": 0.0}, 90, ["A"]) for _ in range(200)]
        self.assertGreater(sum(1 for o in picks if o.action == "chat" and o.target == "A"), 150)

    def test_old_q_values_are_migrated(self):
        path = db_path()
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE q_values (need TEXT, action TEXT, target TEXT, q REAL, n INTEGER)")
        conn.execute("INSERT INTO q_values VALUES ('social', 'chat', 'A', 0.9, 20)")
        conn.commit()
        conn.close()
        d = DriveSystem(path)
        self.assertGreater(d.q("social", "chat", "A"), 0.8)
        self.assertEqual(d.belief("social", "chat", "A")[2], 20)
        DriveSystem(path)                                     # migrating twice does not double the evidence
        self.assertEqual(d.belief("social", "chat", "A")[2], 20)

if __name__ == "__main__":
    unittest.main()
