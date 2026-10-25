import logging
import unittest
from src.evaluation import scenarios

class TestLifeClaims(unittest.TestCase):
    """The seven behavioural claims from the design doc (section 7), on a simulated clock."""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def check(self, scenario):
        r = scenario(seed=0)
        self.assertTrue(r["passed"], f"{r['name']}: {r['measured']} (标准：{r['criterion']})")

    def test_continuity_of_self(self):
        self.check(scenarios.continuity_of_self)

    def test_state_is_causal(self):
        self.check(scenarios.state_is_causal)

    def test_attention_competition(self):
        self.check(scenarios.attention_competition)

    def test_prediction_and_surprise(self):
        self.check(scenarios.prediction_and_surprise)

    def test_learning_who_answers(self):
        self.check(scenarios.learning_who_answers)

    def test_personality_from_experience(self):
        self.check(scenarios.personality_from_experience)

    def test_grounded_introspection(self):
        self.check(scenarios.grounded_introspection)

    def test_social_tact(self):
        self.check(scenarios.social_tact)

    def test_no_bot_loop(self):
        self.check(scenarios.no_bot_loop)

if __name__ == "__main__":
    unittest.main()
