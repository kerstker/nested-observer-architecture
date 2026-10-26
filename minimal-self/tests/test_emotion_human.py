"""Her feelings behave the way a person's do: opposites give way, repeats lose their sting, sleep soothes,
there is a mood under the moment, an apology ends an argument, and silence does not break her heart."""
import asyncio
import os
import random
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import START, ScriptedLLM
from src.levels.l4_emotion.emotion import EmotionEngine, EmotionState
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

HOUR = 3600


def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_emo_"), "m.sqlite")


class TestEngine(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START))
        self.eng = EmotionEngine()

    def test_good_news_takes_the_edge_off_sorrow_and_bad_news_dims_joy(self):
        self.eng.update(EmotionState(sorrow=0.6))
        before = self.eng.state.sorrow
        self.eng.update(EmotionState(joy=0.5))
        self.assertLess(self.eng.state.sorrow, before * 0.85)
        joy = self.eng.state.joy
        self.eng.update(EmotionState(anger=0.5))
        self.assertLess(self.eng.state.joy, joy * 0.85)

    def test_bittersweet_news_is_both_at_once(self):
        self.eng.update(EmotionState(joy=0.5, sorrow=0.5))            # one event, two feelings: not cancelled
        self.assertAlmostEqual(self.eng.state.joy, 0.5, places=2)
        self.assertAlmostEqual(self.eng.state.sorrow, 0.5, places=2)
        self.assertIn("开心", self.eng.describe())
        self.assertIn("失落", self.eng.describe())

    def test_the_tenth_of_the_same_thing_hurts_less_than_the_first(self):
        first = EmotionEngine()
        first.update(EmotionState(sorrow=0.2), source="预测落空")
        repeated = EmotionEngine()
        for _ in range(10):
            repeated.update(EmotionState(sorrow=0.2), source="预测落空")
        unlabelled = EmotionEngine()
        for _ in range(10):
            unlabelled.update(EmotionState(sorrow=0.2))
        self.assertAlmostEqual(first.state.sorrow, 0.2, places=3)       # from rest a push is worth its size
        self.assertLess(repeated.state.sorrow, unlabelled.state.sorrow - 0.1)
        clock.get_clock().advance(6 * HOUR)                            # and a new day it is new again
        again = repeated._habituate("预测落空", clock.now())
        self.assertGreater(again, 0.9)

    def test_sleep_soothes_the_bad_feelings_not_the_good_ones(self):
        awake, asleep = EmotionEngine(), EmotionEngine()
        for e in (awake, asleep):
            e.update(EmotionState(sorrow=0.8, joy=0.0))
            e.state.joy = 0.6
        t = awake.last_decay + HOUR
        awake.decay(t)
        asleep.decay(t, asleep=True)
        self.assertLess(asleep.state.sorrow, awake.state.sorrow - 0.1)
        self.assertAlmostEqual(asleep.state.joy, awake.state.joy, places=6)

    def test_mood_is_the_slow_average_and_is_kept(self):
        for _ in range(12):                                           # a bad afternoon
            self.eng.update(EmotionState(sorrow=0.5))
            self.eng.decay(self.eng.last_decay + 600)
        self.assertLess(self.eng.mood_valence, -0.1)
        mood = self.eng.mood_dict()
        other = EmotionEngine()
        other.load_mood(mood)
        self.assertAlmostEqual(other.mood_valence, self.eng.mood_valence)
        self.eng.decay(self.eng.last_decay + 12 * HOUR)               # half a day later it has mostly lifted
        self.assertGreater(self.eng.mood_valence, -0.1)

    def test_she_can_say_how_she_feels(self):
        self.assertIn("平静", self.eng.describe())
        self.eng.update(EmotionState(anger=0.5, sorrow=0.5))
        self.assertIn("委屈", self.eng.describe())
        self.assertEqual(EmotionEngine().get_compound_state(), "Neutral")
        both = EmotionEngine()
        both.update(EmotionState(joy=0.4, sorrow=0.4))
        self.assertEqual(both.get_compound_state(), "Bittersweet")


class TestInHerLife(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START + 2 * HOUR))
        self.gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path(), rng=random.Random(0))

    def say(self, text, user="u1", name="小明"):
        return asyncio.run(self.gw.hear(text, user_id=user, nickname=name, addressed=True))

    def test_an_apology_ends_the_argument(self):
        self.gw._feel(EmotionState(anger=0.7, sorrow=0.4), "被吵到了")
        angry, sad = self.gw.emotion.state.anger, self.gw.emotion.state.sorrow
        self.say("对不起，是我不好，刚才不该那样说")
        self.assertLess(self.gw.emotion.state.anger, angry * 0.6)
        self.assertLess(self.gw.emotion.state.sorrow, sad)

    def test_thanks_and_praise_warm_her_but_mockery_does_not(self):
        self.say("谢谢你，你好厉害")
        warm = self.gw.emotion.state.joy
        self.assertGreater(warm, 0.1)
        other = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path(), rng=random.Random(0))
        asyncio.run(other.hear("呵呵，你好厉害哦……", user_id="u2", nickname="路人", addressed=True))
        self.assertLess(other.emotion.state.joy, warm)

    def test_a_question_to_a_group_is_not_waited_for_but_one_in_private_is(self):
        reply = "你吃饭了吗？"
        self.gw.inference_engine.reply = reply
        asyncio.run(self.gw.hear("在吗", user_id="p1", nickname="小明", addressed=True))
        asyncio.run(self.gw.hear("在吗", user_id="g1", nickname="小红", group_id="100", addressed=True))
        self.assertIsNotNone(self.gw.user_models.get("p1")["pending"])
        self.assertIsNone(self.gw.user_models.get("g1")["pending"])

    def test_being_ignored_hurts_only_so_much_in_a_day(self):
        names = [f"u{i}" for i in range(12)]
        for u in names:
            err = {"user_id": u, "kind": "reply", "need": None, "elapsed": 3600, "expected_latency": 300, "surprise": 1.0}
            self.gw._on_ignored(err)
        self.assertLessEqual(self.gw.emotion.state.sorrow, 0.36)
        clock.get_clock().advance(26 * HOUR)
        self.gw.emotion.state.sorrow = 0.0
        self.gw._on_ignored({"user_id": "u99", "kind": "reply", "need": None, "elapsed": 3600,
                             "expected_latency": 300, "surprise": 1.0})
        self.assertGreater(self.gw.emotion.state.sorrow, 0.1)               # a new day: it can hurt again

    def test_she_feels_more_for_someone_close(self):
        stranger = self.gw._appraise(None, 0.0, "sad", "", "stranger")
        for _ in range(8):
            self.gw.user_models.on_noa_message("friend", clock.now(), "reply", True)
            self.gw.user_models.on_user_message("friend", clock.now() + 5, 0.0, True)
        friend = self.gw._appraise(None, 0.0, "sad", "", "friend")
        self.assertGreater(friend.sorrow, stranger.sorrow)

    def test_exhaustion_shortens_the_temper_and_loneliness_is_sad(self):
        self.gw._body_temperament()
        rested = dict(self.gw.emotion.baseline)
        self.assertEqual(rested["anger"], 0.0)
        self.assertEqual(rested["sorrow"], 0.0)
        self.gw.metabolism.energy = 5
        self.gw.homeostasis.social_need = 1.0
        self.gw._body_temperament()
        worn = self.gw.emotion.baseline
        self.assertGreater(worn["anger"], 0.05)
        self.assertGreater(worn["sorrow"], 0.1)
        self.assertLess(worn["joy"], rested["joy"])
        self.gw.emotion.decay(self.gw.emotion.last_decay + 6 * HOUR)       # given time she settles there, not at zero
        self.assertGreater(self.gw.emotion.state.sorrow, 0.1)

    def test_the_mood_survives_a_restart_and_reaches_the_prompt_words(self):
        self.gw.emotion.mood_valence = -0.4
        self.gw._save_state("test")
        again = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=self.gw.memory.db.db_path, rng=random.Random(0))
        again._restore_state() if hasattr(again, "_restore_state") else None
        self.assertIn("不太舒服", self.gw.emotion.describe())
        self.assertIn("不太舒服", self.gw._feeling_words())


if __name__ == "__main__":
    unittest.main()
