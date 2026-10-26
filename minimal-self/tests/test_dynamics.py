"""Her formulas on her real life: what counts as being fed or petted, how emotions rise and fade, how surprise
is measured, how needs and energy move, and how her traits drift - the cases come from her own history."""
import asyncio
import math
import os
import random
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, START
from src.levels.l1_perception.sensory import SensoryCortex
from src.levels.l3_metabolism.metabolism import Metabolism
from src.levels.l4_emotion.emotion import EmotionEngine, EmotionState
from src.levels.l5_value.value_system import ValueSystem
from src.levels.l6_homeostasis.homeostasis import Homeostasis
from src.levels.l7_tom.user_model import UserModelStore
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

HOUR = 3600.0

def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_dyn_"), "d.sqlite")

class TestSensation(unittest.TestCase):
    """Something done *to her*, not words that happen to contain a trigger."""
    def setUp(self):
        self.cortex = SensoryCortex()

    def kinds(self, text):
        return set(self.cortex.process_text(text) or {})

    def test_real_feeding_petting_washing_hurting(self):
        self.assertEqual(self.kinds("喂你吃小鱼干"), {"food"})
        self.assertEqual(self.kinds("给你鲱鱼罐头"), {"food"})
        self.assertEqual(self.kinds("摸摸头"), {"pleasure"})
        self.assertEqual(self.kinds("抱抱"), {"pleasure"})
        self.assertEqual(self.kinds("摸摸尾巴"), {"pleasure"})
        self.assertEqual(self.kinds("给你洗澡"), {"clean"})
        self.assertEqual(self.kinds("我要打你"), {"pain"})
        self.assertEqual(self.kinds("摸摸头，给你吃小鱼干"), {"pleasure", "food"})

    def test_ordinary_words_are_not_sensations(self):
        # every one of these used to count (from her real messages)
        for text in ("你喜欢吃什么", "放狗吃菠萝", "可你之前说你最喜欢吃鲱鱼罐头了", "那你要吃鲱鱼罐头吗",
                     "喂，在吗", "喂", "摸摸欧派", "摸摸屁股", "我给你爹掐死了", "今天吃了牛肉面",
                     "great weather today", "the heat", "打开门", "打游戏", "踢足球"):
            self.assertEqual(self.kinds(text), set(), text)

    def test_a_denial_cancels_it(self):
        for text in ("我不会打你的", "别摸你的头", "不会喂你的", "我才不抱你", "don't hit you"):
            self.assertEqual(self.kinds(text), set(), text)

    def test_english_needs_her_as_the_object(self):
        self.assertEqual(self.kinds("let me pet you"), {"pleasure"})
        self.assertEqual(self.kinds("I will feed you"), {"food"})
        self.assertEqual(self.kinds("my path is a pattern"), set())

class TestEmotion(unittest.TestCase):
    def test_from_rest_a_push_is_worth_its_size(self):
        eng = EmotionEngine()
        eng.update(EmotionState(joy=0.5))
        self.assertEqual(eng.state.joy, 0.5)

    def test_pushes_have_diminishing_returns(self):
        eng = EmotionEngine()
        for _ in range(10):
            eng.update(EmotionState(joy=0.3))
        self.assertLess(eng.state.joy, 0.98)
        before = eng.state.joy
        eng.update(EmotionState(joy=0.3))
        self.assertLess(eng.state.joy - before, 0.05)           # nearly full: a push barely moves it

    def test_a_trickle_settles_instead_of_pinning(self):
        """One small push every 3 minutes (what a chatty day gives) used to hold joy at its maximum."""
        eng = EmotionEngine()
        t = 0.0
        eng.last_decay = t
        levels = []
        for _ in range(120):
            t += 180
            eng.decay(t)
            eng.update(EmotionState(joy=0.15))
            levels.append(eng.state.joy)
        self.assertLess(max(levels[40:]), 0.9)
        self.assertGreater(min(levels[40:]), 0.3)

    def test_each_emotion_fades_at_its_own_pace(self):
        eng = EmotionEngine()
        eng.update(EmotionState(joy=0.8, surprise=0.8, sorrow=0.8, anger=0.8))
        eng.decay(eng.last_decay + 180)                          # three minutes
        self.assertLess(eng.state.surprise, 0.25)                # a startle is mostly gone (0.8 -> 0.2)
        self.assertGreater(eng.state.sorrow, 0.7)                # sorrow lingers
        self.assertGreater(eng.state.joy, 0.7)
        self.assertGreater(eng.state.sorrow, eng.state.anger)

    def test_joy_still_halves_in_the_base_half_life(self):
        eng = EmotionEngine(half_life_seconds=1200)
        eng.update(EmotionState(joy=0.8))
        eng.decay(eng.last_decay + 1200)
        self.assertAlmostEqual(eng.state.joy, 0.4, places=3)

class TestNeeds(unittest.TestCase):
    def setUp(self):
        clock.set_clock(SimClock(START))

    def test_hygiene_was_never_lowered_by_anything(self):
        h = Homeostasis()
        for _ in range(48):
            clock.get_clock().advance(HOUR)
            h.tick()
        self.assertGreater(h.hygiene_need, 0.9)                  # awake it only grows ...
        for _ in range(8):
            clock.get_clock().advance(HOUR)
            h.tick(sleeping=True)
        self.assertLess(h.hygiene_need, 0.6)                     # ... and she grooms herself in her sleep
        h.alleviate("hygiene", 0.6)
        self.assertLess(h.hygiene_need, 0.1)

    def test_hunger_and_boredom_grow_slower_asleep(self):
        awake, asleep = Homeostasis(), Homeostasis()
        clock.get_clock().advance(5 * HOUR)
        awake.tick()
        asleep.tick(sleeping=True)
        self.assertLess(asleep.hunger, awake.hunger * 0.5)
        self.assertLess(asleep.curiosity_need, awake.curiosity_need * 0.5)

    def test_tension_eases_with_time_and_stays_in_range(self):
        values = ValueSystem(db_path())
        values.update_tension(0.9)
        values.update_tension(0.9)
        self.assertEqual(values.tension, 1.0)
        values.decay_tension(20 * 60)
        self.assertAlmostEqual(values.tension, 0.5, places=3)
        values.decay_tension(3 * HOUR)
        self.assertLess(values.tension, 0.01)

class TestEnergyBudget(unittest.TestCase):
    def test_a_busy_day_does_not_leave_her_asleep_all_day(self):
        """Steady state: awake drain + the day's replies = what sleep recovers."""
        body = Metabolism()
        cost = body.cognitive_cost * (1 + 75 / 100)                # her mean reply is ~75 characters
        def asleep_share(replies):
            return (24 * body.awake_drain_per_hour + replies * cost) / (24 * (body.awake_drain_per_hour + body.sleep_recovery_per_hour))
        self.assertTrue(0.22 < asleep_share(40) < 0.32)              # a typical day: about 7 h of sleep
        self.assertLess(asleep_share(300), 0.42)
        self.assertLess(asleep_share(600), 0.55)                    # even her busiest day leaves most of it awake

    def test_ordinary_talk_about_food_does_not_feed_or_energise_her(self):
        clock.set_clock(SimClock(START + 2 * HOUR))
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path(), rng=random.Random(0))
        gw.environment.get_context = lambda: {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None,
                                              "is_night": False}
        gw.metabolism.energy = 50.0
        gw.homeostasis.hunger = 0.6
        asyncio.run(gw.hear("你喜欢吃什么，我今天吃了牛肉面", user_id="u1", nickname="小明", addressed=True))
        self.assertLess(gw.metabolism.energy, 50.0)                 # replying costs, nothing was eaten
        self.assertGreater(gw.homeostasis.hunger, 0.55)
        asyncio.run(gw.hear("喂你吃小鱼干", user_id="u1", nickname="小明", addressed=True))
        self.assertLess(gw.homeostasis.hunger, 0.1)
        self.assertGreater(gw.metabolism.energy, 60.0)

    def test_being_washed_lowers_hygiene_and_cheers_her_up(self):
        clock.set_clock(SimClock(START + 2 * HOUR))
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path(), rng=random.Random(0))
        gw.environment.get_context = lambda: {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None,
                                              "is_night": False}
        gw.homeostasis.hygiene_need = 0.7
        asyncio.run(gw.hear("来，给你洗澡", user_id="u1", nickname="小明", addressed=True))
        self.assertLess(gw.homeostasis.hygiene_need, 0.2)
        self.assertGreater(gw.emotion.state.joy, 0.1)

class TestSurprise(unittest.TestCase):
    """Surprise is how unusual an answer time is FOR THAT PERSON, not how far it is from a single average."""
    def ask(self, store, uid, t, latency, rate=0.25):
        store.on_noa_message(uid, t, "reply", True)
        return store.on_user_message(uid, t + latency, 0.0, True, learning_rate=rate)

    def test_someone_who_answers_in_bursts_is_rarely_surprising(self):
        rng = random.Random(3)
        store = UserModelStore(db_path())
        t, fast, n = 0.0, 0, 0
        for i in range(400):
            # real pattern: usually seconds, sometimes after a long break
            latency = rng.lognormvariate(math.log(15), 0.6) if rng.random() < 0.8 else rng.lognormvariate(math.log(3600), 1.0)
            err = self.ask(store, "u1", t, latency)
            t += latency + 60
            if i >= 50:
                n += 1
                fast += err["latency_z"] < -1.5
        self.assertLess(fast / n, 0.2)                              # was ~43% on the real data

    def test_a_real_outlier_is_still_a_surprise(self):
        rng = random.Random(1)
        store = UserModelStore(db_path())
        t = 0.0
        for _ in range(60):
            latency = rng.lognormvariate(math.log(20), 0.3)
            self.ask(store, "u2", t, latency)
            t += latency + 30
        slow = self.ask(store, "u2", t, 4 * HOUR)
        self.assertGreater(slow["latency_z"], 3)
        self.assertGreater(slow["surprise"], 0.5)
        calm = self.ask(store, "u2", t + 5 * HOUR, 20)
        self.assertLess(calm["surprise"], 0.15)

    def test_old_models_without_a_spread_still_work(self):
        store = UserModelStore(db_path())
        m = store.get("u3")
        del m["log_latency_var"]
        err = self.ask(store, "u3", 0.0, 30.0)
        self.assertIn("latency_z", err)
        self.assertIn("log_latency_var", store.get("u3"))

class TestTraitDrift(unittest.TestCase):
    def make(self):
        clock.set_clock(SimClock(START))
        return GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path(), rng=random.Random(0))

    @staticmethod
    def day(valence, arousal, conflicts=0, n=40, searches=0):
        eps = [{"id": i, "kind": "dialogue", "user_id": "u1", "valence": valence, "arousal": arousal,
                "conflict": 1.0 if i < conflicts else 0.0} for i in range(n)]
        eps += [{"id": 1000 + i, "kind": "search", "user_id": "self", "valence": 0.1, "arousal": 0.3, "conflict": 0.0}
                for i in range(searches)]
        return eps

    def run_days(self, gw, eps, days):
        for _ in range(days):
            for _ in range(4):
                gw.narrative.drift_traits(eps)
                clock.get_clock().advance(6 * HOUR)
        return {k: v["value"] for k, v in gw.memory.db.get_persona_traits().items()}

    def test_warm_days_make_her_warmer_but_only_so_far(self):
        gw = self.make()
        before = gw._traits()["affection"]
        after = self.run_days(gw, self.day(0.8, 0.4), 60)["affection"]
        self.assertGreater(after, before + 0.1)
        self.assertLess(after, 0.8)                              # settles; it used to climb to the 0.9 bound
        again = self.run_days(gw, self.day(0.8, 0.4), 30)["affection"]
        self.assertAlmostEqual(again, after, delta=0.01)         # and then it stays

    def test_a_polite_life_does_not_erase_a_defining_trait(self):
        gw = self.make()
        traits = self.run_days(gw, self.day(0.2, 0.3, conflicts=0), 90)
        self.assertGreater(traits["tsundere"], 0.45)             # innate 0.7: it fades a little, never vanishes
        self.assertLess(traits["tsundere"], 0.7)

    def test_nothing_unusual_changes_nothing_much(self):
        gw = self.make()
        before = gw._traits()
        after = self.run_days(gw, self.day(0.15, 0.3, conflicts=2, searches=1), 30)
        for trait in ("affection", "tsundere", "energy", "curiosity"):
            self.assertAlmostEqual(after[trait], before[trait], delta=0.06, msg=trait)

    def test_experience_cannot_take_a_trait_far_from_where_she_began(self):
        gw = self.make()
        traits = self.run_days(gw, self.day(1.0, 1.0, conflicts=40, searches=10), 120)
        innate = {"tsundere": 0.7, "curiosity": 0.8, "affection": 0.5, "energy": 0.6}
        for trait, base in innate.items():
            self.assertLessEqual(abs(traits[trait] - base), 0.26, trait)

    def test_the_daily_limit_still_holds(self):
        gw = self.make()
        before = gw._traits()["affection"]
        for _ in range(8):
            gw.narrative.drift_traits(self.day(1.0, 0.5))
            clock.get_clock().advance(HOUR)
        self.assertLessEqual(gw._traits()["affection"] - before, 0.05 + 1e-9)

if __name__ == "__main__":
    unittest.main()

class TestRestartIsNotAReasonToImproveHerself(unittest.TestCase):
    def test_the_self_check_timer_survives_a_restart(self):
        clock.set_clock(SimClock(START))
        path = db_path()
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=path, rng=random.Random(0))
        gw.last_self_check = clock.now()
        gw.last_evolve_time = clock.now()
        gw._save_state("test")
        clock.get_clock().advance(300)                         # five minutes later the app starts again
        again = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=path, rng=random.Random(0))
        self.assertAlmostEqual(again.last_self_check, START, delta=1)
        self.assertAlmostEqual(again.last_evolve_time, START, delta=1)
        self.assertLess(clock.now() - again.last_self_check, again.self_check_interval)   # not due: no agent run
