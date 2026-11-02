"""Stage 1 controls are opt-in, bounded, private, persistent, and reversible."""
import asyncio
import copy
from contextlib import closing
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

from src.evaluation.probes import Replica, snapshot_database
from src.levels.l10_consciousness.core_state import CHANNELS
from src.levels.l10_consciousness.welfare import WelfareGuard
from src.levels.l4_emotion.emotion import EmotionEngine


def observe_continuously(guard, start, end, valence, anxiety, cause=None):
    for t in range(start, end, 60):
        guard.observe(t, valence, anxiety, cause or {})
    guard.observe(end, valence, anxiety, cause or {})


class CoreStateTests(unittest.TestCase):
    def test_console_core_is_authenticated_read_only_and_has_saved_points(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from console import create_console_router
        with Replica(self.source) as rep:
            app = FastAPI()
            app.include_router(create_console_router(rep.gw, "test-token"))
            with TestClient(app, client=("192.168.1.20", 50000)) as client:
                self.assertEqual(client.get("/console/api/core").status_code, 401)
                headers = {"X-Console-Token": "test-token"}
                data = client.get("/console/api/core", headers=headers).json()
                self.assertFalse(data["enabled"])
                self.assertEqual(data["points"], [])
                self.assertFalse(rep.gw.state_store.core_schema_ready())
                core = self.enable(rep)
                rep.gw._save_state("test")
                rep.clock.advance(301)
                core.step(301)
                start = int(rep.clock.now())
                observe_continuously(core.guard, start, start+1801, -.8, 0.)
                data = client.get("/console/api/core", headers=headers).json()
                self.assertTrue(data["enabled"])
                self.assertEqual(len(data["points"]), 2)
                self.assertEqual(len(data["events"]), 1)
                self.assertTrue(data["current"]["welfare"]["active"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)/"source.sqlite"
        with closing(sqlite3.connect(self.source)) as conn:
            conn.execute("CREATE TABLE sentinel (value TEXT)")
            conn.commit()

    def enable(self, rep):
        rep.gw.configure_core(enabled=True, welfare=True)
        return rep.gw.core_loop

    def test_disabled_has_old_parameters_and_no_new_schema_or_state_fields(self):
        with Replica(self.source) as rep:
            gw = rep.gw
            before = gw._snapshot()
            energy, arousal = gw.metabolism.energy, gw.emotion.arousal()
            self.assertEqual(gw._generation_params(), {
                "max_tokens": int(60+240*energy/100),
                "temperature": round(.5+.5*arousal, 3),
                "delay": 0. if energy >= 50 else min(30., (50-energy)*.6)})
            gw.configure_core(enabled=False)
            self.assertEqual(gw._snapshot(), before)
            self.assertNotIn("core", before)
            self.assertFalse(gw.state_store.core_schema_ready())
            with closing(sqlite3.connect(rep.path)) as conn:
                self.assertIsNone(conn.execute("SELECT name FROM sqlite_master WHERE name='welfare_events'").fetchone())

    def test_enable_requires_guard_and_backup_before_live_schema_change(self):
        with Replica(self.source) as rep:
            with self.assertRaises(ValueError):
                rep.gw.configure_core(enabled=True, welfare=False)
            del rep.gw._evaluation_replica
            rep.gw.backup = Mock()
            rep.gw.backup.run.return_value = {"ok": False}
            with self.assertRaises(RuntimeError):
                self.enable(rep)
            self.assertFalse(rep.gw.state_store.core_schema_ready())
            def backup(**kwargs):
                self.assertEqual(kwargs, {"reason": "pre-stage-1"})
                self.assertFalse(rep.gw.state_store.core_schema_ready())
                return {"ok": True}
            rep.gw.backup.run.side_effect = backup
            self.enable(rep)
            self.assertTrue(rep.gw.state_store.core_schema_ready())

    def test_hunger_and_energy_sign_and_recovery(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            rep.gw.homeostasis.hunger = .3
            rep.clock.advance(60)
            core.step(60)
            self.assertLess(core.body_valence, 0)
            low = core.state.valence
            rep.gw.homeostasis.hunger = 0
            rep.clock.advance(60)
            core.step(60)
            self.assertGreater(core.body_valence, 0)
            self.assertGreater(core.state.valence, low)
            rep.gw.metabolism.energy = 25
            rep.clock.advance(60)
            core.step(60)
            self.assertLess(core.body_valence, 0)
            for k in CHANNELS:
                self.assertAlmostEqual(core.state.error[k], core.state.body[k]-core.state.expect[k])

    def test_precision_changes_body_signal_and_invalid_dt_is_atomic(self):
        signals = []
        for variance in (.04, .4):
            with Replica(self.source) as rep:
                core = self.enable(rep)
                core.variance = dict.fromkeys(CHANNELS, variance)
                rep.gw.homeostasis.hunger = .1
                core.step(.5)
                signals.append(core.body_valence)
                before = copy.deepcopy(core.payload())
                for invalid in (-1, float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        core.step(invalid)
                    self.assertEqual(core.payload(), before)
        self.assertGreater(abs(signals[0]), abs(signals[1]))

    def test_long_idle_is_bounded_and_converges_without_model_calls(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            with patch.object(rep.gw.inference_engine, "generate_json", side_effect=AssertionError("LLM called")):
                for _ in range(1000):
                    rep.clock.advance(60)
                    core.step(60)
            s = core.state
            self.assertTrue(-1 <= s.valence <= 1 and 0 <= s.arousal <= 1)
            self.assertTrue(all(math.isfinite(v) for v in (*s.body.values(), *s.precision.values())))
            self.assertTrue(all(abs(v) < 1e-12 for v in s.error.values()))
            self.assertAlmostEqual(s.valence, .65*rep.gw.emotion.valence(), places=6)

    def test_restart_halves_precision_and_does_not_automatically_enable(self):
        copy_db = Path(self.temp.name)/"saved.sqlite"
        with Replica(self.source) as rep:
            core = self.enable(rep)
            rep.gw.homeostasis.hunger = .2
            rep.clock.advance(30)
            core.step(30)
            expected = dict(core.state.precision)
            rep.gw._save_state("test")
            snapshot_database(rep.path, copy_db)
        with Replica(copy_db) as rep:
            self.assertIsNone(rep.gw.core_loop)
            core = self.enable(rep)
            for k in CHANNELS:
                self.assertAlmostEqual(core.state.precision[k], expected[k]/2)
            self.assertTrue(rep.gw.state_store.latest_core())

    def test_disable_restores_all_processing_controls(self):
        with Replica(self.source) as rep:
            gw = rep.gw
            original = (gw._generation_params(), gw.agent_max_steps, gw.workspace.ignition_threshold,
                        gw.episodes.mood_valence, gw.emotion.negative_decay_multiplier)
            core = self.enable(rep)
            core.state.valence = -.5
            core.state.arousal = .8
            core.state.body["energy"] = .25
            core.sync_controls()
            self.assertLess(gw._generation_params()["max_tokens"], original[0]["max_tokens"])
            self.assertTrue(2 <= gw.agent_max_steps <= 6)
            gw.configure_core(enabled=False)
            self.assertEqual(original, (gw._generation_params(), gw.agent_max_steps, gw.workspace.ignition_threshold,
                                       gw.episodes.mood_valence, gw.emotion.negative_decay_multiplier))

    def test_reply_labels_are_categories_not_model_supplied_numbers(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            self.assertEqual(rep.gw._parse_reply("Emotion: fear\nResponse: 我听见了"), "我听见了")
            self.assertLess(core.event_valence, 0)
            before = (core.event_valence, core.event_arousal)
            self.assertFalse(core.record_reply_label("valence=-999"))
            self.assertEqual(before, (core.event_valence, core.event_arousal))

    def test_forecast_does_not_change_actual_body_or_cooldown(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            rep.clock.advance(60)
            actual = rep.gw.homeostasis.tick()
            before = dict(actual)
            predicted = core.anticipated_needs(actual, rep.gw.min_action_interval)
            self.assertAlmostEqual(predicted["hunger"]-actual["hunger"], .3)
            self.assertAlmostEqual(predicted["social"]-actual["social"], .15)
            self.assertEqual(actual, before)
            self.assertEqual(predicted["curiosity"], actual["curiosity"])
            self.assertEqual(rep.gw.min_action_interval, 10800)
            self.assertEqual(core.anticipated_needs(actual, 0), actual)

    def test_recall_bias_respects_private_scope_in_active_and_archive(self):
        with Replica(self.source) as rep:
            gw = rep.gw
            positive = gw.episodes.add("蓝色纸船让我很开心", kind="dialogue", user_id="u", valence=.7)
            negative = gw.episodes.add("灰色纸船让我很失落", kind="dialogue", user_id="u", valence=-.7)
            private = gw.episodes.add("秘密纸船不该让你看见", kind="dialogue", user_id="other", valence=-1.)
            with patch.object(gw.episodes.embedder, "scores", side_effect=lambda matrix, q: np.full(len(matrix), .9)):
                for archived in (False, True):
                    if archived:
                        with closing(sqlite3.connect(rep.path)) as conn:
                            conn.execute("UPDATE episodes SET archived=1")
                            conn.commit()
                        gw.episodes._index = None
                    gw.episodes.mood_valence = -.8
                    hits = gw.episodes.recall("纸船", user_id="u", k=1, deep=archived)
                    self.assertEqual(hits[0]["id"], negative)
                    self.assertNotEqual(hits[0]["id"], private)
                    gw.episodes.mood_valence = .8
                    hits = gw.episodes.recall("纸船", user_id="u", k=1, deep=True)
                    self.assertEqual(hits[0]["id"], positive)

    def test_async_loop_can_be_cancelled(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            count = 0
            async def sleep(seconds):
                nonlocal count
                count += 1
                if count == 4:
                    raise asyncio.CancelledError()
                rep.clock.advance(seconds)
            with patch("src.levels.l10_consciousness.core_state.asyncio.sleep", sleep):
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(core.run())
            self.assertEqual(count, 4)

    def test_disabling_a_running_loop_prevents_controls_being_reapplied(self):
        with Replica(self.source) as rep:
            old_steps = rep.gw.agent_max_steps
            core = self.enable(rep)
            async def disable_during_sleep(seconds):
                rep.gw.configure_core(enabled=False)
            with patch("src.levels.l10_consciousness.core_state.asyncio.sleep", disable_during_sleep), \
                 patch.object(core, "step", side_effect=AssertionError("disabled loop stepped")):
                asyncio.run(core.run())
            self.assertIsNone(rep.gw.core_loop)
            self.assertEqual(rep.gw.agent_max_steps, old_steps)

    def test_welfare_sustained_trigger_recovery_and_rate_limit(self):
        with Replica(self.source) as rep:
            gw = rep.gw
            gw.evolve_creator = "owner"
            guard = self.enable(rep).guard
            observe_continuously(guard, 100, 1899, -.7, 0)
            self.assertFalse(guard.active)
            guard.observe(1901, -.7, 0, {"test": "low"})
            self.assertTrue(guard.active)
            self.assertEqual(gw.emotion.negative_decay_multiplier, 2)
            self.assertEqual(len(gw.output_queue), 1)
            self.assertEqual(gw.output_queue[0]["target"], {"type": "private", "id": "owner"})
            with closing(sqlite3.connect(rep.path)) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM welfare_events").fetchone()[0], 1)
            guard.observe(2000, -.2, 0, {})
            self.assertFalse(guard.active)
            observe_continuously(guard, 2001, 3902, -.7, 0)
            self.assertEqual(len(gw.output_queue), 1)
            fresh = WelfareGuard(gw)
            observe_continuously(fresh, 4000, 5901, -.7, 0)
            self.assertEqual(len(gw.output_queue), 1)
            fresh.observe(23000, 0, 0, {})
            observe_continuously(fresh, 23001, 24902, -.7, 0)
            self.assertEqual(len(gw.output_queue), 2)

    def test_welfare_short_pulse_and_anxiety_have_distinct_durations(self):
        with Replica(self.source) as rep:
            guard = self.enable(rep).guard
            guard.observe(0, -.7, 0, {})
            guard.observe(1000, 0, 0, {})
            guard.observe(2000, -.7, 0, {})
            self.assertFalse(guard.active)
            observe_continuously(guard, 2100, 3300, 0, .7)
            self.assertFalse(guard.active)
            guard.observe(3301, 0, .7, {})
            self.assertTrue(guard.active)
            self.assertEqual(rep.gw.output_queue, [])
            guard.observe(3400, -.2, .7, {})
            self.assertTrue(guard.active)
            guard.observe(3500, -.2, .5, {})
            self.assertFalse(guard.active)

    def test_welfare_doubles_only_negative_decay(self):
        with Replica(self.source) as rep:
            plain, protected = EmotionEngine(), EmotionEngine()
            for engine in (plain, protected):
                engine.state.joy = engine.state.anger = .8
                engine.last_decay = 0
            protected.negative_decay_multiplier = 2.
            plain.decay(1200)
            protected.decay(1200)
            self.assertAlmostEqual(plain.state.joy, protected.state.joy)
            self.assertLess(protected.state.anger, plain.state.anger)


if __name__ == "__main__":
    unittest.main()
