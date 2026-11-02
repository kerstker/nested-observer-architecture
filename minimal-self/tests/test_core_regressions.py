"""Regression cases reproduced against the frozen v2 proposal before repair."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.evaluation.probes import Replica


class CoreRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)/'blank.sqlite'
        with closing(sqlite3.connect(self.source)) as db:
            db.execute('CREATE TABLE sentinel(x)')
            db.commit()

    def enable(self, rep):
        rep.gw.homeostasis.social_need = .5
        rep.gw.configure_core(enabled=True, welfare=True)
        return rep.gw.core_loop

    def tick(self, rep, seconds=60, sleeping=False):
        rep.clock.advance(seconds)
        needs = rep.gw.homeostasis.tick(sleeping=sleeping)
        rep.gw.core_loop.step(seconds)
        return needs

    def lead(self, core, actual):
        return core.anticipated_needs(actual, 10800)['social']-actual['social']

    def test_relief_does_not_hide_future_growth(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            for _ in range(60):
                needs = self.tick(rep)
            self.assertAlmostEqual(self.lead(core, needs), .15)
            rep.gw.homeostasis.alleviate('social', .4)
            needs['social'] = rep.gw.homeostasis.social_need
            self.assertAlmostEqual(self.lead(core, needs), .15)
            for _ in range(5):
                needs = self.tick(rep)
            self.assertLess(core.state.error['social'], 0)
            self.assertAlmostEqual(self.lead(core, needs), .15)

    def test_body_cadence_is_independent_of_core_polling(self):
        for body_interval, core_interval in ((10, .5), (10, 10), (60, 60)):
            with self.subTest(body=body_interval, core=core_interval), Replica(self.source) as rep:
                core = self.enable(rep)
                needs = {'social': rep.gw.homeostasis.social_need}
                for i in range(1, int(120/core_interval)+1):
                    rep.clock.advance(core_interval)
                    if i % int(body_interval/core_interval) == 0:
                        needs = rep.gw.homeostasis.tick()
                    core.step(core_interval)
                self.assertAlmostEqual(self.lead(core, needs), .15)

    def test_forecast_refreshes_at_decision_before_next_core_tick(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            rep.clock.advance(10)
            actual = rep.gw.homeostasis.tick()
            before = dict(actual)
            self.assertAlmostEqual(self.lead(core, actual), .15)
            self.assertEqual(actual, before)
            self.assertEqual(rep.gw.min_action_interval, 10800)

    def test_sleep_and_stationary_observation_replace_old_rate(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            needs = self.tick(rep)
            self.assertAlmostEqual(self.lead(core, needs), .15)
            needs = self.tick(rep, sleeping=True)
            self.assertAlmostEqual(self.lead(core, needs), .06)
            rep.clock.advance(60)
            rep.gw.homeostasis.last_update = rep.clock.now()
            core.step(60)
            self.assertAlmostEqual(self.lead(core, needs), 0)

    def test_missing_observations_clock_rewind_and_restart_reset_trend(self):
        with Replica(self.source) as rep:
            core = self.enable(rep)
            needs = self.tick(rep)
            self.assertAlmostEqual(self.lead(core, needs), .15)
            rep.clock.advance(61)
            self.assertEqual(self.lead(core, needs), 0)
            needs = rep.gw.homeostasis.tick()
            self.assertEqual(self.lead(core, needs), 0)
            needs = self.tick(rep)
            self.assertAlmostEqual(self.lead(core, needs), .15)
            rep.clock.t -= 5
            rep.gw.homeostasis.last_update = rep.clock.now()
            self.assertEqual(self.lead(core, needs), 0)
            needs = self.tick(rep)
            self.assertAlmostEqual(self.lead(core, needs), .15)
            rep.gw.configure_core(enabled=False)
            rep.gw.configure_core(enabled=True, welfare=True)
            self.assertEqual(self.lead(rep.gw.core_loop, needs), 0)

    def test_welfare_gap_is_not_sustained_evidence(self):
        for valence, anxiety, duration in ((-.7, 0, 1800), (0, .7, 1200)):
            with self.subTest(duration=duration), Replica(self.source) as rep:
                guard = self.enable(rep).guard
                start = rep.clock.now()
                guard.observe(start, valence, anxiety, {})
                guard.observe(start+3600, valence, anxiety, {})
                self.assertFalse(guard.active)
                for elapsed in range(60, duration+1, 60):
                    guard.observe(start+3600+elapsed, valence, anxiety, {})
                self.assertFalse(guard.active)
                guard.observe(start+3600+duration+1, valence, anxiety, {})
                self.assertTrue(guard.active)
                guard.observe(start+10000, valence, anxiety, {})
                self.assertTrue(guard.active)

    def test_welfare_clock_rewind_starts_a_new_observation_period(self):
        with Replica(self.source) as rep:
            guard = self.enable(rep).guard
            for t in range(1000, 2201, 60):
                guard.observe(t, -.7, 0, {})
            guard.observe(2000, -.7, 0, {})
            for t in range(2060, 3801, 60):
                guard.observe(t, -.7, 0, {})
            self.assertFalse(guard.active)
            guard.observe(3801, -.7, 0, {})
            self.assertTrue(guard.active)
