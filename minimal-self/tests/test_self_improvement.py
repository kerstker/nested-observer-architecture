import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock, Clock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace
from src.levels.l17_self_improvement import fitness
from src.levels.l17_self_improvement.agent_bridge import DshAgentBridge
from src.levels.l17_self_improvement.diagnosis import SelfDiagnosis

sys.path.insert(0, os.path.dirname(__file__))
from test_evolve import FAKE_AGENT, _make_repo  # noqa: E402

DIRS = {"calibration_gap": "lower"}

# Stands in for scripts/evaluate_life.py in a fake repository: its results depend on whether the
# fake agent's edit is present, and on NOA_FAKE_MODE (better / worse).
FAKE_EVAL = r'''
import argparse, json, os
p = argparse.ArgumentParser(); p.add_argument("--seed", type=int); p.add_argument("--json"); p.add_argument("--only")
a = p.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")
edited = "fake agent edit" in open(os.path.join(root, "minimal-self", "src", "levels", "l4_emotion", "emotion.py"),
                                   encoding="utf-8").read()
mode = os.environ.get("NOA_FAKE_MODE", "better")
passed = not (edited and mode == "worse")
gap = 0.1 if edited and mode == "better" else 0.3
json.dump({"seed": a.seed, "results": [
    {"id": "metacognitive_calibration", "name": "元认知校准", "passed": passed, "metrics": {"calibration_gap": gap}},
    {"id": "continuity_of_self", "name": "连续的自我", "passed": True, "metrics": {}}]}, open(a.json, "w"))
'''

def run(results_by_seed):
    return fitness.summarise({s: r for s, r in results_by_seed.items()})

def res(passed=True, gap=0.3):
    return [{"id": "m", "name": "元认知校准", "passed": passed, "metrics": {"calibration_gap": gap}},
            {"id": "c", "name": "连续", "passed": True, "metrics": {}}]

class TestFitness(unittest.TestCase):
    def test_a_regression_is_worse_whatever_else_improves(self):
        v = fitness.compare(run({0: res(), 1: res()}), run({0: res(False, 0.01), 1: res(True, 0.01)}),
                            "calibration_gap", DIRS)
        self.assertEqual(v["verdict"], "worse")
        self.assertIn("变差", v["summary"])

    def test_the_target_metric_decides_when_everything_passes(self):
        before = run({0: res(gap=0.3)})
        self.assertEqual(fitness.compare(before, run({0: res(gap=0.2)}), "calibration_gap", DIRS)["verdict"], "better")
        self.assertEqual(fitness.compare(before, run({0: res(gap=0.3)}), "calibration_gap", DIRS)["verdict"], "same")
        self.assertEqual(fitness.compare(before, run({0: res(gap=0.5)}), "calibration_gap", DIRS)["verdict"], "worse")

    def test_a_crashed_evaluation_is_worse(self):
        after = fitness.summarise({}, ["seed 0: 超时"])
        self.assertEqual(fitness.compare(run({0: res()}), after, None, DIRS)["verdict"], "worse")

class TestBridgeWithFitness(unittest.TestCase):
    def _bridge(self):
        repo = tempfile.mkdtemp(prefix="noa_fit_repo_")
        _make_repo(repo)
        os.makedirs(os.path.join(repo, "scripts"))
        with open(os.path.join(repo, "scripts", "evaluate_life.py"), "w", encoding="utf-8") as f:
            f.write(FAKE_EVAL)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "eval"], cwd=repo, check=True)
        fake = os.path.join(repo, "fake_acp.py")
        with open(fake, "w", encoding="utf-8") as f:
            f.write(FAKE_AGENT)
        return DshAgentBridge(repo, data_dir=os.path.join(repo, "data", "evolution"), dsh_exe=sys.executable,
                              dsh_bin=fake, timeout=60, fitness_seeds=(0, 1))

    def tearDown(self):
        os.environ.pop("NOA_FAKE_MODE", None)

    def test_a_better_change_is_offered(self):
        os.environ["NOA_FAKE_MODE"] = "better"
        r = self._bridge().propose("改一下校准", "self", "self_improvement", target_metric="calibration_gap")
        self.assertEqual((r["status"], r["verdict"]), ("tested", "better"), r)
        self.assertIn("calibration_gap 0.3 → 0.1", r["fitness"])

    def test_a_worse_change_is_rejected_and_cannot_be_applied(self):
        os.environ["NOA_FAKE_MODE"] = "worse"
        bridge = self._bridge()
        r = bridge.propose("改一下校准", "self", "self_improvement", target_metric="calibration_gap")
        self.assertEqual((r["status"], r["verdict"]), ("rejected_fitness", "worse"), r)
        self.assertFalse(bridge.apply(r["id"])["ok"])

class FakeBridge:
    def __init__(self, status="tested", verdict="better"):
        self.calls, self.status, self.verdict, self.applied = [], status, verdict, set()

    def propose(self, task, requested_by="unknown", kind="self_evolution", target_metric=None):
        self.calls.append({"task": task, "kind": kind, "target_metric": target_metric, "by": requested_by})
        return {"id": f"p{len(self.calls)}", "status": self.status, "verdict": self.verdict, "files_changed": 1,
                "added": 3, "removed": 1, "tests_total": 140, "tests_failed": 0,
                "fitness": "评估 39/39 → 39/39，calibration_gap 0.4 → 0.1"}

    def list_proposals(self, limit=20):
        return [{"id": p, "status": "applied_pending_restart", "tests_total": 140, "tests_failed": 0}
                for p in self.applied]

def make_gw(bridge):
    clock.set_clock(SimClock(START + 4 * 3600))
    gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"),
                         db_path=os.path.join(tempfile.mkdtemp(prefix="noa_self_imp_"), "s.sqlite"))
    gw.search_tool = StubSearch()
    gw.evolve_enabled, gw.evolve_agent, gw.evolve_creator = True, bridge, "creator"
    gw.metacog.stats["memory"].update(conf=0.9, acc=0.4, n=12)       # she has been overconfident
    return gw

def improve(gw):
    async def go():
        gw._maybe_improve_self(clock.now())
        await asyncio.gather(*list(gw._evolve_tasks))
    asyncio.run(go())

class TestClosedLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        clock.set_clock(Clock())

    def test_she_finds_her_own_problem_from_her_own_data(self):
        gw = make_gw(FakeBridge())
        found = gw.diagnosis.diagnose(gw, clock.now())
        self.assertEqual(found[0]["key"], "overconfident_memory")
        self.assertIn("实际只对了 40%", found[0]["evidence"])
        self.assertIn("calibration_gap", SelfDiagnosis.task_for(found[0]))

    def test_diagnose_propose_measure_report(self):
        bridge = FakeBridge()
        gw = make_gw(bridge)
        improve(gw)
        self.assertEqual(len(bridge.calls), 1)
        call = bridge.calls[0]
        self.assertEqual((call["kind"], call["target_metric"], call["by"]), ("self_improvement", "calibration_gap", "creator"))
        msg = gw.output_queue[-1]
        self.assertEqual(msg["target"], {"type": "private", "id": "creator"})          # only the creator is asked
        self.assertIn("我自己发现", msg["content"])
        self.assertIn("calibration_gap 0.4 → 0.1", msg["content"])
        attempt = gw.diagnosis.attempts()[0]
        self.assertEqual((attempt["status"], attempt["verdict"]), ("tested", "better"))
        goal = next(g for g in gw.goals.active() if g["text"].startswith("改进自己"))
        self.assertEqual(goal["progress"], 0.5)                                            # proposed + measured

    def test_she_does_not_nag_herself(self):
        bridge = FakeBridge()
        gw = make_gw(bridge)
        improve(gw)
        clock.get_clock().advance(2 * 86400)
        gw.last_evolve_time = 0
        improve(gw)
        self.assertEqual(len(bridge.calls), 1)                  # same problem: not again within a week

    def test_a_rejected_change_is_let_go(self):
        gw = make_gw(FakeBridge(status="rejected_fitness", verdict="worse"))
        improve(gw)
        self.assertIn("没采用", gw.output_queue[-1]["content"])
        self.assertFalse([g for g in gw.goals.active() if g["text"].startswith("改进自己")])

    def test_after_applying_she_checks_whether_it_really_helped(self):
        bridge = FakeBridge()
        gw = make_gw(bridge)
        improve(gw)
        bridge.applied.add("p1")                                 # the creator approved it
        gw.metacog.stats["memory"].update(conf=0.6, acc=0.55)    # ...and she lived with it for a while
        clock.get_clock().advance(4 * 86400)
        gw.last_self_check = 0
        gw._maybe_improve_self(clock.now())
        attempt = gw.diagnosis.attempts()[0]
        self.assertIn("真的变好了", attempt["note"])
        self.assertIn("done", [g["status"] for g in gw.goals.history(3)])

    def test_closed_without_creator(self):
        bridge = FakeBridge()
        gw = make_gw(bridge)
        gw.evolve_creator = None
        improve(gw)
        self.assertEqual(bridge.calls, [])

if __name__ == "__main__":
    unittest.main()
