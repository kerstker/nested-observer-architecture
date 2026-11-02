"""Optimization must preserve quality and comparable evidence (E26 integration)."""
import copy
import unittest

from src.levels.l17_self_improvement.fitness import compare, summarise


def result(*, passed=True, accuracy=.95, seconds=10., seed=0):
    return summarise({seed: [{"id": "task", "passed": passed,
                             "metrics": {"accuracy": accuracy, "seconds": seconds}}]})


class QualityGateTests(unittest.TestCase):
    def compare(self, before, after, target="seconds"):
        return compare(before, after, target, {"accuracy": "higher", "seconds": "lower"})

    def test_faster_with_worse_quality_is_rejected(self):
        report = self.compare(result(), result(accuracy=.8, seconds=5.))
        self.assertEqual(report["verdict"], "worse")
        self.assertEqual(report["metric_regressions"], ["accuracy"])

    def test_new_scenario_pass_does_not_hide_metric_regression(self):
        self.assertEqual(self.compare(result(passed=False), result(accuracy=.8))["verdict"], "worse")

    def test_mixed_metrics_are_not_called_same_without_a_target(self):
        self.assertEqual(self.compare(result(), result(accuracy=.8, seconds=5.), None)["verdict"], "worse")

    def test_preserved_quality_and_lower_cost_is_better(self):
        self.assertEqual(self.compare(result(), result(seconds=5.))["verdict"], "better")
        self.assertEqual(self.compare(result(), result())["verdict"], "same")

    def test_replacing_a_failed_sample_is_not_a_fix(self):
        after = result(seed=1)
        self.assertEqual(self.compare(result(passed=False), after)["verdict"], "worse")

    def test_failed_baseline_cannot_approve_an_improvement(self):
        before = result()
        before["errors"] = ["baseline timeout"]
        self.assertEqual(self.compare(before, result(seconds=5.))["verdict"], "worse")

    def test_missing_quality_measurement_rejects_faster_candidate(self):
        after = result(seconds=5.)
        del after["metrics"]["accuracy"]
        self.assertEqual(self.compare(result(), after)["verdict"], "worse")

    def test_changed_metric_cohort_is_rejected_even_with_same_mean(self):
        before = summarise({0: [{"id": "a", "passed": True, "metrics": {"seconds": 10}},
                                {"id": "b", "passed": True, "metrics": {"seconds": 10}}]})
        after = copy.deepcopy(before)
        after["metric_samples"]["seconds"] = ["a@0"]
        after["metrics"]["seconds"] = 5
        self.assertEqual(self.compare(before, after)["verdict"], "worse")

    def test_duplicate_or_nonfinite_results_cannot_pass(self):
        row = {"id": "a", "passed": True, "metrics": {"seconds": 1}}
        self.assertTrue(summarise({0: [row, row]})["errors"])
        for bad in (float("nan"), float("inf"), None, "not measured"):
            with self.subTest(bad=bad):
                self.assertTrue(result(seconds=bad)["errors"])
                self.assertEqual(self.compare(result(), result(seconds=bad))["verdict"], "worse")

    def test_nonfinite_legacy_summary_and_missing_target_are_rejected(self):
        after = result(seconds=5.)
        after["metrics"]["accuracy"] = float("nan")
        self.assertEqual(self.compare(result(), after)["verdict"], "worse")
        self.assertEqual(self.compare(result(), result(), "missing")["verdict"], "worse")

    def test_empty_evidence_and_invalid_threshold_are_rejected(self):
        self.assertEqual(self.compare(summarise({}), summarise({}), None)["verdict"], "worse")
        for threshold in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compare(result(), result(), min_gain=threshold)

    def test_gate_does_not_mutate_evidence(self):
        before, after = result(), result(seconds=5.)
        originals = copy.deepcopy((before, after))
        self.compare(before, after)
        self.assertEqual((before, after), originals)

    def test_explicit_joint_quality_limits_precede_cost(self):
        directions = {"accuracy": "higher", "seconds": "lower"}
        for accuracy, verdict in ((.949, "better"), (.948, "better"), (.947, "worse")):
            with self.subTest(accuracy=accuracy):
                report = compare(result(), result(accuracy=accuracy, seconds=5.), "seconds", directions,
                                 quality_tolerances={"accuracy": .002})
                self.assertEqual(report["verdict"], verdict)
        report = compare(result(), result(seconds=5.), "seconds", directions,
                         quality_tolerances={"unmeasured_quality": .01})
        self.assertEqual(report["verdict"], "worse")
