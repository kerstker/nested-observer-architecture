"""Contracts, independent truth checks and actual native/text-loop integration."""
import asyncio
import itertools
import json
import random
from types import SimpleNamespace

import pytest

from src.levels.l10_consciousness.action_prediction import ActionPrediction, ANSWER, Counts
from src.levels.l8_agency.verified_reasoning import VerifiedReasoner
from src.levels.l8_agency.toolbox import Toolbox, Tool, ToolContext, parse_text_actions
from src.levels.l17_self_improvement.research_gate import Trial, compare_paired
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace
from src.core import clock


@pytest.fixture(autouse=True)
def restore_simulation_clock():
    previous = clock.get_clock()
    try:
        yield
    finally:
        clock.set_clock(previous)


def problem(target=5):
    return {"variables": {"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]},
            "constraints": [{"op": "sum_eq", "vars": ["a", "b"], "value": target},
                            {"op": "all_different", "vars": ["a", "b"]}]}


def test_exact_cache_rechecks_and_preserves_constants():
    solver = VerifiedReasoner()
    first = solver.solve(problem())
    second = solver.solve(problem())
    assert first["status"] == second["status"] == "verified"
    assert second["operations"] < first["operations"]
    assert second["attempts"][0]["tier"] == "L1_exact"
    third = solver.solve(problem(3))
    assert sum(third["assignment"].values()) == 3
    assert third["problem_hash"] != first["problem_hash"]


def test_corrupt_cache_rejected_and_repaired():
    solver = VerifiedReasoner()
    key = solver.solve(problem())["problem_hash"]
    scope, _ = solver.cache[key]
    solver.cache[key] = (scope, {"a": 99, "b": -94})
    result = solver.solve(problem())
    assert result["attempts"][0]["verified"] is False
    assert result["status"] == "verified"
    assert set(result["assignment"].values()) == {2, 3}


def test_neighbor_reuse_is_verified_and_scoped():
    solver = VerifiedReasoner()
    solver.solve(problem(), scope="alice")
    p = problem()
    p["constraints"].append({"op": "sum_le", "vars": ["a"], "value": 3})
    result = solver.solve(p, scope="alice")
    assert result["attempts"][0]["tier"] == "L2_candidate"
    other = solver.solve(p, scope="bob")
    assert other["attempts"][0]["tier"] == "L3_greedy"


def test_negative_cache_is_bounded_and_skips_failed_neighbor():
    solver = VerifiedReasoner(capacity=3)
    solver.solve(problem())
    result = solver.solve(problem(100))
    again = solver.solve(problem(100))
    assert result["status"] == again["status"] == "unsatisfiable"
    assert any(a["tier"] == "L2_candidate" for a in result["attempts"])
    assert not any(a["tier"] == "L2_candidate" for a in again["attempts"])
    for i in range(10):
        solver.solve(problem(i))
    assert len(solver.cache) <= 3 and len(solver.negative) <= 3


def test_verification_and_all_failed_attempts_share_budget():
    solver = VerifiedReasoner()
    for limit in (1, 2, 5, 10):
        result = solver.solve(problem(), operations=limit)
        assert result["status"] == "budget_exhausted"
        assert result["assignment"] is None and result["operations"] <= limit
    solver.solve(problem())
    cached = solver.solve(problem(), operations=2)
    assert cached["assignment"] is None  # even cached output must afford verification


@pytest.mark.parametrize("change", [
    lambda p: p.update(code="print('no')"),
    lambda p: p["variables"].update(a=[True]),
    lambda p: p["constraints"][0].update(value=float("nan")),
    lambda p: p["constraints"][0].update(vars=["a", "a"]),
    lambda p: p["constraints"][0].update(op="eval"),
    lambda p: p["variables"].update(a=[10**20]),
])
def test_untrusted_contract_data_is_never_executed(change):
    p = problem()
    change(p)
    with pytest.raises(ValueError):
        VerifiedReasoner().solve(p)


def test_random_contracts_agree_with_independent_bruteforce():
    rng = random.Random(42)
    solver = VerifiedReasoner()
    for _ in range(80):
        target, upper = rng.randint(-12, 12), rng.randint(-4, 4)
        p = {"variables": {n: [-2, -1, 0, 1, 2] for n in "abc"},
             "constraints": [{"op": "sum_eq", "vars": list("abc"), "value": target},
                             {"op": "sum_le", "vars": ["a"], "value": upper},
                             {"op": "sum_ge", "vars": ["b"], "value": -1},
                             {"op": "all_different", "vars": list("abc")}]}
        truth = [(a, b, c) for a, b, c in itertools.product(range(-2, 3), repeat=3)
                 if a+b+c == target and a <= upper and b >= -1 and len({a, b, c}) == 3]
        result = solver.solve(p)
        assert (result["status"] == "verified") == bool(truth)
        if truth:
            assert tuple(result["assignment"][n] for n in "abc") in truth
        else:
            assert result["status"] == "unsatisfiable"


def model_context(energy=100):
    return SimpleNamespace(metabolism=SimpleNamespace(energy=energy, max_energy=100, is_sleeping=False)), ToolContext("private-id")


def test_forecast_is_frozen_and_scored_before_learning():
    model = ActionPrediction()
    gw, ctx = model_context()
    prediction = model.before_decision(gw, ctx, ["read"])
    frozen = prediction.probabilities
    model.observe(prediction, "read")
    assert prediction.probabilities == frozen
    assert model.report()["metrics"]["action_self"]["log_loss"] == pytest.approx(1.0986122886681098)
    assert dict(model.before_decision(gw, ctx, ["read"]).probabilities[1])["read"] > 1/3


def test_body_and_action_increments_measured_against_paired_baselines():
    model = ActionPrediction()
    for i in range(400):
        gw, ctx = model_context(100 if i % 2 else 5)
        action = "work" if i % 2 else "rest"
        model.observe(model.before_decision(gw, ctx, ["rest", "work"]), action)
        model.observe(model.before_tool(gw, ctx, action), "returned" if action == "work" else "blocked")
    report = model.report()
    assert report["predictive_gain_bits"]["body_to_action"] > .5
    assert report["predictive_gain_bits"]["action_to_outcome"] > .5
    assert report["causal_evidence"] is False


def test_persistence_is_bounded_and_contains_no_conversation_identifiers():
    model = ActionPrediction()
    gw, ctx = model_context()
    model.observe(model.before_decision(gw, ctx, ["read"]), ANSWER)
    state = json.loads(json.dumps(model.to_dict()))
    restored = ActionPrediction()
    restored.load_dict(state)
    assert restored.report()["metrics"] == model.report()["metrics"]
    assert "private-id" not in json.dumps(state)
    c = Counts(capacity=2)
    for i in range(10):
        c.learn(str(i), "read")
    assert len(c.rows) == 2
    restored.load_dict({"version": 1, "models": {"action_self": [["x", {"a": float("nan")}]]}})
    assert all(math_value >= 0 for row in restored.models["action_self"].rows.values() for math_value in row.values())


def test_tool_outcomes_do_not_mislabel_returned_business_failures_as_success():
    gw, ctx = model_context()
    gw.action_prediction = ActionPrediction()
    toolbox = Toolbox(gw, [Tool("business_failure", "", {}, lambda *a: {"success": False})])
    asyncio.run(toolbox.run("business_failure", {}, ctx))
    asyncio.run(toolbox.run("missing", {}, ctx))
    actual = [r["actual"] for r in gw.action_prediction.report()["recent"]]
    assert actual == ["returned", "blocked"]  # transport status, explicitly not task success


def test_response_wrapper_freezes_before_native_or_text_impl_and_isolates_turns():
    gw, ctx = model_context()
    gw.action_prediction = ActionPrediction()
    gw.toolbox = Toolbox(gw, [Tool("read", "", {}, lambda *a: "ok")])
    async def implementation(*args):
        assert gw.action_prediction.metrics["action_self"][0] == 0
        ctx.used.append("read")
        return "done"
    gw._respond_impl = implementation
    assert asyncio.run(GlobalWorkspace._respond(gw, "prompt", "", {}, None, ctx)) == "done"
    assert gw.action_prediction.report()["recent"][-1]["actual"] == "read"


def test_text_protocol_and_owner_tool_permissions():
    gw, _ = model_context()
    gw.owner_id = "owner"
    gw.verified_reasoner = VerifiedReasoner()
    toolbox = Toolbox(gw)
    assert "solve_constraints" not in [t.name for t in toolbox.available(ToolContext("stranger"))]
    name, args = parse_text_actions('Action: solve_constraints ' + json.dumps({"problem": problem()}))[0]
    result = json.loads(asyncio.run(toolbox.run(name, args, ToolContext("console"))))
    assert result["status"] == "verified"


@pytest.mark.parametrize("native", [True, False])
def test_real_workspace_routes_new_solver_and_persists_predictions(native):
    from test_agent_tools import NativeLLM, SequenceLLM, make_gw, hear
    args = {"problem": problem()}
    engine = (NativeLLM([[('solve_constraints', args)], "Thought: 完成\nResponse: 验证通过"])
              if native else SequenceLLM(['Action: solve_constraints ' + json.dumps(args),
                                           "Thought: 完成\nResponse: 验证通过"]))
    gw = make_gw(engine)
    hear(gw, "找两个不同的0到3之间的整数，它们相加等于5", user="console")
    assert gw.verified_reasoner.traces[-1]["status"] == "verified"
    assert gw.action_prediction.report()["recent"][-1]["actual"] == "solve_constraints"
    gw._save_state("research_test")
    state = gw.state_store.latest()["action_prediction"]
    restored = ActionPrediction()
    restored.load_dict(state)
    assert restored.report()["metrics"] == gw.action_prediction.report()["metrics"]


def test_report_schema_allows_reasoning_and_keeps_it_owner_only():
    from test_agent_tools import make_gw, NativeLLM
    gw = make_gw(NativeLLM([]))
    # A mature history must not push the other component beyond the tool cap.
    for _ in range(64):
        gw.verified_reasoner.solve(problem())
        gw.action_prediction.observe(gw.action_prediction.before_tool(gw, ToolContext("console"), "read"), "returned")
    schema = gw.toolbox.tools["look_inside"].parameters
    assert "reasoning" in schema["properties"]["what"]["enum"]
    report = json.loads(asyncio.run(gw.toolbox.run("look_inside", {"what": "reasoning"}, ToolContext("console"))))
    assert report["prediction"]["causal_evidence"] is False
    assert report["reasoning"]["cache_entries"] == 1
    assert "recent" not in report["prediction"]
    public = asyncio.run(gw.toolbox.run("look_inside", {"what": "reasoning"}, ToolContext("public")))
    assert "创造者" in public and "metrics" not in public


def trials(cost=10, correct=True):
    return [Trial(f"p{i}", f"family{i}", seed, correct, cost) for i in range(20) for seed in range(3)]


def test_gate_requires_quality_and_fresh_matched_evidence():
    assert compare_paired(trials(), trials(5), draws=200)["accepted"]
    assert not compare_paired(trials(), trials(5, False), draws=200)["accepted"]
    assert not compare_paired(trials(), trials(5)[:-1], draws=200)["accepted"]
    assert not compare_paired(trials(), trials(5), selection_families=["family1"], draws=200)["accepted"]
    assert not compare_paired(trials(), trials(10), draws=200)["accepted"]


@pytest.mark.parametrize("rows", [trials() + trials()[:1], trials()[:3],
                                   [Trial("p", "f", 0, True, float("nan"))]])
def test_gate_rejects_duplicates_small_samples_and_invalid_cost(rows):
    assert not compare_paired(rows, rows, draws=200)["accepted"]
