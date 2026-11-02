import asyncio
import json
import pytest
from src.levels.l8_agency.planning import Planner, transition
from src.levels.l8_agency.toolbox import ToolContext
from src.core import clock


def task():
    return {"initial": {"prepared": 0, "artifact": 0}, "goal": {"artifact": 1}, "actions": [
        {"name": "prepare", "requires": {}, "effects": {"prepared": 1}, "cost": .2},
        {"name": "fast", "requires": {"prepared": 1}, "effects": {"artifact": 1}, "cost": 1},
        {"name": "safe", "requires": {"prepared": 1}, "effects": {"artifact": 1}, "cost": 1.8}]}


def test_multistep_forecast_requires_observed_goal_before_skill_writeback():
    planner = Planner()
    result = planner.plan(task())
    assert [s["action"] for s in result["steps"]] == ["prepare", "fast"]
    assert not planner.skills and not result["executed"]
    first, second = result["steps"]
    feedback = planner.observe(result["ticket"], first["action"], first["before"], first["predicted_after"])
    assert not feedback["observed_goal"] and not planner.skills
    assert planner.observe(result["ticket"], second["action"], second["before"], second["predicted_after"])["observed_goal"]
    assert planner.skills
    with pytest.raises(ValueError):
        planner.observe(result["ticket"], second["action"], second["before"], second["predicted_after"])


def test_observed_failure_changes_selection_and_invalidates_remaining_forecast():
    planner = Planner()
    for _ in range(6):
        p = task();p["initial"]["prepared"] = 1
        plan = planner.plan(p, self_state={"energy": 0})
        step = plan["steps"][0]
        after = step["before"] if step["action"] == "fast" else step["predicted_after"]
        result = planner.observe(plan["ticket"], step["action"], step["before"], after)
        assert result["causal_source"] == "unidentified"
    plan = planner.plan(p, self_state={"energy": 0})
    assert plan["steps"][0]["action"] == "safe"
    assert planner.plan(p, self_state={"energy": 100})["steps"][0]["action"] == "fast"  # independent unseen-state prior
    assert planner.plan(task(), max_nodes=1)["status"] == "unsolved"


def test_ordering_stale_plan_and_state_mismatch_rejected():
    planner = Planner();plan=planner.plan(task())
    with pytest.raises(ValueError):
        planner.observe(plan["ticket"], "fast", task()["initial"], {"prepared": 0, "artifact": 1})
    step=plan["steps"][0]
    with pytest.raises(ValueError):
        planner.observe(plan["ticket"], step["action"], step["before"], {"artifact": 1})


def test_learns_actual_state_dynamics_and_snapshot_is_detached():
    planner=Planner();p=task();p['initial']['prepared']=1
    plan=planner.plan(p,self_state={'energy':0});step=plan['steps'][0]
    planner.observe(plan['ticket'],step['action'],step['before'],step['before'])
    assert planner.plan(p,self_state={'energy':0})['steps'][0]['action']=='safe'
    saved=planner.to_dict()
    restored=Planner();restored.load_dict(saved)
    assert restored.plan(p,self_state={'energy':0})['steps'][0]['action']=='safe'
    saved['effects'][0][1][0][0]['prepared']=999
    assert planner.to_dict()['effects'] != saved['effects']


def test_restored_skills_revalidate_current_initial_state_and_domain():
    planner=Planner()
    plan=planner.plan(task())
    for step in plan["steps"]:
        planner.observe(plan["ticket"], step["action"], step["before"], step["predicted_after"])
    saved=json.loads(json.dumps(planner.to_dict()))
    restored=Planner();restored.load_dict(saved)
    assert restored.skills and not restored.pending
    p=task();p["initial"]["prepared"]=1
    assert len(restored.plan(p)["steps"]) == 1
    other=restored.plan(p,scope="other")
    assert other["status"] == "predicted"
    restored.load_dict({"version": 1,"skills": [None],"beliefs": [["bad",[float('nan'),0]]]})


@pytest.mark.parametrize("field,value", [("max_steps",0),("max_nodes",20001)])
def test_hard_resource_limits(field,value):
    with pytest.raises(ValueError): Planner().plan(task(),**{field:value})


def test_tool_integration_permissions_and_persistence():
    from test_agent_tools import make_gw, NativeLLM
    previous=clock.get_clock()
    try:
        gw=make_gw(NativeLLM([]));ctx=ToolContext("console")
        result=json.loads(asyncio.run(gw.toolbox.run("plan_actions",{"problem":task()},ctx)))
        for step in result["steps"]:
            feedback=json.loads(asyncio.run(gw.toolbox.run("observe_plan",{
                "ticket":result["ticket"],"action":step["action"],"before":step["before"],"after":step["predicted_after"]},ctx)))
        assert feedback["observed_goal"]
        gw._save_state("planning_test")
        assert gw.state_store.latest()["planner"]["skills"]
        assert "plan_actions" not in [t.name for t in gw.toolbox.available(ToolContext("stranger"))]
    finally: clock.set_clock(previous)
