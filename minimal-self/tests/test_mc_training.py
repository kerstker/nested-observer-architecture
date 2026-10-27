import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.levels.l8_agency.mc_skills import Skills
from src.levels.l8_agency.mc_training import evaluate, observe
from src.levels.l8_agency.game import GameBridge


def state(**overrides):
    return {"state_valid": True, "inventory_valid": True, "x": 0, "y": 64, "z": 0,
            "dimension": "minecraft:overworld", "inventory": {}, **overrides}


@pytest.mark.parametrize("after,expected", [
    (state(inventory={}), "failed"),
    (state(inventory={"minecraft:torch": 2}), "partial"),
    (state(inventory={"minecraft:torch": 4}), "completed"),
    (state(inventory_valid=False), "unverified"),
    (state(dimension="minecraft:the_nether", inventory={"minecraft:torch": 4}), "unverified"),
])
def test_craft_requires_actual_gain_and_enough_output(after, expected):
    result = evaluate("craft", {"item": "minecraft:torch", "count": 4}, state(), after)
    assert result["status"] == expected
    assert result["completed"] == (expected == "completed")


def test_existing_items_are_not_new_crafting():
    inv = state(inventory={"minecraft:torch": 64})
    assert evaluate("craft", {"item": "minecraft:torch"}, inv, inv)["status"] == "failed"
    assert evaluate("goal", {"item": "minecraft:torch"}, inv, inv)["status"] == "completed"


def test_intermediate_goal_material_is_progress_only_with_observed_recipe_step():
    args = {"item": "minecraft:iron_pickaxe"}
    before, after = state(), state(inventory={"minecraft:iron_ingot": 1})
    assert evaluate("goal", args, before, after)["status"] == "failed"
    step = {"kind": "goal_step", "goal": "minecraft:iron_pickaxe", "item": "minecraft:iron_ingot", "gained": 1}
    assert evaluate("goal", args, before, after, facts=[step])["status"] == "partial"
    step["goal"] = "minecraft:shield"
    assert evaluate("goal", args, before, after, facts=[step])["status"] == "failed"


def test_observed_station_satisfies_single_station_goal_but_not_a_stack():
    facts = [{"kind": "station_available", "item": "minecraft:furnace"}]
    assert evaluate("goal", {"item": "minecraft:furnace"}, state(), state(), facts=facts)["completed"]
    assert not evaluate("goal", {"item": "minecraft:furnace", "count": 2}, state(), state(), facts=facts)["completed"]
    assert evaluate("craft", {"item": "torch"}, state(), state(inventory={"minecraft:torch": 4}))["completed"]


@pytest.mark.parametrize("after", [state(y=10), state(dimension="minecraft:the_end"), state(state_valid=False)])
def test_arrival_requires_observed_height_and_dimension(after):
    assert not evaluate("go_to", {"x": 0, "y": 64, "z": 0}, state(x=20), after)["completed"]


def test_partial_route_and_interrupt_are_not_arrival():
    args = {"x": 20, "y": 64, "z": 0}
    assert evaluate("go_to", args, state(), state(x=10))["status"] == "partial"
    assert evaluate("go_to", args, state(), state(x=20), stopped=True)["status"] == "interrupted"
    assert evaluate("go_to", args, state(), state(x=20), error=True)["status"] == "failed"


def test_walking_to_drop_is_not_pickup():
    result = evaluate("collect_drops", {}, state(), state(x=10), progress=5)
    assert result["status"] == "failed" and not result["has_progress"]
    result = evaluate("collect_drops", {}, state(), state(inventory={"minecraft:diamond": 1}))
    assert result["status"] == "partial" and result["evidence"]["collected_items"] == {"minecraft:diamond": 1}


def test_place_requires_the_requested_target_not_a_nested_support_block():
    args = {"item": "minecraft:torch", "x": 1, "y": 64, "z": 1}
    wrong = {"kind": "placed", "item": "minecraft:cobblestone", "x": 1, "y": 63, "z": 1}
    right = {"kind": "placed", **args}
    assert not evaluate("place", args, state(), state(), facts=[wrong])["completed"]
    assert evaluate("place", args, state(), state(), facts=[wrong, right])["completed"]


def test_inventory_preserves_equipment_but_excludes_identity_and_chat():
    def rpc(method, _):
        if method == "player.getState":
            return {"x": 0, "y": 64, "z": 0, "dimension": "minecraft:overworld", "username": "PRIVATE"}
        return {"hotbar": [], "main": [], "armor": [{"id": "minecraft:iron_helmet", "count": 1}],
                "offhand": {"id": "minecraft:shield", "count": 1}, "chat": "PRIVATE"}
    result = observe(rpc)
    assert result["inventory"] == {"minecraft:iron_helmet": 1, "minecraft:shield": 1}
    assert "PRIVATE" not in json.dumps(result)
    assert observe(lambda *a: None)["inventory_valid"] is False


def test_job_failure_cannot_inherit_previous_excursion_progress():
    skills = Skills(lambda *a: {})
    skills.last_outcome = {"kind": "mine", "progress": 9, "reason": "old"}
    skills.mine = lambda **kw: "附近没找到这些方块。"
    seen = []
    skills.on_action = lambda *a: seen.append((dict(skills.last_outcome), dict(skills.last_action)))
    skills.start("mine", blocks="minecraft:iron_ore")
    skills._thread.join(2)
    assert not skills.busy()
    assert seen[0][0] == {}
    assert seen[0][1]["status"] == "unverified"
    assert not seen[0][1]["has_progress"]


def test_pickup_reply_does_not_claim_items_for_reaching_their_position(monkeypatch):
    def rpc(method, _):
        if method == "player.getInventory":
            return {"hotbar": [], "main": []}
        if method == "perception.entities":
            return {"entities": [{"x": 1, "y": 64, "z": 0, "distance": 1}]}
        return {}
    skills = Skills(rpc)
    skills._walk_to = lambda *a, **kw: True
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep", lambda _: None)
    result = skills.collect_drops()
    assert "未确认物品入包" in result
    assert not GameBridge._action_succeeded("collect_drops", result)


def test_equipped_goal_does_not_craft_a_duplicate_shield(monkeypatch):
    calls = []
    class Planner:
        def __init__(self, index):
            pass
        def next_step(self, item, count, have, stations):
            calls.append(dict(have))
            assert have[item] == count == 1
            assert "minecraft:iron_helmet" not in have
            return {"kind": "done"}
    monkeypatch.setattr("src.levels.l8_agency.mc_goal.Planner", Planner)
    skills = Skills(lambda method, params: {"hotbar": [], "main": [],
                    "offhand": {"id": "minecraft:shield", "count": 1},
                    "armor": [{"id": "minecraft:iron_helmet", "count": 1}]} if method == "player.getInventory" else {})
    skills.index = SimpleNamespace(name=lambda x: x)
    skills.goal_item = "minecraft:shield"
    skills.dress = lambda: None
    skills._stations = lambda: set()
    skills._do = lambda *a: pytest.fail("equipped target must not be crafted again")
    assert skills.goal().startswith("目标完成")
    assert calls


def test_direct_goal_entry_respects_failure_backoff_without_mutating_the_world():
    def rpc(method, _):
        assert method == "player.getState"
        return {"x": 1, "y": 64, "z": 1, "dimension": "minecraft:overworld"}
    skills = Skills(rpc)
    skills.goal_item, skills.goal_count = "minecraft:iron_ingot", 3
    skills.memory.feedback("goal", {"item": "minecraft:iron_ingot", "count": 3}, False,
                           "route blocked", (1, 64, 1), "minecraft:overworld", time.time())
    skills.goal = lambda **kw: pytest.fail("failed goal must not restart during cooldown")
    assert "未开始重复任务" in skills.start("goal")
    assert "未开始重复任务" in skills.start("goal", item="minecraft:iron_ingot", count=3, by="quest")
    assert not skills.busy() and skills.training_status()["completed_total"] == 0


def test_stair_waits_for_actual_landing_after_body_reaches_horizontal_target(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    walking = [False]
    def rpc(method, _):
        if method == "player.getState":
            return {"x": 1.5 if walking[0] else .5, "y": 63 if clock[0] >= .15 else 64, "z": .5}
        return {}
    skills = Skills(rpc)
    skills.body = SimpleNamespace(walk_to=lambda *a, **kw: walking.__setitem__(0, True) or True)
    assert skills._walk_to(1.5, 63, .5, reach=.15)
    assert clock[0] >= .15


def test_landing_wait_does_not_accept_a_player_still_on_the_old_ledge(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    skills = Skills(lambda method, _: {"x": 1.5, "y": 64, "z": .5} if method == "player.getState" else {})
    skills.body = SimpleNamespace(walk_to=lambda *a, **kw: True)
    assert not skills._walk_to(1.5, 63, .5, reach=.15)
    assert 1 <= clock[0] <= 1.1


def test_failed_job_does_not_prevent_queued_job_or_mix_their_evidence():
    skills = Skills(lambda *a: {})
    entered, release = threading.Event(), threading.Event()
    def broken(**kwargs):
        entered.set()
        release.wait(2)
        raise ValueError("broken")
    skills.craft = broken
    skills.forage = lambda: "no food"
    skills.start("craft", item="minecraft:torch")
    assert entered.wait(1)
    skills.start("forage")
    release.set()
    skills._thread.join(3)
    history = skills.training_status()["history"]
    assert [a["name"] for a in history] == ["craft", "forage"]
    assert [a["status"] for a in history] == ["failed", "unverified"]
    assert skills.training_status()["completed_total"] == 2


def test_structured_failure_overrides_a_success_sounding_reply():
    game = GameBridge.__new__(GameBridge)
    game.record_action = lambda *a: None
    game.last_pos = None
    game.skills = SimpleNamespace(last_outcome={}, active_job={"id": 1},
                                 last_action={"id": 1, "name": "craft", "status": "failed", "supported": True})
    game._completed_action("craft", {"item": "minecraft:torch"}, "做好了 torch ×4。")
    assert game._fast_work_after == 0


def recorder():
    path = Path(__file__).resolve().parents[2] / "scripts/record_mc_training.py"
    spec = importlib.util.spec_from_file_location("record_mc_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorder_works_without_overlay_server_and_excludes_console_private_data():
    mod = recorder()
    sample = mod.public_sample({"connected": True, "body": {"tick_rate": 14.9}, "journal": "PRIVATE", "username": "PRIVATE"},
                               {"t": 99.9, "body": {"health": 20, "pos": {"x": 1, "y": 64, "z": 2}}, "rationale": "PRIVATE"}, 100)
    report = mod.summarize([sample], [], [])
    assert report["connected_samples"] == report["fresh_samples"] == 1
    assert report["completion_rate"] is None
    assert report["qualification"] == "not_assessed"
    assert "PRIVATE" not in json.dumps(sample)


def test_stale_snapshot_cannot_be_reported_as_live_health_or_stuck():
    mod = recorder()
    sample = mod.public_sample({"connected": True, "body": {"action": "walk"}}, {"t": 1, "body": {"health": 20}}, 100)
    report = mod.summarize([sample], [], [])
    assert report["fresh_samples"] == 0 and report["minimum_observed_health"] is None
    assert report["review_candidates"] == []
