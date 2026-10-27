"""Meals finish on observation, never a wall-clock guess or a competing tool's swap."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.levels.l8_agency.game import GameBridge


def meal_game():
    game = GameBridge.__new__(GameBridge)
    calls = []
    game.skills = SimpleNamespace(pause_until=0)
    game.body = SimpleNamespace(running=True, cancel=lambda: calls.append(("body.cancel", {})))
    game.ready = lambda: True
    game.connected = game.in_world = True
    game.health, game.food = 14, 15
    game.hostiles = []
    game.using_food_until = game.blocking_until = game.last_eat = 0
    game._meal = None
    game._food_retry = {}
    state = {"food": 15, "health": 14, "usingItem": True, "selectedSlot": 4}
    inventory = {"hotbar": [{"slot": 4, "id": "minecraft:bread", "count": 1}], "main": []}
    def rpc(method, args):
        calls.append((method, args))
        if method == "container.state":
            return {"open": False}
        if method == "vision.describeScene":
            return {"lookingAt": {"type": "none"}}
        if method == "player.getState":
            return dict(state)
        if method == "player.getInventory":
            return inventory
        return {"ok": True}
    game._rpc = rpc
    return game, state, inventory, calls


def test_slow_consumption_stays_held_past_old_deadline_and_verifies_inventory():
    game, state, inv, calls = meal_game()
    assert game.survive(100) == "eat"
    assert game.survive(102.2) == "eating"
    assert game.survive(106) == "eating"  # e.g. server ticks lag behind wall time
    assert game.using_food_until == 112
    assert sum(m == "control.stopUsing" for m, _ in calls) == 1
    inv["hotbar"][0]["count"] = 0
    state["food"] = 20
    assert game.survive(107) == "eating"
    assert game.last_eat_result["success"] and game.last_eat_result["consumed"] == 1
    assert game.last_eat_result["item"] == "minecraft:bread"
    assert game.using_food_until == 0 and game.skills.pause_until < game._eat_pause_until
    names = [m for m, _ in calls]
    assert names.index("body.cancel") < names.index("nav.stop") < names.index("control.startUsing")


def test_no_consumption_times_out_and_backs_off_instead_of_reporting_finished():
    game, _, _, calls = meal_game()
    game._eat(100)
    assert game.survive(112) == "eat_failed"
    assert not game.last_eat_result["success"]
    assert game.last_eat_result["usingItem"] is True  # using is not consuming
    assert not game._eat(113)
    assert sum(m == "control.startUsing" for m, _ in calls) == 1
    assert game._eat(128)
    assert game.survive(140) == "eat_failed"
    assert game._food_retry["minecraft:bread"] == {"failures": 2, "after": 170}


@pytest.mark.parametrize("damaged", [{}, {"hotbar": []}, {"hotbar": [], "main": None}, {"hotbar": "bad", "main": []}])
def test_partial_or_failed_inventory_read_cannot_prove_consumption(damaged):
    game, _, inv, _ = meal_game()
    game._eat(100)
    inv.clear()
    inv.update(damaged)
    assert game.survive(105) == "eating"
    assert game.survive(112) == "eat_failed"
    assert game.last_eat_result["consumed"] == 0


def test_food_rise_alone_is_not_attributed_to_specific_item_consumption():
    game, state, _, _ = meal_game()
    game._eat(100)
    state["food"] = 16
    game.survive(102)
    assert game.last_eat_result["success"]
    assert game.last_eat_result["consumed"] == 0 and game.last_eat_result["item"] is None


def test_same_slot_swapped_to_weapon_is_a_failure_not_a_meal():
    game, _, inv, _ = meal_game()
    game._eat(100)
    inv["main"] = [{"id": "minecraft:bread", "count": 1, "slot": 12}]
    inv["hotbar"] = [{"id": "minecraft:diamond_sword", "count": 1, "slot": 4}]
    assert game.survive(102) == "eat_failed"
    assert "手持物" in game.last_eat_result["reason"]


def test_meal_gate_allows_reads_and_stops_but_not_work_mutations(monkeypatch):
    game = GameBridge.__new__(GameBridge)
    game.using_food_until = 112
    game.gw = object()
    bridge = Mock(return_value='{"ok":true}')
    monkeypatch.setattr("src.levels.l8_agency.mc_extra.call_bridge", bridge)
    for method in ("player.getState", "player.getInventory", "nav.status", "container.state", "nav.stop", "control.stop"):
        assert game._rpc(method, {}) == {"ok": True}
    for method in ("control.look", "inventory.selectHotbar", "container.click", "nav.pathTo", "control.stopUsing"):
        assert game._rpc(method, {}) is None
    assert bridge.call_count == 6
    game.using_food_until = 0
    assert game._rpc("control.stopUsing", {}) == {"ok": True}


def test_nonurgent_scheduler_and_look_do_not_interfere_with_meal():
    game = GameBridge.__new__(GameBridge)
    game.using_food_until = 112
    game._players = Mock(side_effect=AssertionError("must not look around"))
    assert game.reflex(102) is None
    assert not game.play_due(102)
    assert asyncio.run(game.auto_work(102)) == ""
    assert asyncio.run(game.run_expert_action({"kind": "skill", "name": "craft"}, 102, None)) == "eating"


def test_observation_exception_releases_meal_and_pause():
    game, _, _, _ = meal_game()
    game._eat(100)
    base = game._rpc
    game._rpc = lambda method, args: (_ for _ in ()).throw(RuntimeError("lost read")) if method == "player.getState" else base(method, args)
    assert game.survive(102) == "eat_failed"
    assert game.using_food_until == 0 and game._meal is None
    assert game.skills.pause_until < game._eat_pause_until


def test_nearby_creeper_can_interrupt_meal_for_escape():
    game, _, _, _ = meal_game()
    game._eat(100)
    game.hostiles = [{"type": "minecraft:creeper", "distance": 2, "uuid": "creeper"}]
    game._flee = Mock(return_value="flee")
    assert game.survive(102) == "flee"
    assert game.using_food_until == 0


@pytest.mark.parametrize("failed_method", ["container.state", "inventory.selectHotbar", "control.startUsing"])
def test_preparation_exception_releases_work_and_use(failed_method):
    game, _, _, calls = meal_game()
    base = game._rpc
    def broken(method, args):
        if method == failed_method:
            raise RuntimeError("preparation lost bridge")
        return base(method, args)
    game._rpc = broken
    assert not game._eat(100)
    assert game.using_food_until == 0 and game._meal is None
    assert game.skills.pause_until < game._eat_pause_until
    assert calls[-1][0] == "control.stopUsing"
    assert not game.last_eat_result["success"]


def test_disconnect_releases_local_meal_ownership():
    game, _, _, _ = meal_game()
    game._eat(100)
    game.connected = False
    assert game.survive(102) == ""
    assert game.using_food_until == 0 and game._meal is None
    assert not game.last_eat_result["success"]


def test_food_view_turns_away_from_furnace_and_verifies_safe_crosshair(monkeypatch):
    game, _, _, calls = meal_game()
    base = game._rpc
    views = iter([{"type": "block", "id": "minecraft:furnace"}, {"type": "none"}])
    def rpc(method, args):
        if method == "vision.describeScene":
            return {"lookingAt": next(views)}
        return base(method, args)
    game._rpc = rpc
    monkeypatch.setattr("src.levels.l8_agency.game.time.sleep", lambda _: None)
    assert game._eat(100)
    looks = [args for method, args in calls if method == "control.look"]
    assert len(looks) == 20
    assert looks[9]["pitch"] == -80 and looks[-1]["pitch"] == -35
    assert max(abs(b["pitch"] - a["pitch"]) for a, b in zip(looks, looks[1:])) < 20
    assert calls[-1][0] == "control.startUsing"


@pytest.mark.parametrize("view", [{}, {"type": "entity"}, {"type": "block", "id": "minecraft:furnace"},
                                  {"type": "block", "id": "unknown:stone"}])
def test_no_safe_view_is_bounded_and_does_not_press_use(view, monkeypatch):
    game, _, _, calls = meal_game()
    base = game._rpc
    game._rpc = lambda method, args: {"lookingAt": view} if method == "vision.describeScene" else base(method, args)
    monkeypatch.setattr("src.levels.l8_agency.game.time.sleep", lambda _: None)
    assert not game._eat(100)
    assert not any(method == "control.startUsing" for method, _ in calls)
    assert game._food_retry["minecraft:bread"]["after"] == 115
    assert game.using_food_until == 0


def test_inert_stone_ceiling_allows_food_in_enclosed_mine(monkeypatch):
    game, _, _, _ = meal_game()
    base = game._rpc
    game._rpc = lambda method, args: {"lookingAt": {"type": "block", "id": "minecraft:stone"}} if method == "vision.describeScene" else base(method, args)
    monkeypatch.setattr("src.levels.l8_agency.game.time.sleep", lambda _: None)
    assert game._eat(100)


def test_container_must_confirm_closed_before_use():
    game, _, _, calls = meal_game()
    base = game._rpc
    game._rpc = lambda method, args: {"open": True} if method == "container.state" else base(method, args)
    assert not game._eat(100)
    assert not any(method == "control.startUsing" for method, _ in calls)
    assert "关闭" in game.last_eat_result["reason"]


def test_container_reopening_during_meal_stops_use_and_closes_without_false_success():
    game, _, _, calls = meal_game()
    assert game._eat(100)
    base = game._rpc
    game._rpc = lambda method, args: {"open": True} if method == "container.state" else base(method, args)
    assert game.survive(102) == "eat_failed"
    assert calls[-2][0] == "control.stopUsing" and calls[-1][0] == "container.close"
    assert not game.last_eat_result["success"] and game.using_food_until == 0
