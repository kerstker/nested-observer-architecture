"""Observed field regressions: full bags, empty hunts, maturity and false stair progress."""
from unittest.mock import Mock

import pytest

from src.levels.l8_agency.mc_skills import Skills
from src.levels.l8_agency.mc_pro import ProPlanner
from test_mc_body_loop import FakeWorld
from test_mc_pro import state


def test_beetroot_from_backpack_becomes_eat_plan_and_selected_food():
    from src.levels.l8_agency.game import GameBridge
    slots = {"hotbar": [{"id": "minecraft:iron_pickaxe", "count": 1, "slot": 0}],
             "main": [{"id": "minecraft:beetroot", "count": 3, "slot": 14}]}
    calls = []
    opened, cursor = [False], [None]
    def rpc(method, args):
        calls.append((method, args))
        if method == "inventory.open":
            opened[0] = True
            return {"opened": True}
        if method == "container.close": opened[0] = False
        if method == "container.state": return {"open": False, "screenOpen": opened[0], "carried": cursor[0]}
        if method == "inventory.click":
            assert opened[0]
            entries = slots["hotbar"] + slots["main"]
            slot = args["slot"]
            old = next((v for v in entries if v["slot"] == slot), None)
            entries = [v for v in entries if v["slot"] != slot]
            if cursor[0]: entries.append(dict(cursor[0], slot=slot))
            cursor[0] = old
            slots["hotbar"] = [v for v in entries if v["slot"] < 9]
            slots["main"] = [v for v in entries if v["slot"] >= 9]
        if method == "vision.describeScene": return {"lookingAt": {"type": "none"}}
        return slots if method == "player.getInventory" else {"ok": True}
    skills = Skills(rpc)
    assert skills._food_count() == 3
    assert not skills.hungry(11)
    action = ProPlanner().survival_plan(state(health=14, food=11, inventory={"minecraft:beetroot": 3}))
    assert action["name"] == "eat" and action["args"]["item"] == "minecraft:beetroot"
    game = GameBridge.__new__(GameBridge)
    from types import SimpleNamespace
    game.skills = SimpleNamespace(pause_until=0)
    game.body = SimpleNamespace(running=False)
    game._rpc = rpc
    assert game._eat(100)
    assert [p["slot"] for m,p in calls if m == "inventory.click"] == [14,8,14]
    assert not opened[0] and cursor[0] is None
    assert not any(m == "inventory.swapSlots" for m,_ in calls)
    assert ("inventory.selectHotbar", {"slot": 8}) in calls
    assert ("control.startUsing", {}) in calls
    assert game.using_food_until == 112


def test_harvesting_planning_and_eating_share_one_vanilla_food_catalog():
    from src.levels.l8_agency.mc_food import FOOD_IDS, FOOD_NAMES
    from src.levels.l8_agency.game import GameBridge
    import src.levels.l8_agency.mc_pro as planning
    skills = Skills(lambda *args: {})
    assert GameBridge.EDIBLE is FOOD_IDS
    assert Skills.FOOD is FOOD_NAMES and planning.FOOD_NAMES is FOOD_NAMES
    for item in FOOD_IDS:
        assert skills._edible(item)
        assert ProPlanner()._food_items(state(inventory={item: 1})) == [(item, 1)]


@pytest.mark.parametrize("item", ["minecraft:poisonous_potato", "minecraft:pufferfish",
                                 "minecraft:rotten_flesh", "minecraft:rabbit_hide",
                                 "minecraft:wheat_seeds", "unknown:beetroot", "unknown:cooked_beef"])
def test_unknown_or_harmful_items_are_not_food_in_any_layer(item):
    from src.levels.l8_agency.game import GameBridge
    skills = Skills(lambda *args: {})
    assert not skills._edible(item)
    assert not ProPlanner()._food_items(state(inventory={item: 1}))
    assert item not in GameBridge.EDIBLE


def test_known_soups_and_fish_are_eatable_but_cooked_food_remains_preferred():
    from src.levels.l8_agency.game import GameBridge
    game = GameBridge.__new__(GameBridge)
    from types import SimpleNamespace
    game.skills = SimpleNamespace(pause_until=0)
    game.body = SimpleNamespace(running=False)
    calls = []
    def rpc(method, args):
        calls.append((method, args))
        if method == "container.state": return {"open": False}
        if method == "vision.describeScene": return {"lookingAt": {"type": "none"}}
        if method == "player.getInventory":
            return {"hotbar": [{"id": "minecraft:chicken", "slot": 0, "count": 1},
                               {"id": "minecraft:cooked_beef", "slot": 1, "count": 1},
                               {"id": "minecraft:beetroot", "slot": 2, "count": 3}]}
        return {"ok": True}
    game._rpc = rpc
    assert game._eat(100)
    assert ("inventory.selectHotbar", {"slot": 1}) in calls
    assert all("minecraft:" + item in game.EDIBLE for item in
               ("beetroot_soup", "rabbit_stew", "mushroom_stew", "cod", "salmon", "tropical_fish", "dried_kelp"))


def test_full_bag_has_priority_over_another_empty_hunt():
    s = state(health=17, food=16, free_slots=0)
    action = ProPlanner().survival_plan(s)
    assert action["name"] == "declutter" and action["args"]["want_free"] == 4
    assert action["args"]["blocks_only"] is True


def test_food_cleanup_never_discards_tools_or_current_task_material(monkeypatch):
    world = FakeWorld()
    items = [{"id": "minecraft:cobblestone", "count": 64, "slot": 0},
             {"id": "minecraft:cobblestone", "count": 64, "slot": 1},
             {"id": "minecraft:dirt", "count": 17, "slot": 2},
             {"id": "minecraft:wooden_pickaxe", "count": 1, "slot": 3},
             {"id": "minecraft:iron_pickaxe", "count": 1, "slot": 4}]
    items += [{"id": "test:quest_" + str(i), "count": 1, "slot": i} for i in range(5, 36)]
    selected, dropped = [0], []
    def rpc(method, args):
        if method == "player.getInventory":
            return {"hotbar": items[:9], "main": items[9:], "selectedSlot": 0}
        if method == "inventory.selectHotbar":
            selected[0] = args["slot"]
        if method == "interact.dropItem":
            dropped.append(items[selected[0]]["id"])
        return world.rpc(method, args)
    skills = Skills(rpc)
    skills.goal_item = "minecraft:dirt"
    skills._smooth_look = Mock()
    skills._walk_to = Mock()
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep", lambda _: None)
    skills.declutter(want_free=4, blocks_only=True)
    assert dropped == ["minecraft:cobblestone"]


def test_empty_hunt_cooldown_switches_strategy_and_allows_other_work():
    s = state(health=17, food=16, free_slots=4, food_search={"hunt_after": 1120, "forage_after": 0})
    assert ProPlanner().survival_plan(s)["name"] == "forage"
    s["food_search"]["forage_after"] = 1045
    assert ProPlanner().survival_plan(s) is None
    s["inventory"] = {"minecraft:wheat": 3}
    assert ProPlanner().survival_plan(s)["args"]["item"] == "minecraft:bread"


def test_empty_hunt_does_not_rescan_animals_during_cooldown():
    rpc = Mock(return_value={})
    skills = Skills(rpc)
    skills.crafter.counts = lambda: {}
    skills.forage = Mock(return_value="探路")
    skills.hunt()
    rpc.reset_mock()
    skills.hunt()
    assert not any(c.args[0] == "perception.entities" for c in rpc.call_args_list)
    assert skills.forage.call_count == 2


def test_food_drop_filter_does_not_treat_dirt_or_poison_as_food(monkeypatch):
    entities = [{"x": 1, "y": 64, "z": 1, "item": {"id": item}} for item in
                ("minecraft:dirt", "minecraft:poisonous_potato", "minecraft:rabbit_hide", "minecraft:bread")]
    skills = Skills(lambda method, args: {"entities": entities})
    skills._walk_to = Mock()
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep", lambda _: None)
    skills._collect_food_drops()
    assert skills._walk_to.call_count == 1


def test_foraging_requires_observed_crop_maturity_and_real_bread(monkeypatch):
    for age in (None, 3, 7):
        world = FakeWorld()
        bag = {}
        crops = [{"id": "minecraft:wheat", "x": x, "y": 64, "z": 1} for x in (1, 2, 3)]
        base = world.rpc
        def rpc(method, args):
            if method == "perception.scan":
                return {"found": crops}
            if method == "perception.blocks" and len(args["positions"]) == 1:
                return {"blocks": [{"id": "minecraft:wheat", "properties": {} if age is None else {"age": age}}]}
            return base(method, args)
        skills = Skills(rpc)
        skills.crafter.counts = lambda: dict(bag)
        def craft(item, n):
            if bag.get("minecraft:wheat", 0) >= 3:
                bag["minecraft:wheat"] -= 3
                bag[item] = n
        skills.crafter.craft = craft
        skills._collect_food_drops = lambda: None
        skills._approach = lambda *a: True
        def harvest(*args):
            bag["minecraft:wheat"] = bag.get("minecraft:wheat", 0) + 1
            return True
        skills._break = Mock(side_effect=harvest)
        skills._walk_to = Mock(return_value=False)
        result = skills.forage()
        assert skills._break.call_count == (3 if age == 7 else 0)
        assert ("实际取得食物 ×1" in result) == (age == 7)


def test_foraging_does_not_claim_food_from_successful_walk_alone():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.crafter.counts = lambda: {}
    skills._walk_to = Mock(return_value=True)
    assert "实际取得食物" not in skills.forage()


def test_stairs_stop_after_two_false_successes_without_descending():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills._dig_step = Mock(return_value=None)
    result = skills.dig_down(10, max_steps=80)
    assert skills._dig_step.call_count == 2
    assert "仅确认 0 级" in result
    assert world.me["y"] == 64


def test_stair_step_targets_tile_center_and_rejects_wrong_height(monkeypatch):
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills._danger_near = lambda cells: None
    skills._break = lambda *args: True
    skills._walk_to = Mock(return_value=True)
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep", lambda _: None)
    result = skills._dig_step([(0, 63, 1)], (0, 63, 1))
    skills._walk_to.assert_called_once_with(.5, 63, 1.5, reach=.15, timeout=6)
    assert "高度" in result
