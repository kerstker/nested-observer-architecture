"""Post-diamond progression is bounded execution and observation, never a fabricated boss completion."""
from unittest.mock import Mock, patch

from src.levels.l8_agency.mc_pro import ProPlanner
from src.levels.l8_agency.mc_skills import Skills
from src.levels.l8_agency.mc_goal import Planner
from test_mc_body_loop import FakeWorld

M = "minecraft:"


def prepared():
    return {"health": 20, "food": 20, "dimension": M + "overworld", "home": {"x": 0, "y": 64, "z": 0},
            "inventory": {M + "diamond_pickaxe": 1, M + "iron_sword": 1, M + "shield": 1,
                          M + "cooked_beef": 20, M + "cobblestone": 64, M + "torch": 32,
                          M + "golden_helmet": 1},
            "worn": {M + "iron_" + kind for kind in ("helmet", "chestplate", "leggings", "boots")}}


def test_expedition_moves_from_real_materials_to_portal_and_dimension():
    p = ProPlanner()
    s = prepared()
    assert p.expedition_plan(s)["name"] == M + "obsidian"
    s["inventory"][M + "obsidian"] = 10
    assert p.expedition_plan(s)["name"] == M + "flint_and_steel"
    s["inventory"][M + "flint_and_steel"] = 1
    assert p.expedition_plan(s)["name"] == "build_portal"
    s["portal"] = {"x": 9, "y": 65, "z": 0}
    assert p.expedition_plan(s)["name"] == "enter_portal"
    s["dimension"] = M + "the_nether"
    assert p.expedition_plan(s)["args"]["item"] == M + "blaze_rod"
    s["inventory"][M + "blaze_rod"] = 8
    assert p.expedition_plan(s)["name"] == "enter_portal"
    s["dimension"] = M + "overworld"
    assert p.expedition_plan(s)["args"]["item"] == M + "ender_pearl"
    s["inventory"][M + "ender_pearl"] = 16
    assert p.expedition_plan(s)["name"] == M + "ender_eye"
    s["inventory"][M + "ender_eye"] = 16
    assert p.expedition_plan(s)["name"] == M + "bow"
    s["inventory"].update({M + "bow": 1, M + "arrow": 128, M + "water_bucket": 1})
    assert p.expedition_plan(s)["name"] == "end_step"  # a next observed action, never completed modpack


def test_expedition_respects_player_goal_and_missing_equipment():
    p = ProPlanner()
    s = prepared()
    s["goal_by"] = "self"
    assert p.expedition_plan(s) is None
    del s["goal_by"]
    s["worn"].remove(M + "iron_chestplate")
    assert p.expedition_plan(s)["name"] == M + "iron_chestplate"
    s = prepared()
    s["dimension"] = M + "the_nether"
    s["inventory"][M + "cooked_beef"] = 2
    s["portal"] = {"x": 1}
    assert p.expedition_plan(s)["name"] == "enter_portal"


def test_resource_hunting_is_a_real_planner_step():
    ix = Mock()
    ix.name = lambda item: item
    planner = Planner(ix)
    assert planner.next_step(M + "blaze_rod", 2, {}, set())["kind"] == "hunt"
    step = planner.next_step(M + "obsidian", 10, {M + "diamond_pickaxe": 1}, set())
    assert step["kind"] == "gather" and step["tier"] == 3


def test_portal_build_resumes_existing_blocks_and_requires_ignition_evidence():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.places.set_home(0, 64, 0)
    skills._walk_to = lambda *a, **kw: True
    skills._smooth_look = lambda *a, **kw: None
    skills.hold = lambda item: True
    skills.crafter.counts = lambda: {M + "obsidian": 10, M + "cobblestone": 8}
    placed = []

    def place(item, x, y, z, **kw):
        world.set_block(x, y, z, item)
        placed.append((item, x, y, z))
        return x, y, z

    skills.place_at = place
    base = world.rpc

    def rpc(method, params):
        if method == "interact.placeBlock":
            for x in (9, 10):
                for y in (65, 66, 67):
                    world.set_block(x, y, 0, M + "nether_portal")
            return {"result": "SUCCESS"}
        return base(method, params)

    skills.rpc = rpc
    assert "点亮成功" in skills.build_portal()
    assert sum(item == M + "obsidian" for item, *_ in placed) == 10
    count = len(placed)
    assert "已经存在" in skills.build_portal()
    assert len(placed) == count
    assert skills.places.near({"x": 0, "y": 64, "z": 0}, ("portal",), within=20)


def test_a_missing_mob_is_not_counted_as_loot_or_a_boss_kill():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.best_tool = lambda kind: M + "iron_sword"
    skills.crafter.counts = lambda: {}
    result = skills.hunt_resource(M + "ender_pearl")
    assert "尚未取得" in result
    assert not any(m == "interact.attackEntity" for m, *_ in world.calls)


def test_blaze_hunt_refuses_wrong_dimension():
    skills = Skills(FakeWorld().rpc)
    assert "维度不对" in skills.hunt_resource(M + "blaze_rod")


def test_boss_checklist_requires_full_armor_and_real_water():
    p = ProPlanner()
    s = {"inventory": {M + "iron_helmet": 1, M + "bucket": 1}}
    checklist = p.boss_prep_checklist(s)
    assert not checklist[0]["done"]
    assert not checklist[6]["done"]
    s["worn"] = {M + "iron_" + kind for kind in ("helmet", "chestplate", "leggings", "boots")}
    s["inventory"][M + "water_bucket"] = 1
    checklist = p.boss_prep_checklist(s)
    assert checklist[0]["done"] and checklist[6]["done"]
