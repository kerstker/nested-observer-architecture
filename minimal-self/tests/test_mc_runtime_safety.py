"""Regression checks for real-time control ownership and observed building/quest evidence."""
import threading
import time
from unittest.mock import Mock

from src.levels.l8_agency.mc_body import PlayerBody, _see_through
from src.levels.l8_agency.mc_skills import Skills
from test_mc_body_loop import FakeWorld


def test_failed_input_is_retried_and_never_acknowledged_locally():
    world = FakeWorld()
    body = PlayerBody(world.rpc)
    body._drive(forward=True)
    world.fail_once.append("control.setInput")
    body._flush_inputs()
    assert not body._sent["forward"]
    body._flush_inputs()
    assert world.inputs["forward"]
    body.stop()


def test_cancel_releases_latched_use_before_next_command():
    world = FakeWorld()
    body = PlayerBody(world.rpc)
    result = []
    worker = threading.Thread(target=lambda: result.append(body.use_item(20)))
    try:
        worker.start()
        deadline = time.monotonic() + 2
        while not world.using and time.monotonic() < deadline:
            time.sleep(0.01)
        assert world.using
        body.cancel()
        worker.join(2)
        assert result == [False]
        assert not world.using
    finally:
        body.stop()
        worker.join(2)


def test_lost_observation_cannot_reassert_forward():
    world = FakeWorld()
    reads = 0

    def rpc(method, params):
        nonlocal reads
        if method == "player.getState":
            reads += 1
            if reads > 3:
                return None
        return world.rpc(method, params)

    body = PlayerBody(rpc, now=world.clock.now, sleep=world.clock.sleep)
    try:
        assert not body.walk_to(0.5, 64, 20, timeout=3)
        assert not any(world.inputs.values())
        assert "感知中断" in body.stats()["note"]
    finally:
        body.stop()


def test_mining_overhead_does_not_require_walking_into_air():
    world = FakeWorld()
    world.set_block(0, 67, 0, "minecraft:stone")
    body = PlayerBody(world.rpc, now=world.clock.now, sleep=world.clock.sleep)
    try:
        assert body.mine_block(0, 67, 0)
        assert world.block_id(0, 67, 0) == "minecraft:air"
    finally:
        body.stop()


def test_unknown_ground_and_lava_are_not_walkable():
    for kind in ("unloaded", "lava"):
        world = FakeWorld()
        base = world.rpc

        def rpc(method, params):
            if method == "perception.blocks" and kind == "unloaded":
                return {"blocks": [{"loaded": False} for _ in params["positions"]]}
            return base(method, params)

        if kind == "lava":
            world.set_block(0, 63, 1, "minecraft:lava")
        body = PlayerBody(rpc, now=world.clock.now, sleep=world.clock.sleep)
        try:
            assert not body.walk_to(0.5, 64, 8, timeout=3)
            assert not any(world.inputs.values())
            assert world.me["z"] < 1
        finally:
            body.stop()


def test_solid_grass_and_glass_block_a_crosshair():
    assert not _see_through({"air": False, "id": "minecraft:grass_block"})
    assert not _see_through({"air": False, "id": "minecraft:glass"})


def test_skills_use_shared_body_and_report_finished_action():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.body = Mock()
    skills.body.walk_to.return_value = True
    assert skills._walk_to(1, 64, 1)
    skills.body.walk_to.assert_called_once()
    assert not any(m == "nav.pathTo" for m, _, _ in world.calls)
    skills.tidy = lambda: "整理完成"
    events = []
    skills.on_action = lambda name, args, result: events.append((name, args, result))
    skills.start("tidy")
    skills._thread.join(2)
    assert events == [("tidy", {}, "整理完成")]


def test_failed_chest_read_preserves_last_observed_contents():
    skills = Skills(lambda *a: None)
    skills.places.add("chest", 0, 64, 0, items={"minecraft:diamond": 2})
    skills._read_chest({"x": 0, "y": 64, "z": 0})
    assert skills.places.where_is("minecraft:diamond")[0]["items"]["minecraft:diamond"] == 2


def test_hut_materials_do_not_prove_walls_and_resume_only_missing_cells():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.places.set_home(0, 64, 0)
    assert not any(skills.hut_status().values())
    _, walls, roof, _ = skills._hut_cells()
    for cell in walls + roof:
        world.set_block(*cell, "minecraft:cobblestone")
    status = skills.hut_status()
    assert status["walls"] and status["roof"] and not status["door"]
    missing = roof[-1]
    world.set_block(*missing, "minecraft:air")
    assert not skills.hut_status()["roof"]
    def walk(x, y, z, **kw):
        world.me.update(x=x, y=y, z=z)
        return True
    skills._walk_to = walk
    skills.crafter.counts = lambda: {"minecraft:cobblestone": 64}
    skills.crafter.craft = lambda *a: "缺材料"
    placed = []

    def place(item, x, y, z, **kw):
        assert world.block_id(x,y,z) == "minecraft:air"
        assert world.solid(x,y-1,z) or any(world.solid(x+dx,y,z+dz)
                                         for dx,dz in ((-1,0),(1,0),(0,-1),(0,1)))
        world.set_block(x, y, z, item)
        placed.append((x, y, z))
        return (x, y, z)

    skills.place_at = place
    result = skills.build_hut()
    scaffold = {(4,64+dy,-5+level) for level in range(1,5) for dy in range(level)} | {(3,67,-1)}
    assert len(scaffold) == 11
    assert set(placed) == scaffold | {missing}
    assert len(placed) == 12
    assert set(placed) & set(walls + roof) == {missing}
    assert all(world.block_id(*p) == "minecraft:cobblestone" for p in walls)
    placed.clear()
    skills.build_hut()
    assert placed == []  # Both structure and completed work stairs are resumed, never rebuilt.
    assert "还缺" in result and "门" in result
    assert skills.hut_status()["roof"]


def test_hut_refuses_to_build_on_unloaded_or_unsafe_foundation():
    world = FakeWorld()
    world.set_block(-2, 63, -2, "minecraft:water")
    skills = Skills(world.rpc)
    skills.places.set_home(0, 64, 0)
    skills._walk_to = lambda *a, **kw: True
    skills.place_at = Mock()
    assert "地基不平" in skills.build_hut()
    skills.place_at.assert_not_called()


def test_home_requires_observed_height_and_never_records_surface_as_cave_home():
    world = FakeWorld()
    world.me.update(x=-1167, y=91, z=296)
    skills = Skills(world.rpc)
    skills.places.set_home(-1167, 10, 296)
    skills.places.add("chest", -1166, 10, 296)
    skills.go_to = Mock(return_value="到了目的地附近")
    skills.store = Mock()
    skills.memory.did = Mock()
    assert "还没回到家" in skills.go_home()
    skills.store.assert_not_called()
    skills.memory.did.assert_not_called()


def test_go_to_rejects_travel_success_without_arrival_and_resolves_vertical_route():
    world = FakeWorld()
    world.me.update(x=0, y=91, z=0)
    skills = Skills(world.rpc)
    skills.travel = Mock(return_value="[done] reached")
    skills.dig_down = Mock(return_value="往下挖到 y=91（0级台阶）。")
    assert "还没到目的地" in skills.go_to(0, 10, 0)
    skills.dig_down.assert_called_once_with(10, max_steps=24)


def test_nav_reached_status_without_actual_arrival_is_not_success(monkeypatch):
    world = FakeWorld()
    world.me.update(x=0, y=91, z=0)
    base = world.rpc
    def rpc(method, args):
        if method == "nav.status":
            return {"active": False, "state": "reached", "distance": 0}
        return base(method, args)
    skills = Skills(rpc)
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep", lambda _: None)
    assert not skills._walk_to(0, 10, 0)


def test_skill_adapter_preserves_planner_height_and_mining_route_arguments():
    from types import SimpleNamespace
    from src.levels.l8_agency.mc_extra import tools
    skills = Mock()
    skills.busy.return_value = False
    game = SimpleNamespace(skills=skills, job_running=lambda: False)
    tool = next(t for t in tools() if t.name == "mc_skill")
    tool.handler(SimpleNamespace(game=game), None,
                 {"name": "dig_down", "to_y": -58, "max_steps": 24})
    skills.start.assert_called_with("dig_down", to_y=-58, max_steps=24)
    tool.handler(SimpleNamespace(game=game), None,
                 {"name": "tunnel", "length": 12, "ores": "minecraft:diamond_ore"})
    skills.start.assert_called_with("tunnel", length=12, ores="minecraft:diamond_ore")


def test_inventory_does_not_repeat_store_when_home_chest_is_missing():
    from src.levels.l8_agency.mc_pro import ProPlanner
    from test_mc_pro import state, stacks
    planner = ProPlanner()
    s = state(home={"x": 0, "y": 64, "z": 0}, inventory=stacks(34), chest_near=False)
    action = planner.inventory_plan(s)
    assert (action["kind"], action["name"]) == ("goal", "minecraft:chest")
    s["pos"] = (0, 80, 0)
    assert planner.inventory_plan(s)["name"] == "go_home"


def test_store_does_not_claim_transfer_when_inventory_did_not_change(monkeypatch):
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.places.set_home(0, 64, 0)
    skills.places.add("chest", 1, 64, 0)
    skills.crafter.counts = lambda: {"minecraft:dirt": 64}
    skills._open_chest = lambda p: True
    skills._read_chest = lambda p: None
    skills._keep = lambda item: 0
    skills.memory.did = Mock()
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep", lambda _: None)
    assert "放进箱子：" not in skills.store()
    skills.memory.did.assert_not_called()


def test_home_requires_the_same_dimension():
    world = FakeWorld()
    world.me.update(dimension="minecraft:the_nether")
    skills = Skills(world.rpc)
    skills.places.set_home(0, 64, 0)
    skills.go_to = Mock()
    assert "另一个维度" in skills.go_home()
    skills.go_to.assert_not_called()


def test_legacy_home_without_anchors_never_starts_an_81_block_descent():
    world = FakeWorld()
    world.me.update(x=-1167, y=91, z=296)
    skills = Skills(world.rpc)
    skills.places.set_home(-1167, 10, 296)
    original = dict(skills.places.home)
    skills.go_to = Mock()
    assert "暂缓回家" in skills.go_home()
    assert skills.home_route_status(world.me)["blocked"]
    skills.go_to.assert_not_called()
    assert skills.places.home == original


def test_home_two_failed_routes_stop_repeated_excavation():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.places.set_home(30, 64, 0)
    skills.places.add("chest", 31, 64, 0)
    skills.go_to = Mock(return_value="走不过去")
    assert "还没回到家" in skills.go_home()
    assert "还没回到家" in skills.go_home()
    assert "暂缓回家" in skills.go_home()
    assert skills.go_to.call_count == 2
    assert skills.home_route_status(world.me)["blocked"]


def test_home_partial_progress_still_has_three_attempt_budget():
    world = FakeWorld()
    world.me.update(x=30, z=0)
    skills = Skills(world.rpc)
    skills.places.set_home(0, 64, 0)
    def partial(*args):
        world.me["x"] -= 3
        return "还没到目的地"
    skills.go_to = Mock(side_effect=partial)
    for _ in range(3):
        assert "还没回到家" in skills.go_home()
    assert "暂缓回家" in skills.go_home()
    assert skills.go_to.call_count == 3


def test_blocked_old_home_does_not_capture_every_planner_action():
    from src.levels.l8_agency.mc_pro import ProPlanner
    from test_mc_pro import state
    s = state(home={"x": 0, "y": 10, "z": 0}, tools={"pickaxe": 1},
              inventory={"minecraft:stone_pickaxe": 1},
              home_route={"blocked": True, "camp_retry_after": 0})
    assert ProPlanner().next_action(s)["name"] == "set_home"
    s["home_route"]["camp_retry_after"] = 1100
    action = ProPlanner().next_action(s)
    assert action is None or action["name"] not in ("go_home", "set_home", "build_hut")
    s["home_route"]["camp_retry_after"] = 0
    s["home"] = None
    s["hostiles"] = [{"type": "minecraft:zombie", "distance": 5}]
    assert ProPlanner().home_plan(s) is None


def test_set_home_keeps_old_marker_until_actual_chest_exists():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.places.set_home(0, 10, 0)
    original = dict(skills.places.home)
    skills.crafter.counts = lambda: {}
    skills.crafter.craft = Mock(return_value="缺木板")
    assert "尚未观测到实际箱子" in skills.set_home()
    assert skills.places.home == original
    assert skills._camp_retry_after > time.time()


def test_set_home_rejects_unknown_terrain_without_replacing_old_marker():
    skills = Skills(lambda method, args: {"x": 0, "y": 64, "z": 0} if method == "player.getState" else {})
    skills.places.set_home(0, 10, 0)
    skills.dig_up = Mock(return_value="观测未知")
    assert "还没挖上去" in skills.set_home()
    assert skills.places.home["y"] == 10


def test_verified_surface_camp_preserves_old_home_as_map_spot():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.places.set_home(0, 10, 0)
    skills.crafter.counts = lambda: {"minecraft:chest": 1}
    skills.crafter.craft = Mock(return_value="缺材料")
    def place(item):
        world.set_block(1, 64, 0, item)
        skills.places.add("chest", 1, 64, 0)
        return (1, 64, 0)
    skills.place_near = place
    skills.store = Mock(return_value="存储完成")
    assert "家安在" in skills.set_home()
    assert skills.places.home["y"] == 64
    assert any(p["kind"] == "spot" and p["y"] == 10 for p in skills.places.places)


def test_missing_skill_context_cannot_bypass_active_work_stop_guard():
    from types import SimpleNamespace
    from src.levels.l8_agency.mc_extra import tools
    skills = Mock()
    skills.busy.return_value = True
    skills.current, skills.progress = "forage", "nearby food"
    game = SimpleNamespace(skills=skills, work_stalled=lambda: False)
    tool = next(t for t in tools() if t.name == "mc_skill")
    result = tool.handler(SimpleNamespace(game=game), None, {"name": "stop"})
    assert "先不停" in result
    skills.stop.assert_not_called()
    result = tool.handler(SimpleNamespace(game=game), SimpleNamespace(origin="game",group_id=None), {"name":"stop"})
    assert "先不停" in result
    skills.stop.assert_not_called()
