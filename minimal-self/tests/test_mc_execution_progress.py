"""Human execution regressions: bounded work, visible transactions, safe discovery and real placement."""
import math
from unittest.mock import Mock

from src.levels.l8_agency.mc_body import PlayerBody, MAX_TURN_DEG_S, TURN_ACCEL_S
from src.levels.l8_agency.mc_craft import select_item
from src.levels.l8_agency.mc_skills import Skills
from test_mc_body_loop import FakeWorld


def test_turn_has_real_acceleration_and_a_bounded_speed():
    now = [0.0]
    looks = []
    # an older mod without control.lookTarget: the loop itself turns the camera in bounded steps
    body = PlayerBody(lambda method, params: (looks.append(params) if method == "control.look" else None) or {"ok": True},
                      now=lambda: now[0])
    body._state = {"yaw": 0, "pitch": 0}
    body._dt = 1 / 15
    for _ in range(20):
        body._turn(170, 0)
        now[0] += body._dt
    deltas = [looks[0]["yaw"]] + [(b["yaw"] - a["yaw"] + 180) % 360 - 180 for a, b in zip(looks, looks[1:])]
    speeds = [d / body._dt for d in deltas]
    assert 0 < speeds[0] < speeds[1] < speeds[2]
    assert max(abs(v) for v in speeds) <= MAX_TURN_DEG_S + .03
    assert max(abs(speeds[i] - speeds[i-1]) for i in range(1, 4)) <= MAX_TURN_DEG_S / TURN_ACCEL_S * body._dt + .04
    assert abs(looks[-1]["yaw"] - 170) < 2


def test_server_click_success_without_the_block_is_not_placement_success():
    world = FakeWorld()
    world.hotbar.append({"slot": 1, "id": "minecraft:cobblestone", "count": 1})
    base = world.rpc
    def rpc(method, params):
        if method == "interact.placeBlock":
            return {"result": "SUCCESS", "placed": True}
        return base(method, params)
    body = PlayerBody(rpc, now=world.clock.now, sleep=world.clock.sleep, perf=world.clock.perf)
    try:
        assert not body.place_block(1, 64, 0, item="minecraft:cobblestone", timeout=3)
        assert world.block_id(1, 64, 0) == "minecraft:air"
        assert "未确认" in body.stats()["note"]
    finally:
        body.stop()


def test_low_health_interrupts_excursion_before_any_destruction():
    calls = []
    def rpc(method, params):
        calls.append(method)
        return {"x": 0, "y": 64, "z": 0, "health": 7, "food": 11} if method == "player.getState" else {}
    skills = Skills(rpc)
    result = skills.dig_down(16)
    assert "生命偏低" in result
    assert "interact.breakBlock" not in calls
    assert skills.work_status()["needs_supply"] is True
    assert skills.last_outcome["progress"] == 0
    assert skills.work_status()["return_position"] == {"x": 0, "y": 64, "z": 0}


def test_nested_stair_work_stops_on_shared_block_budget():
    state = {"x": .5, "y": 64.0, "z": .5, "health": 20, "food": 20}
    skills = Skills(lambda method, params: dict(state) if method == "player.getState" else {})
    skills._blocks = lambda cells: ["minecraft:stone"] * len(cells)
    def dig(cells, stand):
        state.update(x=stand[0] + .5, y=stand[1], z=stand[2] + .5)
        skills._work_progress(broken=40)
    skills._dig_step = dig
    result = skills.dig_down(0)
    assert state["y"] == 62
    assert "采掘数量预算" in result
    assert skills.last_outcome["blocks_broken"] == 80
    assert skills._work is None


def test_cave_exploration_walks_observed_ground_without_digging_a_new_tunnel():
    world = FakeWorld()
    for x in range(-65, 66):
        for z in range(-65, 66):
            world.set_block(x, 67, z, "minecraft:stone")
    skills = Skills(world.rpc)
    skills.light_area = Mock(return_value="附近已观测到照明")
    def walk(x, y, z, **kwargs):
        world.me.update(x=x, y=y, z=z)
        return True
    skills._walk_to = walk
    result = skills.explore_cave(max_nodes=4)
    assert "实际到达 4 处通道节点" in result
    assert skills.work_status()["route_points"] == 4
    assert not any(method == "interact.breakBlock" for method, _, _ in world.calls)


def test_unknown_cave_floor_is_never_treated_as_a_safe_path():
    skills = Skills(lambda method, params: {"x": 0, "y": 64, "z": 0, "health": 20, "food": 20} if method == "player.getState" else {})
    skills.light_area = Mock()
    skills._walk_to = Mock()
    assert "实际到达 0 处通道节点" in skills.explore_cave()
    skills._walk_to.assert_not_called()


def test_visible_ore_requires_exposure_and_harvestable_tool():
    world = FakeWorld()
    world.set_block(2, 64, 0, "minecraft:diamond_ore")
    base = world.rpc
    def rpc(method, params):
        if method == "perception.scan":
            return {"found": [{"id": "minecraft:diamond_ore", "x": 2, "y": 64, "z": 0, "exposed": True}]}
        return base(method, params)
    world.can_harvest = False
    skills = Skills(rpc)
    skills._break = Mock(return_value=True)
    skills._approach = Mock(return_value=True)
    assert skills._dig_visible(["minecraft:diamond_ore"], 4.5) == 0
    skills._break.assert_not_called()
    world.can_harvest = True
    assert skills._dig_visible(["minecraft:diamond_ore"], 4.5) == 1
    skills._break.assert_called_once_with(2, 64, 0)


def test_backpack_selection_is_visible_three_clicks_and_preserves_displaced_tool():
    bag = {2: {"id": "minecraft:diamond_sword", "count": 1}, 12: {"id": "minecraft:torch", "count": 16}}
    cursor = [None]
    opened = [False]
    calls = []
    def rpc(method, params):
        calls.append((method, dict(params)))
        if method == "player.getInventory":
            return {"hotbar": [dict(v, slot=k) for k, v in bag.items() if k < 9],
                    "main": [dict(v, slot=k) for k, v in bag.items() if k >= 9], "selectedSlot": 2}
        if method == "inventory.open":
            opened[0] = True
            return {"opened": True}
        if method == "container.state":
            return {"screenOpen": opened[0]}
        if method == "inventory.click":
            assert opened[0]
            old = bag.pop(params["slot"], None)
            if cursor[0]:
                bag[params["slot"]] = cursor[0]
            cursor[0] = old
        if method == "container.close":
            opened[0] = False
        return {"ok": True}
    pauses = []
    assert select_item(rpc, "minecraft:torch", pause=pauses.append)
    assert [p["slot"] for m, p in calls if m == "inventory.click"] == [12, 2, 12]
    assert bag[2]["id"] == "minecraft:torch" and bag[12]["id"] == "minecraft:diamond_sword"
    assert cursor[0] is None and not opened[0]
    assert len(pauses) == 4 and sum(pauses) >= .5
    assert not any(m == "inventory.swapSlots" for m, _ in calls)


def test_house_has_real_windows_gable_eaves_and_support_first_roof_order():
    skills = Skills(lambda *a: {})
    skills.places.set_home(0, 64, 0)
    plan = skills.house_blueprint()
    assert plan["size"] == (5, 6, 7)
    assert len(plan["windows"]) == 4
    assert set(c["role"] for c in plan["cells"]) == {"base", "post", "wall", "window", "roof"}
    _, _, roof, _ = skills._hut_cells()
    assert roof[0][1] == 67 and abs(roof[0][2]) <= 2
    assert max(p[1] for p in roof) == 69
    assert any(abs(p[2]) == 3 for p in roof)
    assert not any(skills.hut_status().values())


def test_new_jobs_and_goal_changes_queue_without_mutating_running_goal():
    import threading
    skills = Skills(lambda m,p: {"health": 20})
    gate, entered = threading.Event(), threading.Event()
    executed = []
    def work():
        entered.set()
        gate.wait(3)
        executed.append(("first", skills.goal_item))
        return "first done"
    skills.forage = work
    skills.goal_item = "minecraft:iron_ingot"
    skills.light_area = lambda: executed.append(("light", skills.goal_item)) or "light done"
    skills.goal = lambda **a: executed.append(("goal", a)) or "goal done"
    skills.start("forage")
    assert entered.wait(1)
    assert "排队" in skills.start("light_area")
    assert "已在队列" in skills.start("light_area")
    assert "排队" in skills.set_goal("minecraft:diamond", 3)
    assert skills.goal_item == "minecraft:iron_ingot" and not skills._stop.is_set()
    assert [j["name"] for j in skills.work_status()["queued"]] == ["light_area", "goal"]
    gate.set()
    skills._thread.join(3)
    assert [j[0] for j in executed] == ["first", "light", "goal"]
    assert executed[-1][1]["item"] == "minecraft:diamond"
    assert not skills.work_status()["active"]


def test_unverified_emergency_cannot_interrupt_and_stop_clears_queue():
    import threading
    skills = Skills(lambda m,p: {"health": 20} if m == "player.getState" else {})
    gate, entered = threading.Event(), threading.Event()
    def work():
        entered.set()
        gate.wait(3)
        return "done"
    skills.forage = work
    skills.escape = Mock(return_value="escape")
    skills.start("forage")
    assert entered.wait(1)
    assert "排队" in skills.start("escape", _emergency=True)
    assert not skills._stop.is_set()
    skills.stop()
    gate.set()
    skills._thread.join(3)
    skills.escape.assert_not_called()
    assert not skills.work_status()["queued"]


def test_route_uses_plan_and_shared_body_and_verifies_actual_height():
    state = {"x": .5, "y": 64., "z": .5}
    calls = []
    path = [{"x": 2.5, "y": 64, "z": .5}, {"x": 8.5, "y": 64, "z": .5}]
    def rpc(m,p):
        calls.append(m)
        return dict(state) if m == "player.getState" else {"path": path, "started": False} if m == "nav.plan" else {}
    skills = Skills(rpc)
    def follow(points, **kwargs):
        assert points == path
        state.update(x=8.5, y=62)
        return True
    skills.body = Mock()
    skills.body.follow_path.side_effect = follow
    assert not skills._walk_to(8.5, 64, .5)
    assert "nav.pathTo" not in calls and "nav.plan" in calls


def test_cave_probe_never_exceeds_bridge_batch_limit():
    sizes = []
    def rpc(m,p):
        if m == "perception.blocks":
            sizes.append(len(p["positions"]))
            assert len(p["positions"]) <= 512
            return {"blocks": [{"id": "minecraft:air"} for _ in p["positions"]]}
        return {}
    assert Skills(rpc)._cave_frontiers({"x": 0,"y": 64,"z": 0}) == []
    assert len(sizes) > 1


def test_measured_darkness_is_not_hidden_by_a_torch_behind_a_wall():
    skills = Skills(lambda m,p: {"x": 0,"y": 64,"z": 0,"blockLight": 0,"skyLight": 15,"isNight": True} if m == "player.getState" else {"found": [{"x": 2,"y": 64,"z": 0}]})
    skills._torch_behind = Mock(return_value=(1,64,0))
    assert "实际补上火把" in skills.light_area()
    skills._torch_behind.assert_called_once()


def test_near_ore_thin_wall_approach_is_limited_and_refuses_lava(monkeypatch):
    import src.levels.l8_agency.mc_skills as module
    state = {"x": .5,"y": 64,"z": .5,"health": 20,"food": 20}
    skills = Skills(lambda m,p: dict(state) if m == "player.getState" else {"blocks": [{"id": "minecraft:iron_ore", "loaded": True, "canHarvest": True}]})
    ore = {"x": 6,"y": 64,"z": 0,"exposed": True}
    skills._break = Mock(return_value=True)
    skills._danger_near = Mock(return_value=False)
    skills._approach = Mock(return_value=True)
    monkeypatch.setattr(module, "first_obstacle", Mock(side_effect=[(2,65,0,"minecraft:stone"),(3,65,0,"minecraft:stone"),None]))
    assert skills._ore_approach(ore)
    assert skills._break.call_count == 2
    skills._break.reset_mock()
    monkeypatch.setattr(module, "first_obstacle", lambda *a: (2,65,0,"minecraft:stone"))
    assert not skills._ore_approach(ore)
    assert skills._break.call_count == 2
    skills._break.reset_mock()
    skills._danger_near.return_value = True
    assert not skills._ore_approach(ore)
    skills._break.assert_not_called()


def test_roof_work_stair_uses_supported_columns_and_never_walks_on_air():
    import time
    world = FakeWorld()
    skills = Skills(world.rpc)
    placed = []
    def place(item,x,y,z,**kwargs):
        assert world.solid(x,y-1,z) or any(world.solid(x+dx,y,z+dz) for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)))
        world.set_block(x,y,z,item)
        placed.append((x,y,z))
        return (x,y,z)
    def walk(x,y,z,**kwargs):
        assert world.solid(math.floor(x),math.floor(y)-1,math.floor(z))
        world.me.update(x=x,y=y,z=z)
        return True
    skills.place_at = place
    skills._walk_to = walk
    ok,count = skills._roof_scaffold((0,64,0),time.monotonic()+20)
    assert ok and count == 11 and world.me["y"] == 68
    assert len(placed) == 11
    world.set_block(4,63,-5,"minecraft:water")
    assert skills._roof_scaffold((0,64,0),time.monotonic()+20) == (False,0)


def test_house_requirements_reports_real_missing_furniture_without_claiming_stages():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.places.set_home(0,64,0)
    world.set_block(1,64,-1,"minecraft:crafting_table")
    report = skills.house_requirements()
    assert report["observed"] and report["furniture_present"]["crafting_table"]
    assert "minecraft:crafting_table" not in report["materials_needed"]
    assert report["materials_needed"]["minecraft:white_bed"] == 1
    assert report["materials_needed"]["minecraft:glass"] == 4
    assert not skills.hut_status()["furniture"]


def test_injured_player_can_collect_nearby_food_without_hunting_or_exploration(monkeypatch):
    world = FakeWorld()
    world.me.update(health=7.6, food=11)
    bag = {}
    base = world.rpc
    def rpc(m,p):
        if m == "perception.entities":
            return {"entities": [{"kind": "item", "x": 2,"y": 64,"z": 0,"item": {"id": "minecraft:bread"}}]}
        return base(m,p)
    skills = Skills(rpc)
    skills.crafter.counts = lambda: dict(bag)
    skills._walk_to = Mock(side_effect=lambda *a,**kw: bag.update({"minecraft:bread": 1}) or True)
    skills.hunt = Mock()
    monkeypatch.setattr("src.levels.l8_agency.mc_skills.time.sleep",lambda _:None)
    assert "实际取得食物 ×1" in skills.forage()
    assert skills._walk_to.call_count == 1
    skills.hunt.assert_not_called()


def test_injured_player_does_not_forage_near_hostiles_or_explore_unknown_food():
    for enemies in ([], [{"kind": "hostile","distance": 6}]):
        world = FakeWorld()
        world.me.update(health=7.6,food=11)
        base=world.rpc
        skills = Skills(lambda m,p: {"entities": enemies} if m == "perception.entities" else base(m,p))
        skills._walk_to=Mock()
        result=skills.forage()
        assert ("危险" in result) if enemies else ("不带伤远行" in result)
        skills._walk_to.assert_not_called()
