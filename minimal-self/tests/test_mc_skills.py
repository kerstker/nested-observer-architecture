"""Skills of her game body: whole ordinary actions done by code, so she does not think about every swing."""
import time
import unittest

from src.levels.l8_agency.mc_skills import Skills


class FakeWorld:
    """Just enough of her client: a tree, items on the ground, a path that always arrives."""

    def __init__(self, logs=(64, 65, 66, 67), x=5, z=5, items=0):
        self.blocks = {(x, y, z): "minecraft:oak_log" for y in logs}
        self.blocks[(x, max(logs) + 1, z)] = "minecraft:oak_leaves"
        self.items = items
        self.me = {"x": x + 1.5, "y": 64.0, "z": z + 0.5}
        self.calls = []

    def rpc(self, method, params):
        self.calls.append((method, params))
        if method == "player.getInventory":                 # a pickaxe in the bag, as anyone mining has
            return {"hotbar": [{"id": "minecraft:stone_pickaxe", "count": 1, "slot": 0}], "main": [], "selectedSlot": 0}
        if method == "perception.scan":
            if params.get("find"):
                return {"found": [{"id": v, "x": k[0], "y": k[1], "z": k[2], "exposed": True} for k, v in self.blocks.items()
                                  if v in params["find"]]}
            return {"chunks": [{"counts": {"minecraft:oak_log": 4, "minecraft:stripped_oak_log": 1, "minecraft:stone": 900}}]}
        if method == "perception.blocks":
            return {"blocks": [{"id": self.blocks.get((p["x"], p["y"], p["z"]), "minecraft:air" if p["y"] >= 64 else "minecraft:dirt")}
                               for p in params["positions"]]}
        if method == "interact.breakBlock":
            pos = (params["x"], params["y"], params["z"])
            if self.blocks.get(pos, "minecraft:air" if pos[1] >= 64 else "minecraft:dirt") != "minecraft:air":
                self.blocks[(params["x"], params["y"], params["z"])] = "minecraft:air"
                self.items += 1
            return {"started": True}
        if method == "player.getState":
            return dict(self.me)
        if method == "nav.pathTo":
            self.me.update({"x": params["x"], "y": float(params["y"]), "z": params["z"]})
            if self.items and params.get("reachRadius", 2) < 1:
                self.items -= 1                       # walked over one
            return {"started": True}
        if method == "nav.status":
            return {"active": False, "state": "reached", "distance": 0.5}
        if method == "perception.entities":
            return {"entities": [{"x": 6, "y": 64, "z": 6, "distance": 2}] * self.items}
        return {}


def wait(skills, timeout=20):
    end = time.time() + timeout
    while skills.busy() and time.time() < end:
        time.sleep(0.05)


class TestSkills(unittest.TestCase):
    def test_she_chops_a_whole_tree_and_picks_up_the_wood(self):
        world = FakeWorld()
        s = Skills(world.rpc)
        self.assertIn("开始", s.start("chop_tree"))
        wait(s)
        self.assertIn("砍了 4 块", s.last_result)
        self.assertFalse([k for k, v in world.blocks.items() if v.endswith("_log")])
        self.assertEqual(world.blocks[(5, 68, 5)], "minecraft:oak_leaves")       # the leaves are left alone
        self.assertEqual(world.items, 0)                                        # and the logs were picked up
        self.assertFalse([p for m, p in world.calls if m == "interact.breakBlock" and p.get("mode") != "survival"])

    def test_a_trunk_too_high_to_reach_is_left(self):
        world = FakeWorld(logs=tuple(range(64, 76)))
        s = Skills(world.rpc)
        s.start("chop_tree", max_logs=20)
        wait(s)
        self.assertLess(sum(1 for v in world.blocks.values() if v == "minecraft:air"), 12)

    def test_no_tree_no_trouble(self):
        world = FakeWorld()
        world.blocks = {}
        s = Skills(world.rpc)
        s.start("chop_tree")
        wait(s)
        self.assertIn("没有", s.last_result)

    def test_stop_and_status(self):
        world = FakeWorld()
        s = Skills(world.rpc)
        self.assertIn("没在做", s.status())
        self.assertIn("不会", s.start("fly_to_the_moon"))
        s.start("chop_tree")
        s.stop()
        wait(s)
        self.assertIn(("nav.stop", {}), world.calls)
        self.assertFalse(s.busy())

    def test_asking_again_does_not_start_over(self):
        world = FakeWorld()
        s = Skills(world.rpc)
        s.start("chop_tree")
        again = s.start("chop_tree")
        if s.busy():
            self.assertIn("已经在做", again)
        wait(s)
        self.assertIn("砍了 4 块", s.last_result)

    def test_a_tree_out_of_reach_is_walked_to_from_a_free_spot_beside_it(self):
        world = FakeWorld()
        world.me = {"x": 40.0, "y": 64.0, "z": 40.0}                     # far away
        s = Skills(world.rpc)
        s.start("chop_tree")
        wait(s)
        path = [p for m, p in world.calls if m == "nav.pathTo"][0]
        self.assertNotEqual((path["x"], path["z"]), (5, 5))               # not into the trunk itself
        self.assertLessEqual(abs(int(path["x"]) - 5) + abs(int(path["z"]) - 5), 2)
        self.assertIn("砍了 4 块", s.last_result)

    def test_a_tree_she_cannot_reach_is_not_tried_again_right_away(self):
        world = FakeWorld()
        world.me = {"x": 40.0, "y": 64.0, "z": 40.0}
        rpc = world.rpc
        world.rpc = lambda m, p: None if m == "nav.pathTo" else rpc(m, p)
        s = Skills(world.rpc)
        s.start("chop_tree")
        wait(s)
        self.assertIn("换一棵", s.last_result)
        s.start("chop_tree")
        wait(s)
        self.assertIn("都试过了", s.last_result)

    def test_a_branch_up_in_the_canopy_is_not_taken_for_a_tree(self):
        world = FakeWorld(logs=(64, 65, 66), x=9, z=9)                       # a real trunk, a little further away
        world.blocks[(5, 80, 5)] = "minecraft:oak_log"                         # a branch high up, right next to her
        world.me = {"x": 5.5, "y": 64.0, "z": 6.5}
        s = Skills(world.rpc)
        s.start("chop_tree")
        wait(s)
        self.assertIn("砍了 3 块", s.last_result)
        self.assertEqual(world.blocks[(5, 80, 5)], "minecraft:oak_log")

    def test_she_does_not_mine_through_walls(self):
        from src.levels.l8_agency.mc_skills import first_obstacle, sight_cells
        cells = sight_cells((0.5, 65.6, 0.5), (4, 64, 0))
        self.assertTrue(all(c != (4, 64, 0) for c in cells))
        self.assertEqual(cells[0], (0, 65, 0))
        world = FakeWorld(logs=(64,), x=4, z=0)
        world.me = {"x": 0.5, "y": 64.0, "z": 0.5}
        self.assertIsNone(first_obstacle(world.rpc, (0.5, 65.62, 0.5), (4, 64, 0)))
        world.blocks[(2, 64, 0)] = "minecraft:stone"
        world.blocks[(2, 65, 0)] = "minecraft:stone"
        hit = first_obstacle(world.rpc, (0.5, 65.62, 0.5), (4, 64, 0))
        self.assertEqual(hit[3], "minecraft:stone")
        s = Skills(world.rpc)
        self.assertFalse(s._break(4, 64, 0))                                   # the log stays: no mining through stone
        self.assertEqual(world.blocks[(4, 64, 0)], "minecraft:oak_log")
        self.assertFalse([p for m, p in world.calls if m == "interact.breakBlock"])

    def test_leaves_in_the_way_are_knocked_out_first(self):
        world = FakeWorld(logs=(64,), x=4, z=0)
        world.me = {"x": 0.5, "y": 64.0, "z": 0.5}
        world.blocks[(2, 64, 0)] = "minecraft:oak_leaves"
        world.blocks[(2, 65, 0)] = "minecraft:oak_leaves"
        rpc = world.rpc

        def rpc_with_leaves(m, p):                       # leaves break like logs in this fake
            if m == "interact.breakBlock" and world.blocks.get((p["x"], p["y"], p["z"]), "").endswith("leaves"):
                world.calls.append((m, p))
                world.blocks[(p["x"], p["y"], p["z"])] = "minecraft:air"
                return {"started": True}
            return rpc(m, p)
        world.rpc = rpc_with_leaves
        s = Skills(world.rpc)
        s._break(4, 64, 0)
        broke = [(p["x"], p["y"], p["z"]) for m, p in world.calls if m == "interact.breakBlock"]
        self.assertTrue(broke and world.blocks.get(broke[0], "").endswith("air"))
        self.assertIn(broke[0], ((2, 64, 0), (2, 65, 0)))                        # the leaves went first

    def test_turning_is_gradual(self):
        world = FakeWorld()
        world.me.update({"yaw": 0.0, "pitch": 0.0})
        s = Skills(world.rpc)
        t = time.time()
        s._smooth_look(world.me["x"] - 10, world.me["y"] + 1.62, world.me["z"], 0.2)     # a 90-degree turn
        looks = [p for m, p in world.calls if m == "control.look"]
        self.assertGreater(len(looks), 10)
        steps = [abs(b["yaw"] - a["yaw"]) for a, b in zip(looks, looks[1:])]
        self.assertLess(max(steps), 20)                                          # no snap
        self.assertGreater(time.time() - t, 0.2)

    def test_she_mines_the_blocks_she_can_see(self):
        world = FakeWorld(logs=(64,), x=30, z=30)
        world.blocks = {(3, 64, 0): "minecraft:coal_ore", (3, 64, 1): "minecraft:coal_ore", (9, 64, 9): "minecraft:stone"}
        world.me = {"x": 1.5, "y": 64.0, "z": 0.5}
        s = Skills(world.rpc)
        s.start("mine", blocks="minecraft:coal_ore", count=2)
        wait(s)
        self.assertIn("挖了 2 块", s.last_result)
        self.assertEqual(world.blocks[(9, 64, 9)], "minecraft:stone")                # only what was asked
        self.assertIn("要挖什么", s.mine(""))

    def test_a_table_a_mod_renames_once_placed_counts_as_placed(self):
        world = FakeWorld(logs=(64,), x=30, z=30)
        world.me = {"x": 0.5, "y": 64.0, "z": 0.5}
        rpc = world.rpc

        def hands(m, p):
            if m == "player.getInventory":
                return {"hotbar": [{"id": "minecraft:crafting_table", "count": 1, "slot": 0}], "main": [], "selectedSlot": 0}
            if m == "interact.placeBlock":
                world.calls.append((m, p))
                world.blocks[(p["x"], p["y"] + 1, p["z"])] = "visualworkbench:minecraft/crafting_table"
                return {}
            return rpc(m, p)
        s = Skills(hands)
        self.assertEqual(s.place_near("minecraft:crafting_table"), (1, 64, 0))
        self.assertEqual(len([1 for m, _ in world.calls if m == "interact.placeBlock"]), 1)     # one table, not twelve

    def test_the_right_tool_is_taken_and_a_weak_one_is_not_wasted(self):
        world = FakeWorld(logs=(64,), x=3, z=0)
        world.me = {"x": 1.5, "y": 64.0, "z": 0.5}
        world.blocks[(3, 64, 1)] = "minecraft:iron_ore"
        rpc = world.rpc
        bag = {"hotbar": [{"id": "minecraft:wooden_pickaxe", "count": 1, "slot": 0}, {"id": "minecraft:stone_axe", "count": 1, "slot": 1}],
               "main": [], "selectedSlot": 0}
        world.rpc = lambda m, p: bag if m == "player.getInventory" else rpc(m, p)
        s = Skills(world.rpc)
        self.assertTrue(s._break(3, 64, 0))
        self.assertIn(("inventory.selectHotbar", {"slot": 1}), world.calls)              # the axe for the log
        self.assertFalse(s._break(3, 64, 1))                                             # iron ore with a wooden pickaxe: no
        self.assertIn("石镐", s.last_reason)
        self.assertEqual(world.blocks[(3, 64, 1)], "minecraft:iron_ore")

    def test_stairs_down_never_straight_down_and_away_from_lava(self):
        world = FakeWorld(logs=(64,), x=30, z=30)
        world.blocks = {}
        world.me = {"x": 0.5, "y": 64.0, "z": 0.5, "yaw": 0.0}                          # facing +z
        rpc = world.rpc

        def walk(m, p):
            if m == "nav.pathTo":
                world.calls.append((m, p))
                world.me.update({"x": p["x"], "y": float(p["y"]), "z": p["z"]})
                return {"started": True}
            return rpc(m, p)
        world.rpc = walk
        s = Skills(world.rpc)
        self.assertIn("y=60", s.dig_down(to_y=60))
        broke = [(p["x"], p["y"], p["z"]) for m, p in world.calls if m == "interact.breakBlock"]
        self.assertNotIn((0, 63, 0), broke)                                              # not the block under her feet
        self.assertIn((0, 63, 1), broke)
        self.assertEqual([p["z"] for m, p in world.calls if m == "nav.pathTo"], [1.5, 2.5, 3.5, 4.5])
        # lava beside the next step: she turns instead of digging into it
        world.calls.clear()
        world.me = {"x": 0.5, "y": 60.0, "z": 4.5, "yaw": 0.0}
        world.blocks[(1, 59, 5)] = "minecraft:lava"
        s.dig_down(to_y=59)
        self.assertNotIn((0, 59, 5), [(p["x"], p["y"], p["z"]) for m, p in world.calls if m == "interact.breakBlock"])

    def test_dig_down_refuses_when_the_target_is_above_her(self):
        world = FakeWorld()
        world.me = {"x": 0.5, "y": 10.0, "z": 0.5, "yaw": 0.0}
        s = Skills(world.rpc)
        out = s.dig_down(to_y=16)
        self.assertIn("dig_up", out)                                   # tells her the right way out
        self.assertFalse([p for m, p in world.calls if m == "interact.breakBlock"])

    def test_going_for_iron_climbs_to_its_height_first(self):
        world = FakeWorld()
        world.me = {"x": 0.5, "y": 10.0, "z": 0.5, "yaw": 0.0}
        s = Skills(world.rpc)
        climbed = []
        s.mine = lambda *a, **kw: "附近没有找到要挖的方块。"   # nothing in sight
        s.dig_up = lambda y, **kw: climbed.append(y) or "往上挖到 y=16"
        out = s.gather(["minecraft:iron_ore"], 1, y=16)                 # iron is common around y=16
        self.assertEqual(climbed, [16])                                # she climbs there instead of digging at y=10
        self.assertIn("往上", out)

    def test_pillaring_out_of_a_hole(self):
        from src.levels.l8_agency.mc_skills import Skills as _Skills
        placed = []

        class HoleWorld(FakeWorld):
            def __init__(self):
                super().__init__()
                self.me = {"x": 0.5, "y": 11.0, "z": 0.5, "yaw": 0.0}

            def rpc(self, method, params):
                if method == "player.getInventory":
                    return {"hotbar": [{"id": "minecraft:cobblestone", "count": 64, "slot": 0}], "main": [],
                            "selectedSlot": 0}
                if method == "player.getState":
                    return dict(self.me)
                if method == "interact.placeBlock":
                    placed.append((params["x"], params["y"], params["z"], params["face"]))
                    self.me["y"] += 1.0                  # the new block under her feet lifts her
                    return {"placed": True}
                return super().rpc(method, params)
        world = HoleWorld()
        s = Skills(world.rpc)
        s.hold = lambda item: True
        out = s.pillar_up(3)
        self.assertIn("爬高了 3 格", out)
        self.assertEqual(len(placed), 3)
        self.assertTrue(all(f == "up" for *_, f in placed))            # clicks the top of the block below her

    def test_escape_tries_stairs_and_then_pillars(self):
        world = FakeWorld()
        s = Skills(world.rpc)
        s._sky_above = lambda *a, **kw: False
        calls = []
        s.dig_up = lambda y, **kw: calls.append(("dig", y)) or "往上挖到 y=12 停了：前面没有落脚的台阶。"
        s.pillar_up = lambda n=2, **kw: calls.append(("pillar", n)) or "垫方块爬高了 2 格（y=11 → y=13）。"
        out = s.escape(rounds=2)
        self.assertEqual([c[0] for c in calls], ["dig", "pillar", "dig", "pillar"])
        self.assertIn("y=", out)

    def test_cooking_in_a_furnace(self):
        from unittest import mock
        from src.levels.l8_agency import mc_skills
        inv = {0: ["minecraft:raw_iron", 3], 12: ["minecraft:coal", 5]}
        furnace = {}
        cursor = []

        def rpc(m, p):
            if m == "player.getInventory":
                return {"hotbar": [{"id": v[0], "count": v[1], "slot": k} for k, v in inv.items() if k < 9 and v[1]],
                        "main": [{"id": v[0], "count": v[1], "slot": k} for k, v in inv.items() if k >= 9 and v[1]], "selectedSlot": 0}
            if m == "perception.scan":
                return {"pois": [{"id": "minecraft:furnace", "x": 2, "y": 64, "z": 0}]}
            if m == "player.getState":
                return {"x": 0.5, "y": 64.0, "z": 0.5}
            if m == "container.open":
                furnace["open"] = True
                return {}
            if m == "container.state":
                if furnace.get(0) and furnace.get(1):                     # one item cooks per look
                    furnace[0][1] -= 1
                    furnace[2] = ["minecraft:iron_ingot", (furnace.get(2) or ["", 0])[1] + 1]
                    if furnace[0][1] == 0:
                        del furnace[0]
                return {"open": bool(furnace.get("open")), "type": "minecraft:furnace",
                        "slots": [{"slot": k, "id": v[0], "count": v[1]} for k, v in furnace.items() if isinstance(k, int)]}
            if m == "container.click":
                slot, button, mode = p["slot"], p["button"], p["mode"]
                if slot == 2 and mode == "quick_move" and furnace.get(2):
                    inv[20] = ["minecraft:iron_ingot", inv.get(20, ["", 0])[1] + furnace.pop(2)[1]]
                elif slot in (0, 1) and button == 1 and cursor:
                    furnace.setdefault(slot, [cursor[0][0], 0])[1] += 1
                    cursor[0][1] -= 1
                elif slot >= 3:
                    i = slot + 6 if slot < 30 else slot - 30
                    if not cursor and inv.get(i):
                        cursor.append(inv.pop(i))
                    elif cursor:
                        inv[i] = cursor.pop()
                return {}
            if m == "container.close":
                furnace["open"] = False
            return {}
        s = Skills(rpc)
        s._approach = lambda *a: True
        s._smooth_look = lambda *a: None
        with mock.patch.object(mc_skills.time, "sleep", lambda t: None):
            out = s.smelt("minecraft:raw_iron", 3)
        self.assertIn("烧好了 3 个", out)
        self.assertEqual(inv[20], ["minecraft:iron_ingot", 3])
        self.assertEqual(inv[12][1], 4)                                                # one coal is enough for three
        self.assertEqual(Skills.furnace_slot(0), 30)
        self.assertEqual(Skills.furnace_slot(9), 3)

    def test_a_full_bag_throws_out_junk_and_spare_tools_first(self):
        bag = [["minecraft:iron_pickaxe", 1], ["minecraft:wooden_pickaxe", 1], ["minecraft:iron_pickaxe", 1], ["minecraft:leaf_litter", 64], ["minecraft:dirt", 64]] + \
              [["minecraft:oak_log", 64]] * 3 + [["minecraft:coal", 9]] + [["minecraft:diamond", 1]] * 27
        inv = {i: list(v) for i, v in enumerate(bag)}
        calls = []

        def rpc(m, p):
            calls.append((m, p))
            if m == "player.getInventory":
                return {"hotbar": [{"id": v[0], "count": v[1], "slot": k} for k, v in inv.items() if k < 9],
                        "main": [{"id": v[0], "count": v[1], "slot": k} for k, v in inv.items() if k >= 9], "selectedSlot": 0}
            if m == "player.getState":
                return {"x": 0.5, "y": 64.0, "z": 0.5, "yaw": 0.0, "pitch": 0.0}
            if m == "perception.blocks":
                return {"blocks": [{"id": "minecraft:air"} for _ in p["positions"]]}
            if m == "inventory.swapSlots":
                a, b = p["slotA"], p["slotB"]
                inv[a], inv[b] = inv.get(b), inv.get(a)
                inv.update({k: v for k, v in list(inv.items()) if v})
                for k in [k for k, v in inv.items() if not v]:
                    del inv[k]
            if m == "inventory.selectHotbar":
                rpc.sel = p["slot"]
            if m == "interact.dropItem":
                inv.pop(rpc.sel, None)
            return {}
        rpc.sel = 0
        s = Skills(rpc)
        s._walk_to = lambda *a, **k: True
        out = s.declutter(want_free=5)
        self.assertEqual(s.free_slots(), 5)
        self.assertIn("leaf_litter", out)
        self.assertIn("wooden_pickaxe", out)                                          # the iron one stays
        left = [v[0] for v in inv.values()]
        self.assertIn("minecraft:iron_pickaxe", left)
        self.assertEqual(left.count("minecraft:oak_log"), 1)                         # one stack of logs is plenty
        self.assertEqual(left.count("minecraft:diamond"), 27)
        self.assertEqual(left.count("minecraft:iron_pickaxe"), 2)                     # a spare of the best is kept

    def test_go_to_someone(self):
        world = FakeWorld()
        s = Skills(world.rpc, who=lambda name: {"x": 20, "y": 64, "z": 3} if name == "Steve" else None)
        s.start("go_to_player", player="Steve")
        wait(s)
        self.assertIn("到 Steve 身边", s.last_result)
        s.start("go_to_player", player="Nobody")
        wait(s)
        self.assertIn("看不到", s.last_result)


if __name__ == "__main__":
    unittest.main()


class TestStationsAndHome(unittest.TestCase):
    """A player goes back to the table they put down, picks a field table up again, hunts on the surface and
    does not treat a marker deep in a mine as home."""

    def skills(self, tmp=None):
        import tempfile, os
        from src.levels.l8_agency.mc_places import Places
        world = FakeWorld(logs=(70,), x=40, z=40)
        s = Skills(world.rpc)
        s.places = Places(os.path.join(tmp or tempfile.mkdtemp(), "places.json"))
        return world, s

    def test_a_remembered_table_out_of_sight_is_walked_to_instead_of_making_a_new_one(self):
        world, s = self.skills()
        world.blocks[(20, 64, 5)] = "visualworkbench:minecraft/crafting_table"
        s.places.add("crafting_table", 20, 64, 5)
        got = s.find_station("crafting_table")
        self.assertEqual((got["x"], got["z"], got["how"]), (20, 5, "remembered"))
        self.assertIn("nav.pathTo", [m for m, _ in world.calls])

    def test_a_remembered_table_that_is_gone_is_forgotten(self):
        world, s = self.skills()
        s.places.add("crafting_table", 20, 64, 5)          # the block there is air now
        self.assertIsNone(s.find_station("crafting_table"))
        self.assertEqual(s.places.near(world.me, ("crafting_table",)), [])

    def test_a_blast_furnace_is_not_a_furnace(self):
        self.assertFalse(Skills._is_station("minecraft:blast_furnace", "furnace"))
        self.assertTrue(Skills._is_station("minecraft:furnace", "furnace"))

    def test_a_table_put_down_in_the_field_is_picked_up_but_one_at_home_stays(self):
        world, s = self.skills()
        world.blocks[(7, 64, 5)] = "minecraft:crafting_table"
        s.places.add("crafting_table", 7, 64, 5)
        self.assertIn("收回", s.put_away_table((7, 64, 5)))
        self.assertEqual(world.blocks[(7, 64, 5)], "minecraft:air")
        world.blocks[(8, 64, 5)] = "minecraft:crafting_table"
        s.places.set_home(8, 64, 6)
        self.assertEqual(s.put_away_table((8, 64, 5)), "")
        self.assertEqual(world.blocks[(8, 64, 5)], "minecraft:crafting_table")

    def test_a_deep_home_marker_without_chest_or_bed_is_not_usable(self):
        world, s = self.skills()
        s.places.set_home(0, 10, 0)
        self.assertTrue(s.home_route_status({"x": 0, "y": 70, "z": 0})["blocked"])
        s.places.add("chest", 1, 10, 0)
        self.assertFalse(s.home_route_status({"x": 0, "y": 70, "z": 0})["blocked"])

    def test_underground_with_no_animals_she_climbs_out_to_hunt(self):
        world, s = self.skills()
        world.me.update(y=20.0)
        climbed = []
        s._sky_above = lambda x, y, z, depth=24: False
        s.dig_up = lambda to_y, max_steps=0, until_sky=False: climbed.append(until_sky) or "挖不上去"
        out = s.hunt()
        self.assertEqual(climbed, [True])
        self.assertIn("还没挖上去", out)


class TestStairStep(unittest.TestCase):
    def test_a_missing_step_is_filled_with_a_cheap_block_only_on_solid_ground(self):
        s = Skills(lambda m, p: {})
        s.crafter.counts = lambda: {"minecraft:cobblestone": 12}
        placed = []
        s.place_at = lambda item, x, y, z, walk=True: placed.append((item, x, y, z)) or (x, y, z)
        world = {(1, 22, 0): "minecraft:stone"}
        s._blocks = lambda cells: [world.get(tuple(c), "minecraft:air") for c in cells]
        self.assertTrue(s._step_block(1, 23, 0))
        self.assertEqual(placed, [("minecraft:cobblestone", 1, 23, 0)])
        world.clear()                                         # nothing under it and nothing over it: a hole
        self.assertFalse(s._step_block(1, 23, 0))
        s.crafter.counts = lambda: {}
        world[(1, 22, 0)] = "minecraft:stone"
        self.assertFalse(s._step_block(1, 23, 0))             # nothing to build with


class TestClimbOutOfAnOpenCave(unittest.TestCase):
    def test_with_no_rock_to_cut_steps_into_she_towers_up_with_blocks(self):
        s = Skills(lambda m, p: {})
        state = {"x": 0.5, "y": 13.0, "z": 0.5, "yaw": 0}
        s._me = lambda: dict(state)
        s._sky_above = lambda x, y, z, depth=24: state["y"] >= 20
        s._blocks = lambda cells: ["minecraft:air"] * len(cells)     # an open cave: air on every side
        towers = []
        def pillar_up(blocks=4, item=""):
            towers.append(blocks)
            state["y"] += 4
            return f"垫方块爬高了 4 格（y={int(state['y']) - 4} → y={int(state['y'])}）。"
        s.pillar_up = pillar_up
        out = s.dig_up(40, max_steps=20, until_sky=True)
        self.assertTrue(towers)
        self.assertGreater(state["y"], 13)
        self.assertNotIn("停了", out)

    def test_she_gives_up_when_even_towering_does_not_lift_her(self):
        s = Skills(lambda m, p: {})
        state = {"x": 0.5, "y": 13.0, "z": 0.5, "yaw": 0}
        s._me = lambda: dict(state)
        s._blocks = lambda cells: ["minecraft:air"] * len(cells)
        s.pillar_up = lambda blocks=4, item="": "垫方块爬高了 0 格（y=13 → y=13）。"
        self.assertIn("停了", s.dig_up(40, max_steps=20))


class TestLevelACamp(unittest.TestCase):
    def skills(self, world):
        s = Skills(lambda m, p: {})
        s._blocks = lambda cells: [world.get(tuple(c), "minecraft:air") for c in cells]
        s.crafter.counts = lambda: {"minecraft:cobblestone": 20}
        def brk(x, y, z, timeout=8.0, depth=0):
            world[(x, y, z)] = "minecraft:air"
            return True
        s._break = brk
        s.place_at = lambda item, x, y, z, walk=True: world.__setitem__((x, y, z), item) or (x, y, z)
        s._sky_above = lambda x, y, z, depth=24: True
        return s

    def test_a_hillside_is_dug_level_and_a_hole_in_the_floor_filled(self):
        world = {(dx, 63, dz): "minecraft:grass_block" for dx in (-1, 0, 1) for dz in (-1, 0, 1)}
        world.update({(dx, 62, dz): "minecraft:stone" for dx in (-1, 0, 1) for dz in (-1, 0, 1)})
        world[(1, 64, 0)] = world[(1, 64, 1)] = world[(1, 65, 1)] = "minecraft:dirt"    # the slope
        del world[(-1, 63, -1)]                                                          # a hole
        s = self.skills(world)
        self.assertFalse(s._camp_site(0, 64, 0))
        self.assertTrue(s._make_camp_site(0, 64, 0))
        self.assertEqual(world[(-1, 63, -1)], "minecraft:cobblestone")

    def test_someones_wall_or_a_tree_is_never_dug_away(self):
        world = {(dx, 63, dz): "minecraft:grass_block" for dx in (-1, 0, 1) for dz in (-1, 0, 1)}
        world[(1, 64, 0)] = "minecraft:oak_log"
        s = self.skills(world)
        self.assertFalse(s._make_camp_site(0, 64, 0))
        self.assertEqual(world[(1, 64, 0)], "minecraft:oak_log")


class TestNoBaritone(unittest.TestCase):
    def test_without_baritone_nothing_is_sent_to_it(self):
        s = Skills(lambda m, p: {} if m != "baritone.status" else {"available": False})
        self.assertFalse(s.baritone())

class TestCampOnAHillside(unittest.TestCase):
    def test_with_no_flat_ground_she_still_puts_a_chest_down_and_calls_it_home(self):
        import tempfile, os
        from src.levels.l8_agency.mc_places import Places
        s = Skills(lambda m, p: {})
        s.places = Places(os.path.join(tempfile.mkdtemp(), "places.json"))
        state = {"x": 0.5, "y": 94.0, "z": 0.5, "health": 20, "food": 15, "dimension": "minecraft:overworld"}
        s._me = lambda: dict(state)
        s._sky_above = lambda x, y, z, depth=24: True
        s._camp_site = lambda x, y, z: False                  # every spot around is a slope
        s._make_camp_site = lambda x, y, z: False             # and too steep to level
        s._blocks = lambda cells: ["minecraft:grass_block"] * len(cells)
        s.crafter.counts = lambda: {"minecraft:chest": 1, "minecraft:crafting_table": 1, "minecraft:furnace": 1}
        placed = []
        s.place_near = lambda item: placed.append(item) or (1, 94, 0)
        s._block = lambda x, y, z: "minecraft:chest"
        s.store = lambda: ""
        s.places.add("chest", 1, 94, 0)
        out = s.set_home()
        self.assertIn("家安在", out)
        self.assertIn("minecraft:chest", placed)
        self.assertIn("临时营地", s.places.home["note"])


class TestFullBag(unittest.TestCase):
    """The live state at 12:35: 35 slots full, raw copper x142 and bowls x64, a chest in the bag, no home."""

    def skills(self, stacks, chest_on_map=False):
        import tempfile, os
        from src.levels.l8_agency.mc_places import Places
        s = Skills(lambda m, p: {})
        s.places = Places(os.path.join(tempfile.mkdtemp(), "places.json"))
        if chest_on_map:
            s.places.add("chest", 1, 64, 0)
        bag = [{"id": i, "count": n, "slot": k} for k, (i, n) in enumerate(stacks)]
        s._inventory = lambda: {"hotbar": bag[:9], "main": bag[9:], "selectedSlot": 0}
        s.free_slots = lambda: 36 - len(bag)
        s._me = lambda: {"x": 0.5, "y": 64.0, "z": 0.5}
        s._blocks = lambda cells: ["minecraft:air"] * len(cells)
        thrown = []
        s.rpc = lambda m, p: thrown.append(bag[int(p.get("slot", 0))]["id"]) if m == "inventory.selectHotbar" and False else {}
        s._walk_to = lambda *a, **k: True
        s._smooth_look = lambda *a, **k: None
        return s, bag, thrown

    def test_cheap_surplus_counts_as_something_to_throw_out(self):
        stacks = [("minecraft:raw_copper", 64), ("minecraft:raw_copper", 64), ("minecraft:raw_copper", 14),
                  ("minecraft:bowl", 64)] + [(f"minecraft:thing_{i}", 1) for i in range(31)]
        s, bag, _ = self.skills(stacks)
        dropped = []
        s.rpc = lambda m, p: dropped.append(p) if m == "interact.dropItem" else {}
        out = s.declutter(want_free=4, blocks_only=True)
        self.assertIn("扔掉了", out)
        self.assertIn("bowl", out)
        self.assertIn("raw_copper", out)
        self.assertGreaterEqual(len(dropped), 3)

    def test_a_chest_in_the_bag_is_put_down_and_used_before_anything_is_thrown(self):
        stacks = [("minecraft:chest", 1)] + [(f"minecraft:thing_{i}", 1) for i in range(34)]
        s, bag, _ = self.skills(stacks)
        placed = []
        def place_near(item):
            placed.append(item)
            s.places.add("chest", 1, 64, 0)
            return (1, 64, 0)
        s.place_near = place_near
        stored = []
        s.store = lambda: stored.append(1) or "放进箱子：thing×10。"
        s.free_slots = lambda: 36 - len(bag) + (10 if stored else 0)
        out = s.declutter(want_free=4)
        self.assertEqual(placed, ["minecraft:chest"])
        self.assertTrue(stored)
        self.assertIn("就地放了个箱子", out)


class TestOffThePillar(unittest.TestCase):
    """The live case at 14:29: three cobblestone up a one-wide column, air all round, every plan failing."""

    def world(self):
        blocks = {}
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                blocks[(dx, 89, dz)] = "minecraft:dirt"              # the ground
        for y in (90, 91, 92):
            blocks[(0, y, 0)] = "minecraft:cobblestone"            # her column
        state = {"x": 0.5, "y": 93.0, "z": 0.5}
        s = Skills(lambda m, p: {})
        s._me = lambda: dict(state)
        s._blocks = lambda cells: [blocks.get(tuple(c), "minecraft:air") for c in cells]
        s._block = lambda x, y, z: blocks.get((x, y, z), "minecraft:air")
        def brk(x, y, z, timeout=8.0, depth=0):
            blocks[(x, y, z)] = "minecraft:air"
            state["y"] = float(y)                                   # she drops onto the next one down
            return True
        s._break = brk
        return s, state

    def test_a_column_she_built_is_noticed_and_dug_away(self):
        s, state = self.world()
        self.assertEqual(s.perched(), 3)
        out = s.get_off_pillar()
        self.assertIn("挖下来 3 格", out)
        self.assertEqual(state["y"], 90.0)
        self.assertEqual(s.perched(), 0)

    def test_standing_on_ordinary_ground_is_not_perched(self):
        s, state = self.world()
        state["y"] = 90.0
        state["x"] = 2.5
        self.assertEqual(s.perched(), 0)
