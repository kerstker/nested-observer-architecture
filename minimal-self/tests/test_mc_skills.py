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
        if method == "perception.scan":
            if params.get("find"):
                return {"found": [{"id": v, "x": k[0], "y": k[1], "z": k[2]} for k, v in self.blocks.items()
                                  if v in params["find"]]}
            return {"chunks": [{"counts": {"minecraft:oak_log": 4, "minecraft:stripped_oak_log": 1, "minecraft:stone": 900}}]}
        if method == "perception.blocks":
            return {"blocks": [{"id": self.blocks.get((p["x"], p["y"], p["z"]), "minecraft:air" if p["y"] >= 64 else "minecraft:dirt")}
                               for p in params["positions"]]}
        if method == "interact.breakBlock":
            if self.blocks.get((params["x"], params["y"], params["z"]), "").endswith("_log"):
                self.blocks[(params["x"], params["y"], params["z"])] = "minecraft:air"
                self.items += 1
            return {"started": True}
        if method == "player.getState":
            return dict(self.me)
        if method == "nav.pathTo":
            self.me.update({"x": params["x"] + 1.5, "y": float(params["y"]), "z": params["z"] + 0.5})
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
        self.assertLessEqual(abs(path["x"] - 5) + abs(path["z"] - 5), 2)
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
