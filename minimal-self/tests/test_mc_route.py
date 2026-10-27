"""Baritone plans the route; her own body walks it: walk runs, dig through rock, bridge a gap, pillar up, dig
down, re-plan when a step fails, stop when told."""
import math
import tempfile
import os
import unittest

from src.levels.l8_agency.mc_route import RouteFollower
from src.levels.l8_agency.mc_skills import Skills
from src.levels.l8_agency.mc_places import Places


class World:
    """Blocks, her feet, and a Baritone that only plans: `routes` are handed out one per baritone.goto."""

    def __init__(self, routes, me=(0, 64, 0), blocks=None, ground=63):
        self.routes = [list(r) for r in routes]
        self.me = {"x": me[0] + .5, "y": float(me[1]), "z": me[2] + .5}
        self.blocks = dict(blocks or {})
        self.ground = ground
        self.calls = []
        self.current = None
        self.bag = {"minecraft:cobblestone": 32}

    def block(self, c):
        if c in self.blocks:
            return self.blocks[c]
        return "minecraft:stone" if c[1] <= self.ground else "minecraft:air"

    def rpc(self, method, params):
        self.calls.append((method, dict(params or {})))
        if method == "baritone.status":
            return {"available": True, "busy": False}
        if method == "baritone.goto":
            self.current = self.routes.pop(0) if self.routes else None
            return {"started": True}
        if method == "baritone.path":
            if not self.current:
                return {"ready": False}
            return {"ready": True, "positions": [{"x": x, "y": y, "z": z} for x, y, z in self.current]}
        if method == "baritone.stop":
            return {"stopped": True}
        if method == "player.getState":
            return dict(self.me)
        if method == "perception.blocks":
            return {"blocks": [{"id": self.block((p["x"], p["y"], p["z"]))} for p in params["positions"]]}
        return {}

    def sent(self, method):
        return [p for m, p in self.calls if m == method]


class Body:
    """Walks her where she is told, if her body fits there (feet and head free, something under her)."""

    def __init__(self, world):
        self.world = world
        self.legs = []

    def follow_path(self, points, reach=0.45, timeout=10):
        w = self.world
        self.legs.append(len(points))
        for p in points:
            c = (math.floor(p["x"]), int(p["y"]), math.floor(p["z"]))
            free = lambda q: w.block(q).endswith(":air")
            if not (free(c) and free((c[0], c[1] + 1, c[2])) and not free((c[0], c[1] - 1, c[2]))):
                return False
            w.me.update(x=p["x"], y=float(p["y"]), z=p["z"])
        return True

    def cancel(self):
        pass


def skills(world):
    s = Skills(world.rpc)
    s.places = Places(os.path.join(tempfile.mkdtemp(), "places.json"))
    s.body = Body(world)
    s.crafter.counts = lambda: dict(world.bag)

    def brk(x, y, z, timeout=8.0, depth=0):
        world.blocks[(x, y, z)] = "minecraft:air"
        return True
    s._break = brk

    def place_at(item, x, y, z, walk=True):
        world.blocks[(x, y, z)] = item
        world.bag[item] -= 1
        return (x, y, z)
    s.place_at = place_at

    def pillar_up(blocks=1, item=""):
        world.blocks[(math.floor(world.me["x"]), int(world.me["y"]), math.floor(world.me["z"]))] = "minecraft:cobblestone"
        world.me["y"] += 1
        return "垫方块爬高了 1 格。"
    s.pillar_up = pillar_up
    return s


def line(start, n, dx=1, dy=0):
    x, y, z = start
    return [(x + i * dx, y + i * dy, z) for i in range(n + 1)]


class TestRouteFollower(unittest.TestCase):
    def test_a_plain_route_is_walked_in_one_smooth_leg(self):
        world = World([line((0, 64, 0), 8)])
        s = skills(world)
        self.assertEqual(s.baritone_goto(8, 64, 0, near=0), "done")
        self.assertEqual((math.floor(world.me["x"]), world.me["y"]), (8, 64.0))
        self.assertEqual(s.body.legs, [8])                       # one leg, not eight stop-and-go steps
        self.assertTrue(world.sent("baritone.stop"))              # Baritone never walks her itself

    def test_rock_in_the_way_is_mined_then_walked(self):
        wall = {(3, 64, 0): "minecraft:stone", (3, 65, 0): "minecraft:stone"}
        world = World([line((0, 64, 0), 6)], blocks=wall)
        s = skills(world)
        self.assertEqual(s.baritone_goto(6, 64, 0), "done")
        self.assertEqual(world.blocks[(3, 64, 0)], "minecraft:air")
        self.assertEqual(world.blocks[(3, 65, 0)], "minecraft:air")

    def test_a_gap_gets_a_block_under_it(self):
        gap = {(2, 63, 0): "minecraft:air"}
        world = World([line((0, 64, 0), 4)], blocks=gap)
        s = skills(world)
        s._step_block = lambda x, y, z: False         # the "build on top of something" rule must not decide
        self.assertEqual(s.baritone_goto(4, 64, 0), "done")
        self.assertEqual(world.blocks[(2, 63, 0)], "minecraft:cobblestone")

    def test_straight_up_is_a_pillar_and_straight_down_is_dug(self):
        up = [(0, 64, 0), (0, 65, 0), (0, 66, 0)]
        world = World([up])
        s = skills(world)
        self.assertEqual(s.baritone_goto(y=66), "done")
        self.assertEqual(world.me["y"], 66.0)
        down = World([[(0, 64, 0), (0, 63, 0)]])
        s2 = skills(down)
        def fall_after_break(x, y, z, timeout=8.0, depth=0):
            down.blocks[(x, y, z)] = "minecraft:air"
            down.me["y"] = float(y)
            return True
        s2._break = fall_after_break
        self.assertEqual(s2.baritone_goto(y=63), "done")

    def test_a_step_that_keeps_failing_gets_a_fresh_route(self):
        # the first route runs into a block her body cannot pass and that cannot be mined; the second goes round
        bad = {(2, 64, 0): "minecraft:bedrock", (2, 65, 0): "minecraft:bedrock"}
        detour = [(0, 64, 0), (0, 64, 1), (1, 64, 1), (2, 64, 1), (3, 64, 1), (3, 64, 0), (4, 64, 0)]
        world = World([line((0, 64, 0), 4), detour], blocks=bad)
        s = skills(world)
        s._break = lambda x, y, z, timeout=8.0, depth=0: world.block((x, y, z)) != "minecraft:bedrock" and (
            world.blocks.__setitem__((x, y, z), "minecraft:air") or True)
        self.assertEqual(s.baritone_goto(4, 64, 0), "done")
        self.assertEqual(len(world.sent("baritone.goto")), 2)

    def test_no_route_and_stop_are_reported_as_such(self):
        world = World([])
        s = skills(world)
        RouteFollower.PLAN_WAIT, old = 0.3, RouteFollower.PLAN_WAIT
        try:
            self.assertEqual(s.baritone_goto(50, 64, 50), "no_route")
        finally:
            RouteFollower.PLAN_WAIT = old
        world2 = World([line((0, 64, 0), 8)])
        s2 = skills(world2)
        s2._stop.set()
        self.assertEqual(s2.baritone_goto(8, 64, 0), "stopped")

    def test_a_fight_in_the_middle_of_a_route_is_not_counted_as_a_failed_step(self):
        import time
        world = World([line((0, 64, 0), 6)])
        s = skills(world)
        legs = []
        def interrupted(points, reach=0.45, timeout=10):
            legs.append(len(points))
            if len(legs) <= 3:
                s.pause_until = time.time() + 0.05           # the fight thread took the body for a moment
                return False
            return Body(world).follow_path(points)
        s.body.follow_path = interrupted
        self.assertEqual(s.baritone_goto(6, 64, 0), "done")
        self.assertEqual(len(world.sent("baritone.goto")), 1)     # still the first route: no re-plan

    def test_a_step_up_onto_nothing_gets_a_block_put_there_first(self):
        # the live case at y=33: the next step is one up, over air - Baritone means "place, then jump onto it"
        world = World([[(0, 64, 0), (1, 65, 0), (2, 65, 0)]], blocks={(1, 64, 0): "minecraft:air"})
        s = skills(world)
        self.assertEqual(s.baritone_goto(2, 65, 0), "done")
        self.assertEqual(world.blocks[(1, 64, 0)], "minecraft:cobblestone")
