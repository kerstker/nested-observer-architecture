"""Her body loop: the camera turns like a hand, she walks, jumps a step, sprints, mines, places and lets go."""
import math
import threading
import time
import unittest

from src.levels.l8_agency.mc_body import PlayerBody


class FakeClock:
    """A virtual clock: sleeping moves time forward instead of waiting, so the whole loop runs in a blink."""

    def __init__(self):
        self.t = 0.0
        self.perf_calls = 0

    def now(self):
        return self.t

    def sleep(self, seconds):
        if seconds and seconds > 0:
            self.t += float(seconds)
        time.sleep(0.0002)                  # let the other threads have a turn

    def perf(self):
        self.perf_calls += 1
        return self.perf_calls * 0.0005     # every bridge call costs ~1 ms of measured latency


class FakeWorld:
    """A flat plain with a one-block step, a wall, a stone block and a bag - just enough of her client."""

    GROUND = 63

    def __init__(self):
        self.clock = FakeClock()
        self.me = {"x": 0.5, "y": 64.0, "z": 0.5, "yaw": 0.0, "pitch": 0.0,
                   "motion": {"x": 0.0, "y": 0.0, "z": 0.0}, "health": 20.0, "food": 20,
                   "onGround": True, "sprinting": False, "selectedSlot": 0,
                   "dimension": "minecraft:overworld", "gameMode": "survival"}
        self.blocks = {}
        self.calls = []
        self.inputs = {"forward": False, "back": False, "left": False, "right": False,
                       "jump": False, "sneak": False, "sprint": False}
        self.hotbar = [{"slot": 0, "id": "minecraft:stone_pickaxe", "count": 1}]
        self.main = []
        self.using = False
        self.jump_once = False
        self.fail_once = []
        self.requires_tool = True
        self.can_harvest = None             # None: work it out from the bag
        self.best_slot = None               # an override for the tool the client reports
        self.placed = []
        self.smooth_camera = True           # the mod turns the camera itself (control.lookTarget)
        self.look_target = None
        self.turns = []                     # yaw after each simulated camera step

    # --- the world ----------------------------------------------------------------------

    def block_id(self, x, y, z):
        key = (int(x), int(y), int(z))
        if key in self.blocks:
            return self.blocks[key]
        return "minecraft:stone" if key[1] <= self.GROUND else "minecraft:air"

    def solid(self, x, y, z):
        return not self.block_id(x, y, z).endswith(":air")

    def set_block(self, x, y, z, ident):
        self.blocks[(int(x), int(y), int(z))] = ident

    def _tool_slot(self):
        if self.best_slot is not None:
            return int(self.best_slot)
        for entry in self.hotbar:
            if "pickaxe" in str(entry.get("id")):
                return int(entry.get("slot", 0))
        return None

    def _camera_tick(self):
        """The mod's CameraDriver between two reads: at most ~300 deg/s toward the target, never a snap."""
        if self.look_target is None:
            return
        tyaw, tpitch, speed = self.look_target
        step = speed / 15.0
        dyaw = (tyaw - self.me["yaw"] + 180.0) % 360.0 - 180.0
        dpitch = tpitch - self.me["pitch"]
        self.me["yaw"] += max(-step, min(step, dyaw))
        self.me["pitch"] += max(-step, min(step, dpitch))
        self.turns.append(self.me["yaw"])

    def _game_tick(self):
        """What the game does on its own ticks: held keys move her, gravity pulls her down."""
        self._camera_tick()
        yaw = float(self.me["yaw"])
        speed = 7.0 if self.inputs.get("sprint") else 4.3
        step = speed / 15.0
        if self.inputs.get("forward"):
            dx = math.cos(math.radians(yaw + 90.0))
            dz = math.sin(math.radians(yaw + 90.0))
            nx, nz = self.me["x"] + dx * step, self.me["z"] + dz * step
            fy = math.floor(self.me["y"])
            fx, fz = math.floor(nx), math.floor(nz)
            if self.solid(fx, fy, fz) or self.solid(fx, fy + 1, fz):
                # a player jumps up a one-block step instead of stopping dead against it
                if self.inputs.get("jump") and not self.solid(fx, fy + 1, fz) and not self.solid(fx, fy + 2, fz):
                    self.me["y"] = fy + 1.0
                    fy += 1
                    if not self.solid(fx, fy, fz) and not self.solid(fx, fy + 1, fz):
                        self.me["x"], self.me["z"] = nx, nz
            else:
                self.me["x"], self.me["z"] = nx, nz
        below_x = math.floor(self.me["x"])
        below_z = math.floor(self.me["z"])
        if not self.solid(below_x, math.floor(self.me["y"]) - 1, below_z):
            self.me["y"] -= 1.0                       # fell off a step
        self.me["onGround"] = self.solid(below_x, math.floor(self.me["y"]) - 1, below_z)
        self.me["sprinting"] = bool(self.inputs.get("forward") and self.inputs.get("sprint"))

    # --- the bridge ---------------------------------------------------------------------

    def rpc(self, method, params):
        params = params or {}
        self.calls.append((method, dict(params), self.clock.now()))
        if method in self.fail_once:
            self.fail_once.remove(method)
            raise RuntimeError("bridge hiccup: " + method)
        handler = getattr(self, "_rpc_" + method.replace(".", "_"), None)
        if handler is None:
            return {}
        return handler(params)

    def _rpc_player_getState(self, params):
        self._game_tick()
        return dict(self.me)

    def _rpc_control_setInput(self, params):
        for key in self.inputs:
            if key in params:
                self.inputs[key] = bool(params[key])
        return {"ok": True}

    def _rpc_control_lookTarget(self, params):
        if not self.smooth_camera:
            return None                     # an older mod: no such method
        if params.get("clear"):
            self.look_target = None
        else:
            self.look_target = (float(params.get("yaw", self.me["yaw"])), float(params.get("pitch", self.me["pitch"])),
                                float(params.get("speed", 300.0)))
        remaining = 0.0 if self.look_target is None else abs((self.look_target[0] - self.me["yaw"] + 180) % 360 - 180)
        return {"yaw": self.me["yaw"], "pitch": self.me["pitch"], "remaining": remaining, "settled": remaining <= 1.5}

    def _rpc_control_look(self, params):
        yaw = self.me["yaw"]
        pitch = self.me["pitch"]
        if "yaw" in params:
            yaw = float(params["yaw"])
        if "pitch" in params:
            pitch = max(-90.0, min(90.0, float(params["pitch"])))
        self.me["yaw"], self.me["pitch"] = yaw, pitch
        return {"yaw": yaw, "pitch": pitch}

    def _rpc_control_stop(self, params):
        for key in self.inputs:
            self.inputs[key] = False
        return {"ok": True}

    def _rpc_control_jumpOnce(self, params):
        self.jump_once = True
        return {"ok": True}

    def _rpc_control_startUsing(self, params):
        self.using = True
        return {"ok": True}

    def _rpc_control_stopUsing(self, params):
        self.using = False
        return {"ok": True}

    def _rpc_perception_blocks(self, params):
        tool = bool(params.get("tool"))
        out = []
        for position in params.get("positions") or []:
            x = int(math.floor(position.get("x", 0)))
            y = int(math.floor(position.get("y", 0)))
            z = int(math.floor(position.get("z", 0)))
            ident = self.block_id(x, y, z)
            air = ident.endswith(":air")
            block = {"x": x, "y": y, "z": z, "id": ident, "air": air, "loaded": True,
                     "hardness": 0.0 if air else 1.5, "requiresTool": (not air) and self.requires_tool}
            if tool and not air:
                slot = self._tool_slot()
                block["bestSlot"] = slot
                if self.can_harvest is not None:
                    block["canHarvest"] = bool(self.can_harvest)
                else:
                    block["canHarvest"] = slot is not None or not self.requires_tool
            out.append(block)
        return {"blocks": out}

    def _rpc_interact_breakBlock(self, params):
        self.set_block(params["x"], params["y"], params["z"], "minecraft:air")
        return {"started": True, "mode": "survival"}

    def _rpc_interact_stopBreaking(self, params):
        return {"ok": True}

    def _rpc_interact_placeBlock(self, params):
        offsets = {"up": (0, 1, 0), "down": (0, -1, 0), "north": (0, 0, -1), "south": (0, 0, 1),
                   "east": (1, 0, 0), "west": (-1, 0, 0)}
        base = (int(math.floor(params["x"])), int(math.floor(params["y"])), int(math.floor(params["z"])))
        ox, oy, oz = offsets[params.get("face", "up")]
        target = (base[0] + ox, base[1] + oy, base[2] + oz)
        if not self.solid(*base):
            return {"result": "FAIL", "placed": False}
        self.set_block(target[0], target[1], target[2], "minecraft:cobblestone")
        self.placed.append((base, params.get("face", "up"), target))
        return {"result": "SUCCESS", "placed": True,
                "target": {"x": target[0], "y": target[1], "z": target[2]}}

    def _rpc_interact_attackEntity(self, params):
        return {"ok": "attacked " + str(params.get("uuid"))}

    def _rpc_player_getInventory(self, params):
        return {"hotbar": [dict(entry) for entry in self.hotbar],
                "main": [dict(entry) for entry in self.main],
                "selectedSlot": int(self.me["selectedSlot"])}

    def _rpc_inventory_selectHotbar(self, params):
        self.me["selectedSlot"] = int(params["slot"])
        return {"ok": True}

    def _rpc_inventory_swapSlots(self, params):
        return {"ok": True}


def methods(world):
    return [method for method, _, _ in world.calls]


def params_of(world, method):
    return [params for name, params, _ in world.calls if name == method]


class TestBodyLoop(unittest.TestCase):
    def setUp(self):
        self.world = FakeWorld()
        self.bodies = []

    def tearDown(self):
        for body in self.bodies:
            try:
                body.stop()
            except Exception:
                pass

    def body(self, virtual_clock=True, **kwargs):
        """A body on the fake world; by default on a virtual clock so the whole loop runs instantly."""
        if virtual_clock:
            kwargs.setdefault("now", self.world.clock.now)
            kwargs.setdefault("sleep", self.world.clock.sleep)
            kwargs.setdefault("perf", self.world.clock.perf)
        body = PlayerBody(self.world.rpc, **kwargs)
        self.bodies.append(body)
        return body

    # --- frames and telemetry -----------------------------------------------------------

    def test_each_tick_reports_where_she_is_and_how_the_bridge_is_doing(self):
        events = []
        body = self.body(on_event=events.append)
        for _ in range(5):
            body.tick()
        self.assertEqual(len(events), 5)
        self.assertEqual(events[-1]["event"], "tick")
        self.assertEqual(events[-1]["action"], "idle")
        self.assertEqual(events[-1]["pos"], {"x": 0.5, "y": 64.0, "z": 0.5})
        self.assertIn("moved", events[-1])
        stats = body.stats()
        self.assertEqual(stats["ticks"], 5)
        self.assertGreaterEqual(stats["rpc_calls"], 5)
        self.assertGreater(stats["rpc_avg_ms"], 0.0)
        self.assertGreaterEqual(stats["rpc_p95_ms"], stats["rpc_avg_ms"])
        self.assertFalse(stats["running"])

    def test_the_tick_rate_stat_matches_the_loop(self):
        body = self.body()
        body.start()
        time.sleep(0.05)
        rate = body.stats()["tick_rate"]
        self.assertGreaterEqual(rate, 10.0)
        self.assertLessEqual(rate, 20.0)

    def test_reflex_item_use_and_sneaking_come_from_actual_player_state(self):
        events = []
        body = self.body(on_event=events.append)
        self.world.me.update(usingItem=True, sneaking=True)
        body.tick()
        self.assertEqual(events[-1]["action"], "use")
        self.assertTrue(events[-1]["usingItem"])
        self.assertTrue(events[-1]["sneaking"])
        self.assertEqual(body.stats()["action"], "use")
        self.assertFalse(events[-1]["busy"])
        self.assertNotIn("control.startUsing", methods(self.world))
        self.world.me.update(usingItem=False, sneaking=False)
        body.tick()
        self.assertEqual(events[-1]["action"], "idle")
        self.assertFalse(events[-1]["usingItem"])
        self.assertFalse(events[-1]["sneaking"])

    def test_external_navigation_motion_is_visible_while_body_yields(self):
        events = []
        body = self.body(on_event=events.append, skills_busy=lambda: True)
        body.tick()
        self.assertEqual(events[-1]["action"], "yield")
        for _ in range(3):
            self.world.me["z"] += 0.2  # movement reported by the external navigator
            body.tick()
            self.assertEqual(events[-1]["action"], "walk")
            self.assertGreater(events[-1]["moved"], 0.025)
        self.world.me.update(usingItem=True, sneaking=True)
        self.world.me["z"] += 0.2
        body.tick()
        self.assertEqual(events[-1]["action"], "use")
        self.assertTrue(events[-1]["sneaking"])
        # Explicit actions retain their meaning even if an item is also in use.
        self.assertEqual(body._event("mine", dict(self.world.me))["action"], "mine")

    def test_failed_state_read_does_not_replay_cached_use_or_navigation(self):
        events = []
        body = self.body(on_event=events.append)
        self.world.me.update(usingItem=True, sneaking=True)
        body.tick()
        self.world.me["z"] += 1
        self.world.fail_once.append("player.getState")
        body.tick()
        self.assertEqual(events[-1]["action"], "disconnected")
        self.assertIsNone(events[-1]["usingItem"])
        self.assertIsNone(events[-1]["sneaking"])
        self.assertEqual(events[-1]["moved"], 0)

    def test_a_broken_event_callback_does_not_kill_the_body(self):
        def boom(event):
            raise RuntimeError("the log is broken")
        body = self.body(on_event=boom)
        for _ in range(3):
            body.tick()                             # must not raise
        self.assertEqual(body.stats()["ticks"], 3)

    def test_a_bridge_hiccup_is_counted_and_the_body_survives(self):
        body = self.body()
        body.tick()
        self.world.fail_once.append("player.getState")
        body.tick()
        self.assertEqual(body.stats()["rpc_errors"], 1)
        body.tick()
        self.assertEqual(body.stats()["ticks"], 3)

    def test_a_running_loop_refuses_a_manual_tick(self):
        body = self.body()
        body.start()
        with self.assertRaises(RuntimeError):
            body.tick()

    # --- the camera ---------------------------------------------------------------------

    def test_the_camera_turns_in_small_human_steps_not_one_snap(self):
        # the mod turns the camera every frame; the loop only names the target, it does not stream angles
        body = self.body()
        self.assertTrue(body.look_at(5.0, 64.5, 0.5, timeout=3.0))
        yaws = self.world.turns
        self.assertGreaterEqual(len(yaws), 4)
        deltas = [later - earlier for earlier, later in zip(yaws, yaws[1:])]
        self.assertTrue(all(abs(delta) <= 300.0 / 15.0 + 1e-6 for delta in deltas), deltas)
        self.assertAlmostEqual(yaws[-1], -90.0, delta=2.0)
        self.assertLessEqual(len(params_of(self.world, "control.lookTarget")), 2)   # one target, one release
        self.assertEqual(params_of(self.world, "control.look"), [])

    def test_an_older_mod_without_smooth_turning_still_turns_in_small_steps(self):
        self.world.smooth_camera = False
        body = self.body()
        self.assertTrue(body.look_at(5.0, 64.5, 0.5, timeout=3.0))
        looks = params_of(self.world, "control.look")
        self.assertGreaterEqual(len(looks), 4)
        yaws = [entry["yaw"] for entry in looks]
        deltas = [later - earlier for earlier, later in zip(yaws, yaws[1:])]
        self.assertTrue(all(delta <= 1e-6 for delta in deltas), deltas)      # monotone toward the target
        limit = 300.0 / 15.0 + 1e-6                                          # never faster than a wrist
        self.assertTrue(all(abs(delta) <= limit for delta in deltas), deltas)
        self.assertAlmostEqual(yaws[-1], -90.0, delta=2.0)
        self.assertGreater(self.world.clock.t, 0.15)                        # a turn takes time, it is not a snap

    def test_a_point_already_in_the_crosshair_is_not_turned_at_all(self):
        body = self.body()
        eye_height = 64.0 + 1.62
        self.assertTrue(body.look_at(0.5, eye_height, 10.5, timeout=2.0))
        self.assertLessEqual(len(params_of(self.world, "control.look")), 1)

    # --- walking ------------------------------------------------------------------------

    def test_she_jumps_over_a_one_block_step_on_the_way(self):
        self.world.set_block(0, 64, 2, "minecraft:stone")       # a step in the way
        events = []
        body = self.body(on_event=events.append)
        self.assertTrue(body.walk_to(0.5, 64.0, 5.0, reach=1.0, timeout=20.0))
        inputs = params_of(self.world, "control.setInput")
        self.assertTrue(any(entry.get("jump") for entry in inputs), "never jumped at the step")
        self.assertTrue(any(entry.get("forward") for entry in inputs))
        self.assertGreaterEqual(self.world.me["z"], 3.9)
        self.assertTrue(any(event["moved"] > 0 for event in events))

    def test_a_long_flat_run_sprints(self):
        body = self.body()
        self.assertTrue(body.walk_to(0.5, 64.0, 20.5, reach=1.5, timeout=25.0))
        inputs = params_of(self.world, "control.setInput")
        self.assertTrue(any(entry.get("sprint") and entry.get("forward") for entry in inputs))
        self.assertGreaterEqual(self.world.me["z"], 18.5)

    def test_a_two_high_wall_makes_her_give_up_and_let_go(self):
        self.world.set_block(0, 64, 2, "minecraft:stone")
        self.world.set_block(0, 65, 2, "minecraft:stone")
        body = self.body()
        self.assertFalse(body.walk_to(0.5, 64.0, 5.0, reach=1.0, timeout=15.0))
        self.assertFalse(any(self.world.inputs.values()), "keys stayed held after giving up")
        self.assertTrue(body.running)                           # the loop itself is still alive

    def test_a_queued_path_is_walked_waypoint_by_waypoint(self):
        body = self.body()
        self.assertEqual(body.queue_waypoints([(0.5, 64.0, 3.5), {"x": 0.5, "y": 64.0, "z": 6.5}]), 2)
        self.assertTrue(body.follow_path(reach=1.0, timeout=20.0))
        self.assertEqual(len(body.waypoints), 0)
        self.assertGreaterEqual(self.world.me["z"], 5.4)

    # --- hands --------------------------------------------------------------------------

    def test_mining_walks_in_picks_the_tool_and_breaks_it(self):
        self.world.hotbar = [{"slot": 3, "id": "minecraft:iron_pickaxe", "count": 1}]
        self.world.set_block(0, 64, 4, "minecraft:stone")
        body = self.body()
        self.assertTrue(body.mine_block(0, 64, 4, timeout=8.0))
        picked = params_of(self.world, "inventory.selectHotbar")
        self.assertEqual(picked[0], {"slot": 3})
        broken = params_of(self.world, "interact.breakBlock")
        self.assertEqual(broken[0], {"x": 0, "y": 64, "z": 4, "mode": "survival"})
        self.assertEqual(self.world.block_id(0, 64, 4), "minecraft:air")
        self.assertGreaterEqual(len(self.world.turns), 2)                 # she turned toward it, over frames
        self.assertGreater(self.world.me["z"], 0.5)             # she walked toward it first

    def test_mining_stops_when_the_hand_cannot_harvest_the_block(self):
        self.world.can_harvest = False
        self.world.set_block(0, 64, 2, "minecraft:stone")
        body = self.body()
        self.assertFalse(body.mine_block(0, 64, 2))
        self.assertNotIn("interact.breakBlock", methods(self.world))

    def test_placing_clicks_the_face_of_the_block_below(self):
        self.world.hotbar = [{"slot": 0, "id": "minecraft:cobblestone", "count": 10}]
        body = self.body()
        self.assertTrue(body.place_block(0, 64, 2))
        placed = params_of(self.world, "interact.placeBlock")
        self.assertEqual(len(placed), 1)
        self.assertEqual(placed[0], {"x": 0, "y": 63, "z": 2, "face": "up"})
        self.assertEqual(self.world.block_id(0, 64, 2), "minecraft:cobblestone")

    def test_placing_refuses_a_cell_that_is_not_empty(self):
        self.world.set_block(0, 64, 2, "minecraft:stone")
        body = self.body()
        self.assertFalse(body.place_block(0, 64, 2))
        self.assertNotIn("interact.placeBlock", methods(self.world))

    def test_attacking_gets_in_reach_and_swings_at_a_human_cadence(self):
        body = self.body()
        target = {"uuid": "entity-1", "x": 4.5, "y": 64.0, "z": 0.5}
        self.assertTrue(body.attack(target, times=2))
        hits = [stamp for name, _, stamp in self.world.calls if name == "interact.attackEntity"]
        self.assertEqual(len(hits), 2)
        self.assertGreaterEqual(hits[1] - hits[0], 0.6)         # ~0.65 s between swings
        self.assertGreaterEqual(len(self.world.turns), 2)                 # she turned toward it, over frames

    def test_attacking_a_uuid_without_coordinates_still_swings(self):
        body = self.body()
        self.assertTrue(body.attack("entity-2"))
        self.assertEqual(len(params_of(self.world, "interact.attackEntity")), 1)

    def test_using_an_item_holds_the_button_and_lets_go(self):
        body = self.body()
        self.assertTrue(body.use_item(seconds=0.5))
        order = [name for name in methods(self.world) if name in ("control.startUsing", "control.stopUsing")]
        self.assertEqual(order, ["control.startUsing", "control.stopUsing"])
        stamps = [stamp for name, _, stamp in self.world.calls if name in ("control.startUsing", "control.stopUsing")]
        self.assertGreaterEqual(stamps[1] - stamps[0], 0.45)

    # --- not fighting the other threads -------------------------------------------------

    def test_while_her_other_hands_are_busy_the_body_stands_still(self):
        busy = {"on": True}
        events = []
        body = self.body(on_event=events.append, skills_busy=lambda: busy["on"])
        for _ in range(3):
            body.tick()
        self.assertTrue(all(event["action"] == "yield" for event in events))
        self.assertFalse(any(self.world.inputs.values()))
        busy["on"] = False
        body.tick()
        self.assertEqual(events[-1]["action"], "idle")

    def test_stop_lets_go_of_the_keys_and_ends_the_loop(self):
        body = PlayerBody(self.world.rpc, hz=20)                # a real clock: a walk that is still running
        self.bodies.append(body)
        result = []
        walker = threading.Thread(target=lambda: result.append(body.walk_to(0.5, 64.0, 25.0, reach=1.0, timeout=20.0)))
        walker.start()
        end = time.time() + 3.0
        while time.time() < end and not self.world.inputs.get("forward"):
            time.sleep(0.01)
        self.assertTrue(self.world.inputs.get("forward"), "she never started walking")
        body.stop()
        walker.join(3.0)
        self.assertFalse(walker.is_alive(), "the waiting command was never released")
        self.assertEqual(result, [False])
        self.assertFalse(any(self.world.inputs.values()))
        self.assertFalse(body.running)

    def test_cancelling_stops_the_walk_but_keeps_the_body_alive(self):
        body = PlayerBody(self.world.rpc, hz=20)
        self.bodies.append(body)
        result = []
        walker = threading.Thread(target=lambda: result.append(body.walk_to(0.5, 64.0, 25.0, reach=1.0, timeout=20.0)))
        walker.start()
        end = time.time() + 3.0
        while time.time() < end and not self.world.inputs.get("forward"):
            time.sleep(0.01)
        body.cancel()
        walker.join(3.0)
        self.assertFalse(walker.is_alive())
        self.assertEqual(result, [False])
        self.assertTrue(body.running)


if __name__ == "__main__":
    unittest.main()


class TestAimAtAnOpenFace(unittest.TestCase):
    """The live failure at y=23: climbing a stair, the stone over the next step was aimed at from below, where it
    is pressed against more stone, so the crosshair landed on the wrong block and 'stone' could not be mined."""

    def test_the_face_toward_the_hole_is_aimed_at_not_the_one_against_rock(self):
        world = FakeWorld()
        world.me.update(x=0.5, y=23.0, z=0.5, onGround=True)
        for y in range(15, 30):
            for dx in (-2, -1, 0, 1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    world.set_block(dx, y, dz, "minecraft:stone")
        for y in (23, 24, 25):                     # her pocket: feet, head, and the cell over her head dug out
            world.set_block(0, y, 0, "minecraft:air")
        body = PlayerBody(world.rpc, now=world.clock.now, sleep=world.clock.sleep, perf=world.clock.perf)
        body._state = dict(world.me)
        aim = body._face_point(1, 25, 0)          # the stone over the next step (+x, two up)
        self.assertAlmostEqual(aim[0], 1.01, places=2)          # its west side, open to her pocket
        self.assertAlmostEqual(aim[1], 25.5, places=2)
        self.assertIsNone(body._sight_blocker(1, 25, 0))
