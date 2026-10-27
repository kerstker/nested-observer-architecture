"""Walk a route that Baritone planned, with her own body.

Baritone's planner is the best there is at finding a way through terrain (dig here, bridge there, never
through lava), but in this modpack its executor never presses a key. So the planner gives the route as block
positions (where her feet go, step by step) and this module walks it the way a player would:

- a plain step (floor there, room for her body) is walked, several in a row in one smooth leg;
- a step into rock is mined first (head height, then feet), then walked;
- a step over a gap gets a block put under it first (bridging);
- a step straight up is a pillar: jump and put a block where her feet were;
- a step straight down mines the block under her feet.

It works only from what the client reports (`perception.blocks`) and uses the hands Skills already has
(`_break`, `place_at`, `pillar_up`, `_walk_to` / the body's `follow_path`). When she gets off the route or a
step fails twice, it asks Baritone for a fresh route from where she stands.
"""
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Cell = Tuple[int, int, int]


def _cell(p: Dict[str, Any]) -> Cell:
    return (int(p["x"]), int(p["y"]), int(p["z"]))


class RouteFollower:
    PLAN_WAIT = 8.0         # seconds to wait for Baritone to finish a route
    LOOKAHEAD = 12          # steps read and classified in one go
    STEP_RETRIES = 2        # a step that fails this many times: plan again from here
    REPLANS = 6             # fresh routes per trip before giving up

    def __init__(self, skills):
        self.s = skills

    # --- Baritone's route ---------------------------------------------------------------------

    def plan(self, goal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Set the goal, wait for the computed route, take it, and stop Baritone before it acts."""
        rpc = self.s.rpc
        if rpc("baritone.goto", goal) is None:
            return None
        end = time.monotonic() + self.PLAN_WAIT
        try:
            while time.monotonic() < end and not self.s._stop.is_set():
                got = rpc("baritone.path", {}) or {}
                if got.get("ready") and got.get("positions"):
                    return got
                time.sleep(0.15)
            return None
        finally:
            rpc("baritone.stop", {})

    # --- classifying a step -------------------------------------------------------------------

    def _read(self, steps: List[Cell]) -> Dict[Cell, str]:
        cells = set()
        for (x, y, z) in steps:
            cells.update({(x, y - 1, z), (x, y, z), (x, y + 1, z), (x, y + 2, z)})
        order = sorted(cells)
        got = self.s._blocks(order)
        return {c: (b or "") for c, b in zip(order, got)}

    def action(self, prev: Cell, step: Cell, blocks: Dict[Cell, str]) -> str:
        """What a player does to take this step: walk, dig, bridge, pillar or down."""
        passable = self.s._passable
        x, y, z = step
        feet, head, floor = blocks.get((x, y, z), ""), blocks.get((x, y + 1, z), ""), blocks.get((x, y - 1, z), "")
        if (x, z) == (prev[0], prev[2]):
            return "pillar" if y > prev[1] else "down"
        if not passable(feet) or not passable(head):
            return "dig"
        if y > prev[1] and not passable(blocks.get((prev[0], prev[1] + 2, prev[2]), "minecraft:air")):
            return "dig"                                   # a step up needs room over her own head first
        if y <= prev[1] + 1 and passable(floor) and y >= prev[1]:
            # nothing to stand on there: Baritone means "put a block down, then step (or jump) onto it"
            return "bridge" if y == prev[1] else "build_up"
        if y > prev[1]:
            return "up"                                    # a step up: its own jump, not inside a running leg
        return "walk"

    # --- one step -------------------------------------------------------------------------------

    def _clear(self, cells: List[Cell]) -> bool:
        for c in cells:
            b = self.s._block(*c)
            if b and not self.s._passable(b) and not self.s._break(*c, timeout=6.0):
                return False
        return True

    def take(self, prev: Cell, step: Cell, kind: str) -> bool:
        s = self.s
        x, y, z = step
        if kind == "pillar":
            if not self._clear([(x, y + 1, z)]):
                return False
            return "爬高了 0 格" not in s.pillar_up(1)
        if kind == "down":
            if not self._clear([(x, y, z)]):
                return False
            end = time.monotonic() + 3
            while time.monotonic() < end:
                me = s._me() or {}
                if me and math.floor(me["y"]) <= y:
                    return True
                time.sleep(0.15)
            return False
        if kind == "dig":
            need = [(x, y + 1, z), (x, y, z)]
            if y > prev[1]:
                need.insert(0, (prev[0], prev[1] + 2, prev[2]))
            if not self._clear(need):
                return False
        if kind in ("bridge", "build_up") and not self._bridge(prev, (x, y - 1, z)):
            return False
        if kind in ("up", "build_up") or (kind == "dig" and y > prev[1]):
            s._face_toward = (x + .5, y + 1.0, z + .5)
            s.rpc("control.jumpOnce", {})
        return self._walk([step])

    def _bridge(self, prev: Cell, cell: Cell) -> bool:
        """A block into the gap ahead, against the side of the block she stands on (placing over air, the way a
        player bridges; Skills._step_block only builds on top of something)."""
        s = self.s
        have = s.crafter.counts()
        item = next((i for i in s.STEP_BLOCKS if have.get(i, 0) > 0), None)
        if item is None:
            return False
        return bool(s.place_at(item, *cell, walk=False))

    def _walk(self, steps: List[Cell]) -> bool:
        s = self.s
        points = [{"x": x + 0.5, "y": y, "z": z + 0.5} for x, y, z in steps]
        if s.body is not None:
            s._hold_for_reflexes()
            ok = s.body.follow_path(points, reach=0.45, timeout=6.0 + 2.0 * len(points))
        else:
            ok = all(s._walk_to(p["x"], p["y"], p["z"], reach=0.5, timeout=8) for p in points)
        me = s._me() or {}
        x, y, z = steps[-1]
        return bool(ok and me and math.hypot(me["x"] - x - .5, me["z"] - z - .5) <= 1.0 and abs(me["y"] - y) <= 1.2)

    # --- the whole trip -------------------------------------------------------------------------

    @staticmethod
    def _nearest(route: List[Cell], me: Dict[str, Any], start: int) -> int:
        """Her place on the route: the closest step from `start` on (a few back too, after a fight pushed her)."""
        here = (me["x"], me["y"], me["z"])
        lo = max(0, start - 3)
        return min(range(lo, len(route)), key=lambda i: math.dist((route[i][0] + .5, route[i][1], route[i][2] + .5), here))

    def go(self, goal: Dict[str, Any], arrived: Callable[[Dict[str, Any]], bool], timeout: float = 150.0) -> str:
        """Plan, walk, re-plan until `arrived(me)`. Returns done / stopped / timeout / no_route / stuck."""
        s = self.s
        end = time.monotonic() + timeout
        for _ in range(self.REPLANS):
            me = s._me() or {}
            if me and arrived(me):
                return "done"
            if s._stop.is_set():
                return "stopped"
            if time.monotonic() >= end:
                return "timeout"
            got = self.plan(goal)
            if not got:
                return "no_route"
            route = [_cell(p) for p in got["positions"]]
            if self._walk_route(route, end) == "stopped":
                return "stopped"
        me = s._me() or {}
        return "done" if me and arrived(me) else "stuck"

    def _walk_route(self, route: List[Cell], end: float) -> str:
        s = self.s
        me = s._me() or {}
        if not me or len(route) < 2:
            return "replan"
        i = self._nearest(route, me, 0)
        failures = 0
        while i < len(route) - 1:
            if s._stop.is_set():
                return "stopped"
            if time.monotonic() >= end:
                return "timeout"
            window = route[i:i + self.LOOKAHEAD + 1]
            blocks = self._read(window)
            kinds = [self.action(window[k], window[k + 1], blocks) for k in range(len(window) - 1)]
            run = 0
            while run < len(kinds) and kinds[run] == "walk":
                run += 1
            s.progress = f"沿 Baritone 路线走（第 {i + 1}/{len(route)} 步）"
            paused_before = s.pause_until
            if run:
                ok = self._walk(window[1:run + 1])
            else:
                s.progress = f"沿路线：{ {'dig': '挖开', 'bridge': '搭方块过去', 'pillar': '垫方块上去', 'down': '往下挖', 'up': '跳上台阶', 'build_up': '垫一块再跳上去'}[kinds[0]] }"
                ok = self.take(window[0], window[1], kinds[0])
            me = s._me() or me
            if ok:
                failures = 0
                s._work_progress()
                i = max(i + max(1, run), self._nearest(route, me, i))
                continue
            if s.pause_until > paused_before:
                s._hold_for_reflexes()                   # a fight or a meal cut the step short: not a failed step
                i = self._nearest(route, s._me() or me, i)
                continue
            failures += 1
            if failures >= self.STEP_RETRIES:
                return "replan"                          # something changed (a mob, a block): fresh route from here
            i = self._nearest(route, me, i)
        return "end"
