"""
Level 8i: her body while she plays - one continuous, human-paced loop.

The rest of her game code (`mc_skills.Skills`) does one whole thing per call: walk to the tree, chop it,
come back. That is right for a single tool call, but a person does not act in steps - they hold W, move
the mouse, swing, and keep looking while the arm works. This module is that body: a background loop at
10-20 Hz that holds the keys, turns the camera in small human steps, and carries out one command at a
time (`walk_to`, `look_at`, `mine_block`, `place_block`, `attack`, `use_item`).

It talks to the client only through the same bridge RPC callable the rest of her game code uses
(`rpc(method, params) -> dict`), so it can be tested against a fake world and wired in wherever the
game client lives. It never teleports and never snaps the camera: a big turn is drawn over several
frames at no more than ~300 degrees a second, and walking is forward + steering + the odd jump over a
one-block step, exactly like a player.

    body = PlayerBody(game.rpc, on_event=lambda ev: log(ev), skills_busy=skills.busy)
    body.walk_to(120, 64, -30)          # blocks the calling thread until there or timed out
    body.mine_block(121, 64, -30)       # walks within reach, picks the tool, looks, mines
    body.stop()                         # keys up, loop ends

Commands may be sent from any thread (they are queued and handed to the loop one at a time); the
`on_event` callback runs on the loop thread, so keep it short and do not send commands from inside it.
"""
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["PlayerBody"]

# --- numbers that make it look like a person ------------------------------------------------
MAX_TURN_DEG_S = 240.0      # a wrist, not a magic camera
TURN_ACCEL_S = 0.30         # time to reach full mouse speed after standing still
TURN_SETTLE_DEG = 22.0      # slow down in the last degrees, like a hand adjusting a crosshair
EYE_HEIGHT = 1.62           # eyes above her feet
BREAK_REACH = 4.5           # a survival player can break this far away
PLACE_REACH = 4.5
ATTACK_REACH = 3.0
SWING_S = 0.65              # one left-click every ~0.65 s, the usual swing cadence
WALK_REACH = 2.0
STUCK_JUMP_AFTER = 0.30     # seconds pushing against something before a player tries jumping
STUCK_SIDE_AFTER = 0.80     # seconds before trying to go around
STUCK_GIVE_UP_AFTER = 2.60  # seconds before deciding it cannot get there
CLIFF_LOOKAHEAD = 0.60      # how far ahead the feet probe reaches

# Blocks a crosshair goes through / that a player just knocks aside.
SEE_THROUGH = ("air", "cave_air", "void_air", "water", "lava", "grass", "fern", "flower", "torch", "snow",
               "sapling", "vine", "petals", "rail", "button", "carpet", "leaf_litter", "bush", "pressure_plate",
               "tripwire", "glass")

# (offset from the cell we place into, the face of the clicked block that points at it)
PLACE_NEIGHBOURS = (((0, -1, 0), "up"), ((0, 1, 0), "down"),
                    ((0, 0, -1), "south"), ((0, 0, 1), "north"),
                    ((-1, 0, 0), "east"), ((1, 0, 0), "west"))


# --- small helpers --------------------------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _wrap180(degrees: float) -> float:
    """An angle folded into (-180, 180] - so "turn left 350" becomes "turn right 10"."""
    return (float(degrees) + 180.0) % 360.0 - 180.0


def _num(mapping: Any, key: str, default: float = 0.0) -> float:
    """A float from a possibly-missing RPC field, never raising."""
    try:
        value = (mapping or {}).get(key)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _point(value: Any) -> Tuple[float, float, float]:
    """(x, y, z) from a tuple/list or a dict as returned by the bridge."""
    if isinstance(value, dict):
        return (float(value.get("x", 0) or 0), float(value.get("y", 0) or 0), float(value.get("z", 0) or 0))
    x, y, z = value
    return (float(x), float(y), float(z))


def _block_name(block: Any) -> str:
    return str((block or {}).get("id") or "")


def _solid(block: Any) -> bool:
    """Whether that perception.blocks entry is something a body would bump into."""
    if not block:
        return False
    if block.get("loaded") is False:
        return False
    name = _block_name(block).split(":")[-1]
    if name in ("nether_portal", "short_grass", "tall_grass", "fern", "large_fern", "torch", "wall_torch"):
        return False
    if "air" in block:
        return not bool(block.get("air"))
    name = _block_name(block)
    return bool(name) and not name.endswith(":air")


def _see_through(block: Any) -> bool:
    if not _solid(block):
        return True
    name = _block_name(block).split(":")[-1]
    if name.endswith(("_block", "_log", "_wood", "_planks", "_slab", "_stairs")) or "glass" in name:
        return False
    return any(word in name for word in SEE_THROUGH)


class _Job:
    """One queued command: a generator that yields once per body tick and returns True/False at the end."""

    __slots__ = ("kind", "gen", "target", "result", "note", "done", "cancel")

    def __init__(self, kind: str, gen: Any, target: Any = None):
        self.kind = kind
        self.gen = gen
        self.target = target
        self.result = False
        self.note = ""
        self.done = threading.Event()
        self.cancel = threading.Event()


class PlayerBody:
    """Her body at play: a 10-20 Hz loop that holds keys and turns the camera like a human.

    `rpc(method, params)` is the mcpfabric bridge call (method -> dict or None).
    `on_event(dict)` is called once per tick with the body's state (see docs/mc_body.md).
    `skills_busy` is an optional callable (e.g. `skills.busy`): while it is true and no command was
    given, the body stands still instead of fighting the other thread for the keyboard.
    `now`, `sleep` and `perf` can be replaced in tests to run the whole loop on a virtual clock.
    """

    def __init__(self, rpc: Callable[[str, Dict[str, Any]], Any], on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                 hz: float = 15.0, *,
                 now: Optional[Callable[[], float]] = None,
                 sleep: Optional[Callable[[float], None]] = None,
                 perf: Optional[Callable[[], float]] = None,
                 skills_busy: Optional[Callable[[], bool]] = None):
        self.rpc = rpc
        self.on_event: Callable[[Dict[str, Any]], None] = on_event or (lambda event: None)
        self.hz = _clamp(float(hz), 10.0, 20.0)
        self.interval = 1.0 / self.hz
        self._now = now or time.monotonic
        self._sleep = sleep or time.sleep
        self._perf = perf or time.perf_counter
        self.skills_busy = skills_busy or (lambda: False)

        self._lock = threading.RLock()          # the job queue and the thread itself
        self._io_lock = threading.Lock()        # one bridge call at a time
        self._queue: Deque[_Job] = deque()
        self._job: Optional[_Job] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False

        self._state: Dict[str, Any] = {}
        self._state_failures = 0
        self._last_event_pos: Optional[Tuple[float, float, float]] = None
        self._last_tick_t: Optional[float] = None
        self._dt = self.interval
        self._ticks = 0
        self._tick_times: Deque[float] = deque(maxlen=60)

        self._want = {"forward": False, "back": False, "left": False, "right": False,
                      "jump": False, "sneak": False, "sprint": False}
        self._sent = dict(self._want)
        self._jump_ticks = 0

        self._cmd_yaw: Optional[float] = None
        self._cmd_pitch = 0.0
        self._last_turn_t: Optional[float] = None
        self._turn_velocity = [0.0, 0.0]
        # The mod turns the camera itself, every rendered frame (control.lookTarget): the loop only says where.
        # None = not tried yet; False = the bridge has no such method (older mod) until _smooth_retry.
        self._smooth: Optional[bool] = None
        self._smooth_retry = 0.0
        self._smooth_sent: Optional[Tuple[float, float]] = None

        self._last_action = "idle"
        self._note = ""
        self._target: Any = None
        self._ahead = (False, False, False)
        self._next_probe = 0.0

        self.waypoints: Deque[Tuple[float, float, float]] = deque()   # a path the Lead can feed

        self._latency: Deque[float] = deque(maxlen=240)
        self._rpc_calls = 0
        self._rpc_errors = 0
        self._using = False
        self._mining = False

    # --- lifecycle -------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def busy(self) -> bool:
        return self._job is not None or bool(self._queue)

    @property
    def state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self) -> "PlayerBody":
        """Start the loop (commands start it too; this is for an idle body that should keep sensing)."""
        self._ensure_thread()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        """Drop the current command, let go of every key and end the loop. Safe to call twice."""
        self.cancel()
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        with self._lock:
            leftover, self._job = self._job, None
        self._abandon(leftover)
        self._rpc_safe("control.stop", {})
        self._release_inputs()
        self._flush_inputs()

    @staticmethod
    def _abandon(job: Optional[_Job]) -> None:
        """Let a caller waiting on a job go, without running the rest of it."""
        if job is None:
            return
        job.result = False
        job.note = "cancelled"
        job.done.set()

    def cancel(self) -> None:
        """Abandon the running command (keys up) but keep the loop and sensing alive."""
        with self._lock:
            if self._job is not None:
                self._job.cancel.set()
            pending = list(self._queue)
            self._queue.clear()
            self.waypoints.clear()
        for job in pending:
            job.result = False
            job.note = "cancelled"
            job.done.set()
        self._release_inputs()
        self._flush_inputs()
        self._release_interactions()

    def __enter__(self) -> "PlayerBody":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._closed = False
            self._thread = threading.Thread(target=self._run, name="noa-mc-body", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        leftover: Optional[_Job] = None
        while True:
            with self._lock:
                if self._closed:
                    leftover, self._job = self._job, None
                    break
            started = self._now()
            try:
                self._tick_once()
            except Exception:                       # a broken frame must not kill the body
                logger.exception("mc body: tick failed")
            spent = self._now() - started
            self._sleep(max(0.0, self.interval - spent))
        self._abandon(leftover)
        self._release_inputs()

    # --- commands (blocking; any thread; one at a time) ------------------------------------

    def look_at(self, x: float, y: float, z: float, tolerance: float = 2.0, timeout: float = 2.5) -> bool:
        """Turn the camera to a point over several frames; True when it is within `tolerance` degrees."""
        target = (float(x), float(y), float(z))

        def job():
            return (yield from self._look_at_gen(target, tolerance, timeout))

        return self._submit("look", job(), timeout, target=target)

    def walk_to(self, x: float, y: float, z: float, reach: float = WALK_REACH,
                sprint: Optional[bool] = None, timeout: float = 25.0) -> bool:
        """Walk to a position within `reach` blocks; True if it got there. `sprint=None` sprints on long runs."""
        target = (float(x), float(y), float(z))

        def job():
            return (yield from self._walk_to(target[0], target[1], target[2], float(reach), sprint, float(timeout)))

        return self._submit("walk", job(), timeout, target=target)

    def queue_waypoints(self, points: Iterable[Any]) -> int:
        """Append a path (tuples or dicts with x/y/z) to the waypoint queue; returns the queue length."""
        for point in points:
            self.waypoints.append(_point(point))
        return len(self.waypoints)

    def follow_path(self, points: Optional[Iterable[Any]] = None, reach: float = 1.5,
                    sprint: Optional[bool] = None, timeout: float = 30.0) -> bool:
        """Walk through the queue (optionally appending `points` first); False if any leg fails."""
        if points is not None:
            with self._lock:
                self.waypoints.clear()
            self.queue_waypoints(points)

        def job():
            started = self._now()
            while self.waypoints:
                waypoint = self.waypoints[0]
                left = float(timeout) - (self._now() - started)
                if left <= 0:
                    self._note = "路走不完"
                    return False
                ok = yield from self._walk_to(waypoint[0], waypoint[1], waypoint[2],
                                              float(reach), sprint, min(left, 60.0))
                if not ok:
                    return False
                try:
                    self.waypoints.popleft()
                except IndexError:
                    return False
            return True

        return self._submit("walk", job(), timeout)

    def mine_block(self, x: int, y: int, z: int, timeout: float = 10.0) -> bool:
        """Walk within reach, pick the tool the block wants, look at it and mine until it is air."""
        target = (int(x), int(y), int(z))

        def job():
            return (yield from self._mine_job(target[0], target[1], target[2], float(timeout)))

        return self._submit("mine", job(), timeout + 8.0, target=target)

    def place_block(self, x: int, y: int, z: int, item: Optional[str] = None, timeout: float = 8.0) -> bool:
        """Put a block into an empty cell, clicking the face of a neighbour the way a player does."""
        target = (int(x), int(y), int(z))

        def job():
            return (yield from self._place_job(target[0], target[1], target[2], item, float(timeout)))

        return self._submit("place", job(), timeout + 8.0, target=target)

    def attack(self, target: Any, times: int = 1, reach: float = ATTACK_REACH) -> bool:
        """Hit an entity `times` times. `target` is a uuid string or a dict {uuid, x, y, z}."""
        uuid = target if isinstance(target, str) else str((target or {}).get("uuid") or "")
        where = None
        if isinstance(target, dict) and all(key in target for key in ("x", "y", "z")):
            where = _point(target)
        timeout = max(3.0, times * SWING_S + 8.0)

        def job():
            return (yield from self._attack_job(uuid, where, int(times), float(reach)))

        return self._submit("attack", job(), timeout, target=uuid or None)

    def use_item(self, seconds: float = 1.0, hand: str = "main") -> bool:
        """Hold the use button (eating, a bow, a bucket) for `seconds` and let go."""
        def job():
            if self._rpc_safe("control.startUsing", {"hand": hand}) is None:
                self._note = "用不了手上的东西"
                return False
            self._using = True
            yield from self._wait(max(0.0, seconds), "use")
            self._rpc_safe("control.stopUsing", {})
            self._using = False
            return True

        return self._submit("use", job(), seconds + 4.0)

    def jump(self) -> bool:
        """One jump, like tapping space once."""
        def job():
            self._rpc_safe("control.jumpOnce", {})
            yield "jump"
            return True

        return self._submit("jump", job(), 2.0)

    # --- one frame -------------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """Advance the body by one frame by hand (tests / a caller that wants to drive it)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("the body loop is already ticking; call it from the loop instead")
        return self._tick_once()

    def _tick_once(self) -> Dict[str, Any]:
        now = self._now()
        self._dt = _clamp(now - self._last_tick_t, 0.005, 0.25) if self._last_tick_t is not None else self.interval
        self._last_tick_t = now
        self._ticks += 1

        state = self._read_state()
        self._tick_times.append(now)

        with self._lock:
            if self._job is None and self._queue:
                self._job = self._queue.popleft()
            job = self._job

        action = "idle"
        if self._state_failures:
            # Never advance a command from stale coordinates: it could re-press forward immediately
            # after _read_state released it. Resume only on a fresh game observation.
            self._release_inputs()
            if self._state_failures >= 3 and job is not None:
                self._finish(job, False, "感知中断，停止当前动作")
            action = "disconnected"
        elif job is not None:
            self._target = job.target
            if job.cancel.is_set():
                self._finish(job, False, "cancelled")
            else:
                try:
                    action = next(job.gen) or job.kind
                except StopIteration as end:
                    self._finish(job, True if end.value is None else bool(end.value))
                    action = job.kind
                except Exception:
                    logger.exception("mc body: %s failed", job.kind)
                    self._finish(job, False, "error")
                    action = job.kind
        else:
            if self.skills_busy():
                if self._driving():
                    self._rpc_safe("control.stop", {})
                self._release_inputs()
                action = "yield"
            else:
                self._release_inputs()
                action = "idle"

        self._flush_inputs()
        event = self._event(action, state)
        self._last_action = event["action"]
        try:
            self.on_event(event)
        except Exception:
            logger.debug("mc body: on_event raised", exc_info=True)
        return event

    def _read_state(self) -> Dict[str, Any]:
        state = self._rpc_safe("player.getState", {})
        valid = isinstance(state, dict) and all(isinstance(state.get(k), (int, float))
                                               and math.isfinite(state[k]) for k in ("x", "y", "z"))
        if valid:
            self._state_failures = 0
            with self._lock:
                old_dim, new_dim = self._state.get("dimension"), state.get("dimension")
                if old_dim and new_dim and old_dim != new_dim:
                    if self._job is not None:
                        self._job.cancel.set()
                    self._release_inputs()
                self._state = state
            yaw, pitch = state.get("yaw"), state.get("pitch")
            if yaw is not None and (self._cmd_yaw is None or self._job is None):
                self._cmd_yaw = float(yaw)
                self._cmd_pitch = _clamp(_num(state, "pitch"), -90.0, 90.0)
            return self._state
        # The bridge can be busy or the client paused: keep the last known body rather than dying,
        # but let go of the keys if it stays unavailable - blind keys are how a bot walks into lava.
        self._state_failures += 1
        if self._state_failures in (3, 6):
            logger.warning("mc body: no state from the client (%d times), releasing keys", self._state_failures)
            self._rpc_safe("control.stop", {})
            self._release_inputs()
        return self._state

    def _event(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        pos = (_num(state, "x"), _num(state, "y"), _num(state, "z"))
        moved = 0.0
        if self._last_event_pos is not None:
            moved = math.dist(pos, self._last_event_pos)
        self._last_event_pos = pos
        fresh = self._state_failures == 0
        using_item = bool(state.get("usingItem", False)) if fresh else None
        sneaking = bool(state.get("sneaking", False)) if fresh else None
        # Reflexes and the bridge's own navigator can act while no body command is running.
        # Show only observations from a successful player-state read, not a scheduled timer.
        if fresh and action in ("idle", "yield"):
            if using_item:
                action = "use"
            elif moved > 0.025:
                action = "walk"
        event = {
            "event": "tick",
            "t": self._now(),
            "tick": self._ticks,
            "action": action,
            "busy": self._job is not None,
            "pos": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "moved": moved,
            "yaw": round(self._cmd_yaw if self._cmd_yaw is not None else _num(state, "yaw"), 3),
            "pitch": round(self._cmd_pitch, 3),
            "onGround": bool(state.get("onGround", True)),
            "health": _num(state, "health", 20.0),
            "food": int(_num(state, "food", 20)),
            "sprinting": bool(state.get("sprinting", False)),
            "usingItem": using_item,
            "usingHand": state.get("usedItemHand") if fresh else None,
            "hand": {k: v for k, v in state.get("hand", {}).items() if k in ("id", "count", "name", "damage", "maxDamage")} if fresh and isinstance(state.get("hand"), dict) else None,
            "offhand": {k: v for k, v in state.get("offhand", {}).items() if k in ("id", "count", "name", "damage", "maxDamage")} if fresh and isinstance(state.get("offhand"), dict) else None,
            "sneaking": sneaking,
            "blockLight": state.get("blockLight") if fresh else None,
            "skyLight": state.get("skyLight") if fresh else None,
            "isNight": state.get("isNight") if fresh else None,
            "dimension": str(state.get("dimension") or ""),
            "target": list(self._target) if isinstance(self._target, (tuple, list)) else self._target,
            "rpc_ms": round(self._latency[-1], 3) if self._latency else 0.0,
            "tick_rate": round(self._tick_rate(), 1),
            "note": self._note,
        }
        return event

    # --- stats -----------------------------------------------------------------------------

    def _tick_rate(self) -> float:
        times = list(self._tick_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """What the body is doing and how the bridge is behaving (for logs / her toolbox)."""
        latencies = sorted(self._latency)
        count = len(latencies)
        average = sum(latencies) / count if count else 0.0
        p95 = latencies[min(count - 1, int(round(0.95 * (count - 1))))] if count else 0.0
        return {
            "running": self.running,
            "busy": self.busy,
            "action": self._last_action,
            "target": list(self._target) if isinstance(self._target, (tuple, list)) else self._target,
            "note": self._note,
            "ticks": self._ticks,
            "tick_rate": round(self._tick_rate(), 1),
            "rpc_calls": self._rpc_calls,
            "rpc_errors": self._rpc_errors,
            "rpc_avg_ms": round(average, 2),
            "rpc_p95_ms": round(p95, 2),
            "waypoints": len(self.waypoints),
        }

    def status(self) -> str:
        """One line in her words, for a tool result."""
        names = {"idle": "站着", "yield": "让给别的动作", "walk": "走路", "look": "看", "mine": "挖方块",
                 "place": "放方块", "attack": "打", "use": "用东西", "jump": "跳"}
        doing = names.get(self._last_action, self._last_action)
        rate = self._tick_rate()
        return (f"现在{doing}（{rate:.0f}Hz，桥平均 {self.stats()['rpc_avg_ms']:.1f}ms）"
                + (f"：{self._note}" if self._note else ""))

    # --- bridge and keys -------------------------------------------------------------------

    def _rpc_safe(self, method: str, params: Dict[str, Any]) -> Any:
        """One bridge call: timed, serialised, and never raises into the loop."""
        started = self._perf()
        try:
            with self._io_lock:
                return self.rpc(method, params)
        except Exception:
            self._rpc_errors += 1
            logger.warning("mc body: rpc %s failed", method, exc_info=True)
            return None
        finally:
            self._latency.append((self._perf() - started) * 1000.0)
            self._rpc_calls += 1

    def _drive(self, forward: bool = False, back: bool = False, left: bool = False, right: bool = False,
               sprint: bool = False, sneak: bool = False) -> None:
        self._want.update({"forward": forward, "back": back, "left": left, "right": right,
                           "sprint": sprint, "sneak": sneak})

    def _release_inputs(self) -> None:
        for key in self._want:
            self._want[key] = False
        self._jump_ticks = 0

    def _release_interactions(self) -> None:
        """Mining and right-click are latched in the client independently of movement keys."""
        if self._using:
            self._rpc_safe("control.stopUsing", {})
            self._using = False
        if self._mining:
            self._rpc_safe("interact.stopBreaking", {})
            self._mining = False

    def _driving(self) -> bool:
        return any(self._want.values()) or self._jump_ticks > 0

    def _flush_inputs(self) -> None:
        """Send the keys only when they change - the client holds them between ticks."""
        want = dict(self._want)
        if self._jump_ticks > 0:
            want["jump"] = True
            self._jump_ticks -= 1
        if want != self._sent:
            result = self._rpc_safe("control.setInput", dict(want))
            if result is not None and not (isinstance(result, dict) and result.get("ok") is False):
                self._sent = want

    def _submit(self, kind: str, gen: Any, timeout: Optional[float], target: Any = None) -> bool:
        if threading.current_thread() is self._thread:
            raise RuntimeError("PlayerBody commands cannot be issued from on_event / the body loop")
        job = _Job(kind, gen, target)
        self._ensure_thread()
        with self._lock:
            self._queue.append(job)
        wait_for = None if timeout is None else max(0.1, float(timeout)) + 2.5
        if not job.done.wait(wait_for):
            job.cancel.set()
            with self._lock:
                try:
                    self._queue.remove(job)
                except ValueError:
                    pass
            return False
        self._note = job.note or self._note
        return bool(job.result)

    def _finish(self, job: _Job, ok: bool, note: str = "") -> None:
        with self._lock:
            if self._job is job:
                self._job = None
        job.result = bool(ok)
        job.note = note
        if note:
            self._note = note
        self._release_inputs()
        self._release_interactions()
        self._release_look()
        self._flush_inputs()
        job.done.set()

    def _cancelled(self) -> bool:
        job = self._job
        return bool(job is not None and job.cancel.is_set())

    # --- looking ---------------------------------------------------------------------------

    def _eye(self) -> Tuple[float, float, float]:
        state = self._state
        return (_num(state, "x"), _num(state, "y") + EYE_HEIGHT, _num(state, "z"))

    def _look_target(self, x: float, y: float, z: float) -> Tuple[float, float]:
        ex, ey, ez = self._eye()
        dx, dy, dz = float(x) - ex, float(y) - ey, float(z) - ez
        horizontal = math.hypot(dx, dz) or 1e-6
        yaw = math.degrees(math.atan2(dz, dx)) - 90.0
        pitch = -math.degrees(math.atan2(dy, horizontal))
        return yaw, _clamp(pitch, -90.0, 90.0)

    def _turn(self, yaw: float, pitch: float, tolerance: float = 2.0) -> bool:
        """One frame of a human mouse move toward (yaw, pitch); True once close enough."""
        if self._smooth is not False or self._now() >= self._smooth_retry:
            done = self._turn_smooth(yaw, pitch, tolerance)
            if done is not None:
                return done
        if self._cmd_yaw is None:
            state = self._state
            self._cmd_yaw = _num(state, "yaw")
            self._cmd_pitch = _clamp(_num(state, "pitch"), -90.0, 90.0)
        delta_yaw = _wrap180(yaw - self._cmd_yaw)
        delta_pitch = pitch - self._cmd_pitch
        if abs(delta_yaw) <= tolerance and abs(delta_pitch) <= tolerance:
            self._turn_velocity = [0.0, 0.0]
            return True

        now = self._now()
        dt = min(.1, max(.01, self._dt))
        if self._last_turn_t is None or now - self._last_turn_t > .35:
            self._turn_velocity = [0.0, 0.0]
        acceleration = MAX_TURN_DEG_S / TURN_ACCEL_S
        deltas = (delta_yaw, delta_pitch)
        moves = []
        for axis, remaining in enumerate(deltas):
            # Braking distance bounds desired speed; persistent velocity bounds acceleration,
            # including a target reversal while walking. Never teleport the last large angle.
            desired = math.copysign(min(MAX_TURN_DEG_S, math.sqrt(2 * acceleration * abs(remaining))), remaining)
            old = self._turn_velocity[axis]
            speed = old + _clamp(desired - old, -acceleration * dt, acceleration * dt)
            move = speed * dt
            if move * remaining > 0 and abs(move) >= abs(remaining):
                move, speed = remaining, 0.0
            moves.append(move)
            self._turn_velocity[axis] = speed
        next_yaw = _wrap180(self._cmd_yaw + moves[0])
        next_pitch = _clamp(self._cmd_pitch + moves[1], -90.0, 90.0)
        self._last_turn_t = now
        result = self._rpc_safe("control.look", {"yaw": round(next_yaw, 3), "pitch": round(next_pitch, 3)})
        if result is not None and not (isinstance(result, dict) and result.get("ok") is False):
            self._cmd_yaw, self._cmd_pitch = next_yaw, next_pitch
        return False

    def _turn_smooth(self, yaw: float, pitch: float, tolerance: float) -> Optional[bool]:
        """Hand the target to the mod's per-frame camera; None when the bridge cannot (then the loop turns)."""
        state = self._state
        actual_yaw, actual_pitch = _num(state, "yaw"), _clamp(_num(state, "pitch"), -90.0, 90.0)
        sent = self._smooth_sent
        if sent is None or abs(_wrap180(yaw - sent[0])) > 0.5 or abs(pitch - sent[1]) > 0.5:
            result = self._rpc_safe("control.lookTarget", {"yaw": round(yaw, 2), "pitch": round(pitch, 2),
                                                           "speed": MAX_TURN_DEG_S})
            # only a mod that really turns the camera reports how far it still has to go
            if not (isinstance(result, dict) and "remaining" in result):
                self._smooth, self._smooth_retry, self._smooth_sent = False, self._now() + 60.0, None
                return None
            self._smooth, self._smooth_sent = True, (yaw, pitch)
            if isinstance(result, dict) and isinstance(result.get("yaw"), (int, float)):
                actual_yaw, actual_pitch = float(result["yaw"]), float(result.get("pitch", actual_pitch))
        self._cmd_yaw, self._cmd_pitch = actual_yaw, actual_pitch
        return abs(_wrap180(yaw - actual_yaw)) <= tolerance and abs(pitch - actual_pitch) <= tolerance

    def _release_look(self) -> None:
        """A finished command lets go of the camera, so navigation (or the player) can turn it again."""
        if self._smooth and self._smooth_sent is not None:
            self._rpc_safe("control.lookTarget", {"clear": True})
        self._smooth_sent = None

    def _look_toward(self, x: float, y: float, z: float, tolerance: float = 2.0) -> bool:
        yaw, pitch = self._look_target(x, y, z)
        return self._turn(yaw, pitch, tolerance)

    def _look_at_gen(self, point: Tuple[float, float, float], tolerance: float, timeout: float):
        end = self._now() + max(0.1, float(timeout))
        while not self._cancelled():
            if self._look_toward(point[0], point[1], point[2], tolerance):
                return True
            if self._now() > end:
                self._note = "转不过去"
                return False
            yield "look"
        return False

    # --- walking ---------------------------------------------------------------------------

    def _probe_ahead(self, px: float, py: float, pz: float, yaw: float) -> Tuple[bool, bool, bool]:
        """(blocked by a two-high wall, one-block step to jump, cliff) in front of the feet, cached ~0.12 s."""
        now = self._now()
        if now < self._next_probe:
            return self._ahead
        self._next_probe = now + 0.12
        radians = math.radians(yaw + 90.0)
        lookahead = max(CLIFF_LOOKAHEAD, 1.3 if self._want.get("sprint") else .8)
        cell_x = math.floor(px + math.cos(radians) * lookahead)
        cell_z = math.floor(pz + math.sin(radians) * lookahead)
        feet_y = math.floor(py)
        positions = [{"x": cell_x, "y": feet_y, "z": cell_z},
                     {"x": cell_x, "y": feet_y + 1, "z": cell_z},
                     {"x": cell_x, "y": feet_y - 1, "z": cell_z},
                     {"x": cell_x, "y": feet_y - 2, "z": cell_z}]
        got = self._rpc_safe("perception.blocks", {"positions": positions})
        blocks = (got or {}).get("blocks") or []
        if len(blocks) < 4 or any(b.get("loaded") is False for b in blocks):
            self._ahead = (True, False, True)           # unknown ground is not a safe route
            return self._ahead
        if any(_block_name(b).split(":")[-1] in ("lava", "fire", "soul_fire", "magma_block",
                                                 "campfire", "soul_campfire", "powder_snow")
               for b in blocks):
            self._ahead = (True, False, True)
            return self._ahead
        solid = [_solid(block) for block in blocks[:4]]
        blocked = solid[0] and solid[1]
        step_up = solid[0] and not solid[1]
        cliff = (not solid[0]) and (not solid[2]) and (not solid[3])
        self._ahead = (blocked, step_up, cliff)
        return self._ahead

    def _walk_to(self, x: float, y: float, z: float, reach: float, sprint: Optional[bool], timeout: float):
        """Generator: steer forward toward (x, y, z), jumping steps and keeping off cliffs; yields per frame."""
        started = self._now()
        last_dist: Optional[float] = None
        stall_since = started
        last_jump = 0.0
        blocked_since = 0.0
        side_until = 0.0
        side_left = True
        while not self._cancelled():
            now = self._now()
            if now - started > timeout:
                self._note = "走路超时"
                self._release_inputs()
                return False
            state = self._state
            px, py, pz = _num(state, "x"), _num(state, "y"), _num(state, "z")
            dx, dz = x - px, z - pz
            distance = math.hypot(dx, dz)
            if distance <= reach and abs(y - py) <= 1.25:
                self._note = ""
                self._release_inputs()
                return True

            desired_yaw = math.degrees(math.atan2(dz, dx)) - 90.0
            self._turn(desired_yaw, 0.0, tolerance=6.0)          # walk looking ahead, not at the floor
            # A probe in the target direction says nothing about the old camera direction.
            # Turn in place first instead of walking sideways/off a ledge during a 180-degree turn.
            if abs(_wrap180(desired_yaw - (self._cmd_yaw or 0.0))) > 25.0:
                self._release_inputs()
                yield "walk"
                continue
            blocked, step_up, cliff = self._probe_ahead(px, py, pz, desired_yaw)

            if last_dist is None or distance < last_dist - 0.02:
                last_dist = distance
                stall_since = now
            stalled = now - stall_since

            if step_up or stalled > STUCK_JUMP_AFTER:
                if now - last_jump > 0.45:
                    self._jump_ticks = 2
                    last_jump = now
                    self._note = "前面有台阶，跳一下"
            if blocked:
                blocked_since = blocked_since or now
                if now - blocked_since > STUCK_GIVE_UP_AFTER:
                    self._note = "前面挡死了，走不过去"
                    self._release_inputs()
                    return False
            else:
                blocked_since = 0.0
            if stalled > STUCK_GIVE_UP_AFTER + 1.0 and not step_up:
                self._note = "走不过去"
                self._release_inputs()
                return False

            if cliff:                         # ask the route planner; unprobed strafing is not safe
                self._release_inputs()
                self._note = "前方有坑、危险方块或未加载地形，需要重新找路"
                return False
            elif blocked and now - blocked_since > STUCK_SIDE_AFTER:
                self._note = "绕着挡路的方块走"
                if now >= side_until:
                    side_left = not side_left
                    side_until = now + 0.5
                self._next_probe = 0.0
                side_blocked, side_step, side_cliff = self._probe_ahead(
                    px, py, pz, desired_yaw + (-90 if side_left else 90))
                self._next_probe = 0.0
                if side_cliff or side_blocked or side_step:
                    self._release_inputs()
                    self._note = "侧边也没有安全绕路位置"
                    return False
                self._drive(forward=False, left=side_left, right=not side_left)
            else:
                sprint_now = (bool(sprint) if sprint is not None else (distance > 6.0 and abs(y - py) < 1.5))
                sprint_now = sprint_now and _num(state, "food", 20) > 6
                self._drive(forward=True, sprint=sprint_now)
            yield "walk"
        return False

    # --- blocks ----------------------------------------------------------------------------

    def _block_at(self, x: int, y: int, z: int, tool: bool = False) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {"positions": [{"x": int(x), "y": int(y), "z": int(z)}]}
        if tool:
            params["tool"] = True
        got = self._rpc_safe("perception.blocks", params)
        blocks = (got or {}).get("blocks") or []
        return blocks[0] if blocks else None

    def _face_point(self, x: int, y: int, z: int) -> Tuple[float, float, float]:
        """A point on the side of the block that faces her eyes - where a crosshair would land.

        Of the sides turned toward her, one with air in front of it: climbing a stair, the stone over the next
        step has its underside against more stone and its near side open to the hole she is standing in."""
        centre = (x + 0.5, y + 0.5, z + 0.5)
        eye = self._eye()
        delta = (eye[0] - centre[0], eye[1] - centre[1], eye[2] - centre[2])
        axes = sorted((i for i in range(3) if abs(delta[i]) > 0.05), key=lambda i: -abs(delta[i])) or [1]
        if len(axes) > 1:
            fronts = []
            for i in axes:
                n = [int(x), int(y), int(z)]
                n[i] += 1 if delta[i] > 0 else -1
                fronts.append({"x": n[0], "y": n[1], "z": n[2]})
            got = (self._rpc_safe("perception.blocks", {"positions": fronts}) or {}).get("blocks") or []
            if len(got) == len(axes):
                axes = [i for i, b in zip(axes, got) if _see_through(b)] +                        [i for i, b in zip(axes, got) if not _see_through(b)]
        point = list(centre)
        point[axes[0]] += 0.49 if delta[axes[0]] > 0 else -0.49
        return (point[0], point[1], point[2])

    def _sight_blocker(self, x: int, y: int, z: int) -> Optional[str]:
        """What is between her eyes and the block instead of the block (walls, leaves); None if clear."""
        eye = self._eye()
        aim = self._face_point(x, y, z)
        length = math.dist(eye, aim)
        steps = max(2, int(length / 0.2))
        cells: List[Tuple[int, int, int]] = []
        for i in range(1, steps):
            fraction = i / steps
            cell = (math.floor(eye[0] + (aim[0] - eye[0]) * fraction),
                    math.floor(eye[1] + (aim[1] - eye[1]) * fraction),
                    math.floor(eye[2] + (aim[2] - eye[2]) * fraction))
            if cell != (int(x), int(y), int(z)) and cell not in cells:
                cells.append(cell)
        if not cells:
            return None
        got = self._rpc_safe("perception.blocks",
                             {"positions": [{"x": c[0], "y": c[1], "z": c[2]} for c in cells]})
        if got is None:
            return None                                   # cannot tell; let the client decide
        for cell, block in zip(cells, got.get("blocks") or []):
            if _solid(block) and not _see_through(block):
                return _block_name(block).split(":")[-1] or "方块"
        return None

    def _equip(self, item: str) -> bool:
        """Put an item into her hand: select it in the hotbar, or swap it there from the backpack."""
        from src.levels.l8_agency.mc_craft import select_item
        return select_item(self._rpc_safe, item, stopped=self._cancelled, pause=self._sleep)

    def _mine_job(self, x: int, y: int, z: int, timeout: float):
        """Generator: mine one block the way a player does - reach, tool, crosshair, then hold the button."""
        probe = self._block_at(x, y, z, tool=True)
        if probe is None:
            self._note = "看不到那个方块"
            return False
        if probe.get("loaded") is False:
            self._note = "那个位置还没加载出来"
            return False
        if probe.get("air"):
            self._note = "那里已经是空气了"
            return False
        if probe.get("canHarvest") is False:
            self._note = "手上的工具挖不动它"
            return False
        before = _block_name(probe)
        best_slot = probe.get("bestSlot")
        if best_slot is not None:
            self._rpc_safe("inventory.selectHotbar", {"slot": int(best_slot)})

        if math.dist(self._eye(), (x + 0.5, y + 0.5, z + 0.5)) > BREAK_REACH - 0.6 and not (
                yield from self._walk_to(x + 0.5, y, z + 0.5, reach=BREAK_REACH - 0.6,
                                         sprint=None, timeout=min(10.0, max(2.0, timeout)))):
            self._note = "走不到那个方块旁边"
            return False
        aim = self._face_point(x, y, z)
        if not (yield from self._look_at_gen(aim, 3.0, 2.0)):
            self._note = "视线转不过去"
            return False
        blocker = self._sight_blocker(x, y, z)
        if blocker:
            self._note = f"被 {blocker} 挡住了"
            return False
        if self._rpc_safe("interact.breakBlock", {"x": int(x), "y": int(y), "z": int(z), "mode": "survival"}) is None:
            self._note = "没能开始挖"
            return False
        self._mining = True

        end = self._now() + max(2.0, timeout)
        next_poll = self._now() + 0.2
        next_aim = self._now() + 0.4
        while not self._cancelled():
            now = self._now()
            if now > end:
                self._rpc_safe("interact.stopBreaking", {})
                self._note = "挖了很久还没挖掉"
                return False
            if now >= next_poll:
                next_poll = now + 0.2
                current = self._block_at(x, y, z, tool=False)
                if current is None:
                    break
                if current.get("air") or (current.get("id") and _block_name(current) != before):
                    self._note = ""
                    return True
            if now >= next_aim:                          # keep the eyes on it while the arm works
                next_aim = now + 0.4
                self._look_toward(aim[0], aim[1], aim[2], tolerance=3.0)
            yield "mine"
        return False

    def _pick_place_neighbour(self, x: int, y: int, z: int) -> Optional[Tuple[int, int, int, str]]:
        """A solid block next to the empty cell and the face of it to click (floor first, then walls)."""
        eye = self._eye()
        positions, meta = [], []
        for (ox, oy, oz), face in PLACE_NEIGHBOURS:
            positions.append({"x": x + ox, "y": y + oy, "z": z + oz})
            meta.append((x + ox, y + oy, z + oz, face))
        got = self._rpc_safe("perception.blocks", {"positions": positions})
        blocks = (got or {}).get("blocks") or []
        candidates = []
        for (bx, by, bz, face), block in zip(meta, blocks):
            if not _solid(block) or _block_name(block).split(":")[-1] in ("lava", "water"):
                continue
            distance = math.dist(eye, (bx + 0.5, by + 0.5, bz + 0.5))
            if distance > PLACE_REACH:
                continue
            candidates.append(((bx, by, bz, face), 0 if face == "up" else 1, distance))
        if not candidates:
            return None
        candidates.sort(key=lambda entry: (entry[1], entry[2]))
        return candidates[0][0]

    def _place_job(self, x: int, y: int, z: int, item: Optional[str], timeout: float):
        """Generator: put a block into an empty cell by clicking the face of a block next to it."""
        current = self._block_at(x, y, z, tool=False)
        if current is not None and current.get("loaded") is False:
            self._note = "要放的位置还没加载出来"
            return False
        if current is not None and not current.get("air"):
            self._note = "那里已经有方块了"
            return False
        pick = self._pick_place_neighbour(x, y, z)
        if pick is None:
            # A roof/frame target is above the feet. Approach its horizontal position at the
            # current standing height instead of asking the walking loop to stand in mid-air.
            reached = yield from self._walk_to(x + .5, _num(self._state, "y"), z + .5,
                                               reach=2.5, sprint=None, timeout=min(8.0, timeout))
            if reached:
                pick = self._pick_place_neighbour(x, y, z)
        if pick is None:
            self._note = "旁边没有能贴的方块"
            return False
        click_x, click_y, click_z, face = pick
        if item and not self._equip(item):
            self._note = f"背包里没有 {item}"
            return False
        offsets = {"up": (0, 1, 0), "down": (0, -1, 0), "north": (0, 0, -1), "south": (0, 0, 1),
                   "east": (1, 0, 0), "west": (-1, 0, 0)}
        ox, oy, oz = offsets[face]
        click_point = (click_x + 0.5 + ox * 0.5, click_y + 0.5 + oy * 0.5, click_z + 0.5 + oz * 0.5)
        if not (yield from self._look_at_gen(click_point, 3.0, 2.0)):
            self._note = "视线转不过去"
            return False
        blocker = self._sight_blocker(click_x, click_y, click_z)
        if blocker:
            self._note = f"放置面被 {blocker} 挡住了"
            return False
        result = self._rpc_safe("interact.placeBlock",
                                {"x": int(click_x), "y": int(click_y), "z": int(click_z), "face": face})
        if result is None:
            self._note = "没能放下去"
            return False
        if result.get("placed") is False or result.get("result") == "OUT_OF_REACH":
            self._note = "服务器没放上去"
            return False
        # A successful click is not placement. Wait for the expected block to be observed.
        end = self._now() + min(3.0, max(.5, timeout))
        wanted = str(item or "").split(":")[-1]
        while self._now() < end and not self._cancelled():
            yield from self._wait(.18, "place")
            check = self._block_at(x, y, z, tool=False)
            observed = _block_name(check or {}).split(":")[-1].split("/")[-1]
            matches = not wanted or observed == wanted or wanted == "torch" and observed == "wall_torch"
            if check and check.get("loaded", True) and not check.get("air") and matches:
                self._note = ""
                return True
        self._note = "点击后没有观察到目标方块，未确认放置"
        return False

    # --- fighting and waiting ---------------------------------------------------------------

    def _wait(self, seconds: float, label: str):
        end = self._now() + max(0.0, float(seconds))
        while not self._cancelled() and self._now() < end:
            yield label

    def _attack_job(self, uuid: str, where: Optional[Tuple[float, float, float]], times: int, reach: float):
        """Generator: get close, look at the target, then swing at a human cadence."""
        if not uuid:
            self._note = "没有攻击目标"
            return False
        for hit in range(max(1, times)):
            if self._cancelled():
                return False
            if where is not None:
                if not (yield from self._walk_to(where[0], where[1], where[2], reach=reach,
                                                 sprint=None, timeout=6.0)):
                    self._note = "追不上目标"
                    return False
                aim = (where[0], where[1] + 1.2, where[2])        # aim at the body, not the feet
                if not (yield from self._look_at_gen(aim, 4.0, 1.5)):
                    self._note = "看不到目标"
                    return False
            if self._rpc_safe("interact.attackEntity", {"uuid": uuid}) is None:
                self._note = "打不到"
                return False
            if hit + 1 < times:
                yield from self._wait(SWING_S, "attack")
        self._note = ""
        return True
