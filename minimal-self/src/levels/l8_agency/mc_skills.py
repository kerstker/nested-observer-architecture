"""
Level 8i: what her hands know how to do without thinking - skills for the game body.

Thinking (a model call) takes a second or two; a person playing does not think about every swing of the axe.
A skill is one whole, ordinary thing - chop down a tree, pick up what dropped, go over to someone - done by
code against her client (the mcpfabric bridge), in the background, while her mind is free to talk. She starts
one with a single tool call, can see how it is going, and can stop it.

Everything here goes through the same bridge and the same access rules as her other game tools, and only
does what a player can do (walk, break a block in survival, walk over items).
"""
import logging
import math
import threading
import time
from functools import wraps
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from src.levels.l8_agency.mc_food import FOOD_NAMES, is_edible
from src.levels.l8_agency.mc_training import observe, evaluate

logger = logging.getLogger(__name__)

LOG_SUFFIX = "_log"
REACH = 4.5            # how far a survival player can break a block
TRUNK_HEIGHT = 14      # a tall tree, not a skyscraper


SEE_THROUGH = ("air", "grass", "fern", "flower", "torch", "snow", "sapling", "vine", "petals", "rail", "button",
               "carpet", "leaf_litter", "bush", "pressure_plate", "tripwire", "glass")
SOFT = ("leaves", "grass", "fern", "flower", "vine", "bush", "snow", "leaf_litter")    # a player just knocks these out of the way


def face_point(eye: tuple, target: tuple, rpc=None) -> tuple:
    """Where a player looks to hit a block: the middle of a face turned toward them that is open (a face pressed
    against the next block of the trunk cannot be seen). Without `rpc`, the face most turned toward them."""
    c = (target[0] + 0.5, target[1] + 0.5, target[2] + 0.5)
    d = (eye[0] - c[0], eye[1] - c[1], eye[2] - c[2])
    faces = sorted((i for i in range(3) if abs(d[i]) > 0.05), key=lambda i: -abs(d[i])) or [1]
    if rpc is not None and len(faces) > 1:
        nbrs = []
        for i in faces:
            n = list(target)
            n[i] += 1 if d[i] > 0 else -1
            nbrs.append({"x": n[0], "y": n[1], "z": n[2]})
        got = (rpc("perception.blocks", {"positions": nbrs}) or {}).get("blocks") or []
        open_faces = [i for i, b in zip(faces, got)
                      if any(s in str(b.get("id") or "minecraft:air").split(":")[-1] for s in SEE_THROUGH)]
        faces = open_faces + [i for i in faces if i not in open_faces]
    p = list(c)
    p[faces[0]] += 0.49 if d[faces[0]] > 0 else -0.49
    return tuple(p)


def sight_cells(eye: tuple, target: tuple, aim: tuple = None) -> List[tuple]:
    """The block cells the line from her eye to the point she aims at on a block passes through (target excluded)."""
    tx, ty, tz = aim or face_point(eye, target)
    dist = math.dist(eye, (tx, ty, tz))
    n = max(2, int(dist / 0.15))
    out: List[tuple] = []
    for i in range(1, n):
        f = i / n
        c = (math.floor(eye[0] + (tx - eye[0]) * f), math.floor(eye[1] + (ty - eye[1]) * f),
             math.floor(eye[2] + (tz - eye[2]) * f))
        if c != tuple(target) and c not in out:
            out.append(c)
    return out


def first_obstacle(rpc, eye: tuple, target: tuple) -> Optional[tuple]:
    """The first block between her eye and the target that a crosshair would hit instead: (x, y, z, id) or None."""
    cells = sight_cells(eye, target, face_point(eye, target, rpc))
    if not cells:
        return None
    got = (rpc("perception.blocks", {"positions": [{"x": c[0], "y": c[1], "z": c[2]} for c in cells]}) or {}).get("blocks") or []
    for c, b in zip(cells, got):
        block = str(b.get("id") or "")
        name = block.split(":")[-1]
        if block and not any(s in name for s in SEE_THROUGH):
            return (c[0], c[1], c[2], block)
    return None


def bounded_excursion(seconds=90, blocks=72, recovery=False):
    """One budget spans nested mining calls; a new planner decision starts the next excursion."""
    def decorate(fn):
        @wraps(fn)
        def run(self, *args, **kwargs):
            outer = self._work is None
            if outer:
                self._next_supply_check = 0.0
                me = self.rpc("player.getState", {}) or {}
                self._work = {"kind": fn.__name__, "started": time.monotonic(), "deadline": time.monotonic() + seconds,
                              "limit": blocks, "broken": 0, "progress": 0, "reason": "", "recovery": recovery,
                              "origin": {k: me[k] for k in ("x", "y", "z") if k in me}, "last_progress_at": time.time()}
                if self.current == "goal":
                    self._work["deadline"] = min(self._work["deadline"], getattr(self, "_goal_deadline", float("inf")))
            try:
                if not self._work_guard():
                    return "本轮作业暂停：" + self._work["reason"]
                result = fn(self, *args, **kwargs)
                if outer and self._work["reason"]:
                    result += " 本轮作业暂停：" + self._work["reason"]
                return result
            finally:
                if outer:
                    work = self._work
                    self.last_outcome = {"kind": work["kind"], "has_progress": work["progress"] > 0,
                                         "progress": work["progress"], "blocks_broken": work["broken"],
                                         "reason": work["reason"], "return_position": work["origin"],
                                         "last_progress_at": work["last_progress_at"],
                                         "needs_supply": any(q in work["reason"] for q in ("生命", "饱食", "背包", "耐久"))}
                    self._work = None
        return run
    return decorate


class Skills:
    def __init__(self, rpc: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]], who: Callable[[str], Optional[Dict]] = None,
                 patterns=None, index=None):
        self.rpc = rpc                      # method, params -> result dict or None
        self.find_player = who              # name -> entity dict (x, y, z, distance) or None
        self.index = index                  # the modpack's own data (mc_knowledge.ModpackIndex), if there is one
        self.goal_item: Optional[str] = None     # what she is working toward, and who chose it ("self" / "auto")
        self.goal_count = 1
        self.goal_by = ""
        self.goal_note = ""                  # the step it is on, or why it stopped
        self.goal_stuck = False
        from src.levels.l8_agency.mc_craft import Crafter, PatternIndex
        self.crafter = Crafter(rpc, self, patterns or PatternIndex(None), stop=lambda: self._stop.is_set())
        self.current: Optional[str] = None
        self.progress = ""
        self.last_result = ""
        self._stop = threading.Event()
        self.failed: Dict[tuple, float] = {}      # trees she could not get to lately: try another one
        self.last_blocked: Optional[tuple] = None
        self.last_reason = ""
        self.using_screen = False           # standing at a furnace / table on purpose right now
        self.pause_until = 0.0              # reflexes (a fight, eating) hold the body until then
        self.heading: Optional[tuple] = None   # the direction of the current mine
        from src.levels.l8_agency.mc_memory import Experience
        self.memory = Experience()              # what her hands did, what she learned (game sets a file)
        self.on_event: Callable[[str], None] = lambda text: None      # big moments, for her episodic memory
        self._made_tool_at: Dict[str, float] = {}
        self._uses = 0
        from src.levels.l8_agency.mc_places import Places
        self.places = Places()                  # her map: home, chests and what is in them, mines, ore (game sets a file)
        self.travel: Optional[Callable[[int, int, int], str]] = None    # long walks (the game wires the walker in)
        self._thread: Optional[threading.Thread] = None
        self.body = None                       # GameBridge supplies the shared 10-20 Hz executor
        self.on_action: Callable[[str, Dict, str], None] = lambda name, args, result: None
        self._home_attempts: Dict[tuple, Dict[str, Any]] = {}
        self._camp_retry_after = 0.0
        self._hunt_retry_after = 0.0
        self._forage_retry_after = 0.0
        self._forage_visited = {}
        self._work = None
        self.last_outcome = {}
        self._next_supply_check = 0.0
        self._explore_seen = {}
        self._route = []
        self._route_dimension = None
        self._last_light_at = 0.0
        self._cave_attempt_after = 0.0
        self._hut_palette = {}
        self._job_lock = threading.RLock()
        self._pending = deque()
        self._job_serial = 0
        self.active_job = None
        self._action_progress_count = 0
        self._last_progress_at = 0.0
        self.last_action = None
        self._action_history = deque(maxlen=64)
        self._action_facts = deque(maxlen=128)
        self._actions_finished = 0

    # --- running ------------------------------------------------------------------------

    def busy(self) -> bool:
        return self.current is not None and self._thread is not None and self._thread.is_alive()

    def start(self, name: str, **args) -> str:
        urgent = bool(args.pop("_emergency", False))
        fn = {"chop_tree": self.chop_tree, "collect_drops": self.collect_drops, "go_to_player": self.go_to_player,
              "mine": self.mine, "craft": self.craft, "place": self.place, "equip": self.equip, "tidy": self.tidy,
              "smelt": self.smelt, "dig_down": self.dig_down, "dig_up": self.dig_up, "escape": self.escape,
              "tunnel": self.tunnel, "explore_cave": self.explore_cave, "light_area": self.light_area, "goal": self.goal,
              "declutter": self.declutter, "dress": self.dress, "hunt": self.hunt, "forage": self.forage, "set_home": self.set_home,
              "go_home": self.go_home, "build_hut": self.build_hut, "build_portal": self.build_portal,
              "enter_portal": self.enter_portal, "hunt_resource": self.hunt_resource,
              "end_step": self.end_step, "fill_water_bucket": self.fill_water_bucket,
              "store": self.store, "go_to": self.go_to, "get_off_pillar": self.get_off_pillar}.get(name)
        if fn is None:
            return (f"不会 {name}。会的有：goal（朝一个目标一步步做）、chop_tree（砍一棵树）、mine（挖某种方块）、craft（合成）、"
                    "smelt（用熔炉烧）、dig_down（挖楼梯往下）、tunnel（挖矿道）、place（放方块）、equip（拿起合适的工具）、"
                    "tidy（整理快捷栏）、collect_drops（捡掉落物）、go_to_player（走到某人身边）。")
        with self._job_lock:
            signature = (name, args)
            if self.busy():
                if self.active_job and (self.active_job["name"], self.active_job["args"]) == signature:
                    return f"已经在做 {name} 了（{self.progress}），不用再开始一次。"
                if any((job["name"], job["args"]) == signature for job in self._pending):
                    return f"{name} 已在队列中，当前动作完成后执行。"
                if len(self._pending) >= 4:
                    return "动作队列已满，先完成已排好的工作。"
                # An explicit priority request is insufficient by itself: verify immediate danger.
                state = self.rpc("player.getState", {}) or {}
                nearby = (self.rpc("perception.entities", {"radius": 5}) or {}).get("entities", []) if urgent else []
                dangerous = (isinstance(state.get("health"), (int, float)) and state["health"] <= 6
                             or state.get("onFire") or state.get("inLava")
                             or any(e.get("kind") == "hostile" and float(e.get("distance", 99)) <= 4 for e in nearby))
                urgent = bool(urgent and dangerous and name in ("escape", "go_home", "go_to"))
                self._job_serial += 1
                job = {"id": self._job_serial, "name": name, "args": dict(args), "fn": fn, "urgent": urgent}
                if urgent:
                    self._pending.appendleft(job)
                    self._stop.set()
                    if self.body is not None:
                        self.body.cancel()
                    self.rpc("nav.stop", {})
                    self.rpc("interact.stopBreaking", {})
                    return f"已确认近身危险，当前动作安全收尾后优先 {name}。"
                self._pending.append(job)
                return f"已排队 {name}（前面 {len(self._pending) - 1} 项），当前 {self.current} 继续执行。"
            if name == "goal" and not args.get("clear"):
                # mc_goal and the fallback scheduler also enter here, bypassing the
                # expert dispatcher's backoff. Check the same canonical goal at the body boundary.
                item = args.get("item") or self.goal_item
                count = args.get("count", 1) if args.get("item") else self.goal_count
                me = self._me() or {}
                pos = tuple(me[k] for k in ("x", "y", "z")) if all(k in me for k in ("x", "y", "z")) else None
                advice = self.memory.advice("goal", {"item": item, "count": count}, pos,
                                            me.get("dimension", self.places.active_dim), time.time())
                if advice:
                    return "这个目标刚在这里失败，仍在冷却；先处理补给或换路线，未开始重复任务。"
            self._job_serial += 1
            first = {"id": self._job_serial, "name": name, "args": dict(args), "fn": fn, "urgent": urgent}
            self.current = name
            self.active_job = {k: v for k, v in first.items() if k != "fn"}
            self.active_job["started_at"] = time.time()
            self._stop.clear()

            def run():
                job = first
                while job is not None:
                    with self._job_lock:
                        self.active_job = {k: v for k, v in job.items() if k != "fn"}
                        self.active_job["started_at"] = time.time()
                        self.current, self.progress = job["name"], "开始了"
                        self._last_progress_at = time.time()
                        self._action_progress_count = 0
                        self.last_outcome = {}
                        self.last_reason = ""
                        self._action_facts.clear()
                    before = observe(self.rpc)
                    contract_args = dict(job["args"])
                    if job["name"] == "goal" and not contract_args.get("item"):
                        contract_args.update(item=self.goal_item, count=self.goal_count)
                    if job["name"] == "go_to":
                        contract_args = {"x": 0, "y": 64, "z": 0, **contract_args}
                    if job["name"] == "go_home":
                        home = self.places.home or {}
                        contract_args.update({k: home[k] for k in ("x", "y", "z") if k in home})
                        contract_args["dimension"] = home.get("dim", "minecraft:overworld")
                    execution_error = False
                    try:
                        self.last_result = job["fn"](**job["args"])
                    except Exception as e:
                        execution_error = True
                        self.last_result = f"{job['name']} 出错了：{e}"
                    finally:
                        after = observe(self.rpc)
                        verdict = evaluate(job["name"], contract_args, before, after,
                                           progress=self._action_progress_count, stopped=self._stop.is_set(),
                                           error=execution_error, reason=self.last_outcome.get("reason", ""),
                                           facts=self._action_facts)
                        # Only explicit game parameters enter the public training trace.
                        safe_args = {k: contract_args[k] for k in ("item", "count", "blocks", "x", "y", "z", "dimension")
                                     if k in contract_args and isinstance(contract_args[k], (str, int, float, bool, type(None)))}
                        with self._job_lock:
                            self.last_action = {"id": job["id"], "name": job["name"], "args": safe_args,
                                                "started_at": before["t"], "ended_at": after["observed_at"],
                                                "before": before, "after": after, **verdict}
                            self._action_history.append(self.last_action)
                            self._actions_finished += 1
                        logger.info(f"[Skill] {job['name']}: {self.last_result}")
                        if job["name"] not in ("goal", "status"):
                            self.memory.did(f"{job['name']}：{self.last_result[:60]}")
                        try:
                            self.on_action(job["name"], dict(job["args"]), self.last_result)
                        except Exception:
                            logger.exception("game action observer failed")
                    with self._job_lock:
                        if self._pending and (not self._stop.is_set() or self._pending[0]["urgent"]):
                            job = self._pending.popleft()
                            self._stop.clear()
                        else:
                            job = None
                            self.current = None
                            self.active_job = None
            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()
        return f"开始做 {name} 了（在后台进行，可以先说话；进度会写在[眼前]里）。"

    def stop(self) -> str:
        with self._job_lock:
            self._pending.clear()
            self._stop.set()
        if self.body is not None:
            self.body.cancel()
        self.rpc("nav.stop", {})
        self.rpc("baritone.stop", {})
        self.rpc("interact.stopBreaking", {})
        return "停下了，并清空待执行动作。"

    def status(self) -> str:
        if self.busy():
            return f"正在 {self.current}：{self.progress}"
        return f"现在没在做什么。上一件：{self.last_result or '没有'}"

    def training_status(self):
        """Bounded game-only trace; polling it does not issue RPCs or change the world."""
        with self._job_lock:
            active = self.active_job
            return {"version": 1, "active": ({"id": active["id"], "name": active["name"],
                                              "started_at": active["started_at"]} if active else None),
                    "completed_total": self._actions_finished,
                    "history": list(self._action_history), "history_capacity": self._action_history.maxlen}

    # --- small body actions ---------------------------------------------------------------

    def work_status(self):
        work = self._work
        with self._job_lock:
            active = dict(self.active_job) if self.active_job else None
            queue = [{k: v for k, v in j.items() if k not in ("fn", "urgent")} for j in self._pending]
        status = {"active": bool(active or work), "active_job": active, "queued": queue,
                  "next": queue[0]["name"] if queue else None, "kind": (active or {}).get("name"),
                  "current_started_at": (active or {}).get("started_at"), "last_progress_at": self._last_progress_at,
                  "progress": self._action_progress_count, "route_points": len(self._route),
                  "last_outcome": dict(self.last_outcome), "explore_ready": time.monotonic() >= self._cave_attempt_after}
        if not work:
            status.update({k: v for k, v in self.last_outcome.items() if k in ("needs_supply", "return_position", "reason")})
            return status
        status.update(kind=work["kind"], progress=work["progress"],
                      budget_remaining=max(0, round(work["deadline"] - time.monotonic(), 1)),
                      blocks_remaining=max(0, work["limit"] - work["broken"]),
                      return_position=work["origin"], reason=work["reason"], last_progress_at=work["last_progress_at"])
        return status

    def _work_progress(self, broken=0):
        self._last_progress_at = time.time()
        self._action_progress_count += 1
        if self._work is not None:
            self._work["broken"] += broken
            self._work["progress"] += 1
            self._work["last_progress_at"] = self._last_progress_at

    def _work_guard(self):
        work = self._work
        if work is None:
            return not self._stop.is_set()
        if work["reason"]:
            return False
        if self._stop.is_set():
            work["reason"] = "收到停止请求"
        elif time.monotonic() >= work["deadline"]:
            work["reason"] = "到达本轮时间预算，回看路线并重新定目标"
        elif work["broken"] >= work["limit"]:
            work["reason"] = "到达本轮采掘数量预算，返回补给或检查路线"
        elif time.monotonic() >= self._next_supply_check:
            self._next_supply_check = time.monotonic() + 2
            state = self.rpc("player.getState", {}) or {}
            if not work["recovery"] and isinstance(state.get("health"), (float, int)) and state["health"] < 10:
                work["reason"] = "生命偏低，先离开危险并恢复"
            elif not work["recovery"] and isinstance(state.get("food"), (float, int)) and state["food"] <= 8:
                work["reason"] = "饱食不足，先补给"
            inv = self._inventory()
            if isinstance(inv.get("hotbar"), list) and isinstance(inv.get("main"), list):
                items = inv["hotbar"] + inv["main"]
                if not work["recovery"] and len([i for i in items if i.get("count", 0) > 0]) >= 35:
                    work["reason"] = "背包将满，先安全存放再采掘"
                tools = [i for i in items if str(i.get("id", "")).endswith("_pickaxe") and i.get("maxDamage")]
                if tools and max(i["maxDamage"] - i.get("damage", 0) for i in tools) <= 3:
                    work["reason"] = "镐耐久不足，先换备用工具"
        if work["reason"]:
            self.last_reason = work["reason"]
            return False
        return True

    def _me(self) -> Optional[Dict[str, Any]]:
        self._hold_for_reflexes()
        return self.rpc("player.getState", {})

    def _hold_for_reflexes(self):
        """While her reflexes have the body (fighting, running, eating), the work waits."""
        while time.time() < self.pause_until and not self._stop.is_set():
            time.sleep(0.25)

    def free_hands(self):
        """A screen left open (a furnace, the recipe book) freezes walking and mining: close it first."""
        if self.using_screen:
            return
        st = self.rpc("container.state", {}) or {}
        if st.get("open"):
            self.rpc("container.close", {})

    def _walk_to(self, x: float, y: float, z: float, reach: float = 2.0, timeout: float = 25.0) -> bool:
        self._hold_for_reflexes()
        if self._stop.is_set():
            return False
        if self.body is not None:
            me = self.rpc("player.getState", {}) or {}
            if not all(k in me for k in ("x", "y", "z")):
                return False
            self.rpc("nav.stop", {})
            distance = math.dist((me["x"], me["y"], me["z"]), (x, y, z))
            plan = self.rpc("nav.plan", {"x": x, "y": y, "z": z, "reachRadius": min(16, max(0, reach))}) or {}
            path = plan.get("path") or []
            if path:
                ok = self.body.follow_path(path, reach=min(.6, reach), timeout=timeout)
            elif distance <= 3:
                # A close manipulation step can use the body's obstacle/ledge probes directly.
                ok = self.body.walk_to(x, y, z, reach=reach, timeout=timeout)
            elif self.baritone():
                ok = self.baritone_goto(x, y, z, near=max(0, int(reach)), timeout=max(20.0, timeout)) == "done"
            else:
                self.last_reason = "没有已加载可走路线，停止盲走"
                return False
            # The body can reach the horizontal centre before the next game tick
            # applies gravity. Keep the strict height check, but allow bounded settling.
            settle_until = time.monotonic() + min(1.0, max(0.0, timeout))
            arrived = False
            while True:
                actual = self.rpc("player.getState", {}) or {}
                valid = all(k in actual for k in ("x", "y", "z"))
                horizontal = valid and math.hypot(actual["x"] - x, actual["z"] - z) <= reach + .55
                arrived = bool(ok and not self._stop.is_set() and horizontal and abs(actual["y"] - y) <= .75)
                if (arrived or not ok or not horizontal or self._stop.is_set()
                        or time.monotonic() >= settle_until):
                    break
                time.sleep(.05)
            if arrived:
                self._work_progress()
            return arrived
        if self.rpc("nav.pathTo", {"x": x, "y": y, "z": z, "reachRadius": reach, "timeoutSeconds": int(timeout)}) is None:
            return False
        end = time.time() + timeout
        while time.time() < end and not self._stop.is_set():
            time.sleep(0.4)
            st = self.rpc("nav.status", {}) or {}
            if not st.get("active"):
                me = self.rpc("player.getState", {}) or {}
                return bool(me and math.hypot(me["x"] - x, me["z"] - z) <= reach + 0.5
                            and abs(me["y"] - y) <= 2.5)
        self.rpc("nav.stop", {})
        return False

    def _block(self, x: int, y: int, z: int) -> str:
        got = (self.rpc("perception.blocks", {"positions": [{"x": x, "y": y, "z": z}]}) or {}).get("blocks") or [{}]
        return str(got[0].get("id") or "")

    def _smooth_look(self, x: float, y: float, z: float, seconds: float = 0.25) -> None:
        """Turn the camera like a hand moves a mouse: a big turn takes longer than a small one, it speeds up and
        slows down (ease in and out), and it is drawn in small steps (~50 a second), never a snap."""
        if self._stop.is_set() and threading.current_thread() is self._thread:
            return
        if self.body is not None:
            self.body.look_at(x, y, z, timeout=max(2.5, seconds))
            return
        st = self.rpc("player.getState", {}) or {}
        yaw0 = float(st.get("yaw") or 0.0)
        pitch0 = float(st.get("pitch") or 0.0)
        ex = float(st.get("x") or 0.0)
        ey = float(st.get("y") or 0.0) + 1.62
        ez = float(st.get("z") or 0.0)
        dx, dy, dz = x - ex, y - ey, z - ez
        horiz = math.hypot(dx, dz) or 1e-6
        target_yaw = math.degrees(math.atan2(dz, dx)) - 90.0
        target_pitch = -math.degrees(math.atan2(dy, horiz))
        delta = (target_yaw - yaw0 + 180.0) % 360.0 - 180.0
        turn = max(abs(delta), abs(target_pitch - pitch0))
        if turn < 1.5:
            return
        duration = max(seconds, min(0.9, 0.12 + turn / 300.0))     # ~300 degrees a second at most, like a wrist
        steps = max(3, int(duration / 0.02))
        start = time.time()
        for i in range(1, steps + 1):
            t = i / steps
            ease = t * t * (3 - 2 * t)                                  # smoothstep: speed up, then settle
            self.rpc("control.look", {"yaw": yaw0 + delta * ease,
                                      "pitch": pitch0 + (target_pitch - pitch0) * ease})
            wait = start + duration * t - time.time()
            if wait > 0:
                time.sleep(wait)

    def _break(self, x: int, y: int, z: int, timeout: float = 8.0, depth: int = 0) -> bool:
        """Mine a block the way a player does: only what she can see, looking at it the whole time."""
        if not self._work_guard():
            return False
        before = self._block(x, y, z)
        if not before or before.endswith(":air"):
            return False
        if not self._tool_up(before):
            return False
        me = self._me()
        if me:
            eye = (me["x"], me["y"] + 1.62, me["z"])
            hit = first_obstacle(self.rpc, eye, (x, y, z))
            if hit:
                if depth == 0 and any(s in hit[3].split(":")[-1] for s in SOFT):
                    self.progress = f"先把挡着的 {hit[3].split(':')[-1]} 打掉"
                    self._break(hit[0], hit[1], hit[2], timeout=4.0, depth=1)
                    hit = first_obstacle(self.rpc, eye, (x, y, z))
                if hit:
                    self.last_blocked = hit
                    return False                # something solid in the way: no mining through it
        aim = face_point((me["x"], me["y"] + 1.62, me["z"]), (x, y, z), self.rpc) if me else (x + 0.5, y + 0.5, z + 0.5)
        if self.body is not None:
            if self._stop.is_set():
                return False
            ok = self.body.mine_block(x, y, z, timeout=timeout)
            if ok:
                self._work_progress(broken=1)
            return ok
        self._smooth_look(*aim, 0.3)
        if self.rpc("interact.breakBlock", {"x": x, "y": y, "z": z, "mode": "survival"}) is None:
            return False
        end = time.time() + timeout
        while time.time() < end and not self._stop.is_set():
            time.sleep(0.3)
            if self._block(x, y, z) != before:
                self._work_progress(broken=1)
                return True
            self._smooth_look(*aim, 0.1)                           # keep the eyes on it while the arm works
        self.rpc("interact.stopBreaking", {})
        return False

    def tool_for(self, block: str):
        """(tool kind, lowest tier that makes it drop) - from the pack's own tags, else from the block's name."""
        if self.index is not None:
            try:
                kind, tier = self.index.tool_for(block)
                if kind or tier:
                    return kind, tier
            except Exception:
                pass
        name = block.split(":")[-1]
        if any(w in name for w in ("leaves", "air", "grass", "fern", "flower", "torch")) and not name.endswith("_block"):
            return None, 0
        if name.endswith(("_log", "_wood", "_planks", "_stem", "_hyphae")) or name in ("crafting_table", "chest", "barrel", "bookshelf"):
            return "axe", 0
        if any(w in name for w in ("dirt", "sand", "gravel", "clay", "snow", "mud", "soul_soil", "farmland")) or name == "grass_block":
            return "shovel", 0
        if any(w in name for w in ("diamond_ore", "gold_ore", "redstone_ore", "emerald_ore")):
            return "pickaxe", 2
        if any(w in name for w in ("iron_ore", "copper_ore", "lapis_ore")):
            return "pickaxe", 1
        if name in ("obsidian", "crying_obsidian", "ancient_debris"):
            return "pickaxe", 3
        if any(w in name for w in ("stone", "ore", "cobble", "deepslate", "andesite", "diorite", "granite", "tuff", "brick",
                                   "furnace", "basalt", "netherrack", "calcite")):
            return "pickaxe", 0
        return None, 0

    def _tool_up(self, block: str) -> bool:
        """Take the right tool for this block; False when no tool she has is good enough to make it drop
        (a player does not hit iron ore with a wooden pickaxe)."""
        kind, tier = self.tool_for(block)
        if not kind:
            return True
        tool = self.best_tool(kind)
        from src.levels.l8_agency.mc_goal import tool_tier as _tt
        if kind == "pickaxe" and (tool is None or _tt(tool) < tier):
            # stone and ore dug without a (good enough) pickaxe drop nothing: make one, the way anyone would
            self._make_tool(kind, tier)
            tool = self.best_tool(kind)
            if tool is None:
                self.last_reason = "没有镐子，挖了也不掉东西，又做不出新的"
                return False
        if tool:
            self.hold(tool)
            self._uses += 1
            if self._uses % 8 == 0:
                self.mind_the_wear(kind)
        if tier > 0:
            from src.levels.l8_agency.mc_goal import tool_tier
            if tool is None or tool_tier(tool) < tier:
                self.last_reason = f"挖 {block.split(':')[-1]} 要{['木', '石', '铁', '钻石', '下界合金'][tier]}镐以上"
                return False
        return True

    PASSABLE = ("air", "grass", "fern", "flower", "leaf_litter", "snow", "bush", "sapling", "vine", "petals", "moss_carpet")

    def _passable(self, block_id: str) -> bool:
        name = block_id.split(":")[-1]
        if name.endswith(("_block", "_path", "_planks", "_log", "_wood", "_slab", "_stairs")):
            return False                                   # grass_block is ground, not grass
        return not block_id or any(p in name for p in self.PASSABLE)

    def _approach(self, x: int, y: int, z: int) -> bool:
        """Get within reach of a block: stay if already there, else stand on a free spot beside it.
        (Walking "to" a solid block fails: the pathfinder needs somewhere to stand.)"""
        me = self._me()
        if me and self._dist(me, x, y, z) <= REACH - 0.3:
            return True
        spots = [(x + dx, z + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))]
        if me:
            spots.sort(key=lambda p: (p[0] - me["x"]) ** 2 + (p[1] - me["z"]) ** 2)
        for nx, nz in spots:
            got = (self.rpc("perception.blocks", {"positions": [{"x": nx, "y": y + d, "z": nz} for d in (-1, 0, 1)]})
                   or {}).get("blocks") or []
            if len(got) < 3:
                continue
            ground, feet, head = (str(b.get("id") or "") for b in got)
            if not self._passable(ground) and self._passable(feet) and self._passable(head):
                if self._walk_to(nx + .5, y, nz + .5, reach=.5):
                    return True
            if self._stop.is_set():
                return False
        return False

    @staticmethod
    def _dist(me: Dict[str, Any], x: float, y: float, z: float) -> float:
        return math.sqrt((me["x"] - x - 0.5) ** 2 + (me["y"] + 1.6 - y - 0.5) ** 2 + (me["z"] - z - 0.5) ** 2)

    # --- skills ---------------------------------------------------------------------------

    def chop_tree(self, max_logs: int = 10, radius: int = 2, prefer: str = "") -> str:
        """Find the nearest tree trunk, walk to it, break the logs from the bottom up, then pick up what fell.
        `radius` is in chunks (16 blocks), as the client scan counts it."""
        scan = self.rpc("perception.scan", {"radius": radius}) or {}
        kinds = sorted({k for c in scan.get("chunks") or [] for k in (c.get("counts") or {})
                        if k.endswith(LOG_SUFFIX) and "stripped" not in k})
        if not kinds:
            return "附近没有树。"
        if prefer and prefer in kinds:
            kinds = [prefer]                    # the wood the plan asks for, when there is such a tree around
        found = (self.rpc("perception.scan", {"radius": radius, "find": kinds, "findLimit": 64}) or {}).get("found") or []
        me = self._me()
        if not found or not me:
            return "附近没有够得着的树。"
        # the lowest log of the nearest trunk: group by column, take the column closest to her
        columns: Dict[tuple, List[int]] = {}
        for f in found:
            columns.setdefault((f["x"], f["z"]), []).append(f["y"])
        now = time.time()
        columns = {k: v for k, v in columns.items() if now - self.failed.get(k, 0) > 300}
        if not columns:
            return "附近的树我都试过了，走不过去。"
        # A trunk, not a branch up in the canopy: its lowest log stands on the ground, at about her height
        def score(kv):
            (cx, cz), cys = kv
            return (cx - me["x"]) ** 2 + (cz - me["z"]) ** 2 + 4 * (min(cys) - me["y"]) ** 2
        trunk = None
        for (cx, cz), cys in sorted(columns.items(), key=score)[:8]:
            cbase = min(cys)
            if abs(cbase - me["y"]) > 8:
                continue
            below = self._block(cx, cbase - 1, cz)
            name = below.split(":")[-1]
            if below and not below.endswith(":air") and LOG_SUFFIX not in name and "leaves" not in name:
                trunk = (cx, cz, cbase)
                break
            # (a branch, or a log in the air: just not a trunk - not a failure to remember)
        if trunk is None:
            return "附近没有站在地上、够得着的树干。"
        x, z, base = trunk
        logs_before = self._log_count()
        self.progress = f"走去那棵树（{x}, {base}, {z}）"
        if not self._approach(x, base, z):
            self.failed[(x, z)] = now
            return "走不到那棵树，换一棵试试。"
        logs = 0
        for y in range(base, base + TRUNK_HEIGHT):
            if self._stop.is_set() or logs >= max_logs:
                break
            block = self._block(x, y, z)
            if not block.endswith(LOG_SUFFIX):
                if y > base + 1:
                    break                       # past the top of the trunk
                continue
            me = self._me() or me
            if self._dist(me, x, y, z) > REACH:
                break                           # too high to reach from the ground
            self.progress = f"砍第 {logs + 1} 块（{block.split(':')[-1]}）"
            if self._break(x, y, z):
                logs += 1
            elif self.last_blocked:
                self.progress = f"第 {logs + 1} 块被 {self.last_blocked[3].split(':')[-1]} 挡住了"
                break
        if self._stop.is_set():
            return f"砍了 {logs} 块就停了。"
        picked = self.collect_drops(radius=10)
        got = self._log_count() - logs_before
        return f"砍了 {max(got, logs)} 块木头。{picked}"

    def _log_count(self) -> int:
        from src.levels.l8_agency.mc_goal import is_log
        return sum(n for i, n in self.crafter.counts().items() if is_log(i))

    @bounded_excursion(90, 48)
    def mine(self, blocks: str = "", count: int = 8, radius: int = 2) -> str:
        """Dig some blocks of the given kinds nearby - the ones she can actually see, walking up to them first."""
        ids = [b.strip() for b in str(blocks or "").split(",") if b.strip()]
        if not ids:
            return "要挖什么？blocks 写方块 id。"
        found = (self.rpc("perception.scan", {"radius": radius, "find": ids, "findLimit": 64}) or {}).get("found") or []
        me = self._me()
        if not found or not me:
            return "附近没找到这些方块。"
        # exposed ones first (a face open to the air: what a player can see without digging a tunnel), then near ones
        found.sort(key=lambda f: (not f.get("exposed"), (f["x"] - me["x"]) ** 2 + 2 * (f["y"] - me["y"]) ** 2 + (f["z"] - me["z"]) ** 2))
        got = tried = 0
        for f in found:
            if not self._work_guard() or got >= min(32, count) or tried >= min(64, count * 3):
                break
            tried += 1
            if not f.get("exposed") or abs(f["y"] - me["y"]) > 6 or self._dist(me, f["x"], f["y"], f["z"]) > 16:
                continue
            self.progress = f"去挖 {f['id'].split(':')[-1]}（已挖 {got}）"
            reachable = self._ore_approach(f, within=8) if self._dist(me, f["x"], f["y"], f["z"]) <= 8 and "ore" in f["id"] else self._approach(f["x"], f["y"], f["z"])
            if not reachable:
                continue
            if self._break(f["x"], f["y"], f["z"]):
                got += 1
                if "ore" in f["id"]:
                    self.places.add("ore", f["x"], f["y"], f["z"], note=f["id"], merge=8)
            me = self._me() or me
        picked = self.collect_drops(radius=6) if got and self._work_guard() else ""
        return f"挖了 {got} 块。{picked}" if got else "看得见又够得着的一块都没有（要么被挡住，要么太远）。"

    def collect_drops(self, radius: float = 10) -> str:
        """Walk over the items lying around nearby so they end up in the backpack."""
        before = observe(self.rpc)
        got = 0
        for _ in range(8):
            if self._stop.is_set():
                break
            items = (self.rpc("perception.entities", {"radius": radius, "kinds": ["item"]}) or {}).get("entities") or []
            t, where = getattr(self, "dropped_at", (0, None))
            if where and time.time() - t < 120:              # what she just threw away stays thrown away
                items = [e for e in items if (e["x"] - where["x"]) ** 2 + (e["z"] - where["z"]) ** 2 > 36]
            junk = getattr(self, "thrown_ids", set())
            items = [e for e in items if not (isinstance(e.get("item"), dict) and (
                e["item"].get("id") in junk or any(j in str(e["item"].get("id")) for j in self.JUNK)))]
            if not items:
                break
            item = min(items, key=lambda e: e.get("distance", 99))
            self.progress = f"去捡地上的东西（还有 {len(items)} 个）"
            if self._walk_to(item["x"], item["y"], item["z"], reach=0.6, timeout=10):
                got += 1
            time.sleep(0.4)
        after = observe(self.rpc)
        if (before["inventory_valid"] and after["inventory_valid"] and before["state_valid"]
                and after["state_valid"] and before.get("dimension") == after.get("dimension")):
            collected = sum(max(0, n - before["inventory"].get(item, 0)) for item, n in after["inventory"].items())
            if collected:
                return f"实际捡到物品 ×{collected}。"
        return f"走到掉落物附近 {got} 次，但未确认物品入包。" if got else "地上没什么可捡的。"

    # --- hands and bag ---------------------------------------------------------------------

    TOOL_KINDS = {"pickaxe": "_pickaxe", "axe": "_axe", "shovel": "_shovel", "sword": "_sword", "hoe": "_hoe"}
    MATERIAL_RANK = ("wooden", "golden", "stone", "copper", "iron", "diamond", "netherite")

    def _inventory(self) -> Dict[str, Any]:
        return self.rpc("player.getInventory", {}) or {}

    def hold(self, item: str) -> bool:
        """Take an item in hand: pick its hotbar slot, or swap it there from the bag."""
        from src.levels.l8_agency.mc_craft import select_item
        return select_item(self.rpc, item, stopped=self._stop.is_set)

    def best_tool(self, kind: str) -> Optional[str]:
        suffix = self.TOOL_KINDS.get(kind, f"_{kind}")
        inv = self._inventory()
        have = [it["id"] for it in (inv.get("hotbar") or []) + (inv.get("main") or []) if str(it.get("id", "")).endswith(suffix)]
        if not have:
            return None
        rank = lambda i: max((n for n, m in enumerate(self.MATERIAL_RANK) if f":{m}_" in i), default=2)
        return max(have, key=rank)

    def equip(self, kind: str = "pickaxe") -> str:
        tool = self.best_tool(kind)
        if not tool:
            return f"包里没有{kind}。"
        return f"拿起了 {tool.split(':')[-1]}。" if self.hold(tool) else f"没拿起 {tool}。"

    def tidy(self) -> str:
        """The best pickaxe, axe, sword and shovel in hotbar slots 0-3, food in 8 - so her hands find them."""
        moved = []
        for slot, kind in enumerate(("pickaxe", "axe", "sword", "shovel")):
            tool = self.best_tool(kind)
            if not tool:
                continue
            inv = self._inventory()
            if any(it.get("slot") == slot and it.get("id") == tool for it in inv.get("hotbar") or []):
                continue
            src = next((it for it in (inv.get("hotbar") or []) + (inv.get("main") or []) if it.get("id") == tool), None)
            if src:
                from src.levels.l8_agency.mc_craft import move_inventory_stack
                if move_inventory_stack(self.rpc, int(src["slot"]), slot):
                    moved.append(tool.split(":")[-1])
        return ("整理好了：" + "、".join(moved) + " 放到了快捷栏前几格。") if moved else "快捷栏已经挺顺手了。"

    def place(self, item: str = "", x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> str:
        if x is None or y is None or z is None:
            pos = self.place_near(item)
            return f"放好了 {item.split(':')[-1]}（{pos[0]},{pos[1]},{pos[2]}）。" if pos else f"没地方放 {item.split(':')[-1]}。"
        pos = self.place_at(item, int(x), int(y), int(z))
        return f"放好了（{x},{y},{z}）。" if pos else "没放上去（够不着、没有依附面，或者那里有东西）。"

    def place_near(self, item: str) -> Optional[tuple]:
        """Put a block from the bag right next to her, the way a player drops a table down: on the ground if
        there is room, else against a wall (in a one-wide tunnel there is nowhere else)."""
        me = self._me()
        if not me or not item:
            return None
        fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
        ring = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1), (2, 0), (0, 2), (-2, 0), (0, -2))
        for dy in (0, 1):
            for dx, dz in ring:
                pos = self.place_at(item, fx + dx, fy + dy, fz + dz, walk=False)
                if pos:
                    return pos
        return None

    # which face of a neighbouring block a hand clicks to put something into the cell next to it
    FACES = (((0, -1, 0), "up"), ((1, 0, 0), "west"), ((-1, 0, 0), "east"), ((0, 0, 1), "north"), ((0, 0, -1), "south"),
             ((0, 1, 0), "down"))

    def place_at(self, item: str, x: int, y: int, z: int, walk: bool = True) -> Optional[tuple]:
        if self._stop.is_set():
            return None
        cells = [(x, y, z)] + [(x + o[0], y + o[1], z + o[2]) for o, _ in self.FACES]
        got = self._blocks(cells)
        if not self._passable(got[0]) or "water" in got[0] or "lava" in got[0]:
            return None
        # something solid to put it against: the ground first, then a wall, then a ceiling
        support = next(((o, face) for (o, face), b in zip(self.FACES, got[1:])
                        if b and not self._passable(b) and "water" not in b and "lava" not in b), None)
        if support is None:
            return None
        (ox, oy, oz), face = support
        if walk and not self._approach(x + ox, y + oy, z + oz):
            return None
        if not self.hold(item):
            return None
        if self.body is not None:
            if not self.body.place_block(x, y, z, item=item):
                return None
        else:
            self._smooth_look(x + 0.5 + ox * 0.49, y + 0.5 + oy * 0.49, z + 0.5 + oz * 0.49, 0.25)
            self.rpc("interact.placeBlock", {"x": x + ox, "y": y + oy, "z": z + oz, "face": face})
        deadline = time.time() + 2
        want = item.split(":")[-1]
        while time.time() < deadline:
            time.sleep(0.2)
            # mods swap some blocks once placed ("visualworkbench:minecraft/crafting_table"); a torch on a wall
            # becomes a wall_torch
            if self._block(x, y, z).split(":")[-1].split("/")[-1].endswith(want):
                self._action_facts.append({"kind": "placed", "item": item, "x": x, "y": y, "z": z})
                if want in ("chest", "furnace", "crafting_table", "barrel") or want.endswith("_bed"):
                    self.places.add("bed" if want.endswith("_bed") else "chest" if want == "barrel" else want, x, y, z)
                self._work_progress()
                return (x, y, z)
        return None

    # --- Baritone: the legs of a seasoned player --------------------------------------------
    # Walking far, digging down to a layer, getting out to the surface and finding ore are what Baritone (a
    # pathfinder mod, optional) does well: it digs through, bridges, avoids lava and climbs out. When it is
    # installed these legs go to it and the hand-written stairs below stay as the fallback.

    BARITONE_STALL = 25.0    # no movement for this long while it says it is pathing: give up

    def baritone(self) -> bool:
        now = time.monotonic()
        if now < getattr(self, "_baritone_checked", 0):
            return getattr(self, "_baritone_ok", False)
        st = self.rpc("baritone.status", {}) or {}
        self._baritone_ok = bool(st.get("available"))
        self._baritone_checked = now + (60 if self._baritone_ok else 300)
        return self._baritone_ok

    def _baritone_run(self, method: str, params: Dict[str, Any], timeout: float, done=None) -> str:
        """Start one Baritone job and stay with it. Returns "done", "idle" (it stopped on its own), "stalled",
        "stopped", "timeout" or "failed". `done()` may end it early (enough ore in the bag)."""
        if self.rpc(method, params) is None:
            return "failed"
        end = time.monotonic() + timeout
        last_pos, moved_at = None, time.monotonic()
        time.sleep(0.5)
        try:
            while not self._stop.is_set():
                if time.time() < self.pause_until:                 # a fight or a meal has the body
                    self.rpc("baritone.stop", {})
                    self._hold_for_reflexes()
                    if self._stop.is_set():
                        return "stopped"
                    if self.rpc(method, params) is None:          # pick the same job up again afterwards
                        return "failed"
                    moved_at = time.monotonic()
                if done is not None and done():
                    return "done"
                st = self.rpc("baritone.status", {}) or {}
                if not st.get("busy"):
                    return "idle"
                me = self._me() or {}
                pos = tuple(round(me.get(k, 0)) for k in ("x", "y", "z"))
                if pos != last_pos:
                    last_pos, moved_at = pos, time.monotonic()
                    self._work_progress()
                elif time.monotonic() - moved_at > self.BARITONE_STALL:
                    return "stalled"
                if time.monotonic() >= end:
                    return "timeout"
                time.sleep(0.5)
            return "stopped"
        finally:
            self.rpc("baritone.stop", {})

    def baritone_goto(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None,
                      near: int = 0, timeout: float = 120.0) -> str:
        """Baritone plans the way (digging, bridging, no lava); her own body walks it (mc_route).
        Returns done / stopped / timeout / no_route / stuck."""
        params = {k: v for k, v in (("x", x), ("y", y), ("z", z)) if v is not None}
        if near:
            params["near"] = int(near)
        if getattr(self, "_routing", False):
            return "no_route"                            # a step of a route never plans another route
        reach = max(1.0, float(near) + 0.8)

        def arrived(me):
            if x is not None and z is not None and math.hypot(me["x"] - x - .5, me["z"] - z - .5) > reach:
                return False
            if y is not None and (x is None or z is None):
                return abs(math.floor(me["y"]) - int(y)) <= 1
            return y is None or abs(me["y"] - y) <= 1.5
        self.progress = "Baritone 规划路线：" + ",".join(f"{k}={int(v)}" for k, v in params.items() if k != "near")
        from src.levels.l8_agency.mc_route import RouteFollower
        self._routing = True
        try:
            return RouteFollower(self).go(params, arrived, timeout=timeout)
        finally:
            self._routing = False

    def surface_y(self) -> Optional[int]:
        """The height of open ground above her (top of the solid column + 1), from loaded blocks."""
        me = self._me()
        if not me:
            return None
        x, y, z = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
        col = self._blocks([(x, y + k, z) for k in range(2, 96)])
        top = None
        for k, b in enumerate(col, start=2):
            if b and not self._passable(b) and "leaves" not in b:
                top = y + k
        return None if top is None else top + 1

    # --- stations she already has --------------------------------------------------------

    STATION_WALK = 48        # a remembered table/furnace this close is worth walking back to (blocks)

    @staticmethod
    def _is_station(block_id: str, kind: str) -> bool:
        # mods rename placed blocks ("visualworkbench:minecraft/crafting_table"); "blast_furnace" is not a furnace
        return str(block_id or "").split(":")[-1].split("/")[-1] == kind

    def find_station(self, kind: str) -> Optional[Dict[str, Any]]:
        """A crafting table or furnace she can use without making a new one: one in sight first, else one on her
        map within STATION_WALK blocks that is still there when she checks. A player walks back to the table they
        put down a minute ago instead of crafting a 38th."""
        me = self._me() or {}
        if not me:
            return None
        pois = (self.rpc("perception.scan", {"radius": 1}) or {}).get("pois") or []
        seen = sorted((p for p in pois if self._is_station(p.get("id"), kind) and abs(p["y"] - me["y"]) <= 4
                       and math.hypot(p["x"] - me["x"], p["z"] - me["z"]) <= 16),
                      key=lambda p: math.dist((p["x"], p["y"], p["z"]), (me["x"], me["y"], me["z"])))
        if seen:
            return {"x": seen[0]["x"], "y": seen[0]["y"], "z": seen[0]["z"], "how": "seen"}
        for p in self.places.near(me, (kind,), within=self.STATION_WALK, n=4):
            if abs(p["y"] - me["y"]) > 12:
                continue                               # one on another floor of the cave: not a short walk
            self.progress = f"回到记得的{ {'crafting_table': '工作台', 'furnace': '熔炉'}.get(kind, kind) }"
            if not self._walk_to(p["x"] + 0.5, p["y"], p["z"] + 0.5, reach=3.0, timeout=20):
                continue
            if self._is_station(self._block(p["x"], p["y"], p["z"]), kind):
                return {"x": p["x"], "y": p["y"], "z": p["z"], "how": "remembered"}
            self.places.forget_at(p["x"], p["y"], p["z"], kind)      # broken or never hers: stop walking to it
        return None

    def put_away_table(self, pos) -> str:
        """A table put down out in the field is picked back up, as a player does; at home it stays."""
        if not pos:
            return ""
        x, y, z = (int(v) for v in pos)
        home = self.places.home
        if home and math.dist((x, y, z), (home["x"], home["y"], home["z"])) <= 12:
            return ""
        if not self._is_station(self._block(x, y, z), "crafting_table"):
            return ""
        if self._break(x, y, z, timeout=4.0):
            self.places.forget_at(x, y, z, "crafting_table")
            self.collect_drops(radius=4)
            return "把临时工作台收回了包里。"
        return ""

    def craft(self, item: str = "", count: int = 1) -> str:
        if not item:
            return "要做什么？item 写物品 id。"
        self.progress = f"合成 {item.split(':')[-1]}"
        return self.crafter.craft(item, int(count or 1))

    def go_to_player(self, player: str = "") -> str:
        if not self.find_player:
            return "不知道怎么找人。"
        them = self.find_player(player)
        if them is None:
            return f"看不到 {player or '附近的人'}。"
        player = them.get("name") or player or "那个人"
        self.progress = f"走去 {player} 身边"
        ok = self._walk_to(them["x"], them["y"], them["z"], reach=2.5)
        if ok:
            self._smooth_look(them["x"], them["y"] + 1.6, them["z"], 0.3)
        return f"到 {player} 身边了。" if ok else f"没走到 {player} 那里。"

    # --- furnace ----------------------------------------------------------------------------

    FUEL_ORDER = ("minecraft:coal", "minecraft:charcoal", "minecraft:coal_block")

    @staticmethod
    def furnace_slot(inv_slot: int) -> int:
        """Inventory slot -> slot in a furnace menu (input 0, fuel 1, result 2, then her bag and hotbar)."""
        return inv_slot - 6 if inv_slot >= 9 else 30 + inv_slot

    def _open_furnace(self) -> bool:
        st = self.rpc("container.state", {}) or {}
        if st.get("open") and "furnace" in str(st.get("type", "")):
            return True
        if st.get("open"):
            self.rpc("container.close", {})
        f = self.find_station("furnace")
        if f is None:
            if self.crafter.counts().get("minecraft:furnace", 0) == 0:
                self.crafter.craft("minecraft:furnace", 1)
            pos = self.place_near("minecraft:furnace")
            if not pos:
                return False
            f = {"x": pos[0], "y": pos[1], "z": pos[2]}
        if not self._approach(f["x"], f["y"], f["z"]):
            return False
        self._smooth_look(f["x"] + 0.5, f["y"] + 0.5, f["z"] + 0.5, 0.3)
        self.rpc("container.open", {"x": f["x"], "y": f["y"], "z": f["z"]})
        end = time.time() + 3
        while time.time() < end:
            st = self.rpc("container.state", {}) or {}
            if st.get("open") and "furnace" in str(st.get("type", "")):
                return True
            time.sleep(0.2)
        return False

    def _furnace_slots(self) -> Dict[int, Dict[str, Any]]:
        return {int(s["slot"]): s for s in (self.rpc("container.state", {}) or {}).get("slots") or [] if int(s["slot"]) <= 2}

    def _put(self, item: str, target: int, count: int) -> int:
        """Put `count` of an item from the bag into a furnace slot, one right-click each; returns how many went in."""
        put = 0
        for src in [it for it in self.crafter.inventory() if it["id"] == item]:
            if put >= count:
                break
            slot = self.furnace_slot(int(src["slot"]))
            self.crafter._click(slot, 0)
            for _ in range(min(int(src.get("count", 0)), count - put)):
                self.crafter._click(target, 1)
                put += 1
            self.crafter._click(slot, 0)
        return put

    def smelt(self, item: str = "", count: int = 1) -> str:
        """Cook things in a furnace the way a player does: open one (the nearest, or put one down), the raw thing
        on top, fuel below, wait, take what comes out. `item` is what goes in (raw_iron) or what should come out
        (iron_ingot)."""
        have = self.crafter.counts()
        if self.index is not None and have.get(item, 0) == 0:
            src = next((s for s in self.index.smelted_from(item) if have.get(s, 0) > 0), None)
            item = src or item
        count = min(int(count or 1), have.get(item, 0), 64)
        if count <= 0:
            return f"包里没有能烧的 {item.split(':')[-1]}。"
        fuels = [f for f in self.FUEL_ORDER if have.get(f, 0) > 0] or \
                [i for i in have if (i.endswith("_planks") or i.endswith("_log")) and i != item]
        if not fuels:
            return "没有燃料（煤、木炭、木板或原木都行）。"
        self.progress = f"去熔炉烧 {item.split(':')[-1]}"
        if self.free_slots() < 2:
            self.declutter()
        if not self._open_furnace():
            return "没能用上熔炉（附近没有，也放不下）。"
        self.using_screen = True
        try:
            return self._smelt_open(item, count, fuels)
        finally:
            self.using_screen = False
            self.rpc("container.close", {})

    def _smelt_open(self, item: str, count: int, fuels: List[str]) -> str:
        if self._furnace_slots().get(2):
            self.crafter._click(2, 0, "quick_move")
        put = self._put(item, 0, count)
        per = 8 if fuels[0] in self.FUEL_ORDER[:2] else 80 if fuels[0].endswith("coal_block") else 1.5
        need = math.ceil(put / per)
        current = self._furnace_slots().get(1)
        if current and current.get("id") == fuels[0]:
            need = max(0, need - int(current.get("count", 0)))
        self._put(fuels[0], 1, need)
        # what came out is counted in the bag, not in the furnace: a full bag leaves the result where it is
        start_bag = self.crafter.counts()
        got, out_id, end = 0, None, time.time() + put * 10 + 12
        while time.time() < end and not self._stop.is_set():
            time.sleep(2)
            slots = self._furnace_slots()
            out = slots.get(2)
            if out and int(out.get("count", 0)) > 0:
                out_id = out.get("id")
                self.crafter._click(2, 0, "quick_move")
                time.sleep(0.3)
            if out_id:
                got = self.crafter.counts().get(out_id, 0) - start_bag.get(out_id, 0)
            self.progress = f"熔炉里烧着 {item.split(':')[-1]}（拿到 {got}/{put}）"
            if got >= put or (not slots.get(0) and not out):
                break
        if got:
            return f"烧好了 {got} 个。"
        return "烧出来了但拿不出来（背包满了）。" if out_id else "熔炉没烧出东西（燃料不够或者放错了）。"

    # --- going underground like a player: stairs, then a tunnel -----------------------------

    HAZARD = ("lava", "water")

    def _facing(self) -> tuple:
        me = self._me() or {}
        yaw = math.radians(float(me.get("yaw") or 0))
        dx, dz = -math.sin(yaw), math.cos(yaw)
        return (int(round(dx)), 0) if abs(dx) > abs(dz) else (0, int(round(dz)))

    def _blocks(self, cells: List[tuple]) -> List[str]:
        got = (self.rpc("perception.blocks", {"positions": [{"x": c[0], "y": c[1], "z": c[2]} for c in cells]}) or {}).get("blocks") or []
        return [str(b.get("id") or "") for b in got] + [""] * (len(cells) - len(got))

    def _danger_near(self, cells: List[tuple]) -> Optional[str]:
        around = set()
        for (x, y, z) in cells:
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                around.add((x + d[0], y + d[1], z + d[2]))
        for b in self._blocks(sorted(around)):
            if any(h in b for h in self.HAZARD):
                return b
        return None

    def _dig_step(self, cells: List[tuple], stand: tuple) -> Optional[str]:
        """Clear the cells (top first) and step into `stand`; None when done, else why not."""
        if not self._work_guard():
            return self.last_reason or "作业预算结束"
        danger = self._danger_near(cells)
        if danger:
            return f"那边有{'岩浆' if 'lava' in danger else '水'}"
        observed = self._blocks(cells)
        if not all(observed):
            return "前方方块尚未加载，先停下"
        for c, b in zip(cells, observed):
            if self._stop.is_set():
                return "停了"
            self.last_reason = ""
            if not self._passable(b) and not self._break(*c):
                body_note = getattr(self.body, "_note", "") if self.body is not None else ""
                return self.last_reason or (f"{b.split(':')[-1]} 挖不掉（{body_note}）" if body_note
                                            else f"{b.split(':')[-1]} 挖不掉")
        # Corners with a 0.6 radius leave the player's hitbox on the previous stair. Move over
        # the centre, then wait for gravity/stepping to change actual feet height.
        if not self._walk_to(stand[0] + .5, stand[1], stand[2] + .5, reach=.15, timeout=6):
            return "走不过去"
        for _ in range(12):
            me = self._me() or {}
            if me and abs(me["y"] - stand[1]) <= .45:
                return None
            if self._stop.is_set():
                break
            time.sleep(.1)
        return "移动后实际高度仍未达到台阶，停止原地挖掘"

    @bounded_excursion(100, 72)
    def dig_down(self, to_y: int = 16, max_steps: int = 90) -> str:
        """A staircase down, one block forward and one down per step - never straight down (a cave or lava can be
        right under her feet). Turns away from lava, water and deep drops."""
        me = self._me()
        if not me:
            return "不知道自己在哪。"
        to_y = int(to_y)
        if math.floor(me["y"]) <= to_y:
            return f"往下挖不了：我在 y={math.floor(me['y'])}，比 y={to_y} 已经低了；要上去得用 dig_up。"
        if self.baritone():
            start_y = math.floor(me["y"])
            how = self.baritone_goto(y=to_y, timeout=150)
            after = math.floor((self._me() or me)["y"])
            if after <= to_y + 1:
                return f"往下挖到 y={after}（{start_y - after} 级台阶，Baritone）。"
            if start_y - after > 0 and how != "stopped":
                return f"往下挖到 y={after} 停了：Baritone {how}，下降了 {start_y - after} 格（{start_y - after} 级台阶）。"
            me = self._me() or me
        steps, turns, stalled = 0, 0, 0
        lit, unlit = 0, 0
        dx, dz = self._facing()
        while steps < max_steps and not self._stop.is_set():
            me = self._me() or me
            fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
            if fy <= to_y or not self._work_guard():
                break
            nx, nz = fx + dx, fz + dz
            floor = self._blocks([(nx, fy - 2, nz), (nx, fy - 3, nz), (nx, fy - 4, nz)])
            if all(self._passable(b) for b in floor):       # a drop of more than a few blocks: not this way
                why = "前面是个深坑"
            else:
                self.progress = f"挖楼梯往下（y={fy} → {to_y}）"
                why = self._dig_step([(nx, fy + 1, nz), (nx, fy, nz), (nx, fy - 1, nz)], (nx, fy - 1, nz))
            if why is None:
                after = self._me() or {}
                if after and after["y"] <= me["y"] - .5:
                    steps += 1
                    stalled = 0
                    self._work_progress()
                    if steps % 4 == 0:
                        self._dig_visible(self.VISIBLE_ORES, 4.5, limit=2)
                    if steps % 6 == 0:
                        # The previous feet cell is behind us, on the higher stair we just left.
                        # Count light only after place_at observes the actual torch block.
                        if self._torch_behind(fx, fy, fz):
                            lit += 1
                        else:
                            unlit += 1
                    continue
                why = "没有观测到实际高度下降"
            if "高度" in why or "走不过去" in why:
                stalled += 1
                if stalled >= 2:
                    return f"往下挖到 y={fy} 停了：连续两次未实际下降，仅确认 {steps} 级台阶。"
            turns += 1
            if turns > 3:
                return f"往下挖到 y={fy} 停了：{why}。"
            dx, dz = -dz, dx                                    # turn right and try again
        me = self._me() or me
        lighting = f"已确认放置火把 {lit} 支。" if lit else ""
        if unlit:
            lighting += f"有 {unlit} 处照明未确认，需补充火把或检查落点。"
        return f"往下挖到 y={math.floor(me['y'])}（{steps} 级台阶）。" + lighting

    @bounded_excursion(100, 72, recovery=True)
    def dig_up(self, to_y: int = 16, max_steps: int = 120, until_sky: bool = False) -> str:
        """A staircase up: one forward and one up per step, keeping the step under her feet. Stops for lava or
        water above, and for sand or gravel that would fall on her head."""
        me = self._me()
        if not me:
            return "不知道自己在哪。"
        to_y = int(to_y)
        if self.baritone():
            start_y = math.floor(me["y"])
            target = to_y
            if until_sky:
                surface = self.surface_y()
                target = min(to_y, surface) if surface is not None else to_y
            how = self.baritone_goto(y=target, timeout=150)
            me = self._me() or me
            fy = math.floor(me["y"])
            if fy >= target - 1 or (until_sky and self._sky_above(math.floor(me["x"]), fy, math.floor(me["z"]))):
                return f"往上挖到 y={fy}（{max(0, fy - start_y)} 级台阶，Baritone）。"
            if how == "stopped":
                return f"往上挖到 y={fy} 停了：收到停止请求。"
        steps, turns, pillars = 0, 0, 0
        dx, dz = self._facing()
        while steps < max_steps and not self._stop.is_set():
            me = self._me() or me
            fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
            if not self._work_guard() or fy >= to_y or (until_sky and steps % 4 == 0 and self._sky_above(fx, fy, fz)):
                break
            nx, nz = fx + dx, fz + dz
            cells = [(fx, fy + 2, fz), (nx, fy + 2, nz), (nx, fy + 1, nz)]
            falls = lambda b: b.split(":")[-1] in ("sand", "gravel", "red_sand", "suspicious_sand", "suspicious_gravel")
            over_me, over_next = self._blocks([(fx, fy + 3, fz), (nx, fy + 3, nz)])
            if falls(over_me):
                # it would come down on her head: step one block aside on this level and climb from there
                self.progress = "头上有砂砾，往旁边让一格"
                why = self._dig_step([(nx, fy + 1, nz), (nx, fy, nz)], (nx, fy, nz))
                if why is None:
                    continue
            elif falls(over_next):
                why = "前面头上是沙子/砂砾"
            elif self._passable(self._blocks([(nx, fy, nz)])[0]) and not self._step_block(nx, fy, nz):
                why = "前面没有落脚的台阶（也没有方块能垫）"
            else:
                self.progress = f"挖楼梯往上（y={fy} → {to_y}）"
                why = self._dig_step(cells, (nx, fy + 1, nz))
            if why is None:
                steps += 1
                self._work_progress()
                if steps % 5 == 0:
                    self._torch_behind(fx, fy, fz)
                continue
            turns += 1
            if turns > 3:
                # every side is open air (a cave): no rock to cut stairs into. A player towers up instead,
                # until there is a ceiling or a wall close enough to dig stairs again.
                if "落脚" in why and pillars < 3:
                    pillars += 1
                    self.progress = "四周没有台阶可挖：垫方块往上爬"
                    out = self.pillar_up(4)
                    if "爬高了 0 格" not in out and "没有方块" not in out:
                        turns = 0
                        steps += 1
                        me = self._me() or me
                        if until_sky and self._sky_above(math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])):
                            break
                        continue
                return f"往上挖到 y={fy} 停了：{why}。"
            dx, dz = -dz, dx
        me = self._me() or me
        return f"往上挖到 y={math.floor(me['y'])}（{steps} 级台阶）。"

    STEP_BLOCKS = ("minecraft:cobblestone", "minecraft:cobbled_deepslate", "minecraft:dirt", "minecraft:stone",
                   "minecraft:granite", "minecraft:andesite", "minecraft:diorite", "minecraft:tuff", "minecraft:netherrack")

    def _step_block(self, x: int, y: int, z: int) -> bool:
        """Put a cheap block into an empty cell to make a stair step (only on something solid below it)."""
        have = self.crafter.counts()
        item = next((i for i in self.STEP_BLOCKS if have.get(i, 0) > 0), None)
        if item is None or self._passable(self._blocks([(x, y - 1, z)])[0]) and self._passable(self._blocks([(x, y + 1, z)])[0]):
            return False
        self.progress = "前面缺一级台阶：垫一块方块"
        return bool(self.place_at(item, x, y, z, walk=False))

    PILLAR = ("cobblestone", "cobbled_deepslate", "dirt", "stone", "granite", "andesite", "diorite", "tuff",
              "netherrack", "gravel")

    def perched(self) -> int:
        """How many blocks up she stands on a one-wide column with only air around her feet (0 = not perched)."""
        me = self._me()
        if not me or not all(isinstance(me.get(k), (int, float)) for k in ("x", "y", "z")):
            return 0
        x, y, z = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
        ring = self._blocks([(x + dx, y - 1, z + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dz])
        if not ring or sum(1 for b in ring if b and not self._passable(b)) > 1:
            return 0
        col = self._blocks([(x, y - k, z) for k in range(1, 24)])
        height = 0
        for b in col:
            if not b or self._passable(b) or b.split(":")[-1] not in self.PILLAR:
                break
            around = self._blocks([(x + dx, y - height - 1, z + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))])
            if any(a and not self._passable(a) for a in around):
                break                                       # the ground: the column meets solid terrain here
            height += 1
        return height if height >= 2 else 0

    def get_off_pillar(self) -> str:
        """Dig the column under her feet one block at a time until she stands on real ground again."""
        start = self.perched()
        if not start:
            return "没有站在柱子上。"
        me = self._me() or {}
        x, z = math.floor(me["x"]), math.floor(me["z"])
        dug = 0
        for _ in range(start + 1):
            if self._stop.is_set():
                break
            me = self._me() or me
            y = math.floor(me["y"])
            below = self._block(x, y - 1, z)
            around = self._blocks([(x + dx, y - 1, z + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))])
            if (not below or self._passable(below) or below.split(":")[-1] not in self.PILLAR
                    or any(a and not self._passable(a) for a in around)):
                break                                       # her feet are level with the ground around her
            self.progress = f"从自己垫的柱子上下来（已挖 {dug} 格）"
            if not self._break(x, y - 1, z, timeout=4.0):
                break
            dug += 1
            time.sleep(0.4)
        return f"从柱子上挖下来 {dug} 格。" if dug else "柱子挖不动，没下来。"

    def pillar_up(self, blocks: int = 4, item: str = "") -> str:
        """Out of a hole the way players do it: jump, and put a block where her feet just were."""
        me = self._me()
        if not me:
            return "不知道自己在哪。"
        inv = self._inventory()
        have = [str(i.get("id")) for i in (inv.get("hotbar") or []) + (inv.get("main") or [])]
        item = item or next((i for i in ("minecraft:cobblestone", "minecraft:dirt", "minecraft:granite",
                                         "minecraft:stone", "minecraft:oak_planks") if i in have), "")
        if not item:
            return "没有方块可以垫脚。"
        start_y = math.floor(me["y"])
        misses = 0
        for _ in range(int(blocks)):
            if self._stop.is_set():
                break
            me = self._me() or me
            fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
            if not self.hold(item):
                break
            self._smooth_look(fx + .5, fy - 1, fz + .5, .3)
            self.rpc("control.jumpOnce", {})
            time.sleep(0.22)                              # near the top of the jump, above her old feet block
            self.rpc("interact.placeBlock", {"x": fx, "y": fy - 1, "z": fz, "face": "up"})
            time.sleep(0.45)
            if math.floor((self._me() or me)["y"]) <= fy:  # the block did not go down (no room, no support, too late)
                misses += 1
                if misses >= 2:
                    break                                 # stop jumping at nothing
        end_y = math.floor((self._me() or me)["y"])
        return f"垫方块爬高了 {end_y - start_y} 格（y={start_y} → y={end_y}）。"

    def escape(self, rounds: int = 8) -> str:
        """Back toward the sky the way a player gets out of a hole: stairs where there is rock, blocks under her
        feet where there is air."""
        me = self._me()
        if not me:
            return "不知道自己在哪。"
        start = math.floor(me["y"])
        for _ in range(int(rounds)):
            if self._stop.is_set():
                break
            me = self._me() or me
            fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
            if self._sky_above(fx, fy, fz, depth=5):
                return f"爬回地面附近了（y={fy}）。"
            out = self.dig_up(fy + 3, max_steps=10)
            if "停了" not in out:
                continue                                   # the stairs worked
            self.progress = "楼梯走不通，垫方块往上爬"
            out = self.pillar_up(2)
            if "爬高了 0 格" in out:
                return f"被卡住了，爬不上去：{out}"
        me = self._me() or me
        return f"从 y={start} 爬到 y={math.floor(me['y'])}。"

    @bounded_excursion(75, 48)
    def tunnel(self, length: int = 16, ores: str = "") -> str:
        """Branch mining: a 2-high tunnel into new rock, ores in the walls dug on the way, a torch every 8 blocks.
        Walking through air (her own old tunnel, a cave) is not mining: after a few such steps she turns and
        starts a new branch, so every call opens up rock she has not seen."""
        want = [o.strip() for o in str(ores or "").split(",") if o.strip()]
        dx, dz = self.heading or self._facing()
        new = got = turns = idle = steps = 0
        length = min(16, max(1, int(length)))
        while new < length and steps < length * 3 and self._work_guard():
            me = self._me()
            if not me:
                break
            steps += 1
            fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
            nx, nz = fx + dx, fz + dz
            self.progress = f"挖矿道（新挖 {new}/{length}，挖到矿 {got}）"
            cells = [(nx, fy + 1, nz), (nx, fy, nz)]
            solid = sum(not self._passable(b) for b in self._blocks(cells))
            why = self._dig_step(cells, (nx, fy, nz))
            if why or idle >= 4:
                turns += 1
                if turns > 6:
                    break
                dx, dz = -dz, dx                       # a new branch, at a right angle
                idle = 0
                continue
            if solid:
                new += 1
                idle = 0
                if new % 8 == 0:
                    self._torch_behind(fx, fy, fz)
            else:
                idle += 1
            got += self._dig_visible(want or self.VISIBLE_ORES, 8, limit=3)
        self.heading = (dx, dz)
        if want:
            got += self._dig_visible(want, 4)
        return f"矿道新挖了 {new} 格，挖到矿 {got} 块。"

    def _torch_behind(self, x: int, y: int, z: int) -> Optional[tuple]:
        """Use a loaded, empty cell on safe solid ground; return only an observed placed torch."""
        if self.crafter.counts().get("minecraft:torch", 0) <= 0:
            return None
        positions = [{"x": x, "y": y, "z": z}, {"x": x, "y": y - 1, "z": z}]
        blocks = (self.rpc("perception.blocks", {"positions": positions}) or {}).get("blocks") or []
        if len(blocks) != 2 or any(not b.get("loaded", bool(b.get("id"))) for b in blocks):
            return None
        target, support = (str(b.get("id") or "") for b in blocks)
        if not self._passable(target) or not support or self._passable(support):
            return None
        if any(h in target or h in support for h in ("water", "lava", "fire", "magma", "cactus", "campfire")):
            return None
        return self.place_at("minecraft:torch", x, y, z, walk=False)

    VISIBLE_ORES = ["minecraft:" + name for name in ("coal_ore", "iron_ore", "copper_ore", "gold_ore",
                    "diamond_ore", "emerald_ore", "lapis_ore", "redstone_ore", "deepslate_iron_ore",
                    "deepslate_diamond_ore", "deepslate_gold_ore", "nether_quartz_ore")]

    def light_area(self, force: bool = False) -> str:
        """Observe nearby lights and support before adding one; no assumed light level from time alone."""
        now = time.monotonic()
        if not force and now - self._last_light_at < 18:
            return "刚检查过这一处照明。"
        self._last_light_at = now
        me = self._me()
        if not me:
            return "无法观察当前位置，未放火把。"
        x, y, z = (math.floor(me[k]) for k in ("x", "y", "z"))
        block_light, sky_light = me.get("blockLight"), me.get("skyLight")
        measured = isinstance(block_light, (int, float))
        if measured and block_light >= 8:
            return "当前位置实测方块光照充足，保留火把。"
        if not force and not me.get("isNight") and isinstance(sky_light, (int, float)) and sky_light >= 12:
            return "当前位置有充足日光，保留火把。"
        if not measured and not force and self._sky_above(x, y, z, depth=12):
            return "头顶开阔，先保留火把。"
        scan = self.rpc("perception.scan", {"radius": 1, "find": ["minecraft:torch", "minecraft:wall_torch", "minecraft:lantern"], "findLimit": 64}) or {}
        if not measured and any(self._dist(me, p["x"], p["y"], p["z"]) < 6 and abs(p["y"] - y) <= 2 for p in scan.get("found", [])):
            return "附近已观测到照明。"
        for dx, dz in ((-1, 0), (0, -1), (1, 0), (0, 1)):
            pos = self._torch_behind(x + dx, y, z + dz)
            if pos:
                self._work_progress()
                return f"已实际补上火把 {pos}。"
        return "未找到安全落点或没有火把，照明未确认。"

    def _cave_frontiers(self, me):
        """Discover visible covered air openings at several distances, with standable loaded floors."""
        x, y, z = (math.floor(me[k]) for k in ("x", "y", "z"))
        options = [(x + dx * distance, y + dy, z + dz * distance)
                   for distance in (4, 8, 12) for dx, dz in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1))
                   for dy in (0,-1,-2,-3,1)]
        offsets = ((0,0,0),(0,1,0),(0,-1,0),(0,3,0),(0,8,0),(1,1,0),(-1,1,0),(0,1,1),(0,1,-1))
        positions = [{"x": a+dx, "y": b+dy, "z": c+dz} for a,b,c in options for dx,dy,dz in offsets]
        blocks = []
        for start in range(0, len(positions), 512):
            batch = positions[start:start + 512]
            got = (self.rpc("perception.blocks", {"positions": batch}) or {}).get("blocks", [])
            blocks.extend(got if len(got) == len(batch) else [{} for _ in batch])
        candidates = []
        for i, point in enumerate(options):
            parts = blocks[i*len(offsets):(i+1)*len(offsets)]
            if len(parts) != len(offsets) or any(p.get("loaded") is False or not p.get("id") for p in parts):
                continue
            ids = [p["id"] for p in parts]
            feet, head, floor = ids[:3]
            if not (self._passable(feet) and self._passable(head) and not self._passable(floor)
                    and any(not self._passable(v) for v in ids[3:])):
                continue
            if any(h in " ".join(ids) for h in ("lava", "water", "magma", "fire", "cactus")):
                continue
            visits = self._explore_seen.get(point, 0)
            if visits < 2:
                candidates.append((visits, -self._dist(me, *point), point))
        visible = []
        for entry in sorted(candidates)[:16]:
            point = entry[2]
            if not first_obstacle(self.rpc, (me["x"],me["y"]+1.62,me["z"]), (point[0],point[1]+1,point[2])):
                visible.append(entry)
                if len(visible) >= 6:
                    break
        return visible

    @bounded_excursion(90, 36)
    def explore_cave(self, max_seconds: int = 75, max_nodes: int = 12, ores: str = "") -> str:
        """Discover an air opening, follow its real path, and continue along the cave within one budget."""
        self._work["deadline"] = min(self._work["deadline"], time.monotonic() + max(5, min(90, int(max_seconds))))
        self._cave_attempt_after = time.monotonic() + 45
        ids = [v.strip() for v in ores.split(",") if v.strip()] or self.VISIBLE_ORES
        dimension = (self._me() or {}).get("dimension")
        if dimension != self._route_dimension:
            self._route.clear()
            self._explore_seen.clear()
            self._route_dimension = dimension
        moved = mined = 0
        for _ in range(max(1, min(20, int(max_nodes)))):
            if not self._work_guard():
                break
            mined += self._dig_visible(ids, 8, limit=3)
            self.light_area()
            me = self._me()
            if not me:
                break
            candidates = self._cave_frontiers(me)
            if not candidates:
                break
            reached = False
            for _, _, point in candidates[:4]:
                self._explore_seen[point] = self._explore_seen.get(point, 0) + 1
                self.progress = f"沿洞穴开口探路，已到达 {moved} 处、采矿 {mined} 块"
                if self._walk_to(point[0]+.5, point[1], point[2]+.5, reach=.6, timeout=min(18,max(1,self._work["deadline"]-time.monotonic()))):
                    moved += 1
                    self._route.append(point)
                    self._route = self._route[-128:]
                    self._work_progress()
                    self.places.add("cave", *point, note="实际沿通道到达的洞穴节点", merge=5)
                    reached = True
                    break
            if not reached:
                break
        if not moved and not mined:
            self._cave_attempt_after = time.monotonic() + 90
        return f"天然洞穴实际到达 {moved} 处通道节点，采到近处矿 {mined} 块；保留 {len(self._route)} 个已走返程点。"

    def _ore_approach(self, ore, within=8):
        me = self._me()
        target = (ore["x"], ore["y"], ore["z"])
        if not me or not ore.get("exposed") or self._dist(me, *target) > within:
            return False
        probe = (self.rpc("perception.blocks", {"positions": [dict(zip(("x", "y", "z"), target))], "tool": True}) or {}).get("blocks", [])
        if not probe or probe[0].get("loaded") is False or probe[0].get("canHarvest") is False:
            return False
        natural = {"stone", "deepslate", "dirt", "granite", "diorite", "andesite", "tuff", "netherrack"}
        for removed in range(3):
            if not self._work_guard():
                return False
            me = self._me() or me
            obstruction = first_obstacle(self.rpc, (me["x"], me["y"] + 1.62, me["z"]), target)
            if not obstruction:
                return self._approach(*target)
            bx, by, bz, block = obstruction
            if removed >= 2 or block.split(":")[-1] not in natural:
                return False
            if bx == math.floor(me["x"]) and bz == math.floor(me["z"]) and by < me["y"]:
                return False
            if self._danger_near([(bx, by, bz)]) or not self._break(bx, by, bz):
                return False
        return False

    def _dig_visible(self, ids: List[str], within: float, limit: int = 4) -> int:
        """Dig the wanted blocks that are open to her now and close enough to reach without walking far."""
        found = (self.rpc("perception.scan", {"radius": 1, "find": ids, "findLimit": 32}) or {}).get("found") or []
        me = self._me()
        if not me:
            return 0
        got = 0
        for f in sorted(found, key=lambda f: self._dist(me, f["x"], f["y"], f["z"])):
            if got >= limit or not self._work_guard():
                break
            if not f.get("exposed") or self._dist(me, f["x"], f["y"], f["z"]) > within:
                continue
            if f["x"] == math.floor(me["x"]) and f["z"] == math.floor(me["z"]) and f["y"] < me["y"]:
                continue  # never harvest the block supporting her feet
            probe = (self.rpc("perception.blocks", {"positions": [{k: f[k] for k in ("x", "y", "z")}], "tool": True}) or {}).get("blocks") or []
            if not probe or probe[0].get("loaded") is False or probe[0].get("canHarvest") is False:
                continue
            if self._ore_approach(f, within=within) and self._break(f["x"], f["y"], f["z"]):
                got += 1
                me = self._me() or me
                if "ore" in f["id"]:
                    self.places.add("ore", f["x"], f["y"], f["z"], note=f["id"], merge=8)
        if got:
            self.collect_drops(radius=5)
        return got

    def gather(self, blocks: List[str], count: int = 4, y: Optional[int] = None) -> str:
        """Get some of a raw thing the way a miner does: what is in sight first; if there is none, go down to where
        it is common and dig a tunnel there."""
        ids = ",".join(blocks)
        r = self.mine(ids, count)
        if r.startswith("挖了") or self._stop.is_set() or self.last_outcome.get("reason"):
            return r
        if any("ore" in b for b in blocks) and self.baritone():
            drops = {b.split(":")[-1].replace("deepslate_", "").replace("_ore", "") for b in blocks}
            before = self.crafter.counts()

            def enough():
                now = self.crafter.counts()
                return sum(max(0, n - before.get(k, 0)) for k, n in now.items()
                           if any(d in k for d in drops)) >= count
            how = "no_ore"
            tried = set()
            for _ in range(6):
                if enough() or self._stop.is_set() or not self._work_guard():
                    break
                me = self._me() or {}
                found = (self.rpc("perception.scan", {"radius": 3, "find": list(blocks), "findLimit": 48}) or {}).get("found") or []
                # only ore with a face open to the air: what a player can see from a cave or a ravine, no x-ray
                found = [f for f in found if f.get("exposed") and (f["x"], f["y"], f["z"]) not in tried]
                if not found or not me:
                    break
                f = min(found, key=lambda f: math.dist((f["x"], f["y"], f["z"]), (me["x"], me["y"], me["z"])))
                tried.add((f["x"], f["y"], f["z"]))
                self.progress = f"Baritone 规划去 {f['id'].split(':')[-1]}（{f['x']},{f['y']},{f['z']}）"
                how = self.baritone_goto(f["x"], f["y"], f["z"], near=3, timeout=120)
                if how == "stopped":
                    break
                if how == "done":
                    self.mine(",".join(blocks), count, radius=1)
            gained = {k: n - before.get(k, 0) for k, n in self.crafter.counts().items() if n > before.get(k, 0)}
            if gained:
                me = self._me() or {}
                if me:
                    self.places.add("ore", me["x"], me["y"], me["z"], note=blocks[0], merge=8)
                return f"挖了 {sum(gained.values())} 块（Baritone 规划路线）：" + "、".join(
                    f"{k.split(':')[-1]}×{n}" for k, n in gained.items())
            if how == "stopped" or self.last_outcome.get("reason"):
                return f"Baritone 挖矿停了（{how}）。"
        me = self._me() or {}
        here = math.floor(me.get("y", 64))
        if (any("ore" in b for b in blocks) and (y is None or here >= int(y) - 2)
                and time.monotonic() >= self._cave_attempt_after):
            self._cave_attempt_after = time.monotonic() + 90
            explored = self.explore_cave(max_seconds=35, max_nodes=8, ores=ids)
            if self.last_outcome.get("progress", 0) or self.last_outcome.get("reason"):
                return explored
        if y is not None and here - int(y) > 6:
            return self.dig_down(int(y))
        if y is not None and int(y) - here > 2:             # the common height is above her: climb to it first
            return self.dig_up(int(y))
        if all(b.split(":")[-1] in ("stone", "cobblestone", "deepslate") for b in blocks):
            return self.dig_down(here - 6, max_steps=8)          # stone is under everything
        return self.tunnel(20, ids)

    # --- working toward a goal --------------------------------------------------------------

    def set_goal(self, item: Optional[str], count: int = 1, by: str = "self"):
        if self.busy() and threading.current_thread() is not self._thread:
            return self.start("goal", item=item, count=count, by=by, clear=item is None)
        self.goal_item, self.goal_count, self.goal_by, self.goal_note = item, max(1, int(count or 1)), by, ""
        self.goal_stuck = False

    def _stations(self) -> set:
        me = self._me() or {}
        out = set()
        for p in (self.rpc("perception.scan", {"radius": 1}) or {}).get("pois") or []:
            pid = str(p.get("id", "")).split(":")[-1].split("/")[-1]
            if pid in ("crafting_table", "furnace") and abs(p["y"] - me.get("y", 0)) <= 4 \
                    and (p["x"] - me.get("x", 0)) ** 2 + (p["z"] - me.get("z", 0)) ** 2 <= 16 ** 2:
                out.add(f"minecraft:{pid}")
        return out

    def goal(self, item: Optional[str] = "", count: int = 1, max_steps: int = 40, by: str = "", clear: bool = False) -> str:
        """Do the next step toward the goal, again and again, until it is done or stuck. Each step is planned
        afresh from what is in the bag now (a step that brings nothing three times is given up)."""
        from src.levels.l8_agency.mc_goal import Planner
        if clear:
            self.set_goal(None, by=by or "auto")
            return "目标已交回常规进度。"
        if item:
            self.set_goal(item, count, by or self.goal_by or "self")
        if not self.goal_item:
            return "没有目标。"
        if self.index is None:
            return "不知道这个整合包的配方，没法规划。"
        planner, failures, done_steps = Planner(self.index), {}, 0
        name = self.index.name
        try:
            self.dress()                                  # armor made earlier and never put on
        except Exception:
            pass
        self._goal_deadline = time.monotonic() + 150
        for _ in range(min(12, int(max_steps))):
            if time.monotonic() >= self._goal_deadline:
                return f"本轮目标时间预算结束，实际推进 {done_steps} 步；保留目标供下轮复查。"
            if self._stop.is_set():
                return f"朝 {name(self.goal_item)} 做了 {done_steps} 步，停了。"
            have = self.crafter.counts()
            # An equipped target is already owned. Do not count other worn equipment
            # as crafting ingredients or remove it merely to satisfy inventory planning.
            inv = self._inventory()
            equipment = list(inv.get("armor") or [])
            if isinstance(inv.get("offhand"), dict):
                equipment.append(inv["offhand"])
            have[self.goal_item] = have.get(self.goal_item, 0) + sum(
                int(v.get("count", 0)) for v in equipment if v.get("id") == self.goal_item)
            stations = self._stations()
            step = planner.next_step(self.goal_item, self.goal_count, have, stations)
            if step["kind"] == "done":
                if self.goal_item in stations:
                    self._action_facts.append({"kind": "station_available", "item": self.goal_item})
                self.goal_note = "完成了"
                self._after_goal(self.goal_item)
                self.memory.did(f"做成了 {name(self.goal_item)}")
                self.on_event(f"我在 Minecraft 里自己做成了 {name(self.goal_item)}（{done_steps} 步）")
                return f"目标完成：{name(self.goal_item)} ×{self.goal_count}。"
            if step["kind"] == "stuck":
                self.goal_note, self.goal_stuck = step.text(name), True
                self.memory.learn(f"想做 {name(self.goal_item)} 卡住：{step.get('why')}", about=name(self.goal_item))
                return f"朝 {name(self.goal_item)} 卡住了：{step.get('why')}"
            key = step.text(name)
            if failures.get(key, 0) >= 3:
                self.goal_note, self.goal_stuck = f"「{key}」做了三次都没成", True
                return f"朝 {name(self.goal_item)} 卡住了：{key} 做了三次都没成（{self.last_reason or ''}）。"
            self.goal_note = key
            self.progress = f"目标 {name(self.goal_item)}：{key}"
            if self.free_slots() < 4:
                self.declutter()
            self.free_hands()
            if self.hungry():
                self.progress = "饿了，先弄点吃的"
                logger.info(f"[Goal] hungry first: {self.hunt()}")
            if step["kind"] in ("gather", "chop", "smelt"):
                want_item = step.get("input") if step["kind"] == "smelt" else step["item"]
                if self._take_from_chest(want_item, int(step["count"])):
                    self._action_facts.append({"kind": "goal_step", "goal": self.goal_item,
                                               "item": want_item, "source": "chest_inventory_gain"})
                    continue
            if step["kind"] in ("gather", "chop"):
                self.mind_the_wear(step.get("tool") or ("axe" if step["kind"] == "chop" else "pickaxe"))
            before = dict(have)
            result = self._do(step)
            if step["kind"] == "gather" and self.last_outcome.get("reason"):
                self.goal_note = self.last_outcome["reason"]
                return f"目标暂缓：{self.goal_note}；保留已有物资与返程路线。"
            logger.info(f"[Goal] {name(self.goal_item)} | {key} -> {result}")
            self.memory.did(f"{key} → {result[:60]}")
            after = self.crafter.counts()
            worked = after.get(step["item"], 0) > before.get(step["item"], 0)
            if worked:
                self._action_facts.append({"kind": "goal_step", "goal": self.goal_item, "item": step["item"],
                                           "gained": after[step["item"]] - before.get(step["item"], 0)})
            if not worked and failures.get(key, 0) >= 1:
                y = math.floor((self._me() or {}).get("y", 0))
                self.memory.learn(f"{key} 没成（在 y={y}）：{result[:70]}", about=f"{name(step['item'])} {name(self.goal_item)}")
            elif worked and failures.get(key, 0) >= 1:
                self.memory.learn(f"{key} 试了几次后成了：{result[:70]}", about=f"{name(step['item'])} {name(self.goal_item)}")
            if after.get(step["item"], 0) > before.get(step["item"], 0) or step["kind"] in ("craft", "smelt") and result.startswith(("做好了", "烧好了")):
                done_steps += 1
                failures.pop(key, None)
            elif any(after.get(i, 0) > before.get(i, 0) for i in after):
                done_steps += 1
                failures[key] = failures.get(key, 0) + 0.5     # moving (cobblestone from the stairs), not there yet
            else:
                failures[key] = failures.get(key, 0) + 1
        return f"朝 {name(self.goal_item)} 做了 {done_steps} 步，还没做完，歇一下再继续。"

    def _do(self, step) -> str:
        k = step["kind"]
        if k == "chop":
            return self.chop_tree(prefer=step["item"])
        if k == "craft":
            return self.crafter.craft(step["item"], int(step["count"]))
        if k == "smelt":
            return self.smelt(step["input"], int(step["count"]))
        if k == "gather":
            return self.gather(step["blocks"], int(step["count"]), step.get("y"))
        if k == "hunt":
            return self.hunt_resource(step["item"], int(step["count"]))
        return ""

    def _after_goal(self, item: str):
        """Armor is worn and tools go where the hand finds them, as soon as they are made."""
        if item.endswith(("_helmet", "_chestplate", "_leggings", "_boots", "shield")):
            self.dress()
        elif item.endswith(("_pickaxe", "_axe", "_sword", "_shovel")):
            self.tidy()

    # --- keeping the bag usable ---------------------------------------------------------------

    JUNK = ("leaf_litter", "moss_carpet", "_sapling", "_spores", "rotten_flesh", "poisonous_potato", "dead_bush", "_petals",
            "short_grass", "tall_grass", "fern", "dirt", "coarse_dirt", "gravel", "granite", "diorite", "andesite", "tuff",
            "_seeds", "_leaves")
    KEEP = {"minecraft:cobblestone": 64, "minecraft:cobbled_deepslate": 0, "minecraft:stick": 32, "minecraft:torch": 128}
    # cheap things that pile up: past this many the rest is not worth a bag slot (raw copper x142, bowls x64 -
    # a full bag of them blocked hunting, storing and every goal for an hour)
    BULK = {"minecraft:raw_copper": 32, "minecraft:bowl": 4, "minecraft:flint": 8, "minecraft:string": 16,
            "minecraft:bone": 16, "minecraft:arrow": 64, "minecraft:wheat_seeds": 0, "minecraft:beetroot_seeds": 0,
            "minecraft:sand": 32, "minecraft:clay_ball": 16, "minecraft:feather": 16, "minecraft:spider_eye": 0}

    def free_slots(self) -> int:
        inv = self._inventory()
        return 36 - len((inv.get("hotbar") or []) + (inv.get("main") or []))

    def declutter(self, want_free: int = 6, blocks_only: bool = False) -> str:
        """Make room the way a player does when the bag is full: throw out junk (leaf litter, dirt, saplings...),
        then whatever is far more than needed (a few stacks of logs, cobblestone), then worse spare tools."""
        from src.levels.l8_agency.mc_goal import is_log, tool_tier
        st = self.rpc("container.state", {}) or {}
        if st.get("open"):
            self.rpc("container.close", {})
        inv = self._inventory()
        items = [dict(it) for it in (inv.get("hotbar") or []) + (inv.get("main") or [])]
        free = 36 - len(items)
        if free >= want_free:
            return "背包还有地方。"
        me = self._me() or {}
        stored = ""
        if me and not getattr(self, "_storing", False):
            # no chest of hers nearby but one in the bag: put it down here and use it, as a player does with a
            # full inventory far from home (it goes on her map and is found again later)
            if not self.places.near(me, ("chest",), within=24, n=1) and any(
                    it["id"] == "minecraft:chest" for it in items):
                self.progress = "背包满了：就地放个箱子存东西"
                if self.place_near("minecraft:chest"):
                    stored = "就地放了个箱子。"
            if self.places.near(me, ("chest",), within=24, n=1):
                stored += self.store()
                if self.free_slots() >= want_free:
                    return stored
                inv = self._inventory()
                items = [dict(it) for it in (inv.get("hotbar") or []) + (inv.get("main") or [])]
                free = 36 - len(items)
        drop: List[Dict[str, Any]] = []
        name = lambda it: str(it["id"]).split(":")[-1]
        drop += sorted((it for it in items if any(j in name(it) for j in self.JUNK) and not name(it).endswith("_block")),
                       key=lambda it: int(it.get("count", 0)), reverse=True)
        totals: Dict[str, int] = {}
        for it in sorted(items, key=lambda it: -int(it.get("count", 0))):
            group = "logs" if is_log(it["id"]) else "planks" if name(it).endswith("_planks") else it["id"]
            cap = 64 if group in ("logs", "planks") else self.KEEP.get(it["id"])
            if cap is None:
                continue
            totals[group] = totals.get(group, 0) + int(it.get("count", 0))
            if totals[group] - int(it.get("count", 0)) >= cap and it not in drop:
                drop.append(it)                                  # a whole stack beyond what is worth carrying
        bulk: Dict[str, int] = {}
        for it in sorted(items, key=lambda it: -int(it.get("count", 0))):
            cap = self.BULK.get(it["id"])
            if cap is None or it["id"] == self.goal_item or it in drop:
                continue
            n = int(it.get("count", 0))
            bulk[it["id"]] = bulk.get(it["id"], 0) + n
            # a stack past the cap, or one stack far over a small cap (64 bowls when 4 is plenty)
            if bulk[it["id"]] - n >= cap or (cap < 16 and n > 2 * max(1, cap)):
                drop.append(it)
        for kind in ("pickaxe", "axe", "sword", "shovel", "hoe"):
            tools = [it for it in items if name(it).endswith("_" + kind)]
            best = max((tool_tier(it["id"]) for it in tools), default=-1)
            # worse ones go; a spare of the best kind stays (a pickaxe wears out deep in a mine)
            drop += [it for it in tools if tool_tier(it["id"]) < best and it not in drop]
        if blocks_only:
            cheap = {"minecraft:" + name for name in ("dirt", "cobblestone", "cobbled_deepslate", "gravel",
                                                      "andesite", "diorite", "granite", "tuff")}
            reserve = sum(int(it.get("count", 0)) for it in items if it["id"] in cheap)
            safe = []
            for it in drop:
                if (it["id"] in cheap and it["id"] != self.goal_item
                        and reserve - int(it.get("count", 0)) >= 64):
                    safe.append(it)
                    reserve -= int(it.get("count", 0))
                elif it["id"] in self.BULK:
                    safe.append(it)                              # cheap surplus is as safe to drop as dirt
            drop = safe
        drop = drop[:max(0, want_free - free)]
        if not drop:
            return (stored or "") + "背包满了，但没有能扔的。"
        # Throw it the way a player does: face open space, toss it (Q) up and away, then walk off before the
        # pickup delay (2 s) is over - otherwise she just picks it all up again.
        me = self._me() or {}
        fx, fy, fz = math.floor(me.get("x", 0)), math.floor(me.get("y", 64)), math.floor(me.get("z", 0))
        open_dirs = []
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cells = [(fx + dx * k, fy + h, fz + dz * k) for k in (1, 2, 3) for h in (0, 1)]
            open_dirs.append((sum(self._passable(b) for b in self._blocks(cells)), (dx, dz)))
        open_dirs.sort(reverse=True)
        tdx, tdz = open_dirs[0][1]
        self._smooth_look(me.get("x", 0) + tdx * 6, me.get("y", 64) + 3.5, me.get("z", 0) + tdz * 6, 0.3)
        sel = int(inv.get("selectedSlot") or 0)
        thrown = []
        for it in drop:
            if self._stop.is_set():
                break
            slot = int(it["slot"])
            if slot >= 9:                                          # into the hand first (hotbar 8), then toss
                from src.levels.l8_agency.mc_craft import move_inventory_stack
                if not move_inventory_stack(self.rpc, slot, 8):
                    continue
                slot = 8
            self.rpc("inventory.selectHotbar", {"slot": slot})
            self.rpc("interact.dropItem", {"wholeStack": True})
            time.sleep(0.1)
            thrown.append(f"{name(it)}×{it.get('count', 1)}")
        self.rpc("inventory.selectHotbar", {"slot": sel})
        away = next((d for n, d in open_dirs if d == (-tdx, -tdz) and n >= 4), open_dirs[1][1] if len(open_dirs) > 1 else (-tdx, -tdz))
        self._walk_to(fx + away[0] * 4, fy, fz + away[1] * 4, reach=1.0, timeout=4)
        self.dropped_at = (time.time(), {"x": fx + tdx * 2, "z": fz + tdz * 2})
        self.thrown_ids = {str(it["id"]) for it in drop}
        return "扔掉了：" + "、".join(thrown) + "。"


    # --- a player who looks after their gear ----------------------------------------------------

    def mind_the_wear(self, kind: str = "pickaxe", left: int = 12) -> str:
        """A tool about to break is replaced before it does (a new one of the same kind and material, if the bag
        has what it takes); worn ones are not used while a fresh spare is there."""
        tool = self.best_tool(kind)
        if not tool or not self.hold(tool):
            return ""
        hand = (self.rpc("player.getEquipment", {}) or {}).get("mainHand") or {}
        if not hand.get("maxDamage"):
            return ""
        remaining = int(hand["maxDamage"]) - int(hand.get("damage") or 0)
        if remaining > left:
            return ""
        spares = sum(1 for it in self.crafter.inventory() if it["id"] == tool) - 1
        if spares > 0:
            return f"{tool.split(':')[-1]} 快坏了，还有备用的。"
        out = self.crafter.craft(tool, 1)
        logger.info(f"[Skill] {tool} has {remaining} uses left: {out}")
        return out

    def _make_tool(self, kind: str, tier: int = 0) -> str:
        """The best tool of this kind the bag can make right now (stone if there is cobblestone, else wood)."""
        if time.time() - self._made_tool_at.get(kind, 0) < 30:
            return ""
        self._made_tool_at[kind] = time.time()
        have = self.crafter.counts()
        options = []
        if have.get("minecraft:iron_ingot", 0) >= 3:
            options.append("iron")
        if have.get("minecraft:cobblestone", 0) + have.get("minecraft:cobbled_deepslate", 0) >= 3:
            options.append("stone")
        options.append("wooden")
        for mat in options:
            if ["wooden", "stone", "iron"].index(mat) < tier:
                continue
            out = self.crafter.craft(f"minecraft:{mat}_{kind}", 1)
            self.memory.did(f"没有合适的{kind}，做了一把 {mat}_{kind}：{out[:30]}")
            if out.startswith("做好了"):
                self.memory.learn(f"{kind} 用坏了就马上做新的（这次做了 {mat}_{kind}），不然挖了也白挖", about=kind)
                return out
        return ""

    ARMOR_SLOTS = {"helmet": "helmet", "chestplate": "chest", "leggings": "legs", "boots": "boots"}

    def dress(self) -> str:
        """Wear the best armor in the bag and keep a shield in the off hand."""
        from src.levels.l8_agency.mc_goal import tool_tier
        gear = self.rpc("player.getEquipment", {}) or {}
        in_nether = (self.rpc("player.getState", {}) or {}).get("dimension") == "minecraft:the_nether"
        put_on = []
        for piece, slot in self.ARMOR_SLOTS.items():
            worn = (gear.get(slot) or {}).get("id") or ""
            if in_nether and worn.startswith("minecraft:golden_"):
                continue                           # retain piglin protection during the expedition
            have = [it["id"] for it in self.crafter.inventory() if str(it["id"]).endswith("_" + piece)]
            best = max(have, key=tool_tier, default=None)
            if best and (not worn or tool_tier(best) > tool_tier(worn)):
                if self.hold(best):
                    self.rpc("interact.useItem", {"hand": "main"})       # right-click puts it on (swaps the old one)
                    time.sleep(0.3)
                    actual = (self.rpc("player.getEquipment", {}) or {}).get(slot) or {}
                    if actual.get("id") == best:
                        put_on.append(best.split(":")[-1])
        off = (gear.get("offHand") or {}).get("id") or ""
        shield = next((it for it in self.crafter.inventory() if str(it["id"]).endswith("shield")), None)
        if shield and "shield" not in off:
            from src.levels.l8_agency.mc_craft import move_inventory_stack
            if move_inventory_stack(self.rpc, int(shield["slot"]), 40):
                put_on.append("shield（副手）")
        return ("穿戴好了：" + "、".join(put_on)) if put_on else "装备已经是最好的了。"

    FOOD_ANIMALS = ("cow", "pig", "sheep", "chicken", "rabbit", "goat")
    FOOD = FOOD_NAMES

    def hungry(self, food_bar: Optional[float] = None) -> bool:
        """The food bar is going down and there is (almost) nothing to eat in the bag."""
        if food_bar is None:
            food_bar = float((self.rpc("player.getState", {}) or {}).get("food", 20))
        if food_bar > 15:
            return False
        have = self._food_count()
        return have < 3

    def hunt(self, count: int = 3) -> str:
        """Get food like a player in the wild: find an animal, sword out, hit it until it drops, pick it up;
        then cook the meat if there is a furnace to be had."""
        if time.time() < self._hunt_retry_after:
            return self.forage()
        food_before = self._food_count()
        me = self._me() or {}
        if me and not self._sky_above(math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])):
            ents = (self.rpc("perception.entities", {"radius": 32}) or {}).get("entities") or []
            if not any(any(a in str(e.get("type", "")).split(":")[-1] for a in self.FOOD_ANIMALS) for e in ents):
                self.progress = "地下没有动物：先挖楼梯回地面再打猎"
                climbed = self.dig_up(math.floor(me["y"]) + 120, max_steps=160, until_sky=True)
                me = self._me() or me
                if not self._sky_above(math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])):
                    self._hunt_retry_after = time.time() + 60
                    return f"想上地面打猎，还没挖上去：{climbed}"
        got = 0
        for _ in range(int(count) * 3):
            if self._stop.is_set() or got >= int(count):
                break
            ents = (self.rpc("perception.entities", {"radius": 32}) or {}).get("entities") or []
            prey = sorted((e for e in ents if any(a in str(e.get("type", "")).split(":")[-1] for a in self.FOOD_ANIMALS)
                           and "baby" not in str(e.get("name", "")).lower()), key=lambda e: e.get("distance", 99))
            if not prey:
                break
            target = prey[0]
            sword = self.best_tool("sword") or self.best_tool("axe")
            self.progress = f"去打 {str(target.get('type')).split(':')[-1]} 做吃的"
            for _ in range(12):
                if self._stop.is_set():
                    break
                if not self._walk_to(target["x"], target["y"], target["z"], reach=2.5, timeout=8):
                    break
                if sword:
                    self.hold(sword)
                self._smooth_look(target["x"], float(target["y"]) + 0.6, target["z"], 0.15)
                self.rpc("interact.attackEntity", {"uuid": target.get("uuid")})
                time.sleep(0.65)                                       # a full swing, not spam
                now = [e for e in (self.rpc("perception.entities", {"radius": 32}) or {}).get("entities") or []
                       if e.get("uuid") == target.get("uuid")]
                if not now:
                    got += 1
                    break
                target = now[0]
            self.collect_drops(radius=6)
        cooked = ""
        raw = [i for i in self.crafter.counts() if i.split(":")[-1] in ("beef", "porkchop", "mutton", "chicken", "rabbit")]
        if raw and got:
            cooked = self.smelt(raw[0], self.crafter.counts()[raw[0]])
        actual = max(0, self._food_count() - food_before)
        if actual:
            return f"实际取得食物 ×{actual}。{cooked}"
        self._hunt_retry_after = time.time() + 120
        return "没有实际取得动物食物，暂停重复狩猎2分钟。" + self.forage()

    def _food_count(self):
        return sum(n for item, n in self.crafter.counts().items() if self._edible(item))

    def _edible(self, item):
        return is_edible(item)

    def _forage_safe(self):
        me = self._me() or {}
        seen = self.rpc("perception.entities", {"radius": 12}) or {}
        entities = seen.get("entities")
        return bool(me and not me.get("onFire") and not me.get("inLava")
                    and isinstance(entities, list)
                    and not any(e.get("kind") == "hostile" and float(e.get("distance", 0)) <= 12 for e in entities))

    def _collect_food_drops(self, radius=12, deadline=None, cautious=False):
        """Pick only food/ingredients, and only count inventory deltas, never walk attempts."""
        entities = (self.rpc("perception.entities", {"radius": radius, "kinds": ["item"]}) or {}).get("entities", [])
        for e in entities[:24]:
            item = (e.get("item") or {}).get("id", "")
            if not (self._edible(item) or item == "minecraft:wheat"):
                continue
            if self._stop.is_set() or deadline is not None and time.monotonic() >= deadline:
                break
            if cautious:
                me = self._me() or {}
                if not self._forage_safe():
                    break
                if not me or self._dist(me,e["x"],e["y"],e["z"]) > radius:
                    continue
            self._walk_to(e["x"], e["y"], e["z"], reach=.2, timeout=min(8,max(.5,deadline-time.monotonic())) if deadline else 8)
            time.sleep(.2)

    def forage(self) -> str:
        """Food drops, bread, verified ripe crops, then one short loaded-terrain exploration leg."""
        if time.time() < self._forage_retry_after:
            return "觅食路线暂缓，先整理背包或补给，不重复扫描原地。"
        self._forage_retry_after = time.time() + 45
        inv = self._inventory()
        slots = (inv.get("hotbar") or []) + (inv.get("main") or [])
        free = 36 - sum(1 for slot in slots if slot.get("id") and slot.get("count", 0) > 0)
        if free < 2:
            return "采集食物前先腾背包。" + self.declutter(want_free=4, blocks_only=True)
        me = self._me() or {}
        if not me or float(me.get("health", 20)) <= 0:
            return "生命或位置不适合外出觅食，先停在安全处。"
        rescue = float(me.get("health", 20)) < 10
        deadline = time.monotonic() + (25 if rescue else 75)
        before = self._food_count()
        if self.crafter.counts().get("minecraft:wheat", 0) >= 3:
            self.crafter.craft("minecraft:bread", 1)
            if self._food_count() > before:
                self._forage_retry_after = 0
                return "实际取得食物 ×1，使用现有小麦做了面包。"
        if rescue and not self._forage_safe():
            return "低血觅食周围有危险或观测不完整，先避险。"
        if rescue:
            self._collect_food_drops(radius=6,deadline=deadline,cautious=True)
        else:
            self._collect_food_drops()
        crops = {"minecraft:wheat": 7, "minecraft:carrots": 7, "minecraft:potatoes": 7,
                 "minecraft:beetroots": 3, "minecraft:sweet_berry_bush": 2, "minecraft:melon": None}
        found = (self.rpc("perception.scan", {"radius": 2, "find": list(crops)}) or {}).get("found", [])
        found = sorted((p for p in found if p.get("id") in crops and
                        math.dist(tuple(me[k] for k in ("x", "y", "z")), tuple(p[k] for k in ("x", "y", "z"))) <= (6 if rescue else 40)),
                       key=lambda p: self._dist(me, p["x"], p["y"], p["z"]))[:24]
        harvested = 0
        for p in found:
            if harvested >= 3 or self._stop.is_set() or time.monotonic() >= deadline:
                break
            if rescue and not self._forage_safe():
                break
            block = ((self.rpc("perception.blocks", {"positions": [{k: p[k] for k in ("x", "y", "z")}]}) or {}).get("blocks") or [{}])[0]
            age = (block.get("properties") or {}).get("age")
            if block.get("id") != p["id"] or (crops[p["id"]] is not None and
                    (not isinstance(age, (int, float)) or age < crops[p["id"]])):
                continue                         # Missing maturity is unknown, never ripe by assumption.
            self.progress = "采集已确认成熟的食物作物"
            if not self._approach(p["x"], p["y"], p["z"]) or not self._break(p["x"], p["y"], p["z"]):
                continue
            harvested += 1
            if rescue:
                self._collect_food_drops(radius=6,deadline=deadline,cautious=True)
            else:
                self._collect_food_drops()
            seed = {"minecraft:wheat": "minecraft:wheat_seeds", "minecraft:carrots": "minecraft:carrot",
                    "minecraft:potatoes": "minecraft:potato", "minecraft:beetroots": "minecraft:beetroot_seeds"}.get(p["id"])
            if seed and self.crafter.counts().get(seed, 0) > (1 if self._edible(seed) else 0) and self.hold(seed):
                self.rpc("interact.placeBlock", {"x": p["x"], "y": p["y"] - 1, "z": p["z"], "face": "up"})
        if self.crafter.counts().get("minecraft:wheat", 0) >= 3:
            self.crafter.craft("minecraft:bread", 1)
        actual = max(0, self._food_count() - before)
        if actual:
            self._forage_retry_after = 0
            return f"实际取得食物 ×{actual}，来自掉落物/成熟作物/面包。"
        if rescue:
            return "低血救急觅食已检查六格内食物，尚未取得；不带伤远行或狩猎。"
        # Exploration never digs; the shared body/nav still handles every obstacle along the leg.
        me = self._me() or me
        x, y, z = (math.floor(me[k]) for k in ("x", "y", "z"))
        now = time.time()
        self._forage_visited = {k: v for k, v in self._forage_visited.items() if now - v < 600}
        for dx, dz in ((6, 0), (0, 6), (-6, 0), (0, -6)):
            for dy in (0, -1, 1):
                target = (x + dx, y + dy, z + dz)
                key = (me.get("dimension", self.places.active_dim), target[0] // 4, target[2] // 4)
                if key in self._forage_visited:
                    continue
                floor, feet, head = self._blocks([(target[0], target[1] + h, target[2]) for h in (-1, 0, 1)])
                if not self._shelter_block(floor) or not all(b and self._passable(b) for b in (feet, head)):
                    continue
                self._forage_visited[key] = now
                self.progress = "向已观测安全地面探路寻找食物"
                self._walk_to(target[0] + .5, target[1], target[2] + .5, reach=.5, timeout=12)
                after = self._me() or me
                if math.hypot(after["x"] - me["x"], after["z"] - me["z"]) >= 2:
                    return "已实际探路寻找食物；尚未取得食物，下一次重查新位置。"
                return "觅食短途没有实际推进，暂缓此方向，未取得食物。"
        self._forage_retry_after = now + 120
        return "没发现可验证成熟的作物或安全探路方向，暂缓2分钟，不重复原地狩猎。"

    def hunt_resource(self, item: str, count: int = 1) -> str:
        """One bounded resource hunt, using current entity positions and actual collected loot."""
        from src.levels.l8_agency.mc_goal import HUNT, tool_tier
        mob = HUNT.get(item)
        if mob is None:
            return "没有这个资源对应的狩猎动作。"
        hostile = mob not in ("minecraft:cow", "minecraft:chicken", "minecraft:sheep")
        me = self._me() or {}
        if mob == "minecraft:blaze" and me.get("dimension") != "minecraft:the_nether":
            return "烈焰棒需要进入下界并找到烈焰人，当前维度不对。"
        weapon = self.best_tool("sword") or self.best_tool("axe")
        if hostile and (not weapon or tool_tier(weapon) < 2):
            return "主动打敌对生物前先补铁剑或铁斧，不拿挖矿工具硬冲。"
        before = self.crafter.counts().get(item, 0)
        count = max(1, min(16, int(count)))
        deadline = time.monotonic() + 35
        hits = 0
        while not self._stop.is_set() and time.monotonic() < deadline and hits < 24:
            me = self._me() or {}
            if not me or float(me.get("health", 0)) < (14 if hostile else 8) or int(me.get("food", 0)) < 10:
                return "狩猎中止：当前血量、饥饿或感知不足，先恢复。"
            have = self.crafter.counts().get(item, 0)
            if have - before >= count:
                return f"已实际捡到 {item} ×{have - before}。"
            ents = (self.rpc("perception.entities", {"radius": 24}) or {}).get("entities") or []
            targets = [e for e in ents if e.get("type") == mob and e.get("uuid")
                       and all(k in e for k in ("x", "y", "z"))]
            if not targets:
                break
            target = min(targets, key=lambda e: float(e.get("distance", 99)))
            if not self._walk_to(target["x"], target["y"], target["z"], reach=2.7, timeout=5):
                break
            if weapon:
                self.hold(weapon)
            # Reacquire after walking: a moving target's old coordinates are not an aim point.
            fresh = (self.rpc("perception.entities", {"radius": 8}) or {}).get("entities") or []
            target = next((e for e in fresh if e.get("uuid") == target["uuid"]), None)
            if target is None or float(target.get("distance", 99)) > 3.0:
                continue
            self.progress = f"获取 {item.split(':')[-1]}，已实际捡到 {have - before}/{count}"
            self._smooth_look(target["x"], target["y"] + 1.0, target["z"])
            self.rpc("interact.attackEntity", {"uuid": target["uuid"]})
            hits += 1
            time.sleep(.65)
            self.collect_drops(radius=4)
        got = max(0, self.crafter.counts().get(item, 0) - before)
        return (f"本次实际捡到 {item} ×{got}，目标还差 {max(0, count - got)}。" if got else
                f"尚未取得 {item}；附近没有可达目标或安全条件不足，不能把目标消失当作击杀。")


    # --- home, chests, and finding the way back ----------------------------------------------------

    def _sky_above(self, x: int, y: int, z: int, depth: int = 24) -> bool:
        col = self._blocks([(x, y + k, z) for k in range(2, depth)])
        return bool(col) and all(b and (self._passable(b) or "leaves" in b) for b in col)

    def home_route_status(self, me=None) -> Dict[str, Any]:
        """Runtime route budget; never edit a remembered home just because it is unreachable."""
        h = self.places.home
        result = {"blocked": False, "camp_retry_after": self._camp_retry_after}
        if not h:
            return result
        key = tuple(h.get(k) for k in ("x", "y", "z", "dim"))
        record = self._home_attempts.get(key, {})
        if record.get("until", 0) > time.time():
            return {**result, "blocked": True, "reason": "返家路线连续无进展或已用完3次预算，暂缓10分钟",
                    "retry_after": record["until"]}
        # Legacy home markers could be written without having placed even a chest. A distant
        # underground marker with no useful anchor must not cause automatic excavation from above.
        anchors = self.places.near(h, ("chest", "bed"), within=12, n=2)
        if me and abs(me.get("y", h["y"]) - h["y"]) > 12 and not anchors:
            return {**result, "blocked": True, "reason": "旧家址高差超过12格且没有箱子或床的记录，先核实或建安全地面营地"}
        return result

    def _camp_site(self, x: int, y: int, z: int) -> bool:
        floor = self._blocks([(x + dx, y - 1, z + dz) for dx in range(-1, 2) for dz in range(-1, 2)])
        room = self._blocks([(x + dx, y + dy, z + dz) for dx in range(-1, 2)
                             for dz in range(-1, 2) for dy in (0, 1)])
        return (all(self._shelter_block(b) for b in floor)
                and all(b and (self._passable(b) or b in ("minecraft:chest", "minecraft:crafting_table", "minecraft:furnace"))
                        for b in room) and self._sky_above(x, y, z))

    # natural ground she may dig away or fill in to level a camp - never logs, planks or anything placed
    LEVELABLE = ("stone", "dirt", "coarse_dirt", "grass_block", "gravel", "sand", "andesite", "diorite", "granite",
                 "tuff", "deepslate", "cobbled_deepslate", "podzol", "mycelium", "moss_block", "rooted_dirt", "clay",
                 "calcite", "snow_block", "short_grass", "tall_grass", "fern", "large_fern", "snow", "dead_bush")

    def _make_camp_site(self, x: int, y: int, z: int) -> bool:
        """No flat spot nearby (a hillside, a forest floor): level one where she stands, as a player does -
        dig the 3x3 two-high space clear, fill holes in the floor with cobblestone. Natural blocks only."""
        room = [(x + dx, y + dy, z + dz) for dy in (0, 1) for dx in (-1, 0, 1) for dz in (-1, 0, 1)]
        floor = [(x + dx, y - 1, z + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)]
        room_ids, floor_ids = self._blocks(room), self._blocks(floor)
        if not all(room_ids) or not all(floor_ids):
            return False
        natural = lambda b: b.split(":")[-1] in self.LEVELABLE or b.split(":")[-1].endswith("_leaves")
        dig = [c for c, b in zip(room, room_ids) if not self._passable(b)]
        fill = [c for c, b in zip(floor, floor_ids) if self._passable(b)]
        if any(not natural(b) for b, c in zip(room_ids, room) if c in dig):
            return False                                  # a log, a door, someone's wall: not here
        if len(dig) > 12 or len(fill) > 5:
            return False                                  # a cliff face or a pit: too much work for a first camp
        blocks = sum(n for i, n in self.crafter.counts().items() if i in self.STEP_BLOCKS)
        if blocks < len(fill):
            return False
        self.progress = f"附近没有平地：自己平整一块（挖 {len(dig)} 格、填 {len(fill)} 格）"
        for c in sorted(dig, key=lambda c: -c[1]):        # top first, so nothing falls onto the floor
            if self._stop.is_set() or not self._break(*c, timeout=4.0):
                return False
        for c in fill:
            if self._stop.is_set() or not self._step_block(*c):
                return False
        return self._camp_site(x, y, z)

    def go_to(self, x: int = 0, y: int = 64, z: int = 0) -> str:
        """Walk somewhere far: out of the ground first if she is deep below it, then the long walk."""
        me = self._me()
        if not me:
            return "不知道自己在哪。"
        x, y, z = int(x), int(y), int(z)
        if self.baritone():
            how = self.baritone_goto(x, y, z, near=2, timeout=180)
            me = self._me() or me
            if math.hypot(me["x"] - x, me["z"] - z) <= 3.5 and abs(me["y"] - y) <= 3:
                return f"到了 ({x},{y},{z}) 附近。"
            if how == "stopped":
                return "走路已停止，尚未确认到达。"
            self.progress = f"Baritone 没走到（{how}），换手动路线"
        if me["y"] < y - 2 and not self._sky_above(math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])):
            self.progress = f"先挖楼梯回到地面（y={y}）"
            self.dig_up(y, until_sky=True)
        self.progress = f"走去 ({x},{y},{z})"
        if self.travel is not None:
            res = self.travel(x, y, z)
            me = self._me() or {}
            if not me:
                return "走路后无法观测坐标，尚未确认到达。"
            if math.hypot(me["x"] - x, me["z"] - z) <= 3 and abs(me["y"] - y) <= 2.5:
                return f"到了 ({x},{y},{z}) 附近。"
            self.progress = "核对坐标后仍未到达，继续处理路线和高差"
        # without the walker: hops of a few dozen blocks along the straight line
        for _ in range(30):
            me = self._me() or me
            d = math.dist((me["x"], me["z"]), (x, z))
            if self._stop.is_set():
                return "走路已停止，尚未确认到达。"
            if d <= 3 and abs(me["y"] - y) <= 2.5:
                break
            if d <= 8 and abs(me["y"] - y) > 3:
                # A cave base and the surface can share X/Z. Use the existing hazard-checked
                # stair builder rather than claiming arrival or dropping straight down.
                if me["y"] > y:
                    result = self.dig_down(y, max_steps=24)
                else:
                    result = self.dig_up(y, max_steps=24, until_sky=False)
                after = self._me() or {}
                if not after or abs(after["y"] - y) > 2.5:
                    return f"还没到目的地，高差尚未解决：{result}"
                me = after
                d = math.hypot(me["x"] - x, me["z"] - z)
                if d <= 3:
                    break
            k = min(1.0, 32.0 / d)
            if not self._walk_to(me["x"] + (x - me["x"]) * k, y if d < 32 else me["y"], me["z"] + (z - me["z"]) * k,
                                 reach=2.0, timeout=30):
                return f"走到一半走不动了（还差 {int(d)} 格）。"
        me = self._me() or {}
        if me and math.hypot(me["x"] - x, me["z"] - z) <= 3 and abs(me["y"] - y) <= 2.5:
            return f"到了 ({x},{y},{z}) 附近。"
        return "还没到目的地，本次路线步数用完，需继续规划。"

    def set_home(self) -> str:
        """Settle down like a player does on the first day: on the surface, a chest, a crafting table and a
        furnace side by side - and remember the spot as home."""
        me = self._me()
        if not me:
            return "不知道自己在哪。"
        if time.time() < self._camp_retry_after:
            return "附近营地选址或材料还没解决，暂缓重复安家，先补给。"
        self._camp_retry_after = time.time() + 120
        if me.get("dimension", self.places.active_dim) != "minecraft:overworld":
            return "先在主世界选择地面营地，保留当前维度返程点。"
        if float(me.get("health", 20)) < 8:
            return "生命太低，先在安全处恢复再安家。"
        fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
        if not self._sky_above(fx, fy, fz):
            self.progress = "先挖上地面再安家"
            climbed = self.dig_up(fy + 120, max_steps=160, until_sky=True)
            me = self._me() or me
            fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
            if not self._sky_above(fx, fy, fz):
                return f"家要安在地面上，还没挖上去：{climbed}"
        ring = [(0, 0)] + [(dx, dz) for r in (3, 6, 9) for dx, dz in
                            ((r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r))]
        site = next(((fx + dx, fy + dy, fz + dz) for dx, dz in ring for dy in (0, 1, -1)
                     if self._camp_site(fx + dx, fy + dy, fz + dz)), None)
        if site is None and self._make_camp_site(fx, fy, fz):
            site = (fx, fy, fz)
        minimal = False
        if site is None:
            # no flat 3x3 and too steep to level: a player still puts a chest down on any solid ground under the
            # open sky - the hut can come later. Without this she carried a full bag around for an hour.
            under = self._blocks([(fx, fy - 1, fz)])[0]
            if self._sky_above(fx, fy, fz) and under and not self._passable(under) and "water" not in under                     and "lava" not in under:
                site, minimal = (fx, fy, fz), True
        if site is None:
            return "附近还没有观测确认的3×3平整安全地面，保留旧家址，暂不登记新家。"
        if math.hypot(site[0] - me["x"], site[2] - me["z"]) > 1.5:
            if not self._walk_to(*site, reach=1, timeout=15):
                return "安全营地候选点还没走到，保留旧家址。"
        me = self._me() or {}
        if not me:
            return "营地位置观测中断，保留旧家址。"
        fx, fy, fz = (math.floor(me[k]) for k in ("x", "y", "z"))
        if not minimal and not self._camp_site(fx, fy, fz):
            return "落脚点不符合安全营地条件，保留旧家址。"
        made = []
        have = self.crafter.counts()
        wood = sum(n for i, n in have.items() if i.endswith(("_log", "_planks", "_wood", "_stem")))
        if not have.get("minecraft:chest") and wood < 3:
            # no chest and nothing to make one from (her bag was lost when she died): get wood first,
            # instead of reporting "no chest" and standing on the camp site forever
            self.progress = "安家缺木头：先去砍棵树"
            self.chop_tree(max_logs=6)
        for item, need in (("minecraft:chest", True), ("minecraft:crafting_table", False), ("minecraft:furnace", False)):
            kind = item.split(":")[-1]
            if not need and self.places.near(me, (kind,), within=8, n=1):
                continue
            if self.crafter.counts().get(item, 0) == 0:
                self.crafter.craft(item, 1)
            if self.crafter.counts().get(item, 0) and self.place_near(item):
                made.append({"chest": "箱子", "crafting_table": "工作台", "furnace": "熔炉"}[kind])
        chests = self.places.near(me, ("chest",), within=8, n=4)
        if not any(self._block(p["x"], p["y"], p["z"]) in ("minecraft:chest", "minecraft:trapped_chest") for p in chests):
            return "营地尚未观测到实际箱子，保留旧家址，先补木板或放箱子。"
        old = self.places.home
        if old and math.dist((fx, fy, fz), tuple(old[k] for k in ("x", "y", "z"))) > 8:
            self.places.add("spot", old["x"], old["y"], old["z"], note="旧家址，暂缓路线；保留以后核查", dim=old.get("dim"))
        self.places.set_home(fx, fy, fz, note=("临时营地（坡地，箱子先放下）；" if minimal else "已观测安全平地与箱子；")
                             + "、".join(made))
        self._camp_retry_after = 0.0
        self.memory.did(f"在 ({fx},{fy},{fz}) 安了家")
        self.on_event(f"我在 Minecraft 里安了家，在 ({fx},{fy},{fz})" + (f"，放了{'、'.join(made)}" if made else ""))
        stored = self.store() if made and "箱子" in made else ""
        return f"家安在 ({fx},{fy},{fz})" + (f"，放了{'、'.join(made)}" if made else "") + f"。{stored}"

    def go_home(self) -> str:
        h = self.places.home
        if not h:
            return "还没有家（用 set_home 安一个）。"
        before = self._me() or {}
        route = self.home_route_status(before)
        if route["blocked"]:
            return "暂缓回家：" + route["reason"]
        if before.get("dimension", self.places.active_dim) != h.get("dim", "minecraft:overworld"):
            return "家在另一个维度，先通过已知传送门返程。"
        if not before or float(before.get("health", 20)) < 12 or float(before.get("food", 20)) < 8:
            return "返家路上先恢复生命和食物，不带伤继续挖楼梯。"
        key = tuple(h.get(k) for k in ("x", "y", "z", "dim"))
        record = self._home_attempts.setdefault(key, {"attempts": 0, "failed": 0, "until": 0})
        if record["until"] and record["until"] <= time.time():
            record.update(attempts=0, failed=0, until=0)
        record["attempts"] += 1
        distance_before = math.dist(tuple(before[k] for k in ("x", "y", "z")), tuple(h[k] for k in ("x", "y", "z")))
        out = self.go_to(h["x"], h["y"], h["z"])
        me = self._me() or {}
        if (me and math.hypot(me["x"] - h["x"], me["z"] - h["z"]) <= 8
                and abs(me["y"] - h["y"]) <= 2.5
                and me.get("dimension", self.places.active_dim) == h.get("dim", "minecraft:overworld")):
            self.memory.did("回到了家")
            self._home_attempts.pop(key, None)
            return f"回到家了。{self.store()}"
        distance_after = (math.dist(tuple(me[k] for k in ("x", "y", "z")), tuple(h[k] for k in ("x", "y", "z")))
                          if me else distance_before)
        record["failed"] = record["failed"] + 1 if distance_after >= distance_before - 2 else 0
        if record["failed"] >= 2 or record["attempts"] >= 3:
            record["until"] = time.time() + 600
        return f"还没回到家（需要同维度且高度差不超过2.5格）：{out}"

    def house_blueprint(self):
        """A compact timber cabin: stone base, corner posts, windows and a stepped gable roof."""
        home = self.places.home
        if not home:
            return None
        x, y, z = (math.floor(home[k]) for k in ("x", "y", "z"))
        cells = []
        for dy in range(3):
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    if max(abs(dx), abs(dz)) != 2 or (dx == 0 and dz == -2 and dy < 2):
                        continue
                    role = "base" if dy == 0 else "post" if abs(dx) == abs(dz) == 2 else "wall"
                    if dy == 1 and (dx, dz) in ((-2, 0), (2, 0), (-1, 2), (1, 2)):
                        role = "window"
                    cells.append({"pos": (x + dx, y + dy, z + dz), "role": role})
        # Solid stepped roof is placeable from the floor, with a one-block front/back eave.
        for dy, half_width in ((3, 2), (4, 1), (5, 0)):
            for dz in range(-3, 4):
                for dx in range(-half_width, half_width + 1):
                    cells.append({"pos": (x + dx, y + dy, z + dz), "role": "roof"})
        return {"origin": (x, y, z), "size": (5, 6, 7), "style": "stone foundation, timber frame, glazed windows, gable roof",
                "materials_required": {"minecraft:cobblestone": 15, "minecraft:oak_log": 8,
                                       "minecraft:oak_planks": 96, "minecraft:glass": 4, "minecraft:oak_door": 1,
                                       "minecraft:torch": 5, "minecraft:white_bed": 1},
                "cells": cells, "door": (x, y, z - 2), "windows": [c["pos"] for c in cells if c["role"] == "window"]}

    def house_requirements(self):
        """Observed missing structure/furniture, separate from boolean completion stages."""
        plan = self.house_blueprint()
        if not plan:
            return {"materials_needed": {}, "furniture_present": {}, "observed": False}
        x, y, z = plan["origin"]
        default = {"base": "minecraft:cobblestone", "post": "minecraft:oak_log",
                   "wall": "minecraft:oak_planks", "roof": "minecraft:oak_planks", "window": "minecraft:glass"}
        palette = self._hut_palette.get((self.places.active_dim, x, y, z), default)
        ids = self._blocks([c["pos"] for c in plan["cells"]])
        if not all(ids):
            return {"materials_needed": {}, "furniture_present": {}, "observed": False}
        needed = {}
        for cell, block in zip(plan["cells"], ids):
            role = cell["role"]
            exists = "glass" in block if role == "window" else self._shelter_block(block)
            if not exists:
                item = palette[role]
                needed[item] = needed.get(item, 0) + 1
        inside = self._blocks([(x+dx,y+dy,z+dz) for dx in range(-1,2) for dz in range(-1,2) for dy in range(2)])
        present = {kind: any(b.split(":")[-1].split("/")[-1] == kind for b in inside)
                   for kind in ("crafting_table", "furnace", "chest")}
        present["bed"] = any(b.endswith("_bed") for b in inside)
        for kind, exists in present.items():
            if not exists:
                needed["minecraft:white_bed" if kind == "bed" else "minecraft:"+kind] = 1
        if not self._block(*plan["door"]).endswith("_door"):
            needed["minecraft:oak_door"] = 1
        lights = [(x-1,y,z)] + [(x+dx,y,z+dz) for dx,dz in ((-3,0),(3,0),(0,-3),(0,3))]
        needed["minecraft:torch"] = sum(not any(v in b for v in ("torch", "lantern")) for b in self._blocks(lights))
        return {"materials_needed": {i:n for i,n in needed.items() if n}, "furniture_present": present, "observed": True}

    def _roof_scaffold(self, origin, deadline):
        """Build and climb a four-step exterior work stair using ordinary supported placements."""
        x, y, z = origin
        floors = [(x+4,y-1,z+dz) for dz in range(-5,0)]
        if not all(self._shelter_block(b) for b in self._blocks(floors)):
            return False, 0
        if not self._walk_to(x+4.5,y,z-4.5,reach=.4,timeout=15):
            return False, 0
        made = 0
        for level in range(1,5):
            column_z = z-5+level
            for dy in range(level):
                pos = (x+4,y+dy,column_z)
                block = self._block(*pos)
                if self._shelter_block(block):
                    continue
                if self._stop.is_set() or time.monotonic() >= deadline or not block.endswith(":air"):
                    return False, made
                if not self.place_at("minecraft:cobblestone",*pos,walk=False):
                    return False, made
                made += 1
            if not self._walk_to(x+4.5,y+level,column_z+.5,reach=.35,timeout=8):
                return False, made
        bridge = (x+3,y+3,z-1)
        if not self._shelter_block(self._block(*bridge)):
            if not self.place_at("minecraft:cobblestone",*bridge,walk=False):
                return False, made
            made += 1
        return True, made

    def _roof_vantage(self, target, origin):
        x,y,z = origin
        me = self._me() or {}
        if not me:
            return False
        candidates = [(x+4,y+4,z-1),(x+3,y+4,z-1)] + [(x+dx,y+4,z+dz) for dx in range(-2,3) for dz in range(-3,4)]
        candidates = sorted(candidates, key=lambda p: math.dist((me["x"],me["y"],me["z"]),(p[0]+.5,p[1],p[2]+.5)))
        for cx,cy,cz in candidates:
            if math.dist((cx+.5,cy+1.62,cz+.5),(target[0]+.5,target[1]+.5,target[2]+.5)) > 4:
                continue
            floor,feet,head = self._blocks([(cx,cy-1,cz),(cx,cy,cz),(cx,cy+1,cz)])
            if self._shelter_block(floor) and self._passable(feet) and self._passable(head):
                if not first_obstacle(self.rpc,(cx+.5,cy+1.62,cz+.5),target):
                    if self._walk_to(cx+.5,cy,cz+.5,reach=.35,timeout=8):
                        return True
        return False

    def _hut_cells(self):
        plan = self.house_blueprint()
        if not plan:
            return None
        x, y, z = plan["origin"]
        roof = [c["pos"] for c in plan["cells"] if c["role"] == "roof"]
        roof.sort(key=lambda p: (p[1], -p[0], abs(p[2] - (z - 1))))
        return plan["origin"], [c["pos"] for c in plan["cells"] if c["role"] != "roof"], roof, plan["door"]

    def build_portal(self) -> str:
        """Resume a survival 10-obsidian frame near home and verify portal blocks after ignition."""
        me = self._me() or {}
        if me.get("dimension") != "minecraft:overworld":
            return "建下界门先回主世界家附近，避免在未知维度搭错返程点。"
        home = self.places.home
        if not home:
            return "先安家并记住返程坐标，再搭下界门。"
        # Fixed origin makes a partial build recoverable after interruption/restart.
        x, y, z = math.floor(home["x"]) + 8, math.floor(home["y"]), math.floor(home["z"])
        frame = [(x + 1, y, z), (x + 2, y, z)]
        frame += [(x + side, y + dy, z) for dy in range(1, 4) for side in (0, 3)]
        frame += [(x + 1, y + 4, z), (x + 2, y + 4, z)]
        inside = [(x + dx, y + dy, z) for dx in (1, 2) for dy in (1, 2, 3)]
        interior = self._blocks(inside)
        if all("nether_portal" in b for b in interior):
            self.places.add("portal", x + 1, y + 1, z, note="已观测到下界门，往返后再次核查", merge=1)
            return "下界门已经存在，6格传送门内部均已观测确认。"
        # Use an above-ground base supported by existing terrain; never excavate an unknown site.
        if not self._walk_to(x + 1.5, y, z - 2.0, reach=1, timeout=35):
            return "没能走到家旁的建门位置，先解决路线。"
        foundation = self._blocks([(x + dx, y - 1, z) for dx in range(4)]
                                  + [(x + dx, y - 1, z - 1) for dx in range(4)])
        if not all(self._shelter_block(b) for b in foundation):
            return "建门位置的地面不平、有水或未知，保留现场，先选平整安全地面。"
        current = self._blocks(frame)
        inside_ids = self._blocks(inside)
        if any(b and not (b.endswith(":air") or "nether_portal" in b) for b in inside_ids):
            return "门内部被原有方块占用，未破坏原建筑，先清理或另选址。"
        if any(not b or (b != "minecraft:obsidian" and not b.endswith(":air")) for b in current):
            return "门框位置有原物品或未加载地形，未覆盖，先确认场地。"
        needed = sum(b != "minecraft:obsidian" for b in current)
        if self.crafter.counts().get("minecraft:obsidian", 0) < needed:
            return f"剩余门框还需要 {needed} 块黑曜石（已有部分不重建）。"
        # Corner scaffolding gives every survival placement a real adjacent support. It is ordinary
        # cobblestone, stays outside the portal interior and is never counted as obsidian progress.
        for target in ((x, y, z), (x + 3, y, z), (x, y + 4, z), (x + 3, y + 4, z)):
            if self._stop.is_set():
                return "搭门已暂停，现有门框保留，下次接着搭。"
            # Upper corners require side pillars first; defer them below.
            if target[1] != y:
                continue
            block = self._block(*target)
            if block.endswith(":air") and not self.place_at("minecraft:cobblestone", *target):
                return "缺角落支撑块或无法放置，现有门框保留。"
        for target in frame:
            if self._stop.is_set():
                return "搭门已暂停，现有门框保留，下次接着搭。"
            if self._block(*target) == "minecraft:obsidian":
                continue
            if target[1] == y + 4:
                for corner in ((x, y + 4, z), (x + 3, y + 4, z)):
                    if self._block(*corner).endswith(":air") and not self.place_at("minecraft:cobblestone", *corner):
                        return "门顶支撑还没搭好，下次从这里继续。"
            self.progress = "搭下界门，逐块确认门框"
            if not self.place_at("minecraft:obsidian", *target):
                return f"门框 {target} 还未放好，已搭的部分保留。"
        if not self.hold("minecraft:flint_and_steel"):
            return "门框已经搭好，缺打火石点火。"
        # A non-block item uses the clicked block face through the survival interaction RPC.
        self._smooth_look(x + 1.5, y + 1.0, z + 0.5)
        self.rpc("interact.placeBlock", {"x": x + 1, "y": y, "z": z, "face": "up"})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not self._stop.is_set():
            if all("nether_portal" in b for b in self._blocks(inside)):
                self.places.add("portal", x + 1, y + 1, z, note="主世界下界门，已验证6格门内方块", merge=1)
                return "下界门点亮成功，门内6格传送方块已确认；进入后还需核实实际维度。"
            time.sleep(0.2)
        return "门框已搭好，但没有确认点亮，不把它记成可用下界门。"

    def end_step(self) -> str:
        from src.levels.l8_agency.mc_endgame import Endgame
        return Endgame(self).step()

    def fill_water_bucket(self) -> str:
        from src.levels.l8_agency.mc_endgame import Endgame
        return Endgame(self).fill_water_bucket()

    def enter_portal(self) -> str:
        """Walk into a remembered, re-observed portal and wait for an actual dimension change."""
        me = self._me() or {}
        before = me.get("dimension")
        portals = self.places.near(me, ("portal",), within=128, n=3) if me else []
        if not portals:
            return "附近没有记住的传送门，先建门或勘察。"
        portal = next((p for p in portals if "nether_portal" in self._block(int(p["x"]), int(p["y"]), int(p["z"]))), None)
        if not portal:
            return "记忆中的传送门目前未观察到，不能盲走进旧坐标。"
        self.free_hands()
        if before == "minecraft:overworld":
            gear = self.rpc("player.getEquipment", {}) or {}
            if not any(str((gear.get(slot) or {}).get("id", "")).startswith("minecraft:golden_")
                       for slot in self.ARMOR_SLOTS.values()):
                gold = next((it["id"] for it in self.crafter.inventory()
                             if str(it["id"]).startswith("minecraft:golden_")
                             and str(it["id"]).endswith(tuple(self.ARMOR_SLOTS))), None)
                if not gold or not self.hold(gold):
                    return "进入下界前先准备并穿上一件金甲。"
                self.rpc("interact.useItem", {"hand": "main"})
                time.sleep(.3)
                gear = self.rpc("player.getEquipment", {}) or {}
                if not any(str((gear.get(slot) or {}).get("id", "")).startswith("minecraft:golden_")
                           for slot in self.ARMOR_SLOTS.values()):
                    return "还没确认金甲穿上，暂不进门。"
        self.progress = "走进传送门，等待并验证维度变化"
        if not self._walk_to(portal["x"] + .5, portal["y"], portal["z"] + .5, reach=.35, timeout=30):
            return "还没走进传送门。"
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and not self._stop.is_set():
            state = self.rpc("player.getState", {}) or {}
            after = state.get("dimension")
            if after and before and after != before:
                self.places.add("portal", state["x"], state["y"], state["z"], note="刚通过的返程传送门，需要再次观测", merge=2, dim=after)
                self.on_event(f"Minecraft维度已实际从 {before} 变成 {after}，记下返程点")
                return f"维度变化已验证：{before} → {after}；先观察周围和返程门再推进。"
            time.sleep(.25)
        return "还没有观察到维度变化，未把进入下界记成成功。"

    @staticmethod
    def _shelter_block(block: str) -> bool:
        """Only ordinary full blocks count as verified enclosure, never flowers/water/chests."""
        name = str(block).split(":")[-1]
        return name in ("stone", "cobblestone", "cobbled_deepslate", "deepslate", "dirt", "grass_block",
                        "andesite", "diorite", "granite", "bricks", "stone_bricks") or name.endswith(("_planks", "_log"))

    def hut_status(self) -> Dict[str, bool]:
        """Observe the home, not inventory counts. Unknown/unloaded blocks never prove a building exists."""
        shape = self._hut_cells()
        stages = {key: False for key in ("walls", "door", "roof", "light", "furniture", "safety", "windows")}
        if not shape:
            return stages
        (x, y, z), walls, roof, door = shape
        cells = walls + roof + [door, (door[0], y + 1, door[2])]
        ids = self._blocks(cells)
        window_positions = set(self.house_blueprint()["windows"])
        stages["walls"] = all(self._shelter_block(b) or p in window_positions and "glass" in b
                              for p, b in zip(walls, ids[:len(walls)]))
        stages["windows"] = all("glass" in b for b in self._blocks(list(window_positions)))
        stages["roof"] = all(self._shelter_block(b) for b in ids[len(walls):len(walls) + len(roof)])
        stages["door"] = all(b.endswith("_door") for b in ids[-2:])
        inside = [(x + dx, y + dy, z + dz) for dx in range(-1, 2) for dz in range(-1, 2) for dy in range(2)]
        inside_ids = self._blocks(inside)
        stages["light"] = any("torch" in b or "lantern" in b for b in inside_ids)
        short = [b.split(":")[-1].split("/")[-1] for b in inside_ids]
        stages["furniture"] = (all(kind in short for kind in ("crafting_table", "furnace", "chest"))
                                and any(b.endswith("_bed") for b in short))
        outside = self._blocks([(x + dx, y, z + dz) for dx, dz in ((-3, 0), (3, 0), (0, -3), (0, 3))])
        stages["safety"] = stages["door"] and all("torch" in b or "lantern" in b for b in outside)
        return stages

    def build_hut(self) -> str:
        """Build/resume a small survival shelter; never clear an unknown plot or report unbuilt parts done."""
        shape = self._hut_cells()
        if not shape:
            return "还没有家址，先 set_home。"
        (x, y, z), walls, roof, door = shape
        self.free_hands()
        if not self._walk_to(x + 0.5, y, z + 0.5, reach=0.8, timeout=30):
            return "还没走回家址，不能开始搭屋。"
        ground = self._blocks([(x + dx, y - 1, z + dz) for dx in range(-2, 3) for dz in range(-2, 3)])
        if len(ground) != 25 or not all(self._shelter_block(b) for b in ground):
            return "家址 5×5 地基不平、有水或未加载；需要另选平地，暂不往坑里盖。"
        made = 0
        began = time.monotonic()
        plan = self.house_blueprint()
        roles = {c["pos"]: c["role"] for c in plan["cells"]}
        inventory = self.crafter.counts()
        wood = next((i for i in ("minecraft:spruce_planks", "minecraft:oak_planks", "minecraft:birch_planks") if inventory.get(i)), "minecraft:oak_planks")
        log = wood.replace("_planks", "_log")
        choices_by_role = {"base": ("minecraft:cobblestone", "minecraft:cobbled_deepslate"),
                           "post": (log, "minecraft:oak_log", "minecraft:cobblestone"),
                           "wall": (wood, "minecraft:oak_planks", "minecraft:cobblestone"),
                           "roof": ("minecraft:spruce_planks", wood, "minecraft:cobbled_deepslate", "minecraft:cobblestone"),
                           "window": ("minecraft:glass", "minecraft:white_stained_glass")}
        palette_key = (self.places.active_dim, x, y, z)
        if palette_key not in self._hut_palette:
            # Resume the dominant existing material for each role instead of a patchwork roof.
            observed = self._blocks([c["pos"] for c in plan["cells"]])
            palette = {}
            for role, choices in choices_by_role.items():
                existing = [b for c, b in zip(plan["cells"], observed) if c["role"] == role and b in choices]
                palette[role] = max(set(existing), key=existing.count) if existing else next((i for i in choices if inventory.get(i)), choices[0])
            self._hut_palette[palette_key] = palette
        palette = self._hut_palette[palette_key]
        # Windows are not filled with rubble just because glass is currently missing.
        targets = [p for p in walls if roles[p] != "window"] + roof + plan["windows"]
        scaffold_ready = False
        for target in targets:
            if made >= 32 or time.monotonic() - began > 150:
                return f"本轮搭屋预算结束，实际放了 {made} 块；保留蓝图，下次从缺块处接着盖。"
            if self._stop.is_set():
                return f"盖屋已暂停，本次放了 {made} 块；下次从实物状态继续。"
            block = self._block(*target)
            role = roles[target]
            if role == "window" and self._shelter_block(block):
                glass = self.crafter.counts()
                if glass.get("minecraft:glass", 0) or glass.get("minecraft:white_stained_glass", 0):
                    if not self._break(*target):
                        continue
                    block = self._block(*target)
                else:
                    continue
            if self._shelter_block(block) or role == "window" and "glass" in block:
                continue
            if not block or not block.endswith(":air"):
                return f"家址 {target} 被 {block or '未加载方块'} 占用，保留原物品；屋子尚未盖完。"
            have = self.crafter.counts()
            material = palette[role] if have.get(palette[role], 0) else None
            if role == "window" and not material:
                continue
            if not material:
                return f"盖屋材料用完了，本次放了 {made} 块；{role} 还需要 {palette[role]}，保留同一配色续建。"
            self.progress = f"搭木构小屋的 {role}，本轮已放 {made} 块"
            if role == "roof":
                if not scaffold_ready:
                    scaffold_ready, placed = self._roof_scaffold(plan["origin"], began + 150)
                    made += placed
                    if not scaffold_ready:
                        return f"外侧屋顶施工台阶还未就绪，本次放了 {made} 块；需安全地面及圆石，未隔墙放屋顶。"
                if made >= 32 or time.monotonic() - began > 150:
                    return f"施工台阶就绪，本轮放块预算结束，实际放了 {made} 块；下次继续屋顶。"
                if not self._roof_vantage(target, plan["origin"]):
                    return f"屋顶 {target} 暂无安全可见施工位置，保留已建部分。"
            elif scaffold_ready:
                if not self._walk_to(x+.5,y,z+.5,reach=.5,timeout=20):
                    return "屋顶阶段结束，暂未回到室内，保留窗与家具待续。"
                scaffold_ready = False
            if not self.place_at(material, *target, walk=False):
                return f"没能放下 {target}，本次放了 {made} 块；屋子尚未盖完。"
            made += 1
        if (self._me() or {}).get("y", y) > y + 1 and not self._walk_to(x+.5,y,z+.5,reach=.5,timeout=25):
            return "屋顶完成后尚未回到室内，门窗家具下轮继续。"
        have = self.crafter.counts()
        if not self._block(*door).endswith("_door"):
            door_item = next((i for i, n in have.items() if n > 0 and i.endswith("_door") and "iron" not in i), None)
            if door_item is None:
                self.crafter.craft("minecraft:oak_door", 1)
                door_item = "minecraft:oak_door" if self.crafter.counts().get("minecraft:oak_door", 0) else None
            if door_item:
                self.place_at(door_item, *door, walk=False)
        # Furniture/lighting remain real placements. A missing recipe/resource is reported, never inferred.
        for item in ("minecraft:torch", "minecraft:crafting_table", "minecraft:furnace", "minecraft:chest"):
            inside = self._blocks([(x + dx, y + dy, z + dz) for dx in range(-1, 2)
                                   for dz in range(-1, 2) for dy in range(2)])
            if any(b.split(":")[-1].split("/")[-1].endswith(item.split(":")[-1]) for b in inside):
                continue
            if not self.crafter.counts().get(item, 0):
                self.crafter.craft(item, 1)
            if self.crafter.counts().get(item, 0):
                dx, dz = {"minecraft:torch": (-1, 0), "minecraft:crafting_table": (1, -1),
                          "minecraft:furnace": (-1, 1), "minecraft:chest": (-1, -1)}[item]
                self.place_at(item, x + dx, y, z + dz, walk=False)
        have = self.crafter.counts()
        bed = next((i for i, n in have.items() if n > 0 and i.endswith("_bed")), None)
        if bed and all(self._block(x + 1, y, z + dz).endswith(":air") for dz in (0, 1)):
            # Stand directly in front so the bed's head occupies the second reserved interior cell.
            if self._walk_to(x + 1.5, y, z - .5, reach=.25, timeout=8):
                self.place_at(bed, x + 1, y, z, walk=False)
        for dx, dz in ((-3, 0), (3, 0), (0, -3), (0, 3)):
            if "torch" not in self._block(x + dx, y, z + dz) and self.crafter.counts().get("minecraft:torch", 0):
                self.place_at("minecraft:torch", x + dx, y, z + dz)
        status = self.hut_status()
        labels = {"walls": "墙", "roof": "屋顶", "door": "门", "light": "室内照明", "furniture": "家具和床", "safety": "门外照明", "windows": "玻璃窗"}
        missing = [labels[k] for k, verified in status.items() if not verified]
        if missing:
            return f"本次盖屋放了 {made} 块，实物检查还缺：{'、'.join(missing)}。"
        self.memory.did(f"在 ({x},{y},{z}) 盖好了有门、屋顶和照明的起步屋")
        return "起步屋实物检查通过：墙、屋顶、门、照明、工作台、熔炉、箱子和床都在。"

    # what stays on her when she empties her bag into a chest
    CARRY = {"minecraft:torch": 64, "minecraft:coal": 16, "minecraft:cobblestone": 64, "minecraft:stick": 16,
             "minecraft:crafting_table": 1, "minecraft:furnace": 0, "minecraft:bread": 64}

    def _keep(self, item: str) -> int:
        from src.levels.l8_agency.mc_goal import is_gear, is_log
        name = item.split(":")[-1]
        if is_gear(item) or self._edible(item):
            return 10 ** 6
        if is_log(item) or name.endswith("_planks"):
            return 16
        return self.CARRY.get(item, 0)

    def _open_chest(self, p: Dict[str, Any]) -> bool:
        if not self._approach(p["x"], p["y"], p["z"]):
            return False
        self._smooth_look(p["x"] + 0.5, p["y"] + 0.5, p["z"] + 0.5, 0.3)
        self.rpc("container.open", {"x": p["x"], "y": p["y"], "z": p["z"]})
        end = time.time() + 3
        while time.time() < end:
            st = self.rpc("container.state", {}) or {}
            if st.get("open") and any(w in str(st.get("type", "")) for w in ("generic", "chest", "barrel", "shulker")):
                return True
            time.sleep(0.2)
        return False

    def _read_chest(self, p: Dict[str, Any]):
        state = self.rpc("container.state", {})
        if not isinstance(state, dict) or state.get("open") is False or "slots" not in state:
            return                           # a failed read is not proof that the chest is empty
        items: Dict[str, int] = {}
        # ContainerHandlers.state excludes all player-inventory slots at the bridge boundary.
        for sl in state.get("slots") or []:
            if sl.get("id") and int(sl.get("count", 0)) > 0:
                items[sl["id"]] = items.get(sl["id"], 0) + int(sl.get("count", 0))
        self.places.add("chest", p["x"], p["y"], p["z"], items=items, merge=0.5)

    def _chest_transfer(self, item: str, count: int, take: bool = False) -> int:
        """Shift one whole stack or right-click a bounded partial stack, preserving reserved supplies."""
        before = self.crafter.counts().get(item, 0)
        remaining = max(0, min(256, int(count)))
        for _ in range(12):
            if remaining <= 0 or self._stop.is_set():
                break
            state = self.rpc("container.state", {}) or {}
            size = int(state.get("size") or 0)
            if not state.get("open") or size not in (27, 54):
                break  # unknown modded slot layout is not safe to guess
            inv = self.crafter.inventory()
            chest = state.get("slots") or []
            sources = [v for v in (chest if take else inv) if v.get("id") == item and v.get("count", 0) > 0]
            if not sources:
                break
            src = min(sources, key=lambda v: int(v.get("count", 0)))
            inv_menu = lambda slot: size + slot - 9 if slot >= 9 else size + 27 + slot
            source = int(src["slot"]) if take else inv_menu(int(src["slot"]))
            previous = self.crafter.counts().get(item, 0)
            if int(src["count"]) <= remaining:
                self.crafter._click(source, 0, "quick_move")
            else:
                destinations = inv if take else chest
                capacity = 36 if take else size
                occupied = {int(v["slot"]): v for v in destinations}
                dest = next((slot for slot in range(capacity) if slot not in occupied), None)
                if dest is None:
                    dest = next((slot for slot, v in occupied.items() if v.get("id") == item and v.get("count", 0) < 64), None)
                if dest is None:
                    break
                amount = min(remaining, 64 - int(occupied.get(dest, {}).get("count", 0)))
                target = inv_menu(dest) if take else dest
                self.crafter._click(source)
                try:
                    for _ in range(amount):
                        self.crafter._click(target, 1)
                finally:
                    self.crafter._click(source)  # return every untransferred item to its original slot
            after = self.crafter.counts().get(item, 0)
            actual = max(0, after - previous if take else previous - after)
            if not actual:
                break
            remaining -= actual
        after = self.crafter.counts().get(item, 0)
        return max(0, after - before if take else before - after)

    def store(self) -> str:
        """Put what she does not need to carry into her nearest chest, and remember what is in there."""
        me = self._me()
        chest = self.places.near(me, ("chest",), within=24, n=1) if me else []
        if not chest:
            home = self.places.home
            if not me or not home or math.dist(tuple(me[k] for k in ("x", "y", "z")),
                                               tuple(home[k] for k in ("x", "y", "z"))) > 8:
                return "附近没有我的箱子，尚未到可补建箱子的家址。"
            if not self.crafter.counts().get("minecraft:chest", 0):
                self.crafter.craft("minecraft:chest", 1)
            if not self.crafter.counts().get("minecraft:chest", 0) or not self.place_near("minecraft:chest"):
                return "家里的箱子缺失，补建还没完成；需要木板和可放置的地面。"
            chest = self.places.near(me, ("chest",), within=24, n=1)
            if not chest:
                return "放置后还未观测到家里的箱子，稍后核查。"
        p = chest[0]
        self._storing = True
        self.using_screen = True
        try:
            if not self._open_chest(p):
                self.places.forget_at(p["x"], p["y"], p["z"], "chest")      # it is gone (or not hers)
                return "那个箱子打不开了。"
            put = []
            for item, n in sorted(self.crafter.counts().items(), key=lambda kv: -kv[1]):
                extra = n - self._keep(item)
                if extra > 0:
                    moved = self._chest_transfer(item, extra)
                    if moved:
                        put.append(f"{item.split(':')[-1]}×{moved}")
            time.sleep(0.3)
            self._read_chest(p)
        finally:
            self.using_screen = False
            self._storing = False
            self.rpc("container.close", {})
        if put:
            self.memory.did("往箱子里放了 " + "、".join(put[:6]))
        return ("放进箱子：" + "、".join(put[:8]) + "。") if put else "没什么要放的。"

    def _take_from_chest(self, item: Optional[str], count: int) -> bool:
        """If she stored this somewhere not too far, go and get it instead of digging for it again."""
        if not item:
            return False
        me = self._me()
        if not me:
            return False
        for p in self.places.where_is(item):
            if math.dist((p["x"], p["y"], p["z"]), (me["x"], me["y"], me["z"])) > 96:
                continue
            self.progress = f"去箱子里拿 {item.split(':')[-1]}"
            if math.dist((p["x"], p["z"]), (me["x"], me["z"])) > 6:
                self.go_to(p["x"], p["y"], p["z"])
            self.using_screen = True
            try:
                if not self._open_chest(p):
                    continue
                before = self.crafter.counts().get(item, 0)
                self._chest_transfer(item, int(count), take=True)
                time.sleep(0.3)
                self._read_chest(p)
            finally:
                self.using_screen = False
                self.rpc("container.close", {})
            got = self.crafter.counts().get(item, 0) - before
            if got > 0:
                self.memory.did(f"从箱子里拿了 {item.split(':')[-1]}×{got}")
                return True
        return False
