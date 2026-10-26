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
from typing import Any, Callable, Dict, List, Optional

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


class Skills:
    def __init__(self, rpc: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]], who: Callable[[str], Optional[Dict]] = None,
                 patterns=None):
        self.rpc = rpc                      # method, params -> result dict or None
        self.find_player = who              # name -> entity dict (x, y, z, distance) or None
        from src.levels.l8_agency.mc_craft import Crafter, PatternIndex
        self.crafter = Crafter(rpc, self, patterns or PatternIndex(None), stop=lambda: self._stop.is_set())
        self.current: Optional[str] = None
        self.progress = ""
        self.last_result = ""
        self._stop = threading.Event()
        self.failed: Dict[tuple, float] = {}      # trees she could not get to lately: try another one
        self.last_blocked: Optional[tuple] = None
        self._thread: Optional[threading.Thread] = None

    # --- running ------------------------------------------------------------------------

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, name: str, **args) -> str:
        fn = {"chop_tree": self.chop_tree, "collect_drops": self.collect_drops, "go_to_player": self.go_to_player,
              "mine": self.mine, "craft": self.craft, "place": self.place, "equip": self.equip, "tidy": self.tidy}.get(name)
        if fn is None:
            return (f"不会 {name}。会的有：chop_tree（砍一棵树）、mine（挖某种方块）、craft（合成）、place（放方块）、"
                    "equip（拿起合适的工具）、tidy（整理快捷栏）、collect_drops（捡掉落物）、go_to_player（走到某人身边）。")
        if self.busy() and name == self.current:
            return f"已经在做 {name} 了（{self.progress}），不用再开始一次。"
        if self.busy():
            self.stop()
            self._thread.join(timeout=3)
        self._stop.clear()
        self.current, self.progress = name, "开始了"

        def run():
            try:
                self.last_result = fn(**args)
            except Exception as e:                      # a skill never takes the body down with it
                self.last_result = f"{name} 出错了：{e}"
            finally:
                logger.info(f"[Skill] {name}: {self.last_result}")
                self.current = None

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return f"开始做 {name} 了（在后台进行，可以先说话；用 mc_skill status 看进度）。"

    def stop(self) -> str:
        self._stop.set()
        self.rpc("nav.stop", {})
        self.rpc("interact.stopBreaking", {})
        return "停下了。"

    def status(self) -> str:
        if self.busy():
            return f"正在 {self.current}：{self.progress}"
        return f"现在没在做什么。上一件：{self.last_result or '没有'}"

    # --- small body actions ---------------------------------------------------------------

    def _me(self) -> Optional[Dict[str, Any]]:
        return self.rpc("player.getState", {})

    def _walk_to(self, x: float, y: float, z: float, reach: float = 2.0, timeout: float = 25.0) -> bool:
        if self.rpc("nav.pathTo", {"x": x, "y": y, "z": z, "reachRadius": reach, "timeoutSeconds": int(timeout)}) is None:
            return False
        end = time.time() + timeout
        while time.time() < end and not self._stop.is_set():
            time.sleep(0.4)
            st = self.rpc("nav.status", {}) or {}
            if not st.get("active"):
                return st.get("state") == "reached" or (st.get("distance") or 99) <= reach + 0.5
        return False

    def _block(self, x: int, y: int, z: int) -> str:
        got = (self.rpc("perception.blocks", {"positions": [{"x": x, "y": y, "z": z}]}) or {}).get("blocks") or [{}]
        return str(got[0].get("id") or "")

    def _smooth_look(self, x: float, y: float, z: float, seconds: float = 0.25) -> None:
        """Turn the camera like a hand moves a mouse: a big turn takes longer than a small one, it speeds up and
        slows down (ease in and out), and it is drawn in small steps (~50 a second), never a snap."""
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
        before = self._block(x, y, z)
        if not before or before.endswith(":air"):
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
        self._smooth_look(*aim, 0.3)
        if self.rpc("interact.breakBlock", {"x": x, "y": y, "z": z, "mode": "survival"}) is None:
            return False
        end = time.time() + timeout
        while time.time() < end and not self._stop.is_set():
            time.sleep(0.3)
            if self._block(x, y, z) != before:
                return True
            self._smooth_look(*aim, 0.1)                           # keep the eyes on it while the arm works
        self.rpc("interact.stopBreaking", {})
        return False

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
                if self._walk_to(nx, y, nz, reach=1.0):
                    return True
            if self._stop.is_set():
                return False
        return False

    @staticmethod
    def _dist(me: Dict[str, Any], x: float, y: float, z: float) -> float:
        return math.sqrt((me["x"] - x - 0.5) ** 2 + (me["y"] + 1.6 - y - 0.5) ** 2 + (me["z"] - z - 0.5) ** 2)

    # --- skills ---------------------------------------------------------------------------

    def chop_tree(self, max_logs: int = 10, radius: int = 2) -> str:
        """Find the nearest tree trunk, walk to it, break the logs from the bottom up, then pick up what fell.
        `radius` is in chunks (16 blocks), as the client scan counts it."""
        scan = self.rpc("perception.scan", {"radius": radius}) or {}
        kinds = sorted({k for c in scan.get("chunks") or [] for k in (c.get("counts") or {})
                        if k.endswith(LOG_SUFFIX) and "stripped" not in k})
        if not kinds:
            return "附近没有树。"
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
        picked = self.collect_drops(radius=8)
        return f"砍了 {logs} 块木头。{picked}"

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
            if self._stop.is_set() or got >= count or tried >= count * 3:
                break
            tried += 1
            if abs(f["y"] - me["y"]) > 6:
                continue
            self.progress = f"去挖 {f['id'].split(':')[-1]}（已挖 {got}）"
            if not self._approach(f["x"], f["y"], f["z"]):
                continue
            if self._break(f["x"], f["y"], f["z"]):
                got += 1
            me = self._me() or me
        picked = self.collect_drops(radius=6) if got else ""
        return f"挖了 {got} 块。{picked}" if got else "看得见又够得着的一块都没有（要么被挡住，要么太远）。"

    def collect_drops(self, radius: float = 10) -> str:
        """Walk over the items lying around nearby so they end up in the backpack."""
        got = 0
        for _ in range(8):
            if self._stop.is_set():
                break
            items = (self.rpc("perception.entities", {"radius": radius, "kinds": ["item"]}) or {}).get("entities") or []
            if not items:
                break
            item = min(items, key=lambda e: e.get("distance", 99))
            self.progress = f"去捡地上的东西（还有 {len(items)} 个）"
            if self._walk_to(item["x"], item["y"], item["z"], reach=0.6, timeout=10):
                got += 1
            time.sleep(0.4)
        return f"捡了 {got} 次东西。" if got else "地上没什么可捡的。"

    # --- hands and bag ---------------------------------------------------------------------

    TOOL_KINDS = {"pickaxe": "_pickaxe", "axe": "_axe", "shovel": "_shovel", "sword": "_sword", "hoe": "_hoe"}
    MATERIAL_RANK = ("wooden", "golden", "stone", "copper", "iron", "diamond", "netherite")

    def _inventory(self) -> Dict[str, Any]:
        return self.rpc("player.getInventory", {}) or {}

    def hold(self, item: str) -> bool:
        """Take an item in hand: pick its hotbar slot, or swap it there from the bag."""
        inv = self._inventory()
        for it in inv.get("hotbar") or []:
            if it.get("id") == item:
                self.rpc("inventory.selectHotbar", {"slot": int(it["slot"])})
                return True
        main = next((it for it in inv.get("main") or [] if it.get("id") == item), None)
        if not main:
            return False
        sel = int(inv.get("selectedSlot") or 0)
        self.rpc("inventory.swapSlots", {"slotA": int(main["slot"]), "slotB": sel})
        self.rpc("inventory.selectHotbar", {"slot": sel})
        return True

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
                self.rpc("inventory.swapSlots", {"slotA": int(src["slot"]), "slotB": slot})
                moved.append(tool.split(":")[-1])
        return ("整理好了：" + "、".join(moved) + " 放到了快捷栏前几格。") if moved else "快捷栏已经挺顺手了。"

    def place(self, item: str = "", x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> str:
        if x is None or y is None or z is None:
            pos = self.place_near(item)
            return f"放好了 {item.split(':')[-1]}（{pos[0]},{pos[1]},{pos[2]}）。" if pos else f"没地方放 {item.split(':')[-1]}。"
        pos = self.place_at(item, int(x), int(y), int(z))
        return f"放好了（{x},{y},{z}）。" if pos else "没放上去（够不着、没有依附面，或者那里有东西）。"

    def place_near(self, item: str) -> Optional[tuple]:
        """Put a block from the bag on free ground right next to her, the way a player drops a table down."""
        me = self._me()
        if not me or not item:
            return None
        fx, fy, fz = math.floor(me["x"]), math.floor(me["y"]), math.floor(me["z"])
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1), (2, 0), (0, 2), (-2, 0), (0, -2)):
            pos = self.place_at(item, fx + dx, fy, fz + dz, walk=False)
            if pos:
                return pos
        return None

    def place_at(self, item: str, x: int, y: int, z: int, walk: bool = True) -> Optional[tuple]:
        got = (self.rpc("perception.blocks", {"positions": [{"x": x, "y": y, "z": z}, {"x": x, "y": y - 1, "z": z}]}) or {}).get("blocks") or []
        if len(got) < 2:
            return None
        at, below = (str(b.get("id") or "") for b in got)
        if not self._passable(at) or self._passable(below) or "water" in below or "lava" in below:
            return None
        if walk and not self._approach(x, y - 1, z):
            return None
        if not self.hold(item):
            return None
        self._smooth_look(x + 0.5, y - 0.01, z + 0.5, 0.25)          # the top face of the block she builds on
        self.rpc("interact.placeBlock", {"x": x, "y": y - 1, "z": z, "face": "up"})
        deadline = time.time() + 2
        while time.time() < deadline:
            time.sleep(0.2)
            # mods swap some blocks once placed (Visual Workbench: "visualworkbench:minecraft/crafting_table")
            if self._block(x, y, z).split(":")[-1].split("/")[-1] == item.split(":")[-1]:
                return (x, y, z)
        return None

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
