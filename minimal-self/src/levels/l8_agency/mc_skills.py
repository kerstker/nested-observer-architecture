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


class Skills:
    def __init__(self, rpc: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]], who: Callable[[str], Optional[Dict]] = None):
        self.rpc = rpc                      # method, params -> result dict or None
        self.find_player = who              # name -> entity dict (x, y, z, distance) or None
        self.current: Optional[str] = None
        self.progress = ""
        self.last_result = ""
        self._stop = threading.Event()
        self.failed: Dict[tuple, float] = {}      # trees she could not get to lately: try another one
        self._thread: Optional[threading.Thread] = None

    # --- running ------------------------------------------------------------------------

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, name: str, **args) -> str:
        fn = {"chop_tree": self.chop_tree, "collect_drops": self.collect_drops, "go_to_player": self.go_to_player}.get(name)
        if fn is None:
            return f"不会 {name}。会的有：chop_tree（砍一棵树）、collect_drops（捡掉落物）、go_to_player（走到某人身边）。"
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

    def _break(self, x: int, y: int, z: int, timeout: float = 8.0) -> bool:
        before = self._block(x, y, z)
        if not before or before.endswith(":air"):
            return False
        self.rpc("control.lookAt", {"x": x + 0.5, "y": y + 0.5, "z": z + 0.5})
        if self.rpc("interact.breakBlock", {"x": x, "y": y, "z": z, "mode": "survival"}) is None:
            return False
        end = time.time() + timeout
        while time.time() < end and not self._stop.is_set():
            time.sleep(0.3)
            if self._block(x, y, z) != before:
                return True
        self.rpc("interact.stopBreaking", {})
        return False

    PASSABLE = ("air", "grass", "fern", "flower", "leaf_litter", "snow", "bush", "sapling", "vine", "petals", "moss_carpet")

    def _passable(self, block_id: str) -> bool:
        name = block_id.split(":")[-1]
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
        (x, z), ys = min(columns.items(), key=lambda kv: (kv[0][0] - me["x"]) ** 2 + (kv[0][1] - me["z"]) ** 2)
        base = min(ys)
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
        if self._stop.is_set():
            return f"砍了 {logs} 块就停了。"
        picked = self.collect_drops(radius=8)
        return f"砍了 {logs} 块木头。{picked}"

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
            self.rpc("control.lookAt", {"x": them["x"], "y": them["y"] + 1.6, "z": them["z"]})
        return f"到 {player} 身边了。" if ok else f"没走到 {player} 那里。"
