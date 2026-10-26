"""
Level 8j: crafting the way a player does it - by hand, slot by slot.

The mod's recipe-book placement only works for recipes the server has unlocked for her, and in this pack almost
none are (logs in the bag do not unlock planks). A player does not need the recipe book: they put the items
into the grid themselves. So this does exactly that, through the same container clicks a hand makes:

  pick up a stack -> right-click one into each cell that needs it -> put the rest back -> take the result.

Where things go in the grid comes from the recipe files in her own game (vanilla jar + every mod jar, read once
and cached): the bridge reports which items fit a recipe but not where, and for a pickaxe that matters.
Recipes up to 2x2 are made in her own inventory grid; bigger ones at a crafting table - walked to if one is
near, otherwise placed from the bag, otherwise crafted first.
"""
import glob
import json
import logging
import os
import time
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TABLE = "minecraft:crafting_table"


class PatternIndex:
    """Shaped-recipe patterns by recipe id ("minecraft:wooden_pickaxe" -> ["XXX", " # ", " # "])."""

    def __init__(self, game_dir: Optional[str], cache: Optional[str] = None):
        self.game_dir = game_dir
        self.cache = cache
        self.patterns: Dict[str, List[str]] = {}
        self.loaded = False

    def load(self) -> "PatternIndex":
        if self.loaded:
            return self
        self.loaded = True
        if self.cache and os.path.exists(self.cache):
            try:
                with open(self.cache, encoding="utf-8") as f:
                    self.patterns = json.load(f)
                return self
            except (OSError, ValueError):
                pass
        if not self.game_dir or not os.path.isdir(self.game_dir):
            return self
        root = glob.escape(self.game_dir)                  # "[BMC5]" in a folder name is not a glob pattern
        jars = glob.glob(os.path.join(root, "*.jar")) + glob.glob(os.path.join(root, "mods", "*.jar"))
        for jar in jars:
            try:
                with zipfile.ZipFile(jar) as z:
                    for name in z.namelist():
                        parts = name.split("/")
                        # data/<ns>/recipe(s)/<path>.json
                        if len(parts) < 4 or parts[0] != "data" or parts[2] not in ("recipe", "recipes") or not name.endswith(".json"):
                            continue
                        try:
                            doc = json.loads(z.read(name).decode("utf-8"))
                        except (ValueError, UnicodeDecodeError, KeyError):
                            continue
                        if isinstance(doc, dict) and "pattern" in doc and str(doc.get("type", "")).endswith("crafting_shaped"):
                            rid = f"{parts[1]}:{'/'.join(parts[3:])[:-5]}"
                            self.patterns[rid] = [str(row) for row in doc["pattern"]]
            except (zipfile.BadZipFile, OSError):
                continue
        if self.cache:
            try:
                os.makedirs(os.path.dirname(self.cache), exist_ok=True)
                with open(self.cache, "w", encoding="utf-8") as f:
                    json.dump(self.patterns, f)
            except OSError:
                pass
        logger.info(f"[Craft] {len(self.patterns)} shaped recipe patterns from {len(jars)} jars")
        return self

    def get(self, ref: str) -> Optional[List[str]]:
        rid = ref.split("recipe:", 1)[-1]
        return self.load().patterns.get(rid)


class Crafter:
    def __init__(self, rpc: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]], skills, patterns: PatternIndex,
                 stop: Callable[[], bool] = lambda: False):
        self.rpc = rpc
        self.skills = skills            # for walking to / placing a table
        self.patterns = patterns
        self.stopped = stop

    # --- what she has ------------------------------------------------------------------

    def inventory(self) -> List[Dict[str, Any]]:
        inv = self.rpc("player.getInventory", {}) or {}
        return [dict(it, where="hotbar") for it in inv.get("hotbar") or []] + [dict(it, where="main") for it in inv.get("main") or []]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for it in self.inventory():
            out[it["id"]] = out.get(it["id"], 0) + int(it.get("count", 0))
        return out

    # --- recipes -----------------------------------------------------------------------

    def recipes(self, item: str) -> List[Dict[str, Any]]:
        got = (self.rpc("recipes.query", {"items": [item]}) or {}).get("recipes") or []
        out = [r for r in got if r.get("type") == "crafting" and (r.get("result") or {}).get("id") == item]
        # vanilla recipes first: they are the ones the game's quests and players expect
        out.sort(key=lambda r: (not str(r.get("ref", "")).startswith("recipe:minecraft:"),))
        return out

    def layout(self, recipe: Dict[str, Any]) -> Optional[Tuple[List[Tuple[int, int, List[str]]], int, int]]:
        """[(row, col, alternatives)], width, height - or None when the shape cannot be known."""
        ingredients = [list(i or []) for i in recipe.get("ingredients") or []]
        w, h = recipe.get("width"), recipe.get("height")
        if not w or not h:                                  # shapeless: any cells will do
            n = len(ingredients)
            if n == 0 or n > 9:
                return None
            side = 2 if n <= 4 else 3
            return [(i // side, i % side, alts) for i, alts in enumerate(ingredients)], min(n, side), (n + side - 1) // side
        cells: List[Tuple[int, int, List[str]]] = []
        if len(ingredients) == w * h:                       # every cell reported, empty ones as []
            for i, alts in enumerate(ingredients):
                if alts:
                    cells.append((i // w, i % w, alts))
            return cells, w, h
        pattern = self.patterns.get(str(recipe.get("ref", "")))
        if not pattern:
            return None
        it = iter(ingredients)
        for r, row in enumerate(pattern):
            for c, ch in enumerate(row):
                if ch != " ":
                    alts = next(it, None)
                    if alts is None:
                        return None
                    cells.append((r, c, alts))
        return cells, max(len(row) for row in pattern), len(pattern)

    def choose(self, cells, counts: Dict[str, int]) -> Optional[Dict[Tuple[int, int], str]]:
        """Which item goes in each cell, from what she has (None when something is missing)."""
        left = dict(counts)
        pick: Dict[Tuple[int, int], str] = {}
        for r, c, alts in cells:
            have = [a for a in alts if left.get(a, 0) > 0]
            if not have:
                return None
            best = max(have, key=lambda a: left[a])
            left[best] -= 1
            pick[(r, c)] = best
        return pick

    # --- the grid ------------------------------------------------------------------------

    def _menu(self) -> Dict[str, Any]:
        return self.rpc("container.state", {}) or {}

    @staticmethod
    def menu_slot(inv_slot: int, table: bool) -> int:
        """Inventory slot (hotbar 0-8, main 9-35) -> slot index in the open menu."""
        if table:                                           # crafting table: result 0, grid 1-9, then the player
            return inv_slot + 1 if inv_slot >= 9 else 37 + inv_slot
        return inv_slot if inv_slot >= 9 else 36 + inv_slot  # her own inventory screen

    def _click(self, slot: int, button: int = 0, mode: str = "pickup"):
        self.rpc("container.click", {"slot": slot, "button": button, "mode": mode})
        time.sleep(0.05)

    def clear_grid(self, table: bool):
        for s in self._menu().get("slots") or []:
            if 1 <= int(s["slot"]) <= (9 if table else 4):
                self._click(int(s["slot"]), 0, "quick_move")

    def craft_once(self, recipe: Dict[str, Any], table: bool) -> int:
        """Lay the recipe out by hand and take the result. Returns how many came out (0 = failed)."""
        lay = self.layout(recipe)
        if not lay:
            return 0
        cells, w, h = lay
        side = 3 if table else 2
        if w > side or h > side:
            return 0
        pick = self.choose(cells, self.counts())
        if pick is None:
            return 0
        result = recipe["result"]["id"]
        before = self.counts().get(result, 0)
        self.clear_grid(table)
        by_item: Dict[str, List[int]] = {}
        for (r, c), item in pick.items():
            by_item.setdefault(item, []).append(1 + r * side + c)
        for item, targets in by_item.items():
            sources = [it for it in self.inventory() if it["id"] == item]
            while targets and sources:
                src = sources.pop(0)
                slot = self.menu_slot(int(src["slot"]), table)
                self._click(slot, 0)                       # pick the stack up
                n = int(src.get("count", 0))
                while targets and n > 0:
                    self._click(targets.pop(0), 1)         # right-click: one into this cell
                    n -= 1
                self._click(slot, 0)                       # the rest goes back where it was
            if targets:
                self.clear_grid(table)
                return 0
        deadline = time.time() + 2.5
        while time.time() < deadline:
            slots = {int(s["slot"]): s for s in self._menu().get("slots") or []}
            if slots.get(0, {}).get("id") == result:
                self._click(0, 0, "quick_move")            # shift-click the result into the bag
                break
            time.sleep(0.15)
        time.sleep(0.2)
        self.clear_grid(table)
        return max(0, self.counts().get(result, 0) - before)

    # --- a crafting table ----------------------------------------------------------------

    def open_table(self, depth: int) -> bool:
        st = self._menu()
        if st.get("open") and "crafting" in str(st.get("type", "")):
            return True
        if st.get("open"):
            self.rpc("container.close", {})
        # Mods replace the block once it is placed (Visual Workbench: "visualworkbench:minecraft/crafting_table"),
        # so tables are recognised by name among the scan's points of interest, whatever their namespace
        scan = self.rpc("perception.scan", {"radius": 1}) or {}
        me = self.rpc("player.getState", {}) or {}
        found = [p for p in scan.get("pois") or [] if "crafting_table" in str(p.get("id", ""))]
        tables = sorted(found, key=lambda t: (t["x"] - me.get("x", 0)) ** 2 + (t["y"] - me.get("y", 0)) ** 2
                        + (t["z"] - me.get("z", 0)) ** 2)
        table = next((t for t in tables if abs(t["y"] - me.get("y", 0)) <= 4
                      and (t["x"] - me.get("x", 0)) ** 2 + (t["z"] - me.get("z", 0)) ** 2 <= 16 ** 2), None)
        if table is None:
            if self.counts().get(TABLE, 0) == 0 and not self.craft(TABLE, 1, depth + 1).startswith("做好了"):
                return False
            pos = self.skills.place_near(TABLE)
            if not pos:
                return False
            table = {"x": pos[0], "y": pos[1], "z": pos[2]}
        if not self.skills._approach(table["x"], table["y"], table["z"]):
            return False
        self.skills._smooth_look(table["x"] + 0.5, table["y"] + 0.5, table["z"] + 0.5, 0.3)
        self.rpc("container.open", {"x": table["x"], "y": table["y"], "z": table["z"]})
        deadline = time.time() + 3
        while time.time() < deadline:
            st = self._menu()
            if st.get("open") and "crafting" in str(st.get("type", "")):
                return True
            time.sleep(0.2)
        return False

    # --- the whole thing --------------------------------------------------------------------

    def craft(self, item: str, count: int = 1, depth: int = 0) -> str:
        item = item if ":" in item else f"minecraft:{item}"
        if depth > 3:
            return f"做 {item} 要绕的步骤太多了。"
        recipes = self.recipes(item)
        if not recipes:
            return f"不知道 {item} 怎么合成（没查到工作台配方）。"
        made = 0
        tried_sub: Dict[str, int] = {}             # (making planks twice is normal: sticks eat some)
        while made < count and not self.stopped():
            counts = self.counts()
            ready = [r for r in recipes if self.layout(r) and self.choose(self.layout(r)[0], counts) is not None]
            if not ready:
                # something is missing: make it from what she has (planks from logs, sticks from planks...)
                lay = next((self.layout(r) for r in recipes if self.layout(r)), None)
                if not lay:
                    return f"{item} 的摆法查不到。"
                # what is short, counted (3 planks needed and 2 in the bag is short too)
                groups: Dict[tuple, int] = {}
                for _, _, alts in lay[0]:
                    groups[tuple(alts)] = groups.get(tuple(alts), 0) + 1
                missing = [(list(alts), need - sum(counts.get(a, 0) for a in alts)) for alts, need in groups.items()
                           if sum(counts.get(a, 0) for a in alts) < need]
                progressed = False
                for alts, short in missing:
                    for alt in alts[:6]:
                        if tried_sub.get(alt, 0) >= 3:
                            continue
                        tried_sub[alt] = tried_sub.get(alt, 0) + 1
                        if self.craft(alt, short, depth + 1).startswith("做好了"):
                            progressed = True
                            break
                    if progressed:
                        break
                if not progressed:
                    names = "、".join(f"{a[0].split(':')[-1]}×{n}" for a, n in missing)
                    return f"做 {item.split(':')[-1]} 还缺：{names}。"
                continue
            recipe = ready[0]
            cells, w, h = self.layout(recipe)
            table = w > 2 or h > 2
            if table and not self.open_table(depth):
                return "需要工作台，但没能用上（附近没有、也放不下）。"
            got = self.craft_once(recipe, table)
            if table:
                self.rpc("container.close", {})
            if got <= 0:
                return f"摆好了但没出 {item.split(':')[-1]}（可能这个配方服务器不认）。"
            made += got
        return f"做好了 {item.split(':')[-1]} ×{made}。"
