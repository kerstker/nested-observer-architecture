"""
Level 8l: playing toward something, the way a good player does.

A good player does not chop trees because trees are there. They want an iron pickaxe, so they know they need
three iron ingots and two sticks; ingots come from raw iron in a furnace; raw iron from iron ore, which takes a
stone pickaxe and is most common around y=16. They do the first thing on that chain that they can do now, then
the next.

The Planner does that thinking from the modpack's own data (recipes, smelting, block drops, tool tiers): given a
goal and what is in her bag, it returns the single next step - gather, craft, smelt, resource hunt - or says why it is stuck.
The Ladder is the usual progression (wood -> stone -> iron -> diamond tools and armor), so she always has a goal
when nobody gave her one. Executing the steps is the body's job (Skills.goal).
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from src.levels.l8_agency.mc_knowledge import ModpackIndex, TOOL_TIER

PICKAXE_OF_TIER = ("minecraft:wooden_pickaxe", "minecraft:stone_pickaxe", "minecraft:iron_pickaxe", "minecraft:diamond_pickaxe",
                   "minecraft:netherite_pickaxe")
TABLE = "minecraft:crafting_table"
FURNACE = "minecraft:furnace"

# Where raw things come from: the blocks to mine, the tool kind and tier that make them drop, and the height
# (y) where a player goes to find them (None: anywhere near).
GATHER: Dict[str, Tuple[List[str], Optional[str], int, Optional[int]]] = {
    "minecraft:cobblestone": (["minecraft:stone", "minecraft:cobblestone"], "pickaxe", 0, None),
    "minecraft:cobbled_deepslate": (["minecraft:deepslate"], "pickaxe", 0, -8),
    "minecraft:coal": (["minecraft:coal_ore", "minecraft:deepslate_coal_ore"], "pickaxe", 0, None),
    "minecraft:raw_iron": (["minecraft:iron_ore", "minecraft:deepslate_iron_ore"], "pickaxe", 1, 16),
    "minecraft:raw_copper": (["minecraft:copper_ore", "minecraft:deepslate_copper_ore"], "pickaxe", 1, 48),
    "minecraft:lapis_lazuli": (["minecraft:lapis_ore", "minecraft:deepslate_lapis_ore"], "pickaxe", 1, 0),
    "minecraft:raw_gold": (["minecraft:gold_ore", "minecraft:deepslate_gold_ore"], "pickaxe", 2, -16),
    "minecraft:redstone": (["minecraft:redstone_ore", "minecraft:deepslate_redstone_ore"], "pickaxe", 2, -58),
    "minecraft:diamond": (["minecraft:diamond_ore", "minecraft:deepslate_diamond_ore"], "pickaxe", 2, -58),
    "minecraft:emerald": (["minecraft:emerald_ore", "minecraft:deepslate_emerald_ore"], "pickaxe", 2, None),
    "minecraft:sand": (["minecraft:sand"], "shovel", 0, None),
    "minecraft:gravel": (["minecraft:gravel"], "shovel", 0, None),
    "minecraft:flint": (["minecraft:gravel"], "shovel", 0, None),
    "minecraft:dirt": (["minecraft:dirt", "minecraft:grass_block"], "shovel", 0, None),
    "minecraft:clay_ball": (["minecraft:clay"], "shovel", 0, None),
    "minecraft:obsidian": (["minecraft:obsidian"], "pickaxe", 3, None),
}
# Resource hunts are bounded survival actions. Counts are checked in the inventory, never inferred
# from a mob disappearing from perception (which may only mean it walked out of range).
HUNT = {"minecraft:leather": "minecraft:cow", "minecraft:string": "minecraft:spider",
        "minecraft:feather": "minecraft:chicken", "minecraft:white_wool": "minecraft:sheep",
        "minecraft:blaze_rod": "minecraft:blaze", "minecraft:ender_pearl": "minecraft:enderman",
        "minecraft:bone": "minecraft:skeleton", "minecraft:slime_ball": "minecraft:slime"}
# things a body cannot get by mining or crafting from scratch yet (mobs, the Nether...): the planner says so
NOT_YET = {"minecraft:gunpowder": "需要可靠的远程战斗或其他火药来源，避免盲目贴近苦力怕"}
FUEL = {"minecraft:coal": 8, "minecraft:charcoal": 8, "minecraft:coal_block": 80, "minecraft:blaze_rod": 12}

INF = 10 ** 6
COMMON_LOGS = ("oak_log", "birch_log", "spruce_log", "dark_oak_log", "jungle_log")


def is_log(item: str) -> bool:
    name = item.split(":")[-1]
    return (name.endswith(("_log", "_stem")) and "stripped" not in name) or name in ("crimson_hyphae", "warped_hyphae")


def tool_tier(item: str) -> int:
    return max((t for m, t in TOOL_TIER.items() if f":{m}_" in item), default=-1)


def best_tier(have: Dict[str, int], kind: str) -> int:
    return max((tool_tier(i) for i, n in have.items() if n > 0 and i.endswith(f"_{kind}")), default=-1)


GEAR = ("_pickaxe", "_axe", "_shovel", "_hoe", "_sword", "_helmet", "_chestplate", "_leggings", "_boots", "shield", "bow",
        "crossbow", "trident", "fishing_rod", "shears", "flint_and_steel")


def is_gear(item: str) -> bool:
    """Tools, weapons and armor: what she uses, never an ingredient (no melting her pickaxe for nuggets)."""
    return item.split(":")[-1].endswith(GEAR)


def fuel_units(have: Dict[str, int]) -> float:
    units = sum(FUEL.get(i, 0) * n for i, n in have.items())
    units += 1.5 * sum(n for i, n in have.items() if i.endswith("_planks") or is_log(i))
    return units


class Step(dict):
    """kind: gather | chop | craft | smelt | hunt | done | stuck; plus item, count and what the kind needs."""

    def text(self, name=lambda i: i.split(":")[-1]) -> str:
        k = self["kind"]
        if k == "chop":
            return f"砍树拿木头（要 {self['count']} 块）"
        if k == "gather":
            where = f"，去 y≈{self['y']}" if self.get("y") is not None else ""
            return f"挖 {name(self['blocks'][0])} 拿 {name(self['item'])} ×{self['count']}{where}"
        if k == "craft":
            return f"合成 {name(self['item'])} ×{self['count']}"
        if k == "smelt":
            return f"用熔炉把 {name(self['input'])} 烧成 {name(self['item'])} ×{self['count']}"
        if k == "hunt":
            return f"找 {name(self['mob'])} 获取 {name(self['item'])} ×{self['count']}，以实际捡到的数量为准"
        if k == "done":
            return "已经有了"
        return f"卡住了：{self.get('why', '')}"


class Planner:
    def __init__(self, index: ModpackIndex):
        self.ix = index

    # -- how expensive is getting one of an item (to choose between recipes and between alternatives) --
    def cost(self, item: str, have: Dict[str, int], seen: Tuple[str, ...] = (), memo: Optional[Dict[str, float]] = None,
             goal: bool = False) -> float:
        memo = {} if memo is None else memo
        if is_gear(item) and not goal:              # as an ingredient: never
            return INF
        if have.get(item, 0) > 0:
            return 0
        if item in memo:
            return memo[item]
        if item in seen or len(seen) > 7:
            return INF
        if is_log(item):                       # common trees are easier to find than acacia or cherry
            memo[item] = 3 if item.split(":")[-1] in COMMON_LOGS else 5
            return memo[item]
        if item in GATHER:
            memo[item] = 3 + 2 * GATHER[item][2]
            return memo[item]
        if item in HUNT:
            return 30 if item in ("minecraft:blaze_rod", "minecraft:ender_pearl") else 12
        if item in NOT_YET:
            return INF
        best = INF
        for r in self.ix.made_by(item)[:6]:
            if r["t"] not in ("crafting_shaped", "crafting_shapeless", "smelting"):
                continue
            c = 1.0
            for alts in r["in"]:
                c += min((self.cost(a, have, seen + (item,), memo) for a in self.ix.expand(alts)[:12]), default=INF) / max(1, r["n"])
                if c >= INF:
                    break
            best = min(best, c)
        memo[item] = best
        return best

    def pick(self, alts: List[str], have: Dict[str, int], need: int) -> str:
        options = [a for a in self.ix.expand(alts)[:16] if not is_gear(a)] or self.ix.expand(alts)[:1]
        enough = [a for a in options if have.get(a, 0) >= need]
        if enough:
            return max(enough, key=lambda a: have[a])
        memo: Dict[str, float] = {}
        return min(options, key=lambda a: (self.cost(a, have, (), memo), -have.get(a, 0)))

    # -- the next thing to do --
    def next_step(self, goal: str, count: int, have: Dict[str, int], stations: Set[str] = frozenset()) -> Step:
        bag = {k: v for k, v in have.items() if v > 0}
        if bag.get(goal, 0) >= count:
            return Step(kind="done", item=goal, count=count)
        step = self._need(goal, count, bag, set(stations), 0, ())
        return step or Step(kind="done", item=goal, count=count)

    def _tool(self, kind: Optional[str], tier: int, have, stations, depth, path) -> Optional[Step]:
        """A tool that must be in the bag (not used up)."""
        if not kind or best_tier(have, kind) >= tier or (kind != "pickaxe" and tier == 0):
            return None
        tool = PICKAXE_OF_TIER[tier] if kind == "pickaxe" else f"minecraft:{['wooden', 'stone', 'iron', 'diamond'][tier]}_{kind}"
        return self._need(tool, 1, dict(have), stations, depth + 1, path)

    def _need(self, item: str, n: int, have: Dict[str, int], stations: Set[str], depth: int, path: Tuple[str, ...]) -> Optional[Step]:
        h = have.get(item, 0)
        if h >= n:
            have[item] = h - n
            return None
        short = n - h
        have[item] = 0
        if item in path or depth > 10:
            return Step(kind="stuck", item=item, count=short, why=f"做 {self.ix.name(item)} 绕回来了")
        path = path + (item,)
        if is_log(item):
            return Step(kind="chop", item=item, count=short)
        if item in GATHER:
            blocks, kind, tier, y = GATHER[item]
            t = self._tool(kind, tier, have, stations, depth, path)
            if t:
                return t
            return Step(kind="gather", item=item, count=short, blocks=blocks, tool=kind, tier=tier, y=y)
        if item in HUNT:
            return Step(kind="hunt", item=item, count=short, mob=HUNT[item])
        if item in NOT_YET:
            return Step(kind="stuck", item=item, count=short, why=f"缺 {self.ix.name(item)}（{NOT_YET[item]}）")
        memo: Dict[str, float] = {}
        recipes = [r for r in self.ix.made_by(item)[:8] if r["t"] in ("crafting_shaped", "crafting_shapeless", "smelting")]
        if not recipes:
            src = self.ix.dropped_by(item)
            if src:
                kind, tier = self.ix.tool_for(src[0])
                t = self._tool(kind, tier, have, stations, depth, path)
                return t or Step(kind="gather", item=item, count=short, blocks=src[:4], tool=kind, tier=tier, y=None)
            return Step(kind="stuck", item=item, count=short, why=f"不知道 {self.ix.name(item)} 从哪来")

        def rcost(r):
            return sum(min((self.cost(a, have, path, memo) for a in self.ix.expand(alts)[:12]), default=INF) for alts in r["in"])
        recipe = min(recipes, key=rcost)
        if rcost(recipe) >= INF:
            why = next((NOT_YET[a] for alts in recipe["in"] for a in self.ix.expand(alts)[:4] if a in NOT_YET), None)
            return Step(kind="stuck", item=item, count=short, why=f"做 {self.ix.name(item)} 的材料弄不到" + (f"（{why}）" if why else ""))
        if recipe["t"] == "smelting":
            src = self.pick(recipe["in"][0], have, short)
            if FURNACE not in stations:
                s = self._need(FURNACE, 1, dict(have), stations, depth + 1, path)
                if s:
                    return s
            s = self._need(src, short, have, stations, depth + 1, path)
            if s:
                return s
            if fuel_units(have) < short:
                s = self._need("minecraft:coal", math.ceil(short / 8), have, stations, depth + 1, path)
                if s:
                    return s
            return Step(kind="smelt", item=item, count=short, input=src)
        crafts = math.ceil(short / max(1, recipe["n"]))
        groups: Dict[Tuple[str, ...], int] = {}
        for alts in recipe["in"]:
            groups[tuple(alts)] = groups.get(tuple(alts), 0) + 1
        big = recipe.get("w", 0) > 2 or recipe.get("h", 0) > 2 or len(recipe["in"]) > 4      # only a table has 3x3
        if big and TABLE not in stations and have.get(TABLE, 0) == 0:
            s = self._need(TABLE, 1, dict(have), stations, depth + 1, path)
            if s:
                return s
        for alts, per in groups.items():
            s = self._need(self.pick(list(alts), have, per * crafts), per * crafts, have, stations, depth + 1, path)
            if s:
                return s
        return Step(kind="craft", item=item, count=short)


# --- the usual progression, so she always has something worth doing --------------------------------

LADDER: List[Tuple[str, int]] = [
    ("minecraft:wooden_pickaxe", 1), ("minecraft:stone_pickaxe", 1), ("minecraft:stone_axe", 1),
    ("minecraft:stone_sword", 1), ("minecraft:furnace", 1), ("minecraft:torch", 16),
    ("minecraft:iron_pickaxe", 1), ("minecraft:iron_sword", 1), ("minecraft:shield", 1),
    ("minecraft:iron_chestplate", 1), ("minecraft:iron_leggings", 1), ("minecraft:iron_helmet", 1),
    ("minecraft:iron_boots", 1), ("minecraft:bucket", 1), ("minecraft:iron_axe", 1),
    ("minecraft:diamond_pickaxe", 1), ("minecraft:diamond_sword", 1), ("minecraft:diamond_chestplate", 1),
    ("minecraft:diamond_leggings", 1), ("minecraft:diamond_helmet", 1), ("minecraft:diamond_boots", 1),
]
TOOL_KINDS = ("pickaxe", "axe", "sword", "shovel", "hoe")
ARMOR = ("helmet", "chestplate", "leggings", "boots")


def satisfied(item: str, count: int, have: Dict[str, int], worn: Set[str] = frozenset()) -> bool:
    """She already has it, or something at least as good of the same kind (a stone pickaxe is not needed by
    someone with an iron one; armor she is wearing counts)."""
    if have.get(item, 0) >= count:
        return True
    kind = item.split("_")[-1]
    tier = tool_tier(item)
    if kind in TOOL_KINDS:
        return best_tier(have, kind) >= tier >= 0
    if kind in ARMOR:
        owned = set(worn) | {i for i, n in have.items() if n > 0}
        return any(i.endswith(f"_{kind}") and tool_tier(i) >= tier for i in owned)
    return False


def next_goal(have: Dict[str, int], worn: Set[str] = frozenset(), skip: Set[str] = frozenset()) -> Optional[Tuple[str, int]]:
    for item, count in LADDER:
        if item not in skip and not satisfied(item, count, have, worn):
            return item, count
    return None


# --- the pack's quest book: goals the pack itself suggests ------------------------------------------

class QuestBook:
    """The FTB quests whose tasks are all "have this item", in the pack's own order. The client cannot see
    the server's quest progress, so the legacy done set records observed item conditions only. It is not
    evidence that the server accepted a task, reward was claimed, or the modpack was completed."""

    def __init__(self, index: ModpackIndex, path: Optional[str] = None):
        self.ix = index
        self.path = path
        self.done: Set[str] = set()
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    self.done = set(json.load(f))
            except (OSError, ValueError):
                pass

    def _save(self):
        if self.path:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(sorted(self.done), f)
            except OSError:
                pass

    def item_quests(self) -> List[Dict[str, Any]]:
        out = []
        for q in self.ix.load().quests:
            tasks = [t for t in q["tasks"] if t.get("type") != "checkmark"]
            if tasks and all(t.get("type") == "item" and t.get("item") and t["item"] != "ftbquests:missing_item" for t in tasks):
                out.append(q)
        return out

    def wants(self, q: Dict[str, Any]) -> List[Tuple[str, int]]:
        """Each task as a concrete item (a tag task: the member easiest to get) and how many."""
        out = []
        for t in q["tasks"]:
            if t.get("type") != "item":
                continue
            item = t["item"]
            if item.startswith("#"):
                members = self.ix.expand([item])
                if not members:
                    continue
                item = members[0]
            out.append((item, int(t.get("count") or 1)))
        return out

    def check(self, have: Dict[str, int]) -> List[str]:
        """Mark the quests whose items she has now; returns the titles newly done."""
        new = []
        for q in self.item_quests():
            if q["id"] in self.done:
                continue
            want = self.wants(q)
            if want and all(have.get(i, 0) >= n or any(have.get(m, 0) >= n for m in self._members(q, i)) for i, n in want):
                self.done.add(q["id"])
                new.append(q["title"] or q["id"])
        if new:
            self._save()
        return new

    def _members(self, q, item) -> List[str]:
        tags = [t["item"] for t in q["tasks"] if str(t.get("item", "")).startswith("#")]
        return [m for tag in tags for m in self.ix.expand([tag])]

    def next(self, have: Dict[str, int], planner: "Planner", skip: Set[str] = frozenset(),
             stations: Set[str] = frozenset()) -> Optional[Tuple[Dict[str, Any], str, int]]:
        """The quest to work on: not done, every item reachable from here, the cheapest first."""
        best, best_cost = None, INF
        for q in self.item_quests():
            if q["id"] in self.done:
                continue
            want = [(i, n) for i, n in self.wants(q) if have.get(i, 0) < n]
            if not want or any(i in skip for i, _ in want):
                continue
            steps = [planner.next_step(i, n, dict(have), set(stations)) for i, n in want]
            if any(s["kind"] == "stuck" for s in steps):
                continue
            # wood from a tree that may not grow anywhere near (acacia, cherry...): not worth wandering for
            if any(s["kind"] == "chop" and s["item"].split(":")[-1] not in COMMON_LOGS for s in steps):
                continue
            cost = sum(planner.cost(i, {}, goal=True) * n for i, n in want)
            if cost < best_cost:
                best, best_cost = (q, want[0][0], want[0][1]), cost
        return best
