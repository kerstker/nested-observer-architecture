"""
Level 8p: 老玩家怎么玩 - the decisions a good Minecraft player makes without stopping to think.

This module is *pure planning*: it never touches the game. It takes a small dict of what she can see and
returns at most one action for the current moment, exactly the call a veteran makes in a glance:

    "it's getting dark and I have no bed -> dig into the hill"
    "a creeper three blocks away -> walk backwards"
    "I have iron gear and a cheap quest ready -> do the quest, not the diamond chain"
    "diamond target, I am at y=16 -> staircase down to y=-58 first"

`game.py` (GameBridge.auto_work) asks it before running its own goal logic; `skills`/`body` carry the action out.

STATE DICT (all keys optional; defaults are ordinary daytime, healthy, empty-handed)
----------------------------------------------------------------------------------------
    t             float   wall clock, time.time() (used for anti-thrash / shelve expiry; default now)
    hour          float   world/local hour 0-23 (night is 19:00-05:00)
    night         bool    explicit darkness/in-game night (overrides `hour`)
    in_world      bool    her client is in the world (False: loading/disconnected/death screen)
    dead          bool    the death screen is up right now
    pos           (x,y,z) her feet position; `y` may be given separately
    y             float   shorthand for pos[1]
    health        float   0-20
    food          float   0-20
    inventory     {item_id: count}   what is in the bag (namespace or bare names both accepted)
    hostiles      [{"type": "minecraft:zombie", "distance": 3.2, "x":..., "y":..., "z":..., "uuid": "..."}]
                          hostiles in sight, order does not matter (nearest is computed)
    players_near  int     players within ~32 blocks (informational)
    home          {"x","y","z"} | None   where home is
    bed           bool    a bed at home / in the bag
    tools         {"pickaxe": 2, "sword": 2, "axe": 1, "shield": True,
                   "pickaxe": {"tier": 2, "left": 8}}   best tool tier per kind, wear optional
    sky_above     bool    open sky over her head (False: underground)
    dimension     str     "minecraft:overworld" / "the_nether" / "the_end"
    goal          str     the goal she is working on (item id)
    goal_count    int
    goal_by       str     "auto" | "quest" | "self"
    goal_stuck    bool    that goal just got stuck (it is set aside for a while)
    stuck         {item: t} | [item]   goals that got stuck lately (skipped for SHELVE seconds)
    quest_goals   [dict]  what QuestBook.next_goals(...) returned (else self.quests is asked)
    ladder_goals  [{"item","count"}]  the next rungs of mc_goal's ladder (else it is asked lazily)
    worn          set     armor currently worn (used by mc_goal.satisfied when available)
    stations      set     "minecraft:crafting_table" / "minecraft:furnace" nearby
    death         {"pos": (x,y,z), "recovered": False} | "death_pos": (x,y,z)
    hazard        {"kind": "lava"|"cliff"|..., "pos": (x,y,z), "distance": 2.0}
    mine_target   str     item/ore she was told to mine right now (opts into mining_plan)
    task/mode     str     "mine" opts into mining_plan
    busy          bool    a skill/thread has her hands right now (no new work)
    job_running   bool    a node-side background job is running
    free_slots    int     free inventory slots (else estimated from `inventory`)
    chest_near    bool    one of her chests is within reach
    furnace_near  bool    a furnace is within reach / at home
    hut           {stage: bool}  which starter-hut stages are already built (else inferred)

ACTION DICT
-----------
    {"kind": "skill"|"goal"|"walk"|"wait"|"fight"|"shelter"|"home", "name": str, "args": dict, "why": str}
or None (nothing worth doing: do NOT wander without a goal).

    skill    start a Skills skill: skills.start(name, **ProPlanner.skill_kwargs(name, args)) - args may also
             carry descriptive planning fields (spacing, torch_every, target, ...) that the skill does not take
    goal     work toward item `name`: set_goal(name, args["count"], args["by"]) then start("goal")
    walk     go to args x/y/z: skills.start("go_to", ...); args may carry "from" when only a direction is known
    wait     do nothing new this tick (args["seconds"], optional args["item"]); reflexes still run
    fight    do not start a task: args["mode"] is flee|retreat|shield|engage|heal, args["target"], args["distance"]
    shelter  dig in / make a bed: args depth/size/seal/torches -> tunnel/dig_down + place blocks
    home     home business, name go_home|set_home|build_hut: args target / hut checklist / missing stages

PRECEDENCE in next_action (the veteran's checklist, top to bottom)
------------------------------------------------------------------
1. not in world / death screen        -> wait
2. lava, fire, cliff right here       -> back away (hazard_plan)
3. creeper near, low hp, archer       -> flee / retreat / shield (combat_plan, urgent modes)
4. busy hands or a running job        -> None (never interrupt for nothing)
5. a death to recover                 -> walk back to the grave
6. hurt or hungry                     -> eat / cook / hunt / regen (survival_plan)
7. night or hostiles and no shelter   -> home / bed / dig in (night_plan)
8. home business (site, return home)  -> set_home / go_home / build_hut (home_plan)
9. bag almost full                    -> store, go home to store, else declutter
10. a threat already in arm's reach   -> fight
11. tool about to break / too low     -> craft a replacement or upgrade
12. an explicit mining job            -> mining_plan (stairs to the ore layer, branches, torches)
13. otherwise                         -> choose_goal: quest vs ladder by expected value (anti-thrash)*
    * never re-pick a goal that just got stuck; never switch for a marginal gain; never gather extra.

Day schedule (mirrored in mc_expert.md): morning = eat, check wear, take today's one goal; day = push that
goal; dusk = head home, store, smelt; night = craft/smelt/organise inside, never explore.

No game I/O and no heavy imports; the only optional import is mc_goal's ladder, done lazily inside a method.
"""
import logging
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.levels.l8_agency.mc_food import FOOD_NAMES, RAW_FOOD, READY_FOOD, is_edible

logger = logging.getLogger(__name__)

M = "minecraft:"

# --- small tables a veteran carries in their head ---------------------------------------------------

TOOL_TIER = {"wooden": 0, "golden": 0, "stone": 1, "copper": 1, "iron": 2, "diamond": 3, "netherite": 4}
TIER_LABEL = ("木", "石", "铁", "钻石", "下界合金")
PICK_OF_TIER = (M + "wooden_pickaxe", M + "stone_pickaxe", M + "iron_pickaxe", M + "diamond_pickaxe",
                M + "netherite_pickaxe")
TOOL_KINDS = ("pickaxe", "axe", "sword", "shovel", "hoe")
ARMOR_KINDS = ("helmet", "chestplate", "leggings", "boots")

# mc_goal's usual progression (kept here so this module works without importing it; only basic gear is
# ordered anywhere near as strictly as the real ladder - everything after that is compared by value).
LADDER_ORDER: List[Tuple[str, int]] = [
    (M + "wooden_pickaxe", 1), (M + "stone_pickaxe", 1), (M + "stone_axe", 1), (M + "stone_sword", 1),
    (M + "furnace", 1), (M + "torch", 16), (M + "iron_pickaxe", 1), (M + "iron_sword", 1), (M + "shield", 1),
    (M + "iron_chestplate", 1), (M + "iron_leggings", 1), (M + "iron_helmet", 1), (M + "iron_boots", 1),
    (M + "bucket", 1), (M + "iron_axe", 1), (M + "diamond_pickaxe", 1), (M + "diamond_sword", 1),
]
LADDER_BASICS = {i: True for i in (
    M + "wooden_pickaxe", M + "stone_pickaxe", M + "stone_axe", M + "stone_sword", M + "furnace", M + "torch",
    M + "iron_pickaxe", M + "iron_sword", M + "shield", M + "iron_chestplate", M + "iron_leggings",
    M + "iron_helmet", M + "iron_boots")}

# Where each ore is common (mc_expert.md, 1.21): the one thing a veteran never gets wrong.
ORE_Y: Dict[str, Optional[int]] = {
    M + "coal": None, M + "raw_iron": 16, M + "raw_copper": 48, M + "lapis_lazuli": 0,
    M + "raw_gold": -16, M + "redstone": -58, M + "diamond": -58, M + "emerald": None,
}
ORE_PICK: Dict[str, int] = {M + "coal": 0, M + "raw_iron": 1, M + "raw_copper": 1, M + "lapis_lazuli": 1,
                            M + "raw_gold": 2, M + "redstone": 2, M + "diamond": 2, M + "emerald": 2}
ORE_BLOCKS: Dict[str, List[str]] = {
    M + "coal": [M + "coal_ore", M + "deepslate_coal_ore"],
    M + "raw_iron": [M + "iron_ore", M + "deepslate_iron_ore"],
    M + "raw_copper": [M + "copper_ore", M + "deepslate_copper_ore"],
    M + "lapis_lazuli": [M + "lapis_ore", M + "deepslate_lapis_ore"],
    M + "raw_gold": [M + "gold_ore", M + "deepslate_gold_ore"],
    M + "redstone": [M + "redstone_ore", M + "deepslate_redstone_ore"],
    M + "diamond": [M + "diamond_ore", M + "deepslate_diamond_ore"],
    M + "emerald": [M + "emerald_ore", M + "deepslate_emerald_ore"],
}
# (ORE_ALIAS, the short-name -> ore table, is built after the helpers below.)

# The first hour, in the order a speedrunner does it (mc_expert.md).
EARLY_GAME: List[Tuple[str, int, str, str]] = [
    (M + "oak_log", 6, "木头时代", "先砍 1~2 棵树：FallingTree 砍树干最下面一块，整棵倒"),
    (M + "crafting_table", 1, "木头时代", "工作台是后面所有合成的基础"),
    (M + "wooden_pickaxe", 1, "木头时代", "木镐挖圆石，马上换石器"),
    (M + "stone_pickaxe", 1, "石器时代", "圆石 ×3 + 木棍：石镐才挖得动铁/铜/青金石"),
    (M + "stone_axe", 1, "石器时代", "石斧砍树快，也是备用武器"),
    (M + "stone_sword", 1, "石器时代", "天黑之前有把石剑，打僵尸不慌"),
    (M + "furnace", 1, "熔炉与火把", "圆石 ×8：烧铁、烤肉、烧木炭"),
    (M + "torch", 16, "熔炉与火把", "煤/木炭 + 木棍；下矿至少带 16 个"),
    (M + "bread", 3, "食物", "有吃的再出门；面包或熟肉都行，熟食优先"),
    (M + "iron_pickaxe", 1, "铁器时代", "y≈16 挖铁，3 锭先做铁镐（金/钻石/红石都要铁镐）"),
    (M + "iron_sword", 1, "铁器时代", "铁剑：伤害和耐久都比石剑好"),
    (M + "shield", 1, "铁器时代", "盾优先于铁甲：正面伤害和箭都能挡"),
    (M + "iron_chestplate", 1, "铁器时代", "铁甲顺序：胸甲 > 护腿 > 头盔 > 靴子"),
    (M + "iron_leggings", 1, "铁器时代", "护腿是第二件铁甲"),
    (M + "iron_helmet", 1, "铁器时代", "头盔"),
    (M + "iron_boots", 1, "铁器时代", "靴子；整套 24 锭"),
    (M + "bucket", 1, "铁器时代", "水桶：岩浆、落地、清火都有用"),
    (M + "diamond_pickaxe", 1, "钻石时代", "y≈-58 挖钻石；钻石先给镐（效率/时运）"),
]

COOKED_FOOD = READY_FOOD
RAW_ORES = {M + "raw_iron", M + "raw_copper", M + "raw_gold"}
RANGED_ENEMIES = ("skeleton", "stray", "bogged", "pillager", "piglin_brute", "illusioner")
LAVA_KINDS = ("lava", "magma", "fire", "campfire", "lava_cauldron")
CLIFF_KINDS = ("cliff", "fall", "void", "ravine", "drop", "cave")

# The 5x5 starter hut: walls, door, roof, light, then the furniture a first base needs.
HUT_STEPS: List[Dict[str, Any]] = [
    {"stage": "walls", "do": "5x5 木石外墙与立柱，按实物蓝图续建", "items": {}},
    {"stage": "door", "do": "朝空地开 1 扇门（别对着悬崖）", "items": {M + "oak_door": 1}},
    {"stage": "roof", "do": "山墙坡屋顶与前后屋檐，按实物蓝图续建", "items": {}},
    {"stage": "windows", "do": "四面玻璃窗，核验实际玻璃方块", "items": {M + "glass": 4}},
    {"stage": "light", "do": "小屋内放火把并确认实物", "items": {M + "torch": 1}},
    {"stage": "furniture", "do": "床、工作台、熔炉、箱子放屋里", "items": {M + "crafting_table": 1,
                                                                              M + "furnace": 1,
                                                                              M + "chest": 1,
                                                                              M + "white_bed": 1}},
    {"stage": "safety", "do": "小屋外四个方向各放一支火把；出门后另查附近威胁", "items": {M + "torch": 4}},
]
HUT_ORDER = [step["stage"] for step in HUT_STEPS]
HUT_EXPAND = ["围墙和院子", "小麦/土豆地", "分类箱子（矿/木/食物/杂物）", "附魔台 + 15 书架", "下界传送门",
              "传送石"]

BOSS_CHECKLIST = [
    ("铁甲一套（或更好）", "没甲打 boss 一两下就没了"),
    ("铁剑以上", "剑蓄满力 + 下落暴击才打得动"),
    ("盾牌", "盾挡正面攻击和远程"),
    ("弓 + 32 支箭", "打龙水晶这类远程阶段要用"),
    ("熟食 ×8 以上", "回血要饱食"),
    ("方块 ×32", "被围住时往脚下垫高"),
    ("水桶", "落地、岩浆、清火"),
    ("治疗/抗火药水或图腾", "保命"),
]

DAY_PLAN = {
    "morning": "出门前：吃饱、看工具耐久、带火把/方块/吃的、定今天一个目标",
    "day": "推进一个目标；背包快满或工具快坏先处理；天黑前 1 小时往家走",
    "dusk": "回家、存东西、把今天的矿烧上、补火把",
    "night": "屋里合成、熔炼、整理箱子；不外出探索、不挖新矿道；没家就挖进山坡堵门",
}

# Which `args` each Skills skill actually accepts: action dicts also carry descriptive fields (spacing,
# torch_every, target...), and ProPlanner.skill_kwargs() drops those before skills.start(name, **kwargs).
SKILL_ARGS: Dict[str, Tuple[str, ...]] = {
    "chop_tree": ("max_logs", "radius", "prefer"),
    "mine": ("blocks", "count", "radius"),
    "craft": ("item", "count"),
    "smelt": ("item", "count"),
    "dig_down": ("to_y", "max_steps"),
    "dig_up": ("to_y", "max_steps", "until_sky"),
    "tunnel": ("length", "ores"),
    "go_to": ("x", "y", "z"),
    "place": ("item", "x", "y", "z"),
    "collect_drops": ("radius",),
    "equip": ("kind",),
    "dress": (),
    "tidy": (),
    "declutter": ("want_free", "blocks_only"),
    "hunt": ("count",),
    "forage": (),
    "set_home": (),
    "go_home": (),
    "build_hut": (),
    "build_portal": (),
    "enter_portal": (),
    "hunt_resource": ("item", "count"),
    "end_step": (),
    "explore_cave": ("max_seconds", "max_nodes", "ores"),
    "light_area": ("force",),
    "fill_water_bucket": (),
    "escape": ("rounds",),
    "store": (),
    "go_to_player": ("player",),
}


# --- tiny helpers ------------------------------------------------------------------------------------

def short(item: Any) -> str:
    """'minecraft:iron_ore' / 'minecraft:iron_ore[foo]' -> 'iron_ore'."""
    return str(item or "").split("[")[0].split(":")[-1].split("/")[-1]


def human(item: Any) -> str:
    return short(item).replace("_", " ")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pos(value: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(value, dict):
        if all(value.get(k) is not None for k in ("x", "y", "z")):
            return (_num(value["x"]), _num(value["y"]), _num(value["z"]))
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
    return None


def _dist(a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]) -> float:
    if a is None or b is None:
        return 1e9
    return math.dist(a, b)


def tool_tier(item: Any) -> int:
    """0 wood/gold, 1 stone/copper, 2 iron, 3 diamond, 4 netherite, -1 for anything else."""
    return TOOL_TIER.get(short(item).split("_")[0], -1)


def _act(kind: str, name: str, args: Optional[Dict[str, Any]] = None, why: str = "") -> Dict[str, Any]:
    return {"kind": str(kind), "name": str(name), "args": dict(args or {}), "why": str(why)}


def _count(have: Dict[str, int], *items: Any) -> int:
    total = 0
    for item in items:
        item = str(item)
        if not item:
            continue
        total += int(have.get(item, 0) or 0)
        if ":" not in item:
            total += int(have.get(M + item, 0) or 0)
    return total


def _sum(have: Dict[str, int], test) -> int:
    return sum(int(n or 0) for item, n in have.items() if test(short(item), str(item)))


# what a goal/block name means in ore terms ("diamond_pickaxe" is not an ore)
ORE_ALIAS: Dict[str, str] = {}
for _ore, _blocks in ORE_BLOCKS.items():
    ORE_ALIAS[short(_ore)] = _ore
    for _b in _blocks:
        ORE_ALIAS[short(_b)] = _ore


class ProPlanner:
    """A veteran's decisions as pure functions over a small state dict (see the module docstring)."""

    SHELVE = 1200.0        # a goal that just got stuck is left alone this long (20 min)
    SWITCH_HOLD = 45.0     # do not switch goals for a marginal gain sooner than this
    RETURN_HOME_DIST = 96.0
    AT_HOME = 24.0
    NIGHT_FROM, NIGHT_TO = 19.0, 5.0
    FIGHT_HP = 13.0        # below this: do not pick a fight
    FLEE_HP = 9.0          # below this: leave, whatever the gear
    EAT_BELOW = 18.0       # eat when hurt and food is this low ...
    EAT_FOOD = 14.0        # ... or when food is this low anyway
    ATTACK_REACH = 4.0
    CREEPER_KEEP = 12.0
    WEAR_LOW = 12          # tool durability left that means "make a new one first"
    LOW_SLOTS = 4
    QUEST_MAX_COST = 50.0  # a quest costing more than this is not "next" (game.py uses the same 50)

    def __init__(self, index: Any = None, quests: Any = None):
        """`index`: a mc_knowledge.ModpackIndex (only for item names). `quests`: a mc_quests.QuestBook."""
        self.index = index
        self.quests = quests
        self.shelved: Dict[str, float] = {}      # item -> when it got stuck
        self.history: List[Dict[str, Any]] = []  # the last choices (for debugging / no-thrash)
        self._chosen: Optional[Dict[str, Any]] = None
        self._chosen_at = 0.0

    # --- names and the anti-thrash memory -----------------------------------------------------------

    def name(self, item: Any) -> str:
        try:
            if self.index is not None:
                return str(self.index.name(item))
        except Exception:
            pass
        return human(item)

    @staticmethod
    def skill_kwargs(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The subset of an action's `args` that Skills.start(name, **kwargs) accepts (drops planning fields)."""
        allowed = SKILL_ARGS.get(str(name))
        if allowed is None:
            return dict(args or {})
        return {k: v for k, v in (args or {}).items() if k in allowed}

    def shelve(self, item: Any, t: Optional[float] = None) -> None:
        if item:
            self.shelved[str(item)] = _num(t, time.time())

    def shelved_items(self, t: Optional[float] = None) -> set:
        self._prune(_num(t, time.time()))
        return set(self.shelved)

    def _prune(self, t: float) -> None:
        self.shelved = {k: v for k, v in self.shelved.items() if t - v < self.SHELVE}

    def _skip(self, state: Dict[str, Any], t: float) -> set:
        """Items not to choose now: recently stuck, plus whatever the caller says to skip."""
        skip = set(self.shelved)
        stuck = state.get("stuck")
        if isinstance(stuck, dict):
            for item, when in stuck.items():
                if t - _num(when, t) < self.SHELVE:
                    skip.add(str(item))
        elif isinstance(stuck, (list, tuple, set)):
            skip.update(str(i) for i in stuck)
        skip.update(str(i) for i in (state.get("skip") or ()))
        goal = state.get("goal")
        if goal and state.get("goal_stuck"):
            skip.add(str(goal))
            self.shelve(goal, t)                 # it just got stuck: put it aside for a while
        return skip

    def _remember_choice(self, choice: Dict[str, Any], t: float) -> None:
        self._chosen = dict(choice)
        self._chosen_at = t
        self.history = ([{"t": t, "item": choice.get("item"), "by": choice.get("by"),
                          "why": choice.get("why")}] + self.history)[:20]

    # --- reading the state --------------------------------------------------------------------------

    def _have(self, state: Dict[str, Any]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        inv = state.get("inventory") or {}
        if isinstance(inv, dict):
            for item, n in inv.items():
                try:
                    n = int(n or 0)
                except (TypeError, ValueError):
                    n = 0
                if n > 0:
                    out[str(item)] = n
        return out

    def _item(self, state: Dict[str, Any], *items: Any) -> int:
        return _count(self._have(state), *items)

    def _tool_tier(self, state: Dict[str, Any], kind: str) -> int:
        tools = state.get("tools") or {}
        tier = self._tier_of(tools.get(kind)) if isinstance(tools, dict) else -1
        if tier >= 0:
            return tier
        have = self._have(state)
        return max((tool_tier(i) for i in have if short(i).endswith("_" + kind)), default=-1)

    @staticmethod
    def _tier_of(value: Any) -> int:
        if value is None or isinstance(value, bool):
            return -1
        if isinstance(value, dict):
            if value.get("tier") is not None:
                return int(_num(value["tier"], -1))
            for key in ("item", "id", "name"):
                if value.get(key):
                    return tool_tier(value[key])
            return -1
        if isinstance(value, (int, float)):
            return int(value)
        return tool_tier(value)

    @staticmethod
    def _wear(state: Dict[str, Any], kind: str) -> Optional[int]:
        tools = state.get("tools") or {}
        value = tools.get(kind) if isinstance(tools, dict) else None
        if isinstance(value, dict):
            for key in ("left", "durability", "remaining"):
                if value.get(key) is not None:
                    return int(_num(value[key]))
        return None

    def _tool_count(self, state: Dict[str, Any], kind: str) -> int:
        return _sum(self._have(state), lambda s, _i, k=kind: s.endswith("_" + k))

    def _has_shield(self, state: Dict[str, Any]) -> bool:
        tools = state.get("tools") or {}
        value = tools.get("shield") if isinstance(tools, dict) else None
        if value is True:
            return True
        if isinstance(value, (int, float)) and not isinstance(value, bool) and int(value) >= 0:
            return True
        if isinstance(value, str) and value:
            return True
        if isinstance(value, dict) and (value.get("item") or value.get("tier") is not None):
            return True
        return any(short(i).endswith("shield") for i in self._have(state))

    def _potion_count(self, state: Dict[str, Any]) -> int:
        return _sum(self._have(state), lambda s, _i: s in ("potion", "healing_potion", "regeneration_potion",
                                                           "golden_apple", "enchanted_golden_apple"))

    def _cooked_count(self, state: Dict[str, Any]) -> int:
        return _sum(self._have(state), lambda s, i: is_edible(i) and s in COOKED_FOOD)

    def _food_items(self, state: Dict[str, Any]) -> List[Tuple[str, int]]:
        return [(i, n) for i, n in self._have(state).items() if is_edible(i)]

    def _raw_food(self, state: Dict[str, Any]) -> List[Tuple[str, int]]:
        return [(i, n) for i, n in self._have(state).items() if is_edible(i) and short(i) in RAW_FOOD]

    def _raw_ores(self, state: Dict[str, Any]) -> List[Tuple[str, int]]:
        return [(i, n) for i, n in self._have(state).items() if short(i) in {short(o) for o in RAW_ORES}]

    def _torches(self, state: Dict[str, Any]) -> int:
        return self._item(state, M + "torch")

    def _wool(self, state: Dict[str, Any]) -> int:
        return _sum(self._have(state), lambda s, _i: s.endswith("_wool"))

    def _planks(self, state: Dict[str, Any]) -> int:
        return _sum(self._have(state), lambda s, _i: s.endswith("_planks"))

    def _wood(self, state: Dict[str, Any]) -> float:
        have = self._have(state)
        logs = _sum(have, lambda s, _i: s.endswith(("_log", "_wood")))
        return logs + self._planks(state) / 4.0

    def _building_blocks(self, state: Dict[str, Any]) -> int:
        return _sum(self._have(state), lambda s, _i: (
            s in ("cobblestone", "cobbled_deepslate", "stone", "dirt", "sand", "gravel", "granite", "diorite",
                  "andesite", "tuff", "deepslate")
            or s.endswith(("_planks", "_log", "_wood", "_terracotta", "_wool", "_bricks", "_stone_bricks"))))

    def _free_slots(self, state: Dict[str, Any]) -> int:
        if state.get("free_slots") is not None:
            return max(0, int(_num(state.get("free_slots"), 36)))
        used = 0
        for n in self._have(state).values():
            used += max(1, math.ceil(n / 64.0))       # every item takes at least one slot; stacks of 64
        return max(0, 36 - used)

    def _furnace_available(self, state: Dict[str, Any]) -> bool:
        return bool(state.get("furnace_near") or self._item(state, M + "furnace")
                    or (state.get("home") and state.get("at_home")))

    def _night(self, state: Dict[str, Any]) -> bool:
        if state.get("night") is not None:
            return bool(state.get("night"))
        hour = _num(state.get("hour"), 12.0) % 24.0
        return hour >= self.NIGHT_FROM or hour < self.NIGHT_TO

    @staticmethod
    def _y(state: Dict[str, Any]) -> int:
        """Her height: explicit `y` first, else pos[1], else sea level."""
        if state.get("y") is not None:
            return int(_num(state.get("y"), 64))
        spot = _pos(state.get("pos"))
        return int(spot[1]) if spot else 64

    @staticmethod
    def hostiles(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalise `state["hostiles"]` into [{"type", "distance", "pos", "uuid"}], nearest first."""
        raw = state.get("hostiles") or []
        out: List[Dict[str, Any]] = []
        if isinstance(raw, dict):                     # {"minecraft:zombie": 2}
            for kind, n in raw.items():
                for _ in range(max(0, int(_num(n, 0)))):
                    out.append({"type": short(kind), "distance": 16.0, "pos": None, "uuid": None})
        elif isinstance(raw, (list, tuple)):
            for e in raw:
                if isinstance(e, dict):
                    kind = e.get("type") or e.get("kind") or e.get("id") or e.get("name") or "?"
                    out.append({"type": short(kind), "distance": _num(e.get("distance"), 99.0),
                                "pos": _pos(e), "uuid": e.get("uuid")})
                elif e:
                    out.append({"type": short(e), "distance": 16.0, "pos": None, "uuid": None})
        out.sort(key=lambda e: e["distance"])
        return out

    # --- 1. danger right here: lava, fire, cliffs ---------------------------------------------------

    @staticmethod
    def _away(me: Optional[Tuple[float, float, float]], spot: Optional[Tuple[float, float, float]],
              distance: float) -> Optional[Tuple[float, float, float]]:
        if me is None or spot is None:
            return None
        dx, dz = me[0] - spot[0], me[2] - spot[2]
        length = math.hypot(dx, dz)
        if length < 0.01:
            return (me[0] + distance, me[1], me[2])
        return (me[0] + dx / length * distance, me[1], me[2] + dz / length * distance)

    def hazard_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Back away from lava/fire or a cliff edge; digging toward either is the classic way to die."""
        hazard = state.get("hazard")
        if isinstance(hazard, (list, tuple)):
            hazard = hazard[0] if hazard else None
        if not hazard:
            return None
        if isinstance(hazard, str):
            hazard = {"kind": hazard}
        if not isinstance(hazard, dict):
            return None
        kind = short(hazard.get("kind") or hazard.get("type") or "")
        distance = _num(hazard.get("distance"), 2.0)
        spot = _pos(hazard.get("pos"))
        me = _pos(state.get("pos"))
        if kind in LAVA_KINDS and distance <= 6:
            target = self._away(me, spot, 6)
            args: Dict[str, Any] = {"distance": 6, "from": spot, "hazard": kind}
            if target:
                args.update({"x": int(target[0]), "y": int(target[1]), "z": int(target[2])})
            return _act("walk", "go_to", args,
                        "附近有岩浆/火：先退开 6 格，别往那个方向挖；手里留一格方块随时堵，深板岩层尤其小心")
        if kind in CLIFF_KINDS and distance <= 4:
            target = self._away(me, spot, 4)
            args = {"distance": 4, "from": spot, "hazard": kind}
            if target:
                args.update({"x": int(target[0]), "y": int(target[1]), "z": int(target[2])})
            return _act("walk", "go_to", args,
                        "悬崖/深坑边：后退两步、蹲着走，别为了直线距离跳下去；落地伤害是新手最常见的死法")
        return None

    # --- 2. combat ----------------------------------------------------------------------------------

    def combat_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Engage/retreat by hp, enemy kind and gear. Creepers are never meleed; archers are closed behind a shield."""
        hostiles = self.hostiles(state)
        if not hostiles:
            return None
        health = _num(state.get("health"), 20.0)
        nearest = hostiles[0]
        kind, distance = nearest["type"], nearest["distance"]
        crowd = sum(1 for e in hostiles if e["distance"] <= 12.0)
        sword = self._tool_tier(state, "sword")
        axe = self._tool_tier(state, "axe")
        shield = self._has_shield(state)
        target = {"type": kind, "distance": distance, "pos": nearest.get("pos"), "uuid": nearest.get("uuid")}

        if kind == "creeper" and distance <= self.CREEPER_KEEP:
            return _act("fight", "flee", {"mode": "flee", "target": target, "distance": distance, "keep": 12.0},
                        "苦力怕打不死会炸：别站在它旁边等引信，退开 12 格以上再绕")
        if health <= self.FLEE_HP:
            if self._potion_count(state) and distance > 6:
                return _act("fight", "potion", {"mode": "heal", "target": target, "distance": distance},
                            "血太少：先喝治疗药水/吃金苹果，再决定打还是撤")
            return _act("fight", "retreat", {"mode": "retreat", "target": target, "distance": distance,
                                             "sprint": True},
                        "血低于 %d：先撤，退到安全处吃东西回血；别在残血时硬打" % int(self.FLEE_HP))
        if crowd >= 3 and health <= 16:
            return _act("fight", "retreat", {"mode": "retreat", "target": target, "distance": distance},
                        "被 %d 个怪围住：先退，往脚下垫方块站高或堵门，一次只打一个" % crowd)
        if health < self.FIGHT_HP and distance <= 8:
            return _act("fight", "retreat", {"mode": "retreat", "target": target, "distance": distance},
                        "血不到 %d 又没有距离：先拉开，吃一口再回头" % int(self.FIGHT_HP))
        if kind in RANGED_ENEMIES and 4 <= distance <= 16:
            if shield:
                return _act("fight", "shield", {"mode": "shield", "target": target, "distance": distance,
                                                "hold": 0.5},
                            "举盾靠近骷髅/弓手：盾挡正面箭，别站在空地上挨射")
            return _act("fight", "retreat", {"mode": "retreat", "target": target, "distance": distance},
                        "没盾：先找掩体绕过去，在空旷地站着只会被射")
        if distance > 20:
            return None
        weapon = max(sword, axe)
        if distance <= self.ATTACK_REACH:
            if weapon < 1 and health < 18:
                return _act("fight", "retreat", {"mode": "retreat", "target": target, "distance": distance},
                            "手上没有像样的武器：别用拳头打怪，先退，做把石剑/铁剑")
            return _act("fight", "engage", {"mode": "engage", "target": target, "distance": distance,
                                            "crit": True, "jump": True, "shield": shield},
                        "近身：蓄满力再挥，跳起来下落时打是暴击；有盾就挨打前举 0.5 秒")
        if distance <= 16 and health >= self.FIGHT_HP and weapon >= 1:
            return _act("fight", "engage", {"mode": "engage", "target": target, "distance": distance,
                                            "approach": True, "shield": shield},
                        "16 格内：拿剑压上去，一次只打一个，别被引到怪堆里")
        return None

    # --- 3. health and food -------------------------------------------------------------------------

    def survival_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Hurt -> eat/regen, hungry -> eat/cook/hunt. A player never starts a long job on an empty stomach."""
        health = _num(state.get("health"), 20.0)
        food = _num(state.get("food"), 20.0)
        items = self._food_items(state)
        ready = [(i, n) for i, n in items if short(i) not in RAW_FOOD]
        raw = self._raw_food(state)
        if health <= 6:
            if self._potion_count(state):
                return _act("fight", "potion", {"mode": "heal"},
                            "血太少：先喝治疗药水/吃金苹果，再决定打还是撤")
            if items:
                return _act("wait", "eat", {"seconds": 2.5},
                            "血太少：先吃（吃的时候往后退），回血再动，别开新活儿")
            if food >= 18:
                return _act("wait", "regen", {"seconds": 5.0}, "已经吃饱但血太少：在安全处等实际回血")
        hungry = food <= self.EAT_FOOD or (food <= self.EAT_BELOW and health < 20)
        if not hungry:
            return None
        if ready:
            preferred = sorted(ready, key=lambda kv: 0 if (short(kv[0]) in COOKED_FOOD
                                                           or short(kv[0]).startswith("cooked_")) else 1)[0][0]
            return _act("wait", "eat", {"seconds": 2.5, "item": preferred},
                        "饿了（饱食 %d）：先吃 %s，熟食优先；吃东西时别开新活儿" % (int(food), human(preferred)))
        if raw and food > 6 and self._furnace_available(state):
            item, n = raw[0]
            return _act("skill", "smelt", {"item": item, "count": n},
                        "只有生肉：先回家/熔炉烤熟，熟食回得多")
        if raw:
            item, n = raw[0]
            return _act("wait", "eat", {"seconds": 2.5, "item": item},
                        "没熟食：先吃生的顶一下，别空着肚子下矿（饱食低于 7 就不能疾跑）")
        if self._free_slots(state) < 2:
            return _act("skill", "declutter", {"want_free": 4, "blocks_only": True},
                        "没有食物而背包装满了：先腾出4格，才能拾取粮食；暂不重复狩猎")
        if self._item(state, M + "wheat") >= 3:
            return _act("skill", "craft", {"item": M + "bread", "count": 1}, "已有3份小麦：先做面包")
        if self._item(state, M + "hay_block"):
            return _act("skill", "craft", {"item": M + "wheat", "count": 9},
                        "包里有干草块：拆成 9 个小麦，再做 3 个面包，不用去打猎")
        search = state.get("food_search") or {}
        now = _num(state.get("t"), time.time())
        if health < 10:
            if any(e["distance"] <= 16 for e in self.hostiles(state)):
                return _act("fight", "retreat", {"mode": "retreat"}, "低血缺粮又有敌人，先退到安全处")
            if _num(search.get("forage_after"), 0) > now:
                return _act("wait", "food_search", {"seconds": 5}, "刚检查的近处没有食物，等待短途觅食重试；空腹不能靠等待回血")
            return _act("skill", "forage", {}, "低血缺粮不能等待回血：避开战斗，短途拾取食物和成熟作物")
        if self._night(state):
            return None                     # night + no food: shelter first (night_plan), hunt at dawn
        if _num(search.get("hunt_after"), 0) > now:
            if _num(search.get("forage_after"), 0) > now:
                return (None if food >= 12 and health >= 12 else
                        _act("wait", "food_search", {"seconds": 5}, "觅食短途正在冷却，避免原地重复消耗"))
            return _act("skill", "forage", {}, "附近狩猎没有收获：改捡可吃掉落、找成熟作物和短途探路")
        return _act("skill", "hunt", {"count": 3},
                    "没吃的了：打几只动物（顺手烤熟），饱食低于 7 不能疾跑、低于 9 不回血")

    # --- 4. death recovery --------------------------------------------------------------------------

    def death_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """After a respawn: walk back to the grave and pick the gear up before doing anything else."""
        if state.get("dead"):
            return _act("wait", "respawn", {"seconds": 3.0},
                        "死亡界面：等重生；重生后先回死亡点捡东西，别再空手送一次")
        death = state.get("death")
        if isinstance(death, dict):
            spot = _pos(death.get("pos") or death)
            recovered = bool(death.get("recovered"))
        else:
            spot = _pos(death) or _pos(state.get("death_pos"))
            recovered = bool(state.get("death_recovered"))
        if spot is None or recovered:
            return None
        if _num(state.get("health"), 20.0) <= 8:
            return None                     # too hurt to travel: survival_plan handles it first
        me = _pos(state.get("pos"))
        distance = _dist(me, spot)
        if distance > 4:
            return _act("walk", "go_to", {"x": int(spot[0]), "y": int(spot[1]), "z": int(spot[2]),
                                          "grave": True},
                        "回到上次死的地方：东西还在坟里（走之前先吃饱；路上别再送）")
        return _act("skill", "collect_drops", {"radius": 8.0},
                    "到了死的地方：把坟里的东西捡回来，缺工具就先补一把再继续")

    # --- 5. night shelter ---------------------------------------------------------------------------

    def night_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Night (or enough hostiles) without a safe place: home, a bed, or a 3x3 hidey-hole."""
        hostiles = self.hostiles(state)
        nearest = hostiles[0]["distance"] if hostiles else 99.0
        pressured = bool(hostiles) and (nearest <= 12 or sum(1 for e in hostiles if e["distance"] <= 16) >= 3)
        night = self._night(state)
        if not night and not pressured:
            return None
        home = _pos(state.get("home"))
        me = _pos(state.get("pos"))
        distance_home = _dist(me, home)
        if home is not None and distance_home <= self.AT_HOME:
            return self._night_at_home(state, pressured)
        if home is not None:
            return _act("home", "go_home", {"target": {"x": home[0], "y": home[1], "z": home[2]}, "night": True},
                        "天黑了：回家。日落后别赶路、别探索，僵尸和苦力怕比白天多")
        if pressured and nearest <= 8:
            return None                     # combat_plan is already dealing with what is in her face
        if self._wool(state) >= 3 and self._planks(state) >= 3:
            return _act("skill", "craft", {"item": M + "white_bed", "count": 1, "then": "place"},
                        "没有家也没有床：3 羊毛 + 3 木板先做床，睡过去还能设重生点")
        seal = next((i for i in (M + "cobblestone", M + "cobbled_deepslate", M + "dirt", M + "oak_planks")
                     if self._item(state, i)), "")
        return _act("shelter", "dig_in", {"depth": 3, "size": (3, 2, 3), "seal": True, "seal_with": seal,
                                          "torches": 1 if self._torches(state) else 0,
                                          "y": self._y(state)},
                    "没床没家：往山坡/地下挖 3 格（3×3 藏身处），用方块堵住入口、插火把，天亮再出来；"
                    "别在野外站着打怪")

    def _night_at_home(self, state: Dict[str, Any], pressured: bool) -> Dict[str, Any]:
        """At home after dark: seal the hut, store, smelt, or simply sit it out - never go exploring."""
        missing = self._hut_missing(state)
        if missing and not pressured:
            supply = self.house_supply_plan(state)
            if supply:
                return supply
        if missing and self._building_blocks(state) >= 8:
            return _act("home", "build_hut", {"size": 5, "missing": missing, "hut": HUT_STEPS, "later": HUT_EXPAND},
                        ("有怪在附近：先把屋子封起来（有门有光）" if pressured else "夜里适合把屋子弄完：")
                        + "、".join(missing))
        if pressured:
            return _act("wait", "alert", {"seconds": 3.0},
                        "屋里有怪：站在门口打，别冲进黑暗里；门关好，四周点亮")
        if self._free_slots(state) < self.LOW_SLOTS:
            return _act("skill", "store", {},
                        "夜里整理背包：把矿、多余方块、木头存进箱子；火把/吃的/方块随身留")
        raw = self._raw_food(state)
        ores = self._raw_ores(state)
        if (raw or ores) and self._furnace_available(state):
            item, n = (raw or ores)[0]
            return _act("skill", "smelt", {"item": item, "count": n},
                        "夜里在家烧矿/烤肉：把今天的粗矿和生肉处理掉，明早就能做装备")
        if self._item(state, M + "coal", M + "charcoal") and self._item(state, M + "stick") \
                and self._torches(state) < 32:
            return _act("skill", "craft", {"item": M + "torch", "count": 16},
                        "夜里补火把：下矿和屋外照明都要用")
        return _act("wait", "night_work", {"until": "dawn"},
                    "夜里在家合成、整理、等天亮；不外出探索、不挖新矿道")

    # --- 6. home: where and when --------------------------------------------------------------------

    def home_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pick a base site, build the 5x5 starter hut, and know when a veteran heads home."""
        if state.get("skip_home"):
            return None
        home = _pos(state.get("home"))
        if home is None:
            route = state.get("home_route") or {}
            if (_num(route.get("camp_retry_after"), 0) > _num(state.get("t"), time.time())
                    or any(e["distance"] <= 16 for e in self.hostiles(state))
                    or _num(state.get("health"), 20) < 8):
                return None
            if self._tool_tier(state, "pickaxe") >= 1:
                return _act("home", "set_home", {
                    "size": 5, "hut": HUT_STEPS, "later": HUT_EXPAND,
                    "criteria": ["平地", "靠水", "附近有树", "背靠山", "离出生点不远",
                                 "避开刷怪笼/大矿洞/岩浆"]},
                    "有石镐了：该安家。选平地、靠水靠树、背靠山，避开刷怪笼/岩浆/大矿洞，按 5x5 起步屋清单盖")
            return None
        me = _pos(state.get("pos"))
        distance = _dist(me, home)
        hostiles = self.hostiles(state)
        nearest = hostiles[0]["distance"] if hostiles else 99.0
        reasons: List[str] = []
        if distance > self.RETURN_HOME_DIST:
            reasons.append("离家太远")
        if self._night(state) and distance > self.AT_HOME - 8:
            reasons.append("天黑了")
        if (nearest <= 12 or self._free_slots(state) < self.LOW_SLOTS
                or _num(state.get("health"), 20.0) <= 10) and distance > self.AT_HOME - 8:
            reasons.append("家里更安全")
        if reasons:
            return _act("home", "go_home", {"target": {"x": home[0], "y": home[1], "z": home[2]}},
                        "回家：" + "、".join(reasons) + "（一次只做一件事，先回落脚点）")
        if distance <= self.AT_HOME:
            missing = self._hut_missing(state)
            if missing:
                supply = self.house_supply_plan(state)
                if supply:
                    return supply
                if self._building_blocks(state) < 8 and any(s in ("walls", "roof") for s in missing):
                    return _act("skill", "mine", {"blocks": ",".join([M + "stone", M + "cobblestone"]),
                                                  "count": 16},
                                "盖屋要方块：先挖 16 个圆石（顺手挖煤做火把）")
                return _act("home", "build_hut", {"size": 5, "missing": missing, "hut": HUT_STEPS,
                                                  "later": HUT_EXPAND},
                            "起步屋还差：" + "、".join(missing))
        return None

    def house_supply_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use missing physical blueprint cells, never inventory to infer that a house is built."""
        observed = state.get("house_requirements") or {}
        if not observed.get("observed"):
            return None
        needs = dict(observed.get("materials_needed") or {})
        if not (state.get("hut") or {}).get("roof"):
            needs[M + "cobblestone"] = int(needs.get(M + "cobblestone", 0)) + 11  # exterior work stair
        for item, missing in needs.items():
            target = min(64, max(0, int(missing)))
            if target and self._item(state, item) < target and item not in self.shelved:
                return _act("goal", item, {"count": target, "by": "auto"},
                            "房屋实测还缺 %s ×%d，先补本轮建材/家具，再从原蓝图续建" % (human(item), target))
        return None

    def _hut_missing(self, state: Dict[str, Any]) -> List[str]:
        """Which starter-hut stages are not done yet (from `state["hut"]` if given, else from the bag)."""
        built = state.get("hut")
        if isinstance(built, dict):
            return [stage for stage in HUT_ORDER if not built.get(stage)]
        # Carrying building materials does not mean they were placed. The live integration supplies
        # Skills.hut_status(); without that observation the enclosure is still unverified.
        missing: List[str] = ["walls", "roof"]
        if not _sum(self._have(state), lambda s, _i: s.endswith("_door")) and not state.get("door"):
            missing.insert(1, "door")
        if self._torches(state) < 4:
            missing.append("light")
        bed = bool(state.get("bed")) or bool(self._item(state, M + "white_bed", M + "red_bed"))
        if not (bed and self._item(state, M + "chest") and self._item(state, M + "crafting_table")
                and self._item(state, M + "furnace")):
            missing.append("furniture")
        if self._torches(state) < 12:
            missing.append("safety")
        return missing

    # --- 7. bag, tools, mining ----------------------------------------------------------------------

    def inventory_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Bag almost full: store at a chest (home first), then declutter - in that order."""
        free = self._free_slots(state)
        if free >= self.LOW_SLOTS:
            return None
        home = _pos(state.get("home"))
        me = _pos(state.get("pos"))
        distance_home = _dist(me, home)
        at_home = (home is not None and me is not None and distance_home <= 8
                   and abs(me[1] - home[1]) <= 2.5)
        if state.get("chest_near") and (at_home or home is None):
            return _act("skill", "store", {},
                        "背包只剩 %d 格：先到箱子把矿、木头、多余方块存起来（火把/吃的/方块随身留）" % free)
        if at_home:
            if self._item(state, M + "chest"):
                return _act("skill", "store", {}, "家里没有箱子：把随身箱子放下，再实际存放物品")
            if free < 1:
                return _act("skill", "declutter", {"want_free": 2}, "家里缺箱子且背包已满：先腾出合成箱子的空间")
            return _act("goal", M + "chest", {"count": 1}, "家里的箱子缺失：先取得一个箱子，不能反复空存储")
        if home is not None:
            return _act("home", "go_home", {"target": {"x": home[0], "y": home[1], "z": home[2]},
                                            "for": "store"},
                        "背包只剩 %d 格：先回家存，装不下挖到的好东西就白挖了" % free)
        return _act("skill", "declutter", {"want_free": 6},
                    "背包只剩 %d 格又没有箱子：先扔杂物（泥土、树叶、树苗、多余圆石），工具/食物/火把留着"
                    % free)

    def tool_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Replace a tool before it breaks; upgrade a wooden pickaxe as soon as stone is in the bag."""
        for kind in ("pickaxe", "axe", "sword"):
            left = self._wear(state, kind)
            if left is not None and left <= self.WEAR_LOW and self._tool_count(state, kind) <= 1:
                material = self._tool_material(state)
                item = M + material + "_" + kind
                if item in self.shelved:
                    continue
                return _act("skill", "craft", {"item": item, "count": 1},
                            "%s 只剩 %d 耐久：先做新的，别让它断在矿洞/怪堆里" % (kind, left))
        pickaxe = self._tool_tier(state, "pickaxe")
        if pickaxe == 0 and self._item(state, M + "cobblestone", M + "cobbled_deepslate") >= 3 \
                and M + "stone_pickaxe" not in self.shelved:
            return _act("goal", M + "stone_pickaxe", {"count": 1, "by": "ladder"},
                        "还是木镐：有圆石了马上换石镐（铁、铜、青金石都要石镐以上才掉）")
        if pickaxe == 1 and self._item(state, M + "iron_ingot") >= 3 \
                and M + "iron_pickaxe" not in self.shelved:
            return _act("goal", M + "iron_pickaxe", {"count": 1, "by": "ladder"},
                        "有 3 个铁锭：先做铁镐（金、钻石、红石、绿宝石都要铁镐才挖得动）")
        if self._tool_tier(state, "sword") < 1 and self._item(state, M + "cobblestone") >= 2 \
                and self._night(state) and M + "stone_sword" not in self.shelved:
            return _act("goal", M + "stone_sword", {"count": 1, "by": "ladder"},
                        "天要黑了还没有石剑：先做一把，夜里保命用")
        return None

    def _tool_material(self, state: Dict[str, Any]) -> str:
        if self._item(state, M + "iron_ingot") >= 3:
            return "iron"
        if self._item(state, M + "cobblestone", M + "cobbled_deepslate", M + "stone") >= 3:
            return "stone"
        return "wooden"

    def _ore_of(self, target: Any) -> Optional[str]:
        """A goal/item/block name -> the ore it means, or None ('diamond_pickaxe' is not an ore)."""
        text = str(target or "")
        if not text:
            return None
        if text in ORE_Y:
            return text
        return ORE_ALIAS.get(short(text))

    def mining_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Branch mining the way a veteran does it: right y per ore, 3-block branch spacing, torches, spare pick."""
        target = state.get("mine_target") or state.get("goal") or ""
        visible = self.visible_ore_plan(state)
        if visible:
            return visible
        if (state.get("work_status") or {}).get("needs_supply"):
            return _act("skill", "go_home" if state.get("home") else "dig_up",
                        {} if state.get("home") else {"until_sky": True, "max_steps": 24},
                        "本轮采掘预算或补给触底，先沿安全路线撤回，不开新矿道")
        ore = self._ore_of(target)
        if ore is None:
            blocks = ",".join(ORE_BLOCKS[M + "coal"] + ORE_BLOCKS[M + "raw_iron"] + [M + "stone"])
            return _act("skill", "mine", {"blocks": blocks, "count": 8},
                        "没有指定矿：先挖眼前看得见的煤和铁（煤到处都有，铁在 y≈16）")
        need = ORE_PICK.get(ore, 0)
        tier = self._tool_tier(state, "pickaxe")
        if tier < need:
            pick = PICK_OF_TIER[need]
            if pick in self.shelved:
                return None
            return _act("goal", pick, {"count": 1, "by": "ladder"},
                        "挖 %s 要%s镐以上（不然挖了也不掉）：先做 %s"
                        % (human(ore), TIER_LABEL[need], human(pick)))
        if self._torches(state) < 16:
            if self._item(state, M + "coal", M + "charcoal") and self._item(state, M + "stick"):
                return _act("skill", "craft", {"item": M + "torch", "count": 16},
                            "下矿至少 16 个火把：边走边插，别摸黑（光照 0 就刷怪）")
            return _act("skill", "mine", {"blocks": ",".join(ORE_BLOCKS[M + "coal"]), "count": 8},
                        "没火把：先挖煤（煤到处都有），再做火把")
        spot = _pos(state.get("pos"))
        y = self._y(state) if state.get("y") is not None or spot else 64
        layers = ORE_Y.get(ore)
        blocks = ",".join(ORE_BLOCKS.get(ore) or [])
        if state.get("cave_hint") or state.get("open_passages"):
            return _act("skill", "explore_cave", {"max_seconds": 75, "max_nodes": 12, "ores": blocks},
                        "先检查已观察到的洞穴入口，限时探索；没有可走路径就停止而不盲挖")
        if layers is None:
            return _act("skill", "mine", {"blocks": blocks, "count": 8},
                        "%s 没有固定的高度层：见到就挖，别为它专门下一趟矿" % human(ore))
        if y > layers + 4:
            return _act("skill", "dig_down", {"to_y": int(layers), "stairs": True},
                        "挖楼梯下到 y≈%d 找 %s：往前一格往下一格，别垂直往脚下挖（会掉洞或岩浆）"
                        % (layers, human(ore)))
        if y < layers - 2:
            return _act("skill", "dig_up", {"to_y": int(layers)},
                        "人在 y=%d，比 %s 的矿层低：挖楼梯上去到 y≈%d" % (y, human(ore), layers))
        return _act("skill", "tunnel", {"length": 24, "ores": blocks, "branch": 24, "spacing": 3,
                                        "torch_every": 8, "y": int(layers)},
                    "在 y≈%d 鱼骨挖矿：主道 2 格高，每 3 格开一条支道、每条 20~30 格；每 8~10 格插火把；"
                    "听到岩浆声立刻停，先堵再绕" % layers)

    # --- 8. which goal: quest vs ladder, by expected value ------------------------------------------

    def early_game_checklist(self, inventory: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """The speedrun order (wood -> table -> pick -> stone -> furnace -> food -> iron -> diamond), ticked."""
        have = {}
        for item, n in (inventory or {}).items():
            try:
                n = int(n or 0)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                have[str(item)] = n
        out: List[Dict[str, Any]] = []
        for order, (item, count, stage, why) in enumerate(EARLY_GAME, 1):
            if item == M + "oak_log":
                done = _sum(have, lambda s, _i: s.endswith(("_log", "_wood"))) >= count
            elif item == M + "bread":
                done = _sum(have, lambda s, i: is_edible(i)) >= count
            elif item == M + "torch":
                done = _count(have, item) >= count
            else:
                done = _count(have, item) >= count
            out.append({"order": order, "item": item, "count": count, "stage": stage,
                        "done": bool(done), "why": why})
        return out

    def _wood_starved(self, state: Dict[str, Any]) -> bool:
        return self._tool_tier(state, "pickaxe") <= 0 and self._wood(state) < 6

    def choose_goal(self, state: Dict[str, Any], quest_goals: Optional[Iterable[Any]] = None,
                    ladder_goals: Optional[Iterable[Any]] = None) -> Optional[Dict[str, Any]]:
        """Pick what to work on now: basics first, then quest vs ladder by (value / cost), with anti-thrash.

        Returns {"item","count","by":"quest"|"ladder","why","score",("quest")} or None.
        """
        t = _num(state.get("t"), time.time())
        self._prune(t)
        skip = self._skip(state, t)
        have = self._have(state)
        if quest_goals is None and ladder_goals is None and self._wood_starved(state):
            choice = {"item": M + "oak_log", "count": 6, "by": "ladder", "basic": True, "score": 999.0,
                      "why": "开局第一件事：砍 1~2 棵树（FallingTree 砍树干最下面一块，整棵倒）"}
            self._remember_choice(choice, t)
            return choice
        if ladder_goals is not None:
            ladder = [g for g in self._normalize(ladder_goals) if g["item"] not in skip]
        else:
            ladder = [g for g in self._ladder_goals(state, have, skip) if g["item"] not in skip]
        for goal in ladder:
            goal["basic"] = bool(goal.get("basic", goal["item"] in LADDER_BASICS))
        basics = [g for g in ladder if g["basic"]]
        if basics:
            choice = self._ladder_choice(basics[0], have)
            choice["score"] = self._ladder_value(choice, have)
            self._remember_choice(choice, t)
            return choice
        if quest_goals is None:
            quests = self._quest_goals(state, have, skip)
        else:
            quests = [q for q in (quest_goals or ()) if isinstance(q, dict) and q.get("item")
                      and str(q.get("item")) not in skip and q.get("ready") is not False
                      and _num(q.get("cost"), 0.0) <= self.QUEST_MAX_COST]
        candidates: List[Dict[str, Any]] = []
        for quest in quests:
            score = self._quest_value(quest, have)
            if score <= 0:
                continue
            candidates.append({
                "item": str(quest["item"]), "count": max(1, int(_num(quest.get("count"), 1))), "by": "quest",
                "quest": quest, "score": score,
                "why": "任务书：%s（%s）——%s" % (quest.get("title") or human(quest["item"]),
                                                quest.get("chapter") or "?", quest.get("reason") or "现在最划算"),
            })
        for goal in ladder:
            choice = self._ladder_choice(goal, have)
            choice["score"] = self._ladder_value(goal, have)
            candidates.append(choice)
        if not candidates:
            return self._checklist_goal(state, have, skip, t)
        candidates.sort(key=lambda c: (c["score"], 1 if c.get("by") == "quest" else 0), reverse=True)
        best = candidates[0]
        previous = self._chosen
        if (previous and previous.get("item") and previous["item"] != best["item"]
                and previous["item"] not in skip and t - self._chosen_at < self.SWITCH_HOLD
                and _num(previous.get("score"), 0.0) >= best["score"] * 0.8):
            return dict(previous, why=str(previous.get("why") or "") + "（刚定下的目标，先做完再换）")
        self._remember_choice(best, t)
        return best

    def _checklist_goal(self, state: Dict[str, Any], have: Dict[str, int], skip: set,
                        t: float) -> Optional[Dict[str, Any]]:
        """No index and no ladder given: follow the speedrun checklist by hand (pure fallback)."""
        for step in self.early_game_checklist(have):
            if step["done"] or step["item"] in skip:
                continue
            choice = {"item": step["item"], "count": step["count"], "by": "ladder", "score": 1.0,
                      "basic": step["item"] in LADDER_BASICS,
                      "why": "开局第 %d 步（%s）：%s" % (step["order"], step["stage"], step["why"])}
            self._remember_choice(choice, t)
            return choice
        return None

    @staticmethod
    def _normalize(goals: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for goal in goals or ():
            if isinstance(goal, dict):
                item = goal.get("item") or goal.get("id")
                if not item:
                    continue
                out.append(dict(goal, item=str(item), count=max(1, int(_num(goal.get("count"), 1)))))
            elif isinstance(goal, (list, tuple)) and goal:
                out.append({"item": str(goal[0]), "count": max(1, int(_num(goal[1] if len(goal) > 1 else 1, 1))),
                            "by": "ladder"})
            elif isinstance(goal, str) and goal:
                out.append({"item": goal, "count": 1, "by": "ladder"})
        return out

    def _ladder_goals(self, state: Dict[str, Any], have: Dict[str, int], skip: set) -> List[Dict[str, Any]]:
        """The next few rungs of mc_goal's ladder (asked lazily so this module has no import-time coupling)."""
        # A furnace or table standing next to her is one she has: the goal executor counts it as done (Planner
        # treats a nearby station as owned), so asking for it again only made "目标完成：熔炉" every few seconds.
        placed = set(state.get("stations") or ())
        try:
            from src.levels.l8_agency.mc_goal import LADDER, satisfied
            worn = set(state.get("worn") or ())
            out: List[Dict[str, Any]] = []
            for item, count in LADDER:
                if item in skip or (item in placed and count <= 1):
                    continue
                try:
                    if satisfied(item, count, have, worn):
                        continue
                except Exception:
                    pass
                out.append({"item": item, "count": int(count), "by": "ladder",
                            "basic": item in LADDER_BASICS})
                if len(out) >= 4:
                    break
            return out
        except Exception as e:                       # standalone/tests: fall back to the local table
            logger.debug(f"[Pro] no mc_goal ladder: {e}")
        out = []
        for item, count in LADDER_ORDER:
            if item in skip or (item in placed and count <= 1) or self._satisfied(state, item, count, have):
                continue
            out.append({"item": item, "count": int(count), "by": "ladder", "basic": item in LADDER_BASICS})
            if len(out) >= 4:
                break
        return out

    def _satisfied(self, state: Dict[str, Any], item: str, count: int, have: Dict[str, int]) -> bool:
        """Local stand-in for mc_goal.satisfied (only used when mc_goal cannot be imported)."""
        if _count(have, item) >= count:
            return True
        kind = short(item).rsplit("_", 1)[-1]
        if kind in TOOL_KINDS:
            return self._tool_tier(state, kind) >= tool_tier(item)
        if kind in ARMOR_KINDS:
            return any(short(i).endswith("_" + kind) and tool_tier(i) >= tool_tier(item)
                       for i in have)
        return False

    def _quest_goals(self, state: Dict[str, Any], have: Dict[str, int], skip: set) -> List[Dict[str, Any]]:
        raw = state.get("quest_goals")
        if raw is None and self.quests is not None:
            try:
                raw = self.quests.next_goals(4, have, skip=skip, stations=state.get("stations") or ())
            except Exception as e:
                logger.debug(f"[Pro] quest book unavailable: {e}")
                raw = []
        out: List[Dict[str, Any]] = []
        for quest in raw or ():
            if not isinstance(quest, dict) or not quest.get("item"):
                continue
            if str(quest["item"]) in skip:
                continue
            if quest.get("ready") is False or _num(quest.get("cost"), 0.0) > self.QUEST_MAX_COST:
                continue
            out.append(quest)
        return out

    def _quest_value(self, quest: Dict[str, Any], have: Dict[str, int]) -> float:
        """Expected value: reward / (cost + walking), boosted by early chapters and by items already in the bag."""
        cost = max(0.0, _num(quest.get("cost"), 0.0))
        items = quest.get("items") or [(quest.get("item"), quest.get("count") or 1)]
        partial, total = 0.0, 0.0
        for entry in items:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                item, need = entry[0], _num(entry[1], 1.0)
            elif isinstance(entry, dict):
                item, need = entry.get("item"), _num(entry.get("count"), 1.0)
            else:
                continue
            if not item:
                continue
            total += 1.0
            partial += min(1.0, _count(have, item) / max(1.0, need))
        partial = partial / total if total else 0.0
        early = 20.0 if self._early_chapter(quest.get("chapter")) else 0.0
        return (70.0 + early + 15.0 * partial) / (cost + 8.0)

    @staticmethod
    def _early_chapter(chapter: Any) -> bool:
        title = str(chapter or "").lower()
        return any(hint in title for hint in ("tutorial", "overworld", "教程", "主世界", "开局"))

    def _ladder_value(self, goal: Dict[str, Any], have: Dict[str, int]) -> float:
        item = str(goal["item"])
        count = max(1, int(_num(goal.get("count"), 1)))
        basic = bool(goal.get("basic", item in LADDER_BASICS))
        value = 120.0 if basic else 60.0
        partial = min(1.0, _count(have, item) / float(count))
        position = next((i for i, (known, _n) in enumerate(LADDER_ORDER) if known == item), 8)
        cost = 4.0 + 3.0 * position
        return (value + 15.0 * partial) / (cost + 8.0)

    def _ladder_choice(self, goal: Dict[str, Any], _have: Dict[str, int]) -> Dict[str, Any]:
        item = str(goal["item"])
        count = max(1, int(_num(goal.get("count"), 1)))
        return {"item": item, "count": count, "by": "ladder",
                "basic": bool(goal.get("basic", item in LADDER_BASICS)),
                "why": "常规进度：%s ×%d%s" % (self.name(item), count,
                                              "（基础没齐，先补基础）" if item in LADDER_BASICS else "")}

    # --- 9. boss prep (a checklist, not a decision) -------------------------------------------------

    def boss_prep_checklist(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """What a veteran refuses to fight a boss without: ticked against the bag and the tools."""
        have = self._have(state)
        owned = set(have) | set(state.get("worn") or ())
        armored = all(any(i.endswith("_" + kind) and tool_tier(i) >= 2 for i in owned)
                      for kind in ARMOR_KINDS)
        checks = [
            armored,
            self._tool_tier(state, "sword") >= 2,
            self._has_shield(state),
            self._item(state, M + "bow") >= 1 and self._item(state, M + "arrow") >= 32,
            self._cooked_count(state) >= 8,
            self._building_blocks(state) >= 32,
            self._item(state, M + "water_bucket") >= 1,
            bool(self._potion_count(state)) or self._item(state, M + "totem_of_undying") >= 1,
        ]
        return [{"need": need, "done": bool(done), "why": why}
                for (need, why), done in zip(BOSS_CHECKLIST, checks)]

    # --- the one decision ---------------------------------------------------------------------------

    def expedition_plan(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A resumable diamond -> Nether supplies -> Eyes chain, driven only by observed state.

        Stronghold navigation and mod-specific boss mechanics remain separate objectives; an item
        checklist does not certify a dimension, boss kill, quest reward or pack completion.
        """
        if state.get("goal_by") == "self" or self._tool_tier(state, "pickaxe") < 3:
            return None
        have = self._have(state)
        worn = set(state.get("worn") or ())
        dim = str(state.get("dimension") or "")
        if dim == "minecraft:the_end":
            return _act("skill", "end_step", {}, "根据实际末地现场先处理水晶和龙息，分步弓箭战斗并核实返程门")
        if dim not in ("minecraft:overworld", "minecraft:the_nether"):
            return None
        eyes = _count(have, M + "ender_eye")
        if _num(state.get("health"), 20) < 16 or _num(state.get("food"), 20) < 16:
            return None                       # the survival plan restores these first
        if dim == "minecraft:the_nether" and (
                sum(n for i, n in self._food_items(state) if short(i) not in RAW_FOOD) < 8
                or self._building_blocks(state) < 16):
            return (_act("skill", "enter_portal", {}, "下界补给不足，优先回主世界补给")
                    if state.get("portal") else None)
        for kind in ("chestplate", "leggings", "helmet", "boots"):
            if not any(i.endswith("_" + kind) and tool_tier(i) >= 2 for i in set(have) | worn):
                return _act("goal", M + "iron_" + kind, {"count": 1, "by": "expedition"},
                            "下界远征前补齐铁甲，再考虑找要塞")
        if not self._has_shield(state):
            return _act("goal", M + "shield", {"count": 1, "by": "expedition"}, "远征前准备盾牌")
        if dim == "minecraft:overworld" and sum(n for i, n in self._food_items(state) if short(i) not in RAW_FOOD) < 16:
            return _act("skill", "hunt", {"count": 4}, "远征带足16份熟食，先补给再离家")
        if dim == "minecraft:overworld" and self._building_blocks(state) < 64:
            return _act("skill", "mine", {"blocks": M + "stone," + M + "cobblestone", "count": 32},
                        "远征携带一组防爆方块，走过的路径随时能封挡")
        if dim == "minecraft:overworld" and self._torches(state) < 32:
            return _act("goal", M + "torch", {"count": 32, "by": "expedition"}, "准备火把标记去路和返程路")
        if dim == "minecraft:overworld" and (eyes >= 16 or state.get("end_portal") or state.get("stronghold")
                                             or (eyes > 12 and state.get("end_route"))):
            if not _count(have, M + "bow"):
                return _act("goal", M + "bow", {"count": 1, "by": "expedition"}, "准备末地水晶和飞行龙阶段的远程武器")
            if _count(have, M + "arrow") < 128:
                return _act("goal", M + "arrow", {"count": 128, "by": "expedition"}, "末地出发前带两组箭，给水晶和移动目标留余量")
            if not _count(have, M + "water_bucket"):
                return _act("skill", "fill_water_bucket", {}, "取到真实水桶用于末地避险，空桶不算准备完成")
            return _act("skill", "end_step", {}, "跟随真实末影之眼轨迹找要塞，观察门框后填眼，逐步核实末地现场")
        powder = _count(have, M + "blaze_powder") + 2 * _count(have, M + "blaze_rod")
        if dim == "minecraft:the_nether":
            if powder + eyes >= 16:
                return (_act("skill", "enter_portal", {}, "烈焰粉原料已够，沿已验证返程门回主世界")
                        if state.get("portal") else None)
            return _act("skill", "hunt_resource", {"item": M + "blaze_rod", "count": max(1, math.ceil((16 - powder - eyes) / 2))},
                        "在下界寻找可达烈焰人获取烈焰棒；未发现目标就交给地图与视野勘察，不冒称已找到要塞")
        if powder + eyes < 16:
            if not state.get("home"):
                return _act("home", "set_home", {}, "下界远征先留主世界落脚点和返程坐标")
            if not any(i.startswith(M + "golden_") and i.endswith(ARMOR_KINDS) for i in set(have) | worn):
                return _act("goal", M + "golden_helmet", {"count": 1, "by": "expedition"}, "准备金头盔，降低普通猪灵主动攻击风险")
            if not state.get("portal"):
                if _count(have, M + "obsidian") < 10:
                    return _act("goal", M + "obsidian", {"count": 10, "by": "expedition"}, "用钻石镐获取10块黑曜石，保留建门现场进度")
                if not _count(have, M + "flint_and_steel"):
                    return _act("goal", M + "flint_and_steel", {"count": 1, "by": "expedition"}, "合成打火石，生存方式点燃门框")
                return _act("skill", "build_portal", {}, "在家旁安全平地续建并逐块核实下界门")
            return _act("skill", "enter_portal", {}, "补给准备好了，走入已观测到的门并核实维度变化")
        pearls = _count(have, M + "ender_pearl")
        if pearls + eyes < 16:
            return _act("skill", "hunt_resource", {"item": M + "ender_pearl", "count": 16 - pearls - eyes},
                        "收集末影珍珠，只按实际捡到的数量推进；低血量立即停手")
        return _act("goal", M + "ender_eye", {"count": 16, "by": "expedition"},
                    "用真实材料合成16颗末影之眼；随后仍需实地找要塞和门框，不把准备完成当通关")

    @staticmethod
    def _mineable_now(ore, me):
        """The same test Skills.mine applies (a face open to the air, within 16 blocks, within 6 up or down): an
        ore the scan saw behind rock is not 'in sight', and planning it made mine give up every few seconds."""
        if ore.get("exposed") is False:
            return False
        if me is None:
            return True
        return (abs(ore.get("y", me[1]) - me[1]) <= 6
                and math.dist((ore.get("x", me[0]), ore.get("y", me[1]), ore.get("z", me[2])), me) <= 16)

    def visible_ore_plan(self, state):
        me = _pos(state.get("pos"))
        me = tuple(float(v) for v in me) if me else None
        tier = self._tool_tier(state, "pickaxe")
        have = self._have(state)
        for ore in (M + "diamond", M + "raw_iron", M + "coal", M + "raw_gold", M + "redstone", M + "lapis_lazuli"):
            if tier < ORE_PICK[ore] or have.get(ore, 0) >= (16 if ore == M + "diamond" else 32):
                continue
            found = [p for p in state.get("visible_ores", []) if p.get("id", p.get("block")) in ORE_BLOCKS[ore]
                     and self._mineable_now(p, me)]
            if found:
                return _act("skill", "mine", {"blocks": ",".join(ORE_BLOCKS[ore]), "count": min(8, len(found)), "radius": 1},
                            "先收集附近已观察到且工具可采的矿石，避免路过却继续盲挖")
        return None

    def next_action(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The single next thing to do, or None (nothing worth doing - do not wander)."""
        state = dict(state or {})
        if (state.get("home_route") or {}).get("blocked"):
            # Preserve the remembered marker, but do not let every home/night/inventory branch
            # force another failed trip. A safe replacement is registered only by set_home.
            state["home"] = None
            state.pop("hut", None)
        t = _num(state.get("t"), time.time())
        self._prune(t)
        self._skip(state, t)                         # also shelves a goal that just got stuck
        if not state.get("in_world", True):
            return _act("wait", "in_world", {"seconds": 2.0},
                        "客户端不在世界里（加载/断开/死亡界面）：等它回来，别乱按键")
        if state.get("dead"):
            return _act("wait", "respawn", {"seconds": 3.0},
                        "死亡界面：等重生；重生后先回死亡点捡东西")
        action = self.hazard_plan(state)
        if action:
            return action
        fight = self.combat_plan(state)
        if fight and str(fight["args"].get("mode")) in ("flee", "retreat", "shield", "heal"):
            return fight
        if state.get("busy") or state.get("job_running"):
            return None                              # she is doing something: never interrupt for nothing
        action = self.death_plan(state)
        if action:
            return action
        action = self.survival_plan(state)
        if action:
            return action
        if _pos(state.get("home")) is None and not self._night(state):
            # no usable home yet: settle before quests send her down a mine (a player's first-day order);
            # at night the shelter comes first (night_plan below)
            action = self.home_plan(state)
            if action:
                return action
        if (state.get("quests_ui") and self.quests is not None and state.get("goal_by") != "self"
                and _num(state.get("health"), 0) >= 14 and _num(state.get("food"), 0) >= 14 and not state.get("hostiles")):
            actions = self.quests.next_actions(state, capabilities={"quests_ui"}, limit=1)
            if actions:
                return actions[0]
        action = self.night_plan(state)
        if action:
            return action
        light = state.get("light") or {}
        if self._torches(state) and light.get("blockLight", 15) <= 3 and (not state.get("sky_above", True) or self._night(state)):
            return _act("skill", "light_area", {"force": True}, "实际光照偏低，先点亮落脚点再推进目标")
        visible = self.visible_ore_plan(state)
        if visible:
            return visible
        action = self.home_plan(state)
        if action:
            return action
        action = self.inventory_plan(state)
        if action:
            return action
        if fight and str(fight["args"].get("mode")) == "engage" \
                and _num(fight["args"].get("distance"), 99.0) <= 8:
            return fight                             # it is already in her face: fight before starting a job
        action = self.tool_plan(state)
        if action:
            return action
        suggestion = state.get("suggestion")
        if suggestion and state.get("goal_by") != "self" and _num(state.get("health"), 20) >= 14 and _num(state.get("food"), 20) >= 14:
            action = suggestion["action"]
            if action["name"] != "quest":
                return dict(action)
        action = self.visible_ore_plan(state)
        if action:
            return action
        action = self.expedition_plan(state)
        if action:
            return action
        if (self.quests is not None and hasattr(self.quests, "next_actions")
                and state.get("goal_by") != "self" and self._tool_tier(state, "pickaxe") >= 3
                and _num(state.get("health"), 0) >= 16 and _num(state.get("food"), 0) >= 16):
            # These are known-position/dimension actions, not guessed boss kills or fabricated
            # structure coordinates. The expedition preflight above retains priority.
            try:
                actions = self.quests.next_actions(state, capabilities={"enter_portal", "go_to"})
                if actions:
                    return actions[0]
            except Exception:
                logger.debug("quest action planning unavailable", exc_info=True)
        if state.get("mine_target") or state.get("task") == "mine" or state.get("mode") == "mine":
            action = self.mining_plan(state)
            if action:
                return action
        choice = self.choose_goal(state)
        if not choice:
            return None
        return self._goal_action(choice)

    def _goal_action(self, choice: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        item = str(choice.get("item") or "")
        if not item or item in self.shelved:
            return None
        why = choice.get("why") or ("朝 %s 做" % self.name(item))
        if short(item).endswith(("_log", "_wood")):
            return _act("skill", "chop_tree",
                        {"max_logs": min(10, max(1, int(_num(choice.get("count"), 6)))), "prefer": item}, why)
        return _act("goal", item, {"count": max(1, int(_num(choice.get("count"), 1))),
                                   "by": choice.get("by") or "auto"}, why)


__all__ = ["ProPlanner", "ORE_Y", "ORE_BLOCKS", "SKILL_ARGS", "EARLY_GAME", "HUT_STEPS", "BOSS_CHECKLIST",
           "DAY_PLAN"]
