"""
Level 8k: what a good player knows about this game and this modpack.

Two kinds of knowing:
  - the modpack itself, read from her own client once and cached: every item's name (Chinese where the pack has
    it), every recipe (crafting, smelting and the mods' own machines), item and block tags, which tool a block
    needs and how good it must be, what a block drops, and the pack's quests;
  - an experienced player's habits and know-how (mc_expert.md next to this file), looked up by topic.

Nothing here touches the game; the planner (mc_goal) and her mc_know tool read from it.
"""
import glob
import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

TIERS = ("wood", "stone", "iron", "diamond", "netherite")
TOOL_TIER = {"wooden": 0, "golden": 0, "stone": 1, "copper": 1, "iron": 2, "diamond": 3, "netherite": 4}
INDEX_VERSION = 4


# --- reading the pack ------------------------------------------------------------------------

def _ingredient(ing: Any) -> List[str]:
    """One recipe ingredient -> its alternatives: item ids, or "#tag"."""
    if ing is None:
        return []
    if isinstance(ing, str):
        return [ing]
    if isinstance(ing, list):
        out: List[str] = []
        for i in ing:
            out += _ingredient(i)
        return out
    if isinstance(ing, dict):
        if "item" in ing:
            return [str(ing["item"])]
        if "tag" in ing:
            return ["#" + str(ing["tag"])]
        if "id" in ing and isinstance(ing["id"], str):
            return [str(ing["id"])]
        for k in ("items", "children", "ingredients", "base"):        # neoforge compound ingredients
            if k in ing:
                return _ingredient(ing[k])
        if "ingredient" in ing:
            return _ingredient(ing["ingredient"])
    return []


def _result(res: Any) -> Tuple[Optional[str], int]:
    if isinstance(res, str):
        return res, 1
    if isinstance(res, dict):
        rid = res.get("id") or res.get("item")
        if isinstance(rid, dict):
            rid = rid.get("id") or rid.get("item")
        return (str(rid) if rid else None), int(res.get("count") or 1)
    if isinstance(res, list) and res:
        return _result(res[0])
    return None, 1


def parse_recipe(rid: str, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = str(doc.get("type", "")).split(":")[-1] if ":" in str(doc.get("type", "")) else str(doc.get("type", ""))
    ns = str(doc.get("type", "")).split(":")[0] if ":" in str(doc.get("type", "")) else "minecraft"
    out, n = _result(doc.get("result") if "result" in doc else doc.get("output") or doc.get("results"))
    if not out:
        return None
    cells: List[List[str]] = []
    if kind == "crafting_shaped" and isinstance(doc.get("pattern"), list):
        key = {k: _ingredient(v) for k, v in (doc.get("key") or {}).items()}
        for row in doc["pattern"]:
            for ch in str(row):
                if ch != " " and key.get(ch):
                    cells.append(key[ch])
    elif "ingredients" in doc and isinstance(doc["ingredients"], list):
        cells = [a for a in (_ingredient(i) for i in doc["ingredients"]) if a]
    elif "ingredient" in doc:
        cells = [a for a in [_ingredient(doc["ingredient"])] if a]
    elif kind.startswith("smithing"):
        cells = [a for a in (_ingredient(doc.get(k)) for k in ("template", "base", "addition")) if a]
    else:
        cells = [a for a in (_ingredient(v) for k, v in doc.items() if k.startswith(("input", "ingredient"))) if a]
    t = kind if ns == "minecraft" else f"{ns}:{kind}"
    r = {"ref": rid, "t": t, "out": out, "n": n, "in": cells}
    if kind == "crafting_shaped" and isinstance(doc.get("pattern"), list) and doc["pattern"]:
        r["w"], r["h"] = max(len(str(row)) for row in doc["pattern"]), len(doc["pattern"])
    return r


def _loot_items(node: Any, acc: List[str]):
    if isinstance(node, dict):
        if str(node.get("type", "")).endswith("item") and isinstance(node.get("name"), str):
            acc.append(node["name"])
        for v in node.values():
            _loot_items(v, acc)
    elif isinstance(node, list):
        for v in node:
            _loot_items(v, acc)


_SNBT_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]:,]|[^\s{}\[\]:,"]+')


def parse_snbt(text: str) -> Any:
    """Just enough SNBT for FTB Quests files (newline-separated, bare keys, suffixed numbers)."""
    tokens = _SNBT_TOKEN.findall(text)
    pos = 0

    def atom(tok: str) -> Any:
        if tok.startswith('"'):
            return json.loads(tok)
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)[bBsSlLfFdD]?", tok)
        if m:
            return float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        return {"true": True, "false": False}.get(tok, tok)

    def value() -> Any:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "{":
            obj = {}
            while tokens[pos] != "}":
                if tokens[pos] == ",":
                    pos += 1
                    continue
                key = atom(tokens[pos])
                pos += 2                                  # key, ':'
                obj[str(key)] = value()
            pos += 1
            return obj
        if tok == "[":
            arr = []
            while tokens[pos] != "]":
                if tokens[pos] == ",":
                    pos += 1
                    continue
                arr.append(value())
            pos += 1
            return arr
        return atom(tok)
    return value()


def _strip_codes(s: str) -> str:
    return re.sub(r"&[0-9a-fk-orl]", "", str(s or "")).strip()


def find_assets(game_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """The launcher's shared assets folder and this version's asset index name (vanilla Chinese lives there)."""
    index = None
    for js in glob.glob(os.path.join(glob.escape(game_dir), "*.json")):
        try:
            with open(js, encoding="utf-8") as f:
                index = str(json.load(f).get("assets") or "") or index
        except (OSError, ValueError):
            pass
    d = game_dir
    for _ in range(4):
        d = os.path.dirname(d)
        if os.path.isdir(os.path.join(d, "assets", "indexes")):
            return os.path.join(d, "assets"), index
    return None, index


class ModpackIndex:
    def __init__(self, game_dir: Optional[str], cache: Optional[str] = None):
        self.game_dir = game_dir
        self.cache = cache
        self.names: Dict[str, str] = {}           # id -> Chinese name (or English when there is none)
        self.recipes: List[Dict[str, Any]] = []
        self.item_tags: Dict[str, List[str]] = {}
        self.block_tags: Dict[str, List[str]] = {}
        self.drops: Dict[str, List[str]] = {}      # block -> items it drops (without silk touch first)
        self.quests: List[Dict[str, Any]] = []
        self.loaded = False
        self._by_out: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._tag_cache: Dict[Tuple[str, str], Set[str]] = {}

    # building --------------------------------------------------------------------------------

    def load(self) -> "ModpackIndex":
        if self.loaded:
            return self
        self.loaded = True
        if self.cache and os.path.exists(self.cache):
            try:
                with open(self.cache, encoding="utf-8") as f:
                    doc = json.load(f)
                if doc.get("version") == INDEX_VERSION:
                    for k in ("names", "recipes", "item_tags", "block_tags", "drops", "quests"):
                        setattr(self, k, doc.get(k) or getattr(self, k))
                    return self
            except (OSError, ValueError):
                pass
        if self.game_dir and os.path.isdir(self.game_dir):
            self.build()
            if self.cache:
                try:
                    os.makedirs(os.path.dirname(self.cache), exist_ok=True)
                    with open(self.cache, "w", encoding="utf-8") as f:
                        json.dump({"version": INDEX_VERSION, "names": self.names, "recipes": self.recipes,
                                   "item_tags": self.item_tags, "block_tags": self.block_tags, "drops": self.drops,
                                   "quests": self.quests}, f, ensure_ascii=False)
                except OSError:
                    pass
        return self

    def build(self):
        root = glob.escape(self.game_dir)
        jars = glob.glob(os.path.join(root, "*.jar")) + glob.glob(os.path.join(root, "mods", "*.jar"))
        english: Dict[str, str] = {}
        for jar in jars:
            try:
                with zipfile.ZipFile(jar) as z:
                    for name in z.namelist():
                        if name.endswith(".json"):
                            self._read_entry(z, name, english)
            except (zipfile.BadZipFile, OSError):
                continue
        assets, index = find_assets(self.game_dir)
        if assets and index:
            try:
                with open(os.path.join(assets, "indexes", f"{index}.json"), encoding="utf-8") as f:
                    h = json.load(f)["objects"]["minecraft/lang/zh_cn.json"]["hash"]
                with open(os.path.join(assets, "objects", h[:2], h), encoding="utf-8") as f:
                    self._lang(json.load(f), self.names)
            except (OSError, ValueError, KeyError):
                pass
        for k, v in english.items():
            self.names.setdefault(k, v)
        self._read_quests()
        logger.info(f"[Know] modpack index: {len(self.names)} names, {len(self.recipes)} recipes, "
                    f"{len(self.drops)} block drops, {len(self.quests)} quests from {len(jars)} jars")

    @staticmethod
    def _lang(doc: Dict[str, str], into: Dict[str, str]):
        for key, val in doc.items():
            parts = key.split(".")
            if len(parts) == 3 and parts[0] in ("item", "block", "entity") and isinstance(val, str):
                into.setdefault(f"{parts[1]}:{parts[2]}", val)

    def _read_entry(self, z: zipfile.ZipFile, name: str, english: Dict[str, str]):
        parts = name.split("/")
        if len(parts) == 4 and parts[0] == "assets" and parts[2] == "lang" and parts[3] in ("zh_cn.json", "en_us.json"):
            try:
                doc = json.loads(z.read(name).decode("utf-8-sig"))
            except (ValueError, UnicodeDecodeError):
                return
            self._lang(doc, self.names if parts[3] == "zh_cn.json" else english)
            return
        if len(parts) < 4 or parts[0] != "data":
            return
        ns, kind = parts[1], parts[2]
        path = "/".join(parts[3:])[:-5]
        try:
            doc = json.loads(z.read(name).decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError, KeyError):
            return
        if not isinstance(doc, dict):
            return
        if kind in ("recipe", "recipes"):
            r = parse_recipe(f"{ns}:{path}", doc)
            if r:
                self.recipes.append(r)
        elif kind == "tags" and len(parts) >= 5:
            group = parts[3]
            tag = f"{ns}:{'/'.join(parts[4:])[:-5]}"
            vals = [v if isinstance(v, str) else v.get("id") for v in doc.get("values") or []]
            vals = [v for v in vals if v]
            target = self.item_tags if group in ("item", "items") else self.block_tags if group in ("block", "blocks") else None
            if target is not None:
                target[tag] = (vals if doc.get("replace") else target.get(tag, []) + vals)
        elif kind in ("loot_table", "loot_tables") and path.startswith("blocks/"):
            block = f"{ns}:{path[7:]}"
            items: List[str] = []
            _loot_items(doc, items)
            seen = list(dict.fromkeys(items))
            if seen:
                others = [i for i in seen if i != block]
                self.drops[block] = others + ([block] if block in seen else []) if others else seen

    def _read_quests(self):
        qdir = os.path.join(self.game_dir, "config", "ftbquests", "quests")
        lang: Dict[str, Any] = {}
        try:
            with open(os.path.join(qdir, "lang", "en_us.snbt"), encoding="utf-8") as f:
                lang = parse_snbt(f.read())
        except (OSError, IndexError, ValueError):
            pass
        for path in sorted(glob.glob(os.path.join(glob.escape(qdir), "chapters", "*.snbt"))):
            try:
                with open(path, encoding="utf-8") as f:
                    ch = parse_snbt(f.read())
            except (OSError, IndexError, ValueError):
                continue
            title = _strip_codes(lang.get(f"chapter.{ch.get('id')}.title") or ch.get("filename"))
            for q in ch.get("quests") or []:
                tasks = []
                for t in q.get("tasks") or []:
                    item = t.get("item")
                    if isinstance(item, dict):
                        flt = str((item.get("components") or {}).get("ftbfiltersystem:filter") or "")
                        m = re.search(r"item_tag\(([^)]+)\)", flt)
                        item = "#" + m.group(1) if m else item.get("id")
                    tasks.append({k: v for k, v in {"type": t.get("type"), "item": item, "count": t.get("count"),
                                                   "entity": t.get("entity"), "dimension": t.get("dimension"),
                                                   "structure": t.get("structure"), "biome": t.get("biome"),
                                                   "advancement": t.get("advancement")}.items() if v})
                self.quests.append({"id": q.get("id"), "chapter": title,
                                    "title": _strip_codes(lang.get(f"quest.{q.get('id')}.title") or ""),
                                    "text": _strip_codes(lang.get(f"quest.{q.get('id')}.quest_subtitle") or ""),
                                    "deps": q.get("dependencies") or [], "tasks": tasks})

    # asking ----------------------------------------------------------------------------------

    def name(self, item: str) -> str:
        self.load()
        return self.names.get(item) or item.split(":")[-1]

    def tag_members(self, tag: str, blocks: bool = False) -> Set[str]:
        tag = tag.lstrip("#")
        key = (tag, "b" if blocks else "i")
        if key in self._tag_cache:
            return self._tag_cache[key]
        self._tag_cache[key] = set()                  # (a tag that includes itself)
        out: Set[str] = set()
        for v in (self.block_tags if blocks else self.item_tags).get(tag, []):
            v = v.rstrip("?")
            out |= self.tag_members(v, blocks) if v.startswith("#") else {v}
        self._tag_cache[key] = out
        return out

    def expand(self, alts: Iterable[str]) -> List[str]:
        """Ingredient alternatives with tags opened up, vanilla first."""
        out: List[str] = []
        for a in alts:
            for i in (sorted(self.tag_members(a)) if a.startswith("#") else [a]):
                if i not in out:
                    out.append(i)
        return sorted(out, key=lambda i: (not i.startswith("minecraft:"), i))

    def made_by(self, item: str) -> List[Dict[str, Any]]:
        self.load()
        if self._by_out is None:
            self._by_out = {}
            for r in self.recipes:
                self._by_out.setdefault(r["out"], []).append(r)
        rank = {"crafting_shaped": 0, "crafting_shapeless": 0, "smelting": 1, "blasting": 2, "smoking": 2, "campfire_cooking": 3}
        # the ordinary way first: vanilla, a crafting table or a furnace, from the raw thing (not back from a block
        # or from nuggets)
        return sorted(self._by_out.get(item, []),
                      key=lambda r: (not r["ref"].startswith("minecraft:"), any(w in r["ref"] for w in ("nugget", "_block")),
                                     rank.get(r["t"], 4), "_ore" in r["ref"]))

    def smelted_from(self, item: str) -> List[str]:
        out: List[str] = []
        for r in self.made_by(item):
            if r["t"] == "smelting" and r["in"]:
                out += [a for a in self.expand(r["in"][0]) if a not in out]
        return out

    def used_in(self, item: str, limit: int = 12) -> List[str]:
        self.load()
        tags = {"#" + t for t in self.item_tags if item in self.tag_members(t)}
        outs = []
        for r in self.recipes:
            if any(item in alts or tags.intersection(alts) for alts in r["in"]) and r["out"] not in outs:
                outs.append(r["out"])
        outs.sort(key=lambda i: (not i.startswith("minecraft:"), i))
        return outs[:limit]

    def dropped_by(self, item: str) -> List[str]:
        self.load()
        return sorted((b for b, items in self.drops.items() if items and items[0] == item),
                      key=lambda b: (not b.startswith("minecraft:"), b))

    def tool_for(self, block: str) -> Tuple[Optional[str], int]:
        """Which tool kind mines this block well, and the lowest tier that makes it drop (0 = wood/any)."""
        self.load()
        kind = next((k for k in ("pickaxe", "axe", "shovel", "hoe")
                     if block in self.tag_members(f"minecraft:mineable/{k}", True)), None)
        tier = 0
        for t, n in (("minecraft:needs_stone_tool", 1), ("minecraft:needs_iron_tool", 2), ("minecraft:needs_diamond_tool", 3),
                     ("c:needs_netherite_tool", 4), ("neoforge:needs_netherite_tool", 4)):
            if block in self.tag_members(t, True):
                tier = max(tier, n)
        return kind, tier

    def search(self, query: str, limit: int = 5) -> List[str]:
        self.load()
        q = query.strip().lower()
        if not q:
            return []
        if ":" in q and (q in self.names or self.made_by(q) or q in self.drops):
            return [q]
        scored = []
        for i, n in self.names.items():
            nl = n.lower()
            path = i.split(":")[-1]
            if q == nl or q == path:
                s = 0
            elif nl.startswith(q) or path.startswith(q.replace(" ", "_")):
                s = 1
            elif q in nl or q.replace(" ", "_") in path:
                s = 2
            else:
                continue
            scored.append((s, not i.startswith("minecraft:"), len(n), i))
        return [i for *_, i in sorted(scored)[:limit]]

    def recipe_text(self, r: Dict[str, Any]) -> str:
        need: Dict[str, int] = {}
        for alts in r["in"]:
            label = self._alts_label(alts)
            need[label] = need.get(label, 0) + 1
        how = {"crafting_shaped": "合成", "crafting_shapeless": "合成", "smelting": "熔炉烧", "blasting": "高炉烧",
               "smoking": "烟熏炉", "campfire_cooking": "营火烤", "stonecutting": "切石机", "smithing_transform": "锻造台"}.get(r["t"], r["t"])
        return f"{how}：" + " + ".join(f"{k}×{v}" if v > 1 else k for k, v in need.items()) + (f" → {r['n']} 个" if r["n"] > 1 else "")

    def _alts_label(self, alts: List[str]) -> str:
        if len(alts) == 1 and alts[0].startswith("#"):
            members = self.expand(alts)
            return f"任意{self.name(members[0])}类" if members else alts[0]
        return "/".join(self.name(a) for a in alts[:2]) + ("等" if len(alts) > 2 else "")

    def quests_about(self, item: str, limit: int = 3) -> List[Dict[str, Any]]:
        self.load()
        return [q for q in self.quests if any(t.get("item") == item for t in q["tasks"])][:limit]

    def describe(self, item: str) -> str:
        lines = [f"{self.name(item)}（{item}）"]
        kind, tier = self.tool_for(item)
        if kind:
            lines.append(f"挖它用{ {'pickaxe': '镐', 'axe': '斧', 'shovel': '锹', 'hoe': '锄'}[kind]}，"
                         f"至少{['木', '石', '铁', '钻石', '下界合金'][tier]}制" + ("（等级不够挖了什么也不掉）" if tier else ""))
        drops = self.drops.get(item)
        if drops and drops[0] != item:
            lines.append("挖掉会掉：" + "、".join(self.name(d) for d in drops[:3]))
        src = self.dropped_by(item)
        if src:
            lines.append("从这些方块挖到：" + "、".join(self.name(b) for b in src[:4]))
        for r in self.made_by(item)[:4]:
            lines.append(self.recipe_text(r))
        uses = self.used_in(item)
        if uses:
            lines.append("能用来做：" + "、".join(self.name(u) for u in uses[:10]))
        for q in self.quests_about(item):
            lines.append(f"任务「{q['title']}」（{q['chapter']}）需要它")
        return "\n".join(lines)


class Handbook:
    """An experienced player's know-how, by topic (## sections of mc_expert.md, each with a 关键词 line)."""

    def __init__(self, path: Optional[str] = None):
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mc_expert.md")
        self.sections: List[Tuple[str, List[str], str]] = []
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return
        for block in re.split(r"^## ", text, flags=re.M)[1:]:
            title, _, body = block.partition("\n")
            m = re.search(r"^关键词[:：](.*)$", body, flags=re.M)
            words = [w.strip().lower() for w in re.split(r"[,，、\s]+", m.group(1)) if w.strip()] if m else []
            body = re.sub(r"^关键词[:：].*\n?", "", body, flags=re.M).strip()
            self.sections.append((title.strip(), words, body))

    def topics(self) -> List[str]:
        return [t for t, _, _ in self.sections]

    def find(self, query: str, limit: int = 2) -> List[Tuple[str, str]]:
        q = query.lower()
        scored = []
        for title, words, body in self.sections:
            s = (5 if title.lower() in q or q in title.lower() else 0) + sum(2 for w in words if w and w in q)
            if s:
                scored.append((-s, title, body))
        return [(t, b) for _, t, b in sorted(scored)[:limit]]

    def section(self, title: str) -> str:
        return next((b for t, _, b in self.sections if t == title), "")


class Knowledge:
    def __init__(self, index: Optional[ModpackIndex] = None, handbook: Optional[Handbook] = None):
        self.index = index or ModpackIndex(None)
        self.handbook = handbook or Handbook()

    def ask(self, query: str) -> str:
        query = str(query or "").strip()
        if not query:
            return "想查什么？可以写物品名/物品 id，或者一个话题：" + "、".join(self.handbook.topics())
        parts = []
        for title, body in self.handbook.find(query):
            parts.append(f"【{title}】\n{body}")
        items = self.index.search(query, 3)
        for i in items[:2]:
            parts.append(self.index.describe(i))
        if len(items) > 2:
            parts.append("也可能是：" + "、".join(f"{self.index.name(i)}（{i}）" for i in items[2:]))
        if not parts:
            return f"没查到“{query}”。能查的话题：" + "、".join(self.handbook.topics())
        return "\n\n".join(parts)[:2500]
