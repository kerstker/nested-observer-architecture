"""
Level 8m: the pack's own quest book, read the way a veteran reads it.

mc_knowledge reads the FTB quest files from her client (13 chapters, hundreds of quests in BMC5); this module
turns them into a plan:
  - the chapters in the order a player actually goes through them (Tutorial -> Overworld -> Nether / End /
    Aether / Twilight Forest -> The Otherside -> the side chapters -> Dragons last),
  - each quest's dependencies (FTB `dependencies` in the snbt) and its tasks,
  - what she can do right now with what she is carrying (mc_goal.Planner says whether the items are reachable),
  - per-chapter progress, and the one compact line the play prompt can show.

The optional bridge reads the client's server-synchronized FTB team data. Without a fresh snapshot the legacy "done" field
means only observed item conditions, never a confirmed completion or reward claim. Dependencies, submission
rules and server settings can prevent completion despite owning an item. Quests with kill / dimension / advancement / structure
tasks cannot be seen that way: they count in a chapter's total but are never guessed done and are skipped when
picking the next goal (the body can only be sent after an item, not after "kill the Wither").
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# The chapters in the order a player goes through the pack (the names are the ones in the lang files; Chinese
# variants are what a translated pack would call them).
CHAPTER_ORDER = ("tutorial", "overworld", "nether", "end", "aether", "twilight", "otherside", "curiosities",
                 "fishing", "villagers", "decoration", "storage", "dragons")
CHAPTER_HINTS = {
    "tutorial": ("tutorial", "welcome", "start", "教程", "新手", "欢迎", "入门"),
    "overworld": ("overworld", "主世界"),
    "nether": ("nether", "下界", "地狱"),
    "end": ("end", "末地", "末影"),
    "aether": ("aether", "天境", "以太"),
    "twilight": ("twilight", "暮色"),
    "otherside": ("otherside", "other side", "deeper", "dark", "深暗", "彼方"),
    "curiosities": ("curios", "oddit", "奇物", "珍奇", "杂项"),
    "fishing": ("fish", "钓鱼"),
    "villagers": ("villag", "村民"),
    "decoration": ("decor", "装饰", "建筑"),
    "storage": ("storage", "储存", "存储"),
    "dragons": ("dragon", "龙"),
}
MISSING = "ftbquests:missing_item"
INF = 1e6
# The planner prices a chain it cannot finish as INF, but dividing INF by a recipe's output count can leave a big
# finite number behind (bread priced 333337 because its recipe makes 3). Anything past this is that kind of
# arithmetic, not something worth chasing: in this pack every real early/mid item is under 200.
MAX_COST = 5000.0


def _hint(title: str, hint: str) -> bool:
    """Does this chapter title mean that chapter? English by word (so `fishing` matches `fish`), Chinese as text."""
    hint = str(hint or "").strip().lower()
    if not hint:
        return False
    title = str(title or "").lower()
    if any("\u4e00" <= ch <= "\u9fff" for ch in hint):
        return hint in title
    words = re.sub(r"[^a-z0-9]+", " ", title).split()
    return any(w == hint or w.startswith(hint) for w in words)


def chapter_key(title: str) -> Optional[str]:
    """The known chapter a title means (`Tutorial`, `The Otherside`, ...), or None for an unknown one."""
    for key in CHAPTER_ORDER:
        if any(_hint(title, h) for h in CHAPTER_HINTS[key]):
            return key
    return None


class QuestBook:
    """The pack's quest book as chapter -> dependency -> item data, plus what is next and how far each chapter is.

    `index` is a mc_knowledge.ModpackIndex (or anything with `.quests`), or the game folder path (the index is
    built from it), or None (an empty book: everything here answers "nothing" instead of raising).
    `progress` is a json file remembering the quest ids already met, so a quest done while the server was up stays
    done after a restart. `chapter_order` overrides the usual chapter order (tests, or a differently named pack).
    """

    def __init__(self, index: Any = None, progress: Optional[str] = None,
                 chapter_order: Optional[Iterable[str]] = None):
        self._source = index
        self._index: Any = None
        self.progress = progress
        self.done: Set[str] = set()
        self.quests: List[Dict[str, Any]] = []
        self.chapters: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._first_seen: Dict[str, int] = {}
        self._rank_of: Dict[str, Tuple[int, int]] = {}
        self._explicit: List[str] = [str(t) for t in (chapter_order or ())]
        self._planner: Any = None
        self._planner_tried = False
        self._tag_pick: Dict[str, str] = {}
        self._unit_cost: Dict[str, float] = {}
        self._ready = False
        self.observed_dimensions: Set[str] = set()
        self.load_info = {"source": "index", "files": 0, "errors": []}
        self._server_rows = {}
        self._server_at = 0.0
        self._server_session = None
        self._server_reason = "not_polled"
        self._server_announced = set()
        if progress:
            try:
                with open(progress, encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):                 # tolerate a richer progress file
                    self.observed_dimensions = set(saved.get("observed_dimensions") or [])
                    saved = saved.get("done") or []
                self.done = {str(q) for q in saved or []}
            except (OSError, ValueError):
                pass
        self._ensure()

    # --- building ------------------------------------------------------------------------------

    def _ensure(self) -> None:
        if self._ready:
            return
        self._ready = True
        ix = self._source
        if isinstance(ix, str):
            from src.levels.l8_agency.mc_knowledge import ModpackIndex
            ix = ModpackIndex(ix if os.path.isdir(ix) else None)
            try:
                ix.load()
            except Exception as e:                        # a broken pack folder must not take anything down
                logger.warning(f"[Quest] index not readable: {e}")
                ix = ModpackIndex(None)
        self._index = ix
        # GameBridge passes an uninitialised ModpackIndex object, not a folder string.
        # Reading .quests before load() permanently froze the book at zero quests.
        if ix is not None and hasattr(ix, "load"):
            try:
                ix.load()
            except Exception:
                self.load_info["errors"].append("recipe_index_load")
                logger.warning("[Quest] recipe index load failed; reading quest sources independently", exc_info=True)
        raw = getattr(ix, "quests", None) if ix is not None else None
        if callable(raw):
            try:
                raw = raw()
            except Exception:
                raw = None
        game_dir = getattr(ix, "game_dir", None)
        if game_dir:
            fresh = self._read_pack_quests(game_dir)
            if fresh is not None:
                raw = fresh
        for doc in list(raw or []):
            if not isinstance(doc, dict):
                continue
            qid = str(doc.get("id") or f"q{len(self.quests)}")
            if qid in self._by_id:
                continue
            chapter = str(doc.get("chapter") or "").strip() or "?"
            self._first_seen.setdefault(chapter, len(self._first_seen))
            deps = doc.get("deps") if "deps" in doc else doc.get("dependencies")
            deps = [deps] if isinstance(deps, str) else list(deps or [])
            node = {
                "id": qid,
                "chapter": chapter,
                "title": str(doc.get("title") or "").strip() or str(doc.get("text") or "").strip(),
                "text": str(doc.get("text") or ""),
                "description": list(doc.get("description") or []),
                "tasks": [t for t in (doc.get("tasks") or []) if isinstance(t, dict)],
                "deps": [str(d) for d in deps],
                "dependency_requirement": doc.get("dependency_requirement") or "all_completed",
                "min_required_dependencies": int(doc.get("min_required_dependencies") or 0),
                "rewards": list(doc.get("rewards") or []),
                "order": len(self.quests),
            }
            if not node["title"]:                        # a quest the lang files have no name for
                first = next((str(t["item"]) for t in node["tasks"]
                              if t.get("type") == "item" and t.get("item") and not str(t["item"]).startswith("#")), "")
                node["title"] = f"弄到{self.name(first)}" if first else f"任务{len(self.quests) + 1}"
            self.quests.append(node)
            self._by_id[qid] = node
        decks: Dict[str, List[str]] = {}
        for q in self.quests:
            decks.setdefault(q["chapter"], []).append(q["id"])
        for pos, title in enumerate(sorted(decks, key=lambda t: (self._rank(t), self._first_seen[t]))):
            self.chapters.append({"title": title, "key": chapter_key(title), "order": pos,
                                  "total": len(decks[title]), "quests": decks[title]})
            self._rank_of[title] = (self._rank(title), pos)
        for q in self.quests:
            q["chapter_pos"] = self._rank_of[q["chapter"]][1]
        # Old summaries discarded component checks; do not retain their false dragon-breed matches.
        self.done = {qid for qid in self.done if qid in self._by_id and self.inferable(self._by_id[qid])}
        logger.info(f"[Quest] book: {len(self.chapters)} chapters, {len(self.quests)} quests, {len(self.done)} done")

    def _read_pack_quests(self, game_dir):
        """Read current FTB SNBT without losing filters/dependency rules in the recipe cache."""
        from src.levels.l8_agency.mc_knowledge import parse_snbt, _strip_codes
        root = Path(game_dir) / "config" / "ftbquests" / "quests"
        files = sorted((root / "chapters").glob("*.snbt"))
        if not files:
            return None
        lang = {}
        for language in ("en_us", "zh_cn"):
            path = root / "lang" / (language + ".snbt")
            if path.is_file():
                try:
                    lang.update(parse_snbt(path.read_text(encoding="utf-8")))
                except (OSError, ValueError, IndexError, TypeError):
                    self.load_info["errors"].append(path.name)
        output = []
        self.load_info.update(source="live_pack_snbt", files=len(files))
        for path in files:
            try:
                chapter = parse_snbt(path.read_text(encoding="utf-8"))
                title = _strip_codes(lang.get(f"chapter.{chapter.get('id')}.title")
                                     or chapter.get("title") or chapter.get("filename") or path.stem)
                for raw in chapter.get("quests") or []:
                    node = dict(raw, chapter=title)
                    qid = raw.get("id")
                    node["title"] = _strip_codes(lang.get(f"quest.{qid}.title") or raw.get("title") or "")
                    node["text"] = _strip_codes(lang.get(f"quest.{qid}.quest_subtitle") or raw.get("subtitle") or "")
                    description = lang.get(f"quest.{qid}.quest_desc") or raw.get("description") or []
                    node["description"] = [_strip_codes(str(x)) for x in (description if isinstance(description, list) else [description])]
                    tasks = []
                    for task in raw.get("tasks") or []:
                        task = dict(task)
                        item = task.get("item")
                        if isinstance(item, dict):
                            components = item.get("components") or {}
                            filter_text = str(components.get("ftbfiltersystem:filter") or "")
                            match = re.fullmatch(r"item_tag\(([^)]+)\)", filter_text.strip())
                            task["item"] = "#" + match.group(1) if match else item.get("id")
                            task["count"] = task.get("count") or item.get("count") or 1
                            if components and not match:
                                task["item_components"] = dict(components)
                        if task.get("type") == "kill":
                            task["count"] = task.get("value") or task.get("count") or 1
                        tasks.append(task)
                    node["tasks"] = tasks
                    output.append(node)
            except (OSError, ValueError, IndexError, TypeError):
                self.load_info["errors"].append(path.name)
        return output

    def _rank(self, title: str) -> int:
        title = str(title or "")
        key = chapter_key(title)
        for i, want in enumerate(self._explicit):
            if want == title or want == key or _hint(title, want):
                return i
        if key:
            return CHAPTER_ORDER.index(key)
        return len(CHAPTER_ORDER) + self._first_seen.get(title, 0)

    @property
    def index(self) -> Any:
        return self._index

    def planner(self) -> Any:
        """The mc_goal planner over this pack, or None when there is no index to plan from."""
        if not self._planner_tried:
            self._planner_tried = True
            try:
                from src.levels.l8_agency.mc_goal import Planner
                self._planner = Planner(self._index) if self._index is not None else None
            except Exception as e:
                logger.warning(f"[Quest] no planner: {e}")
                self._planner = None
        return self._planner

    def name(self, item: str) -> str:
        try:
            return self._index.name(item)
        except Exception:
            return str(item).split(":")[-1].replace("_", " ")

    # --- what a quest wants --------------------------------------------------------------------

    def item_tasks(self, quest: Dict[str, Any]) -> List[Tuple[str, int]]:
        """The quest's item tasks as (item id or "#tag", count); the other task kinds cannot be seen by her client."""
        out = []
        for t in quest.get("tasks") or []:
            if t.get("type") != "item":
                continue
            if self.server_verified() and any(st.get("id") == t.get("id") and st.get("completed")
                                             for st in self._server_rows.get(quest["id"], {}).get("tasks", [])):
                continue
            item = t.get("item")
            if not item or str(item) == MISSING:
                continue
            out.append((str(item), int(t.get("count") or 1)))
        return out

    def inferable(self, quest: Dict[str, Any]) -> bool:
        """True when every task is an item task: only those can be detected from her inventory."""
        tasks = quest.get("tasks") or []
        return bool(tasks) and all(t.get("type") == "item" and t.get("item") and str(t["item"]) != MISSING
                                   and not t.get("item_components")
                                   for t in tasks)

    def _members(self, tag: str) -> List[str]:
        ix = self._index
        try:
            return list(ix.expand([tag])) if ix is not None and hasattr(ix, "expand") else []
        except Exception:
            return []

    def _unit(self, planner: Any, item: str) -> float:
        """How expensive one of this item is from scratch (cached: the pack does not change while she plays)."""
        if item not in self._unit_cost:
            try:
                self._unit_cost[item] = float(planner.cost(item, {}, goal=True)) if planner is not None else 0.0
            except Exception:
                self._unit_cost[item] = 0.0
        return self._unit_cost[item]

    def _tag_item(self, tag: str) -> Optional[str]:
        """The member of a tag that is easiest for her to get (the pack's own recipe trees decide)."""
        if tag not in self._tag_pick:
            members = self._members(tag)
            if not members:
                return None
            planner = self.planner()
            self._tag_pick[tag] = min(members, key=lambda m: (self._unit(planner, m), members.index(m))) \
                if planner is not None else members[0]
        return self._tag_pick[tag]

    def wants(self, quest: Dict[str, Any], inventory: Optional[Dict[str, int]] = None) -> List[Tuple[str, int]]:
        """Each item task as a concrete item (a tag: the one easiest for her to get) and how many."""
        have = dict(inventory or {})
        out = []
        for item, n in self.item_tasks(quest):
            if item.startswith("#"):
                members = self._members(item)
                if not members:
                    continue
                held = [m for m in members if have.get(m, 0) >= n]
                item = held[0] if held else self._tag_item(item)
                if item is None:
                    continue
            out.append((item, n))
        return out

    def _has(self, item: str, n: int, have: Dict[str, int]) -> bool:
        if have.get(item, 0) >= n:
            return True
        return item.startswith("#") and any(have.get(m, 0) >= n for m in self._members(item))

    def _met(self, quest: Dict[str, Any], have: Dict[str, int], done: Set[str]) -> bool:
        if quest["id"] in done:
            return True
        if not self.inferable(quest):
            return False
        return all(self._has(i, n, have) for i, n in self.item_tasks(quest))

    def server_verified(self) -> bool:
        return bool(self._server_session and self._server_rows and time.monotonic() - self._server_at <= 20)

    def server_status(self) -> Dict[str, Any]:
        fresh = self.server_verified()
        return {"server_verified": fresh, "progress_source": "ftb_client_team_sync" if fresh else "local_observation",
                "synced_quests": len(self._server_rows) if fresh else 0,
                "completed": sum(row["completed"] for row in self._server_rows.values()) if fresh else None,
                "progress_age_seconds": round(time.monotonic() - self._server_at, 2) if self._server_at else None,
                "progress_reason": "synced" if fresh else "stale" if self._server_rows else self._server_reason}

    def ingest_server_progress(self, snapshot: Optional[Dict[str, Any]]) -> bool:
        """Accept only a full, validated, current-session snapshot. Never persist it as local inference."""
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        rows = snapshot.get("quests")
        session = snapshot.get("session")
        valid = (snapshot.get("available") is True and snapshot.get("server_verified") is True
                 and snapshot.get("source") == "ftb_client_team_sync" and snapshot.get("complete_snapshot") is True
                 and isinstance(session, str) and 0 < len(session) <= 80 and isinstance(rows, list)
                 and len(rows) <= 4096 and snapshot.get("total") == len(rows))
        parsed = {}
        if valid:
            for row in rows:
                if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in parsed
                        or any(type(row.get(k)) is not bool for k in ("completed", "started", "can_start"))
                        or not isinstance(row.get("tasks"), list) or len(row["tasks"]) > 256):
                    valid = False
                    break
                for task in row["tasks"]:
                    if (not isinstance(task, dict) or not isinstance(task.get("id"), str)
                            or type(task.get("completed")) is not bool
                            or any(type(task.get(k)) is not int or task[k] < 0 for k in ("progress", "max"))):
                        valid = False
                        break
                if not valid:
                    break
                parsed[row["id"]] = dict(row, tasks=[dict(t) for t in row["tasks"]])
        if not valid or not set(parsed).intersection(self._by_id):
            self._server_rows = {}
            self._server_reason = "unavailable" if snapshot.get("available") is not True else "invalid_or_mismatched_snapshot"
            return False
        if session != self._server_session:
            self._server_announced = set()
        self._server_session, self._server_at = session, time.monotonic()
        self._server_rows = parsed
        self._server_reason = "synced"
        return True

    def done_ids(self, inventory: Optional[Dict[str, int]] = None, done: Optional[Set[str]] = None) -> Set[str]:
        """The quests met with what she is carrying (plus what was remembered), without recording anything."""
        if self.server_verified():
            return {qid for qid, row in self._server_rows.items() if row["completed"] and qid in self._by_id}
        reached = set(self.done if done is None else done)
        have = dict(inventory or {})
        for q in self.quests:
            if q["id"] not in reached and self._met(q, have, reached):
                reached.add(q["id"])
            tasks = q.get("tasks") or []
            if tasks and all(t.get("type") == "dimension" and t.get("dimension") in self.observed_dimensions
                             for t in tasks):
                reached.add(q["id"])
        return reached

    def check(self, inventory: Optional[Dict[str, int]] = None) -> List[str]:
        """Remember newly observed item conditions; these titles are NOT server-confirmed completions."""
        if self.server_verified():
            now = self.done_ids()
            new = [q["title"] or q["id"] for q in self.quests if q["id"] in now - self._server_announced]
            self._server_announced = now
            return new
        before = set(self.done)
        now = self.done_ids(inventory)
        new = [q["title"] or q["id"] for q in self.quests if q["id"] in now and q["id"] not in before]
        if new:
            self.done |= now
            self._save()
        return new

    def mark_done(self, quest_id: str) -> None:
        if str(quest_id) in self._by_id:
            self.done.add(str(quest_id))
            self._save()

    def _save(self) -> None:
        if not self.progress:
            return
        try:
            os.makedirs(os.path.dirname(self.progress) or ".", exist_ok=True)
            with open(self.progress, "w", encoding="utf-8") as f:
                json.dump({"version": 2, "done": sorted(self.done),
                           "evidence": "local_observed_conditions", "server_verified": False,
                           "observed_dimensions": sorted(self.observed_dimensions)}, f)
        except OSError:
            pass

    # --- the next goals ------------------------------------------------------------------------

    def _dep_ready(self, quest: Dict[str, Any], done: Set[str], _visiting=None) -> bool:
        """Local evidence can suggest readiness; unknown kill/checkmark deps are never assumed complete."""
        if self.server_verified():
            return self._server_rows.get(quest["id"], {}).get("can_start") is True
        deps = quest.get("deps") or []
        if not deps:
            return True
        visiting = set(_visiting or ())
        if quest["id"] in visiting:
            return False
        visiting.add(quest["id"])
        rule = quest.get("dependency_requirement") or "all_completed"
        if rule not in ("all_completed", "one_completed", "all_started", "one_started"):
            return False
        # Completing observed conditions implies they were started. Partial server progress is unknown.
        needed = int(quest.get("min_required_dependencies") or (1 if rule.startswith("one_") else len(deps)))
        return sum(dep in done and dep in self._by_id and self._dep_ready(self._by_id[dep], done, visiting)
                   for dep in deps) >= needed

    def _cost(self, planner: Any, item: str, count: int, have: Dict[str, int]) -> float:
        if planner is None:
            return 0.0
        cost = self._unit(planner, item)
        missing = max(0, int(count) - int(have.get(item, 0)))
        return cost * missing if cost < MAX_COST else INF

    def _reachable(self, quest: Dict[str, Any], have: Dict[str, int], planner: Any,
                   stations: Set[str]) -> bool:
        """Does the planner see a way from here to what the quest asks for (its first step is not a dead end)? A
        finite price from the planner is the main test; this catches the last "no idea where that comes from"."""
        if planner is None:
            return True
        for item, n in self.wants(quest, have):
            if have.get(item, 0) >= n:
                continue
            try:
                step = planner.next_step(item, n, dict(have), set(stations))
            except Exception:
                return True
            if step.get("kind") == "stuck":
                return False
        return True

    def _quest_cost(self, want: List[Tuple[str, int]], have: Dict[str, int], planner: Any) -> float:
        return sum(self._cost(planner, i, n, have) for i, n in want)

    def _reason(self, quest: Dict[str, Any], want: List[Tuple[str, int]], ready: bool,
                have: Dict[str, int]) -> str:
        need = "、".join(f"{self.name(i)}×{n}" for i, n in want[:3])
        if not ready:
            return f"前置任务还没完，先做前面的（要{need}）"
        if any(have.get(i, 0) for i, _ in want):
            return f"材料手上已经有一些，差{need}"
        key = chapter_key(quest["chapter"])
        early = "开局" if key in ("tutorial", "overworld") else f"{quest['chapter']}章"
        return f"{early}，要{need}"

    def dependency_chain(self, quest_id: str) -> Dict[str, Any]:
        """Prerequisites before dependants, with broken links/cycles visible instead of ignored."""
        order, missing, cycles, visited, active = [], set(), set(), set(), []

        def visit(qid):
            if qid in active:
                cycles.update(active[active.index(qid):])
                return
            if qid in visited:
                return
            node = self._by_id.get(qid)
            if node is None:
                missing.add(qid)
                return
            active.append(qid)
            for dep in node.get("deps") or []:
                visit(dep)
            active.pop()
            visited.add(qid)
            order.append(qid)

        visit(str(quest_id))
        return {"order": order, "missing": sorted(missing), "cycles": sorted(cycles)}

    def observe_state(self, state: Dict[str, Any]) -> None:
        """A real player-state dimension is evidence of visiting, never evidence of FTB completion."""
        dimension = state.get("dimension")
        if state.get("in_world", True) and isinstance(dimension, str) and dimension and dimension not in self.observed_dimensions:
            self.observed_dimensions.add(dimension)
            self._save()

    def task_status(self, quest: Union[str, Dict[str, Any]], inventory=None) -> List[Dict[str, Any]]:
        node = quest if isinstance(quest, dict) else self._by_id.get(str(quest), {})
        have = dict(inventory or {})
        result = []
        for task in node.get("tasks") or []:
            kind = task.get("type") or "unknown"
            server = next((t for t in self._server_rows.get(node.get("id"), {}).get("tasks", [])
                           if t["id"] == task.get("id")), None) if self.server_verified() else None
            observed = False
            unknown = True
            if kind == "item" and task.get("item") and not task.get("item_components"):
                observed = self._has(str(task["item"]), int(task.get("count") or 1), have)
                unknown = False
            elif kind == "dimension":
                observed = task.get("dimension") in self.observed_dimensions
                unknown = not observed
            result.append({"id": task.get("id"), "type": kind, "target": task.get("item") or task.get("entity")
                           or task.get("dimension") or task.get("structure") or task.get("advancement")
                           or task.get("to_observe"), "count": task.get("count") or task.get("value") or 1,
                           "status": "observed" if observed else "unknown" if unknown else "missing",
                           "requires_components": bool(task.get("item_components")), "server_verified": False})
            if server:
                result[-1].update(status="completed" if server["completed"] else "in_progress",
                                  progress=server["progress"], count=server["max"], server_verified=True)
        return result

    def next_actions(self, state: Dict[str, Any], capabilities: Optional[Iterable[str]] = None,
                     limit: int = 3) -> List[Dict[str, Any]]:
        """Task-directed movement only when a supported controller and a known target exist.

        Kill counts, advancements, checkmarks and structure discovery are not inferred from loot,
        nearby entities or arrival at a coordinate. They remain explicit verification gaps.
        """
        self.observe_state(state)
        if int(limit) <= 0:
            return []
        capabilities = set(capabilities or ())
        have = state.get("inventory") or {}
        done = self.done_ids(have)
        candidates = []
        for q in sorted(self.quests, key=lambda q: (q["chapter_pos"], len(self.dependency_chain(q["id"])["order"]))):
            chain = self.dependency_chain(q["id"])
            row = self._server_rows.get(q["id"], {}) if self.server_verified() else {}
            if "quests_ui" in capabilities and row.get("completed") and row.get("unclaimed_rewards"):
                for reward in q.get("rewards", []):
                    if reward.get("type") in ("item", "xp", "xp_levels") and reward.get("id"):
                        candidates.append({"kind": "quest_ui", "name": "claim", "args": {"quest_id": q["id"], "reward_id": reward["id"]},
                                           "why": f"任务书「{q['title']}」已由服务器确认完成，打开任务书领取奖励"})
                        break
                if len(candidates) >= limit:
                    break
            if q["id"] in done or (not self.server_verified() and (chain["cycles"] or chain["missing"])) or not self._dep_ready(q, done):
                continue
            for task in q.get("tasks") or []:
                if self.server_verified() and any(t["id"] == task.get("id") and t["completed"]
                                                  for t in self._server_rows.get(q["id"], {}).get("tasks", [])):
                    continue
                action = None
                if "quests_ui" in capabilities and row:
                    reading = (task.get("type") == "checkmark" and bool(q.get("description"))
                               and bool(re.search(r"read.*description|阅读.*(?:说明|描述)|阅读描述", q.get("text", ""), re.I)))
                    submission = (task.get("type") == "item" and task.get("item") and not task.get("item_components")
                                  and self._has(task["item"], int(task.get("count") or 1), have))
                    if (reading or submission) and task.get("id"):
                        candidates.append({"kind": "quest_ui", "name": "read" if reading else "submit",
                                           "args": {"quest_id": q["id"], "task_id": task["id"]},
                                           "why": f"任务书「{q['title']}」：{'阅读任务说明后确认' if reading else '按任务条件提交已有物品'}",
                                           "description": q.get("description", []), "verification_required": True})
                        break
                if (task.get("type") == "dimension" and task.get("dimension") == "minecraft:the_nether"
                        and state.get("dimension") == "minecraft:overworld" and state.get("portal")
                        and "enter_portal" in capabilities):
                    action = {"name": "enter_portal", "args": {}}
                elif (task.get("type") == "dimension" and task.get("dimension") == "minecraft:the_end"
                      and state.get("end_portal") and "enter_end" in capabilities):
                    action = {"name": "enter_end", "args": {}}
                elif task.get("type") == "structure" and "go_to" in capabilities:
                    pos = (state.get("known_structures") or {}).get(task.get("structure"))
                    if isinstance(pos, dict) and all(isinstance(pos.get(k), (int, float)) for k in ("x", "y", "z")):
                        if not pos.get("dimension") or pos["dimension"] == state.get("dimension"):
                            action = {"name": "go_to", "args": {k: pos[k] for k in ("x", "y", "z")}}
                if action:
                    candidates.append({"kind": "skill", **action, "quest_id": q["id"],
                                       "why": f"任务书「{q['title']}」：前往已观察到的入口/地点，抵达后核实条件",
                                       "verification_required": True, "server_verified": self.server_verified()})
                    break
            if len(candidates) >= max(0, int(limit)):
                break
        return candidates

    def next_goals(self, limit: int = 5, inventory: Optional[Dict[str, int]] = None,
                   skip: Optional[Iterable[str]] = None, stations: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """The quest goals worth doing now, best first: early chapters, prerequisites met, cheapest first.

        Each one is `{"id", "chapter", "title", "text", "item", "count", "items", "cost", "ready", "reason"}`
        where `text` reads like "Overworld: Obtain a Wooden Log" and `item` is what to hand to
        Skills.set_goal(item, count, "quest").
        """
        have = {k: int(v) for k, v in (inventory or {}).items() if int(v or 0) > 0}
        done = self.done_ids(have)
        skip = {str(s) for s in (skip or ())}
        stations = set(stations or ())
        limit = max(0, int(limit))
        planner = self.planner()
        ranked = []
        for q in self.quests:
            collectable = self.inferable(q) or (self.server_verified() and any(
                t.get("type") == "item" and t.get("item") and not t.get("item_components") for t in q.get("tasks", [])))
            if q["id"] in done or not collectable:
                continue
            want = [(i, n) for i, n in self.wants(q, have) if have.get(i, 0) < n]
            if not want or any(i in skip for i, _ in want):
                continue
            ready = self._dep_ready(q, done)
            chain = self.dependency_chain(q["id"])
            if not self.server_verified() and (chain["cycles"] or chain["missing"]):
                ready = False
            cost = self._quest_cost(want, have, planner)
            if cost >= MAX_COST:                     # one of its items cannot be got at all
                continue
            ranked.append((self._rank_of[q["chapter"]][0], 0 if ready else 1, cost, q["order"], {
                "id": q["id"], "chapter": q["chapter"], "title": q["title"],
                "text": f"{q['chapter']}: {q['title']}", "item": want[0][0], "count": want[0][1],
                "items": want, "cost": round(cost, 2), "ready": ready,
                "dependency_chain": chain["order"], "unverified_dependencies": [d for d in chain["order"]
                                                                                 if d != q["id"] and d not in done],
                "server_verified": self.server_verified(),
                "reason": self._reason(q, want, ready, have),
            }))
        ranked.sort(key=lambda row: row[:4])
        out = []
        for *_rank, goal in ranked:                  # the chain behind only the candidates that might be next
            if len(out) >= limit:
                break
            if self._reachable(self._by_id[goal["id"]], have, planner, stations):
                out.append(goal)
        return out

    def quest_for_item(self, item: str, limit: int = 5) -> List[Dict[str, Any]]:
        """The quests that ask for this item (a quest asking for a tag counts for every member)."""
        item = str(item or "").strip()
        if not item:
            return []
        out = []
        for q in self.quests:
            for want, n in self.item_tasks(q):
                if want == item or (want.startswith("#") and item in self._members(want)):
                    out.append(dict(q, count=n))
                    break
        return out[:limit]

    def prerequisites_for(self, quest: Union[str, Dict[str, Any]], inventory: Optional[Dict[str, int]] = None,
                          stations: Optional[Iterable[str]] = None, steps: int = 16) -> List[Dict[str, Any]]:
        """The item chain to finish that quest from here: mc_goal Steps in the order a player does them."""
        node = quest if isinstance(quest, dict) and "id" in quest else self._by_id.get(str(quest))
        planner = self.planner()
        if node is None or planner is None:
            return []
        have = {k: int(v) for k, v in (inventory or {}).items() if int(v or 0) > 0}
        stations = set(stations or ())
        out: List[Dict[str, Any]] = []
        for item, n in self.wants(node, have):
            for _ in range(steps):
                if have.get(item, 0) >= n or len(out) >= steps * 2:
                    break
                step = planner.next_step(item, n, dict(have), set(stations))
                if step.get("kind") in ("done", None):
                    break
                step["goal"] = item
                out.append(step)
                if step.get("kind") == "stuck":
                    break
                have[step["item"]] = have.get(step["item"], 0) + max(1, int(step.get("count") or 1))
        return out

    # --- how far each chapter is ---------------------------------------------------------------

    def chapter_progress(self, inventory: Optional[Dict[str, int]] = None,
                         stations: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Per chapter: done/total and the quests that can be worked on now (in the pack's own order)."""
        have = {k: int(v) for k, v in (inventory or {}).items() if int(v or 0) > 0}
        done = self.done_ids(have)
        planner = self.planner()
        stations = set(stations or ())
        out = []
        for ch in self.chapters:
            nodes = [self._by_id[i] for i in ch["quests"]]
            reachable = []
            for q in nodes:
                if q["id"] in done or not self.inferable(q):
                    continue
                want = [(i, n) for i, n in self.wants(q, have) if have.get(i, 0) < n]
                if not want or not self._dep_ready(q, done) or self._quest_cost(want, have, planner) >= MAX_COST:
                    continue
                if not self._reachable(q, have, planner, stations):
                    continue
                reachable.append({"id": q["id"], "title": q["title"], "item": want[0][0], "count": want[0][1]})
                if len(reachable) >= 5:              # enough to show what is open in this chapter
                    break
            out.append({
                "chapter": ch["title"], "key": ch["key"], "order": ch["order"],
                "done": sum(1 for q in nodes if q["id"] in done), "total": len(nodes),
                "percent": round(100.0 * sum(1 for q in nodes if q["id"] in done) / max(1, len(nodes))),
                "evidence": "ftb_client_team_sync" if self.server_verified() else "observed_item_conditions",
                "server_verified": self.server_verified(),
                "unknown_tasks": sum(s["status"] == "unknown" for q in nodes for s in self.task_status(q, have)),
                "component_checks_required": sum(bool(t.get("item_components")) for q in nodes for t in q["tasks"]),
                "reachable": reachable,
                "next": reachable[0] if reachable else None,
            })
        return out

    def progress_text(self, inventory: Optional[Dict[str, int]] = None, chapters: int = 3) -> str:
        """One line for the play prompt: the first chapters' progress and what the book suggests next."""
        progress = self.chapter_progress(inventory)
        if not progress:
            return ""
        line = "任务书：" + "、".join(f"{p['chapter']} {p['done']}/{p['total']}" for p in progress[:chapters])
        line += "（服务器同步的团队完成状态；此接口不会领奖）" if self.server_verified() else "（仅观测到的物品条件，服务器完成/领奖未核实）"
        nxt = next((p["next"] for p in progress if p["next"]), None)
        if nxt:
            line += f"；下一步「{nxt['title']}」"
        if self.server_verified():
            pending = []
            for q in self.quests:
                row = self._server_rows.get(q["id"], {})
                if row.get("completed") or not row.get("can_start"):
                    continue
                for status in self.task_status(q, inventory):
                    if status.get("server_verified") and status["status"] == "in_progress" and status["type"] != "item":
                        target = self.name(status["target"]) if status.get("target") else status["type"]
                        pending.append(f"「{q['title']}」{target} {status['progress']}/{status['count']}")
                        break
                if len(pending) == 2:
                    break
            if pending:
                line += "；服务端待办：" + "、".join(pending)
        return line
