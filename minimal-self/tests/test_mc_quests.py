"""The pack's quest book as a plan: chapter order, dependencies, what is reachable, chapter progress."""
import json
import os
import shutil
import tempfile
import unittest
import zipfile

from src.levels.l8_agency.mc_knowledge import Handbook, ModpackIndex
from src.levels.l8_agency.mc_quests import CHAPTER_ORDER, QuestBook, chapter_key

M = "minecraft:"
HERE = os.path.dirname(os.path.abspath(__file__))
EXPERT = os.path.join(os.path.dirname(HERE), "src", "levels", "l8_agency", "mc_expert.md")

NAMES = {M + "bread": "面包", M + "oak_log": "橡木原木", M + "iron_sword": "铁剑", "dmr:dragon_egg": "龙蛋"}


class FakeIndex:
    """The little QuestBook needs from a ModpackIndex: its `.quests`, names, and tags opened up."""

    def __init__(self, quests, tags=None):
        self.quests = quests
        self.tags = tags or {}

    def name(self, item):
        return NAMES.get(item) or item.split(":")[-1].replace("_", " ")

    def expand(self, alts):
        out = []
        for a in alts:
            out.extend(self.tags.get(a.lstrip("#"), []) if a.startswith("#") else [a])
        return out


def quest(qid, chapter, title, tasks, deps=()):
    return {"id": qid, "chapter": chapter, "title": title, "text": "", "deps": list(deps), "tasks": list(tasks)}


def item(item_id, count=None):
    task = {"type": "item", "item": item_id}
    if count:
        task["count"] = count
    return task


FAKE_QUESTS = [
    quest("D1", "Dragons", "Obtain a Dragon Egg", [item("dmr:dragon_egg")]),
    quest("T1", "Tutorial", "Welcome", [item(M + "bread")]),
    quest("O1", "Overworld", "Obtain a Wooden Log", [item("#minecraft:logs", 1)]),
    quest("O2", "Overworld", "Craft an Iron Sword", [item(M + "iron_sword")], deps=["O1"]),
    quest("O3", "The Otherside", "Defeat a Warden", [{"type": "kill", "entity": "deeperdarker:warden"}]),
]


def fake_book(quests=None, **kw):
    return QuestBook(FakeIndex(FAKE_QUESTS if quests is None else quests,
                               tags={"minecraft:logs": [M + "oak_log"]}), **kw)


def synced_rows(rows, session="opaque-session"):
    return {"available": True, "server_verified": True, "source": "ftb_client_team_sync",
            "complete_snapshot": True, "session": session, "total": len(rows), "quests": rows}


def synced_quest(qid, completed=False, can_start=True, tasks=None):
    return {"id": qid, "completed": completed, "started": completed, "can_start": can_start,
            "tasks": tasks or []}


def test_server_evidence_unlocks_kill_dependency_and_overrides_inventory_inference():
    book = fake_book([quest("K", "Overworld", "Kill", [{"id": "TK", "type": "kill", "entity": "minecraft:zombie", "value": 5}]),
                      quest("I", "Overworld", "Iron sword", [item(M + "iron_sword")], deps=["K"])])
    assert not book._dep_ready(book._by_id["I"], set())
    data = synced_rows([synced_quest("K", True, tasks=[{"id": "TK", "progress": 5, "max": 5, "completed": True}]),
                        synced_quest("I", False)])
    assert book.ingest_server_progress(data)
    assert book.done_ids({M + "iron_sword": 1}) == {"K"}  # owning sword is not server completion
    assert book._dep_ready(book._by_id["I"], set())
    assert book.task_status("K")[0]["server_verified"] is True
    assert book.task_status("K")[0]["progress"] == 5
    assert book.next_goals(2, {})[0]["ready"] is True
    assert book.check() == ["Kill"]
    assert book.check() == []
    assert not book.done  # no verified progress is persisted as inferred inventory evidence


def test_unavailable_stale_and_new_session_cannot_keep_verified_completion(monkeypatch):
    book = fake_book([quest("K", "Overworld", "Kill", [{"type": "kill"}])])
    now = [10.0]
    monkeypatch.setattr("src.levels.l8_agency.mc_quests.time.monotonic", lambda: now[0])
    assert book.ingest_server_progress(synced_rows([synced_quest("K", True)]))
    assert book.done_ids() == {"K"}
    now[0] = 31.0
    assert book.done_ids() == set()
    assert book.server_status()["server_verified"] is False
    assert book.ingest_server_progress(synced_rows([synced_quest("K", False)], session="different-world"))
    assert book.done_ids() == set()
    assert not book.ingest_server_progress({"available": False})
    assert not book.server_verified()
    assert not book._server_rows


def test_partial_malformed_or_wrong_book_snapshot_fails_closed():
    book = fake_book()
    partial = synced_rows([synced_quest("T1", True)])
    partial["complete_snapshot"] = False
    assert not book.ingest_server_progress(partial)
    malformed = synced_rows([synced_quest("T1", True, tasks=[{"id": "t", "progress": "10", "max": 10, "completed": True}])])
    assert not book.ingest_server_progress(malformed)
    assert not book.ingest_server_progress(synced_rows([synced_quest("NOT_THIS_BOOK", True)]))
    assert not book.server_verified()


class TestTheBook(unittest.TestCase):
    def test_chapters_come_in_the_order_a_player_goes_through_them(self):
        qb = fake_book()
        self.assertEqual([c["title"] for c in qb.chapters], ["Tutorial", "Overworld", "The Otherside", "Dragons"])
        self.assertEqual([c["total"] for c in qb.chapters], [1, 2, 1, 1])
        self.assertEqual(chapter_key("The Otherside"), "otherside")
        self.assertEqual(chapter_key("The Aether"), "aether")
        self.assertEqual(chapter_key("暮色森林"), "twilight")
        self.assertIsNone(chapter_key("Random Chapter"))
        self.assertEqual(CHAPTER_ORDER[-1], "dragons")
        # an explicit order (a differently named pack, a test) wins over the usual one
        self.assertEqual(QuestBook(FakeIndex(FAKE_QUESTS), chapter_order=["Dragons"]).chapters[0]["title"], "Dragons")

    def test_next_goals_early_chapters_first_and_the_bag_counts(self):
        goals = fake_book().next_goals(3)
        self.assertEqual([g["id"] for g in goals], ["T1", "O1", "O2"])
        self.assertEqual((goals[0]["chapter"], goals[0]["title"]), ("Tutorial", "Welcome"))
        self.assertEqual(goals[0]["text"], "Tutorial: Welcome")
        self.assertEqual((goals[0]["item"], goals[0]["count"]), (M + "bread", 1))
        self.assertTrue(goals[0]["reason"])
        self.assertTrue(goals[0]["ready"])
        # bread in the bag: the tutorial quest is behind her, the overworld ones come first
        self.assertEqual([g["id"] for g in fake_book().next_goals(2, {M + "bread": 1})], ["O1", "O2"])
        # the log task asks for a tag: any oak log satisfies it
        self.assertEqual([g["id"] for g in fake_book().next_goals(2, {M + "oak_log": 1})], ["T1", "O2"])

    def test_a_quest_waits_for_its_dependency_but_is_still_offered(self):
        goals = fake_book().next_goals(4)
        sword = next(g for g in goals if g["id"] == "O2")
        self.assertFalse(sword["ready"])
        self.assertIn("前置", sword["reason"])
        # a task only the client cannot see (a kill) is never made a goal
        self.assertNotIn("O3", [g["id"] for g in goals])
        # a goal set aside is not picked again
        self.assertEqual([g["id"] for g in fake_book().next_goals(2, skip={M + "bread"})], ["O1", "O2"])

    def test_quest_for_item_including_tags(self):
        qb = fake_book()
        self.assertEqual([q["id"] for q in qb.quest_for_item(M + "iron_sword")], ["O2"])
        self.assertEqual([q["id"] for q in qb.quest_for_item(M + "oak_log")], ["O1"])   # asked for as a tag
        self.assertEqual(qb.quest_for_item("nothing:here"), [])

    def test_progress_is_remembered_and_shown(self):
        folder = tempfile.mkdtemp()
        try:
            path = os.path.join(folder, "done.json")
            qb = fake_book(progress=path)
            self.assertEqual(qb.check({M + "bread": 1}), ["Welcome"])
            self.assertEqual(QuestBook(FakeIndex(FAKE_QUESTS), progress=path).done, {"T1"})
            later = fake_book(progress=path)
            self.assertNotIn("T1", [g["id"] for g in later.next_goals(5)])
            progress = later.chapter_progress({M + "bread": 1})
            self.assertEqual([p["chapter"] for p in progress], ["Tutorial", "Overworld", "The Otherside", "Dragons"])
            self.assertEqual((progress[0]["done"], progress[0]["total"]), (1, 1))
            self.assertEqual([q["id"] for q in progress[1]["reachable"]], ["O1"])    # O2 still needs O1
            self.assertEqual(progress[1]["next"]["id"], "O1")
            line = later.progress_text({M + "bread": 1})
            self.assertIn("任务书：Tutorial 1/1", line)
            self.assertIn("下一步", line)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_an_empty_book_answers_instead_of_raising(self):
        for qb in (QuestBook(None), fake_book([]), QuestBook(FakeIndex(None))):
            self.assertEqual(qb.quests, [])
            self.assertEqual(qb.chapters, [])
            self.assertEqual(qb.next_goals(5), [])
            self.assertEqual(qb.next_goals(5, {M + "bread": 1}), [])
            self.assertEqual(qb.chapter_progress({}), [])
            self.assertEqual(qb.progress_text({}), "")
            self.assertEqual(qb.quest_for_item(M + "bread"), [])
            self.assertEqual(qb.check({}), [])
            self.assertEqual(qb.prerequisites_for("QS"), [])


# --- the same book over a small pack read from a game folder -----------------------------------------

def shaped(pattern, key, out, n=1):
    return {"type": "minecraft:crafting_shaped", "pattern": pattern, "key": key, "result": {"id": out, "count": n}}


FILES = {
    "data/minecraft/recipe/oak_planks.json": {"type": "minecraft:crafting_shapeless", "ingredients": [{"tag": "minecraft:oak_logs"}],
                                              "result": {"id": M + "oak_planks", "count": 4}},
    "data/minecraft/recipe/stick.json": shaped(["#", "#"], {"#": {"tag": "minecraft:planks"}}, M + "stick", 4),
    "data/minecraft/recipe/crafting_table.json": shaped(["##", "##"], {"#": {"tag": "minecraft:planks"}}, M + "crafting_table"),
    "data/minecraft/recipe/wooden_pickaxe.json": shaped(["XXX", " # ", " # "], {"X": {"tag": "minecraft:planks"}, "#": {"item": M + "stick"}},
                                                         M + "wooden_pickaxe"),
    "data/minecraft/recipe/stone_pickaxe.json": shaped(["XXX", " # ", " # "], {"X": {"item": M + "cobblestone"}, "#": {"item": M + "stick"}},
                                                       M + "stone_pickaxe"),
    "data/minecraft/recipe/furnace.json": shaped(["XXX", "X X", "XXX"], {"X": {"item": M + "cobblestone"}}, M + "furnace"),
    "data/minecraft/recipe/iron_ingot_from_smelting_raw_iron.json": {"type": "minecraft:smelting", "ingredient": {"item": M + "raw_iron"},
                                                                     "result": {"id": M + "iron_ingot"}},
    "data/minecraft/recipe/iron_sword.json": shaped(["X", "X", "#"], {"X": {"item": M + "iron_ingot"}, "#": {"item": M + "stick"}},
                                                    M + "iron_sword"),
    "data/minecraft/recipe/bow.json": shaped([" #S", "# S", " #S"], {"#": {"item": M + "stick"}, "S": {"item": M + "string"}}, M + "bow"),
    "data/minecraft/tags/item/planks.json": {"values": [M + "oak_planks"]},
    "data/minecraft/tags/item/oak_logs.json": {"values": [M + "oak_log"]},
    "data/minecraft/tags/block/mineable/pickaxe.json": {"values": [M + "stone", M + "iron_ore"]},
    "data/minecraft/tags/block/needs_stone_tool.json": {"values": [M + "iron_ore"]},
    "data/minecraft/loot_table/blocks/iron_ore.json": {"pools": [{"entries": [
        {"type": "minecraft:item", "name": M + "raw_iron"}]}]},
}

TUTORIAL = """{
	filename: "tutorial"
	id: "TUT"
	quests: [
		{ id: "QT1", tasks: [{ id: "A", item: { count: 1, id: "#minecraft:oak_logs" }, type: "item" }] }
		{ dependencies: ["QT1"] id: "QT2", tasks: [{ entity: "minecraft:zombie", id: "B", type: "kill" }] }
	]
}"""
OVERWORLD = """{
	filename: "overworld"
	id: "OW"
	quests: [
		{ id: "QS", tasks: [{ id: "C", item: { count: 1, id: "minecraft:iron_sword" }, type: "item" }] }
		{ id: "QB", tasks: [{ id: "D", item: { id: "minecraft:bow" }, type: "item" }] }
	]
}"""
LANG = """{
	chapter.TUT.title: " &aTutorial"
	chapter.OW.title: " &bOverworld"
	quest.QT1.title: "Obtain a Wooden Log"
	quest.QS.title: "Craft an Iron Sword"
	quest.QB.title: "Craft a Bow"
}"""


def make_pack(root):
    os.makedirs(os.path.join(root, "mods"))
    with zipfile.ZipFile(os.path.join(root, "mods", "pack.jar"), "w") as z:
        for name, doc in FILES.items():
            z.writestr(name, json.dumps(doc))
    qdir = os.path.join(root, "config", "ftbquests", "quests")
    os.makedirs(os.path.join(qdir, "chapters"))
    os.makedirs(os.path.join(qdir, "lang"))
    with open(os.path.join(qdir, "chapters", "tutorial.snbt"), "w", encoding="utf-8") as f:
        f.write(TUTORIAL)
    with open(os.path.join(qdir, "chapters", "overworld.snbt"), "w", encoding="utf-8") as f:
        f.write(OVERWORLD)
    with open(os.path.join(qdir, "lang", "en_us.snbt"), "w", encoding="utf-8") as f:
        f.write(LANG)


class TestOverAPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        make_pack(cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_a_game_folder_becomes_a_book(self):
        qb = QuestBook(self.dir)
        self.assertIsInstance(qb.index, ModpackIndex)
        self.assertEqual([c["title"] for c in qb.chapters], ["Tutorial", "Overworld"])
        self.assertEqual([c["key"] for c in qb.chapters], ["tutorial", "overworld"])
        self.assertEqual(len(qb.quests), 4)
        self.assertEqual({q["chapter"] for q in qb.quests}, {"Tutorial", "Overworld"})

    def test_a_reachable_quest_is_a_goal_and_a_mob_one_is_not(self):
        goals = QuestBook(self.dir).next_goals(5)
        self.assertEqual(goals[0]["id"], "QT1")
        self.assertIn("QS", [g["id"] for g in goals])
        self.assertIn("QB", [g["id"] for g in goals])  # hunt_resource can now collect spider string
        sword = next(g for g in goals if g["id"] == "QS")
        self.assertEqual((sword["item"], sword["count"], sword["text"]), (M + "iron_sword", 1, "Overworld: Craft an Iron Sword"))
        # the log is already in the bag: that task is done, so it is not offered
        self.assertNotIn("QT1", [g["id"] for g in QuestBook(self.dir).next_goals(5, {M + "oak_log": 1})])

    def test_the_item_chain_behind_a_quest(self):
        qb = QuestBook(self.dir)
        steps = qb.prerequisites_for("QS", {M + "stone_pickaxe": 1, M + "stick": 4},
                                     stations={M + "crafting_table", M + "furnace"})
        self.assertEqual((steps[0]["kind"], steps[0]["item"], steps[0]["count"]), ("gather", M + "raw_iron", 2))
        self.assertIn(M + "coal", [s["item"] for s in steps])                # no fuel yet: coal first
        self.assertIn("smelt", [s["kind"] for s in steps])
        self.assertEqual((steps[-1]["kind"], steps[-1]["item"]), ("craft", M + "iron_sword"))
        self.assertTrue(all(s["goal"] == M + "iron_sword" for s in steps))
        self.assertEqual(len(steps), len({(s["kind"], s["item"]) for s in steps}))   # no step repeated
        self.assertEqual(qb.prerequisites_for("nope"), [])

    def test_per_chapter_progress(self):
        qb = QuestBook(self.dir)
        progress = qb.chapter_progress({M + "oak_log": 1})
        self.assertEqual((progress[0]["chapter"], progress[0]["done"], progress[0]["total"]), ("Tutorial", 1, 2))
        self.assertEqual([q["id"] for q in progress[1]["reachable"]], ["QS", "QB"])
        self.assertEqual(progress[1]["next"]["title"], "Craft an Iron Sword")
        line = qb.progress_text({M + "oak_log": 1})
        self.assertIn("Tutorial 1/2", line)
        self.assertIn("下一步", line)


class TestTheVeteranManual(unittest.TestCase):
    SECTIONS = ("老手进度清单", "装备升级的时机", "高效挖矿：高度与分支间距", "战斗细节与药水",
                "选家与 5×5 起步屋", "夜晚安全", "背包管理", "FTB 任务优先级")

    def test_the_new_manual_is_read_by_her_handbook(self):
        with open(EXPERT, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("老手手册", text)
        for title in self.SECTIONS:
            self.assertIn(title, text)
        hb = Handbook(EXPERT)
        topics = hb.topics()
        for title in self.SECTIONS:
            self.assertIn(title, topics)
        self.assertIn("发展路线", topics)                      # the know-how she already had is still there
        self.assertEqual(hb.find("任务优先级")[0][0], "FTB 任务优先级")
        self.assertEqual(hb.find("钻石在哪挖")[0][0], "挖矿")    # and the older section still wins its topic
        self.assertIn("支道", hb.section("高效挖矿：高度与分支间距"))
        self.assertIn("盾牌", hb.section("战斗细节与药水"))


class TestRealPackSemantics(unittest.TestCase):
    def test_lazy_index_is_loaded_before_reading_quest_list(self):
        class Lazy(FakeIndex):
            def __init__(self):
                super().__init__([])
                self.calls = 0

            def load(self):
                self.calls += 1
                self.quests = FAKE_QUESTS

        index = Lazy()
        qb = QuestBook(index)
        self.assertEqual(index.calls, 1)
        self.assertEqual(len(qb.quests), len(FAKE_QUESTS))

    def test_unknown_kill_dependency_cannot_be_skipped_transitively(self):
        qs = [quest("K", "Overworld", "Kill boss", [{"type": "kill", "entity": M + "warden", "count": 1}]),
              quest("I", "Overworld", "Iron", [item(M + "iron_ingot")], ["K"]),
              quest("S", "Overworld", "Sword", [item(M + "iron_sword")], ["I"])]
        qb = QuestBook(FakeIndex(qs))
        sword = next(g for g in qb.next_goals(5, {M + "iron_ingot": 1}) if g["id"] == "S")
        self.assertFalse(sword["ready"])
        self.assertEqual(sword["dependency_chain"], ["K", "I", "S"])
        self.assertIn("K", sword["unverified_dependencies"])
        self.assertEqual(qb.task_status("K", {M + "sculk": 99})[0]["status"], "unknown")

    def test_one_and_minimum_dependencies_are_preserved(self):
        roots = [quest("A", "Tutorial", "A", [item(M + "bread")]),
                 quest("B", "Tutorial", "B", [item(M + "apple")])]
        dependent = quest("C", "Overworld", "C", [item(M + "iron_sword")], ["A", "B"])
        dependent["dependency_requirement"] = "one_completed"
        qb = QuestBook(FakeIndex(roots + [dependent]))
        self.assertTrue(next(g for g in qb.next_goals(5, {M + "bread": 1}) if g["id"] == "C")["ready"])
        dependent["dependency_requirement"] = "all_completed"
        dependent["min_required_dependencies"] = 1
        qb = QuestBook(FakeIndex(roots + [dependent]))
        self.assertTrue(next(g for g in qb.next_goals(5, {M + "bread": 1}) if g["id"] == "C")["ready"])

    def test_dimension_visit_is_observed_but_not_server_verified(self):
        qb = QuestBook(FakeIndex([quest("N", "Nether", "Enter Nether", [{"type": "dimension", "dimension": M + "the_nether"}])]))
        state = {"dimension": M + "overworld", "portal": {"x": 1, "y": 64, "z": 3}, "in_world": True}
        self.assertEqual(qb.next_actions(state, capabilities=()), [])
        action = qb.next_actions(state, capabilities={"enter_portal"})[0]
        self.assertEqual(action["name"], "enter_portal")
        self.assertTrue(action["verification_required"])
        self.assertNotIn("N", qb.done_ids())
        qb.observe_state({"dimension": M + "the_nether", "in_world": False})
        self.assertNotIn("N", qb.done_ids())
        qb.observe_state({"dimension": M + "the_nether", "in_world": True})
        self.assertIn("N", qb.done_ids())
        self.assertFalse(qb.task_status("N")[0]["server_verified"])
        self.assertEqual(qb.next_actions(state, capabilities={"enter_portal"}), [])

    def test_structure_requires_known_position_and_never_means_completed(self):
        qb = QuestBook(FakeIndex([quest("S", "Aether", "Dungeon", [{"type": "structure", "structure": "aether:bronze_dungeon"}])]))
        state = {"dimension": "aether:the_aether"}
        self.assertEqual(qb.next_actions(state, capabilities={"go_to"}), [])
        state["known_structures"] = {"aether:bronze_dungeon": {"x": 10, "y": 70, "z": 12, "dimension": "aether:the_aether"}}
        self.assertEqual(qb.next_actions(state, capabilities={"go_to"})[0]["args"], {"x": 10, "y": 70, "z": 12})
        self.assertNotIn("S", qb.done_ids())

    def test_dependency_cycles_and_missing_ids_are_explained(self):
        qs = [quest("A", "Tutorial", "A", [item(M + "bread")], ["B"]),
              quest("B", "Tutorial", "B", [item(M + "apple")], ["A"]),
              quest("C", "Tutorial", "C", [item(M + "iron_ingot")], ["UNKNOWN"])]
        qb = QuestBook(FakeIndex(qs))
        self.assertEqual(qb.dependency_chain("A")["cycles"], ["A", "B"])
        self.assertEqual(qb.dependency_chain("C")["missing"], ["UNKNOWN"])
        self.assertTrue(all(not g["ready"] for g in qb.next_goals(5)))

    def test_snbt_preserves_breed_components_nested_counts_and_dependency_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            root = os.path.join(temp, "Noa [BMC5]")
            make_pack(root)
            path = os.path.join(root, "config", "ftbquests", "quests", "chapters", "dragons.snbt")
            with open(path, "w", encoding="utf-8") as f:
                f.write('''{ filename:"dragons", id:"DR", quests:[
                  {id:"EGG", dependencies:["QT1","QS"], min_required_dependencies:1, tasks:[
                    {id:"T",type:"item",match_components:"strict",item:{id:"dmr:dragon_egg",count:2,components:{"dmr:dragon_breed":"fire"}}}]}
                  {id:"KILL",tasks:[{id:"K",type:"kill",entity:"minecraft:zombie",value:5}]}
                ]}''')
            qb = QuestBook(ModpackIndex(root))
            self.assertEqual(qb.load_info["source"], "live_pack_snbt")
            self.assertEqual(qb.load_info["errors"], [])
            egg = qb._by_id["EGG"]
            self.assertEqual(egg["tasks"][0]["count"], 2)
            self.assertEqual(egg["tasks"][0]["item_components"], {"dmr:dragon_breed": "fire"})
            self.assertEqual(egg["min_required_dependencies"], 1)
            self.assertNotIn("EGG", qb.done_ids({"dmr:dragon_egg": 64}))
            self.assertNotIn("EGG", [g["id"] for g in qb.next_goals(50)])
            self.assertTrue(qb.task_status("EGG")[0]["requires_components"])
            self.assertEqual(qb._by_id["KILL"]["tasks"][0]["count"], 5)


if __name__ == "__main__":
    unittest.main()
