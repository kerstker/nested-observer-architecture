"""Knowing the pack and playing toward a goal: the index read from jars, the planner, the ladder, the executor."""
import json
import os
import shutil
import tempfile
import unittest
import zipfile

from src.levels.l8_agency.mc_goal import Planner, TABLE, FURNACE, next_goal, satisfied
from src.levels.l8_agency.mc_knowledge import Handbook, Knowledge, ModpackIndex, parse_snbt

M = "minecraft:"


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
    "data/minecraft/recipe/iron_sword.json": shaped(["X", "X", "#"], {"X": {"item": M + "iron_ingot"}, "#": {"item": M + "stick"}},
                                                    M + "iron_sword"),
    "data/minecraft/recipe/furnace.json": shaped(["XXX", "X X", "XXX"], {"X": {"item": M + "cobblestone"}}, M + "furnace"),
    "data/minecraft/recipe/iron_ingot_from_smelting_raw_iron.json": {"type": "minecraft:smelting", "ingredient": {"item": M + "raw_iron"},
                                                                     "result": {"id": M + "iron_ingot"}},
    "data/minecraft/recipe/iron_ingot_from_nuggets.json": shaped(["XXX", "XXX", "XXX"], {"X": {"item": M + "iron_nugget"}}, M + "iron_ingot"),
    "data/minecraft/recipe/iron_nugget.json": {"type": "minecraft:crafting_shapeless", "ingredients": [{"item": M + "iron_ingot"}],
                                               "result": {"id": M + "iron_nugget", "count": 9}},
    "data/minecraft/recipe/iron_nugget_from_smelting.json": {"type": "minecraft:smelting", "ingredient": [{"item": M + "iron_pickaxe"}, {"item": M + "iron_sword"}],
                                                             "result": {"id": M + "iron_nugget"}},
    "data/minecraft/recipe/bow.json": shaped([" #S", "# S", " #S"], {"#": {"item": M + "stick"}, "S": {"item": M + "string"}}, M + "bow"),
    "data/minecraft/tags/item/planks.json": {"values": [M + "oak_planks"]},
    "data/minecraft/tags/item/oak_logs.json": {"values": [M + "oak_log", M + "oak_wood"]},
    "data/minecraft/tags/block/mineable/pickaxe.json": {"values": [M + "stone", M + "iron_ore", "#minecraft:coal_ores"]},
    "data/minecraft/tags/block/coal_ores.json": {"values": [M + "coal_ore"]},
    "data/minecraft/tags/block/needs_stone_tool.json": {"values": [M + "iron_ore"]},
    "data/minecraft/loot_table/blocks/iron_ore.json": {"pools": [{"entries": [{"type": "minecraft:alternatives", "children": [
        {"type": "minecraft:item", "name": M + "iron_ore", "conditions": [{"condition": "silk"}]},
        {"type": "minecraft:item", "name": M + "raw_iron"}]}]}]},
    "assets/minecraft/lang/zh_cn.json": {"item.minecraft.iron_sword": "铁剑", "item.minecraft.iron_ingot": "铁锭",
                                         "block.minecraft.iron_ore": "铁矿石", "item.minecraft.raw_iron": "粗铁"},
}

QUESTS = """{
	filename: "overworld"
	id: "AAA"
	quests: [
		{
			id: "Q1"
			tasks: [{ id: "T1", item: { count: 1, id: "minecraft:iron_sword" }, type: "item" }]
			x: 1.5d
		}
		{
			dependencies: ["Q1"]
			id: "Q2"
			tasks: [{ entity: "minecraft:zombie", id: "T2", type: "kill", value: 3L }]
		}
	]
}"""


def make_pack(root):
    os.makedirs(os.path.join(root, "mods"))
    with zipfile.ZipFile(os.path.join(root, "mods", "pack.jar"), "w") as z:
        for name, doc in FILES.items():
            z.writestr(name, json.dumps(doc))
    q = os.path.join(root, "config", "ftbquests", "quests")
    os.makedirs(os.path.join(q, "chapters"))
    os.makedirs(os.path.join(q, "lang"))
    with open(os.path.join(q, "chapters", "overworld.snbt"), "w", encoding="utf-8") as f:
        f.write(QUESTS)
    with open(os.path.join(q, "lang", "en_us.snbt"), "w", encoding="utf-8") as f:
        f.write('{\n\tchapter.AAA.title: " &aOverworld"\n\tquest.Q1.title: "Get an &eIron Sword"\n}')


class PackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        make_pack(cls.dir)
        cls.ix = ModpackIndex(cls.dir, os.path.join(cls.dir, "cache.json")).load()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)


class TestIndex(PackTest):
    def test_what_the_pack_says(self):
        ix = self.ix
        self.assertEqual(ix.name(M + "iron_sword"), "铁剑")
        self.assertEqual(ix.tool_for(M + "iron_ore"), ("pickaxe", 1))
        self.assertEqual(ix.tool_for(M + "coal_ore"), ("pickaxe", 0))          # through a nested tag
        self.assertEqual(ix.drops[M + "iron_ore"][0], M + "raw_iron")              # not the silk-touch drop
        self.assertEqual(ix.dropped_by(M + "raw_iron"), [M + "iron_ore"])
        self.assertEqual(ix.smelted_from(M + "iron_ingot"), [M + "raw_iron"])
        self.assertEqual(ix.made_by(M + "iron_ingot")[0]["t"], "smelting")         # the ordinary way first
        self.assertEqual(ix.search("铁剑"), [M + "iron_sword"])
        self.assertIn(M + "iron_sword", ix.used_in(M + "iron_ingot"))
        self.assertEqual((ix.made_by(M + "iron_sword")[0]["w"], ix.made_by(M + "iron_sword")[0]["h"]), (1, 3))

    def test_quests_are_read(self):
        q = {x["id"]: x for x in self.ix.quests}
        self.assertEqual(q["Q1"]["title"], "Get an Iron Sword")
        self.assertEqual(q["Q1"]["chapter"], "Overworld")
        self.assertEqual(q["Q1"]["tasks"][0]["item"], M + "iron_sword")
        self.assertEqual(q["Q2"]["deps"], ["Q1"])
        self.assertIn("任务「Get an Iron Sword」", self.ix.describe(M + "iron_sword"))

    def test_the_cache_is_used_the_second_time(self):
        again = ModpackIndex(None, os.path.join(self.dir, "cache.json")).load()
        self.assertEqual(again.name(M + "iron_sword"), "铁剑")
        self.assertEqual(len(again.recipes), len(self.ix.recipes))

    def test_snbt(self):
        self.assertEqual(parse_snbt('{ a: 1.5d, b: [2L 3], "c:d": { e: "x y" }, f: true }'),
                         {"a": 1.5, "b": [2, 3], "c:d": {"e": "x y"}, "f": True})

    def test_handbook_and_asking(self):
        hb = Handbook()
        self.assertIn("挖矿", hb.topics())
        self.assertEqual(hb.find("钻石在哪挖")[0][0], "挖矿")
        k = Knowledge(self.ix, hb)
        out = k.ask("铁剑")
        self.assertIn("铁剑（minecraft:iron_sword）", out)
        self.assertIn("铁锭×2", out)
        self.assertIn("话题", k.ask(""))


class TestPlanner(PackTest):
    def step(self, goal, have, stations=(TABLE, FURNACE), n=1):
        return Planner(self.ix).next_step(goal, n, have, set(stations))

    def test_from_nothing_it_starts_with_a_tree(self):
        self.assertEqual(self.step(M + "iron_sword", {})["kind"], "chop")

    def test_the_chain_to_an_iron_sword(self):
        base = {M + "stone_pickaxe": 1, M + "stick": 4}
        s = self.step(M + "iron_sword", base)
        self.assertEqual((s["kind"], s["item"], s["count"], s["y"]), ("gather", M + "raw_iron", 2, 16))
        s = self.step(M + "iron_sword", dict(base, **{M + "raw_iron": 2}))
        self.assertEqual((s["kind"], s["item"]), ("gather", M + "coal"))           # no fuel yet
        s = self.step(M + "iron_sword", dict(base, **{M + "raw_iron": 2, M + "coal": 1}))
        self.assertEqual((s["kind"], s["input"], s["count"]), ("smelt", M + "raw_iron", 2))
        s = self.step(M + "iron_sword", dict(base, **{M + "iron_ingot": 2}))
        self.assertEqual((s["kind"], s["item"]), ("craft", M + "iron_sword"))
        self.assertEqual(self.step(M + "iron_sword", {M + "iron_sword": 1})["kind"], "done")

    def test_iron_ore_needs_a_stone_pickaxe_first(self):
        s = self.step(M + "iron_sword", {M + "wooden_pickaxe": 1, M + "stick": 4, M + "cobblestone": 3})
        self.assertEqual((s["kind"], s["item"]), ("craft", M + "stone_pickaxe"))

    def test_a_table_is_made_when_none_is_near(self):
        s = self.step(M + "wooden_pickaxe", {M + "oak_planks": 8, M + "stick": 2}, stations=())
        self.assertEqual((s["kind"], s["item"]), ("craft", TABLE))

    def test_her_gear_is_never_melted_down(self):
        s = self.step(M + "iron_nugget", {M + "iron_pickaxe": 1, M + "iron_sword": 1, M + "coal": 4, M + "stone_pickaxe": 1})
        self.assertNotEqual(s.get("input"), M + "iron_pickaxe")
        self.assertNotIn(s["kind"], ("smelt",))

    def test_bow_string_has_a_bounded_hunting_step(self):
        s = self.step(M + "bow", {M + "stick": 3})
        self.assertEqual(s["kind"], "hunt")
        self.assertEqual(s["mob"], M + "spider")
        self.assertEqual(s["item"], M + "string")


class TestQuests(PackTest):
    def test_quests_she_can_do_become_goals_and_are_remembered(self):
        from src.levels.l8_agency.mc_goal import QuestBook
        path = os.path.join(self.dir, "done.json")
        qb = QuestBook(self.ix, path)
        self.assertEqual([q["id"] for q in qb.item_quests()], ["Q1"])          # the kill quest is not an item quest
        q, item, n = qb.next({M + "stone_pickaxe": 1}, Planner(self.ix), set(), {TABLE, FURNACE})
        self.assertEqual((q["id"], item, n), ("Q1", M + "iron_sword", 1))
        self.assertIsNone(qb.next({}, Planner(self.ix), {M + "iron_sword"}))   # a goal set aside is not picked
        self.assertEqual(qb.check({M + "iron_sword": 1}), ["Get an Iron Sword"])
        self.assertEqual(QuestBook(self.ix, path).done, {"Q1"})
        self.assertIsNone(qb.next({}, Planner(self.ix)))


class TestLadder(unittest.TestCase):
    def test_the_usual_progression(self):
        self.assertEqual(next_goal({}), (M + "wooden_pickaxe", 1))
        self.assertEqual(next_goal({M + "iron_pickaxe": 1}), (M + "stone_axe", 1))     # an iron pickaxe beats wood and stone
        self.assertTrue(satisfied(M + "iron_helmet", 1, {}, {M + "diamond_helmet"}))
        self.assertFalse(satisfied(M + "iron_helmet", 1, {}, {M + "leather_helmet"}))
        self.assertEqual(next_goal({}, skip={M + "wooden_pickaxe"}), (M + "stone_pickaxe", 1))


class TestExecutor(PackTest):
    def test_she_works_through_the_chain_by_herself(self):
        from src.levels.l8_agency.mc_skills import Skills
        bag = {M + "stone_pickaxe": 1}
        done = []

        class Body(Skills):
            def _do(self, step):
                done.append(step["kind"] + ":" + step["item"].split(":")[-1])
                if step["kind"] == "chop":
                    bag[M + "oak_log"] = bag.get(M + "oak_log", 0) + 4
                elif step["kind"] == "gather":
                    bag[step["item"]] = bag.get(step["item"], 0) + step["count"]
                elif step["kind"] == "smelt":
                    bag[step["input"]] -= step["count"]
                    bag[step["item"]] = bag.get(step["item"], 0) + step["count"]
                else:
                    r = self.index.made_by(step["item"])[0]
                    for alts in r["in"]:
                        have = next(a for a in self.index.expand(alts) if bag.get(a, 0) > 0)
                        bag[have] -= 1
                    bag[step["item"]] = bag.get(step["item"], 0) + r["n"]
                return "ok"

            def _stations(self):
                return {TABLE, FURNACE}

        s = Body(lambda m, p: {}, index=self.ix)
        s.crafter.counts = lambda: {k: v for k, v in bag.items() if v > 0}
        out = s.goal(M + "iron_sword")
        self.assertIn("目标完成", out)
        self.assertIn("chop:oak_log", done)                                      # sticks need wood
        self.assertLess(done.index("gather:raw_iron"), done.index("smelt:iron_ingot"))
        self.assertIn("smelt:iron_ingot", done)
        self.assertEqual(done[-1], "craft:iron_sword")

    def test_a_step_that_never_works_is_given_up(self):
        from src.levels.l8_agency.mc_skills import Skills
        s = Skills(lambda m, p: {}, index=self.ix)
        s.crafter.counts = lambda: {M + "stone_pickaxe": 1, M + "stick": 2}
        s._stations = lambda: {TABLE, FURNACE}
        s._do = lambda step: "附近没找到"
        out = s.goal(M + "iron_sword")
        self.assertIn("卡住", out)
        self.assertTrue(s.goal_stuck)


if __name__ == "__main__":
    unittest.main()
