"""Veteran play planning (mc_pro): pure decisions for shelter, food, combat, home, mining and goal choice.

No game I/O here either: every test feeds a small state dict and checks the single action a good player would
take, exactly what `game.py` will consult before its own goal logic.
"""
import unittest

from src.levels.l8_agency.mc_pro import EARLY_GAME, ORE_Y, ProPlanner

M = "minecraft:"


def state(**over):
    """An ordinary healthy daytime state, overridden per test."""
    base = {"t": 1000.0, "hour": 9.0, "in_world": True, "pos": (0.0, 64.0, 0.0), "y": 64.0,
            "health": 20.0, "food": 20.0, "inventory": {}, "hostiles": [], "home": None, "bed": False,
            "tools": {}, "players_near": 0, "sky_above": True, "dimension": "minecraft:overworld"}
    base.update(over)
    return base


def stacks(n):
    """n distinct full stacks: an almost-full bag."""
    return {f"minecraft:junk{i}": 64 for i in range(n)}


class TestProPlanner(unittest.TestCase):
    # --- 1. the first hour ------------------------------------------------------------------------

    def test_early_game_begins_with_wood_and_follows_the_speedrun_checklist(self):
        planner = ProPlanner()
        action = planner.next_action(state())
        self.assertEqual((action["kind"], action["name"]), ("skill", "chop_tree"))
        self.assertIn("oak_log", action["args"]["prefer"])

        checklist = planner.early_game_checklist({})
        self.assertEqual(len(checklist), len(EARLY_GAME))
        self.assertEqual([c["order"] for c in checklist], list(range(1, len(EARLY_GAME) + 1)))
        self.assertEqual(checklist[0]["item"], M + "oak_log")
        self.assertFalse(any(c["done"] for c in checklist))

        bag = {M + "oak_log": 8, M + "crafting_table": 1, M + "wooden_pickaxe": 1, M + "stone_pickaxe": 1,
               M + "stone_axe": 1, M + "stone_sword": 1, M + "furnace": 1, M + "torch": 20, M + "bread": 3}
        ticked = planner.early_game_checklist(bag)
        first_open = next(c for c in ticked if not c["done"])
        self.assertEqual(first_open["item"], M + "iron_pickaxe")
        self.assertTrue(all(c["done"] for c in ticked if c["order"] < first_open["order"]))

    # --- 2. night and shelter ----------------------------------------------------------------------

    def test_night_without_bed_or_home_digs_in(self):
        planner = ProPlanner()
        bag = {M + "stone_pickaxe": 1, M + "torch": 8, M + "cobblestone": 16}
        action = planner.next_action(state(hour=22.0, inventory=bag, tools={"pickaxe": 1}))
        self.assertEqual((action["kind"], action["name"]), ("shelter", "dig_in"))
        self.assertEqual(action["args"]["depth"], 3)
        self.assertTrue(action["args"]["seal"])

        # a home far away is better than a hole in the ground
        action = planner.next_action(state(hour=22.0, inventory=bag, tools={"pickaxe": 1},
                                           home={"x": 400, "y": 64, "z": 400}))
        self.assertEqual((action["kind"], action["name"]), ("home", "go_home"))

        # at home after dark: no exploring, do the night's chores inside
        kit = {M + "cobblestone": 64, M + "torch": 32, M + "white_bed": 1, M + "chest": 2,
               M + "crafting_table": 1, M + "furnace": 1, M + "oak_door": 1}
        action = planner.next_action(state(hour=22.0, inventory=kit, home={"x": 0, "y": 64, "z": 0},
                                           hut={k: True for k in ("walls", "roof", "windows", "door", "light", "furniture", "safety")}))
        self.assertEqual((action["kind"], action["name"]), ("wait", "night_work"))
        self.assertIsNone(planner.night_plan(state(hour=12.0, inventory=kit)))

    # --- 3. combat ---------------------------------------------------------------------------------

    def test_creeper_or_low_hp_retreats_and_gear_decides_the_fight(self):
        planner = ProPlanner()
        creeper = {"type": M + "creeper", "distance": 3.0, "x": 1, "y": 64, "z": 1, "uuid": "c"}
        action = planner.combat_plan(state(health=8.0, hostiles=[creeper],
                                          tools={"sword": 2, "shield": True}))
        self.assertEqual(action["kind"], "fight")
        self.assertIn(action["name"], ("flee", "retreat"))
        self.assertEqual(action["args"]["mode"], "flee")

        # while eating/backing off is a body reflex, the planner must say it first thing
        action = planner.next_action(state(health=8.0, hostiles=[creeper], tools={"sword": 2}))
        self.assertEqual((action["kind"], action["name"]), ("fight", "flee"))

        zombie = {"type": M + "zombie", "distance": 3.0, "x": 1, "y": 64, "z": 1, "uuid": "z"}
        action = planner.combat_plan(state(health=18.0, hostiles=[zombie],
                                           tools={"sword": 2, "shield": True}))
        self.assertEqual(action["name"], "engage")
        self.assertTrue(action["args"]["crit"])

        skeleton = {"type": M + "skeleton", "distance": 12.0, "x": 10, "y": 64, "z": 10, "uuid": "s"}
        action = planner.combat_plan(state(health=18.0, hostiles=[skeleton],
                                           tools={"sword": 2, "shield": True}))
        self.assertEqual(action["name"], "shield")

        unarmed = planner.combat_plan(state(health=14.0, hostiles=[zombie], tools={}))
        self.assertEqual(unarmed["name"], "retreat")
        self.assertTrue(planner.boss_prep_checklist(state()))
        self.assertIsNone(planner.combat_plan(state()))

    # --- 4. food and health ------------------------------------------------------------------------

    def test_low_food_eats_cooks_or_hunts(self):
        planner = ProPlanner()
        action = planner.next_action(state(health=16.0, food=8.0))
        self.assertEqual((action["kind"], action["name"]), ("skill", "hunt"))

        action = planner.next_action(state(health=16.0, food=12.0,
                                           inventory={M + "bread": 3}))
        self.assertEqual((action["kind"], action["name"]), ("wait", "eat"))
        self.assertEqual(action["args"]["item"], M + "bread")

        # raw meat at home: cook it before eating it
        action = planner.survival_plan(state(food=12.0, home={"x": 0, "y": 64, "z": 0},
                                             at_home=True, inventory={M + "beef": 2}))
        self.assertEqual((action["kind"], action["name"]), ("skill", "smelt"))

        # nearly dead: eat/regen, never start a job
        action = planner.next_action(state(health=5.0, food=16.0, inventory={M + "bread": 1}))
        self.assertEqual((action["kind"], action["name"]), ("wait", "eat"))

    # --- 5. mining ---------------------------------------------------------------------------------

    def test_diamond_goal_mines_at_y_minus_58_with_spacing_and_torches(self):
        planner = ProPlanner()
        self.assertEqual(ORE_Y[M + "diamond"], -58)
        bag = {M + "torch": 32, M + "iron_pickaxe": 1}
        action = planner.mining_plan(state(goal=M + "diamond", y=16.0, inventory=bag,
                                           tools={"pickaxe": 2}))
        self.assertEqual((action["kind"], action["name"]), ("skill", "dig_down"))
        self.assertEqual(action["args"]["to_y"], -58)

        action = planner.mining_plan(state(goal=M + "diamond", y=-58.0, inventory=bag,
                                           tools={"pickaxe": 2}))
        self.assertEqual(action["name"], "tunnel")
        self.assertEqual(action["args"]["spacing"], 3)
        self.assertEqual(action["args"]["torch_every"], 8)
        self.assertIn("deepslate_diamond_ore", action["args"]["ores"])
        # planning fields do not leak into the skills call
        self.assertEqual(set(planner.skill_kwargs(action["name"], action["args"])), {"length", "ores"})

        # a stone pickaxe cannot get diamond: the veteran makes an iron one first
        action = planner.mining_plan(state(goal=M + "diamond", y=16.0, inventory=bag,
                                           tools={"pickaxe": 1}))
        self.assertEqual((action["kind"], action["name"]), ("goal", M + "iron_pickaxe"))

        # no torches and no coal: torch fuel before the mine trip
        action = planner.mining_plan(state(goal=M + "raw_iron", y=16.0, inventory={M + "stone_pickaxe": 1},
                                           tools={"pickaxe": 1}))
        self.assertEqual((action["kind"], action["name"]), ("skill", "mine"))
        self.assertIn("coal_ore", action["args"]["blocks"])

    # --- 6. quest vs ladder ------------------------------------------------------------------------

    def test_a_cheap_quest_beats_a_far_ladder_rung(self):
        planner = ProPlanner()
        quest_goals = [{"id": "q1", "chapter": "Overworld", "title": "烤面包", "item": M + "bread",
                        "count": 3, "cost": 4.0, "ready": True, "items": [[M + "bread", 3]],
                        "reason": "开局，要面包×3"}]
        choice = planner.choose_goal(state(), quest_goals, [{"item": M + "diamond_pickaxe", "count": 1}])
        self.assertEqual(choice["by"], "quest")
        self.assertEqual(choice["item"], M + "bread")
        self.assertGreater(choice["score"], 0)

        # a wooden pickaxe is a basic: it wins over any quest, however cheap
        choice = planner.choose_goal(state(inventory={M + "oak_log": 8}),
                                     quest_goals, [{"item": M + "wooden_pickaxe", "count": 1}])
        self.assertEqual((choice["by"], choice["item"]), ("ladder", M + "wooden_pickaxe"))

    # --- 7. anti-thrash ----------------------------------------------------------------------------

    def test_a_goal_that_just_got_stuck_is_shelved_and_switched(self):
        planner = ProPlanner()
        bag = {M + "stone_pickaxe": 1, M + "stone_axe": 1, M + "stone_sword": 1, M + "furnace": 1,
               M + "torch": 16, M + "cobblestone": 32}
        one = state(goal=M + "iron_pickaxe", goal_stuck=True, inventory=bag,
                    tools={"pickaxe": 1, "sword": 1},
                    quest_goals=[{"id": "q2", "chapter": "Overworld", "title": "盾牌", "item": M + "shield",
                                  "count": 1, "cost": 6.0, "ready": True, "items": [[M + "shield", 1]]}])
        choice = planner.choose_goal(one)
        self.assertIsNotNone(choice)
        self.assertNotEqual(choice["item"], M + "iron_pickaxe")
        self.assertIn(M + "iron_pickaxe", planner.shelved_items(1000.0))

        # once it is in the skip set it is not chosen again
        again = planner.choose_goal(state(goal=M + "iron_pickaxe", goal_stuck=True, inventory=bag,
                                          tools={"pickaxe": 1, "sword": 1}))
        self.assertNotEqual(again["item"], M + "iron_pickaxe")

    # --- 8. bag full -------------------------------------------------------------------------------

    def test_a_full_bag_stores_at_home_first_then_declutters(self):
        planner = ProPlanner()
        action = planner.inventory_plan(state(inventory=stacks(35)))
        self.assertEqual((action["kind"], action["name"]), ("skill", "declutter"))

        action = planner.inventory_plan(state(inventory=stacks(35),
                                              home={"x": 200, "y": 64, "z": 200}))
        self.assertEqual((action["kind"], action["name"]), ("home", "go_home"))
        self.assertEqual(action["args"]["for"], "store")

        action = planner.inventory_plan(state(inventory=stacks(35), home={"x": 2, "y": 64, "z": 2}, chest_near=True))
        self.assertEqual((action["kind"], action["name"]), ("skill", "store"))
        self.assertIsNone(planner.inventory_plan(state()))

    # --- 9. home site and the 5x5 starter hut ------------------------------------------------------

    def test_home_site_and_starter_hut_plan(self):
        planner = ProPlanner()
        bag = {M + "cobblestone": 32, M + "torch": 16}
        action = planner.home_plan(state(inventory=bag, tools={"pickaxe": 1}, hour=10.0))
        self.assertEqual((action["kind"], action["name"]), ("home", "set_home"))
        self.assertEqual(action["args"]["size"], 5)
        stages = {step["stage"] for step in action["args"]["hut"]}
        self.assertTrue({"walls", "door", "roof", "light", "furniture", "safety"} <= stages)
        self.assertIn("later", action["args"])

        kit = {M + "cobblestone": 64, M + "torch": 32, M + "white_bed": 1,
               M + "chest": 2, M + "crafting_table": 1, M + "furnace": 1, M + "oak_door": 1}
        self.assertIsNone(planner.home_plan(state(inventory=kit, home={"x": 0, "y": 64, "z": 0},
                                                 hour=10.0, hut={k: True for k in ("walls", "roof", "windows", "door", "light", "furniture", "safety")})))
        unbuilt = planner.home_plan(state(inventory=kit, home={"x": 0, "y": 64, "z": 0}, hour=10.0))
        self.assertEqual(unbuilt["name"], "build_hut")
        self.assertIn("walls", unbuilt["args"]["missing"])

        # missing walls and nothing to build with: get blocks first
        action = planner.home_plan(state(inventory={M + "bread": 3}, home={"x": 0, "y": 64, "z": 0},
                                           hour=10.0))
        self.assertEqual((action["kind"], action["name"]), ("skill", "mine"))

        # far from home late in the day: head back
        action = planner.home_plan(state(pos=(300.0, 64.0, 0.0), hour=18.0,
                                         home={"x": 0, "y": 64, "z": 0}))
        self.assertEqual((action["kind"], action["name"]), ("home", "go_home"))

    # --- 10. tools ---------------------------------------------------------------------------------

    def test_worn_tool_is_replaced_before_it_breaks_and_wooden_upgrades_early(self):
        planner = ProPlanner()
        action = planner.tool_plan(state(tools={"pickaxe": {"tier": 2, "left": 5}}))
        self.assertEqual((action["kind"], action["name"]), ("skill", "craft"))
        self.assertTrue(action["args"]["item"].endswith("_pickaxe"))

        action = planner.tool_plan(state(tools={"pickaxe": {"tier": 2, "left": 5}},
                                         inventory={M + "iron_ingot": 3}))
        self.assertEqual(action["args"]["item"], M + "iron_pickaxe")

        action = planner.tool_plan(state(inventory={M + "cobblestone": 8, M + "wooden_pickaxe": 1},
                                         tools={"pickaxe": 0}))
        self.assertEqual((action["kind"], action["name"]), ("goal", M + "stone_pickaxe"))
        self.assertIsNone(planner.tool_plan(state(tools={"pickaxe": 2})))

    # --- 11. death, offline ------------------------------------------------------------------------

    def test_death_recovery_and_offline_wait(self):
        planner = ProPlanner()
        self.assertEqual(planner.next_action({"in_world": False})["name"], "in_world")
        self.assertEqual(planner.next_action({"dead": True, "in_world": True})["name"], "respawn")

        action = planner.next_action(state(death={"pos": (120, 64, 120), "recovered": False}))
        self.assertEqual((action["kind"], action["name"]), ("walk", "go_to"))
        self.assertEqual(action["args"]["x"], 120)
        self.assertIsNone(planner.death_plan(state(death={"pos": (120, 64, 120), "recovered": True})))
        self.assertIsNone(planner.death_plan(state()))

    # --- 12. lava and cliffs -----------------------------------------------------------------------

    def test_lava_or_cliff_makes_her_back_away_first(self):
        planner = ProPlanner()
        action = planner.next_action(state(hazard={"kind": "lava", "pos": (2.0, 63.0, 0.0),
                                                   "distance": 2.0}))
        self.assertEqual((action["kind"], action["name"]), ("walk", "go_to"))
        self.assertLess(action["args"]["x"], 0)          # away from the lava, not into it

        action = planner.next_action(state(hazard={"kind": "cliff", "pos": (0.0, 64.0, 2.0),
                                                   "distance": 1.0}))
        self.assertEqual(action["kind"], "walk")
        self.assertLess(action["args"]["z"], 0)
        self.assertIsNone(planner.hazard_plan(state()))
        self.assertNotEqual(planner.next_action(state())["kind"], "walk")


if __name__ == "__main__":
    unittest.main()
