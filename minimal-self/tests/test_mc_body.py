"""Her body looking after itself: free hands, real mining, worn tools, armor and shield, food, reflexes first."""
import threading
import time
import unittest

from src.levels.l8_agency.mc_skills import Skills


class Body:
    """A tiny world: blocks below y=64 are stone, a bag, her gear, a container that may be open."""

    def __init__(self, bag=None, gear=None, food=20):
        self.blocks = {}
        self.me = {"x": 0.5, "y": 64.0, "z": 0.5, "yaw": 0.0, "pitch": 0.0, "food": food}
        self.bag = {i: list(v) for i, v in enumerate(bag if bag is not None else [["minecraft:iron_pickaxe", 1]])}
        self.gear = gear or {}
        self.open = False
        self.inventory_open = False
        self.cursor = None
        self.selected = 0
        self.calls = []

    def block(self, p):
        return self.blocks.get(p, "minecraft:stone" if p[1] < 64 else "minecraft:air")

    def rpc(self, m, p):
        self.calls.append((m, p))
        if m == "player.getState":
            return dict(self.me)
        if m == "player.getInventory":
            return {"hotbar": [{"id": v[0], "count": v[1], "slot": k} for k, v in self.bag.items() if k < 9],
                    "main": [{"id": v[0], "count": v[1], "slot": k} for k, v in self.bag.items() if k >= 9], "selectedSlot": 0}
        if m == "inventory.open":
            self.inventory_open = True
            return {"opened": True}
        if m == "inventory.selectHotbar":
            self.selected = p["slot"]
        if m == "inventory.click":
            slot = p["slot"]
            if slot == 40:
                current = self.gear.get("offHand", {})
                old = [current["id"], current.get("count", 1)] if current.get("id") else None
                self.gear["offHand"] = {"id": self.cursor[0], "count": self.cursor[1]} if self.cursor else {}
                self.cursor = old
            else:
                old = self.bag.pop(slot, None)
                if self.cursor:
                    self.bag[slot] = self.cursor
                self.cursor = old
        if m == "interact.useItem":
            it = self.bag.get(self.selected)
            if it and it[0].endswith("_chestplate"):
                self.gear["chest"] = {"id": it[0], "count": 1}
        if m == "player.getEquipment":
            return dict(self.gear)
        if m == "perception.blocks":
            return {"blocks": [{"id": self.block((q["x"], q["y"], q["z"]))} for q in p["positions"]]}
        if m == "interact.breakBlock":
            self.blocks[(p["x"], p["y"], p["z"])] = "minecraft:air"
            return {"started": True}
        if m == "nav.pathTo":
            # Navigation receives world coordinates, already centered by Skills.
            self.me.update({"x": p["x"], "y": float(p["y"]), "z": p["z"]})
            return {"started": True}
        if m == "nav.status":
            return {"active": False, "state": "reached"}
        if m == "container.state":
            return {"open": self.open, "screenOpen": self.open or self.inventory_open}
        if m == "container.close":
            self.open = False
            self.inventory_open = False
        return {}


class TestBody(unittest.TestCase):
    def test_a_screen_left_open_is_closed_before_working(self):
        b = Body()
        b.open = True
        s = Skills(b.rpc)
        s.free_hands()
        self.assertFalse(b.open)
        b.open = True
        s.using_screen = True                         # her own furnace: left alone
        s.free_hands()
        self.assertTrue(b.open)

    def test_tunnels_open_new_rock_not_old_tunnels(self):
        b = Body()
        b.me.update({"y": 20.0})
        for z in range(1, 12):                        # an old tunnel straight ahead (+z)
            b.blocks[(0, 20, z)] = b.blocks[(0, 21, z)] = "minecraft:air"
        s = Skills(b.rpc)
        out = s.tunnel(6)
        self.assertIn("新挖了 6 格", out)
        broke = [(p["x"], p["z"]) for m, p in b.calls if m == "interact.breakBlock"]
        self.assertTrue(all(x != 0 for x, _ in broke))          # it turned off the old tunnel into fresh stone
        self.assertEqual(s.heading[0] != 0, True)

    def test_too_deep_for_iron_she_climbs_back_up_by_stairs(self):
        b = Body()
        b.me.update({"y": 40.0})
        b.blocks[(0, 40, 0)] = b.blocks[(0, 41, 0)] = "minecraft:air"           # where she stands
        s = Skills(b.rpc)
        self.assertIn("y=44", s.dig_up(44))
        self.assertEqual([p["y"] for m, p in b.calls if m == "nav.pathTo"], [41, 42, 43, 44])
        # gravel over her own head: a step aside on the same level, then up again
        b.calls.clear()
        b.blocks[(0, 47, 4)] = "minecraft:gravel"
        self.assertIn("y=46", s.dig_up(46))
        self.assertEqual(b.blocks[(0, 47, 4)], "minecraft:gravel")
        b.me.update({"y": -58.0})
        s.mine = lambda ids, count: "附近没找到这些方块。"
        s.dig_up = lambda y: f"up to {y}"
        self.assertEqual(s.gather(["minecraft:iron_ore"], 3, 16), "up to 16")

    def test_in_a_narrow_tunnel_a_table_goes_against_the_wall(self):
        b = Body(bag=[["minecraft:crafting_table", 1]])
        b.me.update({"y": 20.0})
        b.blocks[(0, 20, 0)] = b.blocks[(0, 21, 0)] = "minecraft:air"           # where she stands
        b.blocks[(0, 20, 1)] = b.blocks[(0, 21, 1)] = "minecraft:air"           # the tunnel ahead ...
        b.blocks[(0, 19, 1)] = "minecraft:air"                                   # ... with a hole in its floor
        rpc = b.rpc

        def place(m, p):
            if m == "interact.placeBlock":
                d = {"up": (0, 1, 0), "down": (0, -1, 0), "west": (-1, 0, 0), "east": (1, 0, 0), "north": (0, 0, -1), "south": (0, 0, 1)}[p["face"]]
                b.blocks[(p["x"] + d[0], p["y"] + d[1], p["z"] + d[2])] = "minecraft:crafting_table"
            return rpc(m, p)
        b.rpc = place
        s = Skills(b.rpc)
        pos = s.place_near("minecraft:crafting_table")
        self.assertEqual(pos, (0, 20, 1))
        clicked = [p for m, p in b.calls if m == "interact.placeBlock"][0]
        self.assertNotEqual(clicked["face"], "up")                               # against a wall, not on the floor

    def test_no_pickaxe_no_digging_until_she_makes_one(self):
        b = Body(bag=[["minecraft:stone_axe", 1], ["minecraft:cobblestone", 10]])
        b.blocks[(0, 64, 1)] = "minecraft:stone"
        s = Skills(b.rpc)
        made = []

        def craft(item, n=1, depth=0):
            made.append(item)
            b.bag[5] = [item, 1]
            return "做好了"
        s.crafter.craft = craft
        self.assertTrue(s._break(0, 64, 1))
        self.assertEqual(made, ["minecraft:stone_pickaxe"])                     # made one first, then dug
        b2 = Body(bag=[["minecraft:stone_axe", 1]])
        b2.blocks[(0, 64, 1)] = "minecraft:stone"
        s2 = Skills(b2.rpc)
        s2.crafter.craft = lambda item, n=1, depth=0: "做 x 还缺：木棍"
        self.assertFalse(s2._break(0, 64, 1))                                    # not with an axe
        self.assertIn("没有镐子", s2.last_reason)

    def test_a_tool_about_to_break_is_replaced_first(self):
        b = Body(bag=[["minecraft:iron_pickaxe", 1]],
                 gear={"mainHand": {"id": "minecraft:iron_pickaxe", "damage": 245, "maxDamage": 250}})
        s = Skills(b.rpc)
        made = []
        s.crafter.craft = lambda item, n=1, depth=0: made.append(item) or "做好了"
        s.mind_the_wear("pickaxe")
        self.assertEqual(made, ["minecraft:iron_pickaxe"])
        b.bag[1] = ["minecraft:iron_pickaxe", 1]                     # a spare is there: nothing to make
        made.clear()
        s.mind_the_wear("pickaxe")
        self.assertEqual(made, [])

    def test_armor_goes_on_and_the_shield_into_the_off_hand(self):
        b = Body(bag=[["minecraft:iron_chestplate", 1], ["minecraft:shield", 1], ["minecraft:leather_helmet", 1]],
                 gear={"chest": {"empty": True}, "helmet": {"id": "minecraft:iron_helmet"}, "offHand": {"id": "minecraft:torch"}})
        s = Skills(b.rpc)
        out = s.dress()
        self.assertIn("iron_chestplate", out)
        self.assertNotIn("leather_helmet", out)                       # worse than what she wears
        self.assertEqual([p["slot"] for m, p in b.calls if m == "inventory.click"], [1, 40, 1])
        self.assertEqual(b.gear["offHand"]["id"], "minecraft:shield")
        self.assertFalse(b.inventory_open)
        self.assertIn(("interact.useItem", {"hand": "main"}), b.calls)

    def test_hungry_means_low_bar_and_little_food(self):
        s = Skills(Body(bag=[["minecraft:bread", 1]], food=12).rpc)
        self.assertTrue(s.hungry())
        self.assertFalse(Skills(Body(bag=[["minecraft:cooked_beef", 8]], food=12).rpc).hungry())
        self.assertFalse(Skills(Body(bag=[], food=20).rpc).hungry())

    def test_reflexes_hold_the_work(self):
        b = Body()
        s = Skills(b.rpc)
        s.pause_until = time.time() + 0.6
        t = time.time()
        s._me()
        self.assertGreater(time.time() - t, 0.5)


if __name__ == "__main__":
    unittest.main()
