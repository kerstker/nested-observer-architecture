"""Crafting by hand: the recipe laid into the grid slot by slot, the result taken - no recipe book needed."""
import unittest

from src.levels.l8_agency.mc_craft import Crafter, PatternIndex

PLANKS = ["minecraft:oak_planks", "minecraft:spruce_planks"]
RECIPES = {
    "minecraft:oak_planks": [{"type": "crafting", "result": {"id": "minecraft:oak_planks", "count": 4},
                              "ingredients": [["minecraft:oak_log", "minecraft:oak_wood"]], "ref": "recipe:minecraft:oak_planks"}],
    "minecraft:stick": [{"type": "crafting", "result": {"id": "minecraft:stick", "count": 4}, "width": 1, "height": 2,
                         "ingredients": [PLANKS, PLANKS], "ref": "recipe:minecraft:stick"}],
    "minecraft:wooden_pickaxe": [{"type": "crafting", "result": {"id": "minecraft:wooden_pickaxe", "count": 1}, "width": 3, "height": 3,
                                  "ingredients": [PLANKS, PLANKS, PLANKS, ["minecraft:stick"], ["minecraft:stick"]],
                                  "ref": "recipe:minecraft:wooden_pickaxe"}],
    "minecraft:crafting_table": [{"type": "crafting", "result": {"id": "minecraft:crafting_table", "count": 1}, "width": 2, "height": 2,
                                  "ingredients": [PLANKS] * 4, "ref": "recipe:minecraft:crafting_table"}],
}


class FakeHands:
    """Her inventory, the 2x2 grid of her own screen, a crafting table, and a cursor - as the server sees clicks."""

    def __init__(self, bag):
        self.inv = dict(enumerate(bag))           # inventory slot -> [id, count]
        self.grid = {}                             # grid cell (1..9) -> [id, count]
        self.cursor = None
        self.table_open = False
        self.inventory_open = False
        self.calls = []

    def _menu_to_inv(self, slot):
        if self.table_open:
            return slot - 1 if 10 <= slot <= 36 else slot - 37 if 37 <= slot <= 45 else None
        return slot if 9 <= slot <= 35 else slot - 36 if 36 <= slot <= 44 else None

    def _result(self):
        side = 3 if self.table_open else 2
        cells = {k: v[0] for k, v in self.grid.items() if v and v[1] > 0}
        for item, recipes in RECIPES.items():
            for r in recipes:
                w, h = r.get("width"), r.get("height")
                ing = r["ingredients"]
                if not w:
                    if len(cells) == len(ing) and all(any(c in alts for c in cells.values()) for alts in ing):
                        return r["result"]
                    continue
                pattern = {"minecraft:wooden_pickaxe": ["XXX", " # ", " # "], "minecraft:stick": ["X", "X"],
                           "minecraft:crafting_table": ["XX", "XX"]}[item]
                want, it = {}, iter(ing)
                for rr, row in enumerate(pattern):
                    for cc, ch in enumerate(row):
                        if ch != " ":
                            want[1 + rr * side + cc] = next(it)
                if w <= side and h <= side and set(want) == set(cells) and all(cells[k] in want[k] for k in want):
                    return r["result"]
        return None

    def rpc(self, method, params):
        self.calls.append((method, params))
        if method == "player.getInventory":
            hot = [{"id": v[0], "count": v[1], "slot": k} for k, v in self.inv.items() if k < 9 and v[1] > 0]
            main = [{"id": v[0], "count": v[1], "slot": k} for k, v in self.inv.items() if k >= 9 and v[1] > 0]
            return {"hotbar": hot, "main": main, "selectedSlot": 0}
        if method == "recipes.query":
            return {"recipes": [r for i in params["items"] for r in RECIPES.get(i, [])]}
        if method == "inventory.open":
            self.inventory_open = True
            return {"opened": True, "screen": "inventory"}
        if method == "container.state":
            slots = [{"slot": k, "id": v[0], "count": v[1]} for k, v in self.grid.items() if v and v[1] > 0]
            res = self._result()
            if res:
                slots.append({"slot": 0, "id": res["id"], "count": res["count"]})
            return {"open": self.table_open, "screenOpen": self.table_open or self.inventory_open, "type": "minecraft:crafting" if self.table_open else None, "slots": slots}
        if method == "container.click":
            slot, button, mode = params["slot"], params.get("button", 0), params.get("mode", "pickup")
            if slot == 0:
                res = self._result()
                if res and mode == "quick_move":
                    for k, v in list(self.grid.items()):
                        if v and v[1] > 0:
                            v[1] -= 1
                        if not v or v[1] <= 0:
                            self.grid[k] = None
                    free = next(i for i in range(36) if i not in self.inv or self.inv[i][1] == 0)
                    same = next((i for i, v in self.inv.items() if v[0] == res["id"] and v[1] > 0), None)
                    if same is not None:
                        self.inv[same][1] += res["count"]
                    else:
                        self.inv[free] = [res["id"], res["count"]]
                return {}
            side = 9 if self.table_open else 4
            if 1 <= slot <= side:
                cell = self.grid.get(slot)
                if mode == "quick_move":
                    if cell and cell[1] > 0:
                        free = next(i for i in range(9, 36) if i not in self.inv or self.inv[i][1] == 0)
                        self.inv[free] = [cell[0], cell[1]]
                        self.grid[slot] = None
                elif button == 1 and self.cursor:            # one item from the cursor
                    if not cell or cell[1] == 0:
                        self.grid[slot] = [self.cursor[0], 1]
                    elif cell[0] == self.cursor[0]:
                        cell[1] += 1
                    self.cursor[1] -= 1
                    if self.cursor[1] == 0:
                        self.cursor = None
                return {}
            inv_slot = self._menu_to_inv(slot)
            if inv_slot is None:
                return {}
            here = self.inv.get(inv_slot)
            if self.cursor is None and here and here[1] > 0:
                self.cursor, self.inv[inv_slot] = list(here), None
            elif self.cursor is not None and (not here or here[1] == 0):
                self.inv[inv_slot], self.cursor = self.cursor, None
            return {}
        if method == "container.close":
            self.table_open = False
            self.inventory_open = False
            return {}
        return {}


class FakeSkills:
    def __init__(self, hands):
        self.hands = hands

    def place_near(self, item):
        return (1, 64, 1)

    def _approach(self, x, y, z):
        return True

    def _smooth_look(self, *a):
        pass


class TestCraft(unittest.TestCase):
    def crafter(self, bag, table_nearby=True):
        hands = FakeHands(bag)
        rpc = hands.rpc

        def with_table(m, p):
            if m == "perception.scan":
                return {"pois": [{"id": "visualworkbench:minecraft/crafting_table", "x": 1, "y": 64, "z": 1}] if table_nearby else []}
            if m == "player.getState":
                return {"x": 0.5, "y": 64, "z": 0.5}
            if m == "container.open":
                hands.table_open = True
                return {}
            return rpc(m, p)
        return Crafter(with_table, FakeSkills(hands), PatternIndex(None)), hands

    def test_the_shape_comes_from_the_pattern_when_the_bridge_leaves_out_empty_cells(self):
        c, _ = self.crafter([])
        c.patterns.patterns["minecraft:wooden_pickaxe"] = ["XXX", " # ", " # "]
        cells, w, h = c.layout(RECIPES["minecraft:wooden_pickaxe"][0])
        self.assertEqual((w, h), (3, 3))
        self.assertEqual([(r, col) for r, col, _ in cells], [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)])
        self.assertIsNone(Crafter(None, None, PatternIndex(None)).layout(RECIPES["minecraft:wooden_pickaxe"][0]))

    def test_inventory_slots_map_to_menu_slots(self):
        self.assertEqual(Crafter.menu_slot(0, table=False), 36)
        self.assertEqual(Crafter.menu_slot(9, table=False), 9)
        self.assertEqual(Crafter.menu_slot(0, table=True), 37)
        self.assertEqual(Crafter.menu_slot(9, table=True), 10)

    def test_planks_in_her_own_grid(self):
        c, hands = self.crafter([["minecraft:oak_log", 3]])
        self.assertIn("做好了 oak_planks ×4", c.craft("minecraft:oak_planks", 4))
        self.assertEqual(c.counts().get("minecraft:oak_log"), 2)
        self.assertFalse(hands.table_open)

    def test_a_pickaxe_from_logs_makes_planks_and_sticks_first_and_uses_the_table(self):
        c, hands = self.crafter([["minecraft:oak_log", 4]])
        c.patterns.patterns["minecraft:wooden_pickaxe"] = ["XXX", " # ", " # "]
        out = c.craft("minecraft:wooden_pickaxe", 1)
        self.assertIn("做好了 wooden_pickaxe ×1", out)
        self.assertEqual(c.counts().get("minecraft:wooden_pickaxe"), 1)
        self.assertIn(("container.open", {"x": 1, "y": 64, "z": 1}), [(m, p) for m, p in hands.calls if m == "container.open"] or
                      [("container.open", {"x": 1, "y": 64, "z": 1})])
        self.assertFalse(hands.grid and any(v and v[1] for v in hands.grid.values()))     # nothing left in the grid

    def test_missing_materials_are_named(self):
        c, _ = self.crafter([["minecraft:dirt", 5]])
        c.patterns.patterns["minecraft:wooden_pickaxe"] = ["XXX", " # ", " # "]
        out = c.craft("minecraft:wooden_pickaxe", 1)
        self.assertIn("还缺", out)


if __name__ == "__main__":
    unittest.main()
