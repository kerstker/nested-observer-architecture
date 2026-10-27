"""Her map of the world: home, her chests and what is in them, where she found ore and where she died."""
import os
import tempfile
import unittest

from src.levels.l8_agency.mc_places import Places, direction
from src.levels.l8_agency.mc_skills import Skills


class TestPlaces(unittest.TestCase):
    def test_the_map_is_kept_and_read_like_a_person_would(self):
        path = os.path.join(tempfile.mkdtemp(), "places.json")
        p = Places(path)
        p.set_home(100, 70, -50)
        p.add("chest", 101, 70, -50, items={"minecraft:iron_ingot": 12, "minecraft:cobblestone": 64})
        p.add("ore", 40, 16, 10, note="minecraft:iron_ore", merge=8)
        p.add("ore", 43, 15, 12, note="minecraft:iron_ore", merge=8)            # the same vein
        again = Places(path)
        self.assertEqual(again.home["x"], 100)
        self.assertEqual(len([q for q in again.places if q["kind"] == "ore"]), 1)
        self.assertEqual(again.where_is("minecraft:iron_ingot")[0]["x"], 101)
        me = {"x": 0.0, "y": 70.0, "z": 0.0}
        lines = again.describe(me)
        self.assertTrue(lines[0].startswith("家：(100,70,-50)"))
        self.assertIn("东北", lines[0])                                           # +x east, -z north
        self.assertIn("iron_ingot×12", " ".join(lines))
        self.assertIn("低", direction(me, {"x": 0, "y": 20, "z": 40}))
        self.assertIn("南", direction(me, {"x": 0, "y": 70, "z": 40}))


class Hands:
    """Real slot stacks and carried cursor, including partial right-click transfers."""
    def __init__(self, bag):
        self.slots = {}
        slot = 9
        for item, count in bag.items():
            while count:
                n = min(64, count)
                self.slots[slot] = {"id": item, "count": n}
                slot += 1
                count -= n
        self.chest_slots = {}
        self.cursor = None
        self.open = False
        self.calls = []

    @property
    def bag(self):
        result = {}
        for v in self.slots.values():
            result[v["id"]] = result.get(v["id"], 0) + v["count"]
        return result

    def rpc(self, m, p):
        self.calls.append((m, p))
        if m == "player.getState":
            return {"x": .5, "y": 70., "z": .5, "yaw": 0., "pitch": 0., "food": 20}
        if m == "player.getInventory":
            return {"hotbar": [dict(v, slot=k) for k,v in self.slots.items() if k < 9],
                    "main": [dict(v, slot=k) for k,v in self.slots.items() if k >= 9], "selectedSlot": 0}
        if m == "container.open":
            self.open = True
        if m == "container.close":
            self.open = False
        if m == "container.state":
            return {"open": self.open, "size": 27, "type": "minecraft:generic_9x3",
                    "slots": [dict(v, slot=k) for k,v in self.chest_slots.items()]}
        if m == "container.click":
            assert self.open
            menu = p["slot"]
            slots = self.chest_slots if menu < 27 else self.slots
            slot = menu if menu < 27 else menu - 18 if menu < 54 else menu - 54
            if p.get("mode") == "quick_move":
                other = self.slots if menu < 27 else self.chest_slots
                stack = slots.pop(slot)
                empty = next(k for k in range(36 if menu < 27 else 27) if k not in other)
                other[empty] = stack
            elif p.get("button") == 1 and self.cursor:
                assert slot not in slots or slots[slot]["id"] == self.cursor["id"]
                slots.setdefault(slot, {"id": self.cursor["id"], "count": 0})["count"] += 1
                self.cursor["count"] -= 1
                if not self.cursor["count"]:
                    self.cursor = None
            else:
                old = slots.pop(slot, None)
                if self.cursor:
                    slots[slot] = self.cursor
                self.cursor = old
        return {"ok": True}


class TestChests(unittest.TestCase):
    def test_she_stores_what_she_does_not_carry_and_takes_it_back_when_needed(self):
        h = Hands({"minecraft:cobblestone": 200, "minecraft:raw_iron": 9, "minecraft:iron_pickaxe": 1, "minecraft:torch": 20})
        s = Skills(h.rpc)
        s._approach = lambda *a: True
        s._smooth_look = lambda *a: None
        s.places.add("chest", 2, 70, 0)
        out = s.store()
        self.assertIn("raw_iron×9", out)
        self.assertEqual(h.bag["minecraft:cobblestone"], 64)                      # a stack stays on her
        self.assertEqual(h.bag["minecraft:iron_pickaxe"], 1)                     # gear never goes in
        self.assertEqual(s.places.where_is("minecraft:raw_iron")[0]["items"]["minecraft:raw_iron"], 9)
        self.assertTrue(s._take_from_chest("minecraft:raw_iron", 3))
        self.assertEqual(h.bag["minecraft:raw_iron"], 3)
        self.assertEqual(s.places.where_is("minecraft:raw_iron")[0]["items"]["minecraft:raw_iron"], 6)
        self.assertFalse(s._take_from_chest("minecraft:diamond", 1))


if __name__ == "__main__":
    unittest.main()
