"""
Level 8n: her map of the world - where home is, where she put things, where she has been.

A player keeps a map in their head: home is over there, the chest by the door has the iron, the mine is down
the stairs behind the house, she died by that ravine. This is that map, kept across sessions:

  home, her chests (with what she put in them), furnaces, crafting tables, beds, mines, ore she found,
  the place she last died, and spots she wants to remember.

Distances and directions are given the way a person thinks of them ("east, 120 blocks, 30 lower").
"""
import json
import math
import os
import logging
import time
from typing import Any, Dict, List, Optional

LABELS = {"home": "家", "chest": "箱子", "furnace": "熔炉", "crafting_table": "工作台", "bed": "床", "mine": "矿洞",
          "ore": "矿", "death": "死掉的地方", "spot": "地点", "portal": "传送门"}


def direction(me: Dict[str, float], p: Dict[str, Any]) -> str:
    dx, dz = p["x"] - me["x"], p["z"] - me["z"]
    dist = math.hypot(dx, dz)
    dy = p["y"] - me["y"]
    height = f"，高 {int(dy)} 格" if dy > 4 else f"，低 {int(-dy)} 格" if dy < -4 else ""
    if dist < 3:
        return "就在这" + height
    ang = math.degrees(math.atan2(dx, -dz)) % 360            # 0 = north (-z), 90 = east (+x)
    name = ("北", "东北", "东", "东南", "南", "西南", "西", "西北")[int((ang + 22.5) // 45) % 8]
    return f"在{name}边 {int(dist)} 格{height}"


class Places:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.active_dim = "minecraft:overworld"
        self.on_change = None
        self.places: List[Dict[str, Any]] = []
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    self.places = json.load(f)
            except (OSError, ValueError):
                pass

    def _save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.places, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def add(self, kind: str, x: float, y: float, z: float, note: str = "", items: Optional[Dict[str, int]] = None,
            merge: float = 2.0, dim: Optional[str] = None) -> Dict[str, Any]:
        """Remember a place (the same kind within `merge` blocks is the same place, updated)."""
        x, y, z = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        dim = dim or self.active_dim
        for p in self.places:
            if (p["kind"] == kind and p.get("dim", "minecraft:overworld") == dim
                    and math.dist((p["x"], p["y"], p["z"]), (x, y, z)) <= merge):
                changed = (bool(note) and note != p.get("note")) or (items is not None and items != p.get("items"))
                p.update(t=time.time(), **({"note": note} if note else {}))
                if items is not None:
                    p["items"] = dict(items)
                self._save()
                if changed:
                    self._notify(p)
                return p
        p = {"kind": kind, "x": x, "y": y, "z": z, "note": note, "t": time.time(), "dim": dim}
        if items is not None:
            p["items"] = dict(items)
        self.places.append(p)
        if kind in ("ore", "death", "spot"):                   # these pile up: keep the latest ones
            same = [q for q in self.places if q["kind"] == kind]
            for old in sorted(same, key=lambda q: q["t"])[:-40]:
                self.places.remove(old)
        self._save()
        self._notify(p)
        return p

    def _notify(self, place):
        if self.on_change:
            try:
                self.on_change({**place, "items": dict(place.get("items") or {})})
            except Exception:
                logging.getLogger(__name__).warning("[Game] map observer failed", exc_info=True)

    def forget_at(self, x: float, y: float, z: float, kind: Optional[str] = None):
        before = len(self.places)
        self.places = [p for p in self.places if not (p.get("dim", "minecraft:overworld") == self.active_dim
                                                      and (kind is None or p["kind"] == kind)
                                                      and math.dist((p["x"], p["y"], p["z"]), (x, y, z)) < 1.5)]
        if len(self.places) != before:
            self._save()

    @property
    def home(self) -> Optional[Dict[str, Any]]:
        return next((p for p in self.places if p["kind"] == "home"
                     and p.get("dim", "minecraft:overworld") == self.active_dim), None)

    def set_home(self, x: float, y: float, z: float, note: str = "") -> Dict[str, Any]:
        self.places = [p for p in self.places if p["kind"] != "home"
                       or p.get("dim", "minecraft:overworld") != self.active_dim]
        return self.add("home", x, y, z, note)

    def near(self, me: Dict[str, float], kinds=None, within: Optional[float] = None, n: int = 5) -> List[Dict[str, Any]]:
        dim = me.get("dim") or me.get("dimension") or self.active_dim
        out = [p for p in self.places if (kinds is None or p["kind"] in kinds)
               and p.get("dim", "minecraft:overworld") == dim]
        out.sort(key=lambda p: math.dist((p["x"], p["y"], p["z"]), (me["x"], me["y"], me["z"])))
        if within is not None:
            out = [p for p in out if math.dist((p["x"], p["y"], p["z"]), (me["x"], me["y"], me["z"])) <= within]
        return out[:n]

    def where_is(self, item: str) -> List[Dict[str, Any]]:
        """Her chests that hold this item, most of it first."""
        return sorted((p for p in self.places if p["kind"] == "chest"
                       and p.get("dim", "minecraft:overworld") == self.active_dim
                       and (p.get("items") or {}).get(item, 0) > 0),
                      key=lambda p: -p["items"][item])

    def describe(self, me: Optional[Dict[str, float]], name=lambda i: i.split(":")[-1]) -> List[str]:
        if not me:
            return []
        lines = []
        h = self.home
        lines.append(f"家：({h['x']},{h['y']},{h['z']})，{direction(me, h)}" if h else "家：还没有（找个地面上的好地方安个家）")
        for p in self.near(me, ("chest",), n=2):
            items = sorted((p.get("items") or {}).items(), key=lambda kv: -kv[1])[:6]
            stuff = "、".join(f"{name(i)}×{n}" for i, n in items) or "空的"
            lines.append(f"我的箱子 ({p['x']},{p['y']},{p['z']})，{direction(me, p)}：{stuff}")
        for p in self.near(me, ("furnace", "crafting_table", "bed", "mine"), within=48, n=3):
            lines.append(f"{LABELS.get(p['kind'], p['kind'])} ({p['x']},{p['y']},{p['z']})，{direction(me, p)}")
        ores = self.near(me, ("ore",), within=64, n=3)
        if ores:
            lines.append("记得的矿：" + "；".join(f"{name(p['note'])} {direction(me, p)}" for p in ores))
        d = self.near(me, ("death",), n=1)
        if d and time.time() - d[0]["t"] < 1800:
            lines.append(f"上次死掉的地方 ({d[0]['x']},{d[0]['y']},{d[0]['z']})，{direction(me, d[0])}（东西可能还在坟墓里）")
        return lines
