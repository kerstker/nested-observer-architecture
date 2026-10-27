"""Bounded survival actions from an Eye of Ender to the End encounter.

No /locate, teleport, world writes or reward grants. All coordinates come from observed eye entities,
loaded client blocks or previously observed places. Each invocation performs one recoverable step.
"""
import math
import time

M = "minecraft:"


class Endgame:
    def __init__(self, skills):
        self.s = skills

    def me(self):
        return self.s.rpc("player.getState", {}) or {}

    def entities(self, radius=64):
        # Eyes, crystals and dragon-breath clouds are "other", excluded by the bridge's default.
        return (self.s.rpc("perception.entities", {"radius": radius,
                "kinds": ["other", "hostile", "passive"]}) or {}).get("entities") or []

    def scan(self, ids, radius=2):
        return (self.s.rpc("perception.scan", {"radius": radius, "find": ids, "findLimit": 64}) or {}).get("found") or []

    def step(self):
        me = self.me()
        if not me or self.s._stop.is_set():
            return "末地行动暂停：没有可靠的角色状态。"
        if float(me.get("health", 0)) < 14 or int(me.get("food", 0)) < 14:
            return "末地行动先恢复血量和饥饿，不在低状态继续推进。"
        dim = me.get("dimension")
        if dim == M + "the_end":
            return self.fight()
        if dim != M + "overworld":
            return "找要塞需要先回主世界。"
        found = self.scan([M + "end_portal_frame", M + "end_portal"])
        frames = [b for b in found if b.get("id") == M + "end_portal_frame"]
        active = [b for b in found if b.get("id") == M + "end_portal"]
        if active:
            p = min(active, key=lambda b: math.dist((b["x"], b["y"], b["z"]), (me["x"], me["y"], me["z"])))
            self.s.places.add("end_portal", p["x"], p["y"], p["z"], note="已观测到激活的末地门", dim=dim)
            return self.enter(p, M + "the_end")
        if frames:
            frame = min(frames, key=lambda b: math.dist((b["x"], b["y"], b["z"]), (me["x"], me["y"], me["z"])))
            self.s.places.add("stronghold", frame["x"], frame["y"], frame["z"], note="已观测末地门框", dim=dim)
            if me["y"] > frame["y"] + 4:
                return self.s.dig_down(frame["y"] + 1, max_steps=12)
            if math.hypot(me["x"] - frame["x"], me["z"] - frame["z"]) > 12:
                return self.s.go_to(frame["x"], frame["y"] + 1, frame["z"])
            return self.fill_frames(frames)
        return self.follow_eye()

    def follow_eye(self):
        me = self.me()
        if me.get("dimension") != M + "overworld":
            return "末影之眼导航只在主世界使用。"
        route = getattr(self.s, "_eye_route", None)
        if route and route["remaining"] > 0:
            dx, dz = route["dx"], route["dz"]
            distance = min(96, route["remaining"])
            out = self.s.go_to(round(me["x"] + dx * distance), round(me["y"]), round(me["z"] + dz * distance))
            after = self.me()
            moved = math.hypot(after.get("x", me["x"]) - me["x"], after.get("z", me["z"]) - me["z"])
            route["remaining"] -= min(distance, moved)
            if moved < 2:
                self.s._eye_route = None     # blocked terrain requires fresh planning, not a perpetual leg
            return out
        # Preserve at least the maximum twelve eyes needed for a naturally generated portal.
        if self.s.crafter.counts().get(M + "ender_eye", 0) <= 12:
            return "需要补充末影之眼；保留12颗填门，导航眼不够了。"
        before = {e.get("uuid") for e in self.entities() if e.get("type") == M + "eye_of_ender"}
        self.s.free_hands()
        if not self.s.hold(M + "ender_eye"):
            return "没拿到末影之眼。"
        self.s._smooth_look(me["x"], me["y"] + 12, me["z"] + 8)
        self.s.rpc("interact.useItem", {"hand": "main"})
        deadline = time.monotonic() + 3.5
        points = []
        while time.monotonic() < deadline and not self.s._stop.is_set():
            eyes = [e for e in self.entities(32) if e.get("type") == M + "eye_of_ender"
                    and e.get("uuid") not in before and all(k in e for k in ("x", "y", "z"))]
            if eyes:
                eye = min(eyes, key=lambda e: float(e.get("distance", 99)))
                points.append(dict(eye))
            time.sleep(.15)
        if not points:
            return "没有观测到这次末影之眼的飞行轨迹，不猜要塞方向。"
        end = max(points, key=lambda e: math.hypot(e["x"] - me["x"], e["z"] - me["z"]))
        dx, dz = end["x"] - me["x"], end["z"] - me["z"]
        length = math.hypot(dx, dz)
        if end["y"] < me["y"] - 1 or length < 4:
            self.s.places.add("stronghold_hint", end["x"], end["y"], end["z"],
                              note="末影之眼下沉/近距离轨迹，地下位置待勘察", dim=M + "overworld", merge=8)
            return self.s.dig_down(max(-40, math.floor(me["y"]) - 8), max_steps=8)
        # A short terrain-planned leg; the next invocation throws again and corrects the bearing.
        self.s.places.add("eye_bearing", end["x"], end["y"], end["z"],
                          note="真实末影之眼飞行方向，非要塞精确坐标", dim=M + "overworld", merge=8)
        self.s.progress = "沿实际末影之眼方向走96格，再观测校正"
        self.s._eye_route = {"dx": dx / length, "dz": dz / length, "remaining": 512}
        return self.s.go_to(round(me["x"] + dx / length * 96), round(me["y"]), round(me["z"] + dz / length * 96))

    def fill_frames(self, frames):
        if self.s.crafter.counts().get(M + "ender_eye", 0) < 1:
            return "没有可用末影之眼；已有眼的门框会保留，补足后继续。"
        filled = 0
        for frame in frames[:12]:
            if self.s._stop.is_set():
                return "填门暂停，已填的眼保留。"
            x, y, z = (int(frame[k]) for k in ("x", "y", "z"))
            if not self.s._approach(x, y, z):
                return "有门框尚未能安全靠近，先解决台阶和岩浆附近的路线。"
            if not self.s.hold(M + "ender_eye"):
                break
            before = self.s.crafter.counts().get(M + "ender_eye", 0)
            self.s._smooth_look(x + .5, y + 1.0, z + .5)
            # useItemOn only; unlike a generic air click this cannot throw an eye off a filled frame.
            self.s.rpc("interact.placeBlock", {"x": x, "y": y, "z": z, "face": "up"})
            time.sleep(.2)
            after = self.s.crafter.counts().get(M + "ender_eye", 0)
            filled += max(0, before - after)
            active = self.scan([M + "end_portal"], radius=1)
            if active:
                p = active[0]
                self.s.places.add("end_portal", p["x"], p["y"], p["z"],
                                  note="填眼后实际观测到激活末地门", dim=M + "overworld")
                return "末地门已实际点亮；下一步进入并核实维度，不把填眼回执当成功。"
        return f"本次实际消耗 {filled} 颗眼，但未观测到激活的末地门；需要继续检查其余门框。"

    def enter(self, portal, expected):
        me = self.me()
        # Gear/ammunition is checked immediately before the irreversible trip, not only in a plan.
        have = self.s.crafter.counts()
        if expected == M + "the_end" and (not have.get(M + "bow") or have.get(M + "arrow", 0) < 64
                                         or not have.get(M + "water_bucket")):
            return "进入末地前需弓、至少64支箭和水桶，先补齐。"
        self.s.free_hands()
        if me["y"] > portal["y"] + 4:
            return self.s.dig_down(portal["y"] + 1, max_steps=12)
        # Path onto a frame/edge first. The final half-block step into the portal is intentional.
        if not self.s._walk_to(portal["x"] + .5, portal["y"] + 1, portal["z"] + .5, reach=.45, timeout=25):
            state = self.me()
            if state.get("dimension") != expected:
                return "还没安全走入末地传送门。"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not self.s._stop.is_set():
            state = self.me()
            if state.get("dimension") == expected:
                self.s.places.add("arrival", state["x"], state["y"], state["z"],
                                  note="末地传送后实际落脚点", dim=expected)
                return f"已核实实际维度为 {expected}，先检查脚下和周围。"
            time.sleep(.2)
        return "尚未观测到目标维度，进入动作没有确认成功。"

    def fill_water_bucket(self):
        if self.s.crafter.counts().get(M + "water_bucket"):
            return "已经有装满水的水桶。"
        if not self.s.crafter.counts().get(M + "bucket"):
            return self.s.craft(M + "bucket", 1)
        me = self.me()
        if me.get("dimension") != M + "overworld":
            return "回主世界取水。"
        water = self.scan([M + "water"], radius=1)
        for p in water[:8]:
            if self.s._stop.is_set():
                break
            if not self.s._approach(p["x"], p["y"], p["z"]):
                continue
            self.s.hold(M + "bucket")
            self.s._smooth_look(p["x"] + .5, p["y"] + .5, p["z"] + .5)
            self.s.rpc("interact.useItem", {"hand": "main"})
            time.sleep(.25)
            if self.s.crafter.counts().get(M + "water_bucket"):
                return "已实际取到一桶水。"
        return "附近没有安全可达的水源方块，未把空桶当成水桶。"

    def _retreat(self, hazard, sideways=False):
        me = self.me()
        dx, dz = me["x"] - hazard["x"], me["z"] - hazard["z"]
        angle = math.atan2(dz, dx)
        for offset in ((math.pi / 2, -math.pi / 2, 0) if sideways else (0, math.pi / 2, -math.pi / 2)):
            x, z = me["x"] + 5 * math.cos(angle + offset), me["z"] + 5 * math.sin(angle + offset)
            cells = [(math.floor(me["x"] + (x - me["x"]) * f / 5), math.floor(me["y"]) - 1,
                      math.floor(me["z"] + (z - me["z"]) * f / 5)) for f in range(1, 6)]
            ground = self.s._blocks(cells)
            if ground and all(b in (M + "end_stone", M + "obsidian", M + "bedrock", M + "cobblestone") for b in ground):
                return self.s._walk_to(x, me["y"], z, reach=1, timeout=4)
        return False

    def arena_route(self):
        """Leave the arrival platform along observed End stone, bridging a bounded number of cells."""
        me = self.me()
        if me.get("dimension") != M + "the_end":
            return "不在末地，未执行岛屿路线。"
        fx, fy, fz = (math.floor(me[k]) for k in ("x", "y", "z"))
        if not self.s._sky_above(fx, fy, fz):
            return self.s.dig_up(fy + 12, max_steps=8, until_sky=True)
        stone = self.scan([M + "end_stone"], radius=4)
        candidates = [p for p in stone if p.get("exposed") and fy - 2 <= p["y"] <= fy + 14
                      and math.hypot(p["x"] - me["x"], p["z"] - me["z"]) >= 4]
        if not candidates:
            return "还没观察到可达的末地石岛边，不向未知虚空搭桥。"
        target = min(candidates, key=lambda p: math.dist((p["x"], p["y"], p["z"]), (me["x"], me["y"], me["z"])))
        placed = 0
        for _ in range(6):
            if self.s._stop.is_set():
                break
            me = self.me()
            if not me or not me.get("onGround") or float(me.get("health", 0)) < 14:
                return "搭桥前身体未站稳或血量不足，暂停。"
            fx, fy, fz = (math.floor(me[k]) for k in ("x", "y", "z"))
            dx, dz = target["x"] - fx, target["z"] - fz
            if math.hypot(dx, dz) < 2:
                return f"已沿验证过的地面接近末地石岛边，本次放了 {placed} 块。"
            sx, sz = ((1 if dx > 0 else -1), 0) if abs(dx) >= abs(dz) else (0, (1 if dz > 0 else -1))
            nx, nz = fx + sx, fz + sz
            ground = self.s._block(nx, fy - 1, nz)
            if not ground:
                return "前方区块尚未加载，不继续搭桥。"
            if ground.endswith(":air"):
                if not self.s.crafter.counts().get(M + "cobblestone") or not self.s.place_at(M + "cobblestone", nx, fy - 1, nz, walk=False):
                    return "前方桥面尚未实际放好，停在已有支撑上。"
                placed += 1
            elif ground not in (M + "end_stone", M + "obsidian", M + "bedrock", M + "cobblestone"):
                return "前方不是验证过的安全桥面，暂停。"
            feet_y = fy
            if target["y"] + 1 > fy:
                if not self.s._block(nx, fy, nz).endswith(":air"):
                    return "向上的桥面被占用，先重新找路。"
                if not self.s.place_at(M + "cobblestone", nx, fy, nz, walk=False):
                    return "台阶还没放好，不向虚空跳。"
                placed += 1
                feet_y += 1
            headroom = self.s._blocks([(nx, feet_y, nz), (nx, feet_y + 1, nz)])
            if not all(b.endswith(":air") for b in headroom):
                return "前方站位或头顶不通，保留已有桥面。"
            if not self.s._walk_to(nx + .5, feet_y, nz + .5, reach=.3, timeout=5):
                return "桥面存在但尚未走稳，暂停继续延伸。"
        return f"末地岛边路线已推进，本次确认放置 {placed} 块；下一步重新观察龙和水晶。"

    def fight(self):
        me = self.me()
        if me.get("dimension") != M + "the_end":
            return "当前不在末地，不执行龙战动作。"
        enemies = self.entities()
        dragon = next((e for e in enemies if e.get("type") == M + "ender_dragon"), None)
        clouds = [e for e in enemies if e.get("type") in (M + "area_effect_cloud", M + "dragon_fireball")
                  and float(e.get("distance", 99)) < 9]
        if clouds:
            return "发现龙息/火球，已尝试沿验证有地面的路线避开。" if self._retreat(clouds[0]) else "龙息附近没有验证过的安全退路，暂停射击等待重新观察。"
        if float(me.get("health", 0)) < 14:
            if dragon:
                self._retreat(dragon)
            return "龙战血量不足，先避开并让生存反射吃东西恢复。"
        exits = self.scan([M + "end_portal"], radius=2)
        if exits and dragon is None:
            p = exits[0]
            self.s.places.add("end_exit", p["x"], p["y"], p["z"],
                              note="观察到激活的末地返程门，FTB领奖及模组Boss未核实", dim=M + "the_end")
            return "观察到激活的末地返程门，附近未见存活末影龙；这是末地现场证据，不代表整合包任务全部完成。"
        have = self.s.crafter.counts()
        if not have.get(M + "bow") or not have.get(M + "arrow"):
            return "没有可用弓箭，不能继续远程龙战。"
        crystals = [e for e in enemies if e.get("type") == M + "end_crystal"]
        targets = sorted(crystals, key=lambda e: float(e.get("distance", 99))) or ([dragon] if dragon else [])
        for target in targets[:4]:
            if self.s._stop.is_set():
                return "龙战已暂停。"
            distance = math.dist((me["x"], me["y"], me["z"]), (target["x"], target["y"], target["z"]))
            if distance < 8 and target.get("type") == M + "end_crystal":
                self._retreat(target)
                return "离末地水晶爆炸范围太近，先退到安全距离。"
            if distance > 60:
                continue
            previous = getattr(self.s, "_dragon_shot", None)
            if target.get("type") == M + "ender_dragon" and previous and previous["misses"] >= 3:
                moved = math.dist(previous["pos"], (target["x"], target["y"], target["z"]))
                if moved < 2:
                    return "龙停在原处且连续箭击未掉血，等待离开停栖/无敌阶段，避免浪费箭。"
            # Fully drawn arrow: speed about 3 blocks/tick, gravity .05 blocks/tick².
            # The low ballistic arc raises aim by d²/360; live target motion is rechecked each shot.
            aim_y = target["y"] + (1 if target.get("type") == M + "end_crystal" else 2) + distance * distance / 360
            self.s._smooth_look(target["x"], aim_y, target["z"])
            scene = self.s.rpc("vision.describeScene", {"maxDistance": 64, "rayColumns": 1, "rayRows": 1}) or {}
            if not any(e.get("uuid") == target.get("uuid") for e in scene.get("entities") or []):
                continue                    # cage/wall: change observed firing position, do not fire blind
            self.s.hold(M + "bow")
            before = float(target.get("health", 0))
            if self.s.body is not None:
                self.s.body.use_item(seconds=1.1)
            else:
                self.s.rpc("control.startUsing", {})
                try:
                    time.sleep(1.1)
                finally:
                    self.s.rpc("control.stopUsing", {})
            time.sleep(.6)
            after = next((e for e in self.entities() if e.get("uuid") == target.get("uuid")), None)
            if after and "health" in after:
                old = getattr(self.s, "_dragon_shot", {})
                self.s._dragon_shot = {"pos": (after["x"], after["y"], after["z"]),
                                      "misses": 0 if float(after["health"]) < before else old.get("misses", 0) + 1}
                return f"向末影龙射出一箭，观测血量 {before:g} → {float(after['health']):g}；下一步继续检查水晶与龙息。"
            return "已向观测到的末地水晶/末影龙射箭；目标暂不在视野，未据此判定击杀。"
        if targets:
            if self._retreat(targets[0], sideways=True):
                return "当前水晶/龙被遮挡或距离不合适，已沿安全地面调整射击位置后再观测。"
        return self.arena_route()
