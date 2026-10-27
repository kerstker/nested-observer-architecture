"""The 10 Hz fighting reflex: what starts a fight, how she closes in, swings on a charged sword, backs off,
runs when low, and lets go of the keys and the camera when it is over."""
from src.levels.l8_agency.mc_fight import Fighter


class Arena:
    def __init__(self, mobs=(), me=None):
        self.mobs = list(mobs)
        self.me = {"x": 0.0, "y": 64.0, "z": 0.0, "health": 20.0, "onGround": True,
                   "attackStrength": 1.0, "lookRemaining": 0.0, **(me or {})}
        self.calls = []
        self.t = 0.0
        self.hotbar = [{"slot": 0, "id": "minecraft:stone_pickaxe"}, {"slot": 2, "id": "minecraft:iron_sword"}]
        self.offhand = {}

    def rpc(self, method, params):
        self.calls.append((method, dict(params or {})))
        if method == "perception.entities":
            return {"entities": [dict(m) for m in self.mobs]}
        if method == "player.getState":
            return dict(self.me)
        if method == "player.getInventory":
            return {"hotbar": self.hotbar, "main": [], "selectedSlot": 0}
        if method == "player.getEquipment":
            return {"offHand": self.offhand}
        return {"ok": True}

    def methods(self, name):
        return [p for m, p in self.calls if m == name]

    def fighter(self):
        paused = []
        f = Fighter(self.rpc, pause=paused.append, now=lambda: self.t, sleep=lambda s: None)
        f.paused = paused
        return f


def zombie(distance, visible=True, uuid="z1", x=None):
    return {"uuid": uuid, "kind": "hostile", "type": "minecraft:zombie", "distance": distance,
            "x": distance if x is None else x, "y": 64.0, "z": 0.0, "visible": visible}


def test_a_mob_behind_rock_is_not_a_fight_but_one_hitting_her_is():
    arena = Arena([zombie(8, visible=False)])
    f = arena.fighter()
    assert f.step() == "idle" and not f.fighting
    arena.mobs = [zombie(2.5, visible=False)]
    assert f.step() != "idle" and f.fighting
    assert f.paused                                   # the skills wait while she fights


def test_she_turns_to_it_takes_the_sword_and_closes_in():
    arena = Arena([zombie(7)])
    f = arena.fighter()
    assert f.step() == "close_in"
    assert arena.methods("inventory.selectHotbar") == [{"slot": 2}]
    look = arena.methods("control.lookTarget")[-1]
    assert (look["x"], look["z"]) == (7.0, 0.0)
    assert arena.methods("control.setInput")[-1]["forward"] is True


def test_she_swings_only_when_the_sword_is_charged_and_keeps_her_distance():
    arena = Arena([zombie(2.5)], me={"attackStrength": 0.4})
    f = arena.fighter()
    assert f.step() == "fight"
    assert arena.methods("interact.attackEntity") == []      # spam-clicking is not how a player fights
    arena.me["attackStrength"] = 1.0
    arena.t += 0.1
    f.step()
    assert arena.methods("interact.attackEntity") == [{"uuid": "z1"}]
    arena.mobs = [zombie(1.2)]
    arena.me["attackStrength"] = 0.2
    arena.t += 0.1
    f.step()
    assert arena.methods("control.setInput")[-1]["back"] is True


def test_low_health_turns_away_and_sprints():
    arena = Arena([zombie(3)], me={"health": 5.0})
    f = arena.fighter()
    assert f.step() == "flee"
    keys = arena.methods("control.setInput")[-1]
    assert keys["forward"] and keys["sprint"]
    assert arena.methods("control.lookTarget")[-1]["x"] < 0      # facing away from the zombie at +x


def test_a_creeper_is_not_hugged():
    creeper = {"uuid": "c1", "kind": "hostile", "type": "minecraft:creeper", "distance": 3.0,
               "x": 3.0, "y": 64.0, "z": 0.0, "visible": True}
    arena = Arena([creeper], me={"attackStrength": 0.3})
    f = arena.fighter()
    assert f.step() == "creeper_back"
    assert arena.methods("control.setInput")[-1]["back"] is True


def test_when_it_is_over_she_lets_go_of_keys_and_camera():
    arena = Arena([zombie(2.5)])
    f = arena.fighter()
    f.step()
    arena.mobs = []
    arena.t += 2.5
    assert f.step() == "calm"
    assert not f.fighting and f.kills == 1
    assert arena.methods("control.lookTarget")[-1] == {"clear": True}
    assert not any(arena.methods("control.setInput")[-1].values())


def test_while_eating_it_leaves_a_distant_mob_alone_but_fights_one_that_is_on_her():
    arena = Arena([zombie(7)])
    f = Fighter(arena.rpc, can_fight=lambda: False, now=lambda: arena.t, sleep=lambda s: None)
    assert f.step() == "idle" and arena.methods("interact.attackEntity") == []
    arena.mobs = [zombie(2.5)]
    assert f.step() == "fight" and arena.methods("interact.attackEntity") == [{"uuid": "z1"}]


def test_a_fight_takes_the_legs_from_baritone_first():
    arena = Arena([zombie(7)])
    f = arena.fighter()
    f.step()
    methods = [m for m, _ in arena.calls]
    assert "baritone.stop" in methods
    assert methods.index("baritone.stop") < methods.index("control.setInput")
