"""Fighting reflexes at 10 Hz: see the mob, turn to it, close in, wait for the swing to charge, hit, back off.

A player's fight is decided in tenths of a second. The game loop ticks once a second and the planner even less
often, so a zombie used to get two or three hits in before she noticed it. `Fighter` runs on its own thread,
reads the client directly, and holds the keys and the camera only while a fight is on; when it is over it lets
go and the interrupted work resumes.

It only uses what a player has: what is in line of sight (or already hitting her), the sword's real charge
(`attackStrength` from the client), the shield in the off hand, and the movement keys.
"""
import logging
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# mobs that do not start a fight: leave them alone unless they are already on her
NEUTRAL = ("enderman", "zombified_piglin", "piglin", "spider", "cave_spider", "wolf", "bee", "iron_golem",
           "llama", "polar_bear", "dolphin", "panda", "goat")
RANGED = ("skeleton", "stray", "bogged", "pillager", "piglin_brute", "witch", "blaze", "ghast")
BOSSES = ("ender_dragon", "wither", "elder_guardian", "warden")


def _short(kind: Any) -> str:
    return str(kind or "").split(":")[-1]


class Fighter:
    HZ = 10.0
    IDLE_HZ = 4.0          # looking around for trouble when there is none
    ENGAGE = 10.0          # a visible hostile this close is a fight
    BLIND = 3.5            # one this close is one even through a corner (it is hitting her)
    REACH = 3.0            # melee range a player keeps
    BACKOFF = 1.8          # closer than this, step back while swinging
    FLEE_HP = 7.0          # below this she runs instead
    CREEPER_SAFE = 6.0     # keep at least this far from a creeper that is not being hit
    CALM = 2.0             # no target for this long: the fight is over

    def __init__(self, rpc: Callable[[str, Dict[str, Any]], Any], *, pause: Callable[[float], None] = None,
                 on_event: Callable[[str], None] = None, can_fight: Callable[[], bool] = None,
                 now: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.rpc = rpc
        self.pause = pause or (lambda seconds: None)          # hold the skills and the body loop
        self.on_event = on_event or (lambda text: None)
        self.can_fight = can_fight or (lambda: True)
        self.now = now
        self.sleep = sleep
        self.engaged: Optional[Dict[str, Any]] = None
        self.hostiles: List[Dict[str, Any]] = []
        self.last_seen = 0.0
        self.last_swing = 0.0
        self.swings = 0
        self.kills = 0
        self._sent: Dict[str, bool] = {}
        self._shield = False
        self._strafe = (0.0, False)
        self._armed = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- lifecycle -------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fighting(self) -> bool:
        return self.engaged is not None

    def start(self) -> "Fighter":
        if not self.running:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="mc-fight", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._disengage("停止")

    def _run(self) -> None:
        while not self._stop.is_set():
            started = self.now()
            try:
                self.step()
            except Exception:
                logger.warning("[Fight] step failed", exc_info=True)
            period = 1.0 / (self.HZ if self.engaged else self.IDLE_HZ)
            self.sleep(max(0.0, period - (self.now() - started)))

    # --- one tenth of a second -------------------------------------------------------------

    def threats(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for e in entities:
            if e.get("kind") != "hostile":
                continue
            kind, dist = _short(e.get("type")), float(e.get("distance", 99))
            if kind in BOSSES:
                continue                              # boss fights belong to their own controller
            close = dist <= self.BLIND
            if kind in NEUTRAL and not close:
                continue
            # an older mod does not say what is in sight: then distance alone decides
            if close or (dist <= self.ENGAGE and e.get("visible", True)):
                out.append(e)
        return sorted(out, key=lambda e: float(e.get("distance", 99)))

    def step(self) -> str:
        seen = self.rpc("perception.entities", {"radius": 16, "kinds": ["hostile"]}) or {}
        entities = seen.get("entities") if isinstance(seen, dict) else None
        if not isinstance(entities, list):
            return "blind"
        self.hostiles = sorted((e for e in entities if e.get("kind") == "hostile"),
                               key=lambda e: float(e.get("distance", 99)))
        threats = self.threats(entities)
        if not self.can_fight():
            # mid-meal she does not go looking for a fight, but one that is already on her is fought: holding
            # still to eat while a zombie hits her is how she died at 14:18
            threats = [e for e in threats if float(e.get("distance", 99)) <= self.BLIND + 0.5]
        now = self.now()
        if not threats:
            if self.engaged and now - self.last_seen >= self.CALM:
                gone = self.engaged
                self._disengage("没有威胁了")
                if not any(e.get("uuid") == gone.get("uuid") for e in entities):
                    self.kills += 1
                return "calm"
            if self.engaged:
                self._keys()                          # it stepped out of sight: stop, look around
            return "watch" if self.engaged else "idle"
        target = threats[0]
        if self.engaged is None:
            self.rpc("container.close", {})           # a table or chest left open would eat the keys
            self.rpc("nav.stop", {})
            self.rpc("baritone.stop", {})             # its keys would fight hers
            self.on_event(f"{_short(target.get('type'))} 靠近了（{float(target.get('distance', 0)):.0f} 格）：开打")
            self._armed = False
        self.engaged, self.last_seen = target, now
        self.pause(1.0)                               # the skills and the body loop wait while she fights
        return self._fight(target, now)

    def _fight(self, target: Dict[str, Any], now: float) -> str:
        me = self.rpc("player.getState", {}) or {}
        if not isinstance(me.get("x"), (int, float)):
            return "blind"
        kind, dist = _short(target.get("type")), float(target.get("distance", 99))
        health = float(me.get("health", 20))
        tx, ty, tz = float(target["x"]), float(target["y"]), float(target["z"])
        self.rpc("control.lookTarget", {"x": tx, "y": ty + 1.2, "z": tz, "speed": 520})
        if not self._armed:
            self._arm()

        if health <= self.FLEE_HP:
            # turn away and sprint: backing up is slower than the mob
            away = (me["x"] - tx, me["z"] - tz)
            n = math.hypot(*away) or 1.0
            self.rpc("control.lookTarget", {"x": me["x"] + away[0] / n * 10, "y": me["y"] + 1.6,
                                            "z": me["z"] + away[1] / n * 10, "speed": 600})
            self._block(False)
            self._keys(forward=True, sprint=True, jump=bool(me.get("onGround")) and dist < 3)
            return "flee"

        if kind == "creeper" and dist < self.CREEPER_SAFE and not self._charged(me):
            self._keys(back=True)                     # out of its blast while the swing charges
            return "creeper_back"

        if dist > self.REACH:
            ranged = kind in RANGED
            self._block(ranged and self._has_shield(me))
            look_left = float(me.get("lookRemaining", 0))
            facing = look_left <= 35
            self._keys(forward=facing, sprint=facing and dist > 5 and not self._shield,
                       jump=facing and bool(me.get("onGround")) and float(target.get("y", 0)) > me["y"] + 0.6)
            return "close_in"

        if self._charged(me) and float(me.get("lookRemaining", 0)) <= 25:
            self._block(False)
            self.rpc("interact.attackEntity", {"uuid": target.get("uuid")})
            self.last_swing, self.swings = now, self.swings + 1
            if kind == "creeper":
                self._keys(back=True)                 # knockback, then step out before the fuse restarts
                return "hit_back"
        # between swings: keep the distance and move sideways a little, as a player does
        if now - self._strafe[0] > 0.8:
            self._strafe = (now, not self._strafe[1])
        left = self._strafe[1]
        self._keys(back=dist < self.BACKOFF, left=left, right=not left)
        return "fight"

    # --- hands -------------------------------------------------------------------------------

    @staticmethod
    def _charged(me: Dict[str, Any]) -> bool:
        strength = me.get("attackStrength")
        return strength is None or float(strength) >= 0.92

    def _has_shield(self, me: Dict[str, Any]) -> bool:
        gear = self.rpc("player.getEquipment", {}) or {}
        return "shield" in str((gear.get("offHand") or {}).get("id", ""))

    def _arm(self) -> None:
        """A sword (else an axe) from the hotbar - a key press, not a trip into the bag mid-fight."""
        self._armed = True
        inv = self.rpc("player.getInventory", {}) or {}
        hotbar = inv.get("hotbar") or []
        for suffix in ("_sword", "_axe"):
            weapon = next((it for it in hotbar if str(it.get("id", "")).endswith(suffix)), None)
            if weapon is not None:
                if int(weapon.get("slot", -1)) != int(inv.get("selectedSlot", -2)):
                    self.rpc("inventory.selectHotbar", {"slot": int(weapon["slot"])})
                return

    def _block(self, up: bool) -> None:
        if up and not self._shield:
            self.rpc("control.startUsing", {"hand": "off"})
        elif not up and self._shield:
            self.rpc("control.stopUsing", {})
        self._shield = up

    def _keys(self, **held: bool) -> None:
        want = {k: bool(held.get(k)) for k in ("forward", "back", "left", "right", "jump", "sneak", "sprint")}
        if want != self._sent:
            self.rpc("control.setInput", want)
            self._sent = want

    def _disengage(self, why: str) -> None:
        if self.engaged is None and not self._sent and not self._shield:
            return
        self._block(False)
        if any(self._sent.values()):
            self.rpc("control.setInput", {k: False for k in self._sent})
        self._sent = {}
        self.rpc("control.lookTarget", {"clear": True})
        was = self.engaged
        self.engaged = None
        if was is not None:
            self.on_event(f"战斗结束（{why}）")

    def status(self) -> Dict[str, Any]:
        return {"running": self.running, "fighting": self.fighting,
                "target": _short((self.engaged or {}).get("type")) or None,
                "swings": self.swings, "kills": self.kills}
