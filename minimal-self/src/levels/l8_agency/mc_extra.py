"""
Level 8h: the parts of her game body the MCP server does not expose.

The mcpfabric mod inside her client can open chests, crafting tables and furnaces, move items, look up
recipes and craft, but its MCP server (0.8.0) does not offer those as tools. These tools call the mod's bridge
directly (the same loopback address and token), so she can actually use a chest or a crafting table.
They belong to the "minecraft" group: the same tiers, firewall and lockdown as the rest of her game body.
"""
import json
import urllib.request
from typing import Any, Dict

from src.levels.l8_agency import access

SERVER = "minecraft"

def call_bridge(gw, method: str, params: Dict[str, Any] = None, timeout: float = 10.0, limit: int = 5000) -> str:
    """The bridge's answer as JSON text, cut to `limit` characters for a model (0: whole, for code that parses it)."""
    url, token = getattr(gw, "mc_bridge", (None, None))
    if not url:
        return "游戏身体没有连上。"
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/rpc", data=body,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
    except Exception as e:
        return f"出错了：连不上游戏（{type(e).__name__}）"
    if not out.get("ok", True) or "error" in out:
        err = out.get("error") or {}
        return "出错了：" + str(err.get("message") if isinstance(err, dict) else err)
    text = json.dumps(out.get("result", out), ensure_ascii=False)
    return text[:limit] if limit else text

def _available(gw, ctx) -> bool:
    hub = getattr(gw, "mcp", None)
    if hub is None or SERVER not in hub.config or not getattr(gw, "mc_bridge", None):
        return False
    return access.allowed(hub.config[SERVER], access.tier_of(gw, ctx), "get_self")

def tools():
    from src.levels.l8_agency.toolbox import Tool, _i, _props, _s

    def open_block(gw, ctx, a):
        return call_bridge(gw, "container.open", {"x": a.get("x"), "y": a.get("y"), "z": a.get("z")})

    def state(gw, ctx, a):
        return call_bridge(gw, "container.state")

    def click(gw, ctx, a):
        return call_bridge(gw, "container.click", {"slot": int(a.get("slot", 0)), "button": int(a.get("button", 0) or 0),
                                                   "mode": str(a.get("mode") or "pickup")})

    def transfer(gw, ctx, a):
        p = {"direction": "take" if str(a.get("direction")) == "take" else "put"}
        if a.get("item"):
            p["item"] = str(a["item"])
        if a.get("count"):
            p["count"] = int(a["count"])
        return call_bridge(gw, "container.transfer", p)

    def recipes(gw, ctx, a):
        items = [s.strip() for s in str(a.get("items") or "").split(",") if s.strip()]
        return call_bridge(gw, "recipes.query", {"items": items}) if items else "要查哪些物品的配方？"

    def craft(gw, ctx, a):
        return call_bridge(gw, "craft.place", {"ref": str(a.get("ref") or ""), "all": bool(a.get("all"))})

    def follow(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        who = str(a.get("player") or "").strip()
        if who.lower() in ("", "stop", "停", "不跟了"):
            return game.set_follow(None)
        return game.set_follow(who, float(a.get("minutes") or 20))

    def near(gw, ctx, a):
        game = getattr(gw, "game", None)
        return game.who_is_near(float(a.get("radius") or 48)) if game is not None else "游戏身体没有连上。"

    def skill(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        name = str(a.get("name") or "").strip()
        if name == "stop":
            return game.skills.stop()
        if name == "status":
            return game.skills.status()
        args = {"player": str(a["player"])} if name == "go_to_player" and a.get("player") else {}
        return game.skills.start(name, **args)

    def find(gw, ctx, a):
        ids = [s.strip() for s in str(a.get("blocks") or "").split(",") if s.strip()]
        if not ids:
            return "要找什么方块？比如 minecraft:oak_log"
        # (the scan radius is in chunks of 16 blocks; the tool speaks blocks)
        chunks = max(1, min(12, round(int(a.get("radius") or 32) / 16)))
        got = call_bridge(gw, "perception.scan", {"radius": chunks, "find": ids, "findLimit": int(a.get("limit") or 20)}, limit=0)
        try:
            found = json.loads(got).get("found") or []
        except ValueError:
            return got[:2000]
        return json.dumps(found, ensure_ascii=False)[:3000] if found else "附近没找到。"

    def common(**kw):
        return dict(available=_available, timeout=15, group=SERVER, **kw)

    return [
        Tool("mc_open_block", "在游戏里打开身边的箱子、工作台、熔炉之类（要走到够得着的地方）。",
             _props(["x", "y", "z"], x=_i("方块 x"), y=_i("方块 y"), z=_i("方块 z")), open_block, **common()),
        Tool("mc_container_state", "看现在打开的箱子/工作台/背包界面里每一格是什么。", _props(), state, **common(read_only=True)),
        Tool("mc_container_click", "点界面里的一格（拿起、放下、shift 快速移动、扔出）。",
             _props(["slot"], slot=_i("格子编号"), button=_i("0 左键 1 右键"),
                    mode=_s("pickup / quick_move / throw / swap")), click, **common()),
        Tool("mc_container_transfer", "把东西在打开的箱子和背包之间搬：take 从箱子拿，put 放进箱子。",
             _props(["direction"], direction=_s("take 或 put"), item=_s("物品 id，比如 minecraft:oak_log；不写就全部"),
                    count=_i("最多几个")), transfer, **common()),
        Tool("mc_recipes", "查几样物品的合成配方（返回能用来合成的 ref）。",
             _props(["items"], items=_s("物品 id，逗号分隔，比如 minecraft:crafting_table")), recipes, **common(read_only=True)),
        Tool("mc_skill", "成套的动作，在后台自己做完：chop_tree 砍附近一整棵树并捡木头，collect_drops 捡地上的东西，"
             "go_to_player 走到某人身边（写 player），stop 停下，status 看进度。",
             _props(["name"], name=_s("chop_tree / collect_drops / go_to_player / stop / status"), player=_s("go_to_player 时写玩家名")),
             skill, **common()),
        Tool("mc_find_blocks", "在自己周围找某些方块的位置（比如原木、矿石、箱子），按距离给出坐标。",
             _props(["blocks"], blocks=_s("方块 id，逗号分隔，比如 minecraft:oak_log"), radius=_i("多远以内（格），默认 32"),
                    limit=_i("最多几个，默认 20")), find, **common(read_only=True)),
        Tool("mc_who_is_near", "看看身边有哪些玩家（叫得出名字的会写名字、多远）和哪些生物。",
             _props(radius=_i("多远以内，默认 48 格")), near, **common(read_only=True)),
        Tool("mc_follow", "跟着某个玩家走（身体会自己跟上、看着 ta，直到你说停或者时间到）；player 写 stop 就不跟了。",
             _props(["player"], player=_s("玩家名，或 stop"), minutes=_i("最多跟多久（分钟），默认 20")), follow, **common()),
        Tool("mc_craft", "在打开的工作台（或背包合成格）里按配方摆好材料，然后从结果格拿走。",
             _props(["ref"], ref=_s("mc_recipes 给的 ref"), all=_s("true 就能做多少做多少")), craft, **common()),
    ]
