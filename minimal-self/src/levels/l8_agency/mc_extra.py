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

def call_bridge(gw, method: str, params: Dict[str, Any] = None, timeout: float = 10.0) -> str:
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
    return json.dumps(out.get("result", out), ensure_ascii=False)[:5000]

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
        Tool("mc_craft", "在打开的工作台（或背包合成格）里按配方摆好材料，然后从结果格拿走。",
             _props(["ref"], ref=_s("mc_recipes 给的 ref"), all=_s("true 就能做多少做多少")), craft, **common()),
    ]
