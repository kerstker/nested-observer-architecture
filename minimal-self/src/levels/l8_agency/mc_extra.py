"""
Level 8h: the parts of her game body the MCP server does not expose.

The mcpfabric mod inside her client can open chests, crafting tables and furnaces, move items, look up
recipes and craft, but its MCP server (0.8.0) does not offer those as tools. These tools call the mod's bridge
directly (the same loopback address and token), so she can actually use a chest or a crafting table.
They belong to the "minecraft" group: the same tiers, firewall and lockdown as the rest of her game body.
"""
import http.client
import json
import threading
from typing import Any, Dict
from urllib.parse import urlparse

from src.levels.l8_agency import access

SERVER = "minecraft"

def call_bridge(gw, method: str, params: Dict[str, Any] = None, timeout: float = 10.0, limit: int = 5000) -> str:
    """The bridge's answer as JSON text, cut to `limit` characters for a model (0: whole, for code that parses it)."""
    url, token = getattr(gw, "mc_bridge", (None, None))
    if not url:
        return "游戏身体没有连上。"
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()
    try:
        out = json.loads(_post(url, token, body, timeout))
    except Exception as e:
        return f"出错了：连不上游戏（{type(e).__name__}）"
    if not out.get("ok", True) or "error" in out:
        err = out.get("error") or {}
        return "出错了：" + str(err.get("message") if isinstance(err, dict) else err)
    text = json.dumps(out.get("result", out), ensure_ascii=False)
    return text[:limit] if limit else text

_local = threading.local()

def _post(url: str, token: str, body: bytes, timeout: float) -> str:
    """POST to the bridge over a connection this thread keeps open (reopened once if the game closed it)."""
    u = urlparse(url)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    for attempt in (0, 1):
        conn = getattr(_local, "conn", None)
        if conn is None or getattr(_local, "url", None) != url:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
            _local.conn, _local.url = conn, url
        conn.timeout = timeout
        try:
            conn.request("POST", (u.path.rstrip("/") or "") + "/rpc", body=body, headers=headers)
            return conn.getresponse().read().decode("utf-8")
        except (http.client.HTTPException, OSError):
            conn.close()
            _local.conn = None
            if attempt:
                raise
    raise OSError("unreachable")

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

    def rejoin(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        return game.rejoin_now()

    def look(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        return game.look_now()

    def look_at(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        return game.look_smooth(float(a.get("x")), float(a.get("y")), float(a.get("z")))

    def mine(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        return game.mine_block(a.get("x"), a.get("y"), a.get("z"))

    def place(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        return game.place_block(a.get("x"), a.get("y"), a.get("z"), str(a.get("item") or ""))

    def skill(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        name = str(a.get("name") or "").strip()
        own_turn = getattr(ctx, "origin", "game") == "game" and not getattr(ctx, "group_id", None)
        if name == "status":
            return game.skills.status()
        if name == "stop":
            if own_turn and game.skills.busy() and not game.work_stalled():
                return f"{game.skills.current} 还在推进（{game.skills.progress}），先不停。"
            return game.skills.stop()
        args = {"player": str(a["player"])} if name == "go_to_player" and a.get("player") else {}
        if name == "mine":
            args = {"blocks": str(a.get("blocks") or ""), "count": int(a.get("count") or 8)}
        elif name == "craft":
            args = {"item": str(a.get("item") or ""), "count": int(a.get("count") or 1)}
        elif name == "place":
            args = {"item": str(a.get("item") or "")}
            if all(a.get(k) is not None for k in ("x", "y", "z")):
                args.update(x=int(a["x"]), y=int(a["y"]), z=int(a["z"]))
        elif name == "equip":
            args = {"kind": str(a.get("kind") or "pickaxe")}
        elif name == "smelt":
            args = {"item": str(a.get("item") or ""), "count": int(a.get("count") or 8)}
        elif name == "dig_down":
            to_y = a.get("to_y") if a.get("to_y") is not None else a.get("y")
            args = {"to_y": int(to_y if to_y is not None else 16)}
            if a.get("max_steps") is not None:
                args["max_steps"] = max(1, min(160, int(a["max_steps"])))
        elif name == "dig_up":
            to_y = a.get("to_y") if a.get("to_y") is not None else a.get("y")
            args = {"to_y": int(to_y if to_y is not None else 64)}
            if a.get("max_steps") is not None:
                args["max_steps"] = max(1, min(160, int(a["max_steps"])))
            if "until_sky" in a:
                args["until_sky"] = bool(a["until_sky"])
        elif name == "tunnel":
            args = {"length": int(a.get("length") or a.get("count") or 16),
                    "ores": str(a.get("ores") or a.get("blocks") or "")}
        elif name == "explore_cave":
            args = {"max_seconds": max(5, min(90, int(a.get("max_seconds") or 75))),
                    "max_nodes": max(1, min(20, int(a.get("max_nodes") or 12))), "ores": str(a.get("ores") or a.get("blocks") or "")}
        elif name == "light_area":
            args = {"force": bool(a.get("force", False))}
        elif name == "collect_drops" and a.get("radius") is not None:
            args = {"radius": max(1, min(48, int(a["radius"])))}
        elif name == "declutter" and a.get("want_free") is not None:
            args = {"want_free": max(1, min(20, int(a["want_free"])))}
            if "blocks_only" in a:
                args["blocks_only"] = bool(a["blocks_only"])
        elif name == "chop_tree":
            args = {k: a[k] for k in ("max_logs", "radius", "prefer") if k in a}
        elif name == "go_to":
            if any(a.get(k) is None for k in ("x", "y", "z")):
                return "go_to 要写 x, y, z。"
            args = {"x": int(a["x"]), "y": int(a["y"]), "z": int(a["z"])}
        elif name == "hunt":
            args = {"count": int(a.get("count") or 3)}
        elif name == "hunt_resource":
            args = {"item": str(a.get("item") or ""), "count": int(a.get("count") or 1)}
        elif name == "goal" and a.get("item"):
            args = {"item": str(a["item"]), "count": int(a.get("count") or 1), "by": "self"}
        return game.skills.start(name, **args)

    def know(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None or getattr(game, "knowledge", None) is None:
            return "游戏身体没有连上。"
        return game.knowledge.ask(str(a.get("query") or ""))

    def map_(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        s = game.skills
        me = {"x": game.last_pos[0], "y": game.last_pos[1], "z": game.last_pos[2]} if game.last_pos else None
        name = s.index.name if s.index is not None else (lambda i: i.split(":")[-1])
        mark = str(a.get("mark") or "").strip()
        if mark:
            if not me:
                return "不知道自己在哪，记不了。"
            s.places.add("spot", me["x"], me["y"], me["z"], note=mark)
            game.remember(f"我在 Minecraft 里记下了一个地方：{mark}")
            return f"记下了：{mark}（{int(me['x'])},{int(me['y'])},{int(me['z'])}）。"
        q = str(a.get("find") or "").strip()
        if q:
            ids = s.index.search(q, 3) if s.index is not None and ":" not in q else [q]
            hits = [(i, p) for i in ids for p in s.places.where_is(i)]
            spots = [p for p in s.places.places if q in str(p.get("note", ""))]
            lines = [f"{name(i)}×{p['items'][i]} 在箱子 ({p['x']},{p['y']},{p['z']})" for i, p in hits[:4]]
            lines += [f"{p['note']} ({p['x']},{p['y']},{p['z']})" for p in spots[:4]]
            return "\n".join(lines) or f"我不记得把 {q} 放在哪了。"
        return "\n".join(s.places.describe(me, name)) or "地图上还什么都没有。"

    def remember(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        note = str(a.get("note") or "").strip()
        if not note:
            return "要记住什么？"
        game.skills.memory.learn(note, about=str(a.get("about") or ""), source="self")
        return "记住了。"

    def recall(gw, ctx, a):
        game = getattr(gw, "game", None)
        memory = getattr(game, "game_memory", None)
        if memory is None:
            return "游戏向量记忆尚未启用。"
        query = str(a.get("query") or "").strip()[:500]
        if not query:
            return "要回忆什么？可以问家、箱子、去过的地方或上次怎么做的。"
        pos = getattr(game, "last_pos", None)
        near = ({"x": pos[0], "y": pos[1], "z": pos[2],
                 "dim": getattr(game, "dimension", "") or getattr(getattr(game, "body", None), "state", {}).get("dimension", "")}
                if pos else None)
        hits = memory.recall(query, k=max(1, min(8, int(a.get("limit") or 6))), near=near)
        return json.dumps(hits, ensure_ascii=False)[:8000] if hits else "没有找到相关游戏记忆。"

    def goal(gw, ctx, a):
        game = getattr(gw, "game", None)
        if game is None:
            return "游戏身体没有连上。"
        s = game.skills
        item = str(a.get("item") or "").strip()
        if not item or item in ("status", "看看"):
            if not s.goal_item:
                return "现在没有目标（身体会按常规进度自己找下一个）。"
            why = {"self": "自己定的", "quest": f"任务书：{getattr(s, 'goal_quest', '')}"}.get(s.goal_by, "常规进度")
            return f"目标：{s.goal_item} ×{s.goal_count}（{why}）；现在：{s.goal_note or '还没开始'}"
        if item in ("auto", "自动", "stop", "clear"):
            return s.start("goal", clear=True, by="auto")
        if ":" not in item and s.index is not None:
            found = s.index.search(item, 1)
            item = found[0] if found else f"minecraft:{item}"
        from src.levels.l8_agency.mc_goal import satisfied
        try:
            have = s.crafter.counts()
        except Exception:
            have = {}
        if satisfied(item, int(a.get("count") or 1), have):
            return (f"不用做：包里已经有 {item.split(':')[-1]} 或更好的同类工具"
                    "（做出来也会被当成杂物扔掉）。换一个真正缺的东西，或者用 mc_quest 看任务书要什么。")
        game.shelved.pop(item, None)
        return s.start("goal", item=item, count=int(a.get("count") or 1), by="self")

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
        base = dict(available=_available, timeout=15, group=SERVER)
        base.update(kw)
        return base

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
        Tool("mc_skill", "成套的动作，在后台自己做完（像玩家一样先走过去、看到再动手，不隔空）：chop_tree 砍附近一整棵树并捡木头，"
             "mine 挖附近某种方块（blocks 写 id），craft 合成（item 写物品 id，会自己摆格子、需要时找/放/做工作台、缺木板木棍会先做），"
             "place 放方块（item；写 x,y,z 放到那里，不写就放在身边），equip 拿起合适的工具（kind: pickaxe/axe/sword/shovel），"
             "tidy 把工具整理到快捷栏，smelt 用熔炉烧（item 写要烧的或想烧出的），dig_down/dig_up 挖楼梯下/上（to_y 写目标高度），"
             "tunnel 挖 2 格高的矿道（count 几格，blocks 顺手挖的矿），goal 朝目标一步步做（item 可换目标），"
             "hunt 打动物弄吃的（会自己烤熟），forage 捡可食掉落/采成熟作物/做面包，dress 穿上最好的盔甲、盾牌放副手，declutter 扔掉背包里的杂物，"
             "explore_cave 沿已观察天然洞穴探路并顺手采可见矿，light_area 检查并补照明，set_home 在地面上安家（箱子、工作台、熔炉），build_hut 在家址分阶段搭5×5木石玻璃窗山墙小屋，go_home 回家并把东西放进箱子，store 把东西放进附近我的箱子，"
             "build_portal 在家旁续建并核实下界门，enter_portal 进入已确认的门并验证维度，hunt_resource 按实际掉落获取生物资源（item/count），"
             "end_step 按实际观测分步找要塞/填末地门/进入末地/水晶与龙战，fill_water_bucket 去安全水源取水，"
             "go_to 走去某个坐标（写 x,y,z；在地下会先挖上去），"
             "collect_drops 捡地上的东西，go_to_player 走到某人身边（写 player），stop 停下，status 看进度。",
             _props(["name"], name=_s("goal / chop_tree / mine / craft / smelt / dig_down / dig_up / tunnel / hunt / forage / dress / declutter / "
                                      "explore_cave / light_area / set_home / build_hut / build_portal / enter_portal / hunt_resource / end_step / fill_water_bucket / go_home / store / go_to / place / equip / tidy / collect_drops / "
                                      "go_to_player / stop / status"),
                    player=_s("go_to_player 时写玩家名"), blocks=_s("mine 时写方块 id，逗号分隔"),
                    item=_s("craft / place 时写物品 id，比如 minecraft:wooden_pickaxe"), kind=_s("equip 时写工具种类"),
                    max_seconds=_i("explore_cave 本轮秒数，最多90"), max_nodes=_i("explore_cave 本轮走过节点，最多20"),
                    force={"type": "boolean", "description": "light_area：夜间强制检查露天照明"},
                    count=_i("mine 挖几块 / craft 做几个"), to_y=_i("dig_down/dig_up 的目标高度"),
                    max_steps=_i("楼梯单次最大步数"), length=_i("矿道长度"), ores=_s("矿道顺手挖的矿"),
                    radius=_i("砍树/捡物搜索半径"), max_logs=_i("砍树原木上限"), prefer=_s("优先寻找的原木id"),
                    want_free=_i("清包需要腾出的格数"), until_sky={"type": "boolean", "description": "向上挖到露天即停"},
                    blocks_only={"type": "boolean", "description": "只丢冗余廉价方块，并保留64个建筑方块"},
                    x=_i("place 的 x"), y=_i("place 的 y；dig_down 挖到的高度"), z=_i("place 的 z")),
             skill, **common()),
        Tool("mc_know", "查游戏知识：物品怎么做/有什么用/用什么工具挖/从哪来、这个整合包的模组和任务、老玩家的经验"
                        "（写物品名或 id，或者一个话题：发展路线、挖矿、冶炼、战斗、盔甲、吃饭、夜晚、附魔、下界、整合包、高手习惯……）。",
             _props(["query"], query=_s("要查的东西，比如 铁镐、minecraft:shield、挖矿、整合包")), know, **common(read_only=True)),
        Tool("mc_map", "看我脑子里的地图：家在哪、我的箱子里放了什么、熔炉/工作台/矿洞/见过的矿在哪、上次死在哪；"
                       "find 写物品找它放在哪个箱子；mark 写一个名字把脚下这个地方记下来。",
             _props(find=_s("找东西放在哪，比如 铁锭"), mark=_s("给现在的位置起个名字记下来，比如 钻石矿洞入口")),
             map_, **common(read_only=True)),
        Tool("mc_remember", "把在游戏里学到的经验记下来（怎么做成的、为什么失败、哪里有什么），以后玩的时候会想起来。",
             _props(["note"], note=_s("一句经验，比如 深板岩层铁很少，要回到 y≈16 挖"), about=_s("跟什么有关，比如 铁、钻石、矿道")),
             remember, **common(read_only=True)),
        Tool("mc_recall", "检索自己的游戏经历、家、箱子物品、地图和经验，使用与语义记忆相同的向量编码器。结果是历史观测，需要时到游戏中复查。",
             _props(["query"], query=_s("要回忆什么"), limit=_i("返回条数，1到8，默认6")),
             recall, **common(read_only=True)),
        Tool("mc_goal", "看或换你在游戏里的目标。身体会自己一步步朝目标做（缺什么去拿什么、该下矿就下矿、该烧就烧）。"
                        "item 写想要的东西（id 或名字）；写 auto 交回常规进度；不写就是看现在的目标。",
             _props(item=_s("比如 minecraft:iron_sword、铁镐；auto 交回常规进度"), count=_i("要几个，默认 1")), goal, **common()),
        Tool("mc_find_blocks", "在自己周围找某些方块的位置（比如原木、矿石、箱子），按距离给出坐标。",
             _props(["blocks"], blocks=_s("方块 id，逗号分隔，比如 minecraft:oak_log"), radius=_i("多远以内（格），默认 32"),
                    limit=_i("最多几个，默认 20")), find, **common(read_only=True)),
        Tool("mc_who_is_near", "看看身边有哪些玩家（叫得出名字的会写名字、多远）和哪些生物。",
             _props(radius=_i("多远以内，默认 48 格")), near, **common(read_only=True)),
        Tool("mc_rejoin_world", "掉线或者回到主菜单之后，重新进入上一次玩的世界（自动重连一直没成时用这个）。",
             _props(), rejoin, **common()),
        Tool("mc_look", "真的看一眼游戏画面：截一张图，随这一轮接下来的思考一起交给模型。看清地形、界面、"
                        "怪物动作、远处的东西或卡住的时候用（平时每 20 秒左右也会自动看一眼）。",
             _props(), look, **common(read_only=True)),
        Tool("mc_look_at", "像人一样慢慢把视线转到某个坐标（不是瞬间转头）。看方块、看人、看路时用。",
             _props(["x", "y", "z"], x=_i("坐标 x"), y=_i("坐标 y"), z=_i("坐标 z")), look_at, **common()),
        Tool("mc_mine_block", "像玩家一样挖掉某个坐标的方块：先走近到够得着，再慢慢转过去，自动换最好的工具，"
                              "然后按住挖，直到挖掉。不要用别的方式隔空挖。",
             _props(["x", "y", "z"], x=_i("方块 x"), y=_i("方块 y"), z=_i("方块 z")), mine, **common(timeout=50)),
        Tool("mc_place_block", "像玩家一样在某个空坐标放一个方块：先走近，转向要依附的面，拿好方块再放。"
                               "item 不写就自动从包里挑一种方块。",
             _props(["x", "y", "z"], x=_i("目标空位 x"), y=_i("目标空位 y"), z=_i("目标空位 z"),
                    item=_s("方块物品 id，比如 minecraft:cobblestone；不写就自动挑")), place, **common(timeout=50)),
        Tool("mc_follow", "跟着某个玩家走（身体会自己跟上、看着 ta，直到你说停或者时间到）；player 写 stop 就不跟了。",
             _props(["player"], player=_s("玩家名，或 stop"), minutes=_i("最多跟多久（分钟），默认 20")), follow, **common()),
        Tool("mc_craft", "在打开的工作台（或背包合成格）里按配方摆好材料，然后从结果格拿走。",
             _props(["ref"], ref=_s("mc_recipes 给的 ref"), all=_s("true 就能做多少做多少")), craft, **common()),
    ]
