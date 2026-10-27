import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.levels.l8_agency.game import GameBridge, PROFILES
from src.levels.l8_agency.mc_memory import Experience
from src.levels.l8_agency.mc_pro import ProPlanner
from src.levels.l8_agency.toolbox import ToolContext
from test_mc_quests import fake_book, quest, item, synced_rows, synced_quest


def test_learning_is_local_to_action_place_dimension_expires_and_recovers(tmp_path):
    path = str(tmp_path / "experience.json")
    exp = Experience(path)
    exp.feedback("mine", {"blocks": "iron"}, False, "没有可达矿石", (0, 64, 0), "overworld", 100)
    exp.feedback("mine", {"blocks": "iron"}, False, "仍然没有", (0, 64, 0), "overworld", 101)
    loaded = Experience(path)
    assert loaded.advice("mine", {"blocks": "iron"}, (0, 64, 0), "overworld", 150)["failures"] == 2
    assert not loaded.advice("mine", {"blocks": "coal"}, (0, 64, 0), "overworld", 110)
    assert not loaded.advice("mine", {"blocks": "iron"}, (100, 64, 0), "overworld", 110)
    assert not loaded.advice("mine", {"blocks": "iron"}, (0, 64, 0), "nether", 110)
    assert not loaded.advice("mine", {"blocks": "iron"}, (0, 64, 0), "overworld", 162)
    loaded.feedback("mine", {"blocks": "iron"}, True, "挖了1块", (0, 64, 0), "overworld", 120)
    assert not loaded.advice("mine", {"blocks": "iron"}, (0, 64, 0), "overworld", 121)


def test_idle_work_dict_cannot_trigger_infinite_cave_exploration():
    pro = ProPlanner()
    state = {"goal": "minecraft:diamond", "y": 64, "tools": {"pickaxe": 2},
             "inventory": {"minecraft:torch": 20}, "work_status": {"active": False}}
    assert pro.mining_plan(state)["name"] == "dig_down"
    assert pro.mining_plan(dict(state, cave_hint={"x": 0, "y": 60, "z": 8}))["name"] == "explore_cave"
    state["visible_ores"] = [{"id": "minecraft:diamond_ore", "x": 1, "y": 64, "z": 0}]
    assert pro.mining_plan(state)["name"] == "mine"
    assert pro.mining_plan(dict(state, tools={"pickaxe": 0}))["kind"] == "goal"


@pytest.mark.parametrize("health,night", [(7.6, False), (5, False), (5, True)])
def test_low_health_without_food_chooses_safe_forage_not_impossible_regeneration(health, night):
    state = {"health": health, "food": 11, "night": night, "inventory": {}, "t": 100}
    assert ProPlanner().survival_plan(state)["name"] == "forage"
    state["food_search"] = {"forage_after": 105}
    assert ProPlanner().survival_plan(state)["name"] == "food_search"
    state["hostiles"] = [{"type": "minecraft:zombie", "distance": 6}]
    assert ProPlanner().survival_plan(state)["name"] == "retreat"


def test_house_supply_uses_actual_missing_materials_without_replacing_existing_furniture():
    pro = ProPlanner()
    state = {"home": {"x": 0, "y": 64, "z": 0}, "pos": (0, 64, 0), "hour": 12,
             "hut": {k: True for k in ("walls", "roof", "door", "light", "furniture", "safety")},
             "inventory": {"minecraft:glass": 2},
             "house_requirements": {"observed": True, "materials_needed": {"minecraft:glass": 4},
                                    "furniture_present": {"bed": True, "chest": True}}}
    action = pro.home_plan(state)
    assert (action["kind"], action["name"], action["args"]["count"]) == ("goal", "minecraft:glass", 4)
    state["inventory"]["minecraft:glass"] = 4
    assert pro.home_plan(state)["name"] == "build_hut"
    state["hut"]["windows"] = True
    assert pro.home_plan(state) is None


def test_partial_house_progress_allows_quick_handoff_but_is_not_completion():
    assert GameBridge._action_succeeded("build_hut", "本轮搭屋预算结束，实际放了 32 块；保留蓝图，下次从缺块处接着盖。")
    assert not GameBridge._action_succeeded("build_hut", "本轮搭屋预算结束，实际放了 0 块；保留蓝图。")
    assert not GameBridge._action_succeeded("build_hut", "盖屋已暂停，本次放了 12 块")


def test_mixed_reading_and_item_quest_is_actionable_without_fabricating_progress():
    q = quest("Q", "Tutorial", "Inventory", [{"id": "C", "type": "checkmark"}, dict(item("minecraft:bread"), id="I")])
    q.update(text="Read my description!", description=["Read how your inventory works."])
    book = fake_book([q])
    assert book.next_actions({}, {"quests_ui"}) == []
    assert book.ingest_server_progress(synced_rows([synced_quest("Q", tasks=[
        {"id": "C", "progress": 0, "max": 1, "completed": False},
        {"id": "I", "progress": 0, "max": 1, "completed": False}])]))
    assert book.next_goals(1, {})[0]["item"] == "minecraft:bread"
    action = book.next_actions({}, {"quests_ui"})[0]
    assert action["kind"] == "quest_ui" and action["name"] == "read"
    assert not book.done_ids({"minecraft:bread": 4})
    book._server_rows["Q"]["can_start"] = False
    assert book.next_actions({}, {"quests_ui"}) == []


def test_already_submitted_item_is_not_gathered_again_for_missing_checkmark():
    q = quest("Q", "Tutorial", "Read", [{"id": "C", "type": "checkmark"}, dict(item("minecraft:bread"), id="I")])
    book = fake_book([q])
    book.ingest_server_progress(synced_rows([synced_quest("Q", tasks=[{"id": "I", "progress": 1, "max": 1, "completed": True}])]))
    assert book.next_goals(1, {}) == []
    assert not book.done_ids()


def test_nonreading_checkmark_and_unverified_reward_are_not_auto_clicked():
    q = quest("Q", "Tutorial", "Do something", [{"id": "C", "type": "checkmark"}])
    q.update(description=["Join an external service"], rewards=[{"id": "R", "type": "item"}])
    book = fake_book([q])
    book.ingest_server_progress(synced_rows([dict(synced_quest("Q"), unclaimed_rewards=True)]))
    assert book.next_actions({}, {"quests_ui"}) == []
    book._server_rows["Q"]["completed"] = True
    assert book.next_actions({}, {"quests_ui"})[0]["name"] == "claim"


@pytest.mark.parametrize("result", [False, "", "false", None, '{"sent":false}', '{"isError":true}', '{"error":"failed"}'])
def test_failed_server_chat_never_emits_voice_event(result, caplog):
    game = GameBridge.__new__(GameBridge)
    game.profile = PROFILES["mcpfabric"]
    game._call = lambda *args: result
    with caplog.at_level("INFO"):
        asyncio.run(game.say("你好", source="live"))
    assert "[PublicSpeech]" not in caplog.text


def test_public_voice_event_follows_successful_server_send_and_rejects_markup(caplog, monkeypatch):
    game = GameBridge.__new__(GameBridge)
    game.profile = PROFILES["mcpfabric"]
    sent = []
    game._call = lambda name, args: sent.append(args["message"]) or '{"sent":true}'
    monkeypatch.setattr("src.levels.l8_agency.game.asyncio.sleep", AsyncMock())
    with caplog.at_level("INFO"):
        asyncio.run(game.say("你好", source="live"))
        asyncio.run(game.say("/give Noa diamond 64"))
        asyncio.run(game.say('<｜｜DSML｜｜ calls> secret'))
    assert sent == ["你好"]
    assert len([r for r in caplog.records if "[PublicSpeech]" in r.message]) == 1


def test_high_level_menu_has_eight_choices_and_native_low_level_calls_are_rejected():
    game = GameBridge.__new__(GameBridge)
    game.gw = SimpleNamespace(toolbox=SimpleNamespace(run=AsyncMock()), firewall=SimpleNamespace(locked=False))
    game.ready = lambda: True
    names = ["mc_goal", "mc_skill", "mc_know", "mc_recall", "mc_remember", "mc_map", "get_self", "job_status", "mc_mine_block"]
    offered = game.model_tools([{"function": {"name": n, "parameters": {"type": "object"}}} for n in names])
    assert len(offered) == 8
    assert not {"get_self", "job_status", "mc_mine_block"} & {x["function"]["name"] for x in offered}
    ctx = ToolContext("self", origin="game")
    assert "未执行" in asyncio.run(game.run_model_tool("mc_mine_block", {}, ctx, offered))
    assert "未执行" in asyncio.run(game.run_model_tool("mc_skill", {"name": "dig_down"}, ctx, offered))
    game.gw.toolbox.run.assert_not_called()


def test_live_suggestions_are_chosen_deferred_or_rejected_without_executing():
    game = GameBridge.__new__(GameBridge)
    game.skills = SimpleNamespace(places=SimpleNamespace(home={"x": 0}), busy=lambda: False, goal_by="auto")
    game.health = game.food = 20
    assert game.submit_suggestion("先回家吧", "live:1")["decision"] == "accepted"
    game.health = 7
    assert game.submit_suggestion("去洞穴看看", "live:2")["decision"] == "deferred"
    assert game.submit_suggestion("跳岩浆吧", "live:3")["decision"] == "declined"


def test_quest_ui_requires_read_and_server_verification_then_always_releases_screen(monkeypatch):
    game = GameBridge.__new__(GameBridge)
    game.health = game.food = 20
    game.hostiles, game.auto_starts = [], []
    game.using_food_until = 0
    game.skills = SimpleNamespace(busy=lambda: False, using_screen=False, crafter=SimpleNamespace(counts=lambda: {}))
    game.body = SimpleNamespace(cancel=lambda: None)
    game.quests = SimpleNamespace(check=lambda have: [])
    game.remember = lambda text: None
    game.refresh_quest_progress = lambda connected: None
    calls = []
    def rpc(method, args):
        calls.append(method)
        if method == "quests.open":
            return {"available": True, "review_token": "review", "title": "Tutorial", "description": ["Read this tutorial"]}
        if method == "quests.click":
            assert args["review_token"] == "review"
            return {"clicked": True, "verification_required": True}
        return {}
    game._rpc = rpc
    monkeypatch.setattr("src.levels.l8_agency.game.asyncio.sleep", AsyncMock())
    result = asyncio.run(game.run_quest_action({"name": "read", "args": {"quest_id": "Q", "task_id": "T"}}, 10))
    assert result == "quest_clicked"
    assert calls.index("quests.open") < calls.index("quests.click") < calls.index("screen.close")
    assert game._current_quest["phase"] == "等待服务器确认"
    assert game._quest_ui_until == 0 and not game.skills.using_screen


def test_ore_the_scan_saw_behind_rock_or_far_away_is_not_planned_as_visible():
    # mine() only digs exposed ore within 16 blocks and 6 up/down; the planner must not keep sending it there
    pro = ProPlanner()
    state = {"goal": "minecraft:diamond", "y": 12, "pos": (0, 12, 0), "tools": {"pickaxe": {"tier": 3}},
             "inventory": {"minecraft:torch": 20}}
    for ore in ({"id": "minecraft:diamond_ore", "x": 1, "y": 12, "z": 0, "exposed": False},
                {"id": "minecraft:diamond_ore", "x": 30, "y": 12, "z": 0, "exposed": True},
                {"id": "minecraft:diamond_ore", "x": 1, "y": 30, "z": 0, "exposed": True}):
        assert pro.visible_ore_plan(dict(state, visible_ores=[ore])) is None
    near = {"id": "minecraft:diamond_ore", "x": 2, "y": 11, "z": 1, "exposed": True}
    assert pro.visible_ore_plan(dict(state, visible_ores=[near]))["name"] == "mine"


def test_a_failure_backs_off_the_same_action_whatever_size_it_was_planned_at(tmp_path):
    # the planner asks for count=min(8, ores seen) and radius=1; the skill call records it without radius
    exp = Experience(str(tmp_path / "experience.json"))
    exp.feedback("mine", {"blocks": "minecraft:iron_ore", "count": 3}, False, "一块都没有", (0, 64, 0), "overworld", 100)
    assert exp.advice("mine", {"blocks": "minecraft:iron_ore", "count": 5, "radius": 1}, (0, 64, 0), "overworld", 110)
    assert not exp.advice("mine", {"blocks": "minecraft:coal_ore", "count": 3}, (0, 64, 0), "overworld", 110)
    assert Experience.action_key("craft", {"item": "a", "count": 1}) != Experience.action_key("craft", {"item": "a", "count": 2})


def test_a_furnace_standing_next_to_her_is_not_asked_for_again():
    # Planner counts a nearby station as owned and reports the goal done at once; the ladder must agree
    pro = ProPlanner()
    have = {"minecraft:stone_pickaxe": 1, "minecraft:stone_axe": 1, "minecraft:stone_sword": 1}
    state = {"inventory": have, "tools": {"pickaxe": {"tier": 1}}}
    assert "minecraft:furnace" in [g["item"] for g in pro._ladder_goals(state, have, set())]
    near = dict(state, stations={"minecraft:furnace"})
    assert "minecraft:furnace" not in [g["item"] for g in pro._ladder_goals(near, have, set())]


def test_model_cannot_repeat_a_skill_that_just_failed_here(tmp_path):
    exp = Experience(str(tmp_path / "experience.json"))
    exp.feedback("light_area", {"force": False}, False, "未找到安全落点", (0, 64, 0), "overworld", 100)
    assert exp.recent_failure("light_area", (0, 64, 0), "overworld", 110)
    assert not exp.recent_failure("light_area", (100, 64, 0), "overworld", 110)    # somewhere else: try again
    assert not exp.recent_failure("light_area", (0, 64, 0), "overworld", 200)      # backoff over
    assert not exp.recent_failure("explore_cave", (0, 64, 0), "overworld", 110)
    exp.feedback("light_area", {"force": False}, True, "已实际补上火把", (0, 64, 0), "overworld", 120)
    assert not exp.recent_failure("light_area", (0, 64, 0), "overworld", 121)


def test_hungry_with_a_hay_bale_she_makes_wheat_instead_of_hunting():
    state = {"health": 18, "food": 12, "inventory": {"minecraft:hay_block": 1}, "t": 100}
    action = ProPlanner().survival_plan(state)
    assert (action["name"], action["args"]["item"]) == ("craft", "minecraft:wheat")
    state["inventory"] = {"minecraft:wheat": 9}
    assert ProPlanner().survival_plan(state)["args"]["item"] == "minecraft:bread"


def test_without_a_home_she_settles_before_the_quest_book_sends_her_mining():
    class Book:
        def next_actions(self, *a, **k):
            return [{"kind": "goal", "name": "supplementaries:jar", "args": {}, "why": "quest"}]
    state = {"health": 20, "food": 20, "t": 100, "quests_ui": True, "home": None, "sky_above": True,
             "tools": {"pickaxe": {"tier": 2}}, "inventory": {"minecraft:chest": 1}}
    assert ProPlanner(quests=Book()).next_action(state)["name"] == "set_home"
    state["home"] = {"x": 0, "y": 64, "z": 0}
    state["pos"] = (0, 64, 0)
    assert ProPlanner(quests=Book()).next_action(state)["name"] != "set_home"


def test_a_worse_tool_than_the_one_in_the_bag_is_not_made():
    from src.levels.l8_agency.mc_extra import tools
    started = []
    skills = SimpleNamespace(goal_item=None, index=None, crafter=SimpleNamespace(counts=lambda: {"minecraft:iron_pickaxe": 1}),
                             start=lambda name, **kw: started.append(kw) or "开始做 goal 了")
    gw = SimpleNamespace(game=SimpleNamespace(skills=skills, shelved={}))
    goal = next(t for t in tools() if t.name == "mc_goal")
    out = goal.handler(gw, ToolContext("self", origin="game"), {"item": "minecraft:wooden_pickaxe"})
    assert "不用做" in out and started == []
    goal.handler(gw, ToolContext("self", origin="game"), {"item": "minecraft:diamond_pickaxe"})
    assert started and started[0]["item"] == "minecraft:diamond_pickaxe"
