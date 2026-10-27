import asyncio
import json
import threading
import time
from types import SimpleNamespace

from src.levels.l2_memory.embedding import Embedder
from src.levels.l8_agency.game import GameBridge
from src.levels.l8_agency.mc_game_memory import GameMemory
from src.levels.l8_agency.mc_memory import Experience
from src.levels.l8_agency.mc_places import Places
from src.levels.l8_agency.mc_pro import ProPlanner
from src.levels.l8_agency.mc_runtime import GameRuntime


def test_bridge_constructor_reuses_semantic_embedder(tmp_path):
    shared = Embedder()
    gw = SimpleNamespace(mc_client_dir=str(tmp_path),
                         memory=SimpleNamespace(db=SimpleNamespace(db_path=str(tmp_path / "memory.sqlite"))),
                         episodes=SimpleNamespace(embedder=shared), autonomous_groups=set(),
                         mcp=SimpleNamespace(config={"minecraft": {"profile": "mcpfabric"}}))
    game = GameBridge(gw)
    try:
        assert game.game_memory.embedder is shared
        assert game.skills.body is game.body
        assert game.skills.on_action == game._completed_action
        assert game.runtime.snapshot_path == str(tmp_path / "live" / "overlay" / "game_state.json")
        game.token_average = lambda: {}
        status = game.status()
        assert status["game_memory"]["shared_embedder"] is True
        assert {"source", "files", "chapters", "quests", "errors"} <= status["quests"].keys()
    finally:
        game.runtime.close()
        game.game_memory.close()


def test_slow_embeddings_do_not_block_body_snapshot(tmp_path):
    path = tmp_path / "state.json"
    runtime = GameRuntime(str(path), capacity=1)
    entered, release = threading.Event(), threading.Event()

    def slow():
        entered.set()
        release.wait(2)

    try:
        assert runtime.submit(slow)
        assert entered.wait(1)
        assert runtime.submit(lambda: None)
        assert not runtime.submit(lambda: None)
        runtime.snapshot({"t": 1, "doing": "挖矿"})
        end = time.monotonic() + 1
        while not path.exists() and time.monotonic() < end:
            time.sleep(0.02)
        assert json.loads(path.read_text(encoding="utf-8"))["doing"] == "挖矿"
        assert runtime.dropped == 1
    finally:
        release.set()
        assert runtime.flush()
        runtime.close()


def test_background_start_is_not_a_memory_of_success():
    memory = GameMemory(embedder=Embedder())
    try:
        assert memory.record_action("mc_skill", {"name": "mine"}, "开始做 mine 了（在后台进行）") is None
        assert memory.stats()["records"] == 0
        memory.record_action("mc_skill", {"name": "mine", "blocks": "minecraft:iron_ore"},
                             "挖了 3 块", pos=(1, 16, 4), dim="minecraft:overworld")
        got = memory.recent()[0]
        assert got["dim"] == "minecraft:overworld"
        assert "3块铁矿石" in got["text"]
    finally:
        memory.close()


def test_snapshot_retries_windows_reader_lock_without_memory_error(tmp_path, monkeypatch):
    import os
    real_replace = os.replace
    calls = []
    def locked_once(source, target):
        calls.append(target)
        if len(calls) == 1:
            raise PermissionError("reader has file open")
        return real_replace(source, target)
    monkeypatch.setattr("src.levels.l8_agency.mc_runtime.os.replace", locked_once)
    path = tmp_path / "state.json"
    runtime = GameRuntime(str(path))
    try:
        runtime.snapshot({"t": 123})
        deadline = time.monotonic() + 2
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert json.loads(path.read_text()) == {"t": 123}
        assert runtime.snapshot_retries == 1
        assert runtime.errors == runtime.memory_errors == runtime.snapshot_errors == 0
    finally:
        runtime.close()


def test_memory_error_counter_identifies_worker_failure():
    runtime = GameRuntime()
    try:
        def broken():
            raise ValueError("bad embedding")
        runtime.submit(broken)
        assert runtime.flush()
        assert runtime.errors == runtime.memory_errors == 1
        assert runtime.snapshot_errors == 0
        assert runtime.last_error == "memory:ValueError"
    finally:
        runtime.close()


def test_dimensions_do_not_merge_homes_or_chests():
    places = Places()
    changes = []
    places.on_change = changes.append
    places.set_home(0, 64, 0)
    places.add("chest", 1, 64, 0, items={"minecraft:iron_ingot": 4})
    places.add("chest", 1, 64, 0, items={"minecraft:iron_ingot": 4})
    assert len(changes) == 2  # unchanged chest observations do not spam embeddings
    places.active_dim = "minecraft:the_nether"
    assert places.home is None
    assert places.where_is("minecraft:iron_ingot") == []
    assert places.near({"x": 0, "y": 64, "z": 0}) == []
    places.set_home(0, 64, 0)
    assert len(places.places) == 3
    places.active_dim = "minecraft:overworld"
    assert places.home["dim"] == "minecraft:overworld"


def test_partial_return_home_is_not_remembered_as_arrival():
    memory = GameMemory(embedder=Embedder())
    try:
        memory.record_action("mc_skill", {"name": "go_home"}, "走到一半走不动了（还差 40 格）。")
        got = memory.recent()[0]
        assert "失败" in got["tags"]
        assert "还差 40 格" in got["text"]
    finally:
        memory.close()


def test_game_only_snapshot_never_exports_private_context():
    game = GameBridge.__new__(GameBridge)
    captured = []
    game.runtime = SimpleNamespace(snapshot=captured.append)
    game.gw = SimpleNamespace(inner_monologue="QQ私密消息", attention_schema="私人姓名")
    game.body = SimpleNamespace(state={"dimension": "minecraft:overworld"})
    game.skills = SimpleNamespace(places=Places(), current="mine", goal_item="minecraft:iron_pickaxe")
    game.public_reason = "先做铁镐再下矿"
    game._body_event({"action": "mine", "busy": True, "yaw": 22, "health": 20,
                      "note": "任意内部字符串", "token": "secret", "private": game.gw.inner_monologue})
    encoded = json.dumps(captured[0], ensure_ascii=False)
    assert "QQ私密" not in encoded and "secret" not in encoded and "任意内部" not in encoded
    assert captured[0]["body"]["yaw"] == 22
    assert captured[0]["doing"] == "挖矿"
    assert captured[0]["quests_summary"] == {"available": False, "completed": None, "total": None, "current": None}
    game.quests = SimpleNamespace(server_status=lambda: {"server_verified": True, "completed": 17, "synced_quests": 143,
                                                        "private": "unrelated"})
    game._body_event({"action": "idle"})
    assert captured[-1]["quests_summary"] == {"available": True, "completed": 17, "total": 143, "current": None}
    assert "unrelated" not in json.dumps(captured[-1])


def test_maps_lessons_and_completed_skills_feed_shared_embedder():
    game = GameBridge.__new__(GameBridge)
    emb = Embedder()
    game.game_memory = GameMemory(embedder=emb)
    game.runtime = GameRuntime()
    game.last_pos = (10, 65, 10)
    game.skills = SimpleNamespace(places=Places(), memory=Experience())
    game.skills.places.on_change = game._remember_place
    game.skills.memory.on_learn = game._remember_lesson
    try:
        game.skills.places.set_home(10, 65, 10)
        game.skills.places.add("chest", 11, 65, 10, items={"minecraft:iron_ingot": 12})
        game.skills.memory.learn("铁矿在 y=16 好找", about="铁矿")
        game._completed_action("craft", {"item": "minecraft:iron_pickaxe"}, "合成了铁镐")
        assert game.runtime.flush()
        assert game.game_memory.embedder is emb
        assert game.game_memory.stats()["records"] == 5  # action plus structured retry lesson
        assert any(any(t.startswith("action-key:") for t in row["tags"]) for row in game.game_memory.recent(10))
        assert "铁锭×12" in game.game_memory.recall("箱子铁锭")[0]["text"]
        assert game.game_memory.recall("铁矿", kinds=["lesson"])
    finally:
        game.runtime.close()
        game.game_memory.close()


def test_expert_plan_dispatches_one_goal_not_per_keypress():
    game = GameBridge.__new__(GameBridge)
    goals, calls = [], []
    game.skills = SimpleNamespace(set_goal=lambda *args: goals.append(args))
    game.pro = ProPlanner()
    game.auto_starts = []

    async def run(name, args, ctx):
        calls.append((name, args))
        return "开始做 goal 了"

    game.gw = SimpleNamespace(toolbox=SimpleNamespace(run=run))
    result = asyncio.run(game.run_expert_action({"kind": "goal", "name": "minecraft:iron_pickaxe",
                                               "args": {"count": 1, "by": "quest"},
                                               "why": "任务需要铁镐"}, 123, None))
    assert goals == [("minecraft:iron_pickaxe", 1, "quest")]
    assert calls == [("mc_skill", {"name": "goal"})]
    assert result == "mc_skill"


def test_only_measured_success_accelerates_next_goal(monkeypatch):
    monkeypatch.setattr("src.levels.l8_agency.game.clock.now", lambda: 100.0)
    game = GameBridge.__new__(GameBridge)
    game.record_action = lambda *args: None
    game.last_sched = 80.0
    game._completed_action("mine", {}, "挖了 2 块。捡了 1 次东西。")
    assert game._fast_work_after == 102.0
    for result in ("挖了 0 块。", "开始做 mine 了", "没找到矿石", "挖了 2 块但超时取消了"):
        game._completed_action("mine", {}, result)
        assert game._fast_work_after == 0.0
        assert game.last_sched == 100.0
    assert game._action_succeeded("craft", "做好了 iron_pickaxe ×1。")
    assert not game._action_succeeded("craft", "做好了 iron_pickaxe ×0。")
    assert game.AUTO_PER_HOUR == 240


def test_dragon_controller_keeps_bow_but_normal_mobs_and_low_health_have_reflexes():
    game = GameBridge.__new__(GameBridge)
    game.ready = lambda: True
    game.in_world = game.connected = True
    game.using_food_until = game.blocking_until = 0
    game.health, game.food, game.last_eat = 20, 20, 0
    calls = []
    game._hold_work = lambda seconds: calls.append("hold")
    game._flee = lambda target, now: calls.append(target["type"]) or "flee"
    game.hostiles = [{"type": "minecraft:ender_dragon", "distance": 3}]
    assert game.survive(100) == "end_combat"
    assert calls == []
    game.hostiles.append({"type": "minecraft:creeper", "distance": 3})
    assert game.survive(100) == "flee"
    assert calls == ["hold", "minecraft:creeper"]
    game.hostiles.pop()
    game.health = 3
    assert game.survive(101) == "flee"
    assert calls[-1] == "minecraft:ender_dragon"


def test_quest_poll_is_bounded_read_only_and_disconnect_invalidates(monkeypatch):
    game = GameBridge.__new__(GameBridge)
    game._last_quest_sync = 0
    received, calls = [], []
    game.quests = SimpleNamespace(ingest_server_progress=received.append)
    now = [10.0]
    monkeypatch.setattr("src.levels.l8_agency.game.time.monotonic", lambda: now[0])
    game._rpc = lambda method, params, timeout: calls.append((method, params, timeout)) or {"available": False}
    game.refresh_quest_progress()
    game.refresh_quest_progress()
    assert calls == [("quests.progress", {"offset": 0, "limit": 256}, 2)]
    game.refresh_quest_progress(False)
    assert received[-1] is None
    now[0] = 16
    game.refresh_quest_progress()
    assert len(calls) == 2
