"""Endgame regressions: real observations, bounded survival actions and no invented completion."""
from unittest.mock import Mock, patch

from src.levels.l8_agency.mc_endgame import Endgame
from src.levels.l8_agency.mc_skills import Skills
from src.levels.l8_agency.mc_body import PlayerBody
from test_mc_body_loop import FakeWorld

M = "minecraft:"


def setup():
    world = FakeWorld()
    skills = Skills(world.rpc)
    skills.crafter.counts = lambda: {M + "ender_eye": 16, M + "bow": 1, M + "arrow": 128,
                                   M + "water_bucket": 1, M + "cobblestone": 64}
    skills.hold = lambda item: True
    skills.free_hands = lambda: None
    skills._smooth_look = Mock()
    return world, skills, Endgame(skills)


def test_eye_navigation_requires_an_observed_new_eye_and_requests_other_entities():
    world, s, end = setup()
    s.go_to = Mock()
    with patch("src.levels.l8_agency.mc_endgame.time.monotonic", side_effect=[0, 1, 2, 4]), \
         patch("src.levels.l8_agency.mc_endgame.time.sleep"):
        result = end.follow_eye()
    assert "没有观测" in result
    s.go_to.assert_not_called()
    assert any(m == "perception.entities" and "other" in p["kinds"] for m, p, _ in world.calls)
    assert not any("locate" in m.lower() or "teleport" in m.lower() for m, *_ in world.calls)


def test_eye_flight_drives_direction_instead_of_a_guessed_stronghold_coordinate():
    _, s, end = setup()
    eye = {"uuid": "eye-1", "type": M + "eye_of_ender", "x": 12.5, "y": 72, "z": .5, "distance": 15}
    end.entities = Mock(side_effect=[[], [eye], [eye]])
    s.go_to = Mock(return_value="正在走")
    with patch("src.levels.l8_agency.mc_endgame.time.monotonic", side_effect=[0, 1, 2, 4]), \
         patch("src.levels.l8_agency.mc_endgame.time.sleep"):
        end.follow_eye()
    assert s.go_to.call_args.args == (96, 64, 0)
    assert s._eye_route["dx"] == 1
    assert s.places.near({"x": .5, "y": 64, "z": .5}, ("eye_bearing",), within=32)


def test_observed_frames_are_not_a_teleport_or_a_claim_of_an_active_portal():
    _, s, end = setup()
    s._approach = lambda *a: True
    end.scan = lambda *a, **kw: []
    with patch("src.levels.l8_agency.mc_endgame.time.sleep"):
        result = end.fill_frames([{"x": 1, "y": 63, "z": 0}])
    assert "未观测到激活" in result
    assert not any(p["kind"] == "end_portal" for p in s.places.places)


def test_activated_portal_is_recorded_only_after_block_observation():
    _, s, end = setup()
    s._approach = lambda *a: True
    end.scan = lambda *a, **kw: [{"id": M + "end_portal", "x": 2, "y": 63, "z": 2}]
    with patch("src.levels.l8_agency.mc_endgame.time.sleep"):
        result = end.fill_frames([{"x": 1, "y": 63, "z": 0}])
    assert "实际点亮" in result
    assert any(p["kind"] == "end_portal" for p in s.places.places)


def test_entering_end_without_ammunition_never_walks_into_portal():
    _, s, end = setup()
    s.crafter.counts = lambda: {M + "bow": 1}
    s._walk_to = Mock()
    assert "至少64" in end.enter({"x": 0, "y": 63, "z": 0}, M + "the_end")
    s._walk_to.assert_not_called()


def test_dragon_breath_is_handled_before_bow_firing():
    world, s, end = setup()
    world.me["dimension"] = M + "the_end"
    end.entities = lambda *a: [{"type": M + "area_effect_cloud", "x": 1, "y": 64, "z": 1, "distance": 2}]
    end._retreat = Mock(return_value=True)
    assert "避开" in end.fight()
    assert not any(m == "control.startUsing" for m, *_ in world.calls)


def test_unseen_dragon_never_becomes_a_victory_without_exit_evidence():
    world, s, end = setup()
    world.me["dimension"] = M + "the_end"
    end.entities = lambda *a: []
    end.scan = lambda *a, **kw: []
    end.arena_route = lambda: "寻找安全路线"
    assert end.fight() == "寻找安全路线"
    end.scan = lambda *a, **kw: [{"id": M + "end_portal", "x": 0, "y": 63, "z": 0}]
    result = end.fight()
    assert "现场证据" in result and "不代表" in result
    assert s.places.places[-1]["dim"] == M + "the_end"


def test_shot_requires_visible_target_and_reports_observed_health():
    world, s, end = setup()
    world.me["dimension"] = M + "the_end"
    dragon = {"uuid": "d", "type": M + "ender_dragon", "x": 30, "y": 74, "z": 0, "health": 200}
    end.entities = Mock(side_effect=[[dragon], [{**dragon, "health": 190}]])
    end.scan = lambda *a, **kw: []
    base = s.rpc
    s.rpc = lambda method, params: {"entities": [{"uuid": "d"}]} if method == "vision.describeScene" else base(method, params)
    s.body = Mock()
    with patch("src.levels.l8_agency.mc_endgame.time.sleep"):
        result = end.fight()
    s.body.use_item.assert_called_once_with(seconds=1.1)
    assert "200" in result and "190" in result


def test_dimension_change_aborts_old_world_walk_immediately():
    world = FakeWorld()
    base = world.rpc
    reads = 0

    def rpc(method, params):
        nonlocal reads
        if method == "player.getState":
            reads += 1
            if reads >= 4:
                world.me["dimension"] = M + "the_end"
                world.me["x"] = 100
        return base(method, params)

    body = PlayerBody(rpc, now=world.clock.now, sleep=world.clock.sleep)
    try:
        assert not body.walk_to(.5, 64, 20, timeout=10)
        assert not any(world.inputs.values())
    finally:
        body.stop()
