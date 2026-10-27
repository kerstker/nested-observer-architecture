"""Lighting follows confirmed descent and a verified placed block, never attempted movement."""
from unittest.mock import Mock

import pytest

from src.levels.l8_agency.mc_skills import Skills


def descending_skills(move=True):
    skills = Skills(lambda *args: {})
    state = {"x": .5, "y": 64.0, "z": .5, "yaw": 0}
    skills._me = lambda: dict(state)
    skills._blocks = lambda cells: ["minecraft:stone"] * len(cells)
    def step(cells, stand):
        if move:
            state.update(x=stand[0] + .5, y=float(stand[1]), z=stand[2] + .5)
        return None
    skills._dig_step = step
    return skills, state


def test_stair_lights_each_six_confirmed_descents_on_the_previous_stair():
    skills, state = descending_skills()
    skills._torch_behind = Mock(side_effect=lambda x, y, z: (x, y, z))
    result = skills.dig_down(51, max_steps=13)
    assert state["y"] == 51
    assert [c.args for c in skills._torch_behind.call_args_list] == [(0, 59, 5), (0, 53, 11)]
    assert "已确认放置火把 2 支" in result


def test_no_descent_does_not_light_or_count_attempted_steps():
    skills, state = descending_skills(move=False)
    skills._torch_behind = Mock()
    result = skills.dig_down(40)
    assert "仅确认 0 级台阶" in result
    skills._torch_behind.assert_not_called()


def test_failed_placement_is_reported_as_unconfirmed_light():
    skills, _ = descending_skills()
    skills._torch_behind = Mock(return_value=None)
    result = skills.dig_down(58)
    assert "照明未确认" in result and "已确认放置火把" not in result


@pytest.mark.parametrize("target,support,loaded", [
    ("minecraft:air", "minecraft:air", True),
    ("minecraft:air", "minecraft:lava", True),
    ("minecraft:water", "minecraft:stone", True),
    ("minecraft:air", "minecraft:magma_block", True),
    ("minecraft:air", "minecraft:stone", False),
])
def test_torch_requires_loaded_empty_cell_and_safe_solid_floor(target, support, loaded):
    skills = Skills(lambda method, args: {"blocks": [{"id": target, "loaded": loaded},
                                                    {"id": support, "loaded": loaded}]})
    skills.crafter.counts = lambda: {"minecraft:torch": 16}
    skills.place_at = Mock()
    assert skills._torch_behind(0, 59, 5) is None
    skills.place_at.assert_not_called()


def test_torch_without_inventory_does_not_swap_drop_or_place_anything():
    rpc = Mock()
    skills = Skills(rpc)
    skills.crafter.counts = lambda: {"minecraft:diamond_pickaxe": 1}
    assert skills._torch_behind(0, 59, 5) is None
    rpc.assert_not_called()


def test_torch_result_depends_on_actual_place_at_confirmation():
    skills = Skills(lambda method, args: {"blocks": [{"id": "minecraft:air", "loaded": True},
                                                    {"id": "minecraft:stone", "loaded": True}]})
    skills.crafter.counts = lambda: {"minecraft:torch": 1}
    skills.place_at = Mock(return_value=None)
    assert skills._torch_behind(0, 59, 5) is None
    skills.place_at.assert_called_once_with("minecraft:torch", 0, 59, 5, walk=False)
    skills.place_at.return_value = (0, 59, 5)
    assert skills._torch_behind(0, 59, 5) == (0, 59, 5)
