import asyncio
import pytest
from src.levels.l10_consciousness.tool_protocol import recover_calls, spoken_text
from src.levels.l8_agency.toolbox import ToolContext
from test_agent_tools import NativeLLM, make_gw

def envelope(name="recall", value="洞穴"):
    return (f'<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="{name}">'
            f'<｜｜DSML｜｜ parameter name="query" string="true">{value}</｜｜DSML｜｜ parameter>'
            '</｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>')

SPEC = [{"type": "function", "function": {"name": "recall", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

def test_actual_dsml_is_executed_once_and_followup_has_valid_call_pair():
    llm = NativeLLM([envelope(), "Response: 查完了"])
    gw = make_gw(llm)
    result = asyncio.run(gw._respond_native("test", "", {}, None, ToolContext("10001")))
    assert result == "查完了"
    assert [t["name"] for t in gw.tool_trace] == ["recall"]
    messages = llm.seen[1][0]
    call = next(m for m in messages if m.get("tool_calls"))["tool_calls"][0]
    assert any(m.get("tool_call_id") == call["id"] for m in messages)

@pytest.mark.parametrize("text", [envelope()[:-12], "Response: 示例 " + envelope(), "```xml\n" + envelope() + "\n```",
                                      envelope("run_python"), envelope(value="<bad/>"), '<!DOCTYPE calls>' + envelope()])
def test_incomplete_quoted_unavailable_or_nested_payload_is_never_executed_or_spoken(text):
    assert recover_calls(text, SPEC) == []
    assert spoken_text(text) == ""

def test_live_viewer_cannot_recover_a_tool_and_last_step_never_executes():
    llm = NativeLLM([envelope()] * 4)
    gw = make_gw(llm)
    gw.agent_max_steps = 1
    result = asyncio.run(gw._respond_native("test", "", {}, None, ToolContext("live:123", group_id="live")))
    assert result == ""
    assert not getattr(gw, "tool_trace", [])
    assert len(llm.seen) == 2

def test_schema_rejects_wrong_numeric_argument_and_duplicate_parameter():
    text = envelope().replace('string="true">洞穴', 'string="false">12')
    assert recover_calls(text, SPEC) == []
    parameter = '<｜｜DSML｜｜ parameter name="query" string="true">x</｜｜DSML｜｜ parameter>'
    assert recover_calls(envelope().replace('</｜｜DSML｜｜ invoke>', parameter + '</｜｜DSML｜｜ invoke>'), SPEC) == []

@pytest.mark.parametrize("prefix", ["Thought: 我先看地图\n", "Response: 看看之前的洞穴。\n", "Thought: 找路线\nResponse: 我看看\n"])
def test_complete_calls_after_legacy_reply_wrappers_are_recovered(prefix):
    assert recover_calls(prefix + envelope(), SPEC)[0]["arguments"] == {"query": "洞穴"}
    assert spoken_text(prefix + envelope()) == ""
