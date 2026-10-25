"""Her tools and her tasks: native tool calling with several steps, the tolerant text protocol,
and promises that become work the heartbeat actually carries out."""
import asyncio
import os
import random
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, StubSearch, START
from src.levels.l8_agency.tasks import TaskBook
from src.levels.l8_agency.toolbox import ToolContext, fetch_page, parse_due, parse_text_actions
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None, "is_night": False}

def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_agent_"), "a.sqlite")

class NativeLLM(ScriptedLLM):
    """A model with function calling: plays back a script of turns (tool calls or a final answer)."""
    supports_tools = True

    def __init__(self, turns, reply="嗯"):
        super().__init__(reply)
        self.turns = list(turns)
        self.seen = []          # (messages, tools) of every tool-loop call

    @staticmethod
    def chatml_to_messages(prompt):
        return [{"role": "system", "content": prompt}, {"role": "user", "content": "…"}]

    def chat_tools(self, messages, tools, max_tokens=300, temperature=0.7):
        self.seen.append(([dict(m) for m in messages], tools))
        turn = self.turns.pop(0) if self.turns else "Thought: 好\nResponse: 嗯嗯"
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": []}
        calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(turn)]
        return {"content": "", "tool_calls": calls,
                "raw": {"role": "assistant", "content": "", "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": "{}"}}
                    for c in calls]}}

class SequenceLLM(ScriptedLLM):
    """Text-protocol model: answers the reply prompts in order."""
    def __init__(self, replies):
        super().__init__(replies[-1])
        self.replies = list(replies)

    def generate_json(self, prompt, max_tokens=200, temperature=0.7, **kw):
        out = super().generate_json(prompt, max_tokens, temperature)
        if "[要求]" in prompt or "[要做的事]" in prompt:
            return {"reply": self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]}
        return out

def make_gw(llm):
    clock.set_clock(SimClock(START + 2 * 3600))
    gw = GlobalWorkspace(EventBus(), llm, db_path=db_path(), rng=random.Random(0))
    gw.search_tool = StubSearch()
    gw.environment.get_context = lambda: DAY
    return gw

def hear(gw, text, user="10001", nick="小明", group=None):
    return asyncio.run(gw.hear(text, user_id=user, nickname=nick, group_id=group, addressed=True))

class TestTextProtocol(unittest.TestCase):
    def test_parses_the_forms_models_actually_write(self):
        self.assertEqual(parse_text_actions('Action: SEARCH query="北京天气"'), [("web_search", {"query": "北京天气"})])
        self.assertEqual(parse_text_actions("Action：search query=“北京天气”"), [("web_search", {"query": "北京天气"})])
        self.assertEqual(parse_text_actions('Action: web_search {"query": "显卡价格"}'),
                         [("web_search", {"query": "显卡价格"})])
        (name, args), = parse_text_actions('Action: MEMORIZE title="猫" content="猫喜欢鱼" tags="动物"')
        self.assertEqual((name, args["content"], args["about_person"]), ("remember", "猫喜欢鱼", False))
        self.assertEqual(parse_text_actions("Action: NONE"), [])
        self.assertEqual(parse_text_actions("今天天气不错"), [])

    def test_text_loop_runs_several_rounds(self):
        llm = SequenceLLM(['Thought: 查一下\nResponse: Action：SEARCH query="北京天气"',
                           'Thought: 再看看\nResponse: Action: recall {"query": "北京"}',
                           "Thought: 好\nResponse: 北京今天晴"])
        gw = make_gw(llm)
        reply = hear(gw, "北京天气怎么样")
        self.assertIn("北京今天晴", reply)
        used = [d for d in gw.decisions if d["kind"] == "tools"]
        self.assertTrue(used and "web_search" in used[-1]["detail"] and "recall" in used[-1]["detail"])
        self.assertTrue(any("[工具结果]" in c["prompt"] and "北京天气" in c["prompt"] for c in llm.calls))

class TestNativeLoop(unittest.TestCase):
    def test_search_then_answer(self):
        llm = NativeLLM([[("web_search", {"query": "新游戏"})], "Thought: 查到了\nResponse: 叫《星海》"])
        gw = make_gw(llm)
        reply = hear(gw, "那个新出的游戏叫什么")
        self.assertIn("星海", reply)
        messages, tools = llm.seen[1]
        self.assertTrue(any(m["role"] == "tool" and "新游戏" in m["content"] for m in messages))
        self.assertIn("web_search", [t["function"]["name"] for t in llm.seen[0][1]])
        self.assertEqual(gw.tool_trace[-1]["name"], "web_search")
        self.assertIn("[工具]", llm.seen[0][0][0]["content"])

    def test_steps_are_bounded(self):
        llm = NativeLLM([[("recall", {"query": "x"})]] * 10 + ["Thought: 算了\nResponse: 不记得了"])
        gw = make_gw(llm)
        gw.agent_max_steps = 3
        hear(gw, "你还记得吗")
        self.assertLessEqual(len(llm.seen), 4)
        self.assertIsNone(llm.seen[-1][1])            # the last step gets no tools: she has to answer

    def test_tools_depend_on_where_she_is(self):
        gw = make_gw(NativeLLM([]))
        names = lambda ctx: {t["function"]["name"] for t in gw.toolbox.specs(ctx)}
        private = names(ToolContext("10001"))
        self.assertNotIn("group_history", private)
        self.assertNotIn("evolve", private)
        self.assertNotIn("schedule_task", names(ToolContext("10001", origin="task")))

        class FakeQQ:
            def connected(self):
                return True
        gw.qq = FakeQQ()
        self.assertIn("group_history", names(ToolContext("10001", group_id="555")))

    def test_remember_is_about_the_person_and_private(self):
        llm = NativeLLM([[("remember", {"content": "小明下周要考试"})], "Thought: 记下了\nResponse: 加油"])
        gw = make_gw(llm)
        hear(gw, "我下周要考试")
        facts = gw.semantic.for_user("10001")
        self.assertTrue(any("考试" in f["fact"] for f in facts))
        self.assertFalse(any("考试" in f["fact"] for f in gw.semantic.public_facts()))

class TestTasks(unittest.TestCase):
    def test_schedule_then_heartbeat_does_it(self):
        llm = NativeLLM([
            [("schedule_task", {"kind": "do", "what": "查一下明天北京的天气告诉小明", "minutes": 10})],
            "Thought: 记下了\nResponse: 好，等我查查告诉你",
            [("web_search", {"query": "明天北京天气"})],
            "Thought: 查到了\nResponse: 小明，明天北京晴",
        ])
        gw = make_gw(llm)
        hear(gw, "帮我查下明天北京天气，不急")
        pending = gw.tasks.pending("10001")
        self.assertEqual(len(pending), 1)
        self.assertTrue(pending[0]["goal_id"])
        self.assertEqual(len([t for t in gw.tasks.pending()]), 1)   # no second task from the promise detector

        asyncio.run(gw.tick())
        self.assertEqual(gw.tasks.get(pending[0]["id"])["status"], "pending")   # not due yet
        clock.get_clock().advance(11 * 60)
        asyncio.run(gw.tick())
        t = gw.tasks.get(pending[0]["id"])
        self.assertEqual(t["status"], "done")
        sent = [m for m in gw.output_queue if m.get("type") == "task"]
        self.assertTrue(sent and "明天北京晴" in sent[-1]["content"])
        self.assertEqual(sent[-1]["target"], {"type": "private", "id": "10001"})
        self.assertEqual(gw.goals.get(t["goal_id"])["status"], "done")

    def test_group_task_answers_in_the_group(self):
        llm = NativeLLM([[("schedule_task", {"kind": "remind", "what": "该去吃饭了", "minutes": 30})],
                         "Thought: 好\nResponse: 好的到时候叫你"])
        gw = make_gw(llm)
        hear(gw, "半小时后提醒我吃饭", group="555")
        clock.get_clock().advance(31 * 60)
        asyncio.run(gw.tick())
        sent = [m for m in gw.output_queue if m.get("type") == "task"]
        self.assertEqual(sent[-1]["target"], {"type": "group", "id": "555"})
        self.assertTrue(sent[-1]["in_reply"])

    def test_reminders_ring_while_she_sleeps_but_work_waits(self):
        gw = make_gw(ScriptedLLM("嗯"))
        now = clock.now()
        r = gw.tasks.add("remind", "喝水", now + 60, now, "10001", "小明")
        d = gw.tasks.add("do", "查点东西", now + 60, now, "10001", "小明")
        gw._set_sleeping(True, source="test")
        clock.get_clock().advance(120)
        asyncio.run(gw._run_due_tasks(clock.now()))
        self.assertEqual(gw.tasks.get(r)["status"], "done")
        self.assertEqual(gw.tasks.get(d)["status"], "pending")

    def test_a_promise_to_look_something_up_becomes_a_task(self):
        llm = SequenceLLM(["Thought: 不知道\nResponse: 我也不清楚，等我查查告诉你", "Thought: 查完了\nResponse: 查到了，是周五"])
        gw = make_gw(llm)
        hear(gw, "演唱会是哪天")
        tasks = gw.tasks.pending("10001")
        self.assertEqual(len(tasks), 1)
        self.assertEqual((tasks[0]["kind"], tasks[0]["origin"]), ("do", "promise"))
        clock.get_clock().advance(6 * 60)
        asyncio.run(gw._run_due_tasks(clock.now()))
        self.assertEqual(gw.tasks.get(tasks[0]["id"])["status"], "done")

    def test_console_requests_land_in_the_console_inbox(self):
        gw = make_gw(ScriptedLLM("嗯"))
        now = clock.now()
        gw.tasks.add("remind", "喝水", now, now, "console", "控制台")
        asyncio.run(gw._run_due_tasks(clock.now()))
        self.assertEqual(len(gw.console_inbox), 1)
        self.assertFalse(gw.output_queue)

    def test_limits(self):
        book = TaskBook(db_path())
        for i in range(5):
            book.add("remind", f"事 {i}", 100, 0, "u")
        with self.assertRaises(ValueError):
            book.add("remind", "第六件", 100, 0, "u")
        with self.assertRaises(ValueError):
            book.add("remind", "太远", 40 * 86400, 0, "v")
        tid = book.add("remind", "v 的事", 100, 0, "v")
        self.assertIsNone(book.cancel(tid, "u", False, 1))
        self.assertEqual(book.cancel(tid, "v", False, 1)["status"], "cancelled")

    def test_failed_tasks_retry_then_give_up(self):
        gw = make_gw(SequenceLLM(["（不回）"]))
        now = clock.now()
        tid = gw.tasks.add("do", "查点东西", now, now, "10001", "小明")
        for _ in range(3):
            asyncio.run(gw._run_due_tasks(clock.now()))
            clock.get_clock().advance(11 * 60)
        self.assertEqual(gw.tasks.get(tid)["status"], "failed")

    def test_due_parsing(self):
        base = START + 2 * 3600
        self.assertEqual(parse_due({"minutes": 10}, base), base + 600)
        self.assertGreater(parse_due({"at": "08:00"}, base), base)
        with self.assertRaises(ValueError):
            parse_due({"at": "明天早上"}, base)

class TestWebPages(unittest.TestCase):
    def test_never_opens_local_or_private_addresses(self):
        for url in ("http://127.0.0.1:8080/console", "http://localhost/", "http://192.168.1.1/", "file:///C:/x",
                    "http://169.254.169.254/latest/meta-data"):
            with self.assertRaises(ValueError):
                fetch_page(url)

    def test_read_webpage_tool_uses_the_page_text(self):
        llm = NativeLLM([[("read_webpage", {"url": "https://example.com/a"})], "Thought: 看完了\nResponse: 写的是猫"])
        gw = make_gw(llm)
        gw.page_fetcher = lambda url: "《猫》\n猫是一种动物" if url == "https://example.com/a" else ""
        hear(gw, "帮我看看 https://example.com/a 写了啥")
        self.assertTrue(any(m["role"] == "tool" and "猫是一种动物" in m["content"] for m in llm.seen[1][0]))

if __name__ == "__main__":
    unittest.main()
