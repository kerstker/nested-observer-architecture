"""Reading subtext: a cheap pre-screen decides when to look closer, the reading changes how she answers."""
import asyncio
import os
import tempfile
import unittest

from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM
from src.levels.l7_tom.subtext import SubtextReader
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace


class TestPrescreen(unittest.TestCase):
    def setUp(self):
        self.r = SubtextReader()

    def test_ordinary_messages_cost_nothing_extra(self):
        for text in ("今天吃什么好呢", "帮我查一下天气", "哈哈哈", "晚安", "好的", "（引用了 小明「我好累」）这个是什么游戏"):
            self.assertEqual(self.r.risky(text), [], text)

    def test_risky_messages_are_flagged(self):
        for text in ("我今天好难过", "没事，算了", "你怎么这么冲啊", "你是不是其实就是个AI", "你们真的不爱我了吗",
                     "好好好你说得对", "呵呵"):
            self.assertTrue(self.r.risky(text), text)


class TestReading(unittest.TestCase):
    def test_reading_is_rendered_and_mapped_to_her_coarse_emotion(self):
        r = SubtextReader()
        reading = r.read(lambda *a, **k: {"emotion": "难过", "real_meaning": "说没事其实有事", "approach": "先陪着别开玩笑",
                                          "avoid": "反问"}, "小明", "没事", "", ["淡化"])
        self.assertEqual(r.tom_emotion(reading), "sad")
        text = r.render(reading)
        self.assertIn("先陪着别开玩笑", text)
        self.assertIn("反问", text)

    def test_a_failed_reading_is_just_none(self):
        def boom(*a, **k):
            raise RuntimeError("x")
        self.assertIsNone(SubtextReader().read(boom, "小明", "没事"))
        self.assertIsNone(SubtextReader().read(lambda *a, **k: {"reply": "嗯"}, "小明", "没事"))


class TestInThePrompt(unittest.TestCase):
    def _gw(self, llm):
        return GlobalWorkspace(EventBus(), llm, db_path=os.path.join(tempfile.mkdtemp(prefix="noa_sub_"), "m.sqlite"))

    def test_a_risky_message_gets_read_and_the_reading_reaches_the_prompt(self):
        class Llm(ScriptedLLM):
            def generate_json(self, prompt, max_tokens=200, temperature=0.7):
                if "看懂对方这句话到底是什么意思" in prompt:
                    self.calls.append({"prompt": prompt})
                    return {"emotion": "难过", "real_meaning": "其实很难受", "approach": "先陪着", "avoid": "开玩笑"}
                return super().generate_json(prompt, max_tokens, temperature)
        llm = Llm("嗯，我在")
        gw = self._gw(llm)
        asyncio.run(gw.hear("没事，算了", user_id="u1", nickname="小明", addressed=True))
        prompt = next(c["prompt"] for c in reversed(llm.calls) if "[要求]" in c["prompt"])
        self.assertIn("[我对 ta 这句话的理解]", prompt)
        self.assertIn("先陪着", prompt)
        self.assertEqual(gw.last_subtext["emotion"], "难过")

    def test_an_ordinary_message_makes_no_extra_call(self):
        llm = ScriptedLLM("好呀")
        gw = self._gw(llm)
        asyncio.run(gw.hear("今天吃什么好呢", user_id="u1", nickname="小明", addressed=True))
        self.assertFalse(any("看懂对方这句话到底是什么意思" in c["prompt"] for c in llm.calls))
        self.assertIsNone(gw.last_subtext)


if __name__ == "__main__":
    unittest.main()
