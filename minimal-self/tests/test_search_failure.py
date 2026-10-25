import logging
import os
import tempfile
import unittest

from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace
from src.levels.l19_search.web_search import WebSearchTool

def offline_search_tool() -> WebSearchTool:
    tool = WebSearchTool()
    tool._ddgs_search = lambda q, n: []
    tool._baidu_search = lambda q, n: []
    tool._ddg_search = lambda q, n: []
    return tool

def make_gw() -> GlobalWorkspace:
    db = os.path.join(tempfile.mkdtemp(prefix="noa_search_"), "s.sqlite")
    gw = GlobalWorkspace(EventBus(), ScriptedLLM("北京天气"), db_path=db)
    gw.search_tool = offline_search_tool()
    return gw

class TestSearchFailure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_no_fabricated_results(self):
        tool = offline_search_tool()
        self.assertIsNone(tool.try_search("北京天气"))
        text = tool.search("北京天气")
        self.assertIn("没有搜到", text)
        self.assertNotIn("°C", text)
        self.assertFalse(hasattr(tool, "_mock_search"))

    def test_failed_curiosity_learns_nothing(self):
        gw = make_gw()
        gw.homeostasis.curiosity_need = 0.9
        q_before = gw.drives.q("curiosity", "search")
        gw._trigger_curiosity_behavior()
        self.assertEqual(gw.knowledge.recall("北京天气"), "")             # nothing memorised
        self.assertEqual(gw.homeostasis.curiosity_need, 0.9)              # boredom not relieved
        self.assertLess(gw.drives.q("curiosity", "search"), q_before)     # searching reinforced down
        self.assertEqual(gw.output_queue, [])                             # nothing to "share"
        kinds = [e["kind"] for e in gw.episodes.recent(5)]
        self.assertIn("search_failed", kinds)
        self.assertNotIn("search", kinds)

    def test_successful_share_goes_to_a_private_chat(self):
        from src.evaluation.scenarios import StubSearch
        gw = make_gw()
        gw.search_tool = StubSearch()
        gw._trigger_curiosity_behavior()
        self.assertNotIn("target", gw.output_queue.pop())     # nobody known yet: no target (app drops it)
        gw.user_models.get("u1")["private"] = True
        gw.user_models.get("g_member")                         # only seen in groups
        gw._trigger_curiosity_behavior()
        self.assertEqual(gw.output_queue.pop()["target"], {"type": "private", "id": "u1"})

    def test_conversation_is_told_search_failed(self):
        import asyncio
        gw = make_gw()
        asyncio.run(gw.process_input("今天北京天气怎么样？", user_id="u1", nickname="小明"))
        prompt = gw.inference_engine.last_prompt()
        self.assertIn("搜索失败了", prompt)
        self.assertNotIn("Real-time Search Results", prompt)

if __name__ == "__main__":
    unittest.main()
