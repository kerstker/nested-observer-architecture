import unittest
import asyncio
import os
import tempfile
from src.core.event_bus import EventBus, EventType
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace
from src.core.event_bus import Event


def _tmp_db():
    """Every workspace in these tests must own a throwaway database: the default db_path
    would write the test's sleep events and snapshots into the live memory.sqlite."""
    return os.path.join(tempfile.mkdtemp(prefix="noa_link_"), "l.sqlite")

class DummyLLM:
    def generate_json(self, prompt):
        return {"reply": "嗯……\n刚才在睡觉，你说什么？"}

class TestStateLinkage(unittest.TestCase):
    def test_sleep_affects_output(self):
        bus = EventBus()
        gw = GlobalWorkspace(bus, DummyLLM(), db_path=_tmp_db())
        # Force sleep on: only the first few words get out
        gw._set_sleeping(True, source="test")
        resp = asyncio.run(gw.process_input("你好", user_id="u1", nickname="ker", group_id=None))
        self.assertEqual(resp, "嗯……")

    def test_no_stage_directions(self):
        class Theatrical:
            def generate_json(self, prompt):
                return {"reply": "Thought: 有人来了\nResponse: *耳朵竖起来* 在呢 *甩甩尾巴*，怎么啦？"}
        gw = GlobalWorkspace(EventBus(), Theatrical(), db_path=_tmp_db())
        resp = asyncio.run(gw.process_input("在吗", user_id="u1", nickname="ker", group_id=None))
        self.assertEqual(resp, "在呢 ，怎么啦？".replace(" ，", "，"))
    
    def test_proactive_suppressed_when_sleeping(self):
        bus = EventBus()
        gw = GlobalWorkspace(bus, DummyLLM(), db_path=_tmp_db())
        gw._set_sleeping(True, source="test")
        # Try triggering proactive; should do nothing
        gw._trigger_proactive_behavior()
        self.assertEqual(len(gw.output_queue), 0)
    
    def test_state_event_recording(self):
        bus = EventBus()
        gw = GlobalWorkspace(bus, DummyLLM(), db_path=_tmp_db())
        gw._set_sleeping(True, source="test")
        events = gw.memory.db.get_state_events(state="sleep", limit=1)
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[0]["state"], "sleep")
    
    def test_callbacks_invoked(self):
        bus = EventBus()
        gw = GlobalWorkspace(bus, DummyLLM(), db_path=_tmp_db())
        flags = {"called": False}
        def cb(payload):
            flags["called"] = True
        gw.register_state_callback("sleep", cb)
        gw._set_sleeping(True, source="test")
        self.assertTrue(flags["called"])

if __name__ == "__main__":
    unittest.main()
