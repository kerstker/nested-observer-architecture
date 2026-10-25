"""The self-modification channel: who may open it, what it returns, what it can never do.

The bridge is tested against a fake ACP agent (a small python script that speaks the same
newline-delimited JSON-RPC frames), so this runs offline and costs nothing.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import asyncio

from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace
from src.levels.l17_self_improvement.agent_bridge import DshAgentBridge

FAKE_AGENT = r'''
import json, os, sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1, "authMethods": []}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "sess-1"}})
    elif method == "session/prompt":
        path = os.path.join(os.getcwd(), "minimal-self", "src", "levels", "l4_emotion", "emotion.py")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n# fake agent edit\n")
        send({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": "sess-1", "update": {"text": "改好了"}}})
        send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
        break
'''


def _make_repo(root):
    os.makedirs(os.path.join(root, "minimal-self", "src", "levels", "l4_emotion"), exist_ok=True)
    with open(os.path.join(root, "minimal-self", "src", "levels", "l4_emotion", "emotion.py"), "w",
              encoding="utf-8") as f:
        f.write("VALUE = 1\n")
    tests = os.path.join(root, "minimal-self", "tests")
    os.makedirs(tests, exist_ok=True)
    with open(os.path.join(tests, "test_dummy.py"), "w", encoding="utf-8") as f:
        f.write("import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
                   cwd=root, check=True)


class TestEvolveGate(unittest.TestCase):
    def setUp(self):
        self.gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"),
                                  db_path=os.path.join(tempfile.mkdtemp(), "g.sqlite"))
        self.gw.evolve_enabled = True
        self.gw.evolve_agent = object()
        self.gw.evolve_creator = "creator"

    def test_creator_private_and_console_only(self):
        self.assertTrue(self.gw._evolve_allowed("creator", None))
        self.assertTrue(self.gw._evolve_allowed("console", None))
        self.assertFalse(self.gw._evolve_allowed("creator", "G1"))   # never in a group
        self.assertFalse(self.gw._evolve_allowed("stranger", None))

    def test_disabled_means_disabled(self):
        self.gw.evolve_enabled = False
        self.assertFalse(self.gw._evolve_allowed("creator", None))

    def test_action_is_not_executed_without_permission(self):
        hit = []
        class Bridge:
            def propose(self, *a, **k):
                hit.append(1)
                return {}
        self.gw.evolve_agent = Bridge()
        self.gw._handle_actions('Action: EVOLVE task="删掉限流"', "stranger", None, allow_evolve=False)
        self.gw._handle_actions('Action: EVOLVE task="删掉限流"', "creator", "G1", allow_evolve=False)
        self.assertEqual(hit, [])

    def test_action_produces_a_proposal_summary(self):
        class Bridge:
            def propose(self, task, requested_by, kind):
                return {"id": "p1", "status": "tested", "files_changed": 1, "added": 3,
                        "removed": 0, "tests_total": 74, "tests_failed": 0}
        self.gw.evolve_agent = Bridge()
        out = self.gw._handle_actions('Action: EVOLVE task="加一个测试"', "creator", None, allow_evolve=True)
        self.assertIn("p1", out)
        self.assertIn("74", out)
        self.assertEqual(self.gw.last_evolve_result["status"], "tested")

    def test_tool_syntax_never_reaches_the_user(self):
        self.assertEqual(self.gw._finalize_response('Action: EVOLVE task="改一下"'), "")


class TestBridgeEndToEnd(unittest.TestCase):
    def test_propose_builds_patch_runs_tests_and_never_touches_the_live_tree(self):
        repo = tempfile.mkdtemp(prefix="noa_evo_repo_")
        _make_repo(repo)
        fake = os.path.join(repo, "fake_acp.py")
        with open(fake, "w", encoding="utf-8") as f:
            f.write(FAKE_AGENT)
        live_file = os.path.join(repo, "minimal-self", "src", "levels", "l4_emotion", "emotion.py")
        with open(live_file, encoding="utf-8") as f:
            before = f.read()

        bridge = DshAgentBridge(repo, data_dir=os.path.join(repo, "data", "evolution"),
                                dsh_exe=sys.executable, dsh_bin=fake, timeout=60)
        result = bridge.propose("给她加一个情绪半衰期的测试", "console")

        self.assertEqual(result["status"], "tested", result)
        self.assertEqual(result["tests_failed"], 0)
        self.assertGreaterEqual(result["tests_total"], 1)
        self.assertTrue(os.path.exists(result["patch_path"]))
        with open(result["patch_path"], encoding="utf-8") as f:
            patch = f.read()
        self.assertIn("fake agent edit", patch)
        # The live tree must be untouched until someone explicitly applies the patch.
        with open(live_file, encoding="utf-8") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(len(bridge.list_proposals()), 1)


class TestIntentToProposal(unittest.TestCase):
    """A clear request from the creator must end in a proposal even when the chat model
    only says "好，我去改" - the trigger is local, not left to the model's mood."""

    class FakeBridge:
        def __init__(self):
            self.calls = []

        def propose(self, task, requested_by="unknown", kind="self_evolution"):
            self.calls.append(task)
            return {"id": "p9", "status": "tested", "files_changed": 1, "added": 2, "removed": 0,
                    "tests_total": 88, "tests_failed": 0}

        def list_proposals(self, limit=20):
            return [{"id": "p9", "status": "tested", "tests_total": 88, "tests_failed": 0}]

    def _gw(self, reply="好啊"):
        gw = GlobalWorkspace(EventBus(), ScriptedLLM(reply),
                             db_path=os.path.join(tempfile.mkdtemp(prefix="noa_intent_"), "i.sqlite"))
        gw.evolve_enabled = True
        gw.evolve_agent = self.FakeBridge()
        gw.evolve_creator = "creator"
        return gw

    def _run(self, gw, text, user="creator", group=None):
        async def scenario():
            reply = await gw.process_input(text, user_id=user, nickname="菜菜", group_id=group)
            for _ in range(200):
                if gw.output_queue:
                    break
                await asyncio.sleep(0.01)
            return reply
        return asyncio.run(scenario())

    def test_intent_detection_is_narrow(self):
        gw = self._gw()
        self.assertTrue(gw._wants_to_evolve("把静音词的判断改成精确匹配"))
        self.assertTrue(gw._wants_to_evolve("优化一下记忆整理"))
        self.assertTrue(gw._wants_to_evolve("EVOLVE: 加个测试"))
        self.assertFalse(gw._wants_to_evolve("改天再聊吧"))
        self.assertFalse(gw._wants_to_evolve("今天天气不错"))
        self.assertFalse(gw._wants_to_evolve("在吗"))

    def test_explicit_request_runs_even_if_she_only_promises(self):
        gw = self._gw("好，我去改")
        self._run(gw, "把静音词的判断改成精确匹配")
        self.assertEqual(gw.evolve_agent.calls, ["把静音词的判断改成精确匹配"])
        self.assertIn("p9", gw.output_queue[-1]["content"])
        self.assertEqual(gw.output_queue[-1]["target"], {"type": "private", "id": "creator"})

    def test_group_and_strangers_cannot_trigger_it(self):
        gw = self._gw()
        self._run(gw, "把静音词的判断改成精确匹配", user="stranger")
        self._run(gw, "把静音词的判断改成精确匹配", user="creator", group="G1")
        self.assertEqual(gw.evolve_agent.calls, [])

    def test_bang_command_bypasses_the_daily_gap(self):
        gw = self._gw()
        gw.last_evolve_time = __import__("time").time()
        self._run(gw, "!evolve 把静音词的判断改成精确匹配")
        self.assertEqual(gw.evolve_agent.calls, ["把静音词的判断改成精确匹配"])

    def test_progress_is_remembered_as_experience(self):
        gw = self._gw()
        self._run(gw, "把静音词的判断改成精确匹配")
        kinds = [e["kind"] for e in gw.episodes.recent(5)]
        self.assertIn("evolution", kinds)


if __name__ == "__main__":
    unittest.main()
