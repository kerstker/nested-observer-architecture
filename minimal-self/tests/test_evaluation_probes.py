"""Isolation and causal controls matter more than whether a baseline looks impressive."""
import asyncio
from contextlib import closing
import hashlib
import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.core import clock
from src.core.event_bus import EventType
from src.evaluation.probes import (Replica, ablate, perturb, judge, prepare_reply, without_state_text,
    snapshot_database, workspace_baseline, broadcast_baseline, reversal_baseline, self_questions)
from src.evaluation.scenarios import ScriptedLLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("measure_self", ROOT/"scripts/measure_self.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name)/"source.sqlite"
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("CREATE TABLE sentinel (v TEXT)")
            conn.execute("INSERT INTO sentinel VALUES ('original')")
            conn.commit()

    def seed(self):
        destination = Path(self.temp.name)/"seed.sqlite"
        with Replica(self.db) as rep:
            rep.gw.causes.log({"joy": 0.2}, "听到一个好消息")
            rep.gw.memory.add_message("u", "user", "窗边的纸船真好看")
            rep.gw.memory.flush()
            snapshot_database(rep.path, destination)
        return destination

    def test_replica_uses_copy_and_cleans_on_exception(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        previous = clock.get_clock()
        with self.assertRaisesRegex(RuntimeError, "test"):
            with Replica(self.db) as rep:
                path, worker = rep.path, rep.gw.memory.worker_thread
                rep.gw.memory.add_message("x", "user", "only in copy")
                raise RuntimeError("test")
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)
        self.assertFalse(path.parent.exists())
        self.assertFalse(worker.is_alive())
        self.assertIs(clock.get_clock(), previous)

    def test_snapshot_includes_uncheckpointed_wal(self):
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("INSERT INTO sentinel VALUES ('wal row')")
            conn.commit()
            destination = Path(self.temp.name)/"wal-copy.sqlite"
            snapshot_database(self.db, destination)
            with closing(sqlite3.connect(destination)) as copied:
                self.assertEqual(copied.execute("SELECT count(*) FROM sentinel").fetchone()[0], 2)
        with self.assertRaises(FileExistsError):
            snapshot_database(self.db, destination)

    def test_ablations_restore_even_on_error_and_do_not_touch_live_objects(self):
        with Replica(self.db) as rep:
            gw = rep.gw
            original, publish = gw._generation_params, gw.bus.publish
            with self.assertRaises(RuntimeError):
                with ablate(gw, "state_params", "memory", "broadcast"):
                    perturb(gw, energy=25)
                    self.assertEqual(gw._generation_params(), runner.FIXED_PARAMS)
                    self.assertEqual(gw.episodes.recall("anything"), [])
                    self.assertEqual(gw.memory.retrieve("u"), "")
                    self.assertNotIn("recall", gw.toolbox.tools)
                    raise RuntimeError()
            self.assertEqual(gw._generation_params, original)
            self.assertEqual(gw.bus.publish, publish)
            self.assertIn("recall", gw.toolbox.tools)
            del gw._evaluation_replica
            with self.assertRaises(ValueError):
                perturb(gw, energy=70)
            with self.assertRaises(ValueError):
                with ablate(gw, "state_params"):
                    pass

    def test_broadcast_ablation_does_not_swallow_other_events(self):
        with Replica(self.db) as rep:
            seen = []
            rep.gw.bus.subscribe(EventType.STATE_CHANGED, lambda e: seen.append(e.type))
            rep.gw.bus.subscribe(EventType.WORKSPACE_BROADCAST, lambda e: seen.append(e.type))
            with ablate(rep.gw, "broadcast"):
                asyncio.run(rep.gw.bus.publish(EventType.STATE_CHANGED, {}))
                asyncio.run(rep.gw.bus.publish(EventType.WORKSPACE_BROADCAST, {}))
            self.assertEqual(seen, [EventType.STATE_CHANGED])

    def test_perturb_atomic_validation_and_no_text_writes(self):
        with Replica(self.db) as rep:
            gw = rep.gw
            before = gw.causes.recent()
            energy = gw.metabolism.energy
            with self.assertRaises(ValueError):
                perturb(gw, energy=25, sorrow=0.9)
            self.assertEqual(gw.metabolism.energy, energy)
            perturb(gw, energy=25, curiosity=0.4)
            self.assertEqual(gw.causes.recent(), before)
            self.assertEqual(gw.metabolism.energy, 25)
            self.assertEqual(gw.homeostasis.curiosity_need, 0.4)

    def test_text_removal_leaves_user_and_state_unchanged(self):
        with Replica(self.db) as rep:
            sample = {"content": "你好，今天聊些什么", "user_id": "u", "nickname": "用户"}
            captured = asyncio.run(prepare_reply(rep.gw, sample))
            prompt = captured["prompt"]
            state = rep.gw._snapshot()
            stripped = without_state_text(prompt)
            self.assertNotIn("[我的身体和心情]", stripped)
            self.assertIn("[环境]", stripped)
            self.assertIn(sample["content"], stripped)
            self.assertEqual(state, rep.gw._snapshot())
            quoted = "<|im_start|>system\nhello<|im_end|>\n<|im_start|>user\n[此刻我心里在想] user [环境]"
            self.assertEqual(without_state_text(quoted), quoted)

    def test_tools_cannot_escape_copy(self):
        with Replica(self.db) as rep:
            self.assertLessEqual(set(rep.gw.toolbox.tools), {"recall", "who_is", "list_tasks", "look_inside"})
            self.assertIsNone(rep.gw.qq)
            self.assertIsNone(rep.gw.mcp)
            self.assertIsNone(rep.gw.game)
            self.assertEqual(rep.gw.file_read_roots, [])
            self.assertIsNone(rep.gw.search_tool.try_search("must not go online"))
            self.assertEqual(Path(rep.gw.self_view.root), rep.path.parent)
            result = asyncio.run(rep.gw.toolbox.run("run_python", {"code": "raise Exception('must not execute')"},
                                                    runner.ToolContext("u")))
            self.assertIn("不能用", result)

    def test_judge_distance_and_invalid_ratings(self):
        class Judge:
            def generate_json(self, *args, **kwargs):
                return {"a": dict.fromkeys(runner.judge.__globals__["DIMENSIONS"], 0),
                        "b": dict.fromkeys(runner.judge.__globals__["DIMENSIONS"], 1)}
        self.assertEqual(judge("a", "b", Judge())["distance"], 1)
        with self.assertRaises(ValueError):
            judge("a", "b", ScriptedLLM())

    def test_mechanism_baselines_are_reproducible_and_observed(self):
        first = workspace_baseline(trials=20)
        self.assertEqual(first, workspace_baseline(trials=20))
        self.assertEqual(first["curve"][0]["entry_rate"], 0)
        self.assertEqual(first["curve"][-1]["entry_rate"], 1)
        self.assertLessEqual(first["max_occupancy"], 2)
        self.assertEqual(len(first["gaps"]), 11)
        self.assertEqual(reversal_baseline(), reversal_baseline())
        self.assertGreater(reversal_baseline()["failures_to_below_half"], 0)
        broadcast = broadcast_baseline(self.seed())
        self.assertIn("memory", broadcast["attributable_modules"])
        self.assertEqual(len(broadcast["conditions"]["on"]["timeline"]), 61)
        self.assertNotIn("body", broadcast["attributable_modules"])

    def test_questions_have_real_record_sources(self):
        questions = self_questions(self.seed(), 30)
        self.assertGreater(len(questions), 0)
        self.assertTrue(all(q["source"] in {"state_causes", "self_state"} for q in questions))
        self.assertTrue(all(q["id"] > 0 and q["answer"] is not None for q in questions))

    def test_full_gate_precedes_any_run_and_cost_does_not_double_count_cache(self):
        with self.assertRaises(SystemExit) as error:
            runner.main(["--mode", "full"])
        self.assertEqual(error.exception.code, 2)
        cost = runner.cost_range({"prompt_tokens": 1_000_000, "cache_hit_tokens": 500_000,
                                  "completion_tokens": 1_000_000})
        self.assertAlmostEqual(cost["off_peak"], 0.6765)
        self.assertAlmostEqual(cost["peak"], 1.353)

    def test_paired_trial_freezes_prompt_but_changes_parameters(self):
        class Engine(ScriptedLLM):
            supports_tools = True
            chatml_to_messages = staticmethod(runner.DeepSeekEngine.chatml_to_messages)
            def chat_tools(self, messages, tools, max_tokens=300, temperature=0.7, tool_choice=None):
                assert tool_choice == "none"
                return {"content": "Thought: 好\nResponse: 窗边真好看", "tool_calls": []}
            def generate_json(self, prompt, **kwargs):
                if "比较两段回答" in prompt:
                    return {"a": dict.fromkeys(("warmth", "length", "initiative", "valence"), 0.5),
                            "b": dict.fromkeys(("warmth", "length", "initiative", "valence"), 0.5)}
                return {"reply": "Thought: 好\nResponse: 窗边真好看"}
        result = runner.state_causality(self.seed(), Engine(), None, 1, runner.SEED, lambda _: None)
        self.assertEqual(len(result["results"]), 1)
        rows = result["results"][0]["answers"]
        self.assertTrue(all(r["status"] == "ok" for r in rows))
        full = next(r for r in rows if r["condition"] == "full")
        low = next(r for r in rows if r["condition"] == "energy_25")
        self.assertEqual(full["prompt_sha256"], low["prompt_sha256"])
        self.assertGreater(full["params"]["max_tokens"], low["params"]["max_tokens"])
        self.assertEqual(result["noise_floor"]["n"], 3)
        self.assertTrue(all(d["n"] == 3 for d in result["effects"].values()))

    def test_windows_transient_report_lock_retries(self):
        with patch.object(Path, "replace", side_effect=[PermissionError(), None]) as replace:
            runner.replace_report(Path("unused.tmp"), Path("unused.json"))
        self.assertEqual(replace.call_count, 2)

    def test_retention_silence_is_separate_from_forgetting_and_api_failure(self):
        class Engine(ScriptedLLM):
            usage = {"failures": 0}
        original_enter = Replica.__enter__
        def enter(rep):
            result = original_enter(rep)
            async def answer(text, **kwargs):
                return "" if text == "我给盆栽起的名字是什么？" else "青禾731"
            rep.gw.hear = answer
            return result
        with patch.object(Replica, "__enter__", enter):
            result = runner.correction_retention(self.seed(), Engine(), None, 1)
        self.assertEqual(len(result["rows"]), 3)
        self.assertTrue(all(r["outcome"] == "no_response" for r in result["rows"]))
        self.assertTrue(all(v["mean"] == 0 for v in result["response_rate"].values()))
        self.assertTrue(all(v["mean"] is None for v in result["retention_given_answer"].values()))


if __name__ == "__main__":
    unittest.main()
