"""Her memory is one file: it is backed up online, every backup is checked, the rotation keeps the
right ones, and a damaged backup is never taken for a good one."""
import asyncio
import os
import random
import shutil
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta

from src.core import clock
from src.core.backup import BackupManager, _stamp
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, START
from src.levels.l2_memory.embedding import Embedder
from src.levels.l2_memory.layers import EpisodicMemory
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = 86400

def make_db():
    folder = tempfile.mkdtemp(prefix="noa_bk_")
    path = os.path.join(folder, "memory.sqlite")
    clock.set_clock(SimClock(START))
    gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=path, rng=random.Random(0))
    for i in range(5):
        gw.episodes.add(f"第 {i} 件事：今天和阿杰聊了猫", kind="dialogue", user_id="u1")
    gw.memory.flush()
    return gw, path

BASE = datetime(2026, 1, 1)

def at(day: int, hour: int = 3) -> float:
    return (BASE + timedelta(days=day, hours=hour)).timestamp()

def ymd(day: int) -> str:
    return (BASE + timedelta(days=day)).strftime("%Y%m%d")

class TestBackup(unittest.TestCase):
    def setUp(self):
        self.gw, self.path = make_db()
        self.mgr = BackupManager(self.path)

    def test_online_backup_is_checked_and_matches(self):
        r = self.mgr.run(at(0))
        self.assertTrue(r["ok"], r)
        self.assertTrue(os.path.exists(os.path.join(self.mgr.folder, r["name"])))
        self.assertFalse([n for n in os.listdir(self.mgr.folder) if n.endswith(".partial")])
        conn = sqlite3.connect(os.path.join(self.mgr.folder, r["name"]))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0], 5)
        conn.close()

    def test_backup_while_she_keeps_writing(self):
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                self.gw.episodes.add(f"又一件事 {i}", kind="dialogue", user_id="u1")
                i += 1
        t = threading.Thread(target=writer)
        t.start()
        try:
            results = [self.mgr.run(at(d)) for d in range(3)]
        finally:
            stop.set()
            t.join()
        self.assertTrue(all(r["ok"] for r in results), [r.get("error") for r in results])

    def test_due_only_after_a_day(self):
        self.assertTrue(self.mgr.due(at(0)))
        self.mgr.run(at(0))
        self.assertFalse(self.mgr.due(at(0) + 3600))
        self.assertTrue(self.mgr.due(at(0) + DAY + 1))
        self.assertIsNone(self.mgr.run_if_due(at(0) + 3600))

    def test_rotation_keeps_recent_days_and_one_per_week(self):
        for d in range(0, 40):
            for hour in (3, 15):                         # two a day: only the newest of a day survives
                self.mgr.run(at(d, hour))
        names = [b["name"] for b in self.mgr.list()]
        stamps = [n[7:15] for n in names]
        self.assertEqual(len(stamps), len(set(stamps)))   # one per day at most
        days = {s for s in stamps}
        latest = ymd(39)
        recent = [ymd(d) for d in range(33, 40)]
        self.assertTrue(set(recent) <= days)              # the last 7 days are all there
        self.assertIn(latest, days)
        self.assertLessEqual(len(names), 7 + 4)           # a week's worth plus a few weekly ones
        self.assertGreater(len(names), 7)

    def test_a_damaged_backup_is_refused_and_never_replaces_a_good_one(self):
        good = self.mgr.run(at(0))
        self.assertTrue(good["ok"])
        real = self.mgr.verify

        def damaged(path, before=None, after=None):          # the copy came out short of the live data
            out = real(path, before, after)
            return {**out, "ok": False, "problems": ["episodes: 0 rows, the live database already had 5"]}
        self.mgr.verify = damaged
        bad = self.mgr.run(at(1))
        self.assertFalse(bad["ok"])
        self.assertEqual([b["name"] for b in self.mgr.list()], [good["name"]])
        self.assertFalse([n for n in os.listdir(self.mgr.folder) if n.endswith(".partial")])

    def test_verify_notices_a_truncated_file_and_missing_rows(self):
        r = self.mgr.run(at(0))
        path = os.path.join(self.mgr.folder, r["name"])
        broken = path + ".broken"
        with open(path, "rb") as f:
            data = f.read()
        with open(broken, "wb") as f:
            f.write(data[: len(data) // 2])
        self.assertFalse(self.mgr.verify(broken)["ok"])
        conn = sqlite3.connect(self.path)                        # the live database has more than the copy has
        self.assertFalse(self.mgr.verify(path, before={"episodes": 99})["ok"])
        conn.close()

    def test_restore_check_uses_it_as_her_memory(self):
        r = self.mgr.run(at(0))
        check = self.mgr.restore_check(os.path.join(self.mgr.folder, r["name"]))
        self.assertTrue(check["ok"], check)
        self.assertEqual(check["episodes"], 5)
        self.assertTrue(check["recall_works"])
        self.assertFalse(self.mgr.restore_check(os.path.join(self.mgr.folder, "nothing.sqlite"))["ok"])

    def test_a_second_copy_elsewhere(self):
        other = tempfile.mkdtemp(prefix="noa_bk2_")
        mgr = BackupManager(self.path, extra_dirs=[other])
        r = mgr.run(at(0))
        self.assertEqual(r["copies"], 1)
        self.assertEqual([b["name"] for b in mgr.list(other)], [r["name"]])

    def test_not_enough_space_is_reported_not_attempted(self):
        real = shutil.disk_usage
        shutil.disk_usage = lambda p: shutil._ntuple_diskusage(100, 99, 1)
        try:
            r = self.mgr.run(at(0))
        finally:
            shutil.disk_usage = real
        self.assertFalse(r["ok"])
        self.assertIn("free space", r["error"])
        self.assertEqual(self.mgr.list(), [])

class TestInTheHeartbeat(unittest.TestCase):
    def test_tick_makes_the_backup_and_tells_her_keeper_when_it_fails(self):
        gw, path = make_db()
        gw.environment.get_context = lambda: {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None,
                                              "is_night": False}
        gw.backup = BackupManager(path)
        asyncio.run(gw.tick())
        self.assertEqual(len(gw.backup.list()), 1)
        asyncio.run(gw.tick())                                   # not due again
        self.assertEqual(len(gw.backup.list()), 1)

        gw.evolve_creator = "1001"
        gw.backup = BackupManager(path, folder=os.path.join(path, "not-a-folder"))   # cannot be created
        gw.last_backup_check = 0.0
        asyncio.run(gw.tick())
        alerts = [m for m in gw.output_queue if "备份失败" in m.get("content", "")]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["target"], {"type": "private", "id": "1001"})
        gw.last_backup_check = 0.0
        asyncio.run(gw.tick())                                   # one alert, not one every ten minutes
        self.assertEqual(len([m for m in gw.output_queue if "备份失败" in m.get("content", "")]), 1)

if __name__ == "__main__":
    unittest.main()

class TestDiaryAndLegacy(unittest.TestCase):
    def test_diary_moves_out_of_the_legacy_table_once(self):
        folder = tempfile.mkdtemp(prefix="noa_diary_")
        path = os.path.join(folder, "m.sqlite")
        from src.levels.l2_memory.storage.database import MemoryDatabase
        from src.levels.l2_memory.layers import Diary
        db = MemoryDatabase(path)
        conn = sqlite3.connect(path)
        conn.executemany("INSERT INTO memories (user_id, category, content, confidence, created_at, last_accessed) "
                         "VALUES ('self', 'fact', ?, 1.0, ?, ?)",
                         [("Diary: 今天等了很久", 100.0, 100.0), ("The user is the creator", 101.0, 101.0),
                          ("Diary: 今天聊了猫", 102.0, 102.0)])
        conn.commit()
        conn.close()
        diary = Diary(path)
        self.assertEqual([d["text"] for d in diary.recent()][::-1], ["今天等了很久", "今天聊了猫"])
        Diary(path)                                                    # opening again does not copy twice
        self.assertEqual(len(diary.recent(10)), 2)
        diary.write("新的一篇")
        self.assertEqual(diary.recent(1)[0]["text"], "新的一篇")

    def test_a_dream_writes_the_diary_and_no_message_calls_the_model_for_facts(self):
        clock.set_clock(SimClock(START + 2 * 3600))
        llm = ScriptedLLM("我喜欢小鱼干")
        gw = GlobalWorkspace(EventBus(), llm, db_path=os.path.join(tempfile.mkdtemp(prefix="noa_dg_"), "g.sqlite"),
                             rng=random.Random(0))
        gw.environment.get_context = lambda: {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None,
                                              "is_night": False}
        asyncio.run(gw.hear("我喜欢你，我叫小明，我爱吃鱼", user_id="u1", nickname="小明", addressed=True))
        self.assertFalse([c for c in llm.calls if "Extract any permanent facts" in c["prompt"]])
        conn = sqlite3.connect(gw.memory.db.db_path)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
        conn.close()
        for i in range(25):
            gw.episodes.add(f"和小明聊天 {i}", kind="dialogue", user_id="u1", valence=0.5)
        report = gw._dream_process()
        self.assertTrue(report and report.get("diary"))
        self.assertEqual(gw.diary.recent(1)[0]["text"], report["diary"])

    def test_the_creator_is_her_creator_from_the_config(self):
        clock.set_clock(SimClock(START))
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=os.path.join(tempfile.mkdtemp(prefix="noa_cr_"), "g.sqlite"),
                             rng=random.Random(0))
        gw.evolve_creator = "1001"
        gw._ensure_self()
        self.assertIn("创造者", gw.memory.db.get_relationship("1001")["relation_type"])
