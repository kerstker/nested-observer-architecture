"""Who is who and how people and things relate: scoping, who said what, updates and disputes,
privacy across chats, her own identity, and claims that are only claims."""
import asyncio
import json
import os
import random
import re
import sqlite3
import tempfile
import unittest

from src.core import clock
from src.core.clock import SimClock
from src.core.event_bus import EventBus
from src.evaluation.scenarios import ScriptedLLM, START
from src.levels.l2_memory.graph import RelationGraph, PRIVATE, GROUP, PUBLIC
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace

DAY = 86400

def db_path():
    return os.path.join(tempfile.mkdtemp(prefix="noa_graph_"), "g.sqlite")

class Extractor:
    """Plays the model: for each episode id in the prompt, the relations scripted for that episode's text."""
    def __init__(self, by_text, fail_first=0):
        self.by_text = by_text
        self.fail = fail_first
        self.prompts = []

    def __call__(self, prompt, **kw):
        self.prompts.append(prompt)
        if self.fail:
            self.fail -= 1
            return {"reply": "{\"relations\": [ {\"subject\": \"截断"}
        out = []
        for eid, text in re.findall(r"^\[(\d+)\] \([^)]*\) (.*)$", prompt.split("[经历]")[1], re.M):
            for key, rels in self.by_text.items():
                if key in text:
                    out += [{**r, "episode": int(eid)} for r in rels]
        return {"relations": out}

def make_gw():
    clock.set_clock(SimClock(START))
    gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=db_path(), rng=random.Random(0))
    gw.environment.get_context = lambda: {"time": "12:00", "period": "Day", "weather": "Sunny", "festival": None,
                                          "is_night": False}
    return gw

def say(gw, uid, nick, text, group=None, days_ago=0.0):
    gw.memory.db.upsert_user(uid, nick, nick)
    return gw.episodes.add(f"{nick}：{text}\n我：嗯", kind="dialogue", user_id=uid, group_id=group,
                           t=clock.now() - days_ago * DAY)

R = lambda s, p, o, said, **kw: {"subject": s, "predicate": p, "object": o, "said_by": said,
                                 "subject_kind": kw.pop("sk", "person"), "object_kind": kw.pop("ok", "person"), **kw}

class TestGraph(unittest.TestCase):
    def test_multi_hop_and_who_said_it(self):
        gw = make_gw()
        say(gw, "u1", "阿杰", "阿珂养了只猫叫团子", days_ago=5)
        say(gw, "u1", "阿杰", "团子生病了", days_ago=1)
        ex = Extractor({"养了只猫": [R("阿珂", "养的猫", "团子", "阿杰", ok="pet")],
                        "团子生病": [R("团子", "近况", "生病了", "阿杰", sk="pet", ok="value", single=True)]})
        self.assertEqual(gw.graph.extract(ex, gw._names_in_chat), 2)
        ctx = gw.graph.context_for("阿珂最近怎么样", "u1", "阿杰", None)
        self.assertIn("阿珂 —养的猫→ 团子（阿杰说的", ctx)
        self.assertIn("团子 —近况→ 生病了", ctx)                    # two hops from 阿珂

    def test_same_name_in_two_private_chats_is_two_people(self):
        gw = make_gw()
        say(gw, "u2", "小美", "小王是我弟弟")
        say(gw, "u3", "老郑", "小王是我们前台")
        ex = Extractor({"我弟弟": [R("小王", "姐姐", "小美", "小美")],
                        "我们前台": [R("小王", "职业", "前台", "老郑", ok="value")]})
        gw.graph.extract(ex, gw._names_in_chat)
        a = gw.graph.context_for("小王最近还好吗", "u2", "小美", None)
        b = gw.graph.context_for("小王今天来了吗", "u3", "老郑", None)
        self.assertIn("姐姐", a)
        self.assertNotIn("前台", a)
        self.assertIn("前台", b)
        self.assertNotIn("姐姐", b)
        self.assertEqual(gw.graph.context_for("你认识小王吗", "u9", "陌生人", "g1"), "")   # nothing private leaks

    def test_updates_replace_and_corrections_are_kept_with_their_source(self):
        gw = make_gw()
        say(gw, "u1", "阿杰", "小林下个月结婚", days_ago=6)
        say(gw, "u1", "阿杰", "婚礼推迟到明年了", days_ago=2)
        say(gw, "u4", "阿凯", "白羽去日本出差", group="g1", days_ago=4)
        say(gw, "u5", "小鹿", "阿凯说错了，是韩国", group="g1", days_ago=2)
        say(gw, "u6", "小周", "白羽其实去的是台湾", group="g1", days_ago=1)
        ex = Extractor({"下个月结婚": [R("小林", "婚礼时间", "下个月", "阿杰", ok="value", single=True)],
                        "推迟到明年": [R("小林", "婚礼时间", "明年", "阿杰", ok="value", single=True)],
                        "去日本": [R("白羽", "出差地点", "日本", "阿凯", ok="value", single=True)],
                        "是韩国": [R("白羽", "出差地点", "韩国", "小鹿", ok="value", single=True, corrects=True)],
                        "去的是台湾": [R("白羽", "出差地点", "台湾", "小周", ok="value", single=True)]})
        gw.graph.extract(ex, gw._names_in_chat)
        wedding = gw.graph.context_for("小林什么时候结婚", "u1", "阿杰", None)
        self.assertIn("下个月（阿杰说的，6 天前）【已过时，后来改成：明年】", wedding)
        trip = gw.graph.context_for("白羽去哪", "u7", "路人", "g1")
        self.assertIn("日本（阿凯说的，4 天前）【已过时，后来改成：韩国】", trip)
        self.assertIn("韩国（小鹿说的，2 天前）【有不同说法】", trip)     # a third, uncorrected report contests it
        self.assertIn("台湾（小周说的，昨天）【有不同说法】", trip)

    def test_an_older_report_never_replaces_a_newer_one(self):
        gw = make_gw()
        graph = gw.graph
        scope = (PUBLIC, None, None)
        s = graph.entity("小林", "person", scope)
        new = graph.add(s, "婚礼时间", None, "明年", "u1", "阿杰", None, clock.now() - 1 * DAY, scope, single=True)
        graph.add(s, "婚礼时间", None, "下个月", "u1", "阿杰", None, clock.now() - 6 * DAY, scope, single=True)
        conn = sqlite3.connect(graph.db_path)
        self.assertEqual(conn.execute("SELECT status FROM relations WHERE id = ?", (new,)).fetchone()[0], "current")
        conn.close()

    def test_unreadable_extraction_is_retried_not_lost(self):
        gw = make_gw()
        for i in range(8):
            say(gw, "u1", "阿杰", f"第{i}件事：阿珂养了只猫")
        ex = Extractor({"养了只猫": [R("阿珂", "养的猫", "团子", "阿杰", ok="pet")]}, fail_first=1)
        gw.graph.extract(ex, gw._names_in_chat)
        self.assertIn("养的猫", gw.graph.context_for("阿珂", "u1", "阿杰", None))
        self.assertEqual(gw.graph.stats()["episodes_unread"], 0)

    def test_sensitive_claims_about_people_are_not_kept(self):
        gw = make_gw()
        say(gw, "u4", "飛硬", "Asuna是男同", group="g1")
        gw.graph.extract(Extractor({"男同": [R("Asuna", "性取向", "男同", "飛硬", ok="value")]}), gw._names_in_chat)
        self.assertEqual(gw.graph.context_for("Asuna 是谁", "u5", "x", "g1"), "")

class TestIdentity(unittest.TestCase):
    def test_she_knows_her_number_and_her_real_creator(self):
        gw = make_gw()
        gw.evolve_creator = "1001"
        gw.memory.db.upsert_user("1001", "kerst", "kerst")

        class QQ:
            self_id = "1000000002"
            @staticmethod
            def connected():
                return False
        gw.qq = QQ()
        gw._ensure_self()
        ctx = gw.graph.context_for("1000000002 是谁", "u9", "路人", None)
        self.assertIn("我 —QQ号→ 1000000002", ctx)
        self.assertIn("我 —创造者→ kerst", ctx)
        self.assertIn("我的创造者是 kerst", gw._self_facts())

    def test_claimed_roles_are_recorded_as_claims(self):
        gw = make_gw()
        gw.evolve_creator = "1001"
        conn = sqlite3.connect(gw.memory.db.db_path)
        conn.execute("REPLACE INTO relationships VALUES ('2002', 'self', '用户声称是AI的父亲', 1.0, 0)")
        conn.execute("REPLACE INTO relationships VALUES ('1001', 'self', '创造者', 1.0, 0)")
        conn.commit()
        conn.close()
        gw._ensure_self()
        self.assertTrue(gw.memory.db.get_relationship("2002")["relation_type"].startswith("ta 自称"))
        self.assertEqual(gw.memory.db.get_relationship("1001")["relation_type"], "创造者")

class TestPrivateFacts(unittest.TestCase):
    def test_facts_learned_only_in_private_stay_private(self):
        gw = make_gw()
        eid = say(gw, "u2", "小美", "小王是我弟弟，读高三")
        conn = sqlite3.connect(gw.memory.db.db_path)
        conn.execute("INSERT INTO semantic_facts (t, user_id, fact, confidence, sources, visibility) "
                     "VALUES (?, 'u2', '小美的弟弟小王读高三', 0.7, ?, 'public')", (clock.now(), json.dumps([eid])))
        conn.commit()
        conn.close()
        from src.levels.l2_memory.layers import SemanticMemory
        sem = SemanticMemory(gw.memory.db.db_path, gw.episodes.embedder)       # migration on open
        self.assertEqual(sem.public_facts(), [])

if __name__ == "__main__":
    unittest.main()
