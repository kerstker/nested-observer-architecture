"""
Level 17c: self-diagnosis - what she finds wrong with herself, from her own records.

Self-improvement is a closed loop only if the problem comes from her and the result goes back
to her: findings are read off her lived data (calibration, kept and broken promises, rumination,
failed searches, corrections), each is tied to a fitness metric where one exists, every attempt
is remembered (so she does not try the same thing every day), and after an applied change she
checks, on her real data again, whether it actually helped.
"""
import sqlite3
from typing import Any, Dict, List, Optional

DAY = 86400.0

class SelfDiagnosis:
    RETRY_AFTER = 7 * DAY          # don't attempt the same problem again too soon
    GIVE_UP_AFTER = 3              # rejected this many times: leave it for a month

    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS self_improvement (
            id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, key TEXT, problem TEXT, metric TEXT,
            value_before REAL, proposal_id TEXT, status TEXT, verdict TEXT, fitness TEXT,
            value_after REAL, checked_at REAL, note TEXT
        )''')
        conn.commit()
        conn.close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- reading her life ---------------------------------------------------------

    def _count(self, sql: str, args=()) -> int:
        conn = self._conn()
        try:
            return conn.execute(sql, args).fetchone()[0]
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def measure(self, gw, now: float) -> Dict[str, Dict[str, Any]]:
        """Every quantity she can diagnose herself by, right now."""
        week = now - 7 * DAY
        out: Dict[str, Dict[str, Any]] = {}
        for domain, st in gw.metacog.stats.items():
            if st["n"] >= 5:
                # What is left after her own calibration, not the raw overconfidence she already corrects for
                sure = st.get("cal", st["conf"])
                out[f"overconfident_{domain}"] = {
                    "value": sure - st["acc"], "threshold": 0.15, "metric": "calibration_gap",
                    "text": f"我在{gw.metacog.DOMAIN_WORDS[domain]}上太自信了",
                    "evidence": f"最近 {int(st['n'])} 次有结论的回答里，校准以后我平均还有 {sure:.0%} 的把握，"
                                f"实际只对了 {st['acc']:.0%}",
                    "where": "minimal-self/src/levels/l10_consciousness/metacognition.py（把握的估计和校准）"}
        broken = self._count("SELECT COUNT(*) FROM goals WHERE kind='promise' AND status='abandoned' AND last_progress>=?",
                             (week,))
        kept = self._count("SELECT COUNT(*) FROM goals WHERE kind='promise' AND status='done' AND last_progress>=?",
                           (week,))
        if broken + kept >= 2:
            out["broken_promises"] = {
                "value": broken / (broken + kept), "threshold": 0.34, "metric": "hours_to_keep",
                "text": "我答应别人的事经常做不到",
                "evidence": f"这一周答应了 {broken + kept} 件事，做到 {kept} 件，没做到 {broken} 件",
                "where": "minimal-self/src/levels/l8_agency/goals.py 和 global_workspace.py 里主动找人的部分"}
        failed = self._count("SELECT COUNT(*) FROM episodes WHERE kind='search_failed' AND t>=?", (week,))
        found = self._count("SELECT COUNT(*) FROM episodes WHERE kind='search' AND t>=?", (week,))
        if failed + found >= 4:
            out["search_failing"] = {
                "value": failed / (failed + found), "threshold": 0.5, "metric": None,
                "text": "我上网查东西老是失败",
                "evidence": f"这一周查了 {failed + found} 次，失败 {failed} 次",
                "where": "minimal-self/src/levels/l19_search/web_search.py"}
        letting_go = sum(gw.attention_schema.disengaged.values())
        out["rumination"] = {
            "value": float(letting_go), "threshold": 6, "metric": "rumination_share",
            "text": "我老是被同一件烦心事缠住",
            "evidence": f"我已经有 {letting_go} 次发现自己反复想同一件事、不得不主动放下",
            "where": "minimal-self/src/levels/l10_consciousness/attention_schema.py"}
        stalled = self._count("SELECT COUNT(*) FROM goals WHERE kind!='promise' AND status='abandoned' AND last_progress>=?",
                              (now - 14 * DAY,))
        done = self._count("SELECT COUNT(*) FROM goals WHERE kind!='promise' AND status='done' AND last_progress>=?",
                           (now - 14 * DAY,))
        if stalled + done >= 3:
            out["goals_stall"] = {
                "value": stalled / (stalled + done), "threshold": 0.5, "metric": None,
                "text": "我给自己定的目标大多半途而废",
                "evidence": f"两周里 {stalled + done} 个目标，完成 {done} 个，放弃 {stalled} 个",
                "where": "minimal-self/src/levels/l8_agency/goals.py"}
        return out

    # --- choosing what to work on ------------------------------------------------

    def attempts(self, key: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._conn()
        q = "SELECT * FROM self_improvement" + (" WHERE key=?" if key else "") + " ORDER BY id DESC LIMIT ?"
        rows = conn.execute(q, ((key, limit) if key else (limit,))).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _open(self, key: str, now: float) -> bool:
        past = self.attempts(key, limit=10)
        if past and now - past[0]["t"] < self.RETRY_AFTER:
            return False
        rejected = [a for a in past if a["status"] in ("rejected_fitness", "tests_failed", "no_change", "error")
                    and now - a["t"] < 30 * DAY]
        return len(rejected) < self.GIVE_UP_AFTER

    def diagnose(self, gw, now: float) -> List[Dict[str, Any]]:
        """Problems worth working on, the most serious first."""
        found = []
        for key, m in self.measure(gw, now).items():
            if m["value"] >= m["threshold"] and self._open(key, now):
                found.append({"key": key, "severity": m["value"] / m["threshold"], **m})
        return sorted(found, key=lambda f: -f["severity"])

    @staticmethod
    def task_for(f: Dict[str, Any]) -> str:
        target = (f"其中和这个问题直接相关的指标是 {f['metric']}，改动要让它变好。" if f.get("metric")
                  else "这个问题目前没有专门的评估指标，至少不能让任何评估变差。")
        return (f"这是乃爱自己从她的真实经历里发现的问题：{f['text']}。\n"
                f"依据：{f['evidence']}。\n"
                f"相关代码大概在：{f['where']}。\n"
                "请先读懂相关代码，找到造成这个问题最可能的原因，做一个小而具体的改动。\n"
                "改完以后系统会在副本里跑全部单元测试，再用她的行为评估（每一项、3 个随机种子）作为适应度，"
                f"和改动前比较：任何一项从通过变成不通过，这份改动都会被否决。{target}")

    # --- remembering attempts -----------------------------------------------------

    def record(self, f: Dict[str, Any], now: float) -> int:
        conn = self._conn()
        cur = conn.execute("INSERT INTO self_improvement (t, key, problem, metric, value_before, status) "
                           "VALUES (?, ?, ?, ?, ?, 'running')", (now, f["key"], f["text"], f.get("metric"), f["value"]))
        aid = cur.lastrowid
        conn.commit()
        conn.close()
        return aid

    def finish(self, aid: int, result: Dict[str, Any]):
        conn = self._conn()
        conn.execute("UPDATE self_improvement SET proposal_id=?, status=?, verdict=?, fitness=? WHERE id=?",
                     (result.get("id"), result.get("status"), result.get("verdict"), result.get("fitness"), aid))
        conn.commit()
        conn.close()

    def follow_up(self, gw, now: float, applied: set, min_days: float = 3.0) -> List[Dict[str, Any]]:
        """After a change was applied and she has lived with it for a while: did it really help?"""
        current = self.measure(gw, now)
        checked = []
        for a in self.attempts(limit=50):
            if a["checked_at"] or a["proposal_id"] not in applied or now - a["t"] < min_days * DAY:
                continue
            m = current.get(a["key"])
            after = m["value"] if m else None
            helped = after is not None and after < a["value_before"]
            note = (f"真的变好了：{a['value_before']:.2f} → {after:.2f}" if helped else
                    f"没看出变化：{a['value_before']:.2f} → {after:.2f}" if after is not None else "没有足够的新数据来判断")
            conn = self._conn()
            conn.execute("UPDATE self_improvement SET value_after=?, checked_at=?, note=? WHERE id=?",
                         (after, now, note, a["id"]))
            conn.commit()
            conn.close()
            checked.append({**a, "value_after": after, "helped": helped, "note": note})
        return checked

    def describe(self, limit: int = 3) -> str:
        rows = self.attempts(limit=limit)
        if not rows:
            return "还没有自己发起过改进。"
        return "；".join(f"{r['problem']}（{r['status'] or '进行中'}" + (f"，{r['note']}" if r["note"] else "") + "）"
                        for r in rows)
