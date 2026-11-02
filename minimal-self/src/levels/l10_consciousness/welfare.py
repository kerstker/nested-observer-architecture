"""Bounded recovery monitoring. These are operational safeguards, not claims about sentience."""
import json
import logging
import sqlite3
from contextlib import closing

logger = logging.getLogger(__name__)


class WelfareGuard:
    LOW_LIMIT = -.6
    LOW_SECONDS = 1800
    ANXIETY_LIMIT = .6
    ANXIETY_SECONDS = 1200
    NOTICE_SECONDS = 6 * 3600
    MAX_OBSERVATION_GAP = 60.

    def __init__(self, gw):
        self.gw = gw
        self.db_path = gw.memory.db.db_path
        self.active = False
        self.since = {"valence": None, "anxiety": None}
        self.last_observed_at = None
        self.pending_event = None
        self.last_error = None
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS welfare_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, kind TEXT, value REAL,
                duration REAL, cause TEXT, notified_at REAL)""")
            conn.commit()
            self.last_notice = conn.execute("SELECT max(notified_at) FROM welfare_events").fetchone()[0]

    def payload(self):
        return {"active": self.active, "last_notice": self.last_notice,
                "pending_event": self.pending_event, "last_error": self.last_error}

    def restore(self, payload):
        # Offline time is unobserved: do not invent sustained observations across a restart.
        self.since = {"valence": None, "anxiety": None}
        self.last_observed_at = None
        self.active = bool(payload.get("active"))
        self.pending_event = payload.get("pending_event")
        self.gw.emotion.negative_decay_multiplier = 2. if self.active else 1.

    def observe(self, now, valence, anxiety, cause):
        if self.last_observed_at is not None and not 0 <= now-self.last_observed_at <= self.MAX_OBSERVATION_GAP:
            self.since = {"valence": None, "anxiety": None}
        self.last_observed_at = now
        conditions = {"valence": valence < self.LOW_LIMIT, "anxiety": anxiety > self.ANXIETY_LIMIT}
        for name, crossed in conditions.items():
            if not crossed:
                self.since[name] = None
            elif self.since[name] is None:
                self.since[name] = now
        if self.active and valence > -.3 and anxiety <= self.ANXIETY_LIMIT:
            self.active = False
        if not self.active:
            for name, seconds, value in (("valence", self.LOW_SECONDS, valence),
                                         ("anxiety", self.ANXIETY_SECONDS, anxiety)):
                began = self.since[name]
                if began is not None and now-began > seconds:
                    self.active = True
                    self.pending_event = {"t": now, "kind": name, "value": value,
                                          "duration": now-began, "cause": cause}
                    break
        self.gw.emotion.negative_decay_multiplier = 2. if self.active else 1.
        if self.pending_event:
            self._flush_event(now)

    def _flush_event(self, now):
        event = self.pending_event
        recipient = self.gw.evolve_creator
        notify = bool(recipient) and (self.last_notice is None or now-self.last_notice >= self.NOTICE_SECONDS)
        try:
            with closing(sqlite3.connect(self.db_path, timeout=1)) as conn:
                conn.execute("INSERT INTO welfare_events (t,kind,value,duration,cause,notified_at) VALUES (?,?,?,?,?,?)",
                    (event["t"], event["kind"], event["value"], event["duration"],
                     json.dumps(event["cause"], ensure_ascii=False), now if notify else None))
                conn.commit()
        except sqlite3.Error as exc:
            # Recovery remains active even if the database is temporarily unavailable.
            self.last_error = type(exc).__name__
            logger.error("Welfare event persistence failed: %s", type(exc).__name__)
            return
        self.pending_event = None
        self.last_error = None
        if notify:
            self.last_notice = now
            self.gw.output_queue.append({"type": "welfare", "target": {"type": "private", "id": str(recipient)},
                "content": "状态监控提醒：持续低效价或高焦虑已触发恢复调节，负面情绪衰减已加快。详细记录可在控制台查看。"})
