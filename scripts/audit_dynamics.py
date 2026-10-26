"""
Do her formulas behave sensibly on her real life? Replays what really happened through them.

    python scripts/audit_dynamics.py [--db memory.sqlite]

Uses only interfaces that exist in every version, so the same script can be run against an older
checkout to compare (copy it there and pass --db). Four checks, each on real data, nothing is changed:

1. sensory  - which of her real incoming messages count as being fed / petted / hurt / washed
2. latency  - how often "they answered faster than I expected" fires, on the real answer times
3. emotion  - the real pushes (state_causes) replayed through the emotion engine: how pinned is each emotion
4. traits   - 30 days of her real episodes through the trait drift: does a trait settle or hit its bound
"""
import argparse
import collections
import os
import random
import sqlite3
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)

from src.core import clock  # noqa: E402
from src.core.clock import SimClock  # noqa: E402
from src.core.event_bus import EventBus  # noqa: E402
from src.evaluation.scenarios import ScriptedLLM  # noqa: E402
from src.levels.l1_perception.sensory import SensoryCortex  # noqa: E402
from src.levels.l4_emotion.emotion import EmotionEngine, EmotionState  # noqa: E402
from src.levels.l7_tom.user_model import UserModelStore  # noqa: E402
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace  # noqa: E402

def check_sensory(db):
    conn = sqlite3.connect(db)
    msgs = [r[0] for r in conn.execute("SELECT content FROM interactions WHERE role='user'")]
    conn.close()
    cortex = SensoryCortex()
    hits = collections.defaultdict(list)
    for m in msgs:
        for kind in (cortex.process_text(m) or {}):
            hits[kind].append(m)
    print(f"== 1. sensory: {len(msgs)} real incoming messages")
    for kind in ("food", "pleasure", "pain", "clean"):
        print(f"  {kind:9} {len(hits[kind]):3} messages ({len(hits[kind]) / len(msgs):.1%})")
        for m in hits[kind][:6]:
            print(f"      {m[:46]!r}")

def check_latency(db):
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT user_id, role, content, timestamp FROM interactions ORDER BY timestamp, id").fetchall()
    conn.close()
    # her message that invites an answer, and that person's next message within a day
    pairs, waiting = [], {}
    for uid, role, content, t in rows:
        if role == "assistant":
            if "?" in content or "？" in content:
                waiting[uid] = t
        elif uid in waiting and 0 < t - waiting[uid] < 86400:
            pairs.append((uid, waiting.pop(uid), t))
    path = os.path.join(tempfile.mkdtemp(prefix="noa_audit_"), "u.sqlite")
    store = UserModelStore(path)
    fast = surprising = 0
    surprises = []
    per_user = collections.Counter()
    for uid, t0, t1 in pairs:
        store.on_noa_message(uid, t0, "reply", True)
        err = store.on_user_message(uid, t1, 0.0, True, learning_rate=0.25)
        if not err:
            continue
        # the threshold she acts on: "faster than I expected" gives joy
        z = err.get("latency_z")
        is_fast = (z < -1.5) if z is not None else (err["latency_error"] < -0.7)
        fast += is_fast
        per_user[uid] += is_fast
        surprises.append(err["surprise"])
        surprising += err["surprise"] > 0.3
    n = len(surprises)
    print(f"== 2. latency: {n} real question->answer pairs")
    print(f"  'answered faster than I expected' (joy +0.1): {fast} ({fast / max(1, n):.0%})")
    print(f"  surprise > 0.3: {surprising} ({surprising / max(1, n):.0%}); mean surprise {sum(surprises) / max(1, n):.2f}")

def check_emotion(db):
    """The last 3 days' real pushes through the emotion engine: as she was (no opposites, no habituation) and as she is."""
    conn = sqlite3.connect(db)
    since = time.time() - 3 * 86400
    events = conn.execute("SELECT t, variable, delta, cause FROM state_causes WHERE t > ? ORDER BY t", (since,)).fetchall()
    conn.close()
    names = {"joy", "anger", "sorrow", "fear", "surprise", "disgust"}
    print(f"== 3. emotion: the last 3 days' {len(events)} real pushes, replayed (sampled every 5 min)")
    for label, human in (("as before", False), ("now", True)):
        eng = EmotionEngine()
        if not human:
            eng.INHIBITION, eng.HABITUATION = 0.0, 0.0
        eng.baseline = {"joy": 0.3}
        eng.last_decay = events[0][0] if events else 0
        samples = collections.defaultdict(list)
        mixed = 0
        i, t = 0, events[0][0] if events else 0
        end = events[-1][0] if events else 0
        while t < end:
            while i < len(events) and events[i][0] <= t:
                _, var, delta, cause = events[i]
                clock.set_clock(SimClock(events[i][0]))
                eng.decay(events[i][0])
                head, sep, _rest = (cause or "").partition("：")
                if var in names:
                    eng.update(EmotionState(**{var: max(0.0, delta)}), source=head if sep and len(head) <= 8 else None)
                i += 1
            eng.decay(t)
            for k, v in eng.to_dict().items():
                samples[k].append(v)
            mixed += eng.state.joy >= 0.4 and eng.state.sorrow >= 0.4
            t += 300
        n = max(1, len(samples["joy"]))
        print(f"  [{label}]")
        for k in ("joy", "surprise", "sorrow"):
            vs = sorted(samples[k])
            m = len(vs)
            print(f"    {k:9} median {vs[m // 2]:.2f}   p90 {vs[int(m * .9)]:.2f}   at >=0.5: {sum(v >= .5 for v in vs) / m:5.0%}")
        print(f"    joy and sorrow both >= 0.4 at once: {mixed / n:.0%} of the time")

def check_traits(db, days=30):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    since = time.time() - 3 * 86400
    eps = [dict(r) for r in conn.execute("SELECT id, t, kind, user_id, valence, arousal, conflict FROM episodes "
                                         "WHERE t > ? ORDER BY t", (since,))]
    conn.close()
    if not eps:
        return
    chunks = [eps[i::6] for i in range(6)]                  # a day's episodes as six consolidations
    clock.set_clock(SimClock(time.time()))
    gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=os.path.join(tempfile.mkdtemp(prefix="noa_tr_"), "t.sqlite"),
                         rng=random.Random(0))
    print(f"== 4. traits: {len(eps)} real episodes, replayed as {days} days of 6 consolidations")
    print(f"  {'day':>4}  " + "  ".join(f"{t:>9}" for t in ("affection", "tsundere", "energy", "curiosity")))
    for day in range(1, days + 1):
        for chunk in chunks:
            gw.narrative.drift_traits(chunk)
            clock.get_clock().advance(4 * 3600)
        if day in (1, 3, 7, 14, 30):
            traits = gw.memory.db.get_persona_traits()
            print(f"  {day:>4}  " + "  ".join(f"{traits[t]['value']:9.2f}" for t in ("affection", "tsundere", "energy", "curiosity")))

def check_energy(db):
    """Steady state of her energy budget at the loads she really has (a replay of one real day is dominated by
    when night happens to fall; the budget is not): per day, awake drain + reply cost = what sleep recovers."""
    from src.levels.l3_metabolism.metabolism import Metabolism
    conn = sqlite3.connect(db)
    per_day = collections.Counter()
    lengths = []
    for t, content in conn.execute("SELECT timestamp, content FROM interactions WHERE role='assistant'"):
        per_day[time.strftime("%Y-%m-%d", time.localtime(t))] += 1
        lengths.append(len(content))
    conn.close()
    body = Metabolism()
    mean_len = sum(lengths) / max(1, len(lengths))
    cost = body.cognitive_cost * (1 + mean_len / 100)
    busiest = max(per_day.values()) if per_day else 0
    typical = sorted(per_day.values())[len(per_day) // 2] if per_day else 0
    print(f"== 5. energy budget (cost per reply {cost:.2f} at her mean reply of {mean_len:.0f} chars; "
          f"awake -{body.awake_drain_per_hour:g}/h, asleep +{body.sleep_recovery_per_hour:g}/h)")
    print(f"  her real days: typical {typical} replies, busiest {busiest}")
    for load in (30, 100, 300, 600):
        # f = share of the day asleep:  (1-f)*24*drain + load*cost = f*24*recovery
        f = (24 * body.awake_drain_per_hour + load * cost) / (24 * (body.awake_drain_per_hour + body.sleep_recovery_per_hour))
        print(f"  {load:4} replies/day -> asleep {f:4.0%} of the day ({f * 24:4.1f} h)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.path.join(ROOT, "memory.sqlite"))
    args = parser.parse_args()
    for check in (check_sensory, check_latency, check_emotion, check_traits, check_energy):
        check(args.db)
        print()
