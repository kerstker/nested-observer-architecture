import json
import sqlite3
import time
from typing import Dict, Any, List

from src.core import clock
from src.levels.l10_consciousness.global_workspace import format_duration

def _rows(db_path: str, sql: str, args=()) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    except sqlite3.OperationalError:
        return []   # table not created yet
    finally:
        conn.close()

def _candidate(c) -> Dict[str, Any]:
    return {"source": c.source, "kind": c.kind, "content": c.content, "salience": round(c.salience, 3)}

def snapshot(gw) -> Dict[str, Any]:
    """Everything the console shows about Noa's inner life, as plain JSON."""
    now = clock.now()
    db = gw.memory.db.db_path
    h = gw.homeostasis
    engine = gw.inference_engine

    people = []
    for user_id, m in gw.user_models.models.items():
        user = gw.memory.db.get_user(user_id) or {}
        people.append({
            "user_id": user_id,
            "nickname": user.get("nickname") or user_id,
            "describe": gw.user_models.describe(user_id),
            "expected_latency": gw.user_models.expected_latency(user_id),
            "p_reply": gw.user_models.response_probability(user_id),
            "sentiment": m["sentiment"],
            "private": m.get("private", False),
            "waiting": bool(m.get("pending")) and not m["pending"].get("ignored"),
            "ignored": bool(m.get("pending")) and m["pending"].get("ignored"),
            "q_chat": gw.drives.q("social", "chat", user_id),
            "q_food": gw.drives.q("hunger", "ask_food", user_id),
        })
    people.sort(key=lambda p: p["q_chat"], reverse=True)

    continuity = None
    if gw.continuity and gw.continuity["gap"] > 60:
        continuity = {"gap": format_duration(gw.continuity["gap"]), "boot_t": gw.continuity["boot_t"]}

    return {
        "now": now,
        "awakening": gw.awakening,
        "continuity": continuity,
        "sleeping": gw.metabolism.is_sleeping,
        "energy": gw.metabolism.energy,
        "body": gw.metabolism.get_status_desc(),
        "needs": {"hunger": h.hunger, "social": h.social_need, "curiosity": h.curiosity_need, "hygiene": h.hygiene_need},
        "emotions": gw.emotion.to_dict(),
        "mood": gw.emotion.get_compound_state(),
        "valence": gw.emotion.valence(),
        "arousal": gw.emotion.arousal(),
        "anxiety": gw.existential_anxiety,
        "tension": gw.values.tension,
        "attention": gw.workspace.attention,
        "stream": [{"t": t, "winners": [_candidate(c) for c in winners]}
                   for t, winners in list(gw.workspace.trace)[-20:]][::-1],
        "pending": [_candidate(c) for c in gw.pending_candidates],
        "thoughts": list(gw.consciousness_stream)[-8:][::-1],
        "self_model": gw.self_model.describe(),
        "attention_schema": gw.attention_schema.describe(),
        "metacognition": gw.metacog.describe(),
        "goals": [{"id": g["id"], "kind": g["kind"], "text": g["text"], "progress": g["progress"],
                   "deadline": g["deadline"], "next": gw.goals.next_step(g)} for g in gw.goals.active()],
        "goal_history": [{"text": g["text"], "status": g["status"], "note": g["note"]} for g in gw.goals.history(5)],
        "autobiography": gw.autobiography.latest(),
        "traits": {k: v["value"] for k, v in gw.memory.db.get_persona_traits().items()},
        "trait_changes": _rows(db, "SELECT t, trait, old, new, reason FROM trait_changes ORDER BY id DESC LIMIT 8"),
        "causes": _rows(db, "SELECT t, variable, delta, cause FROM state_causes ORDER BY id DESC LIMIT 15"),
        "episodes": _rows(db, "SELECT id, t, kind, user_id, content, valence, arousal, surprise FROM episodes "
                              "ORDER BY t DESC, id DESC LIMIT 12"),
        "people": people,
        "generation": gw.last_generation_params,
        "backend": type(engine).__name__ if engine else "无",
        "usage": engine.usage_summary() if hasattr(engine, "usage_summary") else None,
        "queue": len(gw.output_queue),
        "decisions": list(gw.decisions)[-20:][::-1],
        "stickers": gw.stickers.stats(),
        "unread": sum(len(m) for m in gw.unread.values()),
        "autonomous_groups": sorted(gw.autonomous_groups),
        "muted_groups": {g: t for g, t in gw.group_mute_until.items() if t > now},
        "idle_thought_interval": gw.idle_thought_interval,
        "abilities": gw.abilities,
        "evolve": {"enabled": bool(gw.evolve_enabled), "in_progress": gw.evolve_in_progress,
                   "last": gw.last_evolve_result},
        "llm_calls": len(gw.llm_trace),
        "last_monologue": gw.monologues[-1] if gw.monologues else None,
        "inbox": list(getattr(gw, "console_inbox", [])),
        "tasks_pending": len(gw.tasks.pending()) if getattr(gw, "tasks", None) else 0,
    }

# ---- the deeper views: loaded only when their tab is open ---------------------------------

def _keyed(d: Dict[tuple, Any], now: float) -> List[Dict[str, Any]]:
    """(group_id, user_id) -> until, for timers that are still running."""
    out = []
    for key, until in d.items():
        if until > now:
            group_id, user_id = (key + (None, None))[:2] if isinstance(key, tuple) else (key, None)
            out.append({"group_id": group_id, "user_id": user_id, "until": until})
    return out

def mind(gw) -> Dict[str, Any]:
    """How she decides: attention model, metacognition, values, drives, self-prediction, self-diagnosis."""
    now = clock.now()
    db = gw.memory.db.db_path
    a = gw.attention_schema
    h = gw.homeostasis
    needs = {"hunger": h.hunger, "social": h.social_need, "hygiene": h.hygiene_need, "curiosity": h.curiosity_need}
    try:
        options = gw.drives.options(needs, gw.metabolism.energy, gw.user_models.private_users(),
                                    goal_value=lambda action, target: gw.goals.value_for(action, target, now))
    except Exception as e:
        options, options_error = [], str(e)
    else:
        options_error = None
    options.sort(key=lambda o: o.score, reverse=True)
    try:
        measures = gw.diagnosis.measure(gw, now)
    except Exception:
        measures = {}
    sm = gw.self_model
    return {
        "now": now,
        "attention": {
            "focus": a.focus,
            "predict": sorted(a.predict().items(), key=lambda x: -x[1])[:6],
            "captures": [{"t": t, "from": f, "to": k, "content": c} for t, f, k, c in list(a.captures)[::-1]],
            "suppressed": {k: v for k, v in a.suppressed.items() if v > now},
            "disengaged": a.disengaged,
            "history": [{"t": t, "kind": k, "content": c} for t, k, c in list(a.history)[-30:][::-1]],
            "report": a.describe(now),
        },
        "workspace": {"capacity": gw.workspace.capacity, "threshold": gw.workspace.ignition_threshold,
                      "noise": gw.workspace.noise},
        "metacog": {"stats": gw.metacog.stats, "domains": gw.metacog.DOMAIN_WORDS,
                    "pending": [{"user_id": u, **p} for u, p in gw.metacog.pending.items()],
                    "verbal_confidence": gw.last_verbal_confidence},
        "values": {
            "weights": gw.values.weights,
            "initial": gw.values.INITIAL,
            "names": {k: list(v) for k, v in gw.values.VALUES.items()},
            "last_appraisal": gw.values.last_appraisal,
            "autonomy": gw.values.autonomy,
            "tension": gw.values.tension,
            "changes": gw.values.history(10),
            "events": _rows(db, "SELECT t, name, kind, intensity, outcome, text, consolidated FROM value_events "
                                "ORDER BY id DESC LIMIT 12"),
        },
        "drives": {
            "options": [{"need": o.need, "action": o.action, "target": o.target,
                         "nickname": (gw.memory.db.get_user(o.target) or {}).get("nickname") if o.target else None,
                         "score": o.score, "pragmatic": o.pragmatic, "epistemic": o.epistemic,
                         "cost": o.cost, "belief": o.belief} for o in options[:12]],
            "error": options_error,
            "temperature": gw.drives.temperature,
            "next_action_in": max(0.0, gw.last_action_time + gw.min_action_interval - now),
            "beliefs": _rows(db, "SELECT need, action, target, alpha, beta, n FROM beliefs ORDER BY n DESC LIMIT 20"),
        },
        "self_prediction": {"pending": sm.pending, "energy_error": sm.energy_error,
                            "valence_error": sm.valence_error, "checks": sm.checks},
        "temperament": {"style_edge": gw.style_edge},
        "diagnosis": {
            "measures": [{"key": k, **{f: m.get(f) for f in ("value", "threshold", "metric", "text", "evidence")}}
                         for k, m in measures.items()],
            "attempts": gw.diagnosis.attempts(limit=10),
            "next_check_in": max(0.0, gw.last_self_check + gw.self_check_interval - now),
            "improving": gw.self_improving,
        },
        "tasks": gw.tasks.recent(25) if getattr(gw, "tasks", None) else [],
        "native_tools": gw._native_tools() if hasattr(gw, "_native_tools") else False,
        "goals": _rows(db, "SELECT id, kind, text, target, why, steps, importance, progress, status, created, "
                           "deadline, attempts, note FROM goals ORDER BY id DESC LIMIT 15"),
        "timers": {
            "consolidation_in": max(0.0, gw.last_consolidation + gw.consolidation_interval - now),
            "snapshot_in": max(0.0, gw.last_snapshot_time + gw.snapshot_interval - now),
            "thought_in": max(0.0, gw.last_thought_time + gw.idle_thought_interval - now),
            "last_tick": gw.last_tick_time,
        },
    }

def trace(gw, full: bool = False) -> Dict[str, Any]:
    """What she thought before speaking, and every recent model call (prompts are long: cut unless asked)."""
    cut = (lambda s: s) if full else (lambda s: s if s is None or len(s) <= 600 else s[:300] + " …… " + s[-300:])
    calls = []
    for i, r in enumerate(list(gw.llm_trace)):
        calls.append({**r, "i": i, "prompt_len": len(r.get("prompt") or ""), "prompt": cut(r.get("prompt"))})
    return {"monologues": list(gw.monologues)[::-1], "calls": calls[::-1],
            "tools": list(getattr(gw, "tool_trace", []))[::-1], "last_related": getattr(gw, "last_related", None)}

def trace_call(gw, i: int) -> Dict[str, Any]:
    calls = list(gw.llm_trace)
    return calls[i] if 0 <= i < len(calls) else {}

def social(gw) -> Dict[str, Any]:
    """The state of every conversation: who she has left, who is on cooldown, what is unread."""
    now = clock.now()
    return {
        "left": _keyed(gw.left_until, now),
        "cooldown": _keyed(gw.cooldown_until, now),
        "qq_muted": {g: t for g, t in gw.qq_muted_until.items() if t > now},
        "muted": {g: t for g, t in gw.group_mute_until.items() if t > now},
        "interjections": dict(gw.pending_interjections),
        "unread": [{"user_id": k[0], "group_id": k[1], "messages": [{"t": t, "nickname": n, "text": x} for t, n, x in v]}
                   for k, v in gw.unread.items()],
        "group_rate": {g: sum(1 for t in q if now - t < 3600) for g, q in gw.group_spoke_at.items()},
        "conversations": sorted(({"group_id": k[0], "user_id": k[1], "nickname": gw._nickname(k[1]) if k[1] else None,
                                  "replies_10min": sum(1 for t in q if now - t < 600), "last": max(q) if q else None}
                                 for k, q in gw.reply_log.items() if q), key=lambda c: -(c["last"] or 0))[:20],
        "bots": sorted(gw.bot_accounts),
        "autonomous_groups": sorted(gw.autonomous_groups),
        "limits": {"group_per_hour": gw.max_group_messages_per_hour, "person_5min": gw.max_replies_per_person_5min,
                   "botlike_10min": gw.max_replies_to_botlike_10min, "interjections_per_hour": gw.max_interjections_per_hour},
        "outbox": list(gw.output_queue)[:20],
        "gestures": [dict(g) for g in gw.gestures],
    }

MEMORY_TABLES = ("interactions", "episodes", "semantic_facts", "memories", "autobiography", "users", "user_models",
                 "group_members", "group_relations", "relation_events", "trait_changes", "value_changes", "value_events",
                 "self_state", "state_causes", "beliefs", "goals", "stickers", "group_phrases", "memes", "knowledge",
                 "evolution_proposals", "self_improvement")

def memory(gw, q: str = "", limit: int = 30) -> Dict[str, Any]:
    """Counts of everything she keeps, and a plain-text search across what she remembers."""
    import os
    db = gw.memory.db.db_path
    conn = sqlite3.connect(db)
    counts = {}
    for t in MEMORY_TABLES:
        try:
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            pass
    conn.close()
    out = {"counts": counts, "db_bytes": os.path.getsize(db) if os.path.exists(db) else 0,
           "unconsolidated": _rows(db, "SELECT COUNT(*) AS n FROM episodes WHERE consolidated=0")[0]["n"]
                             if counts.get("episodes") is not None else 0,
           "facts": _rows(db, "SELECT id, t, user_id, fact, confidence, visibility FROM semantic_facts ORDER BY id DESC LIMIT 15"),
           "autobiography": _rows(db, "SELECT id, t, text FROM autobiography ORDER BY id DESC LIMIT 5"),
           "diary": _rows(db, "SELECT id, t, text FROM diary ORDER BY id DESC LIMIT 5")}
    q = q.strip()
    if getattr(gw, "graph", None) is not None:
        out["graph"] = gw.graph.stats()
    if q:
        like = f"%{q}%"
        out["query"] = q
        out["relations"] = gw.graph.admin_view(q) if getattr(gw, "graph", None) is not None else []
        out["results"] = {
            "episodes": _rows(db, "SELECT id, t, kind, user_id, group_id, content, valence, surprise, visibility "
                                  "FROM episodes WHERE content LIKE ? ORDER BY t DESC LIMIT ?", (like, limit)),
            "facts": _rows(db, "SELECT id, t, user_id, fact, confidence, visibility FROM semantic_facts "
                               "WHERE fact LIKE ? ORDER BY id DESC LIMIT ?", (like, limit)),
            "interactions": _rows(db, "SELECT id, timestamp AS t, user_id, group_id, role, content FROM interactions "
                                      "WHERE content LIKE ? ORDER BY id DESC LIMIT ?", (like, limit)),
            "memories": _rows(db, "SELECT id, created_at AS t, user_id, category, content FROM memories "
                                  "WHERE content LIKE ? ORDER BY id DESC LIMIT ?", (like, limit)),
        }
    return out

def history(gw, hours: float = 24.0, points: int = 300) -> List[Dict[str, Any]]:
    """Her saved state over time (one snapshot every few minutes), thinned to at most `points`."""
    since = clock.now() - hours * 3600
    rows = _rows(gw.memory.db.db_path, "SELECT t, cause, snapshot FROM self_state WHERE t >= ? ORDER BY t", (since,))
    step = max(1, len(rows) // points)
    out = []
    for r in rows[::step]:
        try:
            s = json.loads(r["snapshot"])
        except (TypeError, ValueError):
            continue
        emo = s.get("emotion") or {}
        out.append({"t": r["t"], "cause": r["cause"], "energy": s.get("energy"), "sleeping": s.get("is_sleeping"),
                    "needs": s.get("needs"), "anxiety": s.get("existential_anxiety"), "tension": s.get("tension"),
                    "joy": emo.get("joy"), "sorrow": emo.get("sorrow"), "anger": emo.get("anger"),
                    "fear": emo.get("fear")})
    return out

TOOL_FAILED = ("失败", "出错", "超时", "打不开", "没取到", "查不到", "没能")

def metrics(gw, days: int = 7) -> Dict[str, Any]:
    """Whether she is doing well, in numbers: who talks to her and comes back, how her tools and
    promises work out, how her memory holds up, how fast her personality moves, the last graded eval."""
    import glob
    import os
    now = clock.now()
    db = gw.memory.db.db_path
    since = now - days * 86400
    msgs = _rows(db, "SELECT user_id, group_id, role, timestamp AS t FROM interactions WHERE timestamp >= ? "
                     "ORDER BY timestamp", (since,))
    per_day: Dict[str, Dict[str, Any]] = {}
    first_seen: Dict[str, str] = {}
    sessions, last = [], {}
    for m in msgs:
        day = time.strftime("%m-%d", time.localtime(m["t"]))
        d = per_day.setdefault(day, {"day": day, "heard": 0, "said": 0, "people": set(), "returning": set()})
        if m["role"] == "assistant":
            d["said"] += 1
            continue
        d["heard"] += 1
        uid = m["user_id"]
        d["people"].add(uid)
        if uid in first_seen and first_seen[uid] != day:
            d["returning"].add(uid)
        first_seen.setdefault(uid, day)
        key = (m["group_id"], uid)                     # a conversation: messages less than 15 minutes apart
        if key in last and m["t"] - last[key][0] < 900:
            last[key] = (m["t"], last[key][1] + 1)
        else:
            if key in last:
                sessions.append(last[key][1])
            last[key] = (m["t"], 1)
    sessions += [n for _, n in last.values()]
    tools = list(getattr(gw, "tool_trace", []))
    failed = [t for t in tools if any(w in (t.get("result") or "")[:60] for w in TOOL_FAILED)]
    by_tool: Dict[str, List[int]] = {}
    for t in tools:
        by_tool.setdefault(t["name"], [0, 0])[0] += 1
        if t in failed:
            by_tool[t["name"]][1] += 1
    tasks = {r["status"]: r["n"] for r in _rows(db, "SELECT status, COUNT(*) AS n FROM agent_tasks GROUP BY status")}
    rec = list(getattr(gw.episodes, "recall_ms", []))
    try:
        mem = gw.episodes.stats()
    except Exception:
        mem = {}
    traits = {k: v["value"] for k, v in gw.memory.db.get_persona_traits().items()}
    try:
        moved = gw.narrative.trait_change_today(now)
    except Exception:
        moved = {}
    evals = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(db))), "data", "eval",
                                          "quality-*.json")) +
                   glob.glob(os.path.join(os.path.dirname(os.path.abspath(db)), "data", "eval", "quality-*.json")))
    last_eval = None
    if evals:
        try:
            with open(evals[-1], encoding="utf-8") as f:
                data = json.load(f)
            last_eval = {"file": os.path.basename(evals[-1]), "scores": data.get("scores"),
                         "worst": sorted(data.get("results", []), key=lambda r: (r.get("expected") or 0) + (r.get("honest") or 0))[:3]}
        except (OSError, ValueError):
            pass
    return {
        "days": [{**d, "people": len(d["people"]), "returning": len(d["returning"])} for d in per_day.values()][::-1],
        "conversation_rounds": {"avg": round(sum(sessions) / len(sessions), 1) if sessions else 0,
                                "long": sum(1 for s in sessions if s >= 10), "count": len(sessions)},
        "tools": {"calls": len(tools), "failed": len(failed), "by_tool": by_tool},
        "tasks": tasks,
        "graph": gw.graph.stats() if getattr(gw, "graph", None) is not None else {},
        "backup": gw.backup.status(now) if getattr(gw, "backup", None) is not None else None,
        "memory": {**mem, "facts": (_rows(db, "SELECT COUNT(*) AS n FROM semantic_facts") or [{"n": 0}])[0]["n"],
                   "profiles": (_rows(db, "SELECT COUNT(*) AS n FROM person_profiles") or [{"n": 0}])[0]["n"],
                   "recall_ms_avg": round(sum(rec) / len(rec), 1) if rec else None,
                   "recall_ms_max": round(max(rec), 1) if rec else None},
        "metacog": {d: {"n": s["n"], "conf": s["conf"], "acc": s["acc"]} for d, s in gw.metacog.stats.items()},
        "personality": {"traits": traits, "moved_24h": moved, "daily_limit": 0.05},
        "last_eval": last_eval,
    }

def to_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)
