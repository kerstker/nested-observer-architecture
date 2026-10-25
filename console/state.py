import json
import sqlite3
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
    }

def to_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)
