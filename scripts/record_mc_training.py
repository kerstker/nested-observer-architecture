"""Read-only MC baseline and action recorder, independent of livestream services.

Only GETs the authenticated local console and reads the existing public snapshot.
No game commands, chat, database access, or live configuration changes.
"""
import argparse
import collections
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def public_sample(game, snapshot, now):
    body = game.get("body") or {}
    stamp = snapshot.get("t")
    fresh = isinstance(stamp, (int, float)) and 0 <= now - stamp <= 3
    visible = (snapshot.get("body") or {}) if fresh else {}
    pos = visible.get("pos") or {}
    position = {k: pos[k] for k in ("x", "y", "z") if isinstance(pos.get(k), (int, float)) and math.isfinite(pos[k])}
    return {"t": now, "connected": game.get("connected") is True, "snapshot_fresh": fresh,
            "snapshot_age": round(now - stamp, 3) if isinstance(stamp, (int, float)) else None,
            "body": {k: body.get(k) for k in ("running", "busy", "action", "tick_rate", "rpc_errors", "rpc_p95_ms", "ticks")},
            "state": {**{k: visible.get(k) for k in ("health", "food", "dimension", "onGround")}, "pos": position},
            "training_available": isinstance(game.get("training"), dict),
            "quests_completed": (game.get("quests") or {}).get("completed")}


def summarize(samples, actions, errors):
    rates = [s["body"]["tick_rate"] for s in samples if isinstance(s["body"].get("tick_rate"), (int, float))]
    fresh = [s for s in samples if s["snapshot_fresh"]]
    health = [s["state"]["health"] for s in fresh if isinstance(s["state"].get("health"), (int, float))]
    counts = collections.Counter(a.get("status", "unverified") for a in actions)
    judged = sum(counts[k] for k in ("completed", "partial", "failed", "interrupted"))
    # A stationary walk is a review candidate, never an automatic diagnosis.
    candidates, anchor = [], None
    for s in samples:
        pos = s["state"].get("pos") or {}
        if (not s["connected"] or not s["snapshot_fresh"] or s["body"].get("action") != "walk"
                or len(pos) != 3):
            anchor = None
            continue
        p = tuple(pos[k] for k in ("x", "y", "z"))
        if (anchor is None or s["state"]["dimension"] != anchor[2] or math.dist(p, anchor[1]) > .75):
            anchor = (s["t"], p, s["state"]["dimension"])
        elif s["t"] - anchor[0] >= 15:
            candidates.append({"start": anchor[0], "end": s["t"], "kind": "stationary_walk_review"})
            anchor = (s["t"], p, s["state"]["dimension"])
    return {"schema": 1, "read_only": True, "samples": len(samples),
            "connected_samples": sum(s["connected"] for s in samples), "fresh_samples": len(fresh),
            "body_hz_median": statistics.median(rates) if rates else None,
            "minimum_observed_health": min(health) if health else None,
            "actions_observed": len(actions), "action_statuses": dict(counts),
            "completion_rate": counts["completed"] / len(actions) if actions else None,
            "verdict_coverage": judged / len(actions) if actions else None,
            "training_available": any(s["training_available"] for s in samples),
            "review_candidates": candidates, "errors": dict(collections.Counter(errors)),
            "qualification": "not_assessed", "note": "Observation is not a controlled skill exam or proof of expert ability."}


def run(seconds, interval, destination):
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    headers = {"X-Console-Token": config["CONSOLE_TOKEN"]} if config.get("CONSOLE_TOKEN") else {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    samples, actions, errors, seen = [], [], [], set()
    recording_started = time.time()
    deadline = time.monotonic() + seconds
    with (destination / "samples.jsonl").open("w", encoding="utf-8") as stream, (destination / "actions.jsonl").open("w", encoding="utf-8") as action_stream:
        while time.monotonic() < deadline:
            started = time.monotonic()
            game, snapshot = {}, {}
            try:
                request = urllib.request.Request("http://127.0.0.1:8080/console/api/outside", headers=headers)
                with opener.open(request, timeout=4) as response:
                    game = json.load(response).get("game") or {}
            except Exception as exc:
                errors.append("console:" + type(exc).__name__)
            try:
                snapshot = json.loads((ROOT / "live/overlay/game_state.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append("snapshot:" + type(exc).__name__)
            sample = public_sample(game, snapshot, time.time())
            samples.append(sample)
            stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            training = game.get("training") or {}
            for event in training.get("history", []):
                if event.get("ended_at", 0) < recording_started:
                    continue
                key = (event.get("id"), event.get("started_at"))
                if key in seen:
                    continue
                seen.add(key)
                # The producer exports a game-only contract. Do not export other console fields.
                event = {k: event[k] for k in ("id", "name", "args", "started_at", "ended_at", "before", "after", "status", "completed", "supported", "has_progress", "reason_code", "evidence", "work_reason") if k in event}
                actions.append(event)
                action_stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                action_stream.flush()
            time.sleep(max(0, min(interval - (time.monotonic() - started), deadline - time.monotonic())))
    report = summarize(samples, actions, errors)
    (destination / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["connected_samples"] and report["fresh_samples"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing sessions are never overwritten")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600 or not .5 <= args.interval <= 30:
        parser.error("seconds must be 1..3600 and interval 0.5..30")
    raise SystemExit(run(args.seconds, args.interval, args.output))
