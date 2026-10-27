"""Observed action contracts. Text replies are never completion evidence.

These records describe ordinary game state only; no chat, identities or model journals.
Uncovered skills stay unverified until they have their own postcondition.
"""
import math
import time


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def observe(rpc):
    observation = {"t": time.time(), "state_valid": False, "inventory_valid": False}
    for method in ("player.getState", "player.getInventory"):
        try:
            raw = rpc(method, {})
        except Exception:
            raw = None
        if not isinstance(raw, dict):
            continue
        if method == "player.getState":
            observation["state_valid"] = all(number(raw.get(k)) for k in ("x", "y", "z")) and bool(raw.get("dimension"))
            for k in ("x", "y", "z", "health", "food"):
                if number(raw.get(k)):
                    observation[k] = raw[k]
            if isinstance(raw.get("dimension"), str):
                observation["dimension"] = raw["dimension"]
        elif isinstance(raw.get("hotbar"), list) and isinstance(raw.get("main"), list):
            counts = {}
            armor = raw.get("armor") or []
            if not isinstance(armor, list):
                continue
            rows = raw["hotbar"] + raw["main"] + armor
            if isinstance(raw.get("offhand"), dict):
                rows.append(raw["offhand"])
            valid = True
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not number(row.get("count")):
                    valid = False
                    break
                if row["count"] > 0 and row["id"] != "minecraft:air":
                    counts[row["id"]] = counts.get(row["id"], 0) + int(row["count"])
            observation["inventory_valid"] = valid
            if valid:
                observation["inventory"] = counts
    observation["observed_at"] = time.time()
    return observation


def evaluate(name, args, before, after, *, progress=0, stopped=False, error=False, reason="", facts=()):
    """Verify supported postconditions, keeping progress, interruptions and uncertainty distinct."""
    evidence = {}
    complete, supported, observed, advanced = False, False, False, False
    same_world = (before.get("state_valid") and after.get("state_valid")
                  and before.get("dimension") == after.get("dimension"))
    inventory_valid = bool(same_world and before.get("inventory_valid") and after.get("inventory_valid"))
    if inventory_valid:
        old, new = before["inventory"], after["inventory"]
        delta = {i: new.get(i, 0) - old.get(i, 0) for i in old.keys() | new.keys()
                 if new.get(i, 0) != old.get(i, 0)}
        evidence["inventory_delta"] = delta
    else:
        delta = {}
    if name in ("craft", "goal"):
        supported = isinstance(args.get("item"), str) and bool(args["item"]) and not args.get("clear")
        observed = inventory_valid
        item = args.get("item") if supported else ""
        if name == "craft" and item and ":" not in item:
            item = "minecraft:" + item
        requested = args.get("count", 1)
        count = max(1, int(requested)) if number(requested) else 1
        supported = supported and (requested is None or number(requested))
        amount = (after.get("inventory", {}).get(item, 0) if name == "goal" else delta.get(item, 0))
        evidence.update(target_item=item, target_count=count, observed_count=amount if observed else None)
        complete = supported and observed and amount >= count
        advanced = observed and delta.get(item, 0) > 0
        if name == "goal":
            steps = [f for f in facts if f.get("kind") == "goal_step" and f.get("goal") == item]
            stations = [f for f in facts if f.get("kind") == "station_available" and f.get("item") == item]
            evidence.update(verified_steps=steps, observed_stations=stations)
            advanced = advanced or bool(observed and steps)
            complete = complete or bool(supported and same_world and count <= 1 and stations)
    elif name in ("go_to", "go_home"):
        supported = all(number(args.get(k)) for k in ("x", "y", "z"))
        observed = bool(same_world and args.get("dimension", before.get("dimension")) == after.get("dimension"))
        if supported and observed:
            horizontal = math.hypot(after["x"] - args["x"], after["z"] - args["z"])
            vertical = abs(after["y"] - args["y"])
            evidence.update(horizontal_error=horizontal, vertical_error=vertical)
            complete = horizontal <= (8 if name == "go_home" else 3) and vertical <= 2.5
            target = tuple(args[k] for k in ("x", "y", "z"))
            advanced = math.dist(tuple(before[k] for k in ("x", "y", "z")), target) > math.dist(tuple(after[k] for k in ("x", "y", "z")), target) + .5
    elif name == "place":
        supported = bool(args.get("item"))
        observed = bool(same_world)
        matches = [f for f in facts if f.get("kind") == "placed" and f.get("item") == args.get("item")
                   and all(args.get(k) is None or (number(args[k]) and f.get(k) == int(args[k])) for k in ("x", "y", "z"))]
        complete = observed and bool(matches)
        evidence["placements"] = matches
    elif name == "collect_drops":
        # Inventory gain proves collection, but does not prove every nearby drop was collected.
        supported, observed = True, inventory_valid
        advanced = any(n > 0 for n in delta.values())
        evidence["collected_items"] = {i: n for i, n in delta.items() if n > 0}
    # Unrelated loot or movement cannot count as progress toward a supported contract.
    has_progress = advanced or (not supported and progress > 0)
    if error:
        status, why = "failed", "execution_error"
    elif stopped:
        status, why = "interrupted", "cancelled"
    elif complete:
        status, why = "completed", "postcondition_observed"
    elif supported and not observed:
        status, why = "unverified", "observation_unavailable_or_world_changed"
    elif has_progress:
        status, why = "partial", "progress_without_completion"
    elif supported:
        status, why = "failed", "postcondition_not_met"
    else:
        status, why = "unverified", "contract_not_implemented"
    return {"status": status, "completed": status == "completed", "supported": supported,
            "has_progress": bool(has_progress), "reason_code": why, "evidence": evidence,
            "work_reason": str(reason)[:180]}
