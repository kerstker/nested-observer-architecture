"""A stand-in for the Minecraft MCP server: chat is read from / written to files in $NOA_FAKE_MC."""
import json
import os
import sys
import time

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
DIR = os.environ["NOA_FAKE_MC"]
TOOLS = [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": {}}} for n, d in (
    ("get-position", "Get the current position of the bot"), ("read-chat", "Get recent chat messages from players"),
    ("send-chat", "Send a chat message in-game"), ("move-to-position", "Move the bot"), ("dig-block", "Dig a block"),
    ("find-entity", "Find the nearest entity"))]


def reply(rid, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\n")
    sys.stdout.flush()


def text(t):
    return {"content": [{"type": "text", "text": t}]}


for line in sys.stdin:
    msg = json.loads(line)
    method, rid = msg.get("method"), msg.get("id")
    if method == "initialize":
        reply(rid, {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "fake-mc", "version": "0"}})
    elif method == "tools/list":
        reply(rid, {"tools": TOOLS})
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"].get("arguments") or {}
        with open(os.path.join(DIR, "calls.log"), "a", encoding="utf-8") as f:
            f.write(name + " " + json.dumps(args, ensure_ascii=False) + "\n")
        if name == "read-chat":
            path = os.path.join(DIR, "chat.txt")
            rows = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []
            rows = rows[-int(args.get("count", 10)):]
            body = "".join(f"{i}. {r}\n" for i, r in enumerate(rows, 1))
            reply(rid, text(f"Found {len(rows)} chat message(s):\n\n{body}"))
        elif name == "send-chat":
            with open(os.path.join(DIR, "sent.txt"), "a", encoding="utf-8") as f:
                f.write(args.get("message", "") + "\n")
            reply(rid, text(f'Sent message: "{args.get("message", "")}"'))
        elif name == "get-position":
            reply(rid, text("Position: x=10, y=64, z=-3"))
        else:
            reply(rid, text(f"{name} ok"))
