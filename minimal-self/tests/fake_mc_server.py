"""A stand-in for the Minecraft MCP servers: chat is read from / written to files in $NOA_FAKE_MC.

$NOA_FAKE_MC_MODE=mcpfabric speaks like the mcpfabric bridge (JSON chat events, snake_case tools, admin tools
present); anything else speaks like the mineflayer server.
"""
import json
import os
import sys

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
DIR = os.environ["NOA_FAKE_MC"]
FABRIC = os.environ.get("NOA_FAKE_MC_MODE") == "mcpfabric"
NAMES = (("get_self", "Get local player state"), ("get_recent_chat", "Get recent chat"), ("send_chat", "Send chat"),
         ("navigate_to", "Walk somewhere"), ("break_block", "Break a block"), ("screenshot", "Take a screenshot"),
         ("run_command", "Run a command"), ("give_item", "Give an item"), ("set_block", "Set a block"),
         ("supervised_command", "Supervised command")) if FABRIC else (
    ("get-position", "Get the current position of the bot"), ("read-chat", "Get recent chat messages from players"),
    ("send-chat", "Send a chat message in-game"), ("move-to-position", "Move the bot"), ("dig-block", "Dig a block"),
    ("find-entity", "Find the nearest entity"))
TOOLS = [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": {}}} for n, d in NAMES]


def reply(rid, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\n")
    sys.stdout.flush()


def text(t):
    return {"content": [{"type": "text", "text": t}]}


def chat_rows():
    path = os.path.join(DIR, "chat.txt")
    return open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []


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
            rows = chat_rows()[-int(args.get("count", 10)):]
            body = "".join(f"{i}. {r}\n" for i, r in enumerate(rows, 1))
            reply(rid, text(f"Found {len(rows)} chat message(s):\n\n{body}"))
        elif name == "get_recent_chat":
            events = []
            for i, row in enumerate(chat_rows(), 1):       # "stamp - Player: text"
                _, _, rest = row.partition(" - ")
                player, _, said = rest.partition(": ")
                events.append({"id": i, "type": "chat", "gameTime": i * 20,
                               "data": {"text": f"<{player}> {said}", "sender": player}})
            events.append({"id": 999, "type": "system_message", "gameTime": 0, "data": {"text": "Steve joined the game"}})
            reply(rid, text(json.dumps({"messages": events}, ensure_ascii=False, indent=2)))
        elif name in ("send-chat", "send_chat"):
            with open(os.path.join(DIR, "sent.txt"), "a", encoding="utf-8") as f:
                f.write(args.get("message", "") + "\n")
            reply(rid, text(f'Sent message: "{args.get("message", "")}"'))
        elif name in ("get-position", "get_self"):
            reply(rid, text("Position: x=10, y=64, z=-3"))
        else:
            reply(rid, text(f"{name} ok"))
