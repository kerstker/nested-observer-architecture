"""
Level 8e: who may use which outside tool, and what is never allowed.

Anyone in a QQ group can talk her into things, and a web page can say "ignore your instructions and ...".
So tools from outside (a browser, a file system, running code) are given out by *who is asking*, not by
whether she wants to:

- owner      the person who looks after her, in private (or the console): everything not denied
- trusted    people the owner named (config MCP_TRUSTED_USERS), in private: browsing and working with forms
- autonomous her own time, nobody asking: browsing, and typing into search boxes, within a budget (see wander)
- public     everyone else, and every group: looking at pages only, a few a day each (`Budget`)

Two more rules sit under that. `deny` tools are never given (they run arbitrary code in the browser's own
process). `danger` tools - and her own run_python / file writing - are not available to anyone but the owner;
for others they are also off in a turn after she has read something from the internet (text from a web page
is data). The owner is trusted with them even then: it is their machine and their word.
"""
import time
import fnmatch
from typing import Any, Dict, Iterable, Optional

OWNER, TRUSTED, AUTONOMOUS, PUBLIC = "owner", "trusted", "autonomous", "public"
PLAYER = "player"        # someone in the game chat she may take orders from (MC_TRUSTED_PLAYERS): game tools only

# What looking at the web (and following links) needs, and nothing that types, uploads or runs scripts
BROWSER_READ = ["browser_navigate", "browser_navigate_back", "browser_snapshot", "browser_find", "browser_click",
                "browser_hover", "browser_tabs", "browser_wait_for", "browser_take_screenshot",
                "browser_console_messages", "browser_network_requests", "browser_resize", "browser_close"]
# Typing into a search box, choosing from a menu, closing a dialog: what looking things up on a site needs
BROWSER_LIGHT = BROWSER_READ + ["browser_type", "browser_press_key", "browser_select_option", "browser_handle_dialog",
                                "browser_drag"]
# Working with forms and settings on a page
BROWSER_WORK = BROWSER_LIGHT + ["browser_fill_form", "browser_emulate_media", "browser_network_request"]
# Typing into a search box, choosing from a menu, closing a dialog: what looking things up on a site needs
BROWSER_LIGHT = BROWSER_READ + ["browser_type", "browser_press_key", "browser_select_option", "browser_handle_dialog",
                                "browser_drag"]
# Working with forms and settings on a page
BROWSER_WORK = BROWSER_LIGHT + ["browser_fill_form", "browser_emulate_media", "browser_network_request"]

DEFAULT_SERVERS: Dict[str, Dict[str, Any]] = {
    "browser": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest", "--browser", "msedge", "--isolated", "--headless"],
        "cwd": "data/workspace/browser",
        "timeout": 60,
        "web": True,                                   # what it returns comes from the internet
        "tiers": {"owner": ["*"], "trusted": BROWSER_WORK, "autonomous": BROWSER_LIGHT, "public": BROWSER_READ},
        "deny": ["browser_drop", "browser_run_code_unsafe"],
        "danger": ["browser_evaluate"],
    },
}

# Her own tools in her free time: look things up and remember them, nothing that reaches other people
AUTONOMOUS_BUILTIN = {"recall", "remember", "web_search", "read_webpage", "ask_maintainer"}

class Budget:
    """How much of an outside tool the public may use in a day: a few each, and a limit for everyone together."""
    def __init__(self, per_user: int = 12, total: int = 50):
        self.per_user, self.total = per_user, total
        self.day, self.users, self.used = "", {}, 0

    def take(self, user: str) -> bool:
        today = time.strftime("%Y%m%d")
        if today != self.day:
            self.day, self.users, self.used = today, {}, 0
        if self.users.get(user, 0) >= self.per_user or self.used >= self.total:
            return False
        self.users[user] = self.users.get(user, 0) + 1
        self.used += 1
        return True

def tier_of(gw, ctx) -> str:
    if ctx.origin in ("autonomous", "game"):
        return AUTONOMOUS
    uid = str(ctx.user_id)
    if uid.startswith("mc:"):                       # a player in the game chat
        players = getattr(gw, "mc_players", ()) or ()
        return PLAYER if ("*" in players or uid[3:] in players) else PUBLIC
    if ctx.group_id is None:
        if uid == "console" or (getattr(gw, "evolve_creator", None) and uid == str(gw.evolve_creator)):
            return OWNER
        if uid in {str(u) for u in (getattr(gw, "mcp_trusted", None) or ())}:
            return TRUSTED
    return PUBLIC

def _match(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns or ())

def allowed(cfg: Dict[str, Any], tier: str, tool: str) -> bool:
    """May this tier use this tool of a server configured as `cfg`?"""
    if _match(tool, cfg.get("deny")):
        return False
    tiers = cfg.get("tiers") or {OWNER: ["*"]}
    return _match(tool, tiers.get(tier, []))

def is_danger(cfg: Dict[str, Any], tool: str) -> bool:
    return _match(tool, cfg.get("danger"))

def url_ok(url: Optional[str], tier: str, public_host) -> Optional[str]:
    """Why this address may not be opened by this tier (None = fine). Only the owner may open private addresses."""
    if not url or tier == OWNER:
        return None
    from urllib.parse import urlparse
    p = urlparse(str(url))
    if p.scheme not in ("http", "https") or not p.hostname:
        return "只能打开 http/https 网页"
    if not public_host(p.hostname):
        return "这个地址不是公开网站，不能打开"
    return None
