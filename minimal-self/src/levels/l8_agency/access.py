"""
Level 8e: who may use which outside tool, and what is never allowed.

Anyone in a QQ group can talk her into things, and a web page can say "ignore your instructions and ...".
So tools from outside (a browser, a file system, running code) are given out by *who is asking*, not by
whether she wants to:

- owner      the person who looks after her, in private (or the console): everything not denied
- trusted    people the owner named (config MCP_TRUSTED_USERS), in private: the read-only set
- autonomous her own time, nobody asking: the read-only set, within a budget (see wander)
- public     everyone else, and every group: no outside tools at all

Two more rules sit under that. `deny` tools are never given (they can run arbitrary code in the page or
move files in and out of the machine). `danger` tools - and her own run_python / file writing - are not
available in a turn after she has read something from the internet: text from a web page is data, and if
it were able to steer her into running code, nobody should have let it get that far.
"""
import fnmatch
from typing import Any, Dict, Iterable, Optional

OWNER, TRUSTED, AUTONOMOUS, PUBLIC = "owner", "trusted", "autonomous", "public"

# What looking at the web (and following links) needs, and nothing that types, uploads or runs scripts
BROWSER_READ = ["browser_navigate", "browser_navigate_back", "browser_snapshot", "browser_find", "browser_click",
                "browser_hover", "browser_tabs", "browser_wait_for", "browser_take_screenshot",
                "browser_console_messages", "browser_network_requests", "browser_resize", "browser_close"]

DEFAULT_SERVERS: Dict[str, Dict[str, Any]] = {
    "browser": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest", "--browser", "msedge", "--isolated", "--headless"],
        "cwd": "data/workspace/browser",
        "timeout": 60,
        "web": True,                                   # what it returns comes from the internet
        "tiers": {"owner": ["*"], "trusted": BROWSER_READ, "autonomous": BROWSER_READ, "public": []},
        "deny": ["browser_file_upload", "browser_drop", "browser_run_code_unsafe"],
        "danger": ["browser_evaluate"],
    },
}

# Her own tools in her free time: look things up and remember them, nothing that reaches other people
AUTONOMOUS_BUILTIN = {"recall", "remember", "web_search", "read_webpage", "ask_maintainer"}

def tier_of(gw, ctx) -> str:
    if ctx.origin == "autonomous":
        return AUTONOMOUS
    uid = str(ctx.user_id)
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
