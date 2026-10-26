"""
Level 8c: hands outside the house - tools from MCP servers (the Model Context Protocol).

A MCP server is a program that offers tools over stdin/stdout (a real browser, a file system, a database
...). The hub starts the ones listed in config, asks each what it can do, and lets the toolbox offer those
tools to her next to her own. It speaks just enough of the protocol (JSON-RPC 2.0, newline-delimited) to do
that, so nothing beyond the standard library is needed.

Who may use what is NOT decided here - see `trust.py`-style rules in the toolbox: the hub only knows how to
start servers, list tools and call them, with timeouts, restart after a crash, and results cut to size.
"""
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROTOCOL = "2024-11-05"
MAX_RESULT = 6000

class McpError(Exception):
    pass

class McpServer:
    """One server process. Calls are serialised (a browser has one page in front of it)."""

    def __init__(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None, cwd: str = None,
                 timeout: float = 45.0):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None
        self.tools: List[Dict[str, Any]] = []
        self.error = ""
        self._id = 0
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()                # one call at a time
        self._cond = threading.Condition()
        self.started_at = 0.0
        self.calls = 0

    # --- process ---------------------------------------------------------------------

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        exe = shutil.which(self.command) or self.command
        env = {**os.environ, **self.env}
        self.proc = subprocess.Popen([exe, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, cwd=self.cwd, env=env, bufsize=0,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._pending.clear()
        threading.Thread(target=self._read_loop, args=(self.proc,), daemon=True).start()
        init = self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                            "clientInfo": {"name": "noa", "version": "1"}}, timeout=max(self.timeout, 90.0))
        self._notify("notifications/initialized")
        listing = self._request("tools/list", {})
        self.tools = listing.get("tools") or []
        self.started_at = time.time()
        self.error = ""
        logger.info(f"[MCP] {self.name}: {len(self.tools)} tools ({(init.get('serverInfo') or {}).get('name', '?')})")

    def stop(self):
        proc, self.proc = self.proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    def _read_loop(self, proc):
        try:
            for line in iter(proc.stdout.readline, b""):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "id" in msg and ("result" in msg or "error" in msg):
                    with self._cond:
                        self._pending[msg["id"]] = msg
                        self._cond.notify_all()
        except Exception:
            pass
        with self._cond:                                   # the process is gone: wake whoever waits
            if proc is self.proc:
                self._pending[-1] = {"error": {"message": "server exited"}}
            self._cond.notify_all()

    # --- protocol ----------------------------------------------------------------------

    def _send(self, msg: Dict[str, Any]):
        if not self.alive():
            raise McpError("server is not running")
        try:
            self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except OSError:                                  # the pipe is gone: it died just now
            raise McpError("server is not running")

    def _notify(self, method: str, params: Dict[str, Any] = None):
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def _request(self, method: str, params: Dict[str, Any], timeout: float = None) -> Dict[str, Any]:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.time() + (timeout or self.timeout)
        with self._cond:
            while rid not in self._pending:
                if -1 in self._pending:
                    raise McpError("server exited")
                left = deadline - time.time()
                if left <= 0:
                    raise McpError(f"{method} timed out after {timeout or self.timeout:.0f}s")
                self._cond.wait(min(left, 1.0))
            msg = self._pending.pop(rid)
        if "error" in msg:
            raise McpError(str(msg["error"].get("message") or msg["error"]))
        return msg.get("result") or {}

    def call(self, tool: str, arguments: Dict[str, Any], timeout: float = None) -> str:
        with self._lock:
            for attempt in (0, 1):
                try:
                    if not self.alive():
                        self.start()                       # it died (or was never started): start again
                    self.calls += 1
                    result = self._request("tools/call", {"name": tool, "arguments": arguments or {}}, timeout=timeout)
                    break
                except McpError as e:
                    # Only a call that never reached the server is tried again; one that died half way is not
                    # repeated (it might have clicked something) - the next call starts the server afresh
                    if attempt == 0 and str(e) == "server is not running":
                        self.stop()
                        continue
                    raise
        return render_result(result)

def render_result(result: Dict[str, Any]) -> str:
    """MCP content blocks as text. Images and other binaries are noted, not inlined."""
    parts = []
    for block in result.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind == "image":
            parts.append("（返回了一张图片，这里看不到）")
        elif kind == "resource":
            res = block.get("resource") or {}
            parts.append(str(res.get("text") or res.get("uri") or "（资源）"))
        else:
            parts.append(f"（{kind or '未知'}内容）")
    text = "\n".join(p for p in parts if p).strip() or "（没有内容）"
    if result.get("isError"):
        text = "出错了：" + text
    return text[:MAX_RESULT] + ("……（后面省略）" if len(text) > MAX_RESULT else "")

class McpHub:
    """The configured servers. `config` maps a short name to {command, args, env, tiers, ...}."""

    def __init__(self, config: Dict[str, Dict[str, Any]], root: str = "."):
        self.config = {k: v for k, v in (config or {}).items() if isinstance(v, dict) and v.get("command")
                       and v.get("enabled", True)}
        self.servers: Dict[str, McpServer] = {}
        self.root = root
        self.last_error: Dict[str, str] = {}
        self._starting = threading.Lock()
        for name, c in self.config.items():
            self.servers[name] = McpServer(name, c["command"], c.get("args"), c.get("env"),
                                           cwd=c.get("cwd") or root, timeout=float(c.get("timeout", 45)))

    def start_all(self):
        """Start every server (in the background: a browser takes a few seconds, the bot must not wait)."""
        def run():
            for name, srv in self.servers.items():
                try:
                    srv.start()
                except Exception as e:
                    srv.error = str(e)[:200]
                    self.last_error[name] = srv.error
                    logger.error(f"[MCP] {name} failed to start: {e}")
        threading.Thread(target=run, daemon=True).start()

    def stop_all(self):
        for srv in self.servers.values():
            srv.stop()

    def tools(self) -> List[Dict[str, Any]]:
        """Every tool of every running server: {server, name, description, schema}."""
        out = []
        for name, srv in self.servers.items():
            for t in srv.tools:
                out.append({"server": name, "name": t["name"], "description": t.get("description") or "",
                            "schema": t.get("inputSchema") or {"type": "object", "properties": {}}})
        return out

    def call(self, server: str, tool: str, arguments: Dict[str, Any], timeout: float = None) -> str:
        srv = self.servers.get(server)
        if srv is None:
            raise McpError(f"no such server: {server}")
        return srv.call(tool, arguments, timeout)

    def status(self) -> List[Dict[str, Any]]:
        return [{"name": n, "running": s.alive(), "tools": len(s.tools), "calls": s.calls, "error": s.error,
                 "command": f"{s.command} {' '.join(s.args)}"[:120]} for n, s in self.servers.items()]
