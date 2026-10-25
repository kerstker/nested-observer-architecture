"""
Level 17: the channel through which Noa may change her own code.

Design constraints (these are the whole point of the module):

* The agent never works in the live tree. A throwaway copy of the repository is made,
  the agent edits the copy, and the only thing that comes back is a patch.
* Nothing is applied automatically. `apply()` is a separate, explicit step (console /
  creator), and the running process picks the change up on the next restart.
* The agent runs with a stripped environment: no DeepSeek key, no NapCat token, no console
  token. If it leaks its own context, it leaks nothing that can be used against her.
* Failures never raise into the conversation loop: they come back as a status.

The agent is DeepSeek Harness itself, in its headless ACP profile:
    ELECTRON_RUN_AS_NODE=1 "<DeepSeek Harness.exe>" "<...>/@deepseek-ai/dsh/lib/bin.js" --profile acp
It speaks newline-delimited JSON-RPC (initialize -> session/new -> session/prompt).
"""
import json
import logging
import os
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "__pycache__", "node_modules", "models", ".trae", "data", ".idea", ".venv"}
SKIP_SUFFIXES = (".sqlite", ".sqlite3", ".pyc", ".log", ".gguf", ".jpg", ".png")

DEFAULT_DSH_EXE = r"C:\path\to\DeepSeek Harness.exe"
DEFAULT_DSH_BIN = r"C:\path\to\dsh\node_modules\@deepseek-ai\dsh\lib\bin.js"

PROPOSAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS evolution_proposals (
    id TEXT PRIMARY KEY,
    t REAL,
    requested_by TEXT,
    kind TEXT,
    task TEXT,
    workspace TEXT,
    session_id TEXT,
    status TEXT,
    files_changed INTEGER,
    added INTEGER,
    removed INTEGER,
    tests_total INTEGER,
    tests_failed INTEGER,
    patch_path TEXT,
    report TEXT,
    error TEXT,
    approved_by TEXT,
    applied_at REAL
)
"""

# Everything the agent is told about the rules of the house. Kept short on purpose: the agent
# gets the repository as its workspace and the tests as its feedback loop.
TASK_PREAMBLE = """你在一份「乃爱(Noa)」数字生命项目的副本工作区里工作。这个副本会被丢弃，只有你的 git diff 会被带回审查，所以不要害怕改错，但必须遵守：

1. 只允许修改 minimal-self/src/levels/** 下的代码、以及 console/ 下的界面。禁止修改：config.json、napcat/**、memory.sqlite、data/**、测试文件、以及任何密钥。
2. 不许引入新的第三方依赖。
3. 改动要小：超过 300 行的 diff 会被拒绝。
4. 不要运行 shell 命令、也不要运行测试：这个环境里的 shell 可能不可用。修改完直接用文字说明即可，测试由调用方在副本里统一跑。
5. 不要修改测试来让它变绿。
6. 最后用中文写一段 150 字以内的说明：你改了什么、为什么、需要注意什么。

任务：
"""


def _strip_secrets(env: Dict[str, str]) -> Dict[str, str]:
    out = dict(env)
    for k in list(out):
        upper = k.upper()
        if any(s in upper for s in ("DEEPSEEK_API_KEY", "NAPCAT", "CONSOLE_TOKEN", "OPENAI", "ANTHROPIC")):
            out.pop(k, None)
    out["ELECTRON_RUN_AS_NODE"] = "1"
    out["ELECTRON_NO_ATTACH_CONSOLE"] = "1"
    out.setdefault("PYTHONIOENCODING", "utf-8")
    return out


def _runs_ok(text: str) -> Dict[str, int]:
    """Parse unittest output for the pass/fail counters."""
    total = failed = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Ran ") and " test" in line:
            try:
                total += int(line.split()[1])
            except (IndexError, ValueError):
                pass
        if line.startswith("FAILED ("):
            for part in line[len("FAILED ("):-1].split(","):
                part = part.strip()
                if "=" in part:
                    try:
                        failed += int(part.split("=")[1])
                    except ValueError:
                        pass
    return {"total": total, "failed": failed}


class DshAgentBridge:
    def __init__(self, repo_root: str, data_dir: str = None, dsh_exe: str = None, dsh_bin: str = None,
                 timeout: float = 1200.0, max_patch_lines: int = 300, max_per_day: int = 1):
        self.repo_root = os.path.abspath(repo_root)
        self.data_dir = data_dir or os.path.join(self.repo_root, "data", "evolution")
        self.dsh_exe = dsh_exe or os.environ.get("DSH_EXE") or DEFAULT_DSH_EXE
        self.dsh_bin = dsh_bin or os.environ.get("DSH_BIN") or DEFAULT_DSH_BIN
        self.timeout = timeout
        self.max_patch_lines = max_patch_lines
        self.max_per_day = max_per_day
        os.makedirs(self.data_dir, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ storage
    def _db(self):
        conn = sqlite3.connect(os.path.join(self.repo_root, "memory.sqlite"))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._db()
        try:
            conn.execute(PROPOSAL_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def list_proposals(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._db()
        try:
            rows = conn.execute("SELECT * FROM evolution_proposals ORDER BY t DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def proposals_today(self) -> int:
        day_start = time.time() - (time.time() % 86400)
        conn = self._db()
        try:
            return conn.execute("SELECT COUNT(*) FROM evolution_proposals WHERE t >= ?", (day_start,)).fetchone()[0]
        finally:
            conn.close()

    # ------------------------------------------------------------------ workspace
    def _copy_repo(self, dest: str):
        dest_abs = os.path.abspath(dest)

        def ignore(directory, names):
            skipped = {n for n in names if n in SKIP_DIRS or n.endswith(SKIP_SUFFIXES)}
            # Never descend into the directory the copy itself is being written to.
            for n in names:
                full = os.path.abspath(os.path.join(directory, n))
                if full == dest_abs or dest_abs.startswith(full + os.sep):
                    skipped.add(n)
            return skipped

        shutil.copytree(self.repo_root, dest_abs, ignore=ignore)

    def _git(self, cwd: str, *args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120)
        if proc.returncode != 0 and args[0] not in ("diff", "init", "add", "commit"):
            raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr[:400]}")
        return proc.stdout

    # ------------------------------------------------------------------ agent call
    def _call_agent(self, workspace: str, prompt: str) -> Dict[str, Any]:
        """Run DSH in its ACP profile; returns {'session_id', 'text', 'notifications'} or raises."""
        cmd = [self.dsh_exe, self.dsh_bin, "--profile", "acp"]
        env = _strip_secrets(os.environ)
        proc = subprocess.Popen(cmd, cwd=workspace, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        lines: "queue.Queue[str]" = queue.Queue()

        def pump():
            try:
                for raw in proc.stdout:
                    lines.put(raw.rstrip("\n"))
            finally:
                lines.put("")

        threading.Thread(target=pump, daemon=True).start()

        state = {"session_id": None, "texts": []}

        def send(msg: Dict[str, Any]):
            proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            proc.stdin.flush()

        def wait_for(mid: int, timeout: float) -> Optional[Dict[str, Any]]:
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    line = lines.get(timeout=max(0.1, min(1.0, deadline - time.time())))
                except queue.Empty:
                    continue
                if not line:
                    return None
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == mid:
                    return msg
                # ACP progress/final text arrives as session/update notifications:
                #   {"method":"session/update","params":{"update":{"sessionUpdate":"agent_message_chunk",
                #                                                    "content":{"type":"text","text":"..."}}}}
                if msg.get("method") == "session/update":
                    update = (msg.get("params") or {}).get("update") or {}
                    text = None
                    content = update.get("content")
                    if isinstance(content, dict):
                        text = content.get("text")
                    elif isinstance(content, str):
                        text = content
                    if text is None:
                        for key in ("text", "summary", "title"):
                            if isinstance(update.get(key), str):
                                text = update[key]
                                break
                    if isinstance(text, str) and text:
                        state["texts"].append(text)
            return None

        try:
            send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": 1, "clientCapabilities": {}}})
            if not wait_for(1, 60):
                raise RuntimeError("ACP initialize timed out or was rejected")
            send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                  "params": {"cwd": workspace, "mcpServers": []}})
            opened = wait_for(2, 120)
            if not opened or "result" not in opened:
                raise RuntimeError(f"session/new failed: {str(opened)[:300]}")
            session_id = (opened["result"] or {}).get("sessionId") or (opened["result"] or {}).get("session_id")
            if not session_id:
                raise RuntimeError(f"session/new returned no sessionId: {str(opened)[:300]}")
            state["session_id"] = session_id
            send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                  "params": {"sessionId": session_id,
                             "prompt": [{"type": "text", "text": prompt}]}})
            done = wait_for(3, self.timeout)
            if done is None:
                raise RuntimeError("agent did not finish within the time limit")
            if "error" in done:
                raise RuntimeError(str(done["error"])[:500])
            return {"session_id": session_id, "text": "\n".join(state["texts"])[-4000:]}
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            for stream in (proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ tests
    def _run_tests(self, workspace: str) -> Dict[str, Any]:
        runner = (
            "import glob, subprocess, sys, os\n"
            "files = sorted(glob.glob(os.path.join('tests', 'test_*.py')))\n"
            "total = failed = 0\n"
            "for f in files:\n"
            "    p = subprocess.run([sys.executable, f], capture_output=True, text=True,\n"
            "                       encoding='utf-8', errors='replace', timeout=600)\n"
            "    out = (p.stdout or '') + (p.stderr or '')\n"
            "    summary = [l for l in out.splitlines() if l.startswith('Ran ') or l.startswith('FAILED')\n"
            "               or l.strip() == 'OK']\n"
            "    print('###', f)\n"
            "    print('\\n'.join(summary))\n"
            "    for l in summary:\n"
            "        if l.startswith('Ran '):\n"
            "            total += int(l.split()[1])\n"
            "        if l.startswith('FAILED ('):\n"
            "            for part in l[8:-1].split(','):\n"
            "                if '=' in part:\n"
            "                    failed += int(part.split('=')[1])\n"
            "print('TOTAL', total, 'FAILED', failed)\n"
        )
        minimal = os.path.join(workspace, "minimal-self")
        env = _strip_secrets(os.environ)
        env["PYTHONPATH"] = minimal
        proc = subprocess.run([os.sys.executable, "-c", runner], cwd=minimal, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
        out = (proc.stdout or "") + (proc.stderr or "")
        total = failed = 0
        for line in out.splitlines():
            if line.startswith("TOTAL "):
                parts = line.split()
                total, failed = int(parts[1]), int(parts[3])
        if total == 0:
            counts = _runs_ok(out)
            total, failed = counts["total"], counts["failed"]
        return {"total": total, "failed": failed, "log": out[-6000:]}

    # ------------------------------------------------------------------ public API
    def propose(self, task: str, requested_by: str = "unknown", kind: str = "self_evolution") -> Dict[str, Any]:
        """Make a proposal. Never raises; returns a status dict that is also stored in the DB."""
        pid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        prop_dir = os.path.join(self.data_dir, pid)
        workspace = os.path.join(prop_dir, "workspace")
        result: Dict[str, Any] = {"id": pid, "task": task, "status": "started", "files_changed": 0,
                                  "added": 0, "removed": 0, "tests_total": 0, "tests_failed": 0}
        patch_path = os.path.join(prop_dir, "change.patch")
        try:
            if self.proposals_today() >= self.max_per_day:
                raise RuntimeError("今天的改造额度已经用完了")
            if not os.path.exists(self.dsh_exe):
                raise RuntimeError(f"找不到 DSH 可执行文件：{self.dsh_exe}")
            # The CLI entry lives inside app.asar; a plain os.path.exists() cannot see it,
            # but the Electron binary resolves asar paths for us.
            if not (os.path.exists(self.dsh_bin) or ".asar" in self.dsh_bin.lower()):
                raise RuntimeError(f"找不到 DSH CLI 入口：{self.dsh_bin}")
            os.makedirs(prop_dir, exist_ok=True)
            self._copy_repo(workspace)
            self._git(workspace, "init", "-q")
            self._git(workspace, "add", "-A")
            self._git(workspace, "-c", "user.email=noa@local", "-c", "user.name=noa",
                      "commit", "-qm", "baseline")

            with open(os.path.join(prop_dir, "task.md"), "w", encoding="utf-8") as f:
                f.write(f"requested_by: {requested_by}\n\n{task}\n")

            agent = self._call_agent(workspace, TASK_PREAMBLE + task + "\n")
            result["session_id"] = agent["session_id"]
            result["agent_text"] = agent["text"]

            diff = self._git(workspace, "diff")
            with open(patch_path, "w", encoding="utf-8") as f:
                f.write(diff)
            result["patch_path"] = patch_path
            files = [l.split(" b/")[-1] for l in diff.splitlines() if l.startswith("diff --git")]
            added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
            result.update(files_changed=len(files), added=added, removed=removed,
                          files=files[:20])
            if not diff.strip():
                result["status"] = "no_change"
            elif added + removed > self.max_patch_lines:
                result["status"] = "too_large"
            else:
                tests = self._run_tests(workspace)
                result.update(tests_total=tests["total"], tests_failed=tests["failed"],
                              tests_log=tests["log"])
                result["status"] = "tested" if tests["failed"] == 0 else "tests_failed"
        except Exception as e:
            logger.error(f"[Evolve] {pid} failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)[:800]
        self._save(result, requested_by, kind, workspace)
        return result

    def _save(self, result: Dict[str, Any], requested_by: str, kind: str, workspace: str):
        conn = self._db()
        try:
            conn.execute(
                "REPLACE INTO evolution_proposals (id,t,requested_by,kind,task,workspace,session_id,status,"
                "files_changed,added,removed,tests_total,tests_failed,patch_path,report,error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (result["id"], time.time(), requested_by, kind, result.get("task", ""), workspace,
                 result.get("session_id"), result.get("status"), result.get("files_changed", 0),
                 result.get("added", 0), result.get("removed", 0), result.get("tests_total", 0),
                 result.get("tests_failed", 0), result.get("patch_path"),
                 json.dumps({k: v for k, v in result.items() if k not in ("tests_log", "agent_text")},
                            ensure_ascii=False)[:4000], result.get("error")))
            conn.commit()
        finally:
            conn.close()

    def apply(self, proposal_id: str, approved_by: str = "creator") -> Dict[str, Any]:
        """Apply a proposal's patch to the live tree. Requires a restart to take effect."""
        conn = self._db()
        try:
            row = conn.execute("SELECT * FROM evolution_proposals WHERE id = ?", (proposal_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return {"ok": False, "error": "没有这个提案"}
        if not row["patch_path"] or not os.path.exists(row["patch_path"]):
            return {"ok": False, "error": "提案没有补丁文件"}
        check = subprocess.run(["git", "apply", "--check", row["patch_path"]], cwd=self.repo_root,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        if check.returncode != 0:
            return {"ok": False, "error": "补丁打不上（可能已冲突）：" + check.stderr[:400]}
        applied = subprocess.run(["git", "apply", row["patch_path"]], cwd=self.repo_root,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        if applied.returncode != 0:
            return {"ok": False, "error": applied.stderr[:400]}
        conn = self._db()
        try:
            conn.execute("UPDATE evolution_proposals SET status='applied_pending_restart', approved_by=?, applied_at=? "
                         "WHERE id=?", (approved_by, time.time(), proposal_id))
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "id": proposal_id,
                "note": "补丁已打到线上代码，重启 app.py 后生效"}
