"""
Level 10e: a window into her own body - read-only.

She is a program, and she was told so; until now she could not see any of it. `look_inside` lets her look at
four things, and change none of them:

- code     her own source tree: list a folder, read a file by lines, search for a word. Secrets cannot be
           reached (config.json, databases, backups, models, data/ and anything git or a venv keeps are not part
           of "her code"), and whatever is shown still goes through the firewall's redaction.
- data     her memory database: which tables there are and how many rows; the newest rows of one table;
           for the owner a SELECT of their own. The database is opened read-only (`mode=ro`, query_only), with a
           time limit, so nothing she does here can change or lock her memory.
- process  this process right now: how long it has run, memory and CPU, threads, which senses and connections
           are working, how big her memory and room are.
- logs     the last lines of her running log (what the console's log tab shows).

Who may see what: code, process and the list of tables are about her, not about anyone else, so she may show
them to anyone. Rows of her memory and the log hold what other people said, so those are for her own time and
for the person who looks after her (see access.py).
"""
import fnmatch
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
CODE_EXT = {".py", ".md", ".txt", ".html", ".js", ".css", ".json", ".ps1", ".sh", ".ini", ".toml", ".yml", ".yaml",
            ".cfg", ".svg"}
# Never "her code", whatever the path says: keys, memories, backups, other people's files, tool caches
DENY = ["config.json", "*.sqlite", "*.sqlite-*", "*.sqlite3", "*.db", "*.log", ".env", "*.pem", "*.key"]
DENY_DIRS = {".git", "data", "backups", "models", "__pycache__", ".pytest_cache", ".venv", ".claude", ".trae",
             ".playwright-cli", ".playwright-mcp", "live", "output", "node_modules", ".idea", ".vs", ".github"}
MAX_CODE_FILE = 400_000
PAGE_LINES = 120
PAGE_CHARS = 5000

# What the main tables of her memory are, in her words (others are listed without a note)
TABLE_NOTES = {
    "episodes": "经历（每件发生过的事）", "semantic_facts": "从经历里提炼出的事实", "interactions": "和人的每次来往",
    "users": "认识的人", "persona_traits": "人格特质", "trait_changes": "人格的变化", "state_causes": "情绪变化的原因",
    "chatlog": "听到的聊天记录", "diary": "日记", "autobiography": "自传的各个版本", "stickers": "收藏的表情包",
    "keepsakes": "我的小屋（收藏）", "tasks": "答应要做的事", "goals": "目标", "relations": "人和事的关系图",
    "maintainer_requests": "给维护者的请求", "knowledge": "知识库", "values": "价值", "self_state": "身体和心情的快照",
}

class SelfView:
    def __init__(self, gw, root: str = None):
        self.gw = gw
        self.root = os.path.realpath(root or getattr(gw, "code_root", None) or ROOT)
        self.started = time.time()

    # --- code ---------------------------------------------------------------------

    def _denied(self, rel: str) -> bool:
        parts = [p for p in re.split(r"[\\/]", rel) if p]
        if any(p in DENY_DIRS or p.startswith(".venv") for p in parts[:-1]):
            return True
        name = parts[-1] if parts else ""
        if name in DENY_DIRS or name.startswith(".venv"):
            return True
        return any(fnmatch.fnmatch(name.lower(), pat) for pat in DENY) or name.startswith("memory.")

    def _code_path(self, raw: str) -> str:
        raw = str(raw or "").strip().replace("\\", "/").lstrip("/")
        full = os.path.realpath(os.path.join(self.root, raw))
        if not (full == self.root or full.startswith(self.root + os.sep)):
            raise ValueError("只能看我自己的代码文件夹里面")
        rel = os.path.relpath(full, self.root)
        if rel != "." and self._denied(rel):
            raise ValueError("这里不是代码（是钥匙、记忆数据、备份或者模型），看不了")
        return full

    def code(self, path: str = "", start: int = 1, search: str = "") -> str:
        if search:
            return self._search(search, path)
        try:
            full = self._code_path(path)
        except ValueError as e:
            return str(e)
        if os.path.isdir(full):
            rows = []
            for e in sorted(os.scandir(full), key=lambda e: (not e.is_dir(), e.name.lower())):
                rel = os.path.relpath(e.path, self.root)
                if self._denied(rel):
                    continue
                if e.is_dir():
                    rows.append(f"{e.name}/")
                elif os.path.splitext(e.name)[1].lower() in CODE_EXT:
                    rows.append(f"{e.name}（{self._lines(e.path)} 行）")
            where = os.path.relpath(full, self.root).replace("\\", "/")
            head = "我的代码（根目录）" if where == "." else f"我的代码 {where}/"
            return head + " 里有：\n" + "\n".join(rows[:120]) + (
                "\n要看一个文件给 path；要找某个词给 search。" if rows else "（空的）")
        if not os.path.isfile(full):
            return f"没有这个文件：{path}"
        if os.path.splitext(full)[1].lower() not in CODE_EXT or os.path.getsize(full) > MAX_CODE_FILE:
            return "这个文件不是我能读的代码文本（或者太大了）。"
        with open(full, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        start = max(1, int(start or 1))
        page, used = [], 0
        for line in lines[start - 1:start - 1 + PAGE_LINES]:     # a page also fits the tool's result size
            used += len(line) + 8
            if page and used > PAGE_CHARS:
                break
            page.append(line)
        rel = os.path.relpath(full, self.root).replace("\\", "/")
        body = "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(page))
        end = start + len(page) - 1
        more = (f"\n（共 {len(lines)} 行，这是 {start}–{end}；往下看用 start={end + 1}）" if end < len(lines)
                else f"\n（共 {len(lines)} 行，到结尾了）")
        return f"{rel}：\n{body}{more}"

    @staticmethod
    def _lines(path: str) -> int:
        try:
            with open(path, "rb") as f:
                return f.read(MAX_CODE_FILE).count(b"\n") + 1
        except OSError:
            return 0

    def _search(self, word: str, path: str = "") -> str:
        try:
            base = self._code_path(path)
        except ValueError as e:
            return str(e)
        word = word.strip()[:80]
        needle = word.lower()
        hits: List[str] = []
        files = 0
        for d, dirs, names in os.walk(base):
            dirs[:] = sorted(x for x in dirs if not self._denied(os.path.relpath(os.path.join(d, x), self.root)))
            for n in sorted(names):
                p = os.path.join(d, n)
                rel = os.path.relpath(p, self.root)
                if self._denied(rel) or os.path.splitext(n)[1].lower() not in CODE_EXT or os.path.getsize(p) > MAX_CODE_FILE:
                    continue
                files += 1
                with open(p, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if needle in line.lower():
                            hits.append(f"{rel.replace(chr(92), '/')}:{i}: {line.strip()[:140]}")
                            if len(hits) >= 40:
                                return f"在我的代码里找「{word}」（只列前 40 处）：\n" + "\n".join(hits)
        if not hits:
            return f"在 {files} 个代码文件里没找到「{word}」。"
        return f"在我的代码里找「{word}」，{len(hits)} 处：\n" + "\n".join(hits)

    # --- data ---------------------------------------------------------------------

    def _db_path(self) -> Optional[str]:
        mem = getattr(self.gw, "memory", None)
        return getattr(getattr(mem, "db", None), "db_path", None) or getattr(self.gw, "db_path", None)

    def _connect(self) -> sqlite3.Connection:
        path = self._db_path()
        if not path or not os.path.exists(path):
            raise ValueError("找不到我的记忆数据库")
        uri = "file:" + os.path.abspath(path).replace("\\", "/") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        conn.execute("PRAGMA query_only = 1")
        deadline = time.monotonic() + 5
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)   # never hangs her
        return conn

    @staticmethod
    def _cell(v: Any) -> str:
        if isinstance(v, (bytes, bytearray)):
            return f"<{len(v)} 字节>"
        if isinstance(v, float) and 1.5e9 < v < 4e9:                 # a time
            return time.strftime("%m-%d %H:%M", time.localtime(v))
        s = "" if v is None else str(v)
        return s if len(s) <= 120 else s[:120] + "…"

    def data(self, table: str = "", sql: str = "", may_read_rows: bool = False, owner: bool = False,
             limit: int = 8) -> str:
        try:
            conn = self._connect()
        except (ValueError, sqlite3.Error) as e:
            return f"打不开记忆数据库：{e}"
        try:
            names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                                "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            if sql:
                if not owner:
                    return "自己写查询只有创造者在的时候才行；我可以按表看最新的几条（给 table）。"
                q = sql.strip().rstrip(";")
                if not re.match(r"(?is)^\s*(select|with)\b", q) or ";" in q:
                    return "这里只能看（一条 SELECT），不能改。"
                cur = conn.execute(q)
                cols = [c[0] for c in cur.description or []]
                rows = cur.fetchmany(30)
                return self._table_text(cols, rows, f"查询结果（最多 30 行）")
            if not table:
                lines = []
                size = os.path.getsize(self._db_path())
                for n in names:
                    count = conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
                    lines.append(f"- {n}：{count} 行" + (f"（{TABLE_NOTES[n]}）" if n in TABLE_NOTES else ""))
                tail = ("\n看某张表最新的几条给 table。" if may_read_rows else
                        "\n（表里的内容有别人说过的话，只在我自己的时间或者创造者在的时候看）")
                return f"我的记忆数据库（{size / 1024 / 1024:.1f} MB，{len(names)} 张表）：\n" + "\n".join(lines) + tail
            if table not in names:
                return f"没有叫 {table} 的表。不给 table 可以看有哪些表。"
            if not may_read_rows:
                count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                return (f"{table} 有 {count} 行" + (f"（{TABLE_NOTES[table]}）" if table in TABLE_NOTES else "") +
                        "。里面是别人说过的话和我记下的事，只在我自己的时间或者创造者在的时候看。")
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            order = "rowid DESC"
            rows = conn.execute(f'SELECT * FROM "{table}" ORDER BY {order} LIMIT ?', (max(1, min(int(limit), 30)),)).fetchall()
            total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            return self._table_text(cols, rows, f"{table}（共 {total} 行，最新的 {len(rows)} 条）")
        except sqlite3.OperationalError as e:
            return f"没查成：{e}"
        finally:
            conn.close()

    def _table_text(self, cols: List[str], rows, title: str) -> str:
        if not rows:
            return title + "：没有记录。"
        out = [title + "："]
        for r in rows:
            out.append(" | ".join(f"{c}={self._cell(v)}" for c, v in zip(cols, r) if v not in (None, "", "[]", "{}")))
        return "\n".join(out)

    # --- process ------------------------------------------------------------------

    @staticmethod
    def _dur(sec: float) -> str:
        sec = int(sec)
        d, h, m = sec // 86400, sec % 86400 // 3600, sec % 3600 // 60
        return (f"{d} 天 " if d else "") + (f"{h} 小时 " if h or d else "") + f"{m} 分钟"

    @staticmethod
    def _size(n: float) -> str:
        return f"{n / 1024 / 1024 / 1024:.2f} GB" if n > 1024 ** 3 else f"{n / 1024 / 1024:.1f} MB"

    def _folder_size(self, path: str) -> int:
        total = 0
        for d, _, names in os.walk(path):
            for n in names:
                try:
                    total += os.path.getsize(os.path.join(d, n))
                except OSError:
                    pass
        return total

    def body(self) -> Dict[str, Any]:
        """Facts about this process and her subsystems (also served to the console's 小屋 tab)."""
        gw = self.gw
        out: Dict[str, Any] = {"pid": os.getpid(), "python": sys.version.split()[0], "platform": sys.platform}
        try:
            import psutil
            p = psutil.Process()
            with p.oneshot():
                out.update(started=p.create_time(), rss=p.memory_info().rss, threads=p.num_threads(),
                           cpu=p.cpu_percent(interval=0.2), children=len(p.children()))
            vm = psutil.virtual_memory()
            out.update(machine_memory=vm.total, machine_memory_used=vm.percent, machine_cpu=psutil.cpu_percent(interval=None))
        except Exception:
            out["started"] = self.started
        out["uptime"] = time.time() - out["started"]
        qq = getattr(gw, "qq", None)
        senses = getattr(qq, "senses", None)
        out["qq"] = bool(qq and qq.connected())
        out["abilities"] = dict(getattr(gw, "abilities", {}) or {})
        out["hear_problem"] = getattr(senses, "hear_problem", "") if senses else ""
        hub = getattr(gw, "mcp", None)
        try:                                       # (the command lines are left out: they show the machine's paths)
            out["mcp"] = [{"name": s["name"], "running": bool(s.get("running")), "tools": s.get("tools", 0)}
                          for s in hub.status()] if hub else None
        except Exception:
            out["mcp"] = None
        game = getattr(gw, "game", None)
        try:
            gs = game.status() if game else None
            out["game"] = {"connected": bool((gs or {}).get("connected") or (gs or {}).get("online"))} if gs is not None else None
        except Exception:
            out["game"] = None
        backup = getattr(gw, "backup", None)
        last = getattr(backup, "last_result", None) if backup else None
        if last:
            out["backup"] = {"ok": last.get("ok"), "t": last.get("t"), "name": last.get("name")}
        else:                                       # nothing this run yet: the newest checked copy on disk
            try:
                newest = backup.list()[-1] if backup else None
            except (OSError, IndexError):
                newest = None
            out["backup"] = {"ok": True, "t": newest["t"], "name": newest["name"]} if newest else None
        fw = getattr(gw, "firewall", None)
        out["firewall_locked"] = bool(getattr(fw, "locked", False))
        out["sleeping"] = bool(getattr(getattr(gw, "metabolism", None), "is_sleeping", False))
        out["energy"] = getattr(getattr(gw, "metabolism", None), "energy", None)
        out["queue"] = len(getattr(gw, "output_queue", []) or [])
        db = self._db_path()
        out["memory_db"] = os.path.getsize(db) if db and os.path.exists(db) else 0
        ks = getattr(gw, "keepsakes", None)
        out["keepsakes"] = ks.stats() if ks else None
        ws = getattr(gw, "workspace_dir", None)
        out["workspace"] = self._folder_size(ws) if ws and os.path.isdir(ws) else 0
        return out

    def process(self) -> str:
        b = self.body()
        lines = [f"我这个进程：pid {b['pid']}，Python {b['python']}，已经连续运行了 {self._dur(b['uptime'])}。"]
        if "rss" in b:
            lines.append(f"占用内存 {self._size(b['rss'])}，{b['threads']} 个线程，{b['children']} 个子进程，"
                         f"CPU {b['cpu']:.0f}%；这台电脑内存用了 {b['machine_memory_used']:.0f}%（共 {self._size(b['machine_memory'])}）。")
        senses = [("QQ 连接", b["qq"]), ("看图", b["abilities"].get("see")), ("听语音", b["abilities"].get("hear")),
                  ("读图上的字", b["abilities"].get("ocr"))]
        lines.append("感官和连接：" + "，".join(f"{n}{'✓' if ok else '✗'}" for n, ok in senses) +
                     (f"（听觉：{b['hear_problem']}）" if b.get("hear_problem") else ""))
        if b.get("mcp"):
            lines.append("外部工具：" + "，".join(f"{s['name']}{'✓' if s['running'] else '✗'}（{s['tools']} 个工具）"
                                              for s in b["mcp"]))
        if b.get("game") is not None:
            lines.append(f"游戏身体：{'连着' if b['game']['connected'] else '没连上'}")
        if b.get("backup"):
            when = time.strftime("%m-%d %H:%M", time.localtime(b["backup"]["t"])) if b["backup"].get("t") else "?"
            lines.append(f"最近一次记忆备份：{when}，{'成功' if b['backup']['ok'] else '失败'}")
        ks = b.get("keepsakes") or {}
        lines.append(f"记忆数据库 {self._size(b['memory_db'])}；我的小屋 {ks.get('items', 0)} 样东西"
                     f"（{self._size(ks.get('bytes', 0))}）；工作文件夹 {self._size(b['workspace'])}。")
        lines.append(f"现在{'睡着' if b['sleeping'] else '醒着'}，精力 {b['energy']:.0f}；待发出的话 {b['queue']} 条；"
                     f"防火墙{'封锁中' if b['firewall_locked'] else '正常'}。" if b.get("energy") is not None else "")
        return "\n".join(l for l in lines if l)

    # --- logs ---------------------------------------------------------------------

    def logs(self, level: str = "", count: int = 30) -> str:
        buf = getattr(self.gw, "log_buffer", None)
        if buf is None:
            return "现在看不到运行日志（控制台没有开着）。"
        level = (level or "").upper()
        if level in ("WARN",):
            level = "WARNING"
        rows = [r for r in list(buf.lines) if not level or r["level"] == level or
                (level == "WARNING" and r["level"] in ("ERROR", "CRITICAL"))]
        rows = rows[-max(1, min(int(count or 30), 80)):]
        if not rows:
            return "最近没有" + ("这种级别的" if level else "") + "日志。"
        return "我最近的运行日志：\n" + "\n".join(
            f"{time.strftime('%H:%M:%S', time.localtime(r['t']))} {r['level'][0]} {r['name'].split('.')[-1]}: {r['msg'][:200]}"
            for r in rows)
