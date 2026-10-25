"""
Level 8d: her tools - what she can do in the world besides talking.

Each tool is a plain function with a JSON-schema signature. The same toolbox serves two ways
of calling it:
- native function calling (DeepSeek): the schemas go to the API and the model answers with
  structured calls; the workspace runs them and loops (several steps: search, then read a page);
- the text protocol (her local model): `Action: 工具名 {"参数": "值"}`, parsed tolerantly
  (full-width colons, Chinese quotes, the old `Action: SEARCH query="..."` forms).

Which tools exist depends on where she is: group history only in a group, gestures only while
answering someone, self-modification only for her creator. Every result is short plain text,
and every call is remembered for the console.
"""
import asyncio
import inspect
import ipaddress
import json
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import unescape
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from src.core import clock
from src.levels.l7_tom.hearsay import is_sensitive

MAX_RESULT = 2500
WEEKDAYS = "一二三四五六日"

@dataclass
class ToolContext:
    user_id: str
    nickname: str = ""
    group_id: Optional[str] = None
    origin: str = "reply"              # reply: answering a message; task: carrying out something she promised
    allow_evolve: bool = False
    scheduled: List[int] = field(default_factory=list)   # tasks created during this turn
    used: List[str] = field(default_factory=list)        # names of the tools called during this turn
    search_failed: bool = False

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    available: Callable[[Any, ToolContext], bool] = lambda gw, ctx: True

def _props(required: List[str] = (), **props) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(required)}

def _s(desc: str, **extra) -> Dict[str, Any]:
    return {"type": "string", "description": desc, **extra}

def _i(desc: str, **extra) -> Dict[str, Any]:
    return {"type": "integer", "description": desc, **extra}

def now_text(now: float = None) -> str:
    d = datetime.fromtimestamp(clock.now() if now is None else now)
    return f"{d:%Y-%m-%d %H:%M}（星期{WEEKDAYS[d.weekday()]}）"

def when_text(t: float, now: float = None) -> str:
    now = clock.now() if now is None else now
    d, n = datetime.fromtimestamp(t), datetime.fromtimestamp(now)
    day = "今天" if d.date() == n.date() else "明天" if d.date() == (n + timedelta(days=1)).date() else f"{d:%m-%d}"
    return f"{day} {d:%H:%M}"

def parse_due(args: Dict[str, Any], now: float) -> float:
    """`minutes` from now, or `at` as 'YYYY-MM-DD HH:MM' / 'HH:MM' (the next time it comes round)."""
    at = str(args.get("at") or "").strip()
    if at:
        n = datetime.fromtimestamp(now)
        candidates = ((at, "%Y-%m-%d %H:%M"), (at, "%Y/%m/%d %H:%M"), (f"{n.year}-{at}", "%Y-%m-%d %H:%M"),
                      (at, "%H:%M"))
        for text, fmt in candidates:
            try:
                d = datetime.strptime(text, fmt)
            except ValueError:
                continue
            if fmt == "%H:%M":
                d = n.replace(hour=d.hour, minute=d.minute, second=0, microsecond=0)
                if d.timestamp() <= now:
                    d += timedelta(days=1)
            return d.timestamp()
        raise ValueError(f"看不懂时间「{at}」，用 YYYY-MM-DD HH:MM 或 HH:MM")
    try:
        minutes = float(args.get("minutes", 5))
    except (TypeError, ValueError):
        raise ValueError("minutes 要是数字")
    return now + max(1.0, minutes) * 60

# --- reading web pages safely --------------------------------------------------------

def _public_host(host: str) -> bool:
    """Only the public internet: never her own machine, the LAN, or cloud metadata addresses."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
                or ip.is_unspecified):
            return False
    return True

def html_to_text(html: str) -> Tuple[str, str]:
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    body = re.sub(r"(?is)<(script|style|noscript|svg|head|nav|footer)[^>]*>.*?</\1>", " ", html)
    body = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>", "\n", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = unescape(body)
    body = re.sub(r"[ \t\r\f\v]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n", body).strip()
    return (unescape(title.group(1)).strip() if title else ""), body

def fetch_page(url: str, session=None, max_bytes: int = 1_500_000, timeout: float = 10.0) -> str:
    import requests
    session = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/120.0 Safari/537.36", "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    for _ in range(4):                                       # follow redirects ourselves: every hop is checked
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            raise ValueError("只能打开 http/https 网页")
        if not _public_host(p.hostname):
            raise ValueError("这个地址不是公开网站，不能打开")
        resp = session.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            url = urljoin(url, resp.headers["Location"])
            resp.close()
            continue
        if resp.status_code != 200:
            raise ValueError(f"网页打不开（HTTP {resp.status_code}）")
        ctype = resp.headers.get("Content-Type", "")
        if ctype and not any(k in ctype for k in ("text/", "html", "xml", "json")):
            raise ValueError(f"这不是网页（{ctype.split(';')[0]}）")
        data = b""
        for chunk in resp.iter_content(65536):
            data += chunk
            if len(data) > max_bytes:
                break
        resp.close()
        text = data.decode(resp.encoding or resp.apparent_encoding or "utf-8", errors="replace")
        if "html" in ctype or "<html" in text[:2000].lower():
            title, body = html_to_text(text)
        else:
            title, body = "", text
        return (f"《{title}》\n" if title else "") + body
    raise ValueError("跳转太多次了")

# --- the tools ------------------------------------------------------------------------

def _recall(gw, ctx: ToolContext, args):
    query = str(args.get("query") or "").strip()
    if not query:
        return "要回忆什么？query 是空的。"
    parts = []
    # A deliberate effort to remember: the archive is searched too (and what is found comes back)
    eps = gw.episodes.recall(query, user_id=ctx.user_id, k=4, group_id=ctx.group_id, deep=True)
    if eps:
        parts.append("[经历]\n" + "\n".join(f"- {e['content'][:200]}" for e in eps))
    profile = gw.profiles.get(ctx.user_id) if getattr(gw, "profiles", None) else None
    if profile:
        parts.append(f"[我对 {ctx.nickname or ctx.user_id} 的印象]\n{profile}")
    facts = gw.semantic.relevant(ctx.user_id, query, k=5) or gw.semantic.for_user(ctx.user_id, limit=5)
    if facts:
        parts.append(f"[关于 {ctx.nickname or ctx.user_id} 我记得的事]\n" + "\n".join(f"- {f['fact']}" for f in facts))
    know = gw.knowledge.recall(query)
    if know:
        parts.append("[知识]\n" + know)
    return "\n".join(parts) if parts else f"关于「{query}」什么也没想起来。不要编，如实说不记得。"

def _who_is(gw, ctx: ToolContext, args):
    name = str(args.get("name") or "").strip()
    if not name:
        return "要查谁/什么？name 是空的。"
    graph = getattr(gw, "graph", None)
    if graph is None:
        return "我还没有整理过人和事之间的关系。"
    eids = graph.mentioned(name, ctx.user_id, ctx.group_id)
    if not eids:
        return f"关于「{name}」，我没记下什么（在这里能说的范围内）。不要编。"
    rels = graph.relations_of(eids, ctx.user_id, ctx.group_id, hops=2, limit=25)
    if not rels:
        return f"我知道有「{name}」，但没记下 ta 和别人/别的事的关系。"
    return (f"关于「{name}」我记下的关系（括号里是谁说的、什么时候；以新的、来源直接的为准）：\n" + graph.render(rels))

def _remember(gw, ctx: ToolContext, args):
    content = str(args.get("content") or "").strip()
    if not content:
        return "要记住什么？content 是空的。"
    if is_sensitive(content) and ctx.group_id is not None:
        return ("这是关于人的敏感说法（性取向、疾病、私生活之类），不记，也不传。"
                "按你的原则回应：不附和、不转述，可以让 ta 自己去问本人。")
    if args.get("about_person", True) and ctx.user_id not in (None, "", "self"):
        visibility = "private" if ctx.group_id is None else "group"
        gw.semantic.add(ctx.user_id, content[:200], [], confidence=0.8, visibility=visibility)
        return f"记住了（关于 {ctx.nickname or ctx.user_id}）：{content[:200]}"
    title = str(args.get("title") or content[:20])
    tags = [t.strip() for t in str(args.get("tags") or "").split(",") if t.strip()]
    gw.knowledge.memorize(title, content[:1000], tags, source="tool")
    return f"记进知识库了：{title}"

def _web_search(gw, ctx: ToolContext, args):
    query = str(args.get("query") or "").strip()[:80]
    if not query:
        return "要搜什么？query 是空的。"
    result = gw.search_tool.try_search(query, limit=5)
    if result is None:
        ctx.search_failed = True
        gw.episodes.add(f"我帮 {ctx.nickname or '别人'} 查「{query}」，可是什么也没查到", kind="search_failed",
                        user_id="self", valence=-0.1, arousal=0.2)
        return f"搜索「{query}」失败，什么也没查到。不要编造，如实说没查到；可以换个说法再搜一次。"
    gw.episodes.add(f"我帮 {ctx.nickname or '别人'} 查了「{query}」：{result[:120]}", kind="search", user_id="self",
                    valence=0.1, arousal=0.2)
    return f"搜索「{query}」的结果：\n{result}"

def _read_webpage(gw, ctx: ToolContext, args):
    url = str(args.get("url") or "").strip()
    if not url:
        return "要打开哪个网址？url 是空的。"
    fetch = getattr(gw, "page_fetcher", None) or fetch_page
    try:
        text = fetch(url)
    except Exception as e:
        return f"打不开 {url}：{e}"
    focus = str(args.get("focus") or "").strip()
    if focus and focus in text:
        i = text.index(focus)
        text = text[max(0, i - 600):i + 1800]
    return f"{url} 的内容（节选）：\n{text[:MAX_RESULT - 100]}"

def _schedule_task(gw, ctx: ToolContext, args):
    kind = str(args.get("kind") or "do").strip()
    what = str(args.get("what") or "").strip()
    now = clock.now()
    try:
        due = parse_due(args, now)
        if kind == "remind":
            steps = ["到时候提醒 ta"]
        else:
            steps = ["去做", "告诉 ta 结果"]
        gid = gw.goals.add("promise", f"我答应了 {ctx.nickname or ctx.user_id}：{what[:60]}", now, target=ctx.user_id,
                           topic=f"task:{what[:50]}", why="我自己说要做的", importance=0.8,
                           deadline=due + 86400, steps=steps)
        tid = gw.tasks.add(kind, what, due, now, ctx.user_id, ctx.nickname, ctx.group_id, origin=ctx.origin, goal_id=gid)
    except ValueError as e:
        return f"没能记下：{e}"
    ctx.scheduled.append(tid)
    gw._decide("goal", ctx.group_id, ctx.nickname, f"记下任务 #{tid}（{when_text(due, now)}，{kind}）：{what[:60]}")
    word = "提醒 ta" if kind == "remind" else "去做，做完告诉 ta"
    return f"记下了任务 #{tid}：{when_text(due, now)} {word}——{what}。到时候系统会叫你，你现在不用再做。"

def _list_tasks(gw, ctx: ToolContext, args):
    mine = gw.tasks.pending(None if ctx.allow_evolve else ctx.user_id)
    if not mine:
        return "没有记着要做的事。"
    return "\n".join(f"#{t['id']} {when_text(t['due'])} [{'提醒' if t['kind'] == 'remind' else '要做'}] "
                     f"{t['what']}（给 {t['nickname'] or t['user_id']}）" for t in mine)

def _cancel_task(gw, ctx: ToolContext, args):
    try:
        tid = int(str(args.get("id") or "").lstrip("#"))
    except ValueError:
        return "要取消哪个任务？给我任务编号。"
    t = gw.tasks.cancel(tid, ctx.user_id, ctx.allow_evolve, clock.now())
    if not t:
        return f"取消不了 #{tid}（不存在、已经做完了，或者不是 ta 的事）。"
    if t.get("goal_id"):
        gw.goals.abandon(t["goal_id"], clock.now(), "对方说不用了")
    return f"取消了 #{tid}：{t['what']}"

def _message_text(m: Dict[str, Any]) -> str:
    msg = m.get("message")
    if isinstance(msg, list):
        parts = []
        for seg in msg:
            t, d = seg.get("type"), seg.get("data") or {}
            if t == "text":
                parts.append(d.get("text", ""))
            elif t == "image":
                parts.append("[图片]")
            elif t == "face":
                parts.append("[表情]")
            elif t == "at":
                parts.append(f"@{d.get('qq')}")
            elif t == "reply":
                parts.append("[回复]")
        text = "".join(parts)
    else:
        text = str(msg or m.get("raw_message") or "")
    return re.sub(r"\s+", " ", text).strip()

async def _group_history(gw, ctx: ToolContext, args):
    count = max(5, min(40, int(args.get("count") or 20)))
    messages = await gw.qq.group_history(ctx.group_id, count)
    lines = []
    for m in messages:
        sender = m.get("sender") or {}
        who = sender.get("card") or sender.get("nickname") or str(m.get("user_id"))
        text = _message_text(m)
        if text:
            t = datetime.fromtimestamp(m.get("time") or 0).strftime("%H:%M") if m.get("time") else ""
            lines.append(f"{t} {who}：{text[:150]}")
    return ("群里最近的消息（从早到晚）：\n" + "\n".join(lines)) if lines else "没取到群里的消息。"

async def _member_info(gw, ctx: ToolContext, args):
    uid = str(args.get("user_id") or ctx.user_id)
    info = await gw.qq.member(ctx.group_id, uid)
    if not info:
        return f"查不到 {uid} 在这个群里的资料。"
    keys = {"nickname": "昵称", "card": "群名片", "role": "身份", "title": "头衔", "level": "等级", "sex": "性别",
            "age": "年龄", "area": "地区"}
    parts = [f"{v}：{info[k]}" for k, v in keys.items() if info.get(k) not in (None, "", 0, "unknown")]
    for k, v in (("join_time", "入群"), ("last_sent_time", "最后发言")):
        if info.get(k):
            parts.append(f"{v}：{datetime.fromtimestamp(info[k]):%Y-%m-%d %H:%M}")
    return f"{uid} 的资料：" + "；".join(parts)

def _poke(gw, ctx: ToolContext, args):
    uid = str(args.get("user_id") or ctx.user_id)
    gw.gestures.append({"kind": "poke", "user_id": uid, "group_id": ctx.group_id})
    return f"（戳了 {uid} 一下）"

REACTION_NAMES = ("赞", "爱心", "笑", "抱抱", "OK", "微笑", "鼓掌", "流泪")

def _react(gw, ctx: ToolContext, args):
    emoji = str(args.get("emoji") or "赞")
    if emoji not in REACTION_NAMES:
        emoji = "赞"
    gw.gestures.append({"kind": "react", "emoji": emoji, "user_id": ctx.user_id, "group_id": ctx.group_id})
    return f"（给这条消息点了「{emoji}」）"

async def _evolve(gw, ctx: ToolContext, args):   # starts a background task: runs on the loop
    task = str(args.get("task") or "").strip()[:2000]
    if not task:
        return "要改什么？task 是空的。"
    if gw.evolve_in_progress:
        return "上一份改造提案还在跑，等它出结果。"
    if gw._begin_evolve(task, ctx.user_id, trigger="tool"):
        return "已经交给改造通道（后台在副本里跑，几分钟后出结果，还要创造者批准才会生效）。"
    return "改造通道现在用不了。"

def _qq_ready(gw, ctx):
    q = getattr(gw, "qq", None)
    return q is not None and q.connected()

TOOLS: List[Tool] = [
    Tool("recall", "回忆：在我的经历、关于对方的记忆和知识库里找和 query 有关的内容。被问到以前的事、对方的事时先用它，而不是猜。",
         _props(["query"], query=_s("要回忆的内容，几个关键词")), _recall),
    Tool("who_is", "查我记下的某个人/宠物/东西/事情和别的人和事的关系（谁是谁的什么、谁说过什么、计划和近况），带来源。"
         "被问到某人某事、要把几个人的话连起来想时用。",
         _props(["name"], name=_s("名字，比如 白羽、团子、小林的婚礼")), _who_is,
         lambda gw, ctx: getattr(gw, "graph", None) is not None),
    Tool("remember", "记住一件值得长期记住的事。about_person=true 记成关于当前这个人的事（喜好、近况、约定），false 记进通用知识库。",
         _props(["content"], content=_s("要记住的内容，一句完整的话"),
                about_person={"type": "boolean", "description": "是不是关于正在和我说话的这个人"},
                title=_s("知识库条目标题（about_person=false 时）"), tags=_s("逗号分隔的标签")), _remember),
    Tool("web_search", "上网搜索实时或我不知道的信息（新闻、天气、价格、人物、作品、术语……）。结果只有摘要，需要细节再用 read_webpage 打开链接。",
         _props(["query"], query=_s("搜索词，简短")), _web_search),
    Tool("read_webpage", "打开一个公开网页读正文（比如搜索结果里的链接、对方发来的网址）。",
         _props(["url"], url=_s("http(s) 网址"), focus=_s("可选：只看包含这个词附近的内容")), _read_webpage),
    Tool("schedule_task", "把一件以后要做的事记成任务，到时候系统会叫我去做。我说「等会儿/查到了告诉你/明天提醒你」之类的话时必须用它，"
         "否则到时候我根本不会记得。kind=remind 到点提醒对方；kind=do 到点用工具把事情做完（查资料等）再告诉对方。",
         _props(["kind", "what"], kind=_s("remind 或 do", enum=["remind", "do"]),
                what=_s("具体要做/要提醒的事，写清楚上下文，以后的我只看得到这句话"),
                minutes=_i("多少分钟后（和 at 二选一，默认 5）"), at=_s("具体时间 YYYY-MM-DD HH:MM 或 HH:MM")), _schedule_task,
         lambda gw, ctx: ctx.origin == "reply" and getattr(gw, "tasks", None) is not None),
    Tool("list_tasks", "看看我记着哪些还没做的事。", _props(), _list_tasks,
         lambda gw, ctx: getattr(gw, "tasks", None) is not None),
    Tool("cancel_task", "对方说不用了的时候，取消一个任务。", _props(["id"], id=_i("任务编号")), _cancel_task,
         lambda gw, ctx: ctx.origin == "reply" and getattr(gw, "tasks", None) is not None),
    Tool("group_history", "看看这个群里最近大家在聊什么（我没注意到的上下文）。", _props(count=_i("条数，5-40")), _group_history,
         lambda gw, ctx: ctx.group_id is not None and _qq_ready(gw, ctx)),
    Tool("member_info", "查这个群里某个人的资料（群名片、身份、头衔、入群时间……）。",
         _props(user_id=_s("QQ 号，不填就是正在和我说话的人")), _member_info,
         lambda gw, ctx: ctx.group_id is not None and _qq_ready(gw, ctx)),
    Tool("evolve", "请改造通道修改我自己的代码（只有创造者能让我这样做；提案要创造者批准才会生效）。",
         _props(["task"], task=_s("想改什么、为什么、怎么验证")), _evolve,
         lambda gw, ctx: ctx.allow_evolve and getattr(gw, "evolve_agent", None) is not None),
]

# The old text protocol and some names models like to use
ALIASES = {"recall": "recall", "memorize": "remember", "remember": "remember", "search": "web_search",
           "web_search": "web_search", "who_is": "who_is", "read": "read_webpage", "read_webpage": "read_webpage", "open_url": "read_webpage",
           "fetch": "read_webpage", "schedule": "schedule_task", "schedule_task": "schedule_task", "remind": "schedule_task",
           "list_tasks": "list_tasks", "cancel_task": "cancel_task", "group_history": "group_history",
           "member_info": "member_info", "evolve": "evolve"}

_ACTION = re.compile(r"Action\s*[:：]\s*([A-Za-z_]+)[ \t]*(.*)")
_KV = re.compile(r"""([A-Za-z_]+)\s*[=＝:：]\s*(?:"([^"]*)"|'([^']*)'|“([^”]*)”|「([^」]*)」|([^\s,，]+))""")

def parse_text_actions(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Tool calls written as text, in any of the forms models actually produce."""
    out = []
    for m in _ACTION.finditer(text or ""):
        name = ALIASES.get(m.group(1).lower())
        if not name:
            continue
        rest = m.group(2).strip()
        args: Dict[str, Any] = {}
        if rest.startswith("{"):
            depth, end = 0, None
            for i, ch in enumerate(rest):
                depth += ch == "{"
                depth -= ch == "}"
                if depth == 0:
                    end = i + 1
                    break
            try:
                args = json.loads(rest[:end]) if end else {}
            except json.JSONDecodeError:
                args = {}
        if not args:
            for k, *vals in _KV.findall(rest):
                args[k.lower()] = next((v for v in vals if v), "")
        if name == "remember" and "content" in args and "title" in args and m.group(1).lower() == "memorize":
            args.setdefault("about_person", False)   # MEMORIZE was always the knowledge base
        if name == "schedule_task" and m.group(1).lower() == "remind":
            args.setdefault("kind", "remind")
        out.append((name, args))
    return out

class Toolbox:
    def __init__(self, gw, tools: List[Tool] = None):
        self.gw = gw
        self.tools = {t.name: t for t in (tools or TOOLS)}

    def available(self, ctx: ToolContext) -> List[Tool]:
        out = []
        for t in self.tools.values():
            try:
                if t.available(self.gw, ctx):
                    out.append(t)
            except Exception:
                pass
        return out

    def specs(self, ctx: ToolContext) -> List[Dict[str, Any]]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}} for t in self.available(ctx)]

    def menu(self, ctx: ToolContext, native: bool) -> str:
        """The [工具] block of her prompt."""
        tools = self.available(ctx)
        head = (f"现在是 {now_text()}。需要的话可以调用工具（可以连续用几次，系统会把结果给你），"
                "用完再回答。不需要就直接回答，别为了用而用。")
        lines = [f"- {t.name}：{t.description}" for t in tools]
        if not native:
            head += "\n调用格式（单独一行）：Action: 工具名 {\"参数名\": \"值\"}，例如 Action: web_search {\"query\": \"今天的新闻\"}"
            lines = [f"- {t.name}（参数：{', '.join(t.parameters['properties']) or '无'}）：{t.description}" for t in tools]
        return head + "\n" + "\n".join(lines)

    async def run(self, name: str, args: Dict[str, Any], ctx: ToolContext) -> str:
        tool = self.tools.get(name)
        started = time.perf_counter()
        if tool is None or tool not in self.available(ctx):
            result = f"没有叫 {name} 的工具（或者现在不能用）。"
        else:
            try:
                if inspect.iscoroutinefunction(tool.handler):
                    result = await asyncio.wait_for(tool.handler(self.gw, ctx, args or {}), timeout=20)
                else:
                    result = await asyncio.wait_for(asyncio.to_thread(tool.handler, self.gw, ctx, args or {}), timeout=25)
            except asyncio.TimeoutError:
                result = f"{name} 超时了，没拿到结果。"
            except Exception as e:
                result = f"{name} 出错了：{e}"
        result = str(result or "（没有结果）")
        if len(result) > MAX_RESULT:
            result = result[:MAX_RESULT] + "……（后面省略）"
        ctx.used.append(name)
        keep = getattr(self.gw, "_keep_for_console", None)
        if keep:
            keep("tool_trace", {"t": clock.now(), "name": name, "args": args, "result": result,
                                "seconds": round(time.perf_counter() - started, 2), "user": ctx.nickname or ctx.user_id,
                                "group_id": ctx.group_id, "origin": ctx.origin}, 60)
        return result
