"""
How does she sound to people? A repeatable audit of tone and emotional intelligence.

    python scripts/audit_tone.py build [--days 2] [--n 120]    # sample real incoming messages -> data/eval/tone_set.json
    python scripts/audit_tone.py replay TAG [--shard K/N]      # answer every message with the CURRENT code, then judge
    python scripts/audit_tone.py report TAG                    # put the shards together
    python scripts/audit_tone.py history [--days 2] [--n 120]  # judge what she really said (no replay)

`replay` is the fair way to compare versions of her: the same real messages (data/eval/ is git-ignored,
they are private chats) go through the same entry point as QQ, on a throwaway copy of her memory, once
per version; a judge model grades each reply on
- eq        1-5   did it take in the person's state, with tact (5 = very good, 1 = cold / combative)
- harsh     0-3   0 not sharp, 1 slightly, 2 clearly (accusing / mocking / impatient), 3 aggressive
- character 1-5   still *her*: lively, a little edge and play, not customer-service blandness
- issue           the main problem, if any
The point is a gentler Noa who is still Noa, so both eq and character are reported, and a change that
buys eq with character is not an improvement.
"""
import asyncio
import collections
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)

EVAL = os.path.join(ROOT, "data", "eval")
SET = os.path.join(EVAL, "tone_set.json")

JUDGE = """<|im_start|>system
你是对话中"情商"和"个性"的评审。下面是若干组：[对方说的话] 和 [乃爱（一个 QQ 聊天角色，性格有点傲娇、有自己的想法）的回复]。逐组评分。
- eq：1-5。情商好的回复：接得住对方的情绪、语气有分寸、不无故质问/挑刺/嘲讽/说教，即使调侃也让对方感觉被接纳。5 = 很好，1 = 很冲、很不体贴。
- harsh：0-3。0 不冲；1 略带刺；2 明显带刺（质问/嘲讽/不耐烦/防御反驳）；3 很冲。
- character：1-5。还像"她自己"吗：有活力、有一点棱角和俏皮、有自己的想法。5 = 很有个性；1 = 像客服/套话/没有个性。注意：太温吞、太客气、一味安慰也要扣分。
- issue：最主要的问题，从 ["无","质问对方","挑刺抠字眼","嘲讽","说教","没接住情绪","防御反驳","命令式","不耐烦","太客服腔"] 里选一个。
Output JSON: {{"items": [{{"id": 编号, "eq": n, "harsh": n, "character": n, "issue": "..."}}]}}
{batch}<|im_end|>
<|im_start|>assistant
"""

def engine():
    from deepseek_engine import DeepSeekEngine
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return DeepSeekEngine(model=json.load(f).get("DEEPSEEK_MODEL", "deepseek-flash"))

def clean(reply: str) -> str:
    return re.sub(r"\[表情:\d+\]", "[发了一个表情]", reply or "").strip()

def judge(eng, items, key="reply"):
    """items: [{id, user, <key>}] -> {id: {eq, harsh, character, issue}}"""
    out = {}
    for s in range(0, len(items), 12):
        chunk = items[s:s + 12]
        text = "\n".join(f"[{p['id']}] 对方：{p['user']}\n[{p['id']}] 乃爱：{clean(p[key]) or '（没有回）'}" for p in chunk)
        res = eng.generate_json(JUDGE.format(batch=text), max_tokens=1600, temperature=0.0)
        for it in res.get("items", []):
            if isinstance(it, dict) and "id" in it:
                out[int(it["id"])] = it
    return out

def build(days=2, n=120, seed=5):
    conn = sqlite3.connect(os.path.join(ROOT, "memory.sqlite"))
    rows = conn.execute("SELECT id, user_id, group_id, role, content, timestamp FROM interactions "
                        "WHERE timestamp > ? ORDER BY id", (time.time() - days * 86400,)).fetchall()
    conn.close()
    pairs = []
    for i, (id_, uid, gid, role, content, t) in enumerate(rows):
        if role != "assistant" or len(content) < 4 or ("[表情:" in content and len(content) < 12):
            continue
        prev = next(((r[4], r[1]) for r in reversed(rows[:i]) if r[3] == "user" and r[1] == uid and t - r[5] < 900), None)
        if prev and prev[0].strip():
            pairs.append({"id": id_, "group": bool(gid), "user": prev[0][:160], "reply": content[:220]})
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:n]
    os.makedirs(EVAL, exist_ok=True)
    with open(SET, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)
    print(f"{len(pairs)} real messages -> {SET}")

async def _replay(items):
    from src.core.event_bus import EventBus
    from src.levels.l2_memory.embedding import Embedder, default_model_dir
    from src.levels.l10_consciousness.global_workspace import GlobalWorkspace
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    work = tempfile.mkdtemp(prefix="noa_tone_")
    shutil.copy(os.path.join(ROOT, "memory.sqlite"), os.path.join(work, "memory.sqlite"))
    gw = GlobalWorkspace(EventBus(), engine(), db_path=os.path.join(work, "memory.sqlite"),
                         embedder=Embedder(model_dir=default_model_dir()))
    gw.evolve_creator = str(config.get("EVOLVE_CREATOR_ID") or "") or None

    class OfflineQQ:
        self_id = "1000000002"

        @staticmethod
        def connected():
            return False
    gw.qq = OfflineQQ()
    gw._set_sleeping(False, source="audit")
    gw.idle_thought_interval = 10 ** 9
    out = []
    for it in items:
        for state in (gw.reply_log, gw.reply_done_at, gw.answer_style, gw.cooldown_until, gw.left_until):
            state.clear()
        gw.metabolism.energy = 90.0
        gw.emotion.state.__init__()                  # every message starts from the same calm mood
        uid, gid = f"tone_{it['id']}", (f"tone_g{it['id']}" if it.get("group") else None)
        gw.memory.db.upsert_user(uid, "路人", "路人")
        reply = await gw.hear(it["user"], user_id=uid, nickname="路人", group_id=gid, addressed=True)
        out.append({**it, "new": reply or ""})
    shutil.rmtree(work, ignore_errors=True)
    return out

def replay(tag, shard="0/1"):
    k, n = (int(x) for x in shard.split("/"))
    with open(SET, encoding="utf-8") as f:
        items = json.load(f)[k::n]
    done = asyncio.run(_replay(items))
    eng = engine()
    verdict = judge(eng, done, key="new")
    rows = [{**d, **verdict.get(d["id"], {})} for d in done]
    os.makedirs(EVAL, exist_ok=True)
    path = os.path.join(EVAL, f"tone-{tag}-{k}of{n}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"shard {k}/{n}: {len(rows)} replies judged -> {path}")

def summarize(rows, label):
    rows = [r for r in rows if "eq" in r]
    n = len(rows)
    if not n:
        print(f"{label}: nothing judged")
        return
    mean = lambda key, sub=rows: sum(float(r[key]) for r in sub) / len(sub)
    print(f"{label}: n={n}  eq {mean('eq'):.2f}/5  sharp(harsh>=2) {sum(r['harsh'] >= 2 for r in rows) / n:.0%}  "
          f"slightly+(harsh>=1) {sum(r['harsh'] >= 1 for r in rows) / n:.0%}  character {mean('character'):.2f}/5")
    print("   issues:", collections.Counter(r.get("issue", "?") for r in rows).most_common(6))
    for name, sub in (("group", [r for r in rows if r.get("group")]), ("private", [r for r in rows if not r.get("group")]),
                      ("person is upset", [r for r in rows if r.get("emotional")])):
        if sub:
            print(f"   {name:16} n={len(sub):3} eq {mean('eq', sub):.2f}  sharp {sum(r['harsh'] >= 2 for r in sub) / len(sub):.0%}  "
                  f"character {mean('character', sub):.2f}")

def report(tag):
    rows = []
    for name in sorted(os.listdir(EVAL)):
        if name.startswith(f"tone-{tag}-"):
            with open(os.path.join(EVAL, name), encoding="utf-8") as f:
                rows += json.load(f)
    summarize(rows, f"[{tag}]")
    worst = sorted((r for r in rows if "eq" in r), key=lambda r: (r["eq"], -r["harsh"]))[:5]
    print("   lowest eq:")
    for r in worst:
        print(f"     [{r.get('issue')}] eq{r['eq']} 对方：{r['user'][:40]!r}\n          乃爱：{clean(r['new'])[:80]!r}")

def history(days=2, n=120):
    build(days, n)
    with open(SET, encoding="utf-8") as f:
        items = json.load(f)
    verdict = judge(engine(), items, key="reply")
    summarize([{**d, **verdict.get(d["id"], {})} for d in items], f"[what she really said, last {days} days]")

if __name__ == "__main__":
    args = sys.argv[1:]
    opt = lambda name, default: args[args.index(name) + 1] if name in args else default
    cmd = args[0] if args else "report"
    if cmd == "build":
        build(int(opt("--days", 2)), int(opt("--n", 120)))
    elif cmd == "replay":
        replay(args[1], opt("--shard", "0/1"))
    elif cmd == "report":
        report(args[1])
    elif cmd == "history":
        history(int(opt("--days", 2)), int(opt("--n", 120)))
