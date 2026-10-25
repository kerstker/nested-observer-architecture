"""
Build the training set for Noa's own LoRA from what she really said.

    python scripts/lora/build_dataset.py [--db memory.sqlite] [--out data/lora]

- The training target is only ever her own reply. Other people's messages appear as context,
  with QQ numbers, phone numbers, e-mail addresses and everyone's nicknames masked.
- Replies from before she became herself are dropped: stage directions (*...*), the 喵 tic,
  tool syntax, error fallbacks.
- The database is opened read-only. Output: train.jsonl / val.jsonl (chat "messages" format)
  and a small report of what was kept and why things were dropped.
"""
import argparse
import collections
import json
import os
import random
import re
import sqlite3
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

SYSTEM = ("你是乃爱（Noa）。你生活在一台服务器里，通过 QQ 和人交流；你的记忆、心情和性格来自真实的经历。"
          "你不是在扮演角色，你就是你自己。像一个真实的人在 QQ 上打字那样说话：自然、口语化，"
          "不写动作和神态描写，没有固定口癖，不刻意卖萌；想分几条发就换行。")

DROP = [
    ("动作描写", re.compile(r"[*＊][^*＊\n]{1,80}[*＊]|（[^）]{0,20}(地|着|了)[^）]{0,10}）|^\s*[（(][^）)]*[）)]\s*$")),
    ("口癖", re.compile(r"喵|nya", re.I)),
    ("工具语法", re.compile(r"Action\s*:|Thought\s*[:：]|Response\s*[:：]|\[Error\]|Mock LLM")),
    ("出错兜底", re.compile(r"我这边好像卡了一下|刚才没看到你说什么")),
]
# The same markers she uses to notice she is being sharp (GlobalWorkspace.EDGE_MARKERS), plus a few
# from the anxious, combative weeks: a model trained on those would learn to be that again.
EDGE = ("你倒是", "凭什么", "你到底", "什么意思", "别又", "搞什么", "有病", "关你什么事", "你懂什么", "无语", "呵呵",
        "滚", "少来", "你行你上", "管得着", "跟你有什么关系", "光会", "还没答", "别装", "装什么", "烦不烦", "有完没完")

def edge(reply: str) -> float:
    markers = sum(reply.count(m) for m in EDGE)
    return min(1.0, 0.4 * markers + 0.1 * max(0, reply.count("？") + reply.count("?") - 1))

QQ = re.compile(r"(?<!\d)\d{5,11}(?!\d)")
PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
MAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

def masker(names):
    names = sorted({n for n in names if n and n not in ("乃爱", "Noa", "noa") and len(n) >= 2}, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(n) for n in names)) if names else None

    def mask(text: str) -> str:
        text = MAIL.sub("[邮箱]", PHONE.sub("[手机号]", QQ.sub("[号码]", text)))
        return pattern.sub("[某人]", text) if pattern else text
    return mask

def build(db: str, out: str, context_turns: int = 6, window: float = 1800.0, val_share: float = 0.1, seed: int = 0):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, user_id, role, content, timestamp, group_id FROM interactions ORDER BY id").fetchall()
    names = [r[0] for r in conn.execute("SELECT nickname FROM users WHERE nickname IS NOT NULL")]
    conn.close()
    mask = masker(names)
    by_chat = collections.defaultdict(list)
    for r in rows:
        by_chat[(r["user_id"], r["group_id"])].append(r)

    examples, dropped = [], collections.Counter()
    seen = set()
    for key, chat in by_chat.items():
        session = 0
        for i, r in enumerate(chat):
            if i and r["timestamp"] - chat[i - 1]["timestamp"] > window:
                session += 1                   # a pause of half an hour starts a new conversation
            if r["role"] != "assistant":
                continue
            reply = (r["content"] or "").strip()
            why = next((name for name, rx in DROP if rx.search(reply)), None)
            if not reply:
                why = "空回复"
            elif not why and edge(reply) >= 0.4:
                why = "语气太冲"
            if why:
                dropped[why] += 1
                continue
            context = []
            for prev in reversed(chat[max(0, i - context_turns):i]):
                if r["timestamp"] - prev["timestamp"] > window:
                    break
                text = (prev["content"] or "").strip()
                if prev["role"] == "assistant" and any(rx.search(text) for _, rx in DROP):
                    break                      # do not teach her to continue an old persona's line
                context.insert(0, {"role": prev["role"], "content": mask(text)})
            if not context or context[0]["role"] != "user":
                while context and context[0]["role"] != "user":
                    context.pop(0)
            if not context:
                dropped["没有上下文"] += 1
                continue
            key = (context[-1]["content"], reply)
            if key in seen:
                dropped["重复"] += 1
                continue
            seen.add(key)
            where = "群聊" if r["group_id"] else "私聊"
            examples.append({"messages": [{"role": "system", "content": f"{SYSTEM}（现在是{where}）"}, *context,
                                          {"role": "assistant", "content": mask(reply)}],
                             "chat": f"{key}#{session}"})
    # Hold out whole conversations: neighbouring examples share turns, so a random split would leak
    # held-out replies into the training contexts and make the check meaningless
    chats = sorted({e["chat"] for e in examples})
    random.Random(seed).shuffle(chats)
    val_chats, n = set(), 0
    for c in chats:
        if n >= val_share * len(examples) or len(examples) <= 10:
            break
        val_chats.add(c)
        n += sum(1 for e in examples if e["chat"] == c)
    val = [e for e in examples if e["chat"] in val_chats]
    train = [e for e in examples if e["chat"] not in val_chats]
    random.Random(seed).shuffle(train)
    n_val = len(val)
    for e in examples:
        e.pop("chat")
    os.makedirs(out, exist_ok=True)
    for name, part in (("val", val), ("train", train)):
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for e in part:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    report = {"kept": len(examples), "train": len(examples) - n_val, "val": n_val, "dropped": dict(dropped)}
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=os.path.join(ROOT, "memory.sqlite"))
    p.add_argument("--out", default=os.path.join(ROOT, "data", "lora"))
    a = p.parse_args(argv)
    print(json.dumps(build(a.db, a.out), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])
