"""
The maintainer's side of the request channel: what Noa has asked for, and your answer.

    python scripts/maintainer.py list [--status open|active|done|...] [--all]
    python scripts/maintainer.py show ID
    python scripts/maintainer.py reply ID --status seen|need_info|accepted|done|declined --note "text she will be told"

Meant to be run by the maintenance agent in DeepSeek Harness (see AGENTS.md) or by hand. A reply is
something she is told the next time she is awake: keep the note short and in plain words.
"""
import argparse
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))
sys.path.insert(0, ROOT)

from src.levels.l17_self_improvement.maintainer import STATUSES, Maintainer  # noqa: E402


def stamp(t):
    return time.strftime("%m-%d %H:%M", time.localtime(t))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.path.join(ROOT, "memory.sqlite"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--status", default="active", help="active (default), all, or one status")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("show")
    p.add_argument("id")
    p = sub.add_parser("reply")
    p.add_argument("id")
    p.add_argument("--status", required=True, choices=STATUSES)
    p.add_argument("--note", default="")
    args = parser.parse_args(argv)
    m = Maintainer(args.db)

    if args.cmd == "list":
        status = None if (args.all or args.status == "all") else args.status
        rows = m.list(status)
        if not rows:
            print("没有请求。")
        for r in rows:
            print(f"{r['id']}  [{r['status']:9}] {stamp(r['created'])}  {r['kind']:14} {r['title']}  （{r['requester_name'] or r['requester']}）")
    elif args.cmd == "show":
        r = m.get(args.id)
        if not r:
            print("没有这个请求。")
            return 1
        print(f"{r['id']}  状态：{r['status']}  类型：{r['kind']}  提出：{stamp(r['created'])}  来自：{r['requester_name'] or r['requester']}")
        print(f"标题：{r['title']}")
        print(f"详情：{r['body'] or '（没写）'}")
        if r["response"]:
            print(f"已回复（{stamp(r['updated'])}）：{r['response']}")
    elif args.cmd == "reply":
        if not m.reply(args.id, args.status, args.note):
            print("没有这个请求。")
            return 1
        print(f"已回复 {args.id}：{args.status}。她下次醒着的时候会知道。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
