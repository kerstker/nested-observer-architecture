"""命令行管理她的自我改造提案。

用法（在 thefirstlife-dev 目录下）：
    python scripts/evolve.py list                    # 看所有提案
    python scripts/evolve.py show <id>               # 看某个提案的补丁和测试结果
    python scripts/evolve.py apply <id>              # 把补丁打到线上代码（要重启 app.py 才生效）
    python scripts/evolve.py propose "要改什么"      # 手动让 DSH 做一份提案（不经过她）

提案生成后不会自动生效；apply 只改代码，不重启进程。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))


def _bridge():
    from src.levels.l17_self_improvement.agent_bridge import DshAgentBridge
    config = {}
    path = os.path.join(ROOT, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    return DshAgentBridge(ROOT,
                          dsh_exe=config.get("DSH_EXE") or None,
                          dsh_bin=config.get("DSH_BIN") or None,
                          max_per_day=int(config.get("EVOLVE_MAX_PER_DAY", 1)))


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    bridge = _bridge()

    if cmd == "list":
        rows = bridge.list_proposals(50)
        if not rows:
            print("还没有提案。")
            return 0
        for r in rows:
            print(f"{r['id']}  {r['status']:<24} +{r['added']}/-{r['removed']}  "
                  f"tests {r['tests_total'] - r['tests_failed']}/{r['tests_total']}  "
                  f"by {r['requested_by']}  {(r['task'] or '')[:40]}")
        return 0

    if cmd == "show" and rest:
        rows = [r for r in bridge.list_proposals(200) if r["id"] == rest[0]]
        if not rows:
            print("没有这个提案")
            return 1
        r = rows[0]
        print(json.dumps({k: r[k] for k in r.keys() if k != "patch_path"}, ensure_ascii=False, indent=1))
        if r.get("patch_path") and os.path.exists(r["patch_path"]):
            with open(r["patch_path"], encoding="utf-8") as f:
                print("\n--- patch ---\n" + f.read()[:8000])
        return 0

    if cmd == "apply" and rest:
        result = bridge.apply(rest[0], "cli")
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0 if result.get("ok") else 1

    if cmd == "propose" and rest:
        result = bridge.propose(" ".join(rest), "cli")
        print(json.dumps({k: v for k, v in result.items() if k not in ("tests_log", "agent_text")},
                         ensure_ascii=False, indent=1))
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
