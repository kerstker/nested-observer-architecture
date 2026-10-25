import argparse
from src.levels.l2_memory.storage.database import MemoryDatabase

def main():
    parser = argparse.ArgumentParser(description="Visualize group relations as Graphviz DOT")
    parser.add_argument("--db", default="memory.sqlite", help="Path to SQLite DB")
    parser.add_argument("--group", required=True, help="Group ID")
    parser.add_argument("--limit", type=int, default=100, help="Max edges")
    args = parser.parse_args()

    db = MemoryDatabase(args.db)
    edges = db.get_group_snapshot(args.group, limit=args.limit)
    members = set()
    for e in edges:
        members.add(e["user_id_a"])
        members.add(e["user_id_b"])

    print("graph G {")
    print('  graph [overlap=false];')
    for m in members:
        print(f'  "{m}";')
    for e in edges:
        a = e["user_id_a"]; b = e["user_id_b"]
        w = max(1, int(e["weight"]))
        s = e["sentiment"]
        color = "green" if s > 0 else ("red" if s < 0 else "gray")
        print(f'  "{a}" -- "{b}" [penwidth={w}, color="{color}"];')
    print("}")

if __name__ == "__main__":
    main()
