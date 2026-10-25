import time
import random
from src.levels.l2_memory.storage.database import MemoryDatabase
from src.analysis.group_relations import RealTimeGroupRelationAnalyzer

def synthetic_event(group_id, sender_id, mentions, text="你好"):
    return {
        "message_type": "group",
        "group_id": group_id,
        "message_id": random.randint(100000, 999999),
        "sender": {"user_id": int(sender_id), "nickname": f"user{sender_id}"},
        "message": [{"type": "text", "data": {"text": text}}] + [{"type": "at", "data": {"qq": int(m)}} for m in mentions],
        "raw_message": text
    }

def main():
    db = MemoryDatabase("memory.sqlite")
    analyzer = RealTimeGroupRelationAnalyzer(db)
    group_id = "g1"
    users = [str(i) for i in range(50)]
    start = time.time()
    # simulate 2000 messages
    for i in range(2000):
        sender = random.choice(users)
        mentions = random.sample(users, k=random.randint(0, 2))
        evt = synthetic_event(group_id, sender, mentions, text="测试消息"+str(i))
        analyzer.on_message(evt)
    elapsed = time.time() - start
    print(f"Processed 2000 messages in {elapsed*1000:.1f} ms (~{2000/elapsed:.1f} msg/s)")
    # Check snapshot generation latency
    t0 = time.time()
    snap = db.get_group_snapshot(group_id, limit=100)
    t1 = time.time()
    print(f"Snapshot edges: {len(snap)}, query time: {(t1-t0)*1000:.2f} ms")

if __name__ == "__main__":
    main()
