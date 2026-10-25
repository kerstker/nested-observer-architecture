"""
Level 2d: who and what is related to whom - people, pets, works, places, events - and who said so.

Facts were sentences filed under the person who said them. That cannot hold "罗马 and 白羽 are
roommates" (said by 罗马) together with "罗马 is moving" (said later) so that a question about 白羽
finds both; it cannot tell 小美's brother 小王 from 老郑's receptionist 小王; it cannot know that a
newer report replaced an older one, or that two people say different things.

So relations are kept as a small graph:
- an entity is a named someone/something, *scoped to where she learned of it*: a private chat
  (owner), a group, or public. The same name in two private chats is two different entities.
  QQ users are entities too (user_id), and so is she herself.
- a relation is (subject, predicate, object-or-value), with who said it, when, in which episode,
  and its status: current, superseded (a newer report replaced it) or disputed (someone said otherwise).

Relations are extracted from episodes while she consolidates (the model reads a batch of episodes
with the relations it already knows about the names in them, and says what is new, what replaces
what, and what contradicts what). When she answers, the entities named in the conversation and
their neighbourhood (two hops) come with their sources - only what this chat may see.
"""
import json
import logging
import re
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from src.core import clock
from src.levels.l7_tom.hearsay import is_sensitive

logger = logging.getLogger(__name__)

PRIVATE, GROUP, PUBLIC = "private", "group", "public"
SELF = "self"

def _ago(seconds: float) -> str:
    days = seconds / 86400
    if days < 1:
        return "今天"
    if days < 2:
        return "昨天"
    if days < 30:
        return f"{days:.0f} 天前"
    return f"{days / 30:.0f} 个月前"

class RelationGraph:
    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, aliases TEXT, kind TEXT, user_id TEXT,
            visibility TEXT, group_id TEXT, owner TEXT, first_seen REAL, last_seen REAL
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, subj INTEGER, pred TEXT, obj INTEGER, value TEXT,
            said_by TEXT, said_by_name TEXT, episode INTEGER, t REAL, visibility TEXT, group_id TEXT, owner TEXT,
            status TEXT DEFAULT 'current', replaced_by INTEGER, disputes INTEGER, note TEXT
        )''')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_subj ON relations(subj)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_obj ON relations(obj)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ent_user ON entities(user_id)")
        # Which episodes have been read for relations
        cols = [r[1] for r in conn.execute("PRAGMA table_info(episodes)")]
        if cols and "graphed" not in cols:
            conn.execute("ALTER TABLE episodes ADD COLUMN graphed INTEGER DEFAULT 0")
        conn.commit()
        conn.close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- scope ----------------------------------------------------------------------

    @staticmethod
    def scope_of(visibility: str, group_id: Optional[str], user_id: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
        if visibility == PRIVATE:
            return PRIVATE, None, user_id
        if visibility == GROUP and group_id:
            return GROUP, group_id, None
        return PUBLIC, None, None

    @staticmethod
    def _visible_sql(prefix: str = "") -> str:
        p = prefix
        return (f"({p}visibility = 'public' OR ({p}visibility = 'private' AND {p}owner = ?) "
                f"OR ({p}visibility = 'group' AND {p}group_id = ?))")

    # --- entities -------------------------------------------------------------------

    def ensure_self(self, names: Iterable[str], qq_ids: Iterable[str] = ()) -> int:
        conn = self._conn()
        r = conn.execute("SELECT id, aliases FROM entities WHERE user_id = ?", (SELF,)).fetchone()
        aliases = sorted({n for n in list(names) + [f"QQ{q}" for q in qq_ids] + list(qq_ids) if n})
        if r:
            have = set(json.loads(r["aliases"] or "[]"))
            if not set(aliases) <= have:
                conn.execute("UPDATE entities SET aliases = ? WHERE id = ?", (json.dumps(sorted(have | set(aliases)),
                                                                                       ensure_ascii=False), r["id"]))
            eid = r["id"]
        else:
            eid = conn.execute("INSERT INTO entities (name, aliases, kind, user_id, visibility, first_seen, last_seen) "
                               "VALUES ('乃爱', ?, 'self', ?, 'public', ?, ?)",
                               (json.dumps(aliases, ensure_ascii=False), SELF, clock.now(), clock.now())).lastrowid
        conn.commit()
        conn.close()
        return eid

    def user_entity(self, user_id: str, name: str) -> int:
        """A QQ user: one entity whatever chat they are in (their name may differ by group)."""
        conn = self._conn()
        r = conn.execute("SELECT id, name, aliases FROM entities WHERE user_id = ?", (user_id,)).fetchone()
        now = clock.now()
        if r:
            aliases = set(json.loads(r["aliases"] or "[]"))
            if name and name != r["name"] and name not in aliases:
                aliases.add(name)
                conn.execute("UPDATE entities SET aliases = ?, last_seen = ? WHERE id = ?",
                             (json.dumps(sorted(aliases), ensure_ascii=False), now, r["id"]))
                conn.commit()
            eid = r["id"]
        else:
            eid = conn.execute("INSERT INTO entities (name, aliases, kind, user_id, visibility, first_seen, last_seen) "
                               "VALUES (?, '[]', 'person', ?, 'public', ?, ?)", (name or user_id, user_id, now, now)).lastrowid
            conn.commit()
        conn.close()
        return eid

    def find(self, name: str, scope: Tuple[str, Optional[str], Optional[str]], users: Dict[str, str] = None) -> Optional[int]:
        """The entity this name means here: a known user by that name, else a named thing in this scope."""
        name = (name or "").strip()
        if not name:
            return None
        if users and name in users:
            return self.user_entity(users[name], name)
        vis, gid, owner = scope
        conn = self._conn()
        rows = conn.execute("SELECT id, name, aliases, visibility, user_id FROM entities WHERE name = ? OR aliases LIKE ?",
                            (name, f'%"{name}"%')).fetchall()
        conn.close()
        best = None
        for r in rows:
            if r["user_id"]:                                    # users and herself are known everywhere
                return r["id"]
        for r in rows:
            ok = (r["visibility"] == vis and ((vis == PRIVATE and self._owner(r["id"]) == owner)
                                              or (vis == GROUP and self._group(r["id"]) == gid) or vis == PUBLIC))
            if ok:
                best = r["id"]
        return best

    def _owner(self, eid: int) -> Optional[str]:
        conn = self._conn()
        r = conn.execute("SELECT owner FROM entities WHERE id = ?", (eid,)).fetchone()
        conn.close()
        return r["owner"] if r else None

    def _group(self, eid: int) -> Optional[str]:
        conn = self._conn()
        r = conn.execute("SELECT group_id FROM entities WHERE id = ?", (eid,)).fetchone()
        conn.close()
        return r["group_id"] if r else None

    def entity(self, name: str, kind: str, scope, users: Dict[str, str] = None) -> int:
        eid = self.find(name, scope, users)
        if eid:
            conn = self._conn()
            conn.execute("UPDATE entities SET last_seen = ? WHERE id = ?", (clock.now(), eid))
            conn.commit()
            conn.close()
            return eid
        vis, gid, owner = scope
        conn = self._conn()
        eid = conn.execute("INSERT INTO entities (name, aliases, kind, visibility, group_id, owner, first_seen, last_seen) "
                           "VALUES (?, '[]', ?, ?, ?, ?, ?, ?)",
                           (name, kind or "thing", vis, gid, owner, clock.now(), clock.now())).lastrowid
        conn.commit()
        conn.close()
        return eid

    # --- relations ------------------------------------------------------------------

    def add(self, subj: int, pred: str, obj: Optional[int], value: Optional[str], said_by: Optional[str],
            said_by_name: str, episode: Optional[int], t: float, scope, replaces: Iterable[int] = (),
            disputes: Optional[int] = None, note: str = "", single: bool = False, corrects: bool = False) -> Optional[int]:
        pred = (pred or "").strip()[:30]
        if not pred or (obj is None and not value):
            return None
        vis, gid, owner = scope
        conn = self._conn()
        same = conn.execute("SELECT id FROM relations WHERE subj = ? AND pred = ? AND COALESCE(obj, -1) = ? "
                            "AND COALESCE(value, '') = ? AND status = 'current'",
                            (subj, pred, obj if obj is not None else -1, value or "")).fetchone()
        if same:
            conn.close()
            return same["id"]
        replaces, rivals = list(replaces), []
        if single and not replaces and not disputes:
            # One value at a time (a date, a destination, a status): an older different value in the same
            # chat scope is either updated (same person again, or an explicit correction) or contested
            for r in conn.execute("SELECT id, said_by, said_by_name, t FROM relations WHERE subj = ? AND pred = ? "
                                  "AND status = 'current' AND visibility = ? AND COALESCE(group_id, '') = ? "
                                  "AND COALESCE(owner, '') = ? AND (COALESCE(obj, -1) != ? OR COALESCE(value, '') != ?)",
                                  (subj, pred, vis, gid or "", owner or "", obj if obj is not None else -1, value or "")):
                if r["t"] > t:
                    continue
                same_source = (said_by and r["said_by"] == said_by) or (not said_by and r["said_by_name"] == said_by_name)
                (replaces if same_source or corrects else rivals).append(r["id"])
        rid = conn.execute("INSERT INTO relations (subj, pred, obj, value, said_by, said_by_name, episode, t, visibility, "
                           "group_id, owner, status, disputes, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (subj, pred, obj, (value or None) and value[:120], said_by, said_by_name, episode, t, vis, gid,
                            owner, "disputed" if disputes else "current", disputes, note[:120])).lastrowid
        for old in replaces:
            conn.execute("UPDATE relations SET status = 'superseded', replaced_by = ? WHERE id = ? AND id != ?",
                         (rid, int(old), rid))
        for rival in ([int(disputes)] if disputes else []) + rivals:
            conn.execute("UPDATE relations SET status = 'disputed' WHERE id = ? AND status = 'current'", (rival,))
        if rivals:
            conn.execute("UPDATE relations SET status = 'disputed', disputes = ? WHERE id = ?", (rivals[0], rid))
        conn.commit()
        conn.close()
        return rid

    def _names(self, ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
        ids = list({i for i in ids if i is not None})
        if not ids:
            return {}
        conn = self._conn()
        rows = conn.execute(f"SELECT * FROM entities WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
        conn.close()
        return {r["id"]: dict(r) for r in rows}

    def relations_of(self, eids: Iterable[int], user_id: str, group_id: Optional[str], hops: int = 2,
                     limit: int = 30, include_old: bool = True) -> List[Dict[str, Any]]:
        """Relations around these entities that this chat may see, nearest first."""
        seen_rel: Dict[int, Dict[str, Any]] = {}
        frontier, visited = set(eids), set()
        conn = self._conn()
        for hop in range(hops):
            if not frontier:
                break
            ids = list(frontier)
            visited |= frontier
            rows = conn.execute(
                f"SELECT * FROM relations WHERE (subj IN ({','.join('?' * len(ids))}) OR obj IN ({','.join('?' * len(ids))})) "
                f"AND {self._visible_sql()} " + ("" if include_old else "AND status != 'superseded' ") + "ORDER BY t DESC",
                [*ids, *ids, user_id or "", group_id or ""]).fetchall()
            nxt = set()
            for r in rows:
                if r["id"] not in seen_rel:
                    seen_rel[r["id"]] = {**dict(r), "hop": hop}
                for e in (r["subj"], r["obj"]):
                    if e is not None and e not in visited:
                        nxt.add(e)
            frontier = nxt
        conn.close()
        rels = sorted(seen_rel.values(), key=lambda r: (r["hop"], -r["t"]))[:limit]
        return rels

    def render(self, rels: List[Dict[str, Any]], now: float = None) -> str:
        """Lines she can read: who - relation -> whom/what (who said it, when; replaced/disputed)."""
        now = clock.now() if now is None else now
        names = self._names([r["subj"] for r in rels] + [r["obj"] for r in rels])
        def nm(eid):
            e = names.get(eid)
            return "我" if e and e["user_id"] == SELF else (e["name"] if e else "?")
        by_id = {r["id"]: r for r in rels}
        lines = []
        for r in sorted(rels, key=lambda r: (r["subj"], r["pred"], r["t"])):
            target = nm(r["obj"]) if r["obj"] is not None else r["value"]
            src = "系统记录的事实" if r["said_by"] == SELF else f"{r['said_by_name'] or '有人'}说的"
            line = f"- {nm(r['subj'])} —{r['pred']}→ {target}（{src}，{_ago(now - r['t'])}）"
            if r["status"] == "superseded":
                newer = by_id.get(r["replaced_by"])
                line += "【已过时，被更新的说法取代】" if not newer else f"【已过时，后来改成：{nm(newer['obj']) if newer['obj'] is not None else newer['value']}】"
            elif r["status"] == "disputed":
                line += "【有不同说法】"
            if r["note"]:
                line += f"（{r['note']}）"
            lines.append(line)
        return "\n".join(lines)

    # --- finding what a conversation is about -------------------------------------------

    def mentioned(self, text: str, user_id: str, group_id: Optional[str]) -> List[int]:
        """Entities this chat may see whose name or alias appears in the text (2+ characters)."""
        conn = self._conn()
        rows = conn.execute(f"SELECT id, name, aliases FROM entities WHERE {self._visible_sql()}",
                            (user_id or "", group_id or "")).fetchall()
        conn.close()
        hits = []
        for r in rows:
            for n in [r["name"], *json.loads(r["aliases"] or "[]")]:
                if n and len(n) >= 2 and n in text:
                    hits.append(r["id"])
                    break
        return hits

    def context_for(self, text: str, user_id: str, nickname: str, group_id: Optional[str], limit: int = 18) -> str:
        eids = self.mentioned(text, user_id, group_id)
        conn = self._conn()
        me = conn.execute("SELECT id FROM entities WHERE user_id = ?", (user_id,)).fetchone()
        conn.close()
        rels = self.relations_of(eids, user_id, group_id, hops=2, limit=limit) if eids else []
        if me:                                  # what she knows about the person talking, one hop
            have = {r["id"] for r in rels}
            rels += [r for r in self.relations_of([me["id"]], user_id, group_id, hops=1, limit=6) if r["id"] not in have]
        return self.render(rels[:limit]) if rels else ""

    # --- learning relations from episodes ------------------------------------------------

    EXTRACT = """<|im_start|>system
你在帮乃爱（一个 QQ 聊天角色）整理她记忆里的人和事之间的关系。下面是一些经历（每段前面是编号、时间和是谁说的）。
找出里面提到的、以后可能用得上的关系：人和人（室友、同事、兄弟、恋人、朋友）、人和东西（养的宠物、在玩的游戏、喜欢的作品）、
计划和事件（谁什么时候要去哪、婚礼时间、生病、搬家），以及属性（职业、年级、在哪工作）。
- subject / object 用经历里的称呼；说话人说"我"就写说话人的名字；"你/乃爱"写"乃爱"。
- object 是另一个人或东西时 object_kind 填 person/pet/thing/place/work/org；是一段描述（时间、状态、职业）时 object_kind 填 value。
- predicate 用简短的中文（2-6 个字），站在 subject 的角度：小林 —结婚对象→ 阿珂；阿珂 —养的猫→ 团子；团子 —近况→ 生病去了医院。
- said_by 是说这话的人（经历里的说话人），episode 是编号。
- 下面"已知的关系"里如果有同一件事的旧说法，而这段经历比它新：新说法取代了旧的（改了计划、纠正了错误），填 replaces=[旧编号]；
  两种说法冲突、又不能确定谁对，填 disputes=旧编号。经历比已知说法更早的，不要填 replaces。
- single：这种关系同一时间只能有一个值（婚礼时间、要去的地方、近况、职业、年级）填 true；可以同时有好几个（朋友、室友、喜欢的作品）填 false。
- corrects：这句话是在明确纠正别人之前的说法（"他说错了，其实是……"）填 true。
- 不要写：性取向、疾病（宠物生病可以）、私生活、犯罪、欠债、住址电话这类关于人的敏感说法；也不要写乃爱自己的情绪。
- 没有值得记的就返回空列表。不要编造经历里没有的东西。

[已知的关系]
{known}

[经历]
{episodes}

Output JSON: {{"relations": [{{"subject": "...", "subject_kind": "person", "predicate": "...", "object": "...", "object_kind": "person|pet|thing|place|work|org|value", "said_by": "...", "episode": 编号, "replaces": [], "disputes": null, "single": true, "corrects": false, "note": "可选，比如'说问过本人'"}}]}}
<|im_end|>
<|im_start|>assistant
"""

    def pending_episodes(self, limit: int = 40) -> List[Dict[str, Any]]:
        conn = self._conn()
        # Newest first (what matters now), read in time order within the batch; old backlog comes later
        rows = conn.execute("SELECT id, t, user_id, group_id, kind, content, visibility FROM episodes "
                            "WHERE graphed = 0 AND kind IN ('dialogue', 'proactive') ORDER BY t DESC, id DESC LIMIT ?",
                            (limit,)).fetchall()
        conn.close()
        return sorted((dict(r) for r in rows), key=lambda e: (e["t"], e["id"]))

    def mark_graphed(self, ids: List[int]):
        if not ids:
            return
        conn = self._conn()
        conn.execute(f"UPDATE episodes SET graphed = 1 WHERE id IN ({','.join('?' * len(ids))})", ids)
        conn.commit()
        conn.close()

    def extract(self, llm: Callable[..., Dict[str, Any]], users: Callable[[Dict[str, Any]], Dict[str, str]],
                batch: int = 12, max_batches: int = 6) -> int:
        """Read episodes not yet read for relations. users(episode) -> {name: user_id} of the people in that chat."""
        added = 0
        for _ in range(max_batches):
            eps = self.pending_episodes(batch)
            if not eps:
                break
            # What she already knows about the names in this batch (each in its own chat's scope),
            # so the model can say what is new, what replaces what, and what contradicts what
            known_by_id: Dict[int, Dict[str, Any]] = {}
            for e in eps:
                ids = self.mentioned(e["content"], e["user_id"], e["group_id"])
                for r in self.relations_of(ids, e["user_id"], e["group_id"], hops=1, limit=20, include_old=False) if ids else []:
                    known_by_id[r["id"]] = r
            known_rels = sorted(known_by_id.values(), key=lambda r: -r["t"])[:40]
            known = "\n".join(f"[{r['id']}] " + self.render([r]).lstrip("- ") for r in known_rels) or "（还没有）"
            listing = "\n".join(f"[{e['id']}] ({_ago(clock.now() - e['t'])}{'，群 ' + e['group_id'] if e['group_id'] else '，私聊'}) "
                                f"{e['content'][:300]}" for e in eps)
            try:
                out = llm(self.EXTRACT.format(known=known, episodes=listing), max_tokens=2000, temperature=0.2) or {}
            except Exception as ex:
                logger.error(f"[Graph] extraction failed: {ex}")
                break
            if not isinstance(out.get("relations"), list):
                # Unparseable (often a truncated answer): try again with fewer episodes; a single episode
                # that still fails is set aside (graphed = 2) instead of blocking the queue or being lost silently
                if len(eps) > 3:
                    batch = max(3, len(eps) // 2)
                    logger.warning(f"[Graph] unreadable extraction, retrying with batches of {batch}")
                    continue
                conn = self._conn()
                conn.executemany("UPDATE episodes SET graphed = 2 WHERE id = ?", [(e["id"],) for e in eps])
                conn.commit()
                conn.close()
                continue
            by_id = {e["id"]: e for e in eps}
            known_ok = {r["id"] for r in known_rels}
            for rel in out.get("relations") or []:
                if not isinstance(rel, dict):
                    continue
                ep = by_id.get(rel.get("episode")) if isinstance(rel.get("episode"), int) else None
                if not ep or not rel.get("subject") or not rel.get("predicate") or not rel.get("object"):
                    continue
                line = f"{rel['subject']}{rel['predicate']}{rel['object']}"
                if is_sensitive(line) and not (rel.get("subject_kind") == "pet"):
                    continue
                scope = self.scope_of(ep["visibility"], ep["group_id"], ep["user_id"])
                names = users(ep)
                subj = self.entity(str(rel["subject"]), rel.get("subject_kind") or "person", scope, names)
                okind = rel.get("object_kind") or "value"
                obj, value = (None, str(rel["object"])) if okind == "value" else \
                    (self.entity(str(rel["object"]), okind, scope, names), None)
                said = str(rel.get("said_by") or "")
                said_uid = names.get(said) or (ep["user_id"] if said in ("", None) else None)
                # Only a newer report can replace or dispute an older one (the backlog is read after newer things)
                newer = {i for i in known_ok if known_by_id[i]["t"] <= ep["t"]}
                replaces = [int(i) for i in rel.get("replaces") or [] if isinstance(i, int) and i in newer]
                disputes = rel.get("disputes") if isinstance(rel.get("disputes"), int) and rel.get("disputes") in newer else None
                if self.add(subj, str(rel["predicate"]), obj, value, said_uid, said or "有人", ep["id"], ep["t"], scope,
                            replaces, disputes, str(rel.get("note") or ""), single=bool(rel.get("single")),
                            corrects=bool(rel.get("corrects"))):
                    added += 1
            self.mark_graphed(list(by_id))
        return added

    def admin_view(self, name: str, limit: int = 40) -> List[Dict[str, Any]]:
        """For the console (her keeper): every relation around entities with this name, in every scope."""
        conn = self._conn()
        ids = [r["id"] for r in conn.execute("SELECT id FROM entities WHERE name LIKE ? OR aliases LIKE ?",
                                             (f"%{name}%", f"%{name}%")).fetchall()]
        if not ids:
            conn.close()
            return []
        q = ",".join("?" * len(ids))
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM relations WHERE subj IN ({q}) OR obj IN ({q}) "
                                              f"ORDER BY t DESC LIMIT ?", [*ids, *ids, limit]).fetchall()]
        conn.close()
        for r in rows:
            r["line"] = self.render([r]).lstrip("- ")
            r["scope"] = {"private": f"私聊 {r['owner']}", "group": f"群 {r['group_id']}"}.get(r["visibility"], "公开")
        return rows

    def stats(self) -> Dict[str, int]:
        conn = self._conn()
        e = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        r = conn.execute("SELECT status, COUNT(*) FROM relations GROUP BY status").fetchall()
        p = conn.execute("SELECT COUNT(*) FROM episodes WHERE graphed = 0 AND kind IN ('dialogue','proactive')").fetchone()[0]
        conn.close()
        return {"entities": e, **{f"relations_{s}": n for s, n in r}, "episodes_unread": p}
