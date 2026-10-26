"""
Look at, check and restore her memory backups.

    python scripts/restore_backup.py list
    python scripts/restore_backup.py check [NAME]          # newest if no name: integrity, counts, and a trial restore
    python scripts/restore_backup.py restore NAME          # writes memory.restored-<time>.sqlite next to the live one
    python scripts/restore_backup.py restore NAME --in-place   # replaces memory.sqlite (app.py must be stopped)

`--in-place` first keeps the current memory.sqlite as memory.before-restore-<time>.sqlite, so a restore
can itself be undone.
"""
import argparse
import os
import shutil
import socket
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "minimal-self"))

from src.core.backup import BackupManager  # noqa: E402

def app_is_running(port: int = 8080) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["list", "check", "restore"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--db", default=os.path.join(ROOT, "memory.sqlite"))
    parser.add_argument("--folder")
    args = parser.parse_args()
    mgr = BackupManager(args.db, folder=args.folder)
    backups = mgr.list()

    if args.action == "list":
        for b in backups:
            print(f"{b['name']}  {b['bytes'] / 1e6:6.1f} MB  {time.strftime('%Y-%m-%d %H:%M', time.localtime(b['t']))}")
        print(f"{len(backups)} backups in {mgr.folder}")
        return

    pick = next((b for b in backups if b["name"] == args.name), None) if args.name else (backups[-1] if backups else None)
    if not pick:
        sys.exit("no such backup (use 'list')")
    check = mgr.verify(pick["path"])
    trial = mgr.restore_check(pick["path"])
    print(f"{pick['name']}: integrity/tables {'OK' if check['ok'] else 'PROBLEMS ' + str(check['problems'])}; "
          f"counts {check['counts']}; trial restore {trial}")
    if args.action == "check":
        sys.exit(0 if check["ok"] and trial.get("ok") else 1)

    if not (check["ok"] and trial.get("ok")):
        sys.exit("refusing to restore a backup that does not check out")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if not args.in_place:
        target = os.path.join(os.path.dirname(os.path.abspath(args.db)), f"memory.restored-{stamp}.sqlite")
        shutil.copy2(pick["path"], target)
        print(f"restored to {target}\nto use it: stop app.py, then copy it over memory.sqlite (or run again with --in-place)")
        return
    if app_is_running():
        sys.exit("app.py is running on port 8080: stop it first, a running app would overwrite the restored file")
    keep = os.path.join(os.path.dirname(os.path.abspath(args.db)), f"memory.before-restore-{stamp}.sqlite")
    if os.path.exists(args.db):
        shutil.copy2(args.db, keep)
        print(f"current memory kept as {keep}")
    for ext in ("-wal", "-shm", "-journal"):
        if os.path.exists(args.db + ext):
            os.remove(args.db + ext)
    shutil.copy2(pick["path"], args.db)
    print(f"memory.sqlite is now {pick['name']}")

if __name__ == "__main__":
    main()
