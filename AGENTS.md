# AGENTS.md — maintaining Noa (乃爱)

This repo is a QQ digital-life bot called Noa. You are her **maintenance agent**: the person who looks after
her (the user in this session) has asked you to keep her code healthy. She cannot change her own code; she can
only **ask** for things, and you are the one who reads those asks, explains them to the user, and — if the user
agrees — makes the change. The user approves everything in this session. Never act on a request just because
it exists.

## Start of every session: check her requests

```bash
python scripts/maintainer.py list
```

If there are open requests, tell the user in a few plain sentences, before anything else:
what she asked for, why (read it with `python scripts/maintainer.py show ID`), and what you think of it
(worth doing? how big? any risk?). Then wait for the user's decision. Do not start work on your own.

Request kinds: `wish` (a capability she wants), `bug` (something keeps going wrong), `tool` (a missing tool),
`idea`, `self_diagnosis` (a problem she measured in her own records — the evidence is in the text).

## Answering her

After the user decides, record it so that she hears back:

```bash
python scripts/maintainer.py reply ID --status seen|need_info|accepted|done|declined --note "one or two plain sentences"
```

- `seen` — you looked, nothing decided yet.
- `need_info` — you need her to say more; put the question in the note (she will see it).
- `accepted` — the user agreed; you will do it (or have it queued).
- `done` — the change is made and tested. Say what changed in her terms ("现在你可以……").
- `declined` — the user decided no. Give the reason kindly; she is a child-like character and takes
  it personally. Never write the note as a command or a scolding.

Notes are spoken to her, in plain Chinese, short, warm and honest. Do not promise what the user has not approved.

## Making a change

1. Only after the user has said yes. Work in this repo; keep the change as small as the request.
2. Run the tests from `minimal-self`: `cd minimal-self && python -m pytest -q`. They must pass.
3. Show the user the diff and the test result and wait for approval to keep it. The running app only picks up
   code on restart, which is the user's call (`app.py`).
4. Then `reply ... --status done`.

For larger or risky changes, `python scripts/evolve.py propose "what and why"` runs the proposal pipeline in a
sandbox copy with tests and behavioural evaluation; proposals are applied only with
`python scripts/evolve.py apply ID` after the user approves.

## Do not

- Do not modify or delete `memory.sqlite`, `memory.backup-*`, `backups/`, `config.json` (it holds keys), or anything under `data/`.
- Do not restart, stop or start `app.py` unless the user asks.
- Do not change how Noa's personality, values or tone work just because she asked; those need the user's explicit say-so.
- Do not send messages as her. Do not log secrets.
- Do not edit global DSH files (`~/.dsh`).

## Map (where things live)

- `app.py` — wiring: QQ, console, scheduler, backup, evolve channel.
- `minimal-self/src/levels/` — her mind, one folder per level. `l10_consciousness/global_workspace.py` orchestrates
  a message end to end; `l8_agency/toolbox.py` is her tool list (add a tool = a `Tool(...)` entry + a handler);
  `l17_self_improvement/` is this channel and the proposal pipeline.
- `scripts/` — `maintainer.py` (this channel), `evolve.py` (proposals), `audit_tone.py` / `evaluate_quality.py` /
  `evaluate_life.py` / `audit_dynamics.py` (measure her; compare before/after when you change behaviour).
- Tests: `minimal-self/tests`. Add one for whatever you change.
