"""Real temporary filesystem tasks with dependency, failure and hash verification.
No live files, API, generated code or model weights. Includes a simple-rule control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"minimal-self"))
from src.levels.l8_agency.planning import Planner


def task():
    return {"initial":{"prepared":0,"artifact":0},"goal":{"artifact":1},"actions":[
        {"name":"prepare","requires":{},"effects":{"prepared":1},"cost":.2},
        {"name":"fast","requires":{"prepared":1},"effects":{"artifact":1},"cost":1},
        {"name":"safe","requires":{"prepared":1},"effects":{"artifact":1},"cost":1.8}]}


def episode(planner, seed, low, mode, base):
    payload=random.Random(seed).randbytes(128)
    expected=hashlib.sha256(payload).hexdigest()
    folder=base/f"{mode}_{seed}";folder.mkdir()
    source,artifact=folder/'source.bin',folder/'artifact.bin'
    def observed():
        return {"prepared":int(source.exists() and source.read_bytes()==payload),
                "artifact":int(artifact.exists() and hashlib.sha256(artifact.read_bytes()).hexdigest()==expected)}
    cost, steps = 0.,0
    while steps<6 and not observed()["artifact"]:
        p=task();p["initial"]=observed()
        if mode in ('adaptive','without_self'):
            plan=planner.plan(p,self_state={"energy":0 if low else 100} if mode=='adaptive' else None)
            if plan["status"]!='predicted': break
            path=plan["steps"]
        else:
            name='prepare' if not p['initial']['prepared'] else ('safe' if low and mode=='simple_rule' else 'fast')
            path=[{"action":name}]
        for step in path:
            before=observed(); name=step['action']; steps+=1
            cost += next(a['cost'] for a in p['actions'] if a['name']==name)
            if name=='prepare': source.write_bytes(payload)
            elif name=='safe' or not low:
                if source.exists(): artifact.write_bytes(source.read_bytes())
            after=observed()
            if mode in ('adaptive','without_self'):
                feedback=planner.observe(plan['ticket'],name,before,after)
                if feedback['replan_required']: break
            if after['artifact'] or steps>=6: break
    return {'seed':seed,'low_energy':low,'mode':mode,'correct':bool(observed()['artifact']),
            'action_cost':cost,'actual_actions':steps,'sha256_verified':bool(observed()['artifact'])}


def run():
    planner=Planner()
    with tempfile.TemporaryDirectory(prefix='noa_plan_real_') as directory:
        base=Path(directory)
        training=[episode(planner,seed,seed%2==0,'adaptive',base) for seed in range(20)]
        ablated=Planner()
        for seed in range(20):
            episode(ablated,seed,seed%2==0,'without_self',base)
        # Frozen learned state evaluated on fresh file payloads. No online fitting in controls.
        frozen=planner.to_dict()
        rows=[]
        for seed in range(1000,1060):
            for mode in ['adaptive','without_self','fixed','simple_rule']:
                learner=Planner();learner.load_dict(ablated.to_dict() if mode=='without_self' else frozen)
                rows.append(episode(learner,seed,seed%2==0,mode,base))
        summary={m:{'tasks':len(v:=[r for r in rows if r['mode']==m]),
                    'success_rate':sum(r['correct'] for r in v)/len(v),
                    'mean_action_cost':sum(r['action_cost'] for r in v)/len(v)}
                 for m in ['adaptive','without_self','fixed','simple_rule']}
    return {'scope':'Real temporary file construction with hash verification; controlled actuator failures, not general intelligence.',
            'training_tasks':len(training),'test_payload_seeds':[1000,1059], 'summary':summary,'rows':rows,
            'limitations':['Cost units are declared action costs, not end-to-end latency or API money.',
                           'Simple-rule control has the known failure condition; no claim of superiority over it.',
                           'No human conversational or open-world benchmark.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],indent=2))
