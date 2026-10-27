"""Read-only deployed acceptance probe; does not issue game or chat commands."""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=12);parser.add_argument('--output')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    headers={'X-Console-Token':cfg['CONSOLE_TOKEN']} if cfg.get('CONSOLE_TOKEN') else {}
    rates=[];samples=[];errors=[]
    for i in range(max(1,args.seconds)):
        try:
            req=urllib.request.Request('http://127.0.0.1:8080/console/api/outside',headers=headers)
            with urllib.request.urlopen(req,timeout=4) as resp:g=json.load(resp).get('game') or {}
            with urllib.request.urlopen('http://127.0.0.1:8765/motion',timeout=2) as resp:m=json.load(resp)
            body=g.get('body') or {};rates.append(float(body.get('tick_rate',0)))
            samples.append({'t':time.time(),'connected':g.get('connected') is True,'action':body.get('action'),'tick_rate':body.get('tick_rate'),
                            'rpc_p95_ms':body.get('rpc_p95_ms'),'telemetry':m.get('telemetry'),
                            'memory':g.get('game_memory'),'queue':g.get('memory_queue')})
        except Exception as exc:errors.append(type(exc).__name__)
        if i+1<args.seconds:time.sleep(1)
    latest=samples[-1] if samples else {};memory=latest.get('memory') or {};queue=latest.get('queue') or {}
    report={'checked_at':time.time(),'samples':len(samples),'body_hz':{'min':min(rates,default=0),'median':statistics.median(rates) if rates else 0,'max':max(rates,default=0)},
            'connected':bool(samples) and all(s.get('connected') for s in samples),'shared_embedder':memory.get('shared_embedder') is True,'embedding_backend':memory.get('backend'),
            'embedding_model':memory.get('model_id'),'memory_records':memory.get('records'),
            'fresh_public_telemetry':latest.get('telemetry') is True,'queue':queue,'errors':errors,
            'actions':sorted({str(s.get('action')) for s in samples}),'samples_detail':samples}
    report['passed']=bool(samples and report['connected'] and not errors and min(rates)>=10 and max(rates)<=21 and report['shared_embedder'] and report['fresh_public_telemetry'] and not queue.get('errors') and not queue.get('dropped'))
    if args.output:Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples_detail'},ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
