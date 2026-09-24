"""Schedule frozen policies by query-condition pair; never select outcomes."""
import argparse,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True);a=p.parse_args()
protocol=json.loads((a.study_root/'protocol.json').read_text());out=a.study_root/'evaluation';assert not out.exists();out.mkdir()
workers=[]
for ti,task in enumerate(('cube','pusht','reacher','tworoom')):
    for shard in range(2):
        gpu=2*ti+shard;log=(out/f'worker_{task}_{shard}.log').open('w')
        cmd=['bash',str(a.study_root/'frozen_code/run.sh'),'--study-root',str(a.study_root),'--output',str(out),
             '--task',task,'--shards','2','--shard-index',str(shard)]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);log.close()
        workers.append(dict(task=task,shard=shard,gpu=gpu,pid=proc.pid,command=cmd))
receipt=dict(launched_utc=datetime.now(timezone.utc).isoformat(),expected_outcomes=protocol['expected_outcomes'],workers=workers)
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
