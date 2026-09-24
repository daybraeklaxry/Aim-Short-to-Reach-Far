from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent.parent
for f in (OLD/'budget_code').iterdir():
 if f.suffix in ('.py','.sh') and f.name not in ('run.sh','run_extension.py'):shutil.copy2(f,ROOT/f.name)
shutil.copy2(OLD/'run_long.py',ROOT/'run_long_v2.py')
p=ROOT/'run_long_v2.py';s=p.read_text()
s=s.replace("assignments=[(o,c) for o in range(128) for c in p['conditions']] if a.mode=='evaluate' else [(0,'clean'),(0,'prefix_a')]", "assignments=[(o,c) for o in range(a.start,a.stop) for c in p['conditions']]")
s=s.replace("output=a.output/a.task", "output=a.output/a.task/f'q{a.start:03}-{a.stop:03}'")
s=s.replace("with (output/'outcomes.jsonl').open('x') as records:","existing={}\n        if (output/'outcomes.jsonl').exists():\n            for line in (output/'outcomes.jsonl').open():\n                prior=json.loads(line);existing[(prior['query_ordinal'],prior['condition'])]=prior\n        with (output/'outcomes.jsonl').open('a') as records:")
s=s.replace("row=p['studies'][a.task]['confirm'][ordinal]\n                trace,arrays", "row=p['studies'][a.task]['confirm'][ordinal]\n                if (ordinal,condition) in existing:\n                    completed+=1;continue\n                torch.cuda.reset_peak_memory_stats()\n                trace,arrays")
s=s.replace("stem=output/'traces'", "trace['compact']['peak_reserved_bytes']=torch.cuda.max_memory_reserved()\n                stem=output/'traces'")
s=s.replace("runtime.write(output/'summary.json',dict(completed=True,outcomes=completed,elapsed_seconds=time.perf_counter()-started))", "runtime.write(output/'summary.json',dict(completed=True,outcomes=completed,peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.perf_counter()-started))")
s=s.replace("p.add_argument('--output',type=Path,required=True);", "p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=128);p.add_argument('--output',type=Path,required=True);")
p.write_text(s,encoding='utf-8',newline='\n')
sh=(OLD/'budget_code/run.sh').read_text().replace('"$HERE/run_extension.py" "$@"','script="$1"\nshift\nexec "$PYTHON" "$HERE/$script" "$@"').replace('exec "$PYTHON" script=', 'script=')
# Keep environment selection based on the complete argument vector, then dispatch a named script.
(ROOT/'run.sh').write_text(sh,encoding='utf-8',newline='\n')
print('copied runtime; added resumable query-range LeWM runner')
