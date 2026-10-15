import json,subprocess,os,signal,time,sys,urllib.error
from pathlib import Path
repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash');sys.path.insert(0,str(repo))
from dev.benchmarks.decision_models import call,footprint,check_response
root=Path('/private/tmp/decision-models-extended-20261015');p=json.loads((root/'Julia-1-cases.json').read_text());case=next(c for c in p['cases'] if c['name']=='length_8192')
result=[]
for policy in ['auto','on']:
 cmd=['/Users/ghost233/.lmstudio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.56.0/llama-server','-m',p['model'],'--host','127.0.0.1','--port','18149','-c','8193','-np','1','-ngl','99','-b','8192','-ub','8192','-fa',policy,'--no-warmup','--embd-normalize','-1']
 log=Path('/private/tmp/decision-models-flash-'+policy+'.log').open('w');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 row={'policy':policy,'command':cmd}
 try:
  end=time.monotonic()+90
  while True:
   if proc.poll() is not None:raise RuntimeError((policy,proc.returncode))
   try:call(18149,'/health');break
   except (OSError,urllib.error.URLError):assert time.monotonic()<end;time.sleep(.5)
  for i in range(2):
   start=time.perf_counter();response=call(18149,'/v1/systemone',case['body']);elapsed=time.perf_counter()-start
   check_response(response,8192,case['body']);row[f'run{i}']={'seconds':elapsed,'response':response,'memory':footprint(proc.pid)}
  print(policy,'second response sec',row['run1']['seconds'],'memory',row['run1']['memory'],flush=True)
 finally:
  if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT);proc.wait(timeout=30)
  row['exit']=proc.returncode;log.close();result.append(row)
  Path('/private/tmp/decision-models-flash-policy.json').write_text(json.dumps(result,indent=2))
 assert proc.returncode==0
