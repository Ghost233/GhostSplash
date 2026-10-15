import json,subprocess,signal,time,sys,urllib.request,urllib.error
from pathlib import Path
repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash');sys.path.insert(0,str(repo))
from dev.benchmarks.decision_models import call,check_response
root=Path('/private/tmp/decision-models-extended-20261015');profile=json.loads((root/'Kev-0.8B-cases.json').read_text())
previous=json.loads((root/'correctness-final-defaults.json').read_text())['runs']['Kev-0.8B']['splash']
cmd=['/Users/ghost233/.lmstudio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.56.0/llama-server','-m',profile['model'],'--host','127.0.0.1','--port','18146','-c','8193','-np','1','-ngl','99','-b','2048','-ub','2048','-ctk','bf16','-ctv','bf16','--no-warmup','--embd-normalize','-1']
log=Path('/private/tmp/decision-models-kev-matched-reference.log').open('w')
p=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
out={'command':cmd,'runs':[],'performance_test':False}
try:
 end=time.monotonic()+120
 while True:
  assert p.poll() is None
  try:call(18146,'/health');break
  except (OSError,urllib.error.URLError):assert time.monotonic()<end;time.sleep(.5)
 for case,baseline in zip(profile['cases'],previous):
  response=call(18146,'/v1/systemone',case['body'])
  check_response(response,sum(x['tokens'] for x in case['positions'].values()),case['body'])
  for qid,answer in response['answers'].items():
   assert answer.get('choice')==baseline['response']['answers'][qid].get('choice'),case['name']
  out['runs'].append({'name':case['name'],'response':response});print(case['name'],'PASS',flush=True)
 out['status']='passed'
finally:
 if p.poll() is None:os_error=None;__import__('os').killpg(p.pid,signal.SIGINT);p.wait(timeout=30)
 out['exit']=p.returncode;log.close();Path('/private/tmp/decision-models-kev-matched-correctness.json').write_text(json.dumps(out,indent=2))
assert p.returncode==0
