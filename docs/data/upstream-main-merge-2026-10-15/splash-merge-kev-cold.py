import json,subprocess,time,urllib.request,os,signal
from pathlib import Path
repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash');root=Path('/private/tmp/splash-merge-kev-cold');root.mkdir(exist_ok=True)
modelroot=repo/'install/models-local/.resolved/00c7f8d515847de538982b11503a1f27bcb5542b74d61027a9a0c8d7f3d64968'
profile=json.loads((repo/'docs/data/decision-models-2026-10-15/Kev-0.8B-cases.json').read_text());cases=profile['cases'];result={}
def call(port,path,body=None):
 r=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=120) as f:return json.load(f)
for mode,cwd,binary in [('old','/private/tmp/splash-merge-premerge-python','build/decision-modernbert-20261015/splash'),('new',str(repo),'build/merge-upstream-20261015/splash')]:
 port=18156;cmd=[str(repo/'.venv/bin/python'),'-u','-m','server.server',str(modelroot),'--model',profile['model_id'],'--tokenizer',str(modelroot/'tokenizer'),'--binary',str(repo/binary),'--host','127.0.0.1','--port',str(port),'--max-context','8192','--max-memory',str(4<<30),'--kv-format','bf16','--disable-ane','--no-webui','--queue-size','1','--max-cache-disk','0','--compact-draft-head','off','--prompt-lookup','off','--fast-topk','off']
 with (root/(mode+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   end=time.monotonic()+150
   while True:
    if p.poll() is not None:raise RuntimeError((mode,p.returncode))
    if time.monotonic()>end:raise TimeoutError(mode)
    try:call(port,'/ready');break
    except (OSError,urllib.error.URLError):time.sleep(.3)
   out=[]
   for case in [cases[3],cases[3],*cases[:4]]:
    answer=call(port,'/v1/systemone',case['body']);out.append({'case':case['name'],'response':answer});print(mode,case['name'],answer['answers'],flush=True)
   result[mode]=out
  finally:
   if p.poll() is None:os.killpg(p.pid,signal.SIGINT);p.wait(timeout=40)
   result[mode+'_exit']=p.returncode
   (root/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
   assert p.returncode==0
print('Cold old/new comparison completed',flush=True)
