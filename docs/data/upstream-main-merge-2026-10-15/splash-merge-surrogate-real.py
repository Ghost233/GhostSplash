import json,os,signal,subprocess,time,urllib.request,urllib.error
from pathlib import Path
repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash');out=Path('/private/tmp/splash-merge-surrogate-real');out.mkdir(exist_ok=True);result={}
roots={json.loads(p.read_text()).get('family'):p.parent for p in (repo/'install/models-local/.resolved').glob('*/model.json')}
def call(path,body=None):
 request=urllib.request.Request('http://127.0.0.1:18158'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=120) as f:return json.load(f)
for family,launcher in [('Kev-0.8B','kev08bsplash.sh'),('Julia-1','julia1splash.sh'),('Laya','layasplash.sh')]:
 cmd=['sh',launcher,'--binary',str(repo/'build/merge-upstream-20261015/splash'),'--port','18158']
 with (out/(family+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   end=time.monotonic()+180
   while True:
    if p.poll() is not None:raise RuntimeError((family,p.returncode))
    if time.monotonic()>end:raise TimeoutError(family)
    try:call('/ready');break
    except (OSError,urllib.error.URLError):time.sleep(.3)
   dirty={'model':json.loads((repo/'docs/data/decision-models-2026-10-15'/(family+'-cases.json')).read_text())['model_id'],'state':'document \ud800','questions':{'q':{'type':'choice','instructions':'choose \udfff','criteria':{'a':'first','b':'second'}}}}
   clean=json.loads(json.dumps(dirty).replace('\\ud800','\\ufffd').replace('\\udfff','\\ufffd'))
   actual=call('/v1/systemone',dirty);expected=call('/v1/systemone',clean)
   assert actual['usage']==expected['usage']
   a=actual['answers']['q'];b=expected['answers']['q'];assert a['choice']==b['choice']
   assert all(abs(v-b['probabilities'][k])<=1e-3*max(1,abs(v)) for k,v in a['probabilities'].items())
   result[family]={'actual':actual,'replacement':expected,'status':'passed'}
  finally:
   if p.poll() is None:os.killpg(p.pid,signal.SIGINT);p.wait(timeout=40)
   result.setdefault(family,{})['exit']=p.returncode
   (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
   assert p.returncode==0
 print(family,'surrogate/replacement and token-count comparison PASS; native exit0',flush=True)
