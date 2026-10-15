import json,os,signal,subprocess,time,urllib.request,urllib.error
from pathlib import Path
root=Path.cwd();out=root/'build/worktree-preserved/20261015-ghost-integration/primary-smoke';out.mkdir(exist_ok=True)
ref=json.loads((root/'docs/data/upstream-main-merge-2026-10-15/controlled/correctness.json').read_text())['runs'];results={}
port=18210
def call(path,body=None):
 request=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=180) as r:return json.load(r)
for family,launcher in [('Kev-0.8B','kev08bsplash.sh'),('Julia-1','julia1splash.sh'),('Laya','layasplash.sh')]:
 profile=json.loads((root/'docs/data/decision-models-2026-10-15'/(family+'-cases.json')).read_text());case=profile['cases'][0]
 cmd=['sh',launcher,'--port',str(port)]
 with (out/(family+'.log')).open('w') as log:
  process=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   end=time.monotonic()+240
   while True:
    if process.poll() is not None:raise RuntimeError((family,process.returncode))
    if time.monotonic()>end:raise TimeoutError(family)
    try:call('/ready');break
    except (OSError,urllib.error.URLError):time.sleep(.3)
   answers=[call('/v1/systemone',case['body']) for _ in range(2)]
   for answer in answers:
    expected=ref[family]['new'][0]['response'];assert answer['usage']==expected['usage']
    for qid,a in answer['answers'].items():
     b=expected['answers'][qid];assert a['choice']==b['choice']
     assert all(abs(v-b['probabilities'][k])<=1e-3*max(1,abs(v)) for k,v in a['probabilities'].items())
   status=call('/status');results[family]={'command':cmd,'answers':answers,'status':status,'passed':True}
  finally:
   if process.poll() is None:os.killpg(process.pid,signal.SIGINT);process.wait(timeout=40)
   results.setdefault(family,{})['exit']=process.returncode;(out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');assert process.returncode==0
 text=(out/(family+'.log')).read_text();assert str(root/'build/splash') in text
 assert 'src-d22fd38e821706a8290ca95f7d89c1be5a97b764ef5f3728cd0aa9c52fc29704' in text
 print(family,'primary launcher default engine + actual answer/repeat PASS; exit0',flush=True)
print('PRIMARY DEFAULT LAUNCHERS VERIFIED',flush=True)
