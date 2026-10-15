import json,math,os,signal,subprocess,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash');data=repo/'docs/data/decision-models-2026-10-15';root=Path('/private/tmp/splash-merge-real-controlled');root.mkdir(exist_ok=True)
roots={}
for p in (repo/'install/models-local/.resolved').glob('*/model.json'):
 d=json.loads(p.read_text());roots[d.get('family')]=p.parent
result={'performance_test':False,'runs':{},'cleanup':{},'status':'running'}
def save():(root/'correctness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
def call(port,path,body=None):
 request=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=180) as f:return json.load(f)
def same(a,b):
 assert a['usage']==b['usage']
 for qid,answer in a['answers'].items():
  ref=b['answers'][qid];assert answer['type']==ref['type']
  if answer['type']=='choice':assert answer['choice']==ref['choice']
  if answer['type']=='noul':assert abs(answer['noul']-ref['noul'])<=1e-3
  else:
   for k,v in answer['probabilities'].items():assert abs(v-ref['probabilities'][k])<=1e-3*max(1,abs(v)),(qid,k,v,ref)
def valid(answer,case):
 assert answer['usage']=={'input_tokens':sum(x['tokens'] for x in case['positions'].values()),'output_tokens':0}
 for a in answer['answers'].values():
  if a['type']=='noul':assert math.isfinite(a['noul']) and 0<=a['noul']<=1
  else:
   assert all(math.isfinite(x) and 0<=x<=1 for x in a['probabilities'].values());assert abs(sum(a['probabilities'].values())-1)<1e-6
archived=json.loads((data/'correctness-final-defaults.json').read_text())['runs']
for family in ['Kev-0.8B','Julia-1','Laya']:
 profile=json.loads((data/(family+'-cases.json')).read_text());cases=profile['cases'];result['runs'][family]={}
 for mode in (['old','new'] if family=='Kev-0.8B' else ['old','new','new_full_head']):
  old=mode=='old';port=18157;modelroot=roots[family];label=family+'-'+mode
  binary=repo/('build/decision-modernbert-20261015/splash' if old else 'build/merge-upstream-20261015/splash')
  command=[str(repo/'.venv/bin/python'),'-u','-m','server.server',str(modelroot),'--model',profile['model_id'],'--tokenizer',str(modelroot/'tokenizer'),'--binary',str(binary),'--host','127.0.0.1','--port',str(port),'--max-context','8192','--max-memory',str(4<<30),'--kv-format','bf16','--disable-ane','--no-webui','--queue-size','4','--max-cache-disk','0','--compact-draft-head','off','--prompt-lookup','off','--fast-topk','off']
  if mode=='new_full_head':command+=['--encoder-option-queries','off']
  with (root/(label+'.log')).open('w') as log:
   process=subprocess.Popen(command,cwd='/private/tmp/splash-merge-premerge-python' if old else repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   print(label,'START',process.pid,flush=True)
   try:
    end=time.monotonic()+180
    while True:
     if process.poll() is not None:raise RuntimeError((label,'startup exit',process.returncode))
     if time.monotonic()>end:raise TimeoutError(label)
     try:call(port,'/ready');break
     except (OSError,urllib.error.URLError):time.sleep(.3)
    outputs=[]
    for i,case in enumerate(cases):
     response=call(port,'/v1/systemone',case['body']);valid(response,case);outputs.append({'name':case['name'],'response':response})
     if mode!='old':same(response,result['runs'][family]['old'][i]['response'])
     ref=archived[family]['reference'][i]['response']
     for qid,a in response['answers'].items():
      if a['type']=='choice':assert a['choice']==ref['answers'][qid]['choice']
     result['runs'][family][mode]=outputs;save();print(label,case['name'],'PASS',flush=True)
    repeat=call(port,'/v1/systemone',cases[-2]['body']);same(repeat,outputs[-2]['response'])
    with ThreadPoolExecutor(max_workers=4) as pool: parallel=list(pool.map(lambda case:call(port,'/v1/systemone',case['body']),cases[:4]))
    for i,response in enumerate(parallel):valid(response,cases[i]);same(response,outputs[i]['response'])
    status=call(port,'/status');result['runs'][family][mode+'_status']=status
    if family!='Kev-0.8B':
     m=status['memory'];assert m['kv_bytes']==0 and m['fp16_weight_cache_bytes']==0
     assert m['allocated_bytes']<=m['max_memory_bytes']
     assert m['loaded_weight_bytes']==m['quantized_weight_bytes']+m['auxiliary_weight_bytes']
    result['runs'][family][mode+'_concurrent4']=parallel;save();print(label,'repeat/concurrent4/memory PASS',flush=True)
   finally:
    if process.poll() is None:os.killpg(process.pid,signal.SIGINT);process.wait(timeout=40)
    result['cleanup'][label]={'exit':process.returncode};save();assert process.returncode==0
    print(label,'EXIT',process.returncode,flush=True)
result['status']='passed';save();print('CONTROLLED OLD/NEW CORRECTNESS PASSED',flush=True)
