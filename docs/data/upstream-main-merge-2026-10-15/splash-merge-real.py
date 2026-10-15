import json,math,os,signal,subprocess,sys,time,urllib.request,urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash')
root=Path('/private/tmp/splash-merge-real-models-20261015')
reference='/Users/ghost233/.lmstudio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.56.0/llama-server'
binary=str(repo/'build/merge-upstream-20261015/splash')
result=json.loads((root/'correctness.json').read_text())
reference_runs={name:r['reference'] for name,r in result['runs'].items()}
premerge={name:r['splash'] for name,r in result['runs'].items()}

def call(port,path,body=None):
 request=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
def save():(root/'correctness-final-defaults.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
def valid(response,case):
 assert response['usage']['output_tokens']==0
 assert response['usage']['input_tokens']==sum(x['tokens'] for x in case['positions'].values()),(case['name'],'token count',response['usage'],case['positions'])
 assert set(response['answers'])==set(case['body']['questions'])
 for qid,answer in response['answers'].items():
  if answer['type']=='noul':assert math.isfinite(answer['noul']) and 0<=answer['noul']<=1
  else:
   probs=answer['probabilities'];assert all(math.isfinite(x) and 0<=x<=1 for x in probs.values());assert abs(sum(probs.values())-1)<1e-6
   if answer['type']=='choice':assert answer['choice'] in probs and probs[answer['choice']]==max(probs.values())
def same(a,b):
 for qid,answer in a['answers'].items():
  reference=b['answers'][qid]
  assert answer.get('choice')==reference.get('choice'),('choice mismatch',answer,reference)
  if 'probabilities' in answer:
   for key,value in answer['probabilities'].items():assert abs(value-reference['probabilities'][key])<=1e-3*max(1,abs(value)),(qid,key,value,reference)
  else:assert abs(answer['noul']-reference['noul'])<=1e-3
for family in ['Kev-0.8B','Julia-1','Laya']:
 profile=json.loads((root/(family+'-cases.json')).read_text())
 cases=profile['cases'];result['runs'][family]['reference']=reference_runs[family]
 for mode in (['splash'] if family=='Kev-0.8B' else ['splash','splash_full_head']):
  label=family+'-'+mode+'-defaults';port=18155
  command=[reference,'-m',profile['model'],'--host','127.0.0.1','--port',str(port),'-c','8193','-np','1','-ngl','99','-b','8192','-ub','8192','--no-warmup','--embd-normalize','-1'] if mode=='reference' else ['sh',profile['launcher'],'--binary',binary,'--port',str(port),'--queue-size','4']
  if mode=='splash_full_head': command += ['--encoder-option-queries','off']
  log=(root/(label+'.log')).open('w');process=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print(label,process.pid,'START',flush=True)
  try:
   end=time.monotonic()+180
   while True:
    if process.poll() is not None:raise RuntimeError((label,'startup exit',process.returncode))
    if time.monotonic()>end:raise TimeoutError(label+' startup timeout')
    try:call(port,'/health' if mode=='reference' else '/ready');break
    except (OSError,urllib.error.URLError):time.sleep(.5)
   outputs=[]
   for case in cases:
    response=call(port,'/v1/systemone',case['body']);valid(response,case)
    outputs.append({'name':case['name'],'response':response});result['runs'][family][mode]=outputs;save()
    if mode=='splash': same(response,premerge[family][len(outputs)-1]['response'])
    if mode=='splash_full_head': same(response,result['runs'][family]['splash'][len(outputs)-1]['response'])
    print(label,case['name'],'PASS',response['usage'],flush=True)
   if mode=='splash':
    repeat=call(port,'/v1/systemone',cases[-2]['body']);same(repeat,outputs[-2]['response'])
    # Distinct documents, option order and question types share one backend/pool.
    with ThreadPoolExecutor(max_workers=4) as pool:
     concurrent=list(pool.map(lambda case:call(port,'/v1/systemone',case['body']),cases[:4]))
    for index,response in enumerate(concurrent):valid(response,cases[index]);same(response,outputs[index]['response'])
    result['runs'][family]['concurrent4']=concurrent
    status=call(port,'/status');result['runs'][family]['status']=status
    if family!='Kev-0.8B':
     memory=status['memory'];assert memory['kv_bytes']==0 and memory['fp16_weight_cache_bytes']==0
     assert memory['allocated_bytes']<=memory['max_memory_bytes']
     assert memory['loaded_weight_bytes']==memory['quantized_weight_bytes']+memory['auxiliary_weight_bytes']
    print(label,'repeat/concurrent4/memory PASS',flush=True);save()
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=40)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
   result['cleanup'][label]={'exit':process.returncode};save();log.close()
   assert process.returncode==0,(label,'cleanup exit',process.returncode)
   print(label,'EXIT',process.returncode,flush=True)
 comparisons=[]
 for actual,expected in zip(result['runs'][family]['splash'],result['runs'][family]['reference']):
  differences=[]
  for qid,answer in actual['response']['answers'].items():
   reference_answer=expected['response']['answers'][qid]
   if answer['type']=='choice':assert answer['choice']==reference_answer['choice'],(family,actual['name'],'reference choice mismatch',answer,reference_answer)
   if answer['type']=='noul':differences.append(abs(answer['noul']-reference_answer['noul']))
   else:differences.extend(abs(v-reference_answer['probabilities'][key]) for key,v in answer['probabilities'].items())
  comparisons.append({'name':actual['name'],'maximum_probability_delta':max(differences)})
 result['runs'][family]['comparison']=comparisons;save()
 print(family,'REFERENCE CHOICES/TOKEN COUNTS PASS; maximum probability delta',max(x['maximum_probability_delta'] for x in comparisons),flush=True)
result['status']='passed';save()
print('ALL EXTENDED CORRECTNESS PASS',flush=True)
