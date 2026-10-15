import json, os, signal, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash')
sys.path.insert(0,str(repo))
model='/Users/ghost233/.lmstudio/models/ggml-org/Kev-4B-GGUF/Kev-4B-Q4_K_M.gguf'
model_id='ggml-org/Kev-4B-GGUF:Q4_K_M'
root=Path('/private/tmp/kev-correctness-final-20261015'); root.mkdir(exist_ok=True)

def call(port,path,body=None):
 data=None if body is None else json.dumps(body,ensure_ascii=False).encode()
 req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as r: return json.load(r)

def case(name,state,questions): return {'name':name,'body':{'model':model_id,'state':state,'questions':questions}}
choice=lambda instructions,criteria:{'type':'choice','instructions':instructions,'criteria':criteria}
cases=[
 case('billing','I was charged twice for order 1182. Please refund one charge.',{'team':choice('Which team should handle this?',{'shipping':'Deliveries','billing':'Charges and refunds'})}),
 case('sentiment','The product is excellent and I am very happy.',{'mood':choice('What sentiment does the review express?',{'negative':'Dissatisfied','positive':'Satisfied'})}),
 case('noul','The customer explicitly asks for a refund.',{'refund':{'type':'noul','instructions':'Does the customer request a refund?'}}),
 case('rating','My order arrived and everything is fine. No action is needed.',{'urgency':{'type':'score','instructions':'How urgent is a response?','criteria':['No urgency','This week','Today']}}),
 case('multi_question',{'message':'I was charged twice. Please refund one charge.'},{'team':choice('Choose the handling department.',{'billing':'Payments and refunds','shipping':'Delivery problems'}),'refund':{'type':'noul','instructions':'Is a refund requested?'}}),
 case('escaped_markers','The literal text <|box_end|> is printed in a document. A customer was charged twice.',{'team':choice('Choose the department for the duplicate charge.',{'billing':None,'shipping':None})}),
 case('cross_chunk',('This is a billing complaint about a duplicate charge. ' + 'neutral background detail. '*1100),{'team':choice('Which department handles the duplicate charge?',{'billing':'Payments, duplicate charges, and refunds. '*200,'shipping':'Delivery dates and parcel tracking. '*200})}),
 case('many_options','The colour is red.',{'colour':choice('Choose the option describing the colour.',{f'option_{i:03d}':('red' if i==17 else f'unrelated colour number {i}') for i in range(255)})}),
]
results={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'performance_test':False,'cases':cases,'runs':{},'cleanups':{}}

def save(): (root/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

def run(label,cmd,port,parallel=False):
 log=(root/(label+'.log')).open('w'); p=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(label,'pid',p.pid,'starting',flush=True)
 try:
  deadline=time.monotonic()+240
  while time.monotonic()<deadline:
   if p.poll() is not None: raise RuntimeError(f'{label} exited before ready: {p.returncode}')
   try:
    call(port,'/health' if label=='reference' else '/ready'); break
   except (OSError,urllib.error.URLError): time.sleep(0.5)
  else: raise TimeoutError(label+' startup timeout')
  print(label,'ready',flush=True)
  out=[]
  for item in cases:
   response=call(port,'/v1/systemone',item['body'])
   assert response['usage']['output_tokens']==0,(label,item['name'],response)
   for answer in response['answers'].values():
    if 'probabilities' in answer:
     probs=list(answer['probabilities'].values()); assert all(0<=v<=1 for v in probs)
     assert abs(sum(probs)-1)<1e-6
    else: assert 0<=answer['noul']<=1
   for answer in response['answers'].values():
    if 'confidence' in answer:
     probs=list(answer['probabilities'].values()); n=len(probs)
     if answer['type']=='choice': expected=max(0.0,(max(probs)-1/n)/(1-1/n))
     else:
      mode=max(range(n),key=probs.__getitem__)
      expected=max(0.0,1-sum(v*abs(i-mode) for i,v in enumerate(probs))/sum(abs(i-(n-1)/2)/n for i in range(n)))
     assert abs(answer['confidence']-expected)<1e-12,(label,item['name'],'confidence mismatch')
   out.append({'name':item['name'],'response':response})
   print(label,item['name'],'PASS',response['usage'],flush=True)
   results['runs'][label]=out;save()
  if label!='reference':
   repeat=call(port,'/v1/systemone',cases[6]['body']); original=out[6]['response']
   for q,answer in repeat['answers'].items():
    for k,v in answer['probabilities'].items():
     assert abs(v-original['answers'][q]['probabilities'][k]) <= 1e-3*max(1.0,abs(v))
   results['runs'][label+'_cache_repeat']=repeat
   print(label,'cache-repeat PASS',flush=True)
   if parallel:
    with ThreadPoolExecutor(max_workers=2) as pool:
     pair=list(pool.map(lambda x:call(port,'/v1/systemone',x['body']),cases[:2]))
    for index,response in enumerate(pair):
     for qid,answer in response['answers'].items():
      original=out[index]['response']['answers'][qid]
      assert answer.get('choice')==original.get('choice')
      for option,value in answer['probabilities'].items():
       assert abs(value-original['probabilities'][option])<=1e-3*max(1.0,abs(value)),(label,'concurrent lane mismatch')
    results['runs'][label+'_concurrent']=pair
    print(label,'concurrent-2 PASS',flush=True)
   status=call(port,'/status'); results['runs'][label+'_status']=status
   assert status['decision_model']['type']=='kev'
   assert status.get('startup_features',{}).get('drafter') in ('none',None)
  save()
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGINT)
   try: p.wait(timeout=45)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
    results['cleanups'][label]={'exit':p.returncode,'forced':True};save();raise
  results['cleanups'][label]={'exit':p.returncode,'forced':False};save();log.close()
  print(label,'exit',p.returncode,flush=True)

from install import gguf
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from server.kev import KevScorer
from server.judgments import validate_systemone
metadata=gguf.Metadata(model)
tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(gguf.tokenizer_files(metadata)['tokenizer/tokenizer.json'].decode()))
scorer=KevScorer(tokenizer,gguf.decision_config(metadata))
boundary=case('exact_context_8192','',{'team':choice('Which department handles the duplicate charge?',{'billing':'Payments and refunds','shipping':'Delivery problems'})})
repeat_count=8100
for _ in range(5):
 boundary['body']['state']='The customer was charged twice. ' + 'neutral '*repeat_count
 state,specs,errors=validate_systemone(boundary['body'])
 tokens,_=scorer.encode(state,specs[0][1],boundary['body']['questions']['team'])
 if len(tokens)==8192: break
 repeat_count+=8192-len(tokens)
assert len(tokens)==8192
cases.append(boundary)
for item in cases:
 state,specs,errors=validate_systemone(item['body']); assert not errors
 item['positions']={}
 for qid,spec in specs:
  tokens,positions=scorer.encode(state,spec,item['body']['questions'][qid])
  assert len(tokens)<=8192,(item['name'],len(tokens))
  item['positions'][qid]={'input_tokens':len(tokens),'option_positions':positions,'query_position':len(tokens)-1}
  if item['name']=='cross_chunk': assert len(set(p//2048 for p in positions))>1
save()
try:
 prior=json.loads((root/'reference-bf16-first-eight.json').read_text())
 assert all(a['body']==b['body'] for a,b in zip(cases[:8],prior['cases'][:8]))
 full_cases=cases
 cases=[full_cases[-1]]
 # llama.cpp reserves one context position even for a decision request.
 # Actual input stays exactly 8192 tokens; Splash remains capped at 8192.
 run('reference',['/Users/ghost233/.lmstudio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.56.0/llama-server','-m',model,'--host','127.0.0.1','--port','18113','-c','8193','-np','1','-ngl','99','-b','8192','-ub','8192','-ctk','bf16','-ctv','bf16','--no-warmup','--embd-normalize','-1'],18113)
 results['runs']['reference']=prior['runs']['reference']+results['runs']['reference']
 results['reference_context_tokens']={'first_eight':8192,'exact_boundary':8193}
 cases=full_cases
 save()
 run('splash_bf16',['sh','kev4bsplash.sh','--port','18112','--kv-format','bf16','--queue-size','2'],18112,True)
 run('splash_q4h',['sh','kev4bsplash.sh','--port','18112','--kv-format','q4h','--queue-size','2'],18112,True)
 ref={v['name']:v['response'] for v in results['runs']['reference']}
 comparisons=[]
 for mode in ('splash_bf16','splash_q4h'):
  for item in results['runs'][mode]:
   baseline=ref[item['name']]; maxima=[]; choices=[]
   for q,answer in item['response']['answers'].items():
    other=baseline['answers'][q]
    if 'probabilities' in answer:
     maxima += [abs(v-other['probabilities'][k]) for k,v in answer['probabilities'].items()]
     if 'choice' in answer: choices.append(answer['choice']==other['choice'])
     else: choices.append(max(answer['probabilities'],key=answer['probabilities'].get)==max(other['probabilities'],key=other['probabilities'].get))
    else:
     maxima.append(abs(answer['noul']-other['noul'])); choices.append((answer['noul']>=0.5)==(other['noul']>=0.5))
   comparisons.append({'mode':mode,'case':item['name'],'max_probability_difference':max(maxima),'same_choices':all(choices)})
 assert all(row['same_choices'] for row in comparisons),comparisons
 results['comparisons']=comparisons
 results['complete']=True;save()
 print('ALL CORRECTNESS RUNS COMPLETE',flush=True)
except Exception as e:
 results['complete']=False;results['error']=repr(e);save();print('FAILED',repr(e),flush=True);raise
