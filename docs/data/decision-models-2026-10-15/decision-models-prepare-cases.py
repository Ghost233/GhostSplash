import json,sys
from pathlib import Path
repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash');sys.path.insert(0,str(repo))
from install.gguf import Metadata,tokenizer_files,decision_config
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from server.kev import KevScorer
from server.modernbert import ModernBertScorer
from server.judgments import validate_systemone
models=[('Kev-0.8B','Kev-0.8B-GGUF','Kev-0.8B-Q8_0.gguf','kev08bsplash.sh'),('Julia-1','Julia-1-GGUF','Julia-1-Q8_0.gguf','julia1splash.sh'),('Laya','Laya-GGUF','Laya-Q8_0.gguf','layasplash.sh')]
root=Path('/private/tmp/decision-models-extended-20261015');root.mkdir(exist_ok=True)
base=json.loads(Path('/private/tmp/decision-models-modernbert-real-20261015/results.json').read_text())['runs']['Julia-1-reference']
for name,folder,file,launcher in models:
 path=Path('/Users/ghost233/.lmstudio/models/ggml-org')/folder/file
 metadata=Metadata(path)
 tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(tokenizer_files(metadata)['tokenizer/tokenizer.json'].decode()))
 scorer=(KevScorer if name=='Kev-0.8B' else ModernBertScorer)(tokenizer,decision_config(metadata))
 def prepare(body):
  state,specs,errors=validate_systemone(body); assert not errors,errors
  positions={}
  for qid,spec in specs:
   tokens,markers=scorer.encode(state,spec,body['questions'][qid]);positions[qid]={'tokens':len(tokens),'positions':list(markers),'flags':int(scorer.flags(spec))}
  return positions
 cases=[]
 for case in base:
  body=case['body'].copy();body['model']='ggml-org/'+folder+':Q8_0'
  cases.append({'name':case['name'],'body':body,'positions':prepare(body)})
 for length in [128,512,1049,2048,4096,8192]:
  body=cases[0]['body'].copy();prefix='Record A. A customer was charged twice for one order.';suffix=' The customer requests a refund for the duplicate charge.'
  repeat=max(0,length-80)
  for _ in range(6):
   body['state']=prefix+' neutral'*repeat+suffix
   positions=prepare(body);actual=next(iter(positions.values()))['tokens']
   if actual==length:break
   repeat+=length-actual
   assert repeat>=0
  assert actual==length,(name,length,actual)
  cases.append({'name':f'length_{length}','body':body,'positions':positions})
 multi={'model':'ggml-org/'+folder+':Q8_0','state':'The customer was charged twice and requests a refund.','questions':{
  'team':{'type':'choice','instructions':'Which department handles this?','criteria':{'shipping':'Delivery','billing':'Charges and refunds'}},
  'refund':{'type':'noul','instructions':'Does the customer ask for a refund?'},
  'urgency':{'type':'score','instructions':'How urgent is a response?','criteria':['No urgency','This week','Today']}}}
 cases.append({'name':'multi_question','body':multi,'positions':prepare(multi)})
 record={'name':name,'model':str(path),'model_id':'ggml-org/'+folder+':Q8_0','launcher':launcher,'cases':cases}
 (root/(name+'-cases.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
 print(name,[(c['name'],sum(v['tokens'] for v in c['positions'].values())) for c in cases],flush=True)
