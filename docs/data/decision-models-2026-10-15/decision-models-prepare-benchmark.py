import hashlib,json,sys
from pathlib import Path
repo=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash');sys.path.insert(0,str(repo))
from install.gguf import Metadata,tokenizer_files,decision_config
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from server.kev import KevScorer
from server.modernbert import ModernBertScorer
from server.judgments import validate_systemone
root=Path('/private/tmp/decision-models-extended-20261015')
for name in ['Kev-0.8B','Julia-1','Laya']:
 original=json.loads((root/(name+'-cases.json')).read_text())
 metadata=Metadata(original['model'])
 tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(tokenizer_files(metadata)['tokenizer/tokenizer.json'].decode()))
 scorer=(KevScorer if name=='Kev-0.8B' else ModernBertScorer)(tokenizer,decision_config(metadata))
 def body_for(length,salt):
  body=json.loads(json.dumps(original['cases'][0]['body']))
  prefix=f'Record {salt:04d}. A customer was charged twice for one order.'
  suffix=' The customer requests a refund for the duplicate charge.'
  count=length-80
  for _ in range(7):
   body['state']=prefix+' neutral'*count+suffix
   state,specs,errors=validate_systemone(body);assert not errors
   tokens,positions=scorer.encode(state,specs[0][1],body['questions']['team'])
   if len(tokens)==length: return body
   count+=length-len(tokens);assert count>=0
  raise AssertionError((name,length,salt,len(tokens)))
 result={k:original[k] for k in ['name','model','model_id','launcher']}
 result['bodies']={};result['warmup']={}
 for index,length in enumerate([128,512,2048,8192]):
  result['warmup'][str(length)]=body_for(length,9000+index)
  result['bodies'][str(length)]=[[body_for(length,index*100+round_index*5+sample+1) for sample in range(5)] for round_index in range(3)]
 result['weight_sha256']=hashlib.file_digest(Path(original['model']).open('rb'),'sha256').hexdigest()
 (root/(name+'-benchmark-cases.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(name,'prepared 60 distinct-prefix timed requests + 4 warmups; every token count checked',flush=True)
