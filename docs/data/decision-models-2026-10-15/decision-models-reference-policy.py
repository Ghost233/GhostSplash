import json,os,signal,subprocess,time,urllib.request,urllib.error
from pathlib import Path
binary='/Users/ghost233/.lmstudio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.56.0/llama-server'
model='/Users/ghost233/.lmstudio/models/ggml-org/Julia-1-GGUF/Julia-1-Q8_0.gguf'
log=Path('/private/tmp/decision-models-reference-policy.log').open('w')
command=[binary,'-m',model,'--host','127.0.0.1','--port','18148','-c','8193','-np','1','-ngl','99','-b','8192','-ub','8192','--no-warmup','--embd-normalize','-1','--log-verbosity','3']
p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
 end=time.monotonic()+90
 while True:
  assert p.poll() is None
  try:
   with urllib.request.urlopen('http://127.0.0.1:18148/health',timeout=2) as response:json.load(response)
   break
  except (OSError,urllib.error.URLError):assert time.monotonic()<end;time.sleep(.5)
finally:
 if p.poll() is None:os.killpg(p.pid,signal.SIGINT);p.wait(timeout=30)
 log.close()
assert p.returncode==0
print('reference policy inspection exit',p.returncode)
