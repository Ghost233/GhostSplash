import json, os, signal, socket, subprocess, time, urllib.error, urllib.request
from pathlib import Path
repo = Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash')
port = 18110
with socket.socket() as check:
    check.bind(('127.0.0.1', port))
log = Path('/private/tmp/splash-optional-memory-real-startup.log')
result_path = Path('/private/tmp/splash-optional-memory-real-startup.json')
command = ['sh', str(repo/'qwen3827bmlx4bitsplash.sh'), '--binary', str(repo/'build/optional-memory-fix-20261015/splash'), '--host', '127.0.0.1', '--port', str(port)]
report = {'command': command, 'performance_tested': False, 'ready': False}
process = None
try:
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        while time.monotonic() - started < 300:
            if process.poll() is not None:
                raise RuntimeError(f'launcher exited before Ready: {process.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/status', timeout=3) as response:
                    status = json.load(response)
            except (OSError, urllib.error.URLError, ValueError):
                time.sleep(2)
                continue
            if status.get('ready') is True:
                break
            time.sleep(2)
        else:
            raise RuntimeError('startup did not reach Ready within 300 seconds')
        assert status['memory_audit']['valid'] is True, status['memory_audit']
        optional = status['startup_features']['optional']
        assert optional['compact_draft_head'] and optional['compact_validated'] and optional['fast_topk'] and optional['prompt_lookup']
        assert optional['compact_validation_maximum_error'] == 0
        assert optional['extra_metal_bytes'] == 283639808, optional
        assert abs(status['ane_ffn']['share'] - 14/34) < 1e-6, status['ane_ffn']
        report['ready'] = True
        report['startup_status'] = status
        print('Ready reached; strict memory audit and compact numerical validation passed', flush=True)
        payload = {'model':'mlx-community/Qwen3.8-27B-4bit','messages':[{'role':'user','content':'用一句话解释什么是哈希表。'}], 'max_tokens':32,'temperature':0.7,'top_k':20,'min_p':0,'stream':False}
        request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=180) as response:
            answer = json.load(response)
        assert answer.get('choices'), answer
        assert answer.get('usage',{}).get('completion_tokens',0)>0, answer
        report['sampled_request'] = {'payload':payload,'response':answer}
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/status', timeout=10) as response:
            after = json.load(response)
        assert after['memory_audit']['valid'] is True, after['memory_audit']
        report['after_request_status'] = after
        report['success'] = True
        print('Short sampled request passed; no performance comparison performed', flush=True)
except BaseException as error:
    report['success'] = False
    report['error'] = str(error)
    raise
finally:
    if process is not None:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    report['forced_cleanup'] = True
        report['launcher_exit_code'] = process.returncode
    result_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
