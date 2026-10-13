import copy,json,sys,unittest
from pathlib import Path

root=Path('/Users/ghost233/.codex/worktrees/qwen4b-quantized-ffn/GhostSplash')
module=root/'build/diagnostics/tiel-graph-validation-v1';sys.path.insert(0,str(module))
from tiel_graph_validation import load_certificate,validate_outputs

plan=json.loads((root/'build/tiel-miniplus-matrix-v1/plan.json').read_text())
certificate=load_certificate(module/'certificate.json',plan['runtime_code_sha256'],{k:v['sha256'] for k,v in plan['weights'].items()})
summary=json.loads((root/'build/tiel-miniplus-q4h-output16k-sweep-v1/summary.json').read_text())
reference=next(r for r in summary['records'] if r['profile']=='concurrency-2')
old=json.loads((Path(reference['attempt'])/'stdout.json').read_text())
current=json.loads((root/'build/tiel-miniplus-q4h-output16k-sweep-v3/perf-0-017-isolated-retest-concurrency-2/stdout.json').read_text())
actual=[r['output_token_ids'] for r in current['requests']];expected=[r['output_token_ids'] for r in old['requests']]

class GraphValidationTests(unittest.TestCase):
    def test_original_dynamic_graph_failure_is_classified_without_faking_token_match(self):
        result=validate_outputs(current['parameters'],reference['features'],actual,expected,certificate)
        self.assertFalse(result['token_match_same_profile'])
        self.assertFalse(result['full_token_comparison_required'])
        self.assertEqual(result['first_differences_same_profile'][1]['index'],15)
        self.assertEqual(result['cold_graph_numeric_gate'],'passed')
    def test_single_request_mismatch_still_fails(self):
        p=current['parameters']|{'concurrency':1}
        changed=copy.deepcopy(actual[:1]);changed[0][15]^=1
        with self.assertRaises(AssertionError):validate_outputs(p,reference['features'],changed,actual[:1],certificate)
    def test_single_request_exact_still_passes(self):
        result=validate_outputs(current['parameters']|{'concurrency':1},reference['features'],actual[:1],actual[:1],certificate)
        self.assertTrue(result['full_token_comparison_required'])
    def test_failed_numeric_oracle_is_rejected(self):
        c=certificate|{'status':'failed'}
        with self.assertRaises(AssertionError):validate_outputs(current['parameters'],reference['features'],actual,expected,c)
    def test_original_score_bound_is_enforced(self):
        c=certificate|{'observed_score_relative_max':0.001001}
        with self.assertRaises(AssertionError):validate_outputs(current['parameters'],reference['features'],actual,expected,c)
    def test_original_state_bound_is_enforced(self):
        c=certificate|{'observed_state_cosines':[0.999,1.0]}
        with self.assertRaises(AssertionError):validate_outputs(current['parameters'],reference['features'],actual,expected,c)
    def test_changing_tolerances_is_rejected(self):
        for key,value in [('score_relative_tolerance',0.01),('state_cosine_minimum',0.99)]:
            with self.assertRaises(AssertionError):validate_outputs(current['parameters'],reference['features'],actual,expected,certificate|{key:value})
    def test_cached_or_sampled_workload_is_outside_certificate(self):
        for p in [current['parameters']|{'cold_requests':False},current['parameters']|{'temperature':0.8}]:
            with self.assertRaises(AssertionError):validate_outputs(p,reference['features'],actual,expected,certificate)
    def test_other_format_is_outside_certificate(self):
        with self.assertRaises(AssertionError):validate_outputs(current['parameters']|{'kv_format':'int8'},reference['features'],actual,expected,certificate)
    def test_source_or_weight_change_is_rejected(self):
        for code,weights in [('0'*64,{k:v['sha256'] for k,v in plan['weights'].items()}),(plan['runtime_code_sha256'],{'target':'0'*64})]:
            with self.assertRaises(AssertionError):load_certificate(module/'certificate.json',code,weights)

if __name__=='__main__':unittest.main()
