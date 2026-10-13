"""Tiel cold greedy validation follows the runtime's command-row contract."""
import hashlib
import json
from pathlib import Path

def first_difference(expected, actual):
    for index in range(max(len(expected),len(actual))):
        left=expected[index] if index<len(expected) else None
        right=actual[index] if index<len(actual) else None
        if left!=right:return {'index':index,'reference_token':left,'actual_token':right}
    return None

def validate_outputs(parameters,features,actual,reference,certificate):
    assert len(actual)==len(reference)==parameters['concurrency']
    differences=[first_difference(a,b) for a,b in zip(reference,actual)]
    exact=all(x is None for x in differences)
    if parameters['concurrency']==1:
        assert exact,('single-request full-token divergence',differences)
        return {'policy':'single_request_exact','full_token_comparison_required':True,'token_match_same_profile':True,'first_differences_same_profile':differences,'cold_graph_numeric_gate':'not_required_for_unchanged_single_request_comparison'}
    assert parameters['concurrency'] in (2,4),'uncertified concurrency'
    assert parameters['kv_format']=='q4h' and parameters['tail_prefill_splits']==8
    assert parameters['cold_requests'] and parameters['temperature']==0
    assert parameters['drafter_enabled'] and parameters['drafter_block_size']==8
    assert not parameters['ane_requested'] and not any(features.values())
    assert certificate['status']=='passed' and certificate['native_exit_code']==0
    assert certificate['scope']=='cold_greedy_matching_command_rows'
    assert certificate['score_relative_tolerance']==0.001 and certificate['state_cosine_minimum']==0.999
    assert certificate['observed_score_relative_max']<=0.001
    assert min(certificate['observed_state_cosines'])>0.999
    assert certificate['same_graph_four_lane_tokens_identical']
    assert certificate['controlled_fixed_prefill_prefixes_identical']
    return {'policy':'dynamic_graph_with_matching_rows_numeric_oracle','full_token_comparison_required':False,'token_match_same_profile':exact,'first_differences_same_profile':differences,'cold_graph_numeric_gate':'passed','numerical_certificate_sha256':certificate['certificate_sha256'],'comparison_note':'adaptive prefill partitions and batch rows differ; exact tokens are required by fixed-graph oracle, while cross-graph transcript differences remain diagnostic'}

def load_certificate(path,expected_runtime_sha,expected_weights):
    path=Path(path);data=path.read_bytes();certificate=json.loads(data)
    assert certificate['runtime_code_sha256']==expected_runtime_sha
    assert certificate['weight_sha256']==expected_weights
    for file,wanted in certificate['artifacts'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==wanted,('certificate artifact changed',file)
    certificate['certificate_sha256']=hashlib.sha256(data).hexdigest()
    return certificate
