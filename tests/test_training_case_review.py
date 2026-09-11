"""Regression cases for diagnostics and the observed stale submission instructions."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_submit_then_recovery_is_not_a_failed_episode(tmp_path):
    module = script('review_training_cases')
    row = {'input': '任务简报：{"case_id":"通辽_2025-09-06"}', 'step': 2, 'score': .4,
           'output': '<tool_call>{"name":"submit_forecast",broken}</tool_call>'
                     '<tool_response>{"accepted":false,"errors":["region missing"]}</tool_response>'
                     '<tool_call>{"name":"submit_forecast","arguments":{}}</tool_call>'
                     '<tool_response>{"accepted":true}</tool_response>'}
    result = module.inspect_record(row, tmp_path / '2.jsonl', 1)
    assert result['accepted_submission_observed'] is True
    assert len(result['malformed_closed_tool_calls']) == 1
    assert len(result['tool_errors']) == 1
    assert result['failure_tail'] is None


def test_case_review_rejects_non_training_cases(tmp_path):
    module = script('review_training_cases')
    source = tmp_path / '2.jsonl'
    source.write_text(json.dumps({'input': '{"case_id":"held-out"}', 'output': '', 'step': 2, 'score': 0}))
    with pytest.raises(ValueError, match='outside TRAIN'):
        module.review([source], {'training-only'}, tmp_path / 'review')


def test_regenerated_tool_schema_matches_current_generator():
    builder = script('build_verl_tool_config')
    config = yaml.safe_load((ROOT / 'configs/verl/sitian_tools_columns.yaml').read_text())
    function = next(tool['tool_schema']['function'] for tool in config['tools']
                    if tool['config']['tool_name'] == 'submit_forecast')
    expected = builder._shallow_parameters('submit_forecast', {
        'properties': {'forecast': {'type': 'object'}}, 'required': ['forecast']})
    assert function['parameters'] == expected
    description = function['parameters']['properties']['forecast']['description']
    assert 'issue_date、region' in description
    assert '不要把列再包进 daily' in description
    assert 'daily 必须含AQI等级' not in description
