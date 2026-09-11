import asyncio
import json
from types import SimpleNamespace
import pytest
from scripts.evaluate_rl_skill_pilot import load_arm, paired_ci


def record(cid='case1', submitted=True):
    return {'step': 0, 'outcome_composite': .5 if submitted else 0,
            'sitian_submitted': submitted,
            'sitian_record': json.dumps({'case_id': cid, 'evaluation_seed': 123,
                'forecast': {'daily': []} if submitted else None,
                'termination_reason': 'submitted' if submitted else 'no_submission'})}


def test_failed_cases_are_retained_and_missing_cases_rejected(tmp_path):
    path = tmp_path / '0.jsonl'
    cases = [{'case_id': 'case1'}, {'case_id': 'case2'}]
    path.write_text('\n'.join(json.dumps(r) for r in [record(), record('case2', False)]))
    loaded = load_arm(path, cases, 0)
    assert len(loaded) == 2 and loaded['case2']['outcome'] == 0
    path.write_text(json.dumps(record()))
    with pytest.raises(ValueError, match='population mismatch'):
        load_arm(path, cases, 0)


def test_duplicate_cases_rejected(tmp_path):
    path = tmp_path / '0.jsonl'
    path.write_text('\n'.join([json.dumps(record())] * 2))
    with pytest.raises(ValueError, match='duplicate'):
        load_arm(path, [{'case_id': 'case1'}], 0)


def test_clustered_pairing_uses_case_deltas():
    cases = [{'case_id': str(i), 'issue_date': '2026-06-01'} for i in range(3)]
    before = {str(i): {'outcome': .2 * i} for i in range(3)}
    after = {str(i): {'outcome': .2 * i + .1} for i in range(3)}
    result = paired_ci(cases, before, after, repetitions=100)
    assert result['mean_delta'] == pytest.approx(.1)
    assert result['ci95'] == pytest.approx([.1, .1])
    assert result['issue_week_blocks'] == 1 and result['exploratory']


def test_native_telemetry_preserves_training_sampling_and_masks(monkeypatch):
    pytest.importorskip('verl')
    from sitian.integrations.verl_runtime import SitianToolAgentLoop, ToolAgentLoop
    seen = []
    async def fake_run(self, sampling_params, **kwargs):
        seen.append(sampling_params.copy())
        return SimpleNamespace(extra_fields={}, response_ids=[1, 2], response_mask=[1, 0])
    monkeypatch.setattr(ToolAgentLoop, 'run', fake_run)
    loop = object.__new__(SitianToolAgentLoop)
    async def check():
        output = await loop.run({'temperature': 1}, extra_info={'case_id': 'case1'})
        assert seen[-1] == {'temperature': 1}
        assert output.response_ids == [1, 2] and output.response_mask == [1, 0]
        assert output.reward_score == 0 and output.extra_fields['reward_extra_info']['sitian_submitted'] is False
        info = {'case_id': 'case1', 'evaluation_panel': True, 'evaluation_seed': 20260909}
        await loop.run({'temperature': 1}, extra_info=info, session_id=0)
        await loop.run({'temperature': 1}, extra_info=info, session_id=0)
        assert seen[-1] == seen[-2] and 'seed' in seen[-1]
    asyncio.run(check())
