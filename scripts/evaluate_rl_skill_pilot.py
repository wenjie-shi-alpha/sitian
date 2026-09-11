#!/usr/bin/env python3
"""Fail-closed paired analysis of native veRL validation records."""
import argparse
from collections import Counter, defaultdict
from datetime import date
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sitian.case import CaseBundle
from sitian.agents.scripted import run_baselines
from sitian.hard_metrics import forecast_hard_counts, failed_forecast_hard_counts, sum_counts, summarize_hard_counts, metric_improvements
from sitian.provenance import file_identity, identity_matches_file, case_bundle_snapshot
from sitian.scoring import reward_spec


def load_arm(path, cases, expected_step):
    records = {}
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        record = json.loads(raw['sitian_record'])
        cid = record['case_id']
        if cid in records:
            raise ValueError(f'duplicate case {cid}')
        if raw['step'] != expected_step:
            raise ValueError('wrong checkpoint step')
        submitted = raw['sitian_submitted']
        if not isinstance(submitted, bool) or submitted != (record['termination_reason'] == 'submitted'):
            raise ValueError('inconsistent submission status')
        if submitted != isinstance(record.get('forecast'), dict):
            raise ValueError('missing or unexpected normalized forecast')
        outcome = float(raw['outcome_composite'])
        if not math.isfinite(outcome) or (not submitted and outcome != 0):
            raise ValueError('invalid outcome')
        if record.get('evaluation_seed') is None:
            raise ValueError('unseeded evaluation')
        records[cid] = record | {'submitted': submitted, 'outcome': outcome}
    if set(records) != {c['case_id'] for c in cases}:
        raise ValueError('evaluation population mismatch: missing/unplanned cases')
    return records


def paired_ci(cases, frozen, trained, seed=20260909, repetitions=5000):
    groups = defaultdict(list)
    for c in cases:
        block = date.fromisoformat(c['issue_date']).toordinal() // 7
        groups[block].append(trained[c['case_id']]['outcome'] - frozen[c['case_id']]['outcome'])
    keys = sorted(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sample = [v for k in rng.choices(keys, k=len(keys)) for v in groups[k]]
        values.append(sum(sample) / len(sample))
    values.sort()
    return {'mean_delta': sum(sum(v) for v in groups.values()) / len(cases),
            'ci95': [values[int(.025 * repetitions)], values[int(.975 * repetitions)]],
            'issue_week_blocks': len(keys), 'exploratory': True,
            'warning': 'Few dependent process blocks and one stochastic rollout/case; not a confirmatory national claim.'}


def arm_summary(records, ids, counts):
    return {'cases': len(ids), 'submitted': sum(records[c]['submitted'] for c in ids),
            'submission_rate': sum(records[c]['submitted'] for c in ids) / len(ids),
            'mean_outcome': sum(records[c]['outcome'] for c in ids) / len(ids),
            'termination_reasons': dict(Counter(records[c]['termination_reason'] for c in ids)),
            'hard_metrics_failure_penalized': summarize_hard_counts(sum_counts([counts[c] for c in ids]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--step', type=int, choices=[0, 25, 50], default=0)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    run = Path(args.run_dir).resolve()
    protocol = json.loads((run / 'protocol.json').read_text())
    for ident in protocol['sources'] + protocol['code_and_gate'] + [protocol['panel_identity']]:
        if not identity_matches_file(ident, ROOT / ident['path'], relative_to=ROOT):
            raise ValueError(f"frozen input changed: {ident['path']}")
    if protocol['reward'] != reward_spec():
        raise ValueError('reward changed')
    cases = protocol['cases']
    if case_bundle_snapshot([c['case_dir'] for c in cases], relative_to=ROOT) != protocol['case_snapshot']:
        raise ValueError('case contents changed')
    if args.check_only:
        print('Frozen protocol, input files, evaluation cases and reward verified.')
        return
    frozen = load_arm(run / 'validation/0.jsonl', cases, 0)
    trained = load_arm(run / f'validation/{args.step}.jsonl', cases, args.step)
    for cid in frozen:
        if frozen[cid]['evaluation_seed'] != trained[cid]['evaluation_seed']:
            raise ValueError('paired seed mismatch')
    arms = {'frozen': frozen, f'step_{args.step}': trained}
    for name in protocol['business_references']:
        arms[name] = {}
    counts = {name: {} for name in arms}
    for c in cases:
        cid = c['case_id']
        bundle = CaseBundle.load(Path(c['case_dir']))
        truth = bundle.truth_daily_full()
        if not truth:
            raise ValueError('missing truth')
        baselines = run_baselines(bundle, protocol['business_references'])
        if set(baselines) != set(protocol['business_references']):
            raise ValueError('missing baseline; cannot silently drop case')
        for name, b in baselines.items():
            if b['forecast'] is None or b['composite'] is None:
                raise ValueError('baseline did not submit')
            arms[name][cid] = {'forecast': b['forecast'], 'outcome': b['composite'], 'submitted': True, 'termination_reason': 'submitted'}
        for name, records in arms.items():
            record = records[cid]
            kwargs = {'aqi_standard': bundle.meta.get('aqi_standard')}
            counts[name][cid] = (forecast_hard_counts(record['forecast'], truth, **kwargs) if record['submitted'] else failed_forecast_hard_counts(truth, **kwargs))
    ids = [c['case_id'] for c in cases]
    report = {'endpoint': args.step, 'primary_endpoint': protocol['primary_endpoint'],
              'protocol': file_identity(run / 'protocol.json', relative_to=ROOT),
              'validation_files': [file_identity(run / f'validation/{s}.jsonl', relative_to=ROOT) for s in sorted({0,args.step})],
              'scope': protocol['scope_limits'], 'failure_policy': protocol['failure_policy'],
              'paired_outcome': paired_ci(cases, frozen, trained),
              'arms': {name: arm_summary(records, ids, counts[name]) for name, records in arms.items()}}
    common = [cid for cid in ids if frozen[cid]['submitted'] and trained[cid]['submitted']]
    report['common_valid_diagnostic'] = {'cases': len(common), 'warning': 'Post-treatment subset; diagnostic, not causal effect.'}
    if common:
        before = summarize_hard_counts(sum_counts([counts['frozen'][cid] for cid in common]))
        after = summarize_hard_counts(sum_counts([counts[f'step_{args.step}'][cid] for cid in common]))
        report['common_valid_diagnostic'].update(frozen=before, trained=after, improvements=metric_improvements(after, before))
    report['subgroups'] = {}
    for field in ['split', 'stratum']:
        for value in sorted({c[field] for c in cases}):
            subset = [c for c in cases if c[field] == value]
            subids = [c['case_id'] for c in subset]
            report['subgroups'][f'{field}/{value}'] = {name: arm_summary(records, subids, counts[name]) for name, records in arms.items()}
    report['paired_cases'] = [{'case_id': cid, 'frozen_outcome': frozen[cid]['outcome'], 'trained_outcome': trained[cid]['outcome'], 'frozen_submitted': frozen[cid]['submitted'], 'trained_submitted': trained[cid]['submitted']} for cid in ids]
    target = run / f'comparison_step_{args.step}.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(target), 'paired_outcome': report['paired_outcome'], 'submission_rates': {k: v['submission_rate'] for k,v in report['arms'].items()}}, ensure_ascii=False))

if __name__ == '__main__':
    main()
