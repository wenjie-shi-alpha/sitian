#!/usr/bin/env python3
"""Paired continuation evaluation; baseline is the last scaling checkpoint, not step 0."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from evaluate_rl_skill_pilot import load_arm, paired_ci, arm_summary
from sitian.case import CaseBundle
from sitian.hard_metrics import forecast_hard_counts, failed_forecast_hard_counts

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    selection = json.loads((root / 'selection.json').read_text())
    source = Path(protocol['reward_and_data_guard']).parent
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_rl_skill_pilot.py'),
                    '--run-dir', str(source), '--check-only'], check=True)
    cases = json.loads((source / 'protocol.json').read_text())['cases']
    baseline = selection['baseline_step']
    validation = root / 'long_train/validation'
    before = load_arm(validation / f'{baseline}.jsonl', cases, baseline)
    for step in (25, selection['long_training_target']):
        after = load_arm(validation / f'{step}.jsonl', cases, step)
        if any(before[cid]['evaluation_seed'] != after[cid]['evaluation_seed'] for cid in before):
            raise ValueError('Paired evaluation seed mismatch')
        arms = {f'step_{baseline}': before, f'step_{step}': after}
        counts = {name: {} for name in arms}
        for case in cases:
            bundle = CaseBundle.load(Path(case['case_dir']))
            truth = bundle.truth_daily_full()
            for name, records in arms.items():
                row = records[case['case_id']]
                kwargs = {'aqi_standard': bundle.meta.get('aqi_standard')}
                counts[name][case['case_id']] = (forecast_hard_counts(row['forecast'], truth, **kwargs)
                    if row['submitted'] else failed_forecast_hard_counts(truth, **kwargs))
        ids = [case['case_id'] for case in cases]
        report = {'baseline_step': baseline, 'endpoint': step,
                  'scope': 'Continuation from trained baseline; does not estimate improvement over pretrained step zero.',
                  'paired_outcome': paired_ci(cases, before, after),
                  'arms': {name: arm_summary(rows, ids, counts[name]) for name, rows in arms.items()}}
        (root / f'comparison_step_{step}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
