#!/usr/bin/env python3
"""Freeze a small development panel before any RL skill measurements."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sitian.provenance import file_identity, case_bundle_snapshot
from sitian.scoring import reward_spec


def select(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row['extra_info']['stratum']].append(row)
    rng = random.Random(seed)
    for key in sorted(groups):
        groups[key].sort(key=lambda r: r['extra_info']['case_id'])
        rng.shuffle(groups[key])
    result = []
    while len(result) < count:
        for key in sorted(groups):
            if groups[key] and len(result) < count:
                result.append(groups[key].pop())
        if not any(groups.values()) and len(result) < count:
            raise ValueError('not enough cases')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise SystemExit('protocol already frozen; refusing overwrite')
    seed = 20260909
    panel = []
    sources = []
    for cohort, count in [('val', 48), ('challenge_winter', 16)]:
        path = ROOT / 'data/verl' / f'{cohort}.parquet'
        sources.append(file_identity(path, relative_to=ROOT))
        for row in select(pq.read_table(path).to_pylist(), count, seed):
            row['extra_info'].update(evaluation_panel=True, evaluation_seed=seed)
            panel.append(row)
    train_path = ROOT / 'data/verl/train.parquet'
    train = pq.read_table(train_path).to_pylist()
    ids = [r['extra_info']['case_id'] for r in panel]
    assert len(ids) == len(set(ids)) == 64
    assert not set(ids) & {r['extra_info']['case_id'] for r in train}
    pq.write_table(pa.Table.from_pylist(panel), out / 'panel.parquet', compression='zstd')
    cases = []
    for row in panel:
        info = row['extra_info']
        path = next(iter(info['tools_kwargs'].values()))['create_kwargs']['case_dir']
        cases.append({k: info[k] for k in ['case_id', 'issue_date', 'stratum', 'split']} | {'case_dir': path})
    tracked = ['src/sitian/integrations/verl_runtime.py', 'src/sitian/integrations/verl_bridge.py',
               'src/sitian/scoring.py', 'src/sitian/schema.py', 'src/sitian/env.py',
               'src/sitian/hard_metrics.py', 'scripts/run_verl_smoke.sh',
               'scripts/run_rl_skill_pilot.sh', 'scripts/evaluate_rl_skill_pilot.py',
               'scripts/prepare_rl_skill_pilot.py', 'data/verl/manifest.json',
               'configs/verl/sitian_tools.yaml', 'configs/verl/sitian_agent_loop.yaml']
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'exploratory development RL skill pilot; not national generalization confirmation',
        'steps': 50, 'evaluations': [0, 25, 50], 'primary_endpoint': 50,
        'model': 'Qwen/Qwen3-8B original BF16, fresh LoRA; no smoke adapter',
        'train_cases': len(train), 'seed': seed, 'evaluation_n': 1,
        'evaluation_temperature': 1.0, 'thinking': True,
        'primary_comparison': 'same native tool agent at step 50 minus frozen step 0',
        'primary_metric': 'mean outcome_composite over every planned case, failures zero',
        'secondary_metrics': ['submission rate', 'ordinal MAE', 'event CSI/POD/FAR',
                              'interval score and coverage', 'PM2.5/PM10/O3 midpoint MAE'],
        'business_references': ['guidance', 'persistence', 'climatology'],
        'failure_policy': 'all cases retained; hard metrics use explicit failed_forecast_hard_counts penalties; common-valid subset diagnostic only',
        'uncertainty': 'paired 7-day issue-date block bootstrap; few clusters, exploratory only',
        'decision': 'No success claim from training reward alone. Separate submission gains from common-valid forecast changes. Step 25 diagnostic; no endpoint cherry-picking.',
        'scope_limits': 'Stratum-balanced 48 warm-val + 16 winter cases, not prevalence representative. One rollout/case. Test and spatial OOD test sealed. Agency-specific claim needs fixed-evidence/flow ablations.',
        'sources': sources + [file_identity(train_path, relative_to=ROOT)],
        'panel_identity': file_identity(out / 'panel.parquet', relative_to=ROOT),
        'code_and_gate': [file_identity(ROOT / f, relative_to=ROOT) for f in tracked],
        'case_snapshot': case_bundle_snapshot([r['case_dir'] for r in cases], relative_to=ROOT),
        'reward': reward_spec(), 'cases': cases,
        'strata': dict(Counter((r['split'] + '/' + r['stratum']) for r in cases)),
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'panel': len(panel), 'train': len(train), 'strata': protocol['strata']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
