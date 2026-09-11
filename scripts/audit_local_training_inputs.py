#!/usr/bin/env python3
"""Freeze unchanged audited datasets with freshly validated local runtime gates."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sitian.provenance import case_bundle_snapshot, file_identity, identity_matches_file, score_implementation_matches
from sitian.scoring import reward_spec

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--runtime-audit', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / 'data/verl/manifest.json'
    data = json.loads(source.read_text())
    require(data['reward'] == reward_spec(), 'dataset reward identity differs')
    require(data['expert_corpus_used_as_training_input'] is False, 'expert training inputs forbidden')
    require(data['truth_exposed_in_prompt_or_ground_truth_field'] is False, 'truth exposed')
    checks = {}
    for split, identity in data['source_manifests'].items():
        path = ROOT / identity['path']
        require(identity_matches_file(identity, path), f'{split}: source manifest changed')
        cases = [ROOT / name for name in json.loads(path.read_text())]
        actual = case_bundle_snapshot(cases, relative_to=ROOT)
        expected = data['input_snapshots'][split]
        require(all(actual[k] == expected[k] for k in ('sha256', 'cases', 'files', 'bytes')),
                f'{split}: input or hidden truth snapshot changed')
        checks[split] = {k: actual[k] for k in ('sha256', 'cases', 'files', 'bytes')}
    replacements = {
        'verl_runtime_contract': args.runtime_audit.resolve(),
        'verl_agent_loop_config': ROOT / 'configs/verl/sitian_agent_loop.yaml',
    }
    for name, identity in data['gate_artifacts'].items():
        if name not in replacements:
            require(identity_matches_file(identity, ROOT / identity['path']), f'gate artifact changed: {name}')
    runtime = json.loads(args.runtime_audit.read_text())
    require(runtime['passed'] and all(runtime['checks'].values()), 'local runtime audit failed')
    require(runtime['reward'] == reward_spec(), 'local runtime reward differs')
    for identity in runtime['inputs'].values():
        require(identity_matches_file(identity, ROOT / identity['path']), f'runtime input changed: {identity["path"]}')
    reward = json.loads((ROOT / data['gate_artifacts']['reward_contract_audit']['path']).read_text())
    require(reward['passed'] and score_implementation_matches(reward, ROOT), 'reward audit stale or failed')
    out = args.out_dir.resolve()
    require(not (out / 'manifest.json').exists(), 'local manifest already frozen')
    out.mkdir(parents=True, exist_ok=True)
    for name, identity in data['outputs'].items():
        path = source.parent / name
        require(identity_matches_file(dict(identity, path=str(path)), path), f'Parquet changed: {name}')
        shutil.copy2(path, out / name)
    data['local_adaptation'] = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'parent_manifest': file_identity(source, relative_to=ROOT),
        'verified_case_snapshots': checks,
        'previous_runtime_gates': {name: data['gate_artifacts'][name] for name in replacements},
        'reason': 'Byte-identical Parquet and case snapshots; re-audited native runtime after host migration.',
    }
    for name, path in replacements.items():
        data['gate_artifacts'][name] = file_identity(path, relative_to=ROOT)
    (out / 'manifest.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'passed': True, 'verified_splits': checks, 'out': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
