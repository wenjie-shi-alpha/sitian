#!/usr/bin/env python3
"""Continue useful GRPO updates while scaling batches, then train to a cumulative target.

Run through run_local_training_chain.sh in the pinned veRL environment.
Every segment saves a resumable optimizer checkpoint; selection never rolls weights back.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf

from review_training_cases import review
from summarize_batch_benchmark import read_step_metrics
from summarize_scaling_benchmark import occupancy

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def checkpoint_identity(path, *, data_required=True):
    paths = [path / 'actor' / f'{kind}_world_size_4_rank_{rank}.pt'
             for rank in range(4) for kind in ('model', 'optim', 'extra_state')]
    if data_required:
        paths.append(path / 'data.pt')
    if not all(p.is_file() and p.stat().st_size > 0 for p in paths):
        raise ValueError(f'Incomplete four-rank checkpoint: {path}')
    return [identity(p) for p in paths]


def reset_data_checkpoint(source, destination):
    """Copy actor state unchanged, deliberately omit the old population's iterator."""
    checkpoint_identity(source)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / 'actor', destination / 'actor')
    before = {p['path'].split('/actor/')[-1]: p['sha256']
              for p in checkpoint_identity(source, data_required=False)}
    after = {p['path'].split('/actor/')[-1]: p['sha256']
             for p in checkpoint_identity(destination, data_required=False)}
    if before != after or (destination / 'data.pt').exists():
        raise ValueError('Data transition changed actor state')
    write_json(destination.parent / 'data_transition.json', {
        'source': str(source), 'destination': str(destination),
        'actor_shards_identical': True, 'data_iterator_reset': True,
        'reason': 'smoke_train (64 cases) -> full train (3061 cases); incompatible iterator populations',
        'retained': ['LoRA', 'optimizer', 'scheduler', 'per-rank RNG', 'global step'],
    })


def read_steps(log):
    return read_step_metrics(log)


def measure(arm, resources, phase, batch, first, last, *, allow_extra_steps=False):
    steps = read_steps(arm / 'train.log')
    if allow_extra_steps:
        steps = {step: metrics for step, metrics in steps.items() if first <= step <= last}
    if sorted(steps) != list(range(first, last + 1)):
        raise ValueError(f'Unexpected optimizer step sequence: {sorted(steps)}')
    for metric in steps.values():
        for key in ('actor/pg_loss', 'actor/grad_norm', 'timing_s/step', 'global_seqlen/mean'):
            if not math.isfinite(metric[key]):
                raise ValueError(f'Nonfinite {key}')
        if metric['actor/grad_norm'] <= 0:
            raise ValueError('Zero gradient update')
    samples = [json.loads(line) for line in resources.read_text().splitlines()]
    gpu = [row for row in samples if row.get('phase') == phase and 'gpus' in row]
    seconds = sum(m['timing_s/step'] for m in steps.values())
    tokens = sum(m['global_seqlen/mean'] * 4 for m in steps.values())
    rollouts = [json.loads(line) for step in steps
                for line in (arm / 'rollouts' / f'{step}.jsonl').read_text().splitlines()]
    if len(rollouts) != len(steps) * batch * 8:
        raise ValueError('Wrong number of retained training trajectories')
    return {'batch': batch, 'max_num_seqs': 16, 'first_step': first, 'last_step': last,
            'updates': len(steps), 'retained_case_occurrences': len(steps) * batch,
            'trajectories': len(rollouts), 'step_seconds_total': seconds,
            'sequence_tokens_per_second': tokens / seconds,
            'case_occurrences_per_minute': len(steps) * batch * 60 / seconds,
            'peak_gpu_mib': max(g['memory_mib'] for row in gpu for g in row['gpus']),
            'min_host_available_mib': min(row['host_mem_available_mib'] for row in gpu),
            'occupancy': occupancy(samples, phase), 'metrics_by_step': steps}


def audit_update(before, after, before_step, after_step, output):
    """Inspect every rank's local DTensor shard without initializing a GPU process group."""
    import torch
    torch.set_num_threads(2)
    ranks = []
    for rank in range(4):
        def read(folder, kind):
            return torch.load(folder / 'actor' / f'{kind}_world_size_4_rank_{rank}.pt',
                              map_location='cpu', weights_only=False)
        left, right = read(before, 'model'), read(after, 'model')
        if not left or left.keys() != right.keys():
            raise ValueError('Changed adapter structure')
        changed = 0
        for key in left:
            a, b = left[key], right[key]
            a = a.to_local() if hasattr(a, 'to_local') else a
            b = b.to_local() if hasattr(b, 'to_local') else b
            if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f'Invalid adapter tensor: rank={rank} key={key}')
            changed += int(not torch.equal(a, b))
        if not changed:
            raise ValueError(f'Rank {rank}: no adapter value changed')
        observed = []
        for folder, expected in ((before, before_step), (after, after_step)):
            optimizer = read(folder, 'optim')
            steps = sorted({float(item['step']) for item in optimizer['state'].values()})
            extra = read(folder, 'extra_state')
            if steps != [float(expected)] or extra['lr_scheduler']['last_epoch'] != expected or not extra['rng']:
                raise ValueError(f'Optimizer/scheduler/RNG continuity failed on rank {rank}')
            observed.append(steps)
            del optimizer, extra
        ranks.append({'rank': rank, 'adapter_tensors': len(left), 'changed_local_tensors': changed,
                      'optimizer_steps_before_after': observed})
        del left, right
    report = {'passed': True, 'before': str(before), 'after': str(after), 'ranks': ranks}
    write_json(output, report)
    return report


def make_config(base, parent, destination, arm, batch, target, train_file, panel, *, long_run=False):
    cfg = OmegaConf.load(base)
    cfg.data.train_files = str(train_file)
    cfg.data.val_files = str(panel)
    cfg.data.seed = 20260909
    cfg.data.train_batch_size = batch
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = batch
    cfg.actor_rollout_ref.rollout.max_num_seqs = 16
    cfg.actor_rollout_ref.rollout.gpu_memory_utilization = 0.65
    cfg.actor_rollout_ref.rollout.disable_log_stats = False
    # The original registry is frozen with earlier experiments. The regenerated
    # registry removes its stale daily/AQI instructions without changing reward.
    cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path = str(ROOT / 'configs/verl/sitian_tools_columns.yaml')
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = str(parent)
    cfg.trainer.default_local_dir = str(destination)
    cfg.trainer.experiment_name = arm.parent.name + '_' + arm.name
    cfg.trainer.total_training_steps = target
    cfg.trainer.save_freq = 1 if not long_run else 5
    cfg.trainer.max_actor_ckpt_to_keep = 4
    cfg.trainer.rollout_data_dir = str(arm / 'rollouts')
    cfg.trainer.val_before_train = long_run
    cfg.trainer.test_freq = 25 if long_run else -1
    cfg.trainer.validation_data_dir = str(arm / 'validation') if long_run else None
    cfg.data.val_batch_size = 8
    val = cfg.actor_rollout_ref.rollout.val_kwargs
    val.n, val.temperature, val.top_p, val.top_k, val.do_sample = 1, 1.0, 1.0, -1, True
    if cfg.actor_rollout_ref.actor.optim.lr_scheduler_type != 'constant':
        raise ValueError('Changing segment targets requires a constant LR schedule')
    required = {'model', 'optimizer', 'extra'}
    if not required <= set(cfg.actor_rollout_ref.actor.checkpoint.load_contents):
        raise ValueError('Checkpoint must restore model, optimizer, and extra state')
    return cfg


def observe_progress(root, arm, allowed, reviewed, target, prior_cases, batch):
    """Diagnostics must never terminate the optimizer process."""
    try:
        completed = set(read_steps(arm / 'train.log'))
        if completed - reviewed:
            paths = [arm / 'rollouts' / f'{step}.jsonl' for step in sorted(completed)]
            if all(path.is_file() for path in paths):
                review(paths, allowed, arm / 'case_review')
                write_json(root / 'progress.json', {
                    'phase': arm.name, 'effective_updates': max(completed), 'target': target,
                    'retained_case_occurrences_mainline': prior_cases + len(completed) * batch,
                    'updated_utc': datetime.now(timezone.utc).isoformat()})
                return completed
    except Exception as error:
        # A partial event/log/JSONL write can be retried on the next poll. Final
        # training and checkpoint audits remain mandatory and fail on bad data.
        print(f'Nonfatal progress observation error: {error!r}', file=sys.stderr, flush=True)
        try:
            write_json(root / 'monitor_warning.json', {
                'error': repr(error), 'utc': datetime.now(timezone.utc).isoformat()})
        except Exception:
            pass
    return reviewed


def resume_chain(args):
    """Preserve the failed run and its baseline; restore all saved training state."""
    source = args.resume_chain.resolve()
    root = ROOT / 'data/experiments' / args.run_id
    if root.resolve() == source:
        raise ValueError('Recovery requires a new run directory')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'launched.lock').mkdir()
    (root / 'phase').write_text('recovery_audit\n')
    previous = source / 'long_train'
    old_config = OmegaConf.load(previous / 'resolved_config.yaml')
    selection = json.loads((source / 'selection.json').read_text())
    protocol = json.loads((source / 'protocol.json').read_text())
    pilot = Path(protocol['reward_and_data_guard']).parent
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_rl_skill_pilot.py'),
                    '--run-dir', str(pilot), '--check-only'], check=True)
    for field in ('train', 'validation_panel', 'tool_config'):
        if identity(Path(protocol[field]['path'])) != protocol[field]:
            raise ValueError(f'Frozen {field} changed')
    checkpoints = list(Path(old_config.trainer.default_local_dir).glob('global_step_*'))
    parent = max(checkpoints, key=lambda path: int(path.name.removeprefix('global_step_')))
    step = int(parent.name.removeprefix('global_step_'))
    if step >= args.total_steps or args.total_steps != selection['long_training_target']:
        raise ValueError('Recovery must continue to the existing unfinished target')
    parent_files = checkpoint_identity(parent)
    old_lineage = json.loads((previous / 'lineage.json').read_text())
    audit_update(Path(old_lineage['parent']), parent, old_lineage['initial_step'], step,
                 root / 'recovery_checkpoint_audit.json')
    profiles = json.loads((source / 'profiles.json').read_text())
    batch = selection['selected_batch']
    retained = measure(previous, source / 'resources.jsonl', 'long_train', batch,
                       old_lineage['initial_step'] + 1, step, allow_extra_steps=True)
    profiles.append(retained)
    write_json(root / 'profiles.json', profiles)
    write_json(root / 'recovered_training_summary.json', retained)
    recovery = {'source_run': str(source), 'parent': str(parent), 'restored_step': step,
                'parent_files': parent_files,
                'completed_but_unsaved_steps_excluded': sorted(s for s in read_steps(previous / 'train.log') if s > step),
                'preserved_baseline_step': selection['baseline_step'],
                'changes': ['Strict numeric log parser and native TensorBoard metrics',
                            'Nonfatal progress diagnostics', 'Checkpoint every update; keep last four',
                            'Reuse completed frozen baseline validation'],
                'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(root / 'recovery.json', recovery)
    protocol = protocol | {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'recovery': recovery,
        'parent': str(parent), 'mainline_initial_effective_steps': step,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code': [identity(ROOT / 'scripts' / name) for name in (
            'run_local_training_chain.py', 'run_local_training_chain.sh', 'summarize_batch_benchmark.py',
            'review_training_cases.py', 'evaluate_training_chain.py', 'monitor_local_training.py')]}
    write_json(root / 'protocol.json', protocol)
    write_json(root / 'selection.json', selection | {'recovery_checkpoint': str(parent)})
    arm = root / 'long_train'
    arm.mkdir()
    validation = arm / 'validation'
    validation.mkdir()
    baseline = previous / 'validation' / f"{selection['baseline_step']}.jsonl"
    shutil.copy2(baseline, validation / baseline.name)
    # Preserve any completed scheduled evaluation across a later recovery too.
    for path in (previous / 'validation').glob('*.jsonl'):
        if int(path.stem) <= step and path.name != baseline.name:
            shutil.copy2(path, validation / path.name)
    destination = ROOT / 'data/checkpoints' / args.run_id / 'long_train'
    cfg = make_config(previous / 'resolved_config.yaml', parent, destination, arm, batch,
                      args.total_steps, Path(protocol['train']['path']),
                      Path(protocol['validation_panel']['path']), long_run=True)
    cfg.trainer.val_before_train = False
    cfg.trainer.save_freq = 1
    OmegaConf.save(cfg, arm / 'resolved_config.yaml')
    env = os.environ | {'SITIAN_TRAIN_BATCH_SIZE': str(batch), 'SITIAN_SMOKE_STEPS': str(args.total_steps),
                        'TENSORBOARD_DIR': str(arm / 'tensorboard'), 'PYTHONUNBUFFERED': '1'}
    with (arm / 'runtime_audit.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT / 'scripts/audit_verl_runtime_contract.py'),
                        '--resolved-config', str(arm / 'resolved_config.yaml'),
                        '--tool-config', str(cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path),
                        '--out', str(arm / 'runtime_audit.json')],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    lineage = {'phase': 'long_train', 'parent': str(parent), 'initial_step': step,
               'target_step': args.total_steps, 'batch': batch, 'data_iterator_restored': True,
               'parent_files': parent_files, 'config': identity(arm / 'resolved_config.yaml'),
               'baseline_reused': identity(baseline), 'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(arm / 'lineage.json', lineage)
    import pandas as pd
    allowed = {item['case_id'] for item in pd.read_parquet(protocol['train']['path'])['extra_info']}
    review([previous / 'rollouts' / f'{s}.jsonl' for s in range(old_lineage['initial_step'] + 1, step + 1)],
           allowed, root / 'recovered_case_review')
    prior_cases = sum(profile['retained_case_occurrences'] for profile in profiles)
    write_json(root / 'progress.json', {'phase': 'long_train', 'effective_updates': step,
               'target': args.total_steps, 'retained_case_occurrences_mainline': prior_cases,
               'updated_utc': datetime.now(timezone.utc).isoformat()})
    (root / 'phase').write_text('long_train\n')
    monitor = subprocess.Popen([sys.executable, str(ROOT / 'scripts/monitor_local_training.py'),
                                '--run-dir', str(root), '--rollout-metrics'])
    child = None
    try:
        command = ['timeout', '--signal=TERM', '--kill-after=60s', '129600',
                   'uv', 'run', '--no-sync', '--frozen', '--all-packages', '--extra', 'vllm', '--extra', 'fsdp',
                   'python', '-m', 'verl.trainer.main_ppo', '--config-path', str(arm), '--config-name', 'resolved_config']
        with (arm / 'train.log').open('w') as log:
            child = subprocess.Popen(command, cwd=Path(os.environ['SITIAN_VERL_ROOT']), env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            reviewed = set()
            while child.poll() is None:
                reviewed = observe_progress(root, arm, allowed, reviewed, args.total_steps, prior_cases, batch)
                time.sleep(10)
            status = child.wait()
        (arm / 'exit_code').write_text(str(status) + '\n')
        if status:
            raise RuntimeError(f'Resumed training exited with {status}')
        final = measure(arm, root / 'resources.jsonl', 'long_train', batch, step + 1, args.total_steps)
        after = destination / f'global_step_{args.total_steps}'
        write_json(arm / 'checkpoint_identity.json', checkpoint_identity(after))
        audit_update(parent, after, step, args.total_steps, arm / 'checkpoint_update_audit.json')
        review(sorted((arm / 'rollouts').glob('*.jsonl')), allowed, arm / 'case_review')
        observe_progress(root, arm, allowed, set(), args.total_steps, prior_cases, batch)
        write_json(root / 'long_training_summary.json', final)
        lineage.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                       completed_effective_steps=args.total_steps, checkpoint=str(after),
                       new_effective_updates=args.total_steps - step)
        write_json(arm / 'lineage.json', lineage)
        subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_training_chain.py'),
                        '--run-dir', str(root)], check=True)
        (root / 'phase').write_text('complete\n')
        (root / 'exit_code').write_text('0\n')
    except BaseException as error:
        write_json(root / 'failure.json', {'error': repr(error), 'recovery_parent': str(parent),
                   'checkpoint_directory': str(destination), 'utc': datetime.now(timezone.utc).isoformat()})
        (root / 'phase').write_text('failed\n')
        (root / 'exit_code').write_text('1\n')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
        monitor.terminate()
        monitor.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--source-unit')
    parser.add_argument('--resume-chain', type=Path,
                        help='Continue a stopped chain in a new run directory from its latest saved checkpoint')
    parser.add_argument('--total-steps', type=int, default=50)
    args = parser.parse_args()
    if args.resume_chain:
        resume_chain(args)
        return
    if not args.source_run or not args.source_unit:
        parser.error('--source-run and --source-unit are required for a new scaling chain')
    root = ROOT / 'data/experiments' / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / 'launched.lock').mkdir()  # No accidental restart of an existing run.
    source = args.source_run.resolve()
    pilot = ROOT / 'data/experiments/qwen3_8b_4gpu_20260911_r2'
    train = pilot / 'dataset/train.parquet'
    panel = pilot / 'panel.parquet'
    source_parent = ROOT / 'data/checkpoints' / f'{source.name}_b16_s16/global_step_1'
    verl = Path(os.environ['SITIAN_VERL_ROOT'])
    checkpoint_root = ROOT / 'data/checkpoints' / args.run_id
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'total_effective_updates_target': args.total_steps,
                'parent': str(source_parent), 'mainline_initial_effective_steps': 1,
                'independent_benchmark_branches_excluded': True,
                'train': identity(train), 'validation_panel': identity(panel),
                'tool_config': identity(ROOT / 'configs/verl/sitian_tools_columns.yaml'),
                'case_driven_change': 'Regenerated shallow submit schema: top-level interval columns, issue_date/region; remove stale daily/AQI requirement.',
                'reward_and_data_guard': str(pilot / 'protocol.json'),
                'test_batches': [32, 64], 'max_num_seqs_per_gpu': 16, 'rollout_gpu_memory_utilization': .65,
                'scale_gate': 'GPU peak <85GiB; host available >32GiB; batch32 token and case throughput >=95% of batch16. Preemptions are reported as efficiency diagnostics.',
                'selection': 'Among safe profiles within 5% of best token/s, choose maximum cases/min; within 5% of that choose smallest batch.',
                'limitations': ['Sequential on-policy learning; different cases and weights, not a controlled A/B quality comparison.',
                                'Retained case occurrences are not unique coverage; DAPO may sample and discard additional groups.',
                                'Validation baseline is the checkpoint before long training, not pretrained step zero.',
                                'Batch16 uses the old tool description and 55% rollout memory; batch32 and later share the corrected description and 65% rollout memory.'],
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'code': [identity(ROOT / 'scripts' / name) for name in (
                    'run_local_training_chain.py', 'run_local_training_chain.sh', 'review_training_cases.py',
                    'evaluate_training_chain.py', 'monitor_local_training.py')]}
    write_json(root / 'protocol.json', protocol)
    (root / 'phase').write_text('waiting_for_parent\n')
    deadline = time.monotonic() + 7200
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.source_unit]).returncode == 0:
        if time.monotonic() > deadline:
            raise TimeoutError('Parent benchmark did not stop')
        time.sleep(10)
    if (source / 'exit_code').read_text().strip() != '0' or (source / 'phase').read_text().strip() != 'complete':
        raise ValueError('Parent benchmark failed; no checkpoint adopted')
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_rl_skill_pilot.py'),
                    '--run-dir', str(pilot), '--check-only'], check=True)
    write_json(root / 'parent_checkpoint_identity.json', checkpoint_identity(source_parent))
    import pandas as pd
    allowed = {item['case_id'] for item in pd.read_parquet(train)['extra_info']}
    if len(allowed) != 3061:
        raise ValueError('Unexpected full training population')
    parent = checkpoint_root / 'full_data_entry/global_step_1'
    reset_data_checkpoint(source_parent, parent)
    profiles = [measure(source / 'b16_s16', source / 'resources.jsonl', 'b16_s16', 16, 1, 1)]
    review(sorted((source / 'b16_s16/rollouts').glob('*.jsonl')), allowed, root / 'parent_cases')
    write_json(root / 'profiles.json', profiles)
    step = 1
    monitor = subprocess.Popen([sys.executable, str(ROOT / 'scripts/monitor_local_training.py'),
                                '--run-dir', str(root), '--rollout-metrics'])
    def segment(batch, target, *, long_run=False):
        nonlocal parent, step
        name = 'long_train' if long_run else f'b{batch}_s16'
        arm = root / name
        arm.mkdir()
        destination = checkpoint_root / name
        cfg = make_config(source / 'b16_s16/resolved_config.yaml', parent, destination, arm,
                          batch, target, train, panel, long_run=long_run)
        OmegaConf.save(cfg, arm / 'resolved_config.yaml')
        env = os.environ | {'SITIAN_TRAIN_BATCH_SIZE': str(batch), 'SITIAN_SMOKE_STEPS': str(target),
                            'TENSORBOARD_DIR': str(arm / 'tensorboard')}
        with (arm / 'runtime_audit.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/audit_verl_runtime_contract.py'),
                            '--resolved-config', str(arm / 'resolved_config.yaml'),
                            '--tool-config', str(cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path),
                            '--out', str(arm / 'runtime_audit.json')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        lineage = {'phase': name, 'parent': str(parent), 'initial_step': step, 'target_step': target,
                   'batch': batch, 'config': identity(arm / 'resolved_config.yaml'),
                   'parent_files': checkpoint_identity(parent, data_required=step != 1),
                   'data_iterator_restored': step != 1, 'started_utc': datetime.now(timezone.utc).isoformat()}
        write_json(arm / 'lineage.json', lineage)
        (root / 'phase').write_text(name + '\n')
        timeout = '129600' if long_run else '7200'
        command = ['timeout', '--signal=TERM', '--kill-after=60s', timeout,
                   'uv', 'run', '--no-sync', '--frozen', '--all-packages', '--extra', 'vllm', '--extra', 'fsdp',
                   'python', '-m', 'verl.trainer.main_ppo', '--config-path', str(arm), '--config-name', 'resolved_config']
        with (arm / 'train.log').open('w') as log:
            child = subprocess.Popen(command, cwd=verl, env=env, stdout=log, stderr=subprocess.STDOUT)
            reviewed = set()
            while child.poll() is None:
                # Only review after native step metrics, which follow complete rollout dumps.
                reviewed = observe_progress(root, arm, allowed, reviewed, args.total_steps,
                                            sum(p['retained_case_occurrences'] for p in profiles), batch)
                time.sleep(10)
            status = child.wait()
        (arm / 'exit_code').write_text(str(status) + '\n')
        if status:
            raise RuntimeError(f'{name} failed with exit {status}; latest saved checkpoint remains usable')
        result = measure(arm, root / 'resources.jsonl', name, batch, step + 1, target)
        after = destination / f'global_step_{target}'
        write_json(arm / 'checkpoint_identity.json', checkpoint_identity(after))
        audit_update(parent, after, step, target, arm / 'checkpoint_update_audit.json')
        review(sorted((arm / 'rollouts').glob('*.jsonl')), allowed, arm / 'case_review')
        lineage.update(completed_utc=datetime.now(timezone.utc).isoformat(), completed_effective_steps=target,
                       checkpoint=str(after), new_effective_updates=target - step)
        write_json(arm / 'lineage.json', lineage)
        parent, step = after, target
        return result
    try:
        profiles.append(segment(32, step + 1))
        write_json(root / 'profiles.json', profiles)
        previous, current = profiles[-2:]
        safe = lambda p: (p['peak_gpu_mib'] < 85 * 1024 and p['min_host_available_mib'] > 32 * 1024
                          and p['occupancy'].get('available'))
        scale = (safe(current) and current['sequence_tokens_per_second'] >= .95 * previous['sequence_tokens_per_second']
                 and current['case_occurrences_per_minute'] >= .95 * previous['case_occurrences_per_minute'])
        write_json(root / 'scale_decision.json', {'test_batch64': scale, 'batch32': current, 'reference_batch16': previous})
        if scale:
            profiles.append(segment(64, step + 1))
            write_json(root / 'profiles.json', profiles)
        candidates = [p for p in profiles if safe(p)]
        if not candidates:
            raise ValueError('No safe profile for long training')
        best_tokens = max(p['sequence_tokens_per_second'] for p in candidates)
        candidates = [p for p in candidates if p['sequence_tokens_per_second'] >= .95 * best_tokens]
        best_cases = max(p['case_occurrences_per_minute'] for p in candidates)
        chosen = min((p for p in candidates if p['case_occurrences_per_minute'] >= .95 * best_cases), key=lambda p: p['batch'])
        write_json(root / 'selection.json', {'selected_batch': chosen['batch'], 'max_num_seqs': 16,
                   'resume_from_latest_checkpoint': str(parent), 'baseline_step': step,
                   'long_training_target': args.total_steps, 'selection_rule': protocol['selection'],
                   'profiles': profiles, 'limitations': protocol['limitations']})
        final = segment(chosen['batch'], args.total_steps, long_run=True)
        write_json(root / 'long_training_summary.json', final)
        subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_training_chain.py'), '--run-dir', str(root)], check=True)
        (root / 'phase').write_text('complete\n')
        (root / 'exit_code').write_text('0\n')
    except BaseException as error:
        write_json(root / 'failure.json', {'phase': (root / 'phase').read_text().strip(),
                   'last_verified_checkpoint': str(parent), 'last_verified_effective_step': step,
                   'error': repr(error), 'utc': datetime.now(timezone.utc).isoformat()})
        (root / 'exit_code').write_text('1\n')
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)


if __name__ == '__main__':
    main()
