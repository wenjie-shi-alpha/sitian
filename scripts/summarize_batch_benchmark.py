#!/usr/bin/env python3
"""Compare completed equal-work batch-size arms using measured steps and rollouts."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ANSI = re.compile(r'\x1b\[[0-9;]*m')
NUMBER = r'[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)'
VALUE = re.compile(r'([\w/.-]+):(' + NUMBER + r')(?=\s|$)')
METRIC = re.compile(r'([\w/.-]+):(?:np\.(?:float\d+|int\d+)\((' + NUMBER
                    + r')\)|(' + NUMBER + r')(?=\s|$))')


def parse_step_metrics(text):
    """Read numeric metrics without interpreting interleaved Ray timestamps."""
    result = {}
    for line in ANSI.sub('', text).splitlines():
        match = re.search(r'\bstep:(\d+) - ', line)
        if match:
            metrics = {key: float(wrapped or plain)
                       for key, wrapped, plain in METRIC.findall(line[match.end():])}
            if 'timing_s/step' in metrics:
                result.setdefault(int(match[1]), {}).update(metrics)
    return result


def read_step_metrics(log):
    result = parse_step_metrics(log.read_text(errors='replace'))
    # Console output can be buffered or interleaved across Ray processes. Native
    # scalar events provide timely metrics; preserve console precision if present.
    directory = log.parent / 'tensorboard'
    if any(directory.glob('events.out.tfevents.*')):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()
        by_step = {}
        for key in events.Tags().get('scalars', []):
            for event in events.Scalars(key):
                by_step.setdefault(event.step, {})[key] = event.value
        for step, metrics in by_step.items():
            if 'timing_s/step' in metrics:
                result[step] = metrics | result.get(step, {})
    return result


def read_arm(root, batch, resources, *, arm_name=None, target_prompts=8):
    arm_name = arm_name or f'batch_{batch}'
    arm = root / arm_name
    if (arm / 'exit_code').read_text().strip() != '0':
        raise ValueError(f'batch {batch} did not complete')
    steps = read_step_metrics(arm / 'train.log')
    if sorted(steps) != list(range(1, target_prompts // batch + 1)):
        raise ValueError(f'batch {batch}: missing/unexpected step metrics')
    for metrics in steps.values():
        for key in ('actor/pg_loss', 'actor/grad_norm', 'timing_s/step', 'global_seqlen/mean'):
            if not math.isfinite(metrics[key]):
                raise ValueError(f'batch {batch}: nonfinite {key}')
    records = [json.loads(line) for path in sorted((arm / 'rollouts').glob('*.jsonl'))
               for line in path.read_text().splitlines()]
    prompts = Counter(hashlib.sha256(row['input'].encode()).hexdigest() for row in records)
    if len(records) != target_prompts * 8 or len(prompts) != target_prompts or set(prompts.values()) != {8}:
        raise ValueError(f'batch {batch}: expected {target_prompts} prompts with 8 trajectories each')
    seconds = sum(m['timing_s/step'] for m in steps.values())
    sequence_tokens = sum(m['global_seqlen/mean'] * 4 for m in steps.values())
    samples = [row for row in resources if row.get('phase') == arm_name and 'gpus' in row]
    peak_gpu = {str(index): max(gpu['memory_mib'] for row in samples for gpu in row['gpus']
                                if gpu['index'] == index) for index in range(4)}
    return {
        'batch': batch, 'steps': len(steps), 'retained_prompts': len(prompts), 'trajectories': len(records),
        'step_seconds_total': seconds,
        'generation_seconds_total': sum(m['timing_s/gen'] for m in steps.values()),
        'actor_update_seconds_total': sum(m['timing_s/update_actor'] for m in steps.values()),
        'sequence_tokens_total_including_tools': sequence_tokens,
        'sequence_tokens_per_second_all_gpus': sequence_tokens / seconds,
        'retained_prompts_per_minute': len(prompts) * 60 / seconds,
        'gpu_peak_used_mib_sampled': peak_gpu,
        'minimum_host_available_mib': min(row['host_mem_available_mib'] for row in samples),
        'actor_peak_allocated_gib': max(m['actor/perf/max_memory_allocated_gb'] for m in steps.values()),
        'prompt_hashes': sorted(prompts), 'metrics_by_step': steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    resources = [json.loads(line) for line in (root / 'resources.jsonl').read_text().splitlines()]
    arms = {str(batch): read_arm(root, batch, resources) for batch in (4, 8)}
    a, b = arms['4'], arms['8']
    comparison = {
        'same_retained_prompts': a['prompt_hashes'] == b['prompt_hashes'],
        'overlap_prompts': len(set(a['prompt_hashes']) & set(b['prompt_hashes'])),
        'equal_case_work_speedup_batch8_over_batch4': a['step_seconds_total'] / b['step_seconds_total'],
        'sequence_token_throughput_ratio_batch8_over_batch4': b['sequence_tokens_per_second_all_gpus'] / a['sequence_tokens_per_second_all_gpus'],
        'sequence_token_volume_ratio_batch8_over_batch4': b['sequence_tokens_total_including_tools'] / a['sequence_tokens_total_including_tools'],
    }
    result = {'arms': arms, 'comparison': comparison,
              'limitations': ['Single short run per arm; no statistical significance claim.',
                              'Initialization and pre-training validation excluded from timed steps.',
                              'Batch4 makes two optimizer updates; batch8 makes one. Training quality is not assessed.',
                              'Sequence tokens include prompts and tool text, not just generated assistant tokens.',
                              'GPU usage sampled every two seconds; very brief peaks may be missed.']}
    (root / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 四卡全局 batch 对比', '',
             '两组均保留 8 个案例、64 条轨迹；每卡采样并发上限均为 8，训练微批 token 上限均为 32K。', '',
             '| 指标 | batch=4 × 2 步 | batch=8 × 1 步 |', '|---|---:|---:|']
    for label, key in [('整步累计耗时（秒）', 'step_seconds_total'),
                       ('采样累计耗时（秒）', 'generation_seconds_total'),
                       ('梯度更新累计耗时（秒）', 'actor_update_seconds_total'),
                       ('四卡总序列 token/s（含工具文本）', 'sequence_tokens_per_second_all_gpus'),
                       ('案例/分钟', 'retained_prompts_per_minute'),
                       ('actor 分配显存峰值（GiB）', 'actor_peak_allocated_gib')]:
        lines.append(f'| {label} | {a[key]:.2f} | {b[key]:.2f} |')
    lines += ['', f"入选案例重合：{comparison['overlap_prompts']}/8。",
              f"等案例数量吞吐比（batch8 / batch4）：{comparison['equal_case_work_speedup_batch8_over_batch4']:.3f}。",
              f"序列 token 吞吐比：{comparison['sequence_token_throughput_ratio_batch8_over_batch4']:.3f}。", '',
              '每组仅一次短测，不代表稳定长跑结果或训练质量提升。初始化不计入步骤耗时；',
              '两组优化器更新次数不同，生成长度可能不同。资源采样间隔为 2 秒。', '']
    (root / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
