#!/usr/bin/env python3
"""Summarize global-batch scaling and rollout occupancy on the local four GPUs."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_batch_benchmark import read_arm


def occupancy(samples, phase):
    complete = []
    for row in samples:
        if row.get('phase') != phase or len(row.get('rollout', [])) != 4:
            continue
        replicas = [item.get('metrics', {}) for item in row['rollout']]
        if not all('num_requests_running' in metrics for metrics in replicas):
            continue
        running = [m['num_requests_running'] for m in replicas]
        waiting = [m.get('num_requests_waiting', 0) for m in replicas]
        complete.append((row, replicas, running, waiting))
    active = [item for item in complete if sum(item[2]) + sum(item[3]) > 0]
    if not active:
        return {'available': False, 'complete_scrapes': len(complete)}
    preemptions = max(sum(m.get('num_preemptions_total', 0) for m in item[1]) for item in complete)
    # A replica is idle by this proxy when it has neither running nor queued requests.
    idle = [sum(r + w == 0 for r, w in zip(item[2], item[3])) for item in active]
    sparse_gpu = [sum(g['utilization_percent'] < 10 for g in item[0]['gpus']) >= 2 for item in active]
    return {'available': True, 'complete_scrapes': len(complete), 'active_scrapes': len(active),
            'active_scrapes_with_at_least_two_idle_replicas_fraction': sum(n >= 2 for n in idle) / len(idle),
            'mean_idle_replicas_during_active_rollout': sum(idle) / len(idle),
            'active_scrapes_with_at_least_two_gpus_below_10_percent_fraction': sum(sparse_gpu) / len(sparse_gpu),
            'mean_running_requests_total': sum(sum(item[2]) for item in active) / len(active),
            'peak_running_requests_total': max(sum(item[2]) for item in active),
            'peak_waiting_requests_total': max(sum(item[3]) for item in active),
            'max_preemptions_total': preemptions,
            'generation_tokens_total': max(sum(m.get('generation_tokens_total', 0) for m in item[1]) for item in complete),
            'max_kv_cache_usage_fraction': max(max(m.get('kv_cache_usage_perc', m.get('gpu_cache_usage_perc', 0)) for m in item[1]) for item in complete)}


def hashes(path):
    return {hashlib.sha256(json.loads(line)['input'].encode()).hexdigest() for line in path.read_text().splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    resources = [json.loads(line) for line in (root / 'resources.jsonl').read_text().splitlines()]
    arms = {}
    for spec in protocol['arms']:
        result = read_arm(root, spec['batch'], resources, arm_name=spec['name'],
                          target_prompts=protocol['target_retained_prompts_per_arm'])
        result['max_num_seqs'] = spec['max_num_seqs']
        result['occupancy'] = occupancy(resources, spec['name'])
        if result['occupancy']['available']:
            result['occupancy']['generated_tokens_per_generation_second'] = (
                result['occupancy']['generation_tokens_total'] / result['generation_seconds_total'])
        arms[spec['name']] = result
    a, b = arms['b8_s16'], arms['b16_s16']
    comparison = {
        'same_retained_prompts': a['prompt_hashes'] == b['prompt_hashes'],
        'overlap_prompts': len(set(a['prompt_hashes']) & set(b['prompt_hashes'])),
        'case_throughput_ratio_batch16_over_batch8': a['step_seconds_total'] / b['step_seconds_total'],
        'sequence_token_throughput_ratio_batch16_over_batch8': b['sequence_tokens_per_second_all_gpus'] / a['sequence_tokens_per_second_all_gpus'],
        'sequence_token_volume_ratio_batch16_over_batch8': b['sequence_tokens_total_including_tools'] / a['sequence_tokens_total_including_tools'],
    }
    reference = json.loads((args.reference_dir / 'comparison.json').read_text())['arms']['8']
    first = a['metrics_by_step'][1]
    reference_prompts = set(reference['prompt_hashes'])
    current_prompts = hashes(root / 'b8_s16/rollouts/1.jsonl')
    concurrency = {
        'same_retained_prompts': reference_prompts == current_prompts,
        'overlap_prompts': len(reference_prompts & current_prompts),
        'reference_step_seconds': reference['step_seconds_total'],
        'current_step_seconds': first['timing_s/step'],
        'reference_generation_seconds': reference['generation_seconds_total'],
        'current_generation_seconds': first['timing_s/gen'],
        'case_throughput_ratio_concurrency16_over_concurrency8': reference['step_seconds_total'] / first['timing_s/step'],
        'sequence_token_throughput_ratio_concurrency16_over_concurrency8':
            (first['global_seqlen/mean'] * 4 / first['timing_s/step']) / reference['sequence_tokens_per_second_all_gpus'],
        'note': 'Historical reference, first step only; new arms also enable lightweight vLLM metrics.',
    }
    result = {'arms': arms, 'batch_comparison': comparison, 'concurrency_comparison': concurrency,
              'limitations': protocol['limitations'] + ['Occupancy is a polling proxy; metric refresh lag and GPU sampling can affect it.',
                                                       'Active-rollout samples include ramp-up and tool-call pauses; they are not pure tail-wait measurements.']}
    (root / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 四卡 batch 与采样并发扩展测试', '',
             '| 指标 | batch8 / 并发16 × 2步 | batch16 / 并发16 × 1步 |', '|---|---:|---:|']
    for label, key in [('累计步骤耗时（秒）', 'step_seconds_total'), ('累计采样耗时（秒）', 'generation_seconds_total'),
                       ('累计梯度更新耗时（秒）', 'actor_update_seconds_total'),
                       ('四卡序列 token/s（含提示与工具文本）', 'sequence_tokens_per_second_all_gpus'),
                       ('案例/分钟', 'retained_prompts_per_minute')]:
        lines.append(f'| {label} | {a[key]:.2f} | {b[key]:.2f} |')
    lines += ['', f"案例重合：{comparison['overlap_prompts']}/16。",
              f"batch16 / batch8 案例吞吐比：{comparison['case_throughput_ratio_batch16_over_batch8']:.3f}。",
              f"batch16 / batch8 token 吞吐比：{comparison['sequence_token_throughput_ratio_batch16_over_batch8']:.3f}。", '',
              '并发 8→16 的历史对照仅比较 batch8 的第一步，详情及资源占用代理指标见 comparison.json。',
              '每组一次短测；两组优化器更新次数不同，不评估训练质量。请求占用采样不是精确的尾部等待计时。', '']
    (root / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps({'batch': comparison, 'concurrency': concurrency}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
