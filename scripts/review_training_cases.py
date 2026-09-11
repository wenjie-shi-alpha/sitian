#!/usr/bin/env python3
"""Build an inspectable case ledger from TRAINING rollout dumps only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

CALL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
RESPONSE = re.compile(r'<tool_response>\s*(.*?)\s*</tool_response>', re.S)
CASE = re.compile(r'"case_id"\s*:\s*"([^"\n]+)"')


def inspect_record(row, source, line_number):
    match = CASE.search(row['input'])
    if not match:
        raise ValueError(f'No case identity in {source}:{line_number}')
    text = row['output']
    calls, malformed, responses = [], [], []
    for block in CALL.findall(text):
        try:
            call = json.loads(block)
            calls.append(call.get('name', 'unknown') if isinstance(call, dict) else 'non_object')
        except json.JSONDecodeError as error:
            malformed.append({'error': str(error), 'excerpt': block[:240]})
    for block in RESPONSE.findall(text):
        try:
            responses.append(json.loads(block))
        except json.JSONDecodeError:
            pass  # Some upstream tool responses are plain text.
    accepted = any(isinstance(item, dict) and item.get('accepted') is True for item in responses)
    tool_errors = [item for item in responses if isinstance(item, dict)
                   and (item.get('error') or item.get('accepted') is False)]
    return {'case_id': match[1], 'step': row['step'], 'source': str(source),
            'line': line_number, 'prompt_sha256': hashlib.sha256(row['input'].encode()).hexdigest(),
            'score': row['score'], 'accepted_submission_observed': accepted,
            'response_characters_including_tools': len(text), 'tool_calls': calls,
            'malformed_closed_tool_calls': malformed,
            'unclosed_tool_call_tags': max(0, text.count('<tool_call>') - text.count('</tool_call>')),
            'tool_errors': tool_errors,
            'failure_tail': text[-1000:] if not accepted else None}


def review(paths, allowed_case_ids, destination):
    records = []
    for path in paths:
        for index, line in enumerate(path.read_text().splitlines(), 1):
            item = inspect_record(json.loads(line), path, index)
            if item['case_id'] not in allowed_case_ids:
                raise ValueError(f"CDD input is outside TRAIN population: {item['case_id']}")
            records.append(item)
    if not records:
        raise ValueError('No completed training rollouts')
    grouped = defaultdict(list)
    for row in records:
        grouped[row['case_id']].append(row)
    cases = []
    for cid, rows in grouped.items():
        lengths = [row['response_characters_including_tools'] for row in rows]
        cases.append({'case_id': cid, 'trajectories': len(rows),
                      'accepted': sum(row['accepted_submission_observed'] for row in rows),
                      'malformed_trajectories': sum(bool(row['malformed_closed_tool_calls']) for row in rows),
                      'min_response_characters': min(lengths), 'max_response_characters': max(lengths),
                      'length_ratio_max_min': max(lengths) / max(min(lengths), 1),
                      'mean_reward': sum(row['score'] for row in rows) / len(rows)})
    summary = {'trajectories': len(records), 'distinct_cases': len(cases),
               'accepted_submission_observed': sum(r['accepted_submission_observed'] for r in records),
               'malformed_tool_call_trajectories': sum(bool(r['malformed_closed_tool_calls']) for r in records),
               'tool_error_trajectories': sum(bool(r['tool_errors']) for r in records),
               'unclosed_tool_call_trajectories': sum(bool(r['unclosed_tool_call_tags']) for r in records),
               'tool_call_counts': dict(Counter(c for r in records for c in r['tool_calls']))}
    result = {'summary': summary, 'cases': sorted(cases, key=lambda c: c['accepted'] / c['trajectories']),
              'records': records, 'limitations': [
                  'Only retained training groups; DAPO-filtered groups are not included.',
                  'Text parsing is diagnostic, not a replacement for native runtime submission metrics.',
                  'Characters include tool text and do not measure token count or wall time.',
                  'Malformed model JSON is a model behavior case, not evidence of a runtime bug.',
              ]}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'cases.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 训练案例复盘', '', f"共 {len(records)} 条轨迹、{len(cases)} 个训练案例。",
             f"观察到成功提交 {summary['accepted_submission_observed']} 条；含无效工具 JSON 的轨迹 {summary['malformed_tool_call_trajectories']} 条。", '',
             '| 案例 | 成功提交 / 轨迹 | 无效 JSON 轨迹 | 最长/最短响应字符比 |', '|---|---:|---:|---:|']
    for case in result['cases']:
        lines.append(f"| {case['case_id']} | {case['accepted']}/{case['trajectories']} | {case['malformed_trajectories']} | {case['length_ratio_max_min']:.2f} |")
    lines += ['', '开发决策：保留失败轨迹的真实零奖励与现有工具校验，避免自动修补 JSON 改变训练任务。',
              '以同一案例中的长短轨迹、无效调用和无提交样本建立回归线索；扩大全局 batch 测量等待是否下降。',
              '病例来源、行号、失败片段保存在 cases.json。字符长度包含工具返回；不能当作精确耗时。',
              '仅分析训练集；不使用验证集或封存测试集进行案例驱动修改。', '']
    (destination / 'cases.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', type=Path, nargs='+', required=True)
    parser.add_argument('--train-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    allowed = {item['case_id'] for item in pd.read_parquet(args.train_file)['extra_info']}
    paths = [path for folder in args.rollouts for path in sorted(folder.glob('*.jsonl'))]
    print(json.dumps(review(paths, allowed, args.out), ensure_ascii=False))


if __name__ == '__main__':
    main()
