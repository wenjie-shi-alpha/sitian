#!/usr/bin/env python3
"""Plot request occupancy and physical GPU utilization from a completed benchmark."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--phases', nargs='+', default=['b8_s16', 'b16_s16'])
    parser.add_argument('--title', default='Same 16 cases / 128 trajectories: request occupancy and GPU utilization')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = args.run_dir.resolve()
    rows = [json.loads(line) for line in (root / 'resources.jsonl').read_text().splitlines()]
    peak_waiting = max(sum(item.get('metrics', {}).get('num_requests_waiting', 0)
                           for item in row.get('rollout', []))
                       for row in rows if row.get('phase') in args.phases)
    peak_running = max(sum(item.get('metrics', {}).get('num_requests_running', 0)
                           for item in row.get('rollout', []))
                       for row in rows if row.get('phase') in args.phases)
    colors = ['#276FBF', '#D97706', '#16856B', '#9A55AD']
    fig, axes = plt.subplots(len(args.phases), 2, figsize=(13, 3.5 * len(args.phases) + .5),
                             squeeze=False, constrained_layout=True)
    max_minutes = 0
    queue_axes = []
    for index, phase in enumerate(args.phases):
        match = re.fullmatch(r'b(\d+)_s(\d+)', phase)
        title = f'Batch {match[1]} / concurrency {match[2]}' if match else phase
        if args.phases == ['b8_s16', 'b16_s16']:
            title += ': two updates' if index == 0 else ': one update'
        data = [row for row in rows if row.get('phase') == phase and len(row.get('rollout', [])) == 4
                and all('num_requests_running' in item.get('metrics', {}) for item in row['rollout'])]
        first_active = next(i for i, row in enumerate(data)
                            if sum(item['metrics']['num_requests_running'] + item['metrics'].get('num_requests_waiting', 0)
                                   for item in row['rollout']) > 0)
        data = data[first_active:]
        start = datetime.fromisoformat(data[0]['utc'])
        x = [(datetime.fromisoformat(row['utc']) - start).total_seconds() / 60 for row in data]
        max_minutes = max(max_minutes, max(x))
        left, right = axes[index]
        running = [[row['rollout'][replica]['metrics']['num_requests_running'] for row in data] for replica in range(4)]
        waiting = [sum(item['metrics'].get('num_requests_waiting', 0) for item in row['rollout']) for row in data]
        left.stackplot(x, running, labels=[f'Replica {i}' for i in range(4)], colors=colors, alpha=.85)
        queued = left.twinx()
        queue_axes.append(queued)
        queued.plot(x, waiting, color='#333333', linestyle='--', linewidth=1.4, label='Queued total (right axis)')
        queued.set(ylabel='Queued requests', ylim=(0, max(66, peak_waiting * 1.05)))
        queued.spines['top'].set_visible(False)
        left.set(title=title, ylabel='Running requests (stacked)', ylim=(0, max(66, peak_running * 1.03)))
        for gpu in range(4):
            usage = [next(g['utilization_percent'] for g in row['gpus'] if g['index'] == gpu) for row in data]
            right.plot(x, usage, color=colors[gpu], linewidth=1.1, alpha=.8, label=f'GPU {gpu}')
        right.set(title='Physical GPU utilization', ylabel='GPU utilization (%)', ylim=(-2, 105))
        for ax in (left, right):
            ax.set_xlabel('Minutes after first rollout request')
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
    for ax in axes.flat:
        ax.set_xlim(0, max_minutes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    queue_handles, queue_labels = queue_axes[0].get_legend_handles_labels()
    axes[0, 0].legend(handles + queue_handles, labels + queue_labels,
                      ncol=2, fontsize=8, loc='upper right')
    axes[0, 1].legend(ncol=4, fontsize=8, loc='lower right')
    fig.suptitle(args.title + '\nZero rollout requests can include policy training. Telemetry polled about every 2 seconds.', fontsize=12)
    fig.savefig(root / 'occupancy.png', dpi=160)
    fig.savefig(root / 'occupancy.svg')
    (root / 'plot_runtime.json').write_text(json.dumps({'matplotlib': matplotlib.__version__}, indent=2) + '\n')
    print(root / 'occupancy.png')


if __name__ == '__main__':
    main()
