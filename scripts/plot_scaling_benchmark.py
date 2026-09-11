#!/usr/bin/env python3
"""Plot request occupancy and physical GPU utilization from a completed benchmark."""
import argparse
from datetime import datetime
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = args.run_dir.resolve()
    rows = [json.loads(line) for line in (root / 'resources.jsonl').read_text().splitlines()]
    colors = ['#276FBF', '#D97706', '#16856B', '#9A55AD']
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    max_minutes = 0
    for index, (phase, title) in enumerate([
        ('b8_s16', 'Batch 8 / concurrency 16: two updates'),
        ('b16_s16', 'Batch 16 / concurrency 16: one update'),
    ]):
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
        left.plot(x, waiting, color='#333333', linestyle='--', linewidth=1.4, label='Queued total')
        left.set(title=title, ylabel='Running requests (stacked) / queued total')
        left.set_ylim(0, max(66, max(waiting) * 1.05))
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
    axes[0, 0].legend(ncol=3, fontsize=8, loc='upper right')
    axes[0, 1].legend(ncol=4, fontsize=8, loc='lower right')
    fig.suptitle('Same 16 cases / 128 trajectories: request occupancy and GPU utilization\n'
                 'Zero rollout requests can include policy training. Telemetry polled about every 2 seconds.', fontsize=12)
    fig.savefig(root / 'occupancy.png', dpi=160)
    fig.savefig(root / 'occupancy.svg')
    (root / 'plot_runtime.json').write_text(json.dumps({'matplotlib': matplotlib.__version__}, indent=2) + '\n')
    print(root / 'occupancy.png')


if __name__ == '__main__':
    main()
