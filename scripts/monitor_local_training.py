#!/usr/bin/env python3
"""Sample whole-GPU usage and host RAM during a bounded local benchmark."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import re
import urllib.request


class RolloutMetrics:
    """Discover this run's vLLM endpoints and read counters without a Prometheus server."""
    def __init__(self):
        self.log = None
        self.phase = None
        self.endpoints = []
        self.pool = ThreadPoolExecutor(max_workers=4)

    def scrape(self, endpoint):
        names = ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
                 'gpu_cache_usage_perc', 'generation_tokens_total', 'prompt_tokens_total',
                 'num_preemptions_total', 'request_success_total')
        try:
            with urllib.request.urlopen(f'http://{endpoint}/metrics', timeout=1) as response:
                lines = response.read().decode().splitlines()
            values = {}
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                name = line.split('{', 1)[0].split()[0].removeprefix('vllm:')
                if name in names:
                    values[name] = values.get(name, 0) + float(line.rsplit(' ', 1)[-1])
            return {'endpoint': endpoint, 'metrics': values}
        except (OSError, ValueError) as error:
            return {'endpoint': endpoint, 'error': str(error)}

    def read(self, root, phase):
        if phase != self.phase:
            if self.log:
                self.log.close()
            self.log, self.endpoints, self.phase = None, [], phase
        path = root / phase / 'train.log'
        if self.log is None and path.is_file():
            self.log = path.open(errors='replace')
        if self.log:
            for line in self.log:
                match = re.search(r'LLMServerManager: (\[[^\n]*\])', line)
                if match:
                    self.endpoints = ast.literal_eval(match.group(1))
        return list(self.pool.map(self.scrape, self.endpoints))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--rollout-metrics', action='store_true')
    args = parser.parse_args()
    rollout = RolloutMetrics() if args.rollout_metrics else None
    with (args.run_dir / 'resources.jsonl').open('a', buffering=1) as output:
        while True:
            try:
                raw = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits',
                ], text=True, timeout=5)
                gpus = []
                for line in raw.splitlines():
                    index, memory, utilization, power = line.split(', ')
                    gpus.append({'index': int(index), 'memory_mib': float(memory),
                                 'utilization_percent': float(utilization), 'power_w': float(power)})
                mem = {k: int(v.strip().split()[0]) for k, v in
                       (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
                phase = args.run_dir / 'phase'
                row = {'utc': datetime.now(timezone.utc).isoformat(),
                       'phase': phase.read_text().strip() if phase.exists() else 'preflight',
                       'gpus': gpus, 'host_mem_available_mib': mem['MemAvailable'] / 1024}
                if rollout:
                    row['rollout'] = rollout.read(args.run_dir, row['phase'])
                output.write(json.dumps(row) + '\n')
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                output.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'error': str(error)}) + '\n')
            time.sleep(2)


if __name__ == '__main__':
    main()
