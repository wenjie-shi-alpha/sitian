#!/usr/bin/env python3
"""Sample whole-GPU usage and host RAM during a bounded local benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
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
                output.write(json.dumps(row) + '\n')
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                output.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'error': str(error)}) + '\n')
            time.sleep(2)


if __name__ == '__main__':
    main()
