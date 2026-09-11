#!/usr/bin/env python3
"""Check a sealed bundle by default; --start explicitly launches a fresh run."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sitian.training_ready import verify_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bundle",required=True,type=Path)
    group=p.add_mutually_exclusive_group();group.add_argument("--start",action="store_true");group.add_argument("--check-only",action="store_true")
    a=p.parse_args();out=a.bundle.resolve();certificate=verify_bundle(out)
    cfg=yaml.safe_load((out/"train.yaml").read_text())
    if cfg["trainer"]["resume_mode"]!="disable" or cfg["trainer"].get("resume_from_path") is not None:
        raise ValueError("this protocol requires a fresh base model, no old checkpoint")
    print(json.dumps({"status":certificate["status"],"verified_cases":certificate["case_snapshot"]["cases"],
                      "actual_publication_verified":False,"training_started":False}),flush=True)
    if not a.start: return
    checkpoints=Path(cfg["trainer"]["default_local_dir"])
    if checkpoints.exists() and any(checkpoints.iterdir()): raise ValueError("checkpoint directory is not empty")
    environment={**os.environ,**json.loads((out/"runtime_environment.json").read_text())}
    runtime=ROOT/".local/verl-upstream"
    subprocess.run([str(runtime/".venv/bin/python"),"-c",
        "import torch; assert torch.cuda.device_count() >= 4; assert all(torch.cuda.mem_get_info(i)[0] >= 80*1024**3 for i in range(4)), 'Need four GPUs with >=80 GiB free each'"],env=environment,check=True)
    (out/"future_training").mkdir(exist_ok=True)
    (out/"future_training/launch.lock").mkdir()
    command=[str(runtime/".venv/bin/python"),"-m","verl.trainer.main_ppo","--config-path",str(out),"--config-name","train"]
    with (out/"future_training/train.log").open("x") as log:
        result=subprocess.run(command,cwd=runtime,env=environment,stdout=log,stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__=="__main__": main()
