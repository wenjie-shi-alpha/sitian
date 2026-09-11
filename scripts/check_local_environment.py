#!/usr/bin/env python3
"""Check this interpreter; --gpu additionally executes kernels on every GPU."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "checks": {}, "errors": []}
    if sys.version_info < (3, 11):
        report["errors"].append("sitian requires Python >=3.11; use 3.12 for veRL")
    modules = ["sitian"] if args.gpu else [
        "sitian", "PIL", "numpy", "sklearn", "lightgbm", "xgboost", "pyarrow",
        "xarray", "netCDF4", "cfgrib", "pygrib", "eccodes", "rasterio", "cdsapi", "ee",
    ]
    for name in modules:
        try:
            module = importlib.import_module(name)
            report["checks"][name] = getattr(module, "__version__", "import OK")
        except Exception as exc:
            report["errors"].append(f"{name}: {exc}")
    if args.gpu:
        try:
            import torch
            report["checks"]["torch"] = torch.__version__
            report["cuda_runtime"] = torch.version.cuda
            report["cuda_arches"] = torch.cuda.get_arch_list()
            report["gpus"] = []
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    x = torch.randn(128, 128, device=f"cuda:{index}", dtype=torch.bfloat16)
                    result = x @ x.T
                    assert torch.isfinite(result).all().item(), "non-finite matmul"
                    torch.cuda.synchronize()
                    report["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                           "capability": torch.cuda.get_device_capability(index),
                                           "bf16_matmul": "passed"})
            for package in ("vllm", "transformers", "peft", "ray", "verl"):
                try:
                    report["checks"][package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    report["checks"][package] = "not installed (training-only dependencies optional)"
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, check=True)
            report["driver_versions"] = result.stdout.splitlines()
        except Exception as exc:
            report["errors"].append(f"GPU: {exc}")
    report["passed"] = not report["errors"]
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
