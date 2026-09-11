#!/usr/bin/env python3
"""Fail before training if any dataset row points at a missing or wrong case."""
import argparse
import json

import pyarrow.parquet as pq

from sitian.paths import resolve_case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    checked = {}
    rows = 0
    for dataset in args.datasets:
        for row in pq.read_table(dataset, columns=["extra_info"]).to_pylist():
            info = row["extra_info"]
            paths = {
                tool["create_kwargs"]["case_dir"]
                for tool in (info.get("tools_kwargs") or {}).values() if tool
            }
            if len(paths) != 1:
                raise ValueError(f"{dataset}: expected exactly one case path for {info.get('case_id')}")
            value = paths.pop()
            if value not in checked:
                path = resolve_case_dir(value)
                checked[value] = json.loads((path / "case.json").read_text(encoding="utf-8"))["case_id"]
            if checked[value] != info["case_id"]:
                raise ValueError(f"{dataset}: case identity mismatch for {value}")
            rows += 1
    print(json.dumps({"datasets": args.datasets, "rows": rows, "cases": len(checked), "passed": True}))


if __name__ == "__main__":
    main()
