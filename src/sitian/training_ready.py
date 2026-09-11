"""Fail-closed integrity checks for a prepared offline training bundle."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .provenance import case_bundle_snapshot_matches, file_identity

ROOT=Path(__file__).resolve().parents[2]


def verify_bundle(directory):
    root=Path(directory).resolve()
    certificate=json.loads((root/"certificate.json").read_text())
    if (certificate.get("status")!="offline_training_ready"
            or certificate.get("actual_publication_verified") is not False
            or certificate.get("training_started") is not False):
        raise ValueError("not an offline training-readiness certificate")
    if not certificate.get("checks") or not all(value is True for value in certificate["checks"].values()):
        raise ValueError("readiness checks incomplete")
    for expected in certificate["files"]:
        path=Path(expected["path"])
        actual=file_identity(path if path.is_absolute() else ROOT/path)
        if actual["sha256"]!=expected["sha256"] or actual["bytes"]!=expected["bytes"]:
            raise ValueError(f"stale training input/code: {path}")
    if not case_bundle_snapshot_matches(certificate["case_snapshot"],ROOT):
        raise ValueError("case snapshot changed; rebuild dependent indices and Parquet")
    verl=ROOT/".local/verl-upstream"
    commit=subprocess.check_output(["git","-C",str(verl),"rev-parse","HEAD"],text=True).strip()
    if commit!=certificate["verl_commit"]:
        raise ValueError("veRL revision changed")
    from .scoring import reward_spec
    if reward_spec()!=certificate["reward"]:
        raise ValueError("reward contract changed")
    return certificate
