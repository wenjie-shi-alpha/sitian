import json

import pytest

from sitian.case import CaseBundle
from sitian.provenance import case_bundle_snapshot, file_identity
from sitian.scoring import reward_spec
from sitian import training_ready


def certificate(tmp_path, monkeypatch):
    monkeypatch.setattr(training_ready, "ROOT", tmp_path)
    monkeypatch.setattr(training_ready.subprocess, "check_output", lambda *a, **k: "pinned\n")
    case_path=tmp_path/"case"
    CaseBundle("test", "2025-01-01", "test", 1, truth={"daily":{"2025-01-02":{"pm25_avg":30}}}).save(case_path)
    code=tmp_path/"implementation.py";code.write_text("version = 1\n")
    record={"status":"offline_training_ready","actual_publication_verified":False,"training_started":False,
        "checks":{"all_passed":True},"files":[file_identity(code,relative_to=tmp_path)],
        "case_snapshot":case_bundle_snapshot([case_path],relative_to=tmp_path),
        "verl_commit":"pinned","reward":reward_spec()}
    (tmp_path/"certificate.json").write_text(json.dumps(record))
    return record


def test_integrity_guard_rejects_modified_code(tmp_path, monkeypatch):
    record=certificate(tmp_path,monkeypatch)
    assert training_ready.verify_bundle(tmp_path)==record
    (tmp_path/"implementation.py").write_text("version = 2\n")
    with pytest.raises(ValueError,match="stale training input/code"):
        training_ready.verify_bundle(tmp_path)


def test_integrity_guard_rejects_modified_hidden_truth(tmp_path, monkeypatch):
    certificate(tmp_path,monkeypatch)
    (tmp_path/"case/truth.json").write_text('{"daily":{}}')
    with pytest.raises(ValueError,match="case snapshot changed"):
        training_ready.verify_bundle(tmp_path)


def test_certificate_never_promotes_assumptions_to_real_publication(tmp_path, monkeypatch):
    record=certificate(tmp_path,monkeypatch)
    record["actual_publication_verified"]=True
    (tmp_path/"certificate.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="not an offline"):
        training_ready.verify_bundle(tmp_path)
