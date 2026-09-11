import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_open_cams.py"
    spec = importlib.util.spec_from_file_location("fetch_open_cams_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Remote:
    request_id = "test-request-id"

    def __init__(self, ready):
        self.ready = ready
        self.deleted = False

    @property
    def status(self):
        return "successful" if self.ready else "running"

    @property
    def results_ready(self):
        return self.ready

    def download(self, target):
        Path(target).write_bytes(b"result")

    def get_results(self):
        return SimpleNamespace(location="https://example.invalid/result", content_length=6)

    def update(self):
        pass

    def delete(self):
        self.deleted = True


class _Client:
    def __init__(self, remote):
        self.remote = remote

    def submit(self, collection_id, request):
        return self.remote

    def get_remote(self, request_id):
        assert request_id == self.remote.request_id
        return self.remote


def test_async_retrieval_records_request_and_downloads(tmp_path, monkeypatch):
    cams = _module()
    remote = _Remote(ready=True)
    target = tmp_path / "result.zip"
    def fake_run(command, check, timeout):
        target.write_bytes(b"result")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cams.subprocess, "run", fake_run)
    retrieval = cams._retrieve_with_deadline(
        _Client(remote), {"variable": ["x"]}, target, 1, 0.01
    )
    assert retrieval["ads_request_id"] == "test-request-id"
    assert retrieval["elapsed_seconds"] >= 0
    assert target.read_bytes() == b"result"
    assert not remote.deleted


def test_async_retrieval_resumes_persisted_request(tmp_path, monkeypatch):
    cams = _module()
    remote = _Remote(ready=True)
    client = _Client(remote)
    request = {"variable": ["x"]}
    target = tmp_path / "result.zip"
    digest = sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    Path(str(target) + ".request.json").write_text(json.dumps({
        "ads_request_id": remote.request_id,
        "submitted_at": "2026-08-30T00:00:00Z",
        "request_sha256": digest,
    }), encoding="utf-8")
    def fail_submit(*args):
        raise AssertionError("a persisted job must not be resubmitted")
    client.submit = fail_submit
    def fake_run(command, check, timeout):
        target.write_bytes(b"result")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cams.subprocess, "run", fake_run)
    retrieval = cams._retrieve_with_deadline(client, request, target, 1, 0.01)
    assert retrieval["ads_request_id"] == remote.request_id


def test_async_retrieval_cancels_wall_clock_timeout(tmp_path):
    cams = _module()
    remote = _Remote(ready=False)
    with pytest.raises(TimeoutError, match="was cancelled"):
        cams._retrieve_with_deadline(
            _Client(remote), {"variable": ["x"]}, tmp_path / "result.zip", 0, 0.01
        )
    assert remote.deleted


def test_publish_from_stage_is_atomic_and_hashed(tmp_path):
    cams = _module()
    staged = tmp_path / "stage" / "payload.nc"
    target = tmp_path / "external" / "payload.nc"
    staged.parent.mkdir()
    target.parent.mkdir()
    staged.write_bytes(b"verified payload")
    byte_count, digest = cams._publish_from_stage(staged, target, tmp_path / "publish.lock")
    assert byte_count == len(b"verified payload")
    assert digest == sha256(b"verified payload").hexdigest()
    assert target.read_bytes() == b"verified payload"
    assert not target.with_suffix(".nc.part").exists()
