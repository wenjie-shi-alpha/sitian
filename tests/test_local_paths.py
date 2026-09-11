import pytest

from sitian import paths


def test_case_paths_survive_changed_worker_cwd_and_explicit_migration(tmp_path, monkeypatch):
    project = tmp_path / "new"
    case = project / "cases" / "national" / "sample"
    case.mkdir(parents=True)
    (case / "case.json").write_text("{}")
    monkeypatch.setattr(paths, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SITIAN_LEGACY_PROJECT_ROOT", raising=False)
    assert paths.resolve_case_dir("cases/national/sample") == case
    assert paths.resolve_case_dir(case) == case
    with pytest.raises(FileNotFoundError, match="SITIAN_LEGACY_PROJECT_ROOT"):
        paths.resolve_case_dir("/old/sitian/cases/national/sample")
    monkeypatch.setenv("SITIAN_LEGACY_PROJECT_ROOT", "/old/sitian")
    assert paths.resolve_case_dir("/old/sitian/cases/national/sample") == case
    # A similarly prefixed project must never be silently remapped.
    with pytest.raises(FileNotFoundError):
        paths.resolve_case_dir("/old/sitian-other/cases/national/sample")
    with pytest.raises(FileNotFoundError):
        paths.resolve_case_dir("cases/national/missing")
