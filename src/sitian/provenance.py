"""评测产物的轻量来源追踪（不依赖 git 工作树）。"""
from __future__ import annotations

import hashlib
from pathlib import Path


CASE_VISIBLE_FILES = (
    "observations.json", "diagnostics.json", "guidance.json",
    "previous_forecast.json", "evidence.json",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path, *, relative_to: str | Path | None = None) -> dict:
    p = Path(path).resolve()
    shown = p.relative_to(Path(relative_to).resolve()) if relative_to else p
    return {"path": str(shown), "sha256": file_sha256(p), "bytes": p.stat().st_size}


def identity_matches_file(
    identity: dict | None,
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> bool:
    """Check that a stored identity still represents the exact file contents."""
    if not isinstance(identity, dict) or not identity.get("path"):
        return False
    try:
        current = file_identity(path, relative_to=relative_to)
    except OSError:
        return False
    return (
        identity.get("sha256") == current["sha256"]
        and identity.get("bytes") == current["bytes"]
    )


def score_implementation_matches(identities: dict | None, repo_root: str | Path) -> bool:
    """Validate both files that define whether and how a forecast is scored."""
    identities = identities or {}
    root = Path(repo_root)
    return identity_matches_file(
        identities.get("scoring_implementation"),
        root / "src/sitian/scoring.py", relative_to=root,
    ) and identity_matches_file(
        identities.get("schema_implementation"),
        root / "src/sitian/schema.py", relative_to=root,
    )


def case_bundle_snapshot(
    case_dirs: list[str | Path],
    *,
    relative_to: str | Path,
    include_truth: bool = True,
) -> dict:
    """Hash every file that can affect one replayed task or its rule reward."""
    root = Path(relative_to).resolve()
    resolved = sorted({Path(path).resolve() for path in case_dirs})
    names = ("case.json", *CASE_VISIBLE_FILES, *(("truth.json",) if include_truth else ()))
    digest = hashlib.sha256()
    files = total_bytes = 0
    shown_paths = []
    for case_dir in resolved:
        try:
            shown = case_dir.relative_to(root).as_posix()
        except ValueError:
            shown = case_dir.as_posix()
        shown_paths.append(shown)
        if (case_dir / "expert.json").exists():
            raise ValueError(f"expert file forbidden in national replay input: {shown}")
        for name in names:
            path = case_dir / name
            if not path.is_file():
                continue
            digest.update(shown.encode("utf-8"))
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            size = path.stat().st_size
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files += 1
            total_bytes += size
    return {
        "algorithm": "sha256(relative_case\\0filename\\0content)-v1",
        "cases": len(resolved),
        "files": files,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "case_paths": shown_paths,
        "included": list(names),
        "excluded_and_forbidden": ["expert.json"],
    }


def case_bundle_snapshot_matches(snapshot: dict | None, repo_root: str | Path) -> bool:
    """Recompute a stored case snapshot to reject stale probe/baseline artifacts."""
    if not isinstance(snapshot, dict):
        return False
    root = Path(repo_root).resolve()
    paths = snapshot.get("case_paths")
    if not isinstance(paths, list) or not paths:
        return False
    try:
        current = case_bundle_snapshot(
            [Path(path) if Path(path).is_absolute() else root / path for path in paths],
            relative_to=root,
            include_truth="truth.json" in (snapshot.get("included") or []),
        )
    except (OSError, ValueError):
        return False
    return all(current.get(key) == snapshot.get(key) for key in (
        "algorithm", "cases", "files", "bytes", "sha256", "case_paths",
        "included", "excluded_and_forbidden",
    ))
