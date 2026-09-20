"""Unit checks for the git-free Amarel compute-node source guard.

These checks exercise `experiments/remote/amarel_source_guard.sh` directly so the compute-node
fallback chain (git metadata, then submit-time stamp) is regression-covered without a cluster.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "experiments/remote/amarel_source_guard.sh"
BASH = shutil.which("bash") or "/bin/bash"
COMMIT = "571b260cd274c2633c55fcef7bdda0b9bcec9de0"
OTHER_COMMIT = "e44cb2c3819fb844932fb2d5cc5227bce486716a"


def _run_guard(
    tmp_path: Path,
    root: Path,
    expected: str,
    stamp: Path | None = None,
    *,
    with_git: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Call `amarel_require_source_commit` in a controlled environment."""

    stamp_argument = f'"{stamp}"' if stamp is not None else '""'
    script = (
        "set -euo pipefail\n"
        f'source "{GUARD}"\n'
        f'amarel_require_source_commit "{root}" "{expected}" {stamp_argument}\n'
    )
    environment = dict(os.environ)
    environment.pop("AIM3_GIT_BIN", None)
    if not with_git:
        empty_path = tmp_path / "empty-path"
        empty_path.mkdir(exist_ok=True)
        environment["PATH"] = str(empty_path)
    return subprocess.run(
        [BASH, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def _write_loose_ref_checkout(root: Path, commit: str, branch: str = "sjc") -> None:
    gitdir = root / ".git"
    (gitdir / "refs" / "heads").mkdir(parents=True)
    (gitdir / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    (gitdir / "refs" / "heads" / branch).write_text(f"{commit}\n", encoding="utf-8")


def test_loose_ref_checkout_resolves_without_git(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    _write_loose_ref_checkout(checkout, COMMIT)

    result = _run_guard(tmp_path, checkout, COMMIT)

    assert result.returncode == 0, result.stderr


def test_detached_head_and_packed_ref_worktree_resolve_without_git(tmp_path: Path) -> None:
    detached = tmp_path / "detached"
    detached.mkdir()
    (detached / ".git").mkdir()
    (detached / ".git" / "HEAD").write_text(f"{COMMIT}\n", encoding="utf-8")
    assert _run_guard(tmp_path, detached, COMMIT).returncode == 0

    worktree_gitdir = tmp_path / "main.git" / "worktrees" / "dyn"
    worktree_gitdir.mkdir(parents=True)
    (worktree_gitdir / "HEAD").write_text("ref: refs/heads/dyn\n", encoding="utf-8")
    (worktree_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (tmp_path / "main.git" / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{COMMIT} refs/heads/dyn\n",
        encoding="utf-8",
    )
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_gitdir}\n", encoding="utf-8")

    result = _run_guard(tmp_path, worktree, COMMIT)

    assert result.returncode == 0, result.stderr


def test_changed_commit_without_git_is_rejected(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    _write_loose_ref_checkout(checkout, OTHER_COMMIT)

    result = _run_guard(tmp_path, checkout, COMMIT)

    assert result.returncode != 0
    assert "Source commit changed after submission" in result.stderr


def test_submit_time_stamp_is_the_documented_fallback(tmp_path: Path) -> None:
    checkout = tmp_path / "snapshot"
    checkout.mkdir()
    stamp = tmp_path / "source_commit.txt"
    stamp.write_text(f"{COMMIT}\n", encoding="utf-8")

    result = _run_guard(tmp_path, checkout, COMMIT, stamp)

    assert result.returncode == 0, result.stderr
    assert "submit-time stamp" in result.stderr


def test_mismatched_stamp_is_rejected(tmp_path: Path) -> None:
    checkout = tmp_path / "snapshot"
    checkout.mkdir()
    stamp = tmp_path / "source_commit.txt"
    stamp.write_text(f"{OTHER_COMMIT}\n", encoding="utf-8")

    result = _run_guard(tmp_path, checkout, COMMIT, stamp)

    assert result.returncode != 0
    assert "does not match" in result.stderr


def test_missing_metadata_and_missing_stamp_is_rejected(tmp_path: Path) -> None:
    checkout = tmp_path / "snapshot"
    checkout.mkdir()

    result = _run_guard(tmp_path, checkout, COMMIT, tmp_path / "absent.txt")

    assert result.returncode != 0
    assert "Cannot verify source identity" in result.stderr


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required for this check")
def test_git_path_still_verifies_the_repository(tmp_path: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert _run_guard(tmp_path, ROOT, head, with_git=True).returncode == 0
    assert _run_guard(tmp_path, ROOT, OTHER_COMMIT, with_git=True).returncode != 0
