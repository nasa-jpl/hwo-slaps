"""Source provenance from real Git states and the package that was imported."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pytest
from threadpoolctl import threadpool_limits

from hwoslaps.provenance import capture_provenance, git_revision


def git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), *arguments], check=True, capture_output=True).stdout


@pytest.fixture
def source_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    package = root / "src/pkg"
    package.mkdir(parents=True)
    (package / "module.py").write_text("x = 1\n")
    (root / "tracked.txt").write_text("outside package\n")
    git(root, "add", "-A")
    git(root, "-c", "user.name=George Vassilakis", "-c", "user.email=90641828+GeorgeVassilakis@users.noreply.github.com",
        "commit", "-qm", "Synthetic source for provenance tests")
    return root, package


@pytest.mark.parametrize("state", ["clean", "tracked", "untracked_inside", "untracked_outside", "untracked_directory"])
def test_git_revision_states(source_repo, state):
    root, package = source_repo
    if state == "tracked":
        (root / "tracked.txt").write_text("changed\n")
    elif state == "untracked_inside":
        (package / "new.py").write_bytes(b"untracked bytes\n")
    elif state == "untracked_outside":
        (root / "other.py").write_text("outside\n")
    elif state == "untracked_directory":
        package = root / "installed"
        package.mkdir()
        (package / "module.py").write_text("installed wheel\n")
    record = git_revision(package)
    if state == "untracked_directory":
        assert record is None
        return
    assert record.commit == git(root, "rev-parse", "HEAD").decode().strip()
    changed = state in ("tracked", "untracked_inside")
    assert record.dirty == changed
    if not changed:
        assert record.dirty_paths == () and record.worktree_sha256 is None
    else:
        expected = hashlib.sha256(git(root, "diff", "--binary", "HEAD"))
        if state == "untracked_inside":
            expected.update(b"\0src/pkg/new.py\0untracked bytes\n")
        assert record.worktree_sha256 == expected.hexdigest()
        assert record.dirty_paths == (("tracked.txt",) if state == "tracked" else ("src/pkg/new.py",))


def test_non_repository_is_null_and_failure_inside_tracked_repository_propagates(tmp_path, source_repo):
    assert git_revision(tmp_path) is None
    root, package = source_repo
    branch = git(root, "symbolic-ref", "HEAD").decode().strip()
    (root / ".git" / branch).write_text("0" * 40 + "\n")
    with pytest.raises(subprocess.CalledProcessError):
        git_revision(package)


def test_capture_records_the_imported_package_from_an_unrelated_working_directory(source_repo, monkeypatch):
    import hwoslaps
    root, _ = source_repo
    actual_package = Path(hwoslaps.__file__).resolve().parent
    # Pytest's source-path setting is not inherited by child interpreters.
    monkeypatch.setenv("PYTHONPATH", str(actual_package.parent))
    expected = git(actual_package, "rev-parse", "HEAD").decode().strip()
    program = "from hwoslaps.provenance import capture_provenance; import json; print(json.dumps(capture_provenance(command=['validate'])))"
    output = subprocess.run([sys.executable, "-c", program], cwd=root, check=True, capture_output=True, text=True).stdout
    record = json.loads(output)
    assert Path(record["package_path"]) == actual_package
    assert record["source"]["commit"] == expected
    assert record["hwoslaps_version"] == "1.0.0"
    assert record["command"] == ["validate"]
    assert record["python"] == ".".join(map(str, sys.version_info[:3]))
    assert record["packages"]["numpy"]["version"] == importlib.metadata.version("numpy")
    try:
        autolens_version = importlib.metadata.version("autolens")
    except importlib.metadata.PackageNotFoundError:
        assert record["packages"]["autolens"] is None
    else:
        assert record["packages"]["autolens"]["version"] == autolens_version


def test_capture_reports_live_blas_thread_counts():
    import numpy as np
    np.ones((2, 2)) @ np.ones((2, 2))
    with threadpool_limits(limits=2):
        record = capture_provenance()
    pools = [pool for pool in record["thread_pools"] if pool["user_api"] == "blas"]
    assert pools and all(pool["num_threads"] == 2 for pool in pools)
