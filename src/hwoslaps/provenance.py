"""The imported package's source revision and the environment that executed a run."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from threadpoolctl import threadpool_info

RECORDED_DISTRIBUTIONS = ("numpy", "scipy", "astropy", "PyYAML", "threadpoolctl", "hcipy", "autoconf",
                          "autoarray", "autogalaxy", "autolens", "autofit", "nautilus-sampler", "jax",
                          "jaxlib", "matplotlib")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class GitRevision:
    commit: str
    dirty: bool
    dirty_paths: tuple[str, ...]
    worktree_sha256: str | None

    def to_mapping(self):
        return {**asdict(self), "dirty_paths": list(self.dirty_paths)}


def git_revision(directory):
    if shutil.which("git") is None:
        return None
    location = Path(directory).resolve()
    top = subprocess.run(["git", "--no-optional-locks", "-C", str(location), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode:
        return None
    root = Path(top.stdout.strip()).resolve()
    relative = location.relative_to(root).as_posix()
    def git(*args):
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, check=True).stdout
    if not git("ls-files", "-z", "--", relative):
        return None
    commit = git("rev-parse", "HEAD").decode().strip()
    tracked = git("status", "--porcelain=v1", "-z", "--no-renames", "--untracked-files=no")
    tracked_paths = [record[3:].decode() for record in tracked.split(b"\0") if record]
    untracked = sorted(record.decode() for record in git("ls-files", "-z", "--others", "--exclude-standard",
                                                        "--", relative).split(b"\0") if record)
    dirty_paths = tuple(sorted(set(tracked_paths) | set(untracked)))
    digest = None
    if dirty_paths:
        hashed = hashlib.sha256(git("diff", "--binary", "HEAD"))
        for filename in untracked:
            hashed.update(b"\0" + filename.encode() + b"\0" + (root / filename).read_bytes())
        digest = hashed.hexdigest()
    return GitRevision(commit, bool(dirty_paths), dirty_paths, digest)


def distribution_record(name):
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    direct = distribution.read_text("direct_url.json")
    source = None
    if direct is not None:
        record = json.loads(direct)
        source = {"url": record["url"], "editable": record.get("dir_info", {}).get("editable", False),
                  "vcs_commit": record.get("vcs_info", {}).get("commit_id")}
        parsed = urlparse(record["url"])
        if parsed.scheme == "file" and source["vcs_commit"] is None:
            revision = git_revision(unquote(parsed.path))
            source["revision"] = None if revision is None else revision.to_mapping()
    return {"version": distribution.version, "source": source}


def capture_provenance(*, command=None):
    import hwoslaps
    from ._version import __version__
    package = Path(hwoslaps.__file__).resolve().parent
    revision = git_revision(package)
    return {"hwoslaps_version": __version__, "package_path": str(package),
            "source": None if revision is None else revision.to_mapping(),
            "python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": {name: distribution_record(name) for name in RECORDED_DISTRIBUTIONS},
            "thread_pools": [{key: entry.get(key) for key in ("user_api", "internal_api", "num_threads", "version")}
                             for entry in threadpool_info()],
            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "command": None if command is None else list(command)}
