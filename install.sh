#!/usr/bin/env bash
# Install the validated scientific stack; dependency pins live in pyproject.toml.
set -euo pipefail
TASK_SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TASK_ENV=hwo-slaps
TASK_PREFIX=
TASK_PYTHON=3.11
TASK_GPU=0
TASK_EDITABLE=
TASK_ENV_SELECTED=0
usage() {
    cat <<'HELP'
Usage: bash install.sh [--env-name NAME | --prefix DIR] [--python VERSION]
                      [--cpu | --gpu] [--editable-backends DIR]
Dependency pins are read from pyproject.toml. Patches are applied only to the
selected environment and must match the packaged original/patched hashes.
HELP
}
while [ $# -gt 0 ]; do
    case "$1" in
        --env-name)
            [ -z "$TASK_PREFIX" ] || { usage; exit 2; }
            TASK_ENV=${2:?--env-name requires NAME}; TASK_ENV_SELECTED=1; shift 2;;
        --prefix)
            [ "$TASK_ENV_SELECTED" -eq 0 ] || { usage; exit 2; }
            TASK_PREFIX=${2:?--prefix requires DIR}; shift 2;;
        --python) TASK_PYTHON=${2:?--python requires VERSION}; shift 2;;
        --gpu) TASK_GPU=1; shift;;
        --cpu) TASK_GPU=0; shift;;
        --editable-backends) TASK_EDITABLE=${2:?--editable-backends requires DIR}; shift 2;;
        --help|-h) usage; exit 0;;
        *) usage; exit 2;;
    esac
done
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ -n "$TASK_PREFIX" ]; then
    if [ ! -d "$TASK_PREFIX/conda-meta" ]; then
        conda create --prefix "$TASK_PREFIX" "python=$TASK_PYTHON" -y
    fi
    conda activate "$TASK_PREFIX"
else
    if ! conda env list --json | python -c 'import json,sys,pathlib; raise SystemExit(not any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$TASK_ENV"; then
        conda create --name "$TASK_ENV" "python=$TASK_PYTHON" -y
    fi
    conda activate "$TASK_ENV"
fi
TASK_EXTRAS=all
if [ "$TASK_GPU" -eq 1 ]; then TASK_EXTRAS=all,cuda12; fi
python -m pip install -e "$TASK_SOURCE[$TASK_EXTRAS]"
if [ -n "$TASK_EDITABLE" ]; then
    python - "$TASK_SOURCE/pyproject.toml" "$TASK_EDITABLE" <<'PY'
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import urlparse
project = tomllib.loads(Path(sys.argv[1]).read_text())["project"]
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)
def repository_url(url):
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    parsed = urlparse(url)
    return parsed.hostname, parsed.path.rstrip("/").removesuffix(".git").lower()
def git(directory, *arguments):
    return subprocess.run(["git", "--no-optional-locks", "-C", str(directory), *arguments],
                          check=True, capture_output=True, text=True).stdout.strip()
for extra, checkout in (("lensing", "PyAutoLens"), ("optics", "hcipy")):
    requirement = next(value for value in project["optional-dependencies"][extra] if "git+" in value)
    match = re.fullmatch(r"\S+ @ git\+(.+)@([0-9a-f]{40})", requirement)
    if match is None:
        raise ValueError(f"expected a pinned Git requirement, got {requirement}")
    url, commit = match.groups()
    directory = root / checkout
    if not directory.exists():
        subprocess.run(["git", "clone", url, str(directory)], check=True)
    else:
        if Path(git(directory, "rev-parse", "--show-toplevel")).resolve() != directory.resolve():
            raise RuntimeError(f"{directory}: expected a standalone dependency checkout")
        origin = git(directory, "remote", "get-url", "origin")
        if repository_url(origin) != repository_url(url):
            raise RuntimeError(f"{directory}: origin {origin!r} differs from the pinned repository {url!r}")
        if git(directory, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError(f"{directory}: refusing to install a dirty dependency checkout")
    subprocess.run(["git", "-C", str(directory), "fetch", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(directory), "checkout", "--detach", commit], check=True)
    if git(directory, "rev-parse", "HEAD") != commit or git(directory, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError(f"{directory}: checkout does not match the clean pinned source")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(directory)], check=True)
PY
fi
python - "$TASK_SOURCE/tools/patches/autoarray-2026.5.14.2" <<'PY'
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
import sys
if importlib.metadata.version("autoarray") != "2026.5.14.2":
    raise RuntimeError("the autoarray patches require version 2026.5.14.2")
spec = importlib.util.find_spec("autoarray")
if spec is None or spec.origin is None:
    raise RuntimeError("autoarray is not installed in the selected environment")
site = Path(spec.origin).resolve().parent.parent
patches = Path(sys.argv[1])
expected = {}
for line in (patches / "SHA256SUMS").read_text().splitlines():
    digest, state, filename = line.split()
    expected.setdefault(filename, {})[state] = digest
for filename, digests in expected.items():
    actual = hashlib.sha256((site / filename).read_bytes()).hexdigest()
    if actual not in digests.values():
        raise RuntimeError(f"{filename}: unexpected SHA-256 {actual}; refusing to patch")
for filename, digests in expected.items():
    target = site / filename
    if hashlib.sha256(target.read_bytes()).hexdigest() == digests["patched"]:
        continue
    diff = patches / (target.stem + ".diff")
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-d", str(site), "-i", str(diff.resolve())], check=True)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != digests["patched"]:
        raise RuntimeError(f"{filename}: patched SHA-256 {actual} differs from the validated patch")
PY
python - "$TASK_GPU" <<'PY'
import sys
import autolens
import autofit
import hcipy
import hwoslaps
import jax
required = ("make_hexike_basis", "SegmentedHexikeSurface", "make_segment_hexike_surface_from_hex_aperture")
missing = [name for name in required if not hasattr(hcipy, name)]
if missing:
    raise RuntimeError(f"HCIPy is missing the validated hexike API: {missing}")
if sys.argv[1] == "1" and jax.default_backend() != "gpu":
    raise RuntimeError("--gpu requires a working CUDA JAX backend")
print("hwoslaps", hwoslaps.__version__, hwoslaps.__file__)
print("jax", jax.__version__, jax.devices())
PY
printf '%s\n' 'Validate with: python tools/run_backend_tests.py tests -q -m "not xtx_gpu and not xtx_multi_gpu"'
