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
TASK_REUSE=0
usage() {
    cat <<'HELP'
Usage: bash install.sh [--env-name NAME | --prefix DIR] [--python VERSION]
                      [--cpu | --gpu] [--reuse] [--editable-backends DIR]
The scientific stack supports Python 3.11 and 3.12. Existing environments are
preserved unless --reuse explicitly authorizes dependency installation in them.
Dependency pins are read from pyproject.toml and constraints/validated.txt.
Patches are applied only to the
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
        --reuse) TASK_REUSE=1; shift;;
        --editable-backends) TASK_EDITABLE=${2:?--editable-backends requires DIR}; shift 2;;
        --help|-h) usage; exit 0;;
        *) usage; exit 2;;
    esac
done
[[ "$TASK_PYTHON" =~ ^3\.(11|12)(\.[0-9]+)?$ ]] || {
    printf '%s\n' 'The scientific stack requires Python 3.11 or 3.12.' >&2; exit 2;
}
command -v conda >/dev/null || { printf '%s\n' 'conda must be available on PATH.' >&2; exit 1; }
TASK_CONDA_BASE=$(conda info --base)
TASK_BASE_PYTHON="$TASK_CONDA_BASE/bin/python"
[ -x "$TASK_BASE_PYTHON" ] || { printf '%s\n' 'The conda base interpreter is unavailable.' >&2; exit 1; }
source "$TASK_CONDA_BASE/etc/profile.d/conda.sh"
# Check command success separately from parsing. A failed lookup must never be
# interpreted as an absent environment, even with no bare python on PATH.
TASK_CONDA_INFO=$(conda info --json)
TASK_CONDA_ENVS=$(conda env list --json)
TASK_PREFIX=$("$TASK_BASE_PYTHON" - "$TASK_PREFIX" "$TASK_ENV" "$TASK_CONDA_INFO" "$TASK_CONDA_ENVS" <<'PY'
import json
from pathlib import Path
import re
import sys

requested, name, info_text, envs_text = sys.argv[1:]
info, listing = json.loads(info_text), json.loads(envs_text)
envs, env_dirs = listing["envs"], info["envs_dirs"]
if (not isinstance(envs, list) or not isinstance(env_dirs, list) or not env_dirs
        or any(not isinstance(path, str) or not path for path in (*envs, *env_dirs))):
    raise RuntimeError("conda returned an invalid environment inventory")
if requested:
    target = Path(requested).expanduser().absolute()
else:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) or name in {"base", "root"}:
        raise ValueError("select a non-base environment name without path separators")
    candidates = {Path(path).absolute() for path in envs if Path(path).name == name}
    # A partially-created or unregistered prefix still must not be replaced.
    candidates.update(Path(directory).absolute() / name for directory in env_dirs
                      if (Path(directory) / name).exists() or (Path(directory) / name).is_symlink())
    if len(candidates) > 1:
        raise RuntimeError("environment name is ambiguous; select it with --prefix")
    target = next(iter(candidates)) if candidates else Path(env_dirs[0]).absolute() / name
if target.resolve() == Path(info["root_prefix"]).resolve():
    raise ValueError("refusing to install the scientific stack in the conda base environment")
print(target)
PY
)
if [ -e "$TASK_PREFIX" ] || [ -L "$TASK_PREFIX" ]; then
    [ "$TASK_REUSE" -eq 1 ] || {
        printf 'Existing prefix preserved: %s. Use --reuse to install into this environment.\n' "$TASK_PREFIX" >&2
        exit 1
    }
    [ -d "$TASK_PREFIX/conda-meta" ] && [ -x "$TASK_PREFIX/bin/python" ] || {
        printf '%s\n' 'The existing prefix is not a usable conda environment.' >&2; exit 1;
    }
else
    [ "$TASK_REUSE" -eq 0 ] || { printf '%s\n' '--reuse requires an existing environment.' >&2; exit 1; }
    conda create --prefix "$TASK_PREFIX" "python=$TASK_PYTHON" -y
fi
conda activate "$TASK_PREFIX"
TASK_ENV_PYTHON="$CONDA_PREFIX/bin/python"
"$TASK_ENV_PYTHON" - "$TASK_PREFIX" "$TASK_PYTHON" <<'PY'
from pathlib import Path
import sys

prefix = Path(sys.argv[1]).resolve(strict=True)
if Path(sys.prefix).resolve(strict=True) != prefix or not Path(sys.executable).resolve(strict=True).is_relative_to(prefix):
    raise RuntimeError("the activated interpreter does not belong to the selected environment")
requested = tuple(map(int, sys.argv[2].split(".")))
if sys.version_info[:len(requested)] != requested:
    raise RuntimeError("the existing environment Python differs from --python; select its version explicitly")
PY
TASK_EXTRAS=all
if [ "$TASK_GPU" -eq 1 ]; then TASK_EXTRAS=all,cuda12; fi
"$TASK_ENV_PYTHON" -m pip install -c "$TASK_SOURCE/constraints/validated.txt" -e "$TASK_SOURCE[$TASK_EXTRAS]"
if [ -n "$TASK_EDITABLE" ]; then
    "$TASK_ENV_PYTHON" - "$TASK_SOURCE/pyproject.toml" "$TASK_EDITABLE" <<'PY'
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
"$TASK_ENV_PYTHON" - "$TASK_SOURCE/tools/patches/autoarray-2026.5.14.2" "$CONDA_PREFIX" <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
prefix = Path(sys.prefix).resolve(strict=True)
if prefix != Path(sys.argv[2]).resolve(strict=True) or not Path(sys.executable).resolve(strict=True).is_relative_to(prefix):
    raise RuntimeError("the patch interpreter does not belong to the selected conda environment")
distributions = [dist for dist in importlib.metadata.distributions()
                 if dist.metadata["Name"].lower().replace("_", "-") == "autoarray"]
if len(distributions) != 1:
    raise RuntimeError("autoarray must have exactly one distribution in the selected environment")
distribution = distributions[0]
if distribution.version != "2026.5.14.2":
    raise RuntimeError("the autoarray patches require version 2026.5.14.2")
site = Path(distribution.locate_file("")).resolve(strict=True)
install_roots = {Path(sysconfig.get_path(name)).resolve(strict=True) for name in ("purelib", "platlib")}
if site not in install_roots or not site.is_relative_to(prefix):
    raise RuntimeError("autoarray distribution metadata is outside the selected environment install root")
direct_url = distribution.read_text("direct_url.json")
if direct_url is not None and json.loads(direct_url).get("dir_info", {}).get("editable", False):
    raise RuntimeError("refusing to patch an editable autoarray checkout")
files = distribution.files
if files is None:
    raise RuntimeError("autoarray distribution has no installed-file ownership record")
owned_files = {str(filename) for filename in files}
metadata_files = [Path(distribution.locate_file(filename)).resolve(strict=True) for filename in files
                  if filename.name in ("METADATA", "PKG-INFO")]
if len(metadata_files) != 1 or not metadata_files[0].is_relative_to(site):
    raise RuntimeError("autoarray distribution metadata has an ambiguous or external origin")
spec = importlib.util.find_spec("autoarray")
package = site / "autoarray"
init = package / "__init__.py"
if (spec is None or spec.origin is None or "autoarray/__init__.py" not in owned_files
        or init.resolve(strict=True) != init
        or Path(spec.origin).resolve(strict=True) != init
        or tuple(Path(location).resolve(strict=True) for location in (spec.submodule_search_locations or ())) != (package,)):
    raise RuntimeError("autoarray module is shadowed or does not belong to the selected environment distribution")
patches = Path(sys.argv[1])
expected = {}
for line in (patches / "SHA256SUMS").read_text().splitlines():
    digest, state, filename = line.split()
    expected.setdefault(filename, {})[state] = digest
targets = {}
for filename, digests in expected.items():
    target = site / filename
    resolved = target.resolve(strict=True)
    if (filename not in owned_files or resolved != target or not resolved.is_relative_to(package)
            or not resolved.is_relative_to(prefix)):
        raise RuntimeError(f"{filename}: patch target is not owned by the selected environment distribution")
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual not in digests.values():
        raise RuntimeError(f"{filename}: unexpected SHA-256 {actual}; refusing to patch")
    targets[filename] = resolved
for filename, digests in expected.items():
    target = targets[filename]
    if hashlib.sha256(target.read_bytes()).hexdigest() == digests["patched"]:
        continue
    diff = patches / (target.stem + ".diff")
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-d", str(site), "-i", str(diff.resolve())], check=True)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != digests["patched"]:
        raise RuntimeError(f"{filename}: patched SHA-256 {actual} differs from the validated patch")
PY
"$TASK_ENV_PYTHON" - "$TASK_GPU" <<'PY'
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
