#!/usr/bin/env bash
# Install the validated scientific stack; dependency pins live in pyproject.toml.
set -euo pipefail
TASK_SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TASK_ENV=hwo-slaps
TASK_PREFIX=
TASK_PYTHON=3.12
TASK_GPU=0
TASK_ENV_SELECTED=0
TASK_REUSE=0
usage() {
    cat <<'HELP'
Usage: bash install.sh [--env-name NAME | --prefix DIR] [--python VERSION]
                      [--cpu | --gpu] [--reuse]
The scientific stack requires Python 3.12. Existing environments are
preserved unless --reuse explicitly authorizes dependency installation in them.
Dependency pins are read from pyproject.toml and constraints/validated.txt.
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
        --help|-h) usage; exit 0;;
        *) usage; exit 2;;
    esac
done
[[ "$TASK_PYTHON" =~ ^3\.12(\.[0-9]+)?$ ]] || {
    printf '%s\n' 'The scientific stack requires Python 3.12.' >&2; exit 2;
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
    raise RuntimeError(f"the existing environment runs Python {'.'.join(map(str, sys.version_info[:3]))}, not "
                       f"--python {sys.argv[2]}; select a matching Python 3.12 environment or create a new one")
PY
TASK_EXTRAS=all
if [ "$TASK_GPU" -eq 1 ]; then TASK_EXTRAS=all,cuda12; fi
"$TASK_ENV_PYTHON" -m pip install -c "$TASK_SOURCE/constraints/validated.txt" -e "$TASK_SOURCE[$TASK_EXTRAS]"
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
