#!/usr/bin/env bash
#
# Install HWO-SLAPS and its developer dependencies.
#
# The science stack currently requires GitHub checkouts of PyAutoLens and HCIPy:
# PyAutoLens for the current nonlinear-validation backend, and HCIPy for the
# hexike API that is not available in released packages used by this project.

set -euo pipefail

ENV_NAME="hwo-slaps"
PYTHON_VERSION="3.11"
INSTALL_GPU_JAX=0
UPDATE_GIT_REPOS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT_ROOT="${HWOSLAPS_DEV_ROOT:-$(dirname "$SCRIPT_DIR")}"

PYAUTOLENS_REPO_URL="${PYAUTOLENS_REPO_URL:-https://github.com/PyAutoLabs/PyAutoLens.git}"
HCIPY_REPO_URL="${HCIPY_REPO_URL:-https://github.com/ehpor/hcipy.git}"
PYAUTOLENS_DIR="${PYAUTOLENS_DIR:-$CHECKOUT_ROOT/PyAutoLens}"
HCIPY_DIR="${HCIPY_DIR:-$CHECKOUT_ROOT/hcipy}"

usage() {
    cat <<EOF
Usage: bash install.sh [options]

Options:
  --env-name NAME       Conda environment name. Default: hwo-slaps
  --python VERSION     Python version for new envs. Default: 3.11
  --gpu                Install JAX with CUDA 12 support for NVIDIA GPUs.
  --cpu                Install CPU JAX. Default.
  --checkout-root DIR  Directory for PyAutoLens and HCIPy checkouts.
                       Default: parent directory of this repo.
  --no-pull            Do not pull existing dependency checkouts.
  --help               Show this message.

Environment overrides:
  HWOSLAPS_DEV_ROOT    Default checkout root.
  PYAUTOLENS_REPO_URL  PyAutoLens Git URL.
  HCIPY_REPO_URL       HCIPy Git URL.
  PYAUTOLENS_DIR       Existing or desired PyAutoLens checkout path.
  HCIPY_DIR            Existing or desired HCIPy checkout path.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --gpu)
            INSTALL_GPU_JAX=1
            shift
            ;;
        --cpu)
            INSTALL_GPU_JAX=0
            shift
            ;;
        --checkout-root)
            CHECKOUT_ROOT="$2"
            PYAUTOLENS_DIR="$CHECKOUT_ROOT/PyAutoLens"
            HCIPY_DIR="$CHECKOUT_ROOT/hcipy"
            shift 2
            ;;
        --no-pull)
            UPDATE_GIT_REPOS=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "================================================"
echo "     HWO-SLAPS Developer Installation"
echo "================================================"
echo ""
echo "Environment:       $ENV_NAME"
echo "Python:            $PYTHON_VERSION"
echo "Checkout root:     $CHECKOUT_ROOT"
echo "PyAutoLens dir:    $PYAUTOLENS_DIR"
echo "HCIPy dir:         $HCIPY_DIR"
if [[ "$INSTALL_GPU_JAX" -eq 1 ]]; then
    echo "JAX mode:          CUDA 12 GPU"
else
    echo "JAX mode:          CPU"
fi
echo ""

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found. Install Miniconda or Anaconda first."
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "git not found. Install git first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Found existing conda env '$ENV_NAME'."
else
    echo "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

clone_or_update() {
    local repo_url="$1"
    local target_dir="$2"
    local label="$3"

    if [[ -d "$target_dir/.git" ]]; then
        echo "Found existing $label checkout at $target_dir."
        if [[ "$UPDATE_GIT_REPOS" -eq 1 ]]; then
            echo "Updating $label with git pull --ff-only."
            git -C "$target_dir" pull --ff-only
        else
            echo "Skipping pull for $label."
        fi
    elif [[ -e "$target_dir" ]]; then
        echo "$target_dir exists but is not a git checkout."
        echo "Set ${label}_DIR to a valid checkout or remove the directory."
        exit 1
    else
        echo "Cloning $label from $repo_url to $target_dir."
        mkdir -p "$(dirname "$target_dir")"
        git clone "$repo_url" "$target_dir"
    fi
}

clone_or_update "$PYAUTOLENS_REPO_URL" "$PYAUTOLENS_DIR" "PyAutoLens"
clone_or_update "$HCIPY_REPO_URL" "$HCIPY_DIR" "HCIPy"

echo "Installing base runtime and test dependencies."
python -m pip install \
    numpy \
    scipy \
    matplotlib \
    pyyaml \
    astropy \
    tqdm \
    numba \
    pytest \
    nautilus-sampler

if [[ "$INSTALL_GPU_JAX" -eq 1 ]]; then
    echo "Installing JAX with CUDA 12 support."
    python -m pip install -U "jax[cuda12]"
else
    echo "Installing CPU JAX."
    python -m pip install -U jax
fi

echo "Installing PyAutoLens from editable Git checkout."
python -m pip install -e "$PYAUTOLENS_DIR"

echo "Installing HCIPy from editable Git checkout."
python -m pip install -e "$HCIPY_DIR"

echo "Installing HWO-SLAPS from editable checkout."
python -m pip install -e "$SCRIPT_DIR"

echo ""
echo "Running import and backend checks."
python - <<'PY'
import autolens as al
import autofit as af
import hcipy
import jax
import numpy
import yaml

print("autolens", getattr(al, "__version__", "unknown"), al.__file__)
print("autofit", getattr(af, "__version__", "unknown"), af.__file__)
print("hcipy", getattr(hcipy, "__version__", "unknown"), hcipy.__file__)
print("jax", jax.__version__)
print("jax devices", jax.devices())
print("jax backend", jax.default_backend())

required_hexike = [
    "make_hexike_basis",
    "SegmentedHexikeSurface",
    "make_segment_hexike_surface_from_hex_aperture",
]
missing = [name for name in required_hexike if not hasattr(hcipy, name)]
if missing:
    raise RuntimeError(
        "HCIPy checkout is missing required hexike symbols: "
        + ", ".join(missing)
    )

print("All import checks passed.")
PY

if [[ "$INSTALL_GPU_JAX" -eq 1 ]]; then
    echo ""
    echo "Verifying CUDA JAX backend was selected."
    python - <<'PY'
import jax

backend = jax.default_backend()
if backend != "gpu":
    raise RuntimeError(
        f"Expected JAX GPU backend for --gpu install, got {backend!r}. "
        "Check NVIDIA driver, CUDA compatibility, and JAX CUDA wheel install."
    )
print("CUDA JAX backend verified.")
PY
fi

echo ""
echo "================================================"
echo "     Installation complete"
echo "================================================"
echo ""
echo "Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "Recommended validation checks:"
echo "    python -m pytest -q tests/test_installation.py"
echo "    python -m pytest -q tests/test_nonlinear_dataset_builder.py tests/test_nonlinear_autolens_model_builder_runtime.py tests/test_nonlinear_autolens_runner.py"
