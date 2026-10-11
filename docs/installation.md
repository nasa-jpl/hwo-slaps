# Installation

The core package needs Python 3.11 or newer. The pinned scientific stack supports
Python 3.11 and 3.12. HWO-SLAPS has two layers:

- **The core package** reads and validates configurations and works with saved results.
  It depends only on NumPy, SciPy, PyYAML, Astropy and threadpoolctl.
- **The scientific stack** adds the lensing, optics and inference libraries
  (PyAutoLens, HCIPy, JAX and Nautilus). You need it to run forecasts,
  simulations and nonlinear fits.

Most users want the full scientific environment.

## Get the code

```bash
git clone https://github.com/nasa-jpl/hwo-slaps.git
cd hwo-slaps
```

## Install the scientific environment

The repository includes an installer that creates a conda environment with the
pinned dependency versions from `pyproject.toml` and `constraints/validated.txt`:

```bash
bash install.sh --cpu --env-name hwo-slaps
conda activate hwo-slaps
```

On a Linux machine with an NVIDIA GPU and a CUDA 12 compatible driver, install the
GPU build of JAX, including its CUDA runtime dependencies:

```bash
bash install.sh --gpu --env-name hwo-slaps-gpu
conda activate hwo-slaps-gpu
```

The installer:

- creates the environment if it does not exist, using Python 3.11 by default
  (`--python 3.12` selects Python 3.12, `--prefix` installs into a directory);
- preserves existing environments by default. To install or update dependencies
  in an existing environment, pass `--reuse` and select its Python version with
  `--python` if it differs from the default;
- installs HWO-SLAPS in editable mode with all optional dependencies;
- applies two small performance patches to AutoArray. It first checks that the
  installed AutoArray version and file hashes are the ones the patches were written
  for, and stops if they differ.

Run `bash install.sh --help` for every option.

For example, to update a Python 3.12 environment explicitly:

```bash
bash install.sh --cpu --env-name hwo-slaps --python 3.12 --reuse
```

## Install only the core package

For reading results or validating configurations on a machine without the
scientific stack:

```bash
python -m pip install .
```

## Check the installation

```bash
hwoslaps --help
hwoslaps validate configs/minimal.yaml
```

The second command prints the configuration's name and its digest, a hash of its
scientific content:

```text
minimal: valid, digest 783bbd948ef42e566ae6e711e240f4e7e7e8df5375ec380a7d18917daeff1ffa
```

If you installed the scientific environment, continue with the [quickstart](quickstart.md),
which runs a forecast in a few seconds on a laptop CPU.

## Running on a GPU

The JAX engine runs on whatever devices its process can see. Choose the GPU before
Python starts, and enable double precision:

```bash
export CUDA_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=1
hwoslaps forecast configs/minimal.yaml --engine jax --masses 1e7 1e8 1e9 -o out/minimal_gpu
```

Large configurations, such as the [HWO reference](examples/hwo.md) with its
999 × 999 pixel PSF, are much faster on a GPU. The reference (CPU) engine agrees
with the JAX engine to a few parts per million and is the right choice for small
problems and for testing.
