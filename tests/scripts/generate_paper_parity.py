#!/usr/bin/env python3
"""Generate paper-parity anchors from the submitted RASTI-26-183 code.

Every expected value under ``tests/fixtures/paper_parity`` comes from the
paper commit 41621de (tag ``rasti-26-183-submitted``). This script never
imports the engine under test: each lane runs in a child process whose
``hwoslaps`` must resolve inside an extracted ``git archive 41621de`` tree.

Run on xtx only, from the root of an engine checkout synced there, then
copy ``tests/fixtures/paper_parity`` back to the checkout::

    ROOT=/data/home/gvassilakis/engine-second-pass-20261005
    PAPER=$ROOT/paper_parity_gen/rasti_41621de   # git archive 41621de | tar -x
    env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
        /data/home/gvassilakis/Software/miniconda3/envs/hwo-slaps/bin/python \\
        tests/scripts/generate_paper_parity.py --paper-tree $PAPER \\
        --out tests/fixtures/paper_parity --work $ROOT/paper_parity_gen/work \\
        --gpu 0

``--work`` holds per-lane scratch results and may be deleted afterwards.
``--kernel-anchor`` instead writes only ``k1_paper_kernel.json`` (the K1 anchor: the paper's HWO
reference truth kernels at 999x999 and 51x51, from ``configs/psf_states/science_hwo35.yaml``),
in one reference-lane child, and touches no other file.

The parent process writes the scene inputs (engine YAML, synthetic source
asset and detector kernel) into ``--out`` and then runs one child per lane
with the environment of the test lane it anchors, so that floating-point
behaviour matches ``tools/run_backend_tests.py``:

``reference``
    ``CUDA_VISIBLE_DEVICES=""``, ``JAX_PLATFORMS=cpu``, ``NUMBA_DISABLE_JIT=1``;
    reference Fisher templates and the NumPy nonlinear likelihood.
``reference_numba``
    As ``reference`` with ``NUMBA_DISABLE_JIT=0``. Diagnostic only: the
    manifest records whether the reference numbers depend on the JIT policy.
``jax_gpu``
    ``CUDA_VISIBLE_DEVICES=<--gpu>``, ``JAX_ENABLE_X64=1``,
    ``NUMBA_DISABLE_JIT=0``; JAX grid templates, the JAX likelihood and the
    fresh-profile half chi-squared objective with its gradient.

Each child mirrors the backend launcher: it works in an empty temporary
directory and pushes the AutoArray configuration before importing AutoLens.
The parent then writes one ``<scene>.npz`` per scene and ``manifest.json``.

The Fisher scenes follow the paper's ladder and PSF-knowledge runners
(``scripts/run_ladder.py`` and ``scripts/run_psf_knowledge_map.py``): one
detector built by ``run_ladder._build_detector`` from a noisy no-subhalo
observation, pointed at each mass by ``run_ladder._point_detector_at_rung``,
reduced by ``FisherDetector.compute_grid_map`` over the full square lattice.
The nonlinear scene follows ``scripts/run_nonlinear_validation.py``: a noisy
injected observation passed to ``run_psf_mismatch_case`` under the
``consistent_sampling_v2`` objective, stopped where the validator would start
its searches, then evaluated at fixed parameter vectors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import yaml

PAPER_COMMIT = "41621de8ca861e425e1caddb7a599d5c1032751f"
PAPER_TAG = "rasti-26-183-submitted"
PRIOR_TABLE = "configs/psf_priors/jwst_wss_drift_v1.yaml"
ENGINE_PRIOR_TABLE = "../../../" + PRIOR_TABLE
"""The shipped prior table, relative to ``tests/fixtures/paper_parity``."""
SOURCE_ASSET = "source_image.npz"
DETECTOR_KERNEL = "detector_kernel.npy"
PIXEL_SCALE = 0.03
OBJECTIVE_VERSION = "consistent_sampling_v2"
LOG10_M200_RANGE = (6.0, 9.7)
"""Freed-fit mass support of the paper validation (design_freeze_v1 fit block)."""

FISHER_FIELDS = ("q_asimov", "fisher_raw", "fisher_profiled", "sigma_amplitude", "degradation")
MISMATCH_FIELDS = (
    "amplitude_hat", "q_mismatch", "z_mismatch",
    "amplitude_spurious", "q_spurious", "z_spurious",
)
GRID_FIELD = {
    "q_asimov": "q_asimov_2d", "fisher_raw": "fisher_raw_2d",
    "fisher_profiled": "fisher_profiled_2d",
    "sigma_amplitude": "sigma_amplitude_profiled_2d", "degradation": "degradation_2d",
    "amplitude_hat": "amplitude_hat_2d", "q_mismatch": "q_mismatch_2d",
    "z_mismatch": "z_mismatch_2d", "amplitude_spurious": "amplitude_spurious_2d",
    "q_spurious": "q_spurious_2d", "z_spurious": "z_spurious_2d",
}
LIKELIHOOD_FIELDS = ("log_likelihood", "figure_of_merit", "chi_squared", "noise_normalization")

PINNED_THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
LANES = ("reference", "reference_numba", "jax_gpu")
STORED_LANES = ("reference", "jax_gpu")
PAPER_ONLY_KEYS = (
    "plotting",
    "modeling.enabled",
    "modeling.detection",
    "modeling.fisher.mode",
    "modeling.fisher.map.engine",
    "modeling.fisher.finite_diff.slope",
    "modeling.fisher.finite_diff.multipole_comp",
    "modeling.fisher.finite_diff.shear_comp",
)
"""Keys the 41621de schema requires that carry no science for these scenes."""


def _lane_environment(lane, gpu):
    common = {name: "1" for name in PINNED_THREADS}
    if lane == "jax_gpu":
        return {**common, "CUDA_VISIBLE_DEVICES": str(gpu), "JAX_ENABLE_X64": "1",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "NUMBA_DISABLE_JIT": "0"}
    jit = "0" if lane == "reference_numba" else "1"
    return {**common, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "NUMBA_DISABLE_JIT": jit}


# ---------------------------------------------------------------------------
# Scenes: science inputs shared verbatim by the engine YAML and the paper run
# ---------------------------------------------------------------------------

def _lensing(light, subhalo):
    return {
        "grid": {"shape": [100, 100], "pixel_scale": PIXEL_SCALE, "over_sample_size": 4},
        "lens_galaxy": {"redshift": 0.2, "mass": {
            "type": "Isothermal", "einstein_radius": 1.0,
            "centre": [0.0, 0.0], "ell_comps": [0.1, 0.0]}},
        "source_galaxy": {"redshift": 0.6, "light": light},
        "subhalo": subhalo,
        "cosmology": "Planck15",
    }


EXPONENTIAL = {"type": "Exponential", "centre": [-0.03, 0.08],
               "ell_comps": [0.14516129, 0.25142673],
               "intensity": 2.0, "effective_radius": 0.11}
IMAGE = {"type": "Image", "asset_path": SOURCE_ASSET, "centre": [-0.03, 0.08],
         "rotation_deg": 30.0, "total_flux": 0.29, "flux_scale": 1.0, "size_scale": 1.0}
NFW = {"enabled": True, "mass": 1.0e8, "model": "NFW",
       "concentration": {"model": "moline2017_eq7", "x_sub": 1.0, "h": None},
       "position": {"type": "angle", "angle": 90.0, "offset_pixels": 0}}
OPTICAL_PSF = {
    "telescope": {"gap_size": 0.006, "segment_point_to_point": 1.65,
                  "pupil_diameter": 7.225765, "num_rings": 2,
                  "focal_length": 144.0, "supersampling_factor": 2},
    "hres_psf": {"num_pix": 128, "wavelength": 5.0e-7, "num_airy": 4,
                 "sampling": 5, "save_highres_psf_npy": False},
    "kernel": {"shape_native": [17, 17]},
    "aberrations": {
        "enable_segment_pistons": False, "enable_segment_tiptilts": False,
        "enable_segment_hexikes": True, "enable_global_zernikes": True,
        "segment_pistons": {}, "segment_tiptilts": {},
        "segment_hexikes": {0: {4: 10.0}, 3: {5: 8.0}, 7: {6: 12.0}},
        "global_zernikes": {4: 5.0, 5: 5.0, 8: 5.0},
    },
}
KERNEL_PSF = {"provider": "kernel", "kernel": {
    "path": DETECTOR_KERNEL, "pixel_scale_arcsec": PIXEL_SCALE, "normalize": True}}
OBSERVATION = {"exposure_time": 900.0, "throughput": 1.0, "detector": {
    "gain": 1.0, "read_noise": 0.2, "dark_current": 0.002, "sky_background": 1.0}}
DELTA_FIT_PSF = {"mode": "delta", "delta": {
    "prior_table": ENGINE_PRIOR_TABLE, "amplitude_rms_nm": 10.0,
    "seed": 20261005, "family": "combined"}}


def _fisher(spacing, half_width, *, psf_nuisance):
    fisher = {
        "snr_threshold": 3.0,
        "include_background_offset": True,
        "finite_diff": {"centre_arcsec": 1.0e-3, "einstein_radius_arcsec": 1.0e-3,
                        "ell_comp": 1.0e-3, "source_intensity_frac": 1.0e-2,
                        "source_reff_frac": 1.0e-2},
        "mask_mode": "all_pixels",
        "include_psf_nuisance": psf_nuisance,
        "compute_psf_mode_scan": False,
        "map": {"type": "grid", "grid": {
            "spacing_arcsec": spacing, "half_width_arcsec": half_width, "annulus": None}},
    }
    if psf_nuisance:
        fisher.update({
            "psf_mode_steps": {"segment_pistons": 1.0, "segment_tiptilts": 0.1,
                               "segment_hexikes": 1.0, "global_zernikes": 1.0},
            "psf_mode_prior_sigmas": {"segment_hexikes": 5.0, "global_zernikes": 5.0},
            "psf_basis": {"segment_hexikes": {"segments": [0, 3], "mode_nolls": [1, 2]},
                          "global_zernikes": {"mode_nolls": [4, 5]}},
        })
    return fisher


def _scene(run_name, *, light, subhalo, psf, fisher, fit_psf=None, global_seed=11):
    modeling = {"fisher": fisher}
    if fit_psf is not None:
        modeling["fit_psf"] = copy.deepcopy(fit_psf)
    return {
        "run_name": run_name,
        "global_seed": global_seed,
        "lensing": _lensing(copy.deepcopy(light), copy.deepcopy(subhalo)),
        "psf": copy.deepcopy(psf),
        "observation": copy.deepcopy(OBSERVATION),
        "modeling": modeling,
    }


def _subhalo(model):
    return {"enabled": True, "mass": 1.0e8, "model": model,
            "position": {"type": "angle", "angle": 90.0, "offset_pixels": 0}}


SCENES = {
    "p1_optical_matched": {
        "science": _scene("paper_parity_p1", light=EXPONENTIAL, subhalo=NFW, psf=OPTICAL_PSF,
                          fisher=_fisher(0.4, 1.2, psf_nuisance=True)),
        "masses_msun": [1.0e7, 1.0e8, 1.0e9],
        "description": "segmented-pupil optical PSF, Exponential source, NFW subhalo, "
                       "all-pixel mask, scalar and PSF-mode nuisance profiling",
    },
    "p2_delta_knowledge_error": {
        "science": _scene("paper_parity_p2", light=EXPONENTIAL, subhalo=NFW, psf=OPTICAL_PSF,
                          fisher=_fisher(0.4, 1.2, psf_nuisance=True), fit_psf=DELTA_FIT_PSF),
        "masses_msun": [1.0e7, 1.0e8, 1.0e9],
        "description": "P1 analysed with a delta PSF knowledge error drawn from the "
                       "shipped JWST WSS drift prior at 10 nm RMS",
    },
    "p3_image_source_kernel": {
        "science": _scene("paper_parity_p3", light=IMAGE, subhalo=NFW, psf=KERNEL_PSF,
                          fisher=_fisher(0.6, 1.2, psf_nuisance=False)),
        "masses_msun": [1.0e8, 1.0e9],
        "description": "synthetic image source and an externally supplied detector kernel",
    },
    "p4_subhalo_sis": {
        "science": _scene("paper_parity_p4_sis", light=EXPONENTIAL, subhalo=_subhalo("SIS"),
                          psf=KERNEL_PSF, fisher=_fisher(0.6, 0.6, psf_nuisance=False)),
        "masses_msun": [1.0e8, 1.0e9],
        "description": "SIS subhalo with the detector kernel",
    },
    "p4_subhalo_pointmass": {
        "science": _scene("paper_parity_p4_pointmass", light=EXPONENTIAL,
                          subhalo=_subhalo("PointMass"), psf=KERNEL_PSF,
                          fisher=_fisher(0.6, 0.6, psf_nuisance=False)),
        "masses_msun": [1.0e8, 1.0e9],
        "description": "point-mass subhalo with the detector kernel",
    },
}

NONLINEAR_SCENE = "n1_nonlinear_likelihood"
NONLINEAR = {
    "scene": "p2_delta_knowledge_error",
    "mass_msun": 1.0e9,
    "position_yx_arcsec": [0.4, -0.8],
    "fit_mode": "freed",
    "dataset_kind": "noisy",
    "background_treatment": "subtract_known",
    "log10_m200_range": list(LOG10_M200_RANGE),
    "perturbation_fraction_of_prior_width": 0.05,
    "description": "noisy injected NFW trial at the P2 maximum-q node, fitted with the "
                   "delta fit kernel; smooth and freed models at truth and perturbed",
}


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------

SOURCE_COMPONENTS = (
    # (amplitude, y0, x0, sigma_major, axis_ratio, position angle in degrees)
    (1.0, 0.000, 0.000, 0.060, 0.55, 30.0),
    (0.6, 0.045, -0.035, 0.030, 0.80, -20.0),
    (0.4, -0.050, 0.040, 0.025, 0.60, 75.0),
)
SOURCE_SHAPE = (48, 48)
SOURCE_PIXEL_SCALE = 0.01
KERNEL_SHAPE = (11, 11)
KERNEL_COMPONENTS = (
    # Same layout as SOURCE_COMPONENTS, in detector pixels.
    (0.85, 0.0, 0.0, 0.9, 0.85, 20.0),
    (0.15, 0.0, 0.0, 2.6, 0.70, -35.0),
)


def _elliptical_gaussians(shape, scale, components):
    ny, nx = shape
    y = (np.arange(ny) - (ny - 1) / 2.0) * scale
    x = (np.arange(nx) - (nx - 1) / 2.0) * scale
    yy, xx = np.meshgrid(y, x, indexing="ij")
    image = np.zeros(shape)
    for amplitude, y0, x0, sigma, axis_ratio, angle in components:
        theta = np.deg2rad(angle)
        u = (xx - x0) * np.cos(theta) + (yy - y0) * np.sin(theta)
        v = -(xx - x0) * np.sin(theta) + (yy - y0) * np.cos(theta)
        image += amplitude * np.exp(-0.5 * ((u / sigma) ** 2 + (v / (axis_ratio * sigma)) ** 2))
    return image


def _engine_input_text(name):
    """Final-schema inputs from the same fixed scientific scene used by the paper lane."""
    science = SCENES[name]["science"]
    lensing = science["lensing"]
    light = lensing["source_galaxy"]["light"]
    subhalo = lensing["subhalo"]
    grid = lensing["grid"]
    mass = lensing["lens_galaxy"]["mass"]
    fisher = science["modeling"]["fisher"]
    mapped = fisher["map"]["grid"]
    detector = science["observation"]["detector"]
    text = f'''run_name: {science["run_name"]}
seed: {science["global_seed"]}
cosmology: {{name: {lensing["cosmology"]}}}
scene:
  grid: {{shape: {grid["shape"]}, pixel_scale_arcsec: {grid["pixel_scale"]}, over_sample_size: {grid["over_sample_size"]}}}
  lens:
    redshift: {lensing["lens_galaxy"]["redshift"]}
    mass:
      mass: {{type: {mass["type"]}, centre: {mass["centre"]}, einstein_radius: {mass["einstein_radius"]}, ell_comps: {mass["ell_comps"]}}}
  source:
    redshift: {lensing["source_galaxy"]["redshift"]}
    light:
'''
    if light["type"] == "Image":
        text += f'''      light: {{type: Image, asset_path: ../{light["asset_path"]}, centre: {light["centre"]},
              rotation_deg: {light["rotation_deg"]}, total_flux: {light["total_flux"]}, flux_scale: {light["flux_scale"]}, size_scale: {light["size_scale"]}}}
'''
    else:
        text += f'''      light: {{type: {light["type"]}, centre: {light["centre"]}, ell_comps: {light["ell_comps"]},
              intensity: {light["intensity"]}, effective_radius: {light["effective_radius"]}}}
'''
    if subhalo["model"] == "NFW":
        concentration = subhalo["concentration"]
        h = "null" if concentration["h"] is None else str(concentration["h"])
        text += f'''  subhalo:
    type: NFW
    concentration: {{kind: {concentration["model"]}, x_sub: {concentration["x_sub"]}, h: {h}}}
    redshift: null
'''
    else:
        text += f'''  subhalo: {{type: {subhalo["model"]}, redshift: null}}
'''
    text += "psf:\n"
    psf = science["psf"]
    if psf.get("provider") == "kernel":
        kernel = psf["kernel"]
        text += f'''  truth: {{kind: kernel, path: ../{kernel["path"]}, pixel_scale_arcsec: {kernel["pixel_scale_arcsec"]}, normalize: true}}
'''
    else:
        telescope = psf["telescope"]
        high = psf["hres_psf"]
        wavelength_nm = high["wavelength"] * 1e9
        diameter = telescope["pupil_diameter"]
        sample = high["wavelength"] / diameter * (180.0 / np.pi * 3600.0) / high["sampling"]
        oversampling = int(np.ceil(grid["pixel_scale"] / sample))
        aberrations = psf["aberrations"]
        text += f'''  truth:
    kind: optical
    pupil: {{kind: hex_segmented, diameter_m: {diameter}, pixels: {high["num_pix"]}, supersampling: {telescope["supersampling_factor"]}, rings: {telescope["num_rings"]},
            segment_point_to_point_m: {telescope["segment_point_to_point"]}, gap_m: {telescope["gap_size"]}}}
    focal_length_m: {telescope["focal_length"]}
    wavelength_nm: {wavelength_nm}
    detector_oversampling: {oversampling}
    kernel_shape: {psf["kernel"]["shape_native"]}
    wavefront:
      segment_hexikes: {aberrations["segment_hexikes"]}
      zernikes: {aberrations["global_zernikes"]}
'''
    if "fit_psf" in science["modeling"]:
        delta = science["modeling"]["fit_psf"]["delta"]
        text += f'''  model:
    kind: knowledge_error
    draw:
      prior: {{packaged: {Path(delta["prior_table"]).stem}}}
      amplitude_rms_nm: {delta["amplitude_rms_nm"]}
      seed: {delta["seed"]}
      family: {delta["family"]}
'''
    else:
        text += "  model: {kind: matched}\n"
    difference = fisher["finite_diff"]
    fixed = "[source.light.light.rotation_deg]" if light["type"] == "Image" else "[]"
    text += f'''instrument:
  detector: {{gain_e_per_adu: {detector["gain"]}, read_noise_e: {detector["read_noise"]}, dark_current_e_per_s: {detector["dark_current"]}}}
observation:
  exposure_time_s: {science["observation"]["exposure_time"]}
  sky: {{rate_e_per_s: {detector["sky_background"]}}}
forecast:
  positions: {{kind: grid, spacing_arcsec: {mapped["spacing_arcsec"]}, half_width_arcsec: {mapped["half_width_arcsec"]}, annulus: null}}
  mask: {{kind: all_pixels}}
  nuisances:
    fixed: {fixed}
    steps: {{position: {difference["centre_arcsec"]}, einstein_radius: {difference["einstein_radius_arcsec"]}, ellipticity: {difference["ell_comp"]}, amplitude: {difference["source_intensity_frac"]}, size: {difference["source_reff_frac"]}}}
    priors: {{}}
    background_offset: true
'''
    if fisher["include_psf_nuisance"]:
        modes = fisher["psf_basis"]
        text += f'''    wavefront:
      modes:
        segment_hexikes: {{segments: {modes["segment_hexikes"]["segments"]}, nolls: {modes["segment_hexikes"]["mode_nolls"]}}}
        zernikes: {{nolls: {modes["global_zernikes"]["mode_nolls"]}}}
      step_nm: {fisher["psf_mode_steps"]["segment_hexikes"]}
      prior_sigma_nm: {fisher["psf_mode_prior_sigmas"]["segment_hexikes"]}
'''
    else:
        text += "    wavefront: null\n"
    return text + "  noise_covariance: null\n"


def write_inputs(out, *, legacy=False):
    """Write the synthetic asset, the detector kernel and the engine YAML."""
    out.mkdir(parents=True, exist_ok=True)
    sb = _elliptical_gaussians(SOURCE_SHAPE, SOURCE_PIXEL_SCALE, SOURCE_COMPONENTS)
    sb = sb / (SOURCE_PIXEL_SCALE**2 * sb.sum())
    metadata = {"format_version": 1, "provenance": {
        "generator": "tests/scripts/generate_paper_parity.py",
        "kind": "sum of elliptical Gaussians",
        "components": [list(component) for component in SOURCE_COMPONENTS],
    }}
    np.savez(out / SOURCE_ASSET, sb=sb, pixel_scale_arcsec=np.float64(SOURCE_PIXEL_SCALE),
             metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    kernel = _elliptical_gaussians(KERNEL_SHAPE, 1.0, KERNEL_COMPONENTS)
    np.save(out / DETECTOR_KERNEL, kernel)
    engine = out / "engine"
    engine.mkdir(exist_ok=True)
    for name in SCENES:
        (engine / f"{name}.yaml").write_text(_engine_input_text(name), encoding="utf-8")
    if not legacy:
        return
    for name, scene in SCENES.items():
        header = (
            f"# Paper-parity scene {name}: {scene['description']}.\n"
            "# Written by tests/scripts/generate_paper_parity.py; the paper-side\n"
            "# configuration used for the expected values is in manifest.json.\n"
        )
        text = yaml.safe_dump(scene["science"], sort_keys=False, default_flow_style=None)
        (out / f"{name}.yaml").write_text(header + text, encoding="utf-8")


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Child lane: paper code only
# ---------------------------------------------------------------------------

def _array_digest(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(repr(array.shape).encode() + array.tobytes()).hexdigest()


def _check_lane_environment(lane, gpu):
    expected = _lane_environment(lane, gpu)
    wrong = {name: os.environ.get(name) for name, value in expected.items()
             if os.environ.get(name) != value}
    if lane == "jax_gpu" and "JAX_PLATFORMS" in os.environ:
        wrong["JAX_PLATFORMS"] = os.environ["JAX_PLATFORMS"]
    for name in ("PYTHONPATH", "HWOSLAPS_FISHER_GRID_WORKERS"):
        if os.environ.get(name):
            wrong[name] = os.environ[name]
    if wrong:
        raise RuntimeError(f"lane {lane} runs with the wrong environment: {wrong}")


def _enter_backend_harness():
    """Mirror tools/run_backend_tests.py: empty working directory, AutoArray config."""
    os.chdir(tempfile.mkdtemp(prefix="paper-parity-"))
    from autoconf import conf

    spec = importlib.util.find_spec("autoarray")
    if spec is None or spec.origin is None:
        raise RuntimeError("the supported PyAutoLabs environment is required")
    conf.instance.push(str(Path(spec.origin).resolve().parent / "config"), keep_first=True)


def _import_paper(paper_tree):
    for entry in (paper_tree / "scripts", paper_tree / "src"):
        sys.path.insert(0, str(entry))
    import hwoslaps

    if not Path(hwoslaps.__file__).resolve().is_relative_to(paper_tree / "src"):
        raise RuntimeError(f"hwoslaps resolved outside the paper tree: {hwoslaps.__file__}")


def _set_path(config, dotted, value):
    keys = dotted.split(".")
    for key in keys[:-1]:
        config = config.setdefault(key, {})
    config[keys[-1]] = value


def paper_config(science, out, paper_tree, *, engine):
    """Return the 41621de-schema configuration for one scene."""
    config = copy.deepcopy(science)
    light = config["lensing"]["source_galaxy"]["light"]
    if "asset_path" in light:
        light["asset_path"] = str((out / light["asset_path"]).resolve())
    if config["psf"].get("provider") == "kernel":
        config["psf"]["kernel"]["path"] = str((out / config["psf"]["kernel"]["path"]).resolve())
    fit_psf = config["modeling"].get("fit_psf")
    if fit_psf is not None and fit_psf["mode"] == "delta":
        fit_psf["delta"]["prior_table"] = str(paper_tree / PRIOR_TABLE)
    for dotted, value in (
        ("plotting", {"enabled": False, "output_dir": tempfile.gettempdir()}),
        ("modeling.enabled", True),
        ("modeling.detection", "fisher"),
        ("modeling.fisher.mode", "map"),
        ("modeling.fisher.map.engine", engine),
        ("modeling.fisher.finite_diff.slope", 1.0e-3),
        ("modeling.fisher.finite_diff.multipole_comp", 1.0e-3),
        ("modeling.fisher.finite_diff.shear_comp", 1.0e-3),
    ):
        _set_path(config, dotted, value)
    return config


def _validate_paper_config(config):
    from hwoslaps.config import validation

    if config["psf"].get("provider") == "kernel":
        # The 41621de schema has no detector-kernel provider; the kernel is
        # handed to the paper detector as a PSF object instead.
        validation.validate_top_level({**config, "psf": {}})
        validation.validate_lensing_config(config["lensing"])
        validation.validate_observation_config(config["observation"])
        validation.validate_modeling_config(config["modeling"])
    else:
        validation.validate_or_raise(config)


def _paper_truth_psf(config):
    if config["psf"].get("provider") == "kernel":
        from hwoslaps.psf.utils import make_pyauto_kernel

        spec = config["psf"]["kernel"]
        kernel = make_pyauto_kernel(values=np.load(spec["path"]),
                                    pixel_scales=float(spec["pixel_scale_arcsec"]),
                                    normalize=bool(spec["normalize"]))
        return types.SimpleNamespace(kernel=kernel, kernel_pixel_scale=float(spec["pixel_scale_arcsec"]),
                                     config=None)
    from hwoslaps.psf import generate_psf_system

    return generate_psf_system(config["psf"], full_config=config)


def _kernel_sha256(kernel):
    from hwoslaps.psf.mismatch import _kernel_sha256 as paper_kernel_sha256
    from hwoslaps.psf.utils import pyauto_kernel_native

    return paper_kernel_sha256(pyauto_kernel_native(kernel))


def _fisher_scene(name, out, paper_tree, lane):
    import run_ladder

    scene = SCENES[name]
    engine = "jax" if lane == "jax_gpu" else "reference"
    config = paper_config(scene["science"], out, paper_tree, engine=engine)
    _validate_paper_config(config)
    start = time.perf_counter()
    psf = _paper_truth_psf(config)
    detector = run_ladder._build_detector(config, psf)
    build_seconds = time.perf_counter() - start
    fields = FISHER_FIELDS + (MISMATCH_FIELDS if detector.mismatch_enabled else ())
    rows = {field: [] for field in fields}
    positions = None
    for mass in scene["masses_msun"]:
        logm = float(np.log10(mass))
        if 10.0**logm != mass:
            raise ValueError(f"mass {mass!r} does not survive the ladder's log10 rung")
        run_ladder._point_detector_at_rung(detector, logm)
        grid = detector.compute_grid_map()
        nodes = np.argwhere(grid.evaluated_mask_2d)
        node_positions = np.column_stack((grid.y_coords[nodes[:, 0]], grid.x_coords[nodes[:, 1]]))
        if positions is not None and not np.array_equal(positions, node_positions):
            raise RuntimeError("grid layout changed between masses")
        positions = node_positions
        for field in fields:
            rows[field].append(np.asarray(getattr(grid, GRID_FIELD[field]))[nodes[:, 0], nodes[:, 1]])
    arrays = {f"{lane}__{field}": np.vstack(values) for field, values in rows.items()}
    arrays["positions_yx"] = positions
    arrays["masses_msun"] = np.asarray(scene["masses_msun"], dtype=float)
    record = {
        "seconds": {"detector_build": build_seconds, "total": time.perf_counter() - start},
        "paper_config": config,
        "truth_kernel_sha256": _kernel_sha256(psf.kernel),
        "fit_kernel_sha256": _kernel_sha256(detector.model_psf_data.kernel),
        "profiled_nuisance_names": list(detector.nuisance_names),
        "nuisance_prior_precision": [float(value) for value in detector.prior_precision_diagonal],
        "pixels_unmasked": int(detector.pixels_unmasked),
        "diagnostic_digests": {
            "mu0_adu": _array_digest(detector.mu0_adu_2d),
            "mu0_model_adu": _array_digest(detector.mu0_model_adu_2d),
            "sigma_adu": _array_digest(detector.sigma_adu_2d),
            "fisher_mask": _array_digest(detector.mask_2d.astype(float)),
            "nuisance_images": {name: _array_digest(image) for name, image
                                in zip(detector.nuisance_names, detector.nuisance_images)},
        },
    }
    if detector.fit_psf_delta is not None:
        delta = detector.fit_psf_delta
        record["fit_psf_delta"] = {key: delta[key] for key in (
            "delta_id", "requested_amplitude_rms_nm", "measured_draw_rms_nm", "family", "seed",
            "prior_table_sha256", "truth_psf_config_hash", "fit_psf_config_hash",
            "truth_kernel_sha256", "fit_kernel_sha256", "draw_aberrations")}
    return arrays, record


class _Captured(Exception):
    """Raised by the capturing validator where the paper would start searching."""


class _CapturingValidator:
    """Stand in for NonlinearMetricValidator up to its first search."""

    def validate_case(self, dataset, dataset_metadata, full_config, trial, **kwargs):
        self.captured = dict(dataset=dataset, metadata=dataset_metadata,
                             full_config=full_config, trial=trial, **kwargs)
        raise _Captured


def _truth_vector(paths, config, mass, position):
    lensing = config["lensing"]
    blocks = {("lens", "mass"): lensing["lens_galaxy"]["mass"],
              ("source", "light"): lensing["source_galaxy"]["light"]}
    values = []
    for path in paths:
        if path[0] != "galaxies":
            raise ValueError(f"unexpected prior path {path}")
        galaxy, component, name = path[1], path[2], path[3]
        if component == "subhalo":
            value = math.log10(mass) if name == "log10_m200" else position[int(path[4][-1])]
        else:
            block = blocks[(galaxy, component)][name]
            value = block[int(path[4][-1])] if len(path) == 5 else block
        values.append(float(value))
    return np.asarray(values)


def _nonlinear_scene(out, paper_tree, lane):
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.modeling.nonlinear.autolens_model_builder import (
        autofit_model_from_spec, smooth_model_spec_from_config, subhalo_model_spec_from_trial,
    )
    from hwoslaps.modeling.nonlinear.autolens_runner import AutoLensFitRunner, NonlinearSearchSettings
    from hwoslaps.modeling.nonlinear.mass_mapping import build_mass_mapping_context
    from hwoslaps.modeling.nonlinear.psf_mismatch import run_psf_mismatch_case
    from hwoslaps.modeling.nonlinear.trial import subhalo_truth_config, trial_from_fisher_map_position
    from hwoslaps.modeling.nonlinear.validator import _validate_fit_psf_dataset
    from hwoslaps.observation import generate_observation
    from hwoslaps.psf import generate_psf_system
    from hwoslaps.psf.utils import pyauto_kernel_native

    spec = NONLINEAR
    start = time.perf_counter()
    config = paper_config(SCENES[spec["scene"]]["science"], out, paper_tree, engine="reference")
    config["nonlinear_rendering"] = {"objective_version": OBJECTIVE_VERSION}
    _validate_paper_config(config)
    mass = float(spec["mass_msun"])
    position = tuple(float(value) for value in spec["position_yx_arcsec"])
    injected = subhalo_truth_config(config, mass, position, enabled=True)
    lensing = generate_lensing_system(injected["lensing"], full_config=injected)
    psf = generate_psf_system(injected["psf"], full_config=injected)
    observation = generate_observation(lensing_data=lensing, psf_data=psf,
                                       observation_config=injected["observation"],
                                       full_config=injected)
    trial = trial_from_fisher_map_position(injected, lensing, mass, position,
                                           fisher_q=None, case_id="paper_parity_n1")
    mass_context = build_mass_mapping_context(injected, log10_m200_range=LOG10_M200_RANGE)
    validator = _CapturingValidator()
    try:
        run_psf_mismatch_case(validator, observation, injected, trial,
                              fit_mode=spec["fit_mode"], dataset_kind=spec["dataset_kind"],
                              background_treatment=spec["background_treatment"],
                              mass_context=mass_context)
    except _Captured:
        pass
    captured = validator.captured
    dataset, metadata, full_config = captured["dataset"], captured["metadata"], captured["full_config"]
    _validate_fit_psf_dataset(full_config, dataset, metadata, captured["psf_case"], False,
                              captured["expected_psf_fit_sha256"])
    specs = {
        "smooth": smooth_model_spec_from_config(full_config, priors_config=None),
        "freed": subhalo_model_spec_from_trial(full_config, trial=trial, priors_config=None,
                                               fit_mode=spec["fit_mode"], mass_context=mass_context),
    }
    use_jax = lane == "jax_gpu"
    with tempfile.TemporaryDirectory() as output_dir:
        runner = AutoLensFitRunner(NonlinearSearchSettings(use_jax=use_jax), output_dir=output_dir)
        analysis = runner.make_analysis(dataset, model_metadata=dict(specs["freed"].metadata))
        arrays = {}
        for role, model_spec in specs.items():
            model = autofit_model_from_spec(model_spec)
            paths = list(model.unique_prior_paths)
            priors = list(model.priors_ordered_by_id)
            lower = np.asarray([prior.lower_limit for prior in priors], dtype=float)
            upper = np.asarray([prior.upper_limit for prior in priors], dtype=float)
            truth = _truth_vector(paths, full_config, mass, position)
            signs = np.where(np.arange(truth.size) % 2 == 0, 1.0, -1.0)
            perturbed = truth + spec["perturbation_fraction_of_prior_width"] * (upper - lower) * signs
            vectors = np.vstack((truth, perturbed))
            if np.any(vectors < lower) or np.any(vectors > upper):
                raise ValueError(f"{role} evaluation vectors leave the prior box")
            normalized = (vectors - lower) / (upper - lower)
            arrays[f"{role}_prior_paths"] = np.asarray([".".join(path) for path in paths])
            arrays[f"{role}_prior_lower"] = lower
            arrays[f"{role}_prior_upper"] = upper
            arrays[f"{role}_vectors"] = vectors
            arrays[f"{role}_vectors_normalized"] = normalized
            fits = [analysis.fit_from(instance=model.instance_from_vector(vector=list(vector)))
                    for vector in vectors]
            arrays[f"{lane}__{role}_log_likelihood_function"] = np.asarray(
                [float(analysis.log_likelihood_function(model.instance_from_vector(vector=list(vector))))
                 for vector in vectors])
            for field in LIKELIHOOD_FIELDS:
                arrays[f"{lane}__{role}_{field}"] = np.asarray([float(getattr(fit, field)) for fit in fits])
            if use_jax:
                from hwoslaps.modeling.nonlinear.fresh_profile import make_jax_objective

                objective, _, _, _ = make_jax_objective(analysis, model, lower, upper)
                values = [objective(z) for z in normalized]
                arrays[f"{lane}__{role}_half_chi2"] = np.asarray([value for value, _ in values])
                arrays[f"{lane}__{role}_half_chi2_gradient"] = np.vstack([gradient for _, gradient in values])
    record = {
        "seconds": {"total": time.perf_counter() - start},
        "paper_config": injected,
        "truth_kernel_sha256": _kernel_sha256(psf.kernel),
        "dataset": {
            "psf_case": captured["psf_case"],
            "psf_fit_sha256": metadata.psf_fit_sha256,
            "mask_name": metadata.mask_name,
            "n_unmasked_pixels": int(metadata.n_unmasked_pixels),
            "objective_version": metadata.objective_version,
            "generation_sub_size": int(metadata.generation_sub_size),
            "data_digest": _array_digest(dataset.data.native),
            "noise_map_digest": _array_digest(dataset.noise_map.native),
            "psf_digest": _array_digest(pyauto_kernel_native(dataset.psf)),
            "mask_digest": _array_digest(np.asarray(dataset.mask).astype(float)),
        },
        "trial": {
            "mass_msun": trial.mass_msun, "position_yx_arcsec": list(trial.position_yx_arcsec),
            "model": trial.model, "profile_class": trial.profile_class,
            "lens_redshift": trial.lens_redshift, "source_redshift": trial.source_redshift,
            "kappa_s": trial.kappa_s, "scale_radius_arcsec": trial.scale_radius_arcsec,
            "concentration": trial.concentration,
        },
        "mass_context_hash": mass_context.context_hash,
    }
    return arrays, record


def _versions():
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "jax", "jaxlib", "autolens", "autoarray", "autogalaxy",
                 "autofit", "autoconf", "hcipy", "astropy", "numba"):
        versions[name] = str(importlib.import_module(name).__version__)
    return versions


K1_FIXTURE = "k1_paper_kernel.json"
K1_STATE = "configs/psf_states/science_hwo35.yaml"
K1_SHAPES = ((999, 999), (51, 51))
"""The paper's HWO reference truth state and the two supports it was used at: the Fisher
ladders (999 x 999, ``scripts/run_ladder.py`` KERNEL_SHAPE_NATIVE) and the nonlinear fits
(51 x 51, the state file's own ``kernel.shape_native``)."""


def kernel_anchor(paper_tree, out, gpu):
    """Write the K1 anchor: the paper truth kernels at both supports, from the 41621de code."""
    from hwoslaps.psf import generate_psf_system
    from hwoslaps.psf.utils import pyauto_kernel_native

    state_path = paper_tree / K1_STATE
    with state_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    kernels = {}
    for shape in K1_SHAPES:
        rung = copy.deepcopy(config)
        rung["psf"]["kernel"]["shape_native"] = list(shape)
        psf = generate_psf_system(rung["psf"], full_config=rung)
        values = np.asarray(pyauto_kernel_native(psf.kernel), dtype=np.float64)
        kernels[f"{shape[0]}x{shape[1]}"] = {
            "shape": list(values.shape), "sha256": _kernel_sha256(psf.kernel),
            "peak": float(values.max()), "sum": float(values.sum()), "total_rms_nm": float(psf.total_rms_nm)}
    record = {
        "description": "K1: the HWO reference truth PSF of RASTI-26-183 (science_hwo35) at the Fisher "
                       "and nonlinear-fit supports, from generate_psf_system of the paper commit",
        "paper_commit": PAPER_COMMIT,
        "paper_tag": PAPER_TAG,
        "generator": "tests/scripts/generate_paper_parity.py --kernel-anchor",
        "digest_format": "sha256 of b'<ny>x<nx>:' followed by the float64 C-order kernel bytes",
        "state": K1_STATE,
        "state_sha256": _file_sha256(state_path),
        "pixel_scale_arcsec": config["lensing"]["grid"]["pixel_scale"],
        "psf": config["psf"],
        "provenance_note": config["provenance_note"],
        "kernels": kernels,
        "versions": _versions(),
        "environment": {name: os.environ.get(name) for name in _lane_environment("reference", gpu)},
    }
    (out / K1_FIXTURE).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run_lane(lane, paper_tree, out, work, gpu, scenes):
    _check_lane_environment(lane, gpu)
    _enter_backend_harness()
    _import_paper(paper_tree)
    if lane == "jax_gpu":
        import jax

        if not jax.config.jax_enable_x64 or jax.default_backend() != "gpu":
            raise RuntimeError("the jax_gpu lane needs a 64-bit CUDA JAX backend")
    lane_dir = work / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    for name in scenes:
        if name == NONLINEAR_SCENE:
            arrays, record = _nonlinear_scene(out, paper_tree, lane)
        else:
            arrays, record = _fisher_scene(name, out, paper_tree, lane)
        record["versions"] = _versions()
        record["environment"] = {name: os.environ.get(name) for name in _lane_environment(lane, gpu)}
        np.savez(lane_dir / f"{name}.npz", **arrays)
        (lane_dir / f"{name}.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
        print(f"[{lane}] {name}: {record['seconds']}", flush=True)


# ---------------------------------------------------------------------------
# Parent: inputs, lanes, assembly
# ---------------------------------------------------------------------------

def _max_relative_difference(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(a), np.abs(b))
    difference = np.abs(a - b)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(scale > 0, difference / scale, 0.0)
    return float(np.max(relative)) if relative.size else 0.0


def assemble(out, work, scenes, invocation):
    manifest = {
        "paper_commit": PAPER_COMMIT,
        "paper_tag": PAPER_TAG,
        "generator": "tests/scripts/generate_paper_parity.py",
        "invocation": invocation,
        "lanes": {lane: _lane_environment(lane, "<gpu>") for lane in LANES},
        "paper_only_keys": list(PAPER_ONLY_KEYS),
        "inputs": {
            SOURCE_ASSET: _file_sha256(out / SOURCE_ASSET),
            DETECTOR_KERNEL: _file_sha256(out / DETECTOR_KERNEL),
            PRIOR_TABLE: None,
        },
        "scenes": {},
    }
    for name in scenes:
        stored = {}
        lane_records = {}
        for lane in LANES:
            npz = work / lane / f"{name}.npz"
            if not npz.exists():
                continue
            with np.load(npz) as data:
                arrays = {key: data[key] for key in data.files}
            lane_records[lane] = json.loads((work / lane / f"{name}.json").read_text())
            if lane in STORED_LANES:
                for key, value in arrays.items():
                    if key in stored and not key.startswith(lane) and not np.array_equal(stored[key], value):
                        raise RuntimeError(f"{name}: lane {lane} disagrees on input {key}")
                    stored[key] = value
            else:
                with np.load(work / "reference" / f"{name}.npz") as baseline:
                    manifest.setdefault("numba_jit_invariance", {})[name] = {
                        key.split("__", 1)[1]: _max_relative_difference(
                            baseline[key.replace(lane, "reference", 1)], value)
                        for key, value in arrays.items() if key.startswith(lane + "__")
                    }
        np.savez_compressed(out / f"{name}.npz", **stored)
        entry = {
            "fixture": f"{name}.npz",
            "lanes": {lane: {key: record[key] for key in ("seconds", "versions", "environment")}
                      for lane, record in lane_records.items()},
        }
        if name in SCENES:
            entry.update(engine_config=f"engine/{name}.yaml", description=SCENES[name]["description"],
                         masses_msun=SCENES[name]["masses_msun"])
        else:
            entry.update(engine_config=f"engine/{NONLINEAR['scene']}.yaml", **NONLINEAR)
        reference = lane_records["reference"]
        for key, value in reference.items():
            if key not in ("seconds", "versions", "environment"):
                entry[key] = value
        if "jax_gpu" in lane_records:
            entry["paper_config_jax_gpu_differs_only_in"] = sorted(
                _diff_paths(reference["paper_config"], lane_records["jax_gpu"]["paper_config"]))
            entry["reference_vs_jax_gpu_max_relative_difference"] = {
                key.split("__", 1)[1]: _max_relative_difference(stored[key], stored[key.replace("reference", "jax_gpu", 1)])
                for key in stored if key.startswith("reference__") and key.replace("reference", "jax_gpu", 1) in stored
            }
        if "fit_psf_delta" in entry:
            manifest["inputs"][PRIOR_TABLE] = entry["fit_psf_delta"]["prior_table_sha256"]
        manifest["scenes"][name] = entry
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")


def _diff_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        paths = []
        for key in sorted(set(a) | set(b), key=str):
            paths += _diff_paths(a.get(key), b.get(key), f"{prefix}{key}.")
        return paths
    return [] if a == b else [prefix.rstrip(".")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-tree", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES) + [NONLINEAR_SCENE])
    parser.add_argument("--lane", choices=LANES, help="internal: run one lane in this process")
    parser.add_argument("--kernel-anchor", action="store_true",
                        help=f"write only {K1_FIXTURE} (the K1 anchor), in the reference lane")
    parser.add_argument("--inputs-only", action="store_true", help="write only synthetic assets and final-schema engine YAML")
    args = parser.parse_args()
    if args.inputs_only:
        if args.lane is not None or args.kernel_anchor:
            parser.error("--inputs-only cannot be combined with --lane or --kernel-anchor")
        write_inputs(args.out.resolve())
        return 0
    if args.paper_tree is None or args.work is None or args.gpu is None:
        parser.error("--paper-tree, --work and --gpu are required for paper execution")
    paper_tree = args.paper_tree.resolve()
    out = args.out.resolve()
    work = args.work.resolve()
    unknown = set(args.scenes) - set(SCENES) - {NONLINEAR_SCENE}
    if unknown:
        raise ValueError(f"unknown scenes {sorted(unknown)}")
    if args.lane is not None and args.kernel_anchor:
        _check_lane_environment(args.lane, args.gpu)
        _enter_backend_harness()
        _import_paper(paper_tree)
        kernel_anchor(paper_tree, out, args.gpu)
        return 0
    if args.lane is not None:
        run_lane(args.lane, paper_tree, out, work, args.gpu, args.scenes)
        return 0
    for name in PINNED_THREADS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"pin {name}=1 before running the generator")
    if (paper_tree / ".git").exists() or not (paper_tree / "src" / "hwoslaps").is_dir():
        raise ValueError("--paper-tree must be an extracted git archive of 41621de")
    base = {key: value for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "HWOSLAPS_FISHER_GRID_WORKERS", "JAX_PLATFORMS",
                           "JAX_ENABLE_X64", "CUDA_VISIBLE_DEVICES", "NUMBA_DISABLE_JIT")}
    if args.kernel_anchor:
        command = [sys.executable, str(Path(__file__).resolve()), "--paper-tree", str(paper_tree),
                   "--out", str(out), "--work", str(work), "--gpu", str(args.gpu),
                   "--lane", "reference", "--kernel-anchor"]
        subprocess.run(command, env={**base, **_lane_environment("reference", args.gpu)}, check=True)
        return 0
    write_inputs(out, legacy=True)
    for lane in LANES:
        command = [sys.executable, str(Path(__file__).resolve()), "--paper-tree", str(paper_tree),
                   "--out", str(out), "--work", str(work), "--gpu", str(args.gpu),
                   "--lane", lane, "--scenes", *args.scenes]
        subprocess.run(command, env={**base, **_lane_environment(lane, args.gpu)}, check=True)
    assemble(out, work, args.scenes, " ".join(sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
