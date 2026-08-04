#!/usr/bin/env python3
"""Regenerate frozen lensing-physics scalar anchors."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy import constants as const

SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from _lensing_physics_helpers import (  # noqa: E402
    Planck15CosmologyAdapter,
    bootstrap_hwoslaps_namespace,
    load_constants_module,
    load_mass_models_module,
    load_master_config,
    load_module,
)


def _nfw_lensing_terms(
    mass_msun: float,
    concentration: float,
    z_lens: float,
    z_source: float,
    cosmology,
    constants_module,
    mass_models,
):
    rs_kpc, rho_s = mass_models.nfw_scale_parameters(
        mass_msun,
        concentration,
        z_lens,
        cosmology,
    )
    D_l_m = float(cosmology.angular_diameter_distance(z_lens).value) * constants_module.MPC_TO_M
    D_s_m = float(cosmology.angular_diameter_distance(z_source).value) * constants_module.MPC_TO_M
    D_ls_m = (
        float(cosmology.angular_diameter_distance_z1z2(z_lens, z_source).value)
        * constants_module.MPC_TO_M
    )
    sigma_crit = (const.c.value**2 / (4.0 * np.pi * const.G.value)) * (D_s_m / (D_l_m * D_ls_m))
    rs_m = rs_kpc * constants_module.KPC_TO_M
    kappa_s = (rho_s * rs_m) / sigma_crit
    scale_radius_arcsec = (rs_m / D_l_m) * constants_module.ARCSEC_PER_RAD
    return {
        "rs_kpc": float(rs_kpc),
        "rho_s_kg_m3": float(rho_s),
        "kappa_s": float(kappa_s),
        "scale_radius_arcsec": float(scale_radius_arcsec),
    }


def _load_lensing_generator_module():
    """Load the lensing generator module without package side effects."""
    bootstrap_hwoslaps_namespace()
    load_module("constants.py", "hwoslaps.constants")
    load_module("lensing/mass_models.py", "hwoslaps.lensing.mass_models")
    load_module("lensing/utils.py", "hwoslaps.lensing.utils")
    return load_module("lensing/generator.py", "hwoslaps.lensing.generator")


def _build_integration_config(model_name: str) -> dict:
    """Return deterministic config for integration summary anchors."""
    anchor_masses = {
        "PointMass": 1.0e8,
        "SIS": 1.0e8,
        "NFW": 1.0e9,
    }
    cfg = copy.deepcopy(load_master_config())
    cfg["run_name"] = f"physics-{model_name.lower()}"
    cfg["global_seed"] = 11
    cfg["lensing"]["grid"]["shape"] = [64, 64]
    cfg["lensing"]["subhalo"]["enabled"] = True
    cfg["lensing"]["subhalo"]["mass"] = anchor_masses[model_name]
    cfg["lensing"]["subhalo"]["model"] = model_name
    cfg["lensing"]["subhalo"]["position"] = {"type": "direct", "centre": [0.08, -0.05]}
    if model_name != "NFW":
        cfg["lensing"]["subhalo"].pop("concentration", None)
    return cfg


def _build_integration_image_summary(generator_module) -> dict:
    """Build integration image summaries for PointMass, SIS, and NFW."""
    summaries = {}
    for model_name in ("PointMass", "SIS", "NFW"):
        cfg = _build_integration_config(model_name)
        lensing_data = generator_module.generate_lensing_system(
            cfg["lensing"],
            full_config=cfg,
        )
        key = "pointmass" if model_name == "PointMass" else model_name.lower()
        summaries[key] = {
            "shape": list(lensing_data.image.shape),
            "total_flux": float(np.sum(lensing_data.image)),
            "peak": float(np.max(lensing_data.image)),
        }
    return summaries


def build_anchor_payload(include_integration: bool = False) -> dict:
    """Build frozen scalar anchors for PM/SIS/NFW physics."""
    mass_models = load_mass_models_module()
    constants_module = load_constants_module()
    cosmology = Planck15CosmologyAdapter()

    inputs = {
        "point_mass": {"mass_msun": 1.0e8, "z_lens": 0.2, "z_source": 2.5},
        "sis": {"mass_msun": 1.0e8, "z_lens": 0.2, "z_source": 2.5},
        "nfw": {
            "mass_msun": 1.0e9,
            "z_lens": 0.2,
            "z_source": 2.5,
            "concentration_model": "moline2017_eq7",
            "x_sub": 1.0,
            "h": cosmology.reduced_h,
        },
    }

    point_mass_theta = mass_models.einstein_radius_point_mass(
        inputs["point_mass"]["mass_msun"],
        inputs["point_mass"]["z_lens"],
        inputs["point_mass"]["z_source"],
        cosmology,
    )

    sis_sigma_v = mass_models.sigma_v_from_m200_sis(
        inputs["sis"]["mass_msun"],
        inputs["sis"]["z_lens"],
        cosmology,
    )
    sis_theta = mass_models.einstein_radius_sis_m200(
        inputs["sis"]["mass_msun"],
        inputs["sis"]["z_lens"],
        inputs["sis"]["z_source"],
        cosmology,
    )

    nfw_concentration = mass_models.concentration_mass_relation(
        inputs["nfw"]["mass_msun"],
        model=inputs["nfw"]["concentration_model"],
        x_sub=inputs["nfw"]["x_sub"],
        h=inputs["nfw"]["h"],
    )
    nfw_terms = _nfw_lensing_terms(
        inputs["nfw"]["mass_msun"],
        nfw_concentration,
        inputs["nfw"]["z_lens"],
        inputs["nfw"]["z_source"],
        cosmology,
        constants_module,
        mass_models,
    )

    integration_image_summary = {
        "pointmass": None,
        "sis": None,
        "nfw": None,
    }
    note = "Integration image summary anchors are optional and unset by default."

    if include_integration:
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        try:
            import autolens  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cannot generate integration anchors because `autolens` is unavailable. "
                "Install/use the full environment and rerun with --include-integration."
            ) from exc

        generator_module = _load_lensing_generator_module()
        integration_image_summary = _build_integration_image_summary(generator_module)
        note = (
            "Integration image summary anchors were generated from deterministic "
            "64x64 runs with global_seed=11."
        )

    return {
        "metadata": {
            "schema_version": 1,
            "cosmology": "Planck15",
            "note": note,
        },
        "inputs": inputs,
        "scalars": {
            "point_mass": {"theta_e_arcsec": float(point_mass_theta)},
            "sis": {
                "sigma_v_km_s": float(sis_sigma_v),
                "theta_e_arcsec": float(sis_theta),
            },
            "nfw": {
                "c200": float(nfw_concentration),
                "rs_kpc": nfw_terms["rs_kpc"],
                "rho_s_kg_m3": nfw_terms["rho_s_kg_m3"],
                "kappa_s": nfw_terms["kappa_s"],
                "scale_radius_arcsec": nfw_terms["scale_radius_arcsec"],
            },
        },
        "integration_image_summary": integration_image_summary,
    }


def main() -> int:
    """Write anchor fixture JSON to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(TESTS_DIR / "fixtures" / "lensing_physics_anchors.json"),
        help="Path to output JSON fixture.",
    )
    parser.add_argument(
        "--include-integration",
        action="store_true",
        help="Generate integration image-summary anchors (requires autolens).",
    )
    args = parser.parse_args()

    payload = build_anchor_payload(include_integration=args.include_integration)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Wrote anchors to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
