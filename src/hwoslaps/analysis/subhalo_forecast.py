"""Fold Fisher sensitivity maps with CDM and WDM subhalo populations.

The fold evaluates

``mu = integral n(m) A(m) dm``

with masses in solar masses, lens-plane areas in square kiloparsecs, and
projected number densities in ``kpc^-2 Msun^-1``. Positive neighboring
areas are interpolated in log mass and log area; intervals touching zero
area are interpolated linearly in area against log mass.

========================  ==============================
Quantity                  Unit
========================  ==============================
``m``, ``M_hm``, ``m0``   ``Msun``
``A``                     lens-plane ``kpc^2``
``n(m)``                  ``kpc^-2 Msun^-1``
``Sigma_sub``             ``kpc^-2``
angles and grid spacing   arcseconds
``m_WDM``                 ``keV``
========================  ==============================

Notes
-----
O'Riordan, Despali, Vegetti, Lovell & Moline 2023, MNRAS
(arXiv:2211.15679), Eq. 10 use
``m^-alpha1 [1 + (alpha2 M_hm/m)^beta]^gamma`` with
``alpha1=1.9``, ``alpha2=1.1``, ``beta=1.0``, and ``gamma=-0.5`` for
their Lovell 2020 M_max definition. Their Eq. 11 normalizes with
``f_sub`` and projected host mass inside ``2 theta_E`` over
``1e6--1e11 Msun`` following Vegetti & Koopmans 2009b.

The O'Riordan et al. headline 1.43 detections per lens is per unit
``f_sub`` (their Fig. 9 left axis). At ``f_sub`` near ``1e-2`` it is about
0.014 per Euclid lens, or one detection per roughly 64--76 lenses
(their Sec. 4.3). This corrects the factor-100 Item 10 brief erratum.

The projected SHMF follows Gilman et al. 2020 (arXiv:1908.06983):
``d^2N/(dm dA) = (Sigma_sub/m0) (m/m0)^alpha``, with
``m0=1e8 Msun``, ``alpha`` in ``U(-1.95, -1.85)``, default
``alpha=-1.9``, and ``Sigma_sub`` in ``kpc^-2``. Their prior is
``U(0, 0.1)`` and fiducial plot value is 0.012; masses are M200 relative
to the critical density. The optional normalization presets use
``f_sub=5e-3`` for hydro and ``1e-2`` for DMO, from Despali & Vegetti
2017 as quoted by O'Riordan et al. 2023; Hsueh et al. 2020 measure about
``2e-2`` from seven lensed quasars.

The WDM presets are ``lovell20_bound=(4.2, 2.5, -0.2)`` from Lovell
2020, ``lovell14=(1.0, 1.0, -1.3)`` from Lovell et al. 2014 as used by
Gilman et al. 2020 Eq. 11, and
``oriordan23_mmax=(1.1, 1.0, -0.5)`` from O'Riordan et al. 2023 Eq. 10.
The half-mode conversion is Schneider et al. 2012 as given in Gilman
et al. 2020 Eq. 10:
``M_hm=3e8 (m_DM/3.3 keV)^-3.33 Msun``.

For ``statistic='mismatch'``, the folded objects are real injected
subhaloes analyzed under a wrong PSF. This is a completeness channel,
not a false-positive forecast; ``q_spurious_2d`` is deliberately excluded.

This first forecast omits line-of-sight haloes, making a subhalo-only floor
that is conservative on counts (Vegetti et al. 2018). It assumes a
homogeneous canonical-lens sample and omits subhalo mass-measurement errors,
following the O'Riordan et al. 2023 counting precedent. It treats SHMF mass
as the pipeline M200, models only the WDM count suppression, and does not
apply the Bose et al. 2016 WDM concentration suppression to ``A(m)``.
Concentration sensitivity belongs to the L1a sweep. Multi-forecast
aggregation and manifest orchestration remain S1 work.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


_SCHEMA_VERSION = 1

_SUPPRESSION_PRESETS = {
    "lovell20_bound": (4.2, 2.5, -0.2),
    "lovell14": (1.0, 1.0, -1.3),
    "oriordan23_mmax": (1.1, 1.0, -0.5),
}
"""WDM suppression coefficients from Lovell 2020, Lovell et al. 2014,
and O'Riordan et al. 2023, respectively (`dict`)."""

_F_SUB_PRESETS = {
    "hydro_dv17": 5.0e-3,
    "dmo_dv17": 1.0e-2,
}
"""Despali & Vegetti 2017 hydro and DMO substructure fractions (`dict`)."""

_ROBUSTNESS_FIELDS = (
    "shift_dex",
    "masses_shift_plus",
    "masses_shift_minus",
    "mass_range_folded_shift_plus",
    "mass_range_folded_shift_minus",
    "mu_cdm_shift_plus",
    "mu_cdm_shift_minus",
    "mu_per_bin_cdm_shift_plus",
    "mu_per_bin_cdm_shift_minus",
    "mu_wdm_shift_plus",
    "mu_wdm_shift_minus",
    "D_shift_plus",
    "D_shift_minus",
    "N_req_shift_plus",
    "N_req_shift_minus",
    "N_req_ceil_shift_plus",
    "N_req_ceil_shift_minus",
    "N_req_single_bin_shift_plus",
    "N_req_single_bin_shift_minus",
)

_BASE_ARTIFACT_MEMBERS = {
    "schema_version",
    "forecast_id",
    "ladder_masses_msun",
    "detectable_area_arcsec2",
    "detectable_area_kpc2",
    "statistic",
    "detection_q_threshold",
    "sigma_sub_kpc2",
    "normalization_mode",
    "normalization_preset_json",
    "resolved_f_sub",
    "from_f_sub_json",
    "shmf_slope",
    "pivot_mass_msun",
    "suppression",
    "suppression_abc",
    "mhm_grid_msun",
    "m_wdm_kev",
    "mu_cdm",
    "mu_per_bin_cdm",
    "mu_wdm",
    "mu_per_bin_wdm",
    "D_per_lens",
    "N_req",
    "N_req_ceil",
    "N_req_single_bin",
    "mass_range_folded_msun",
    "robustness_present",
    "inputs_verified",
    "map_manifest_json",
    "source_digests_json",
    "revision_provenance_json",
    "config_json",
    "content_digest",
}

_GRID_REQUIRED_MEMBERS = {
    "y_coords",
    "x_coords",
    "spacing_arcsec",
    "centre_yx",
    "detection_q_threshold",
    "evaluated_mask_2d",
    "detectable_mask_2d",
    "q_asimov_2d",
    "z_asimov_2d",
    "fisher_raw_2d",
    "fisher_profiled_2d",
    "sigma_amplitude_profiled_2d",
    "degradation_2d",
    "absorbed_fraction_2d",
    "num_positions_evaluated",
    "num_detectable",
    "detectable_area_arcsec2",
    "max_z_asimov",
    "median_z_asimov",
    "subhalo_mass",
    "subhalo_model",
    "lens_einstein_radius",
}


@dataclass(frozen=True, kw_only=True)
class SubhaloForecastData:
    """Persistable output of one mass-function fold.

    Parameters
    ----------
    schema_version : `int`
        Artifact schema version.
    forecast_id : `str`
        First 16 hexadecimal characters of the canonical input digest.
    ladder_masses_msun : `numpy.ndarray`
        Stored map masses in ladder order.
    detectable_area_arcsec2 : `numpy.ndarray`
        Fold-time detectable areas in square arcseconds.
    detectable_area_kpc2 : `numpy.ndarray`
        Fold-time detectable areas in lens-plane square kiloparsecs.
    statistic : `str`
        Folded statistic, either ``"matched"`` or ``"mismatch"``.
    detection_q_threshold : `float`
        Fold-time q threshold.
    sigma_sub_kpc2 : `float`
        Resolved SHMF normalization.
    normalization_mode : `str`
        Direct or host-fraction normalization mode.
    normalization_preset : `str`, optional
        Host-fraction preset, or `None` for an explicit fraction.
    resolved_f_sub : `float`, optional
        Resolved projected substructure fraction.
    from_f_sub : `dict`, optional
        Full normalized host-fraction block.
    shmf_slope : `float`
        CDM projected SHMF power-law slope.
    pivot_mass_msun : `float`
        SHMF pivot mass.
    suppression : `str`
        WDM suppression preset name.
    suppression_abc : `tuple` of `float`
        Coefficients in ``[1 + (a M_hm/m)^b]^c``.
    mhm_grid_msun : `numpy.ndarray`
        Half-mode mass grid.
    m_wdm_kev : `numpy.ndarray`
        Thermal-relic equivalent grid. A zero half-mode mass maps to NaN.
    mu_cdm : `float`
        Expected CDM detections per lens.
    mu_per_bin_cdm : `numpy.ndarray`
        CDM expectation in consecutive ladder bins.
    mu_wdm : `numpy.ndarray`
        WDM expectation per half-mode mass.
    mu_per_bin_wdm : `numpy.ndarray`
        WDM expectation by ladder bin and half-mode mass.
    D_per_lens : `numpy.ndarray`
        Mass-binned Poisson KL divergence per lens under CDM truth.
    N_req : `numpy.ndarray`
        Lenses required at the configured log-likelihood threshold.
    N_req_ceil : `numpy.ndarray`
        Ceiling of each finite required-lens count.
    N_req_single_bin : `numpy.ndarray`
        Total-count-only required-lens count.
    mass_range_folded_msun : `tuple` of `float`
        Closed ladder mass range used by the fold.
    robustness : `dict`, optional
        Complete plus/minus mass-relabeling forecast, or `None`.
    inputs_verified : `bool`
        Whether every map passed snapshot binding and congruence.
    map_manifest : `list` of `dict`
        Ordered paths, file hashes, q-grid hashes, and stored masses. Each
        q-grid digest hashes ``"{rows}x{cols}:"`` followed by contiguous
        float64 bytes of the array actually folded: ``q_asimov_2d`` for
        matched and ``q_mismatch_2d`` for mismatch.
    source_digests : `dict`
        SHA-256 digests of the analysis and plotting source files.
    revision_provenance : `dict`
        Source-revision state captured at run time.
    config : `dict`
        Canonicalized validated fold configuration.

    Notes
    -----
    The discrimination significance after ``N`` homogeneous lenses is
    ``Z(N) = sqrt(2 N D)``. The mismatch statistic measures completeness
    for real injected subhaloes under a wrong PSF, not false positives.
    """

    schema_version: int = 1
    forecast_id: str
    ladder_masses_msun: np.ndarray
    detectable_area_arcsec2: np.ndarray
    detectable_area_kpc2: np.ndarray
    statistic: str
    detection_q_threshold: float
    sigma_sub_kpc2: float
    normalization_mode: str
    normalization_preset: Optional[str]
    resolved_f_sub: Optional[float]
    from_f_sub: Optional[dict]
    shmf_slope: float
    pivot_mass_msun: float
    suppression: str
    suppression_abc: tuple[float, float, float]
    mhm_grid_msun: np.ndarray
    m_wdm_kev: np.ndarray
    mu_cdm: float
    mu_per_bin_cdm: np.ndarray
    mu_wdm: np.ndarray
    mu_per_bin_wdm: np.ndarray
    D_per_lens: np.ndarray
    N_req: np.ndarray
    N_req_ceil: np.ndarray
    N_req_single_bin: np.ndarray
    mass_range_folded_msun: tuple[float, float]
    robustness: Optional[dict]
    inputs_verified: bool
    map_manifest: list[dict]
    source_digests: dict
    revision_provenance: dict
    config: dict


def _canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by all identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _reject_unknown_keys(mapping: dict, supported: set[str], path: str) -> None:
    """Reject unsupported mapping keys with a path-qualified message."""
    unsupported = sorted(set(mapping) - supported)
    if unsupported:
        raise ValueError(
            f"{path} contains unsupported keys: " + ", ".join(unsupported)
        )


def _require_mapping(value: Any, path: str) -> dict:
    """Require a dictionary value."""
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a dictionary")
    return value


def _require_list(value: Any, path: str) -> list:
    """Require a list or tuple and return a list copy."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list")
    return list(value)


def _finite_number(value: Any, path: str) -> float:
    """Require a finite non-boolean scalar number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _positive_number(value: Any, path: str) -> float:
    """Require a positive finite scalar number."""
    result = _finite_number(value, path)
    if result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _nonnegative_number(value: Any, path: str) -> float:
    """Require a non-negative finite scalar number."""
    result = _finite_number(value, path)
    if result < 0.0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _integer_at_least(value: Any, minimum: int, path: str) -> int:
    """Require a non-boolean integer at or above a lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return int(value)


def _required(mapping: dict, key: str, path: str) -> Any:
    """Return a required mapping value."""
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {path}")
    return mapping[key]


def validate_subhalo_forecast_config(config: dict) -> dict:
    """Validate and canonicalize a standalone subhalo-fold configuration.

    Parameters
    ----------
    config : `dict`
        Mapping whose sole top-level key is ``subhalo_forecast``.

    Returns
    -------
    normalized : `dict`
        Deep normalized copy with floats and lists canonicalized and all
        documented defaults made explicit.

    Raises
    ------
    ValueError
        Raised for a missing, unknown, mistyped, non-finite, or
        out-of-domain value. Error messages include the offending path.
    """
    root = _require_mapping(config, "config")
    _reject_unknown_keys(root, {"subhalo_forecast"}, "config")
    forecast = _require_mapping(
        _required(root, "subhalo_forecast", "config"),
        "subhalo_forecast",
    )
    supported = {
        "maps",
        "statistic",
        "detection_q_threshold",
        "allow_unverified_maps",
        "lens_plane",
        "shmf",
        "wdm",
        "integration",
        "discrimination",
        "robustness",
    }
    _reject_unknown_keys(forecast, supported, "subhalo_forecast")

    maps = _require_list(
        _required(forecast, "maps", "subhalo_forecast"),
        "subhalo_forecast.maps",
    )
    if len(maps) < 3:
        raise ValueError("subhalo_forecast.maps must contain at least 3 entries")
    normalized_maps = []
    for index, raw_map in enumerate(maps):
        path = f"subhalo_forecast.maps[{index}]"
        map_config = _require_mapping(raw_map, path)
        _reject_unknown_keys(map_config, {"path", "mass_msun"}, path)
        map_path = _required(map_config, "path", path)
        if not isinstance(map_path, str):
            raise ValueError(f"{path}.path must be a string")
        normalized_maps.append({
            "path": map_path,
            "mass_msun": _positive_number(
                _required(map_config, "mass_msun", path),
                f"{path}.mass_msun",
            ),
        })

    statistic = forecast.get("statistic", "matched")
    if not isinstance(statistic, str) or statistic not in {"matched", "mismatch"}:
        raise ValueError(
            "subhalo_forecast.statistic must be one of: 'matched', 'mismatch'"
        )
    threshold = _positive_number(
        forecast.get("detection_q_threshold", 10.0),
        "subhalo_forecast.detection_q_threshold",
    )
    allow_unverified = forecast.get("allow_unverified_maps", False)
    if not isinstance(allow_unverified, bool):
        raise ValueError("subhalo_forecast.allow_unverified_maps must be boolean")

    lens_plane = _require_mapping(
        _required(forecast, "lens_plane", "subhalo_forecast"),
        "subhalo_forecast.lens_plane",
    )
    _reject_unknown_keys(
        lens_plane,
        {"lens_redshift", "cosmology"},
        "subhalo_forecast.lens_plane",
    )
    lens_redshift = _positive_number(
        _required(
            lens_plane,
            "lens_redshift",
            "subhalo_forecast.lens_plane",
        ),
        "subhalo_forecast.lens_plane.lens_redshift",
    )
    cosmology = _required(
        lens_plane,
        "cosmology",
        "subhalo_forecast.lens_plane",
    )
    if cosmology != "Planck15":
        raise ValueError(
            "subhalo_forecast.lens_plane.cosmology supports only 'Planck15'"
        )

    shmf = _require_mapping(
        _required(forecast, "shmf", "subhalo_forecast"),
        "subhalo_forecast.shmf",
    )
    _reject_unknown_keys(
        shmf,
        {"slope", "pivot_mass_msun", "normalization"},
        "subhalo_forecast.shmf",
    )
    slope = _finite_number(shmf.get("slope", -1.9), "subhalo_forecast.shmf.slope")
    if not -3.0 < slope < -1.0:
        raise ValueError(
            "subhalo_forecast.shmf.slope must be strictly between -3 and -1"
        )
    pivot = _positive_number(
        shmf.get("pivot_mass_msun", 1.0e8),
        "subhalo_forecast.shmf.pivot_mass_msun",
    )
    normalization = _require_mapping(
        _required(shmf, "normalization", "subhalo_forecast.shmf"),
        "subhalo_forecast.shmf.normalization",
    )
    _reject_unknown_keys(
        normalization,
        {"sigma_sub_kpc2", "from_f_sub"},
        "subhalo_forecast.shmf.normalization",
    )
    modes = [
        key
        for key in ("sigma_sub_kpc2", "from_f_sub")
        if key in normalization
    ]
    if len(modes) != 1:
        raise ValueError(
            "subhalo_forecast.shmf.normalization must contain exactly one "
            "of sigma_sub_kpc2 or from_f_sub"
        )
    normalized_normalization = {}
    if modes[0] == "sigma_sub_kpc2":
        normalized_normalization["sigma_sub_kpc2"] = _positive_number(
            normalization["sigma_sub_kpc2"],
            "subhalo_forecast.shmf.normalization.sigma_sub_kpc2",
        )
    else:
        path = "subhalo_forecast.shmf.normalization.from_f_sub"
        host = _require_mapping(normalization["from_f_sub"], path)
        _reject_unknown_keys(
            host,
            {
                "preset",
                "f_sub",
                "mass_range_msun",
                "aperture_factor",
                "host_slope",
                "source_redshift",
                "einstein_radius_arcsec",
            },
            path,
        )
        choices = [key for key in ("preset", "f_sub") if key in host]
        if len(choices) != 1:
            raise ValueError(f"{path} must contain exactly one of preset or f_sub")
        normalized_host = {}
        if choices[0] == "preset":
            preset = host["preset"]
            if not isinstance(preset, str) or preset not in _F_SUB_PRESETS:
                raise ValueError(
                    f"{path}.preset must be one of: 'hydro_dv17', 'dmo_dv17'"
                )
            normalized_host["preset"] = preset
        else:
            f_sub = _finite_number(host["f_sub"], f"{path}.f_sub")
            if not 0.0 < f_sub < 1.0:
                raise ValueError(f"{path}.f_sub must be strictly between 0 and 1")
            normalized_host["f_sub"] = f_sub
        mass_range = _require_list(
            _required(host, "mass_range_msun", path),
            f"{path}.mass_range_msun",
        )
        if len(mass_range) != 2:
            raise ValueError(f"{path}.mass_range_msun must have length 2")
        mass_range = [
            _positive_number(value, f"{path}.mass_range_msun[{index}]")
            for index, value in enumerate(mass_range)
        ]
        if mass_range[0] >= mass_range[1]:
            raise ValueError(f"{path}.mass_range_msun must be strictly increasing")
        aperture = _positive_number(
            _required(host, "aperture_factor", path),
            f"{path}.aperture_factor",
        )
        host_slope = _finite_number(
            _required(host, "host_slope", path),
            f"{path}.host_slope",
        )
        if not 1.0 < host_slope < 3.0:
            raise ValueError(f"{path}.host_slope must be strictly between 1 and 3")
        source_redshift = _positive_number(
            _required(host, "source_redshift", path),
            f"{path}.source_redshift",
        )
        if source_redshift <= lens_redshift:
            raise ValueError(
                f"{path}.source_redshift must be greater than lens_plane.lens_redshift"
            )
        einstein_radius = _positive_number(
            _required(host, "einstein_radius_arcsec", path),
            f"{path}.einstein_radius_arcsec",
        )
        normalized_host.update({
            "mass_range_msun": mass_range,
            "aperture_factor": aperture,
            "host_slope": host_slope,
            "source_redshift": source_redshift,
            "einstein_radius_arcsec": einstein_radius,
        })
        normalized_normalization["from_f_sub"] = normalized_host

    wdm = _require_mapping(forecast.get("wdm", {}), "subhalo_forecast.wdm")
    _reject_unknown_keys(
        wdm,
        {"suppression", "custom_abc", "half_mode_mass_grid"},
        "subhalo_forecast.wdm",
    )
    suppression = wdm.get("suppression", "lovell20_bound")
    supported_suppressions = set(_SUPPRESSION_PRESETS) | {"custom"}
    if (
        not isinstance(suppression, str)
        or suppression not in supported_suppressions
    ):
        raise ValueError(
            "subhalo_forecast.wdm.suppression must be one of: "
            "'lovell20_bound', 'lovell14', 'oriordan23_mmax', 'custom'"
        )
    if suppression == "custom":
        if "custom_abc" not in wdm:
            raise ValueError(
                "subhalo_forecast.wdm.custom_abc is required for custom suppression"
            )
        custom = _require_list(
            wdm["custom_abc"],
            "subhalo_forecast.wdm.custom_abc",
        )
        if len(custom) != 3:
            raise ValueError("subhalo_forecast.wdm.custom_abc must have length 3")
        abc = [
            _finite_number(value, f"subhalo_forecast.wdm.custom_abc[{index}]")
            for index, value in enumerate(custom)
        ]
        if abc[0] <= 0.0:
            raise ValueError("subhalo_forecast.wdm.custom_abc[0] must be positive")
        if abc[1] <= 0.0:
            raise ValueError("subhalo_forecast.wdm.custom_abc[1] must be positive")
        if abc[2] >= 0.0:
            raise ValueError("subhalo_forecast.wdm.custom_abc[2] must be negative")
    else:
        if "custom_abc" in wdm:
            raise ValueError(
                "subhalo_forecast.wdm.custom_abc is allowed only for custom suppression"
            )
        abc = None
    grid = _require_mapping(
        wdm.get("half_mode_mass_grid", {}),
        "subhalo_forecast.wdm.half_mode_mass_grid",
    )
    _reject_unknown_keys(
        grid,
        {"log10_min_msun", "log10_max_msun", "num"},
        "subhalo_forecast.wdm.half_mode_mass_grid",
    )
    log_min = _finite_number(
        grid.get("log10_min_msun", 6.0),
        "subhalo_forecast.wdm.half_mode_mass_grid.log10_min_msun",
    )
    log_max = _finite_number(
        grid.get("log10_max_msun", 9.0),
        "subhalo_forecast.wdm.half_mode_mass_grid.log10_max_msun",
    )
    if log_min >= log_max:
        raise ValueError(
            "subhalo_forecast.wdm.half_mode_mass_grid requires "
            "log10_min_msun < log10_max_msun"
        )
    num = _integer_at_least(
        grid.get("num", 25),
        2,
        "subhalo_forecast.wdm.half_mode_mass_grid.num",
    )

    integration = _require_mapping(
        forecast.get("integration", {}),
        "subhalo_forecast.integration",
    )
    _reject_unknown_keys(
        integration,
        {"samples_per_bin"},
        "subhalo_forecast.integration",
    )
    samples = _integer_at_least(
        integration.get("samples_per_bin", 128),
        2,
        "subhalo_forecast.integration.samples_per_bin",
    )
    discrimination = _require_mapping(
        forecast.get("discrimination", {}),
        "subhalo_forecast.discrimination",
    )
    _reject_unknown_keys(
        discrimination,
        {"delta_logl_threshold"},
        "subhalo_forecast.discrimination",
    )
    delta_logl = _positive_number(
        discrimination.get("delta_logl_threshold", 5.0),
        "subhalo_forecast.discrimination.delta_logl_threshold",
    )
    robustness = _require_mapping(
        forecast.get("robustness", {}),
        "subhalo_forecast.robustness",
    )
    _reject_unknown_keys(
        robustness,
        {"mass_axis_shift_dex"},
        "subhalo_forecast.robustness",
    )
    shift = _nonnegative_number(
        robustness.get("mass_axis_shift_dex", 0.25),
        "subhalo_forecast.robustness.mass_axis_shift_dex",
    )

    normalized = {
        "subhalo_forecast": {
            "maps": normalized_maps,
            "statistic": statistic,
            "detection_q_threshold": threshold,
            "allow_unverified_maps": allow_unverified,
            "lens_plane": {
                "lens_redshift": lens_redshift,
                "cosmology": cosmology,
            },
            "shmf": {
                "slope": slope,
                "pivot_mass_msun": pivot,
                "normalization": normalized_normalization,
            },
            "wdm": {
                "suppression": suppression,
                **({"custom_abc": abc} if abc is not None else {}),
                "half_mode_mass_grid": {
                    "log10_min_msun": log_min,
                    "log10_max_msun": log_max,
                    "num": num,
                },
            },
            "integration": {"samples_per_bin": samples},
            "discrimination": {"delta_logl_threshold": delta_logl},
            "robustness": {"mass_axis_shift_dex": shift},
        }
    }
    return normalized


def sigma_sub_from_f_sub(
    f_sub,
    mass_range_msun,
    aperture_factor,
    host_slope,
    lens_redshift,
    source_redshift,
    einstein_radius_arcsec,
    cosmology_name,
    slope,
    pivot_mass_msun,
) -> float:
    """Convert a projected substructure fraction to ``Sigma_sub``.

    Parameters
    ----------
    f_sub : `float`
        Projected substructure mass fraction inside the aperture.
    mass_range_msun : sequence of `float`
        Ordered SHMF normalization mass limits.
    aperture_factor : `float`
        Aperture radius in units of the Einstein radius.
    host_slope : `float`
        Circular power-law host slope, strictly between one and three.
    lens_redshift : `float`
        Lens redshift.
    source_redshift : `float`
        Source redshift, greater than the lens redshift.
    einstein_radius_arcsec : `float`
        Host Einstein radius in arcseconds.
    cosmology_name : `str`
        Supported pipeline cosmology name.
    slope : `float`
        Projected SHMF slope.
    pivot_mass_msun : `float`
        SHMF pivot mass.

    Returns
    -------
    sigma_sub : `float`
        Number-density normalization in square-kiloparsec inverse units.

    Notes
    -----
    With ``s2=slope+2``, ``L=ln(m_hi/m_lo)``, and ``x_lo=m_lo/m0``,
    the mass integral per unit normalization is
    ``m0*x_lo**s2*expm1(s2*L)/s2``. At exactly ``s2=0`` its stable limit
    is ``m0*L``.
    """
    f_sub = _finite_number(f_sub, "f_sub")
    if not 0.0 < f_sub < 1.0:
        raise ValueError("f_sub must be strictly between 0 and 1")
    mass_range = _require_list(mass_range_msun, "mass_range_msun")
    if len(mass_range) != 2:
        raise ValueError("mass_range_msun must have length 2")
    mass_lo = _positive_number(mass_range[0], "mass_range_msun[0]")
    mass_hi = _positive_number(mass_range[1], "mass_range_msun[1]")
    if mass_lo >= mass_hi:
        raise ValueError("mass_range_msun must be strictly increasing")
    aperture_factor = _positive_number(aperture_factor, "aperture_factor")
    host_slope = _finite_number(host_slope, "host_slope")
    if not 1.0 < host_slope < 3.0:
        raise ValueError("host_slope must be strictly between 1 and 3")
    lens_redshift = _positive_number(lens_redshift, "lens_redshift")
    source_redshift = _positive_number(source_redshift, "source_redshift")
    if source_redshift <= lens_redshift:
        raise ValueError("source_redshift must be greater than lens_redshift")
    theta_e = _positive_number(
        einstein_radius_arcsec,
        "einstein_radius_arcsec",
    )
    slope = _finite_number(slope, "slope")
    if not -3.0 < slope < -1.0:
        raise ValueError("slope must be strictly between -3 and -1")
    pivot = _positive_number(pivot_mass_msun, "pivot_mass_msun")

    from hwoslaps.lensing.generator import _get_cosmology

    cosmology = _get_cosmology(cosmology_name)
    sigma_crit = cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        redshift_0=lens_redshift,
        redshift_1=source_redshift,
    )
    aperture_radius = aperture_factor*theta_e
    kappa_bar = (theta_e/aperture_radius)**(host_slope - 1.0)
    sigma_host = kappa_bar*float(sigma_crit)
    s2 = slope + 2.0
    log_range = math.log(mass_hi/mass_lo)
    if s2 == 0.0:
        mass_integral = pivot*log_range
    else:
        mass_integral = (
            pivot*(mass_lo/pivot)**s2*math.expm1(s2*log_range)/s2
        )
    return float(f_sub*sigma_host/mass_integral)


def wdm_suppression(masses_msun, half_mode_mass_msun, a, b, c) -> np.ndarray:
    """Evaluate a WDM suppression curve.

    Parameters
    ----------
    masses_msun : array-like
        Positive finite subhalo masses.
    half_mode_mass_msun : `float`
        Finite non-negative half-mode mass. Exactly zero returns exact ones.
    a, b, c : `float`
        Finite suppression coefficients with ``a>0``, ``b>0``, and ``c<0``.

    Returns
    -------
    suppression : `numpy.ndarray`
        ``[1 + (a M_hm/m)^b]^c`` at every input mass.
    """
    masses = np.asarray(masses_msun, dtype=float)
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_msun must contain finite positive values")
    half_mode = _nonnegative_number(
        half_mode_mass_msun,
        "half_mode_mass_msun",
    )
    a = _positive_number(a, "a")
    b = _positive_number(b, "b")
    c = _finite_number(c, "c")
    if c >= 0.0:
        raise ValueError("c must be negative")
    if half_mode == 0.0:
        return np.ones_like(masses, dtype=float)
    return np.power(1.0 + np.power(a*half_mode/masses, b), c)


def half_mode_mass_from_thermal_kev(m_wdm_kev) -> float:
    """Convert a positive thermal-relic mass to half-mode mass.

    Parameters
    ----------
    m_wdm_kev : `float`
        Thermal-relic mass in kiloelectronvolts.

    Returns
    -------
    m_hm_msun : `float`
        Half-mode mass in solar masses.

    Notes
    -----
    The conversion is
    ``M_hm = 3e8 (m_WDM/3.3 keV)^-3.33 Msun``.
    """
    thermal = _positive_number(m_wdm_kev, "m_wdm_kev")
    return float(3.0e8*(thermal/3.3)**-3.33)


def thermal_kev_from_half_mode_mass(m_hm_msun) -> float:
    """Convert a non-negative half-mode mass to thermal-relic mass.

    Parameters
    ----------
    m_hm_msun : `float`
        Half-mode mass in solar masses. Zero reports NaN because the CDM
        limit has no finite thermal-relic equivalent.

    Returns
    -------
    m_wdm_kev : `float`
        Thermal-relic mass in kiloelectronvolts, or NaN at zero half-mode
        mass.

    Notes
    -----
    The inverse conversion is
    ``m_WDM = 3.3 (M_hm/3e8 Msun)^(-1/3.33) keV``.
    """
    half_mode = _nonnegative_number(m_hm_msun, "m_hm_msun")
    if half_mode == 0.0:
        return float("nan")
    return float(3.3*(half_mode/3.0e8)**(-1.0/3.33))


def _q_grid_digest(array: np.ndarray) -> str:
    """Hash a q grid with its shape prefix and contiguous float64 bytes."""
    values = np.ascontiguousarray(array, dtype=np.float64)
    prefix = f"{values.shape[0]}x{values.shape[1]}:".encode("ascii")
    return hashlib.sha256(prefix + values.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the full SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _precheck_grid_members(path: Path, statistic: str) -> None:
    """Check members needed for loud G0 and G3 errors before loading."""
    try:
        with np.load(path, allow_pickle=False) as stored:
            members = set(stored.files)
            missing = sorted(_GRID_REQUIRED_MEMBERS - members)
            if missing:
                gate = "G3" if "q_asimov_2d" in missing else "G0"
                raise ValueError(
                    f"{gate} map {path} missing members: " + ", ".join(missing)
                )
            if statistic == "mismatch":
                for name in (
                    "mismatch_enabled",
                    "q_mismatch_2d",
                    "amplitude_hat_2d",
                ):
                    if name not in members:
                        raise ValueError(f"G3 map {path} missing {name}")
                if not bool(stored["mismatch_enabled"]):
                    raise ValueError(f"G3 map {path} mismatch_enabled is not true")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"G0 could not inspect map {path}: {exc}") from exc


def _validate_grid_map_structure(grid_map: Any, path: Path, statistic: str) -> None:
    """Apply per-map structural gate G0 and statistic gate G3."""
    for name in ("y_coords", "x_coords"):
        values = np.asarray(getattr(grid_map, name))
        if values.ndim != 1:
            raise ValueError(f"G0 map {path} {name} must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"G0 map {path} {name} must be finite")
        differences = np.diff(values)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError(f"G0 map {path} {name} must be strictly monotone")
        if not np.allclose(
            differences,
            grid_map.spacing_arcsec,
            rtol=1.0e-9,
            atol=0.0,
        ):
            raise ValueError(
                f"G0 map {path} coordinate steps must equal spacing_arcsec"
            )
    if not math.isfinite(grid_map.spacing_arcsec) or grid_map.spacing_arcsec <= 0.0:
        raise ValueError(f"G0 map {path} spacing_arcsec must be finite and positive")
    shape = (len(grid_map.y_coords), len(grid_map.x_coords))
    for name, value in vars(grid_map).items():
        if isinstance(value, np.ndarray) and name not in {"y_coords", "x_coords"}:
            if value.ndim != 2 or value.shape != shape:
                raise ValueError(
                    f"G0 map {path} {name} shape must be exactly {shape}"
                )
    mask_names = ["evaluated_mask_2d", "detectable_mask_2d"]
    for name in ("mismatch_detectable_mask_2d", "false_positive_mask_2d"):
        if getattr(grid_map, name) is not None:
            mask_names.append(name)
    for name in mask_names:
        if np.asarray(getattr(grid_map, name)).dtype != np.bool_:
            raise ValueError(f"G0 map {path} {name} must have boolean dtype")
    evaluated = grid_map.evaluated_mask_2d

    def validate_q(name: str) -> None:
        q_values = getattr(grid_map, name)
        if q_values is None:
            raise ValueError(f"G3 map {path} missing {name}")
        if not np.all(np.isfinite(q_values[evaluated])):
            raise ValueError(
                f"G0 map {path} {name} must be finite on evaluated nodes"
            )
        if not np.all(np.isnan(q_values[~evaluated])):
            raise ValueError(
                f"G0 map {path} {name} must be NaN exactly off evaluated nodes"
            )

    validate_q("q_asimov_2d")
    if statistic == "mismatch":
        if not grid_map.mismatch_enabled:
            raise ValueError(f"G3 map {path} mismatch_enabled is not true")
        if grid_map.amplitude_hat_2d is None:
            raise ValueError(f"G3 map {path} missing amplitude_hat_2d")
        validate_q("q_mismatch_2d")
        validate_q("amplitude_hat_2d")


def _byte_identical(first: np.ndarray, second: np.ndarray) -> bool:
    """Return whether two arrays have identical shape, dtype, and bytes."""
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes() == second.tobytes()
    )


def _redacted_snapshot(snapshot: dict, path: Path) -> dict:
    """Redact exactly the three approved runner-ladder fields."""
    redacted = deepcopy(snapshot)
    try:
        redacted.pop("run_name")
        redacted["plotting"].pop("output_dir")
        redacted["lensing"]["subhalo"].pop("mass")
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"G4 snapshot {path} lacks the redaction path {exc}"
        ) from exc
    return redacted


def _close_redshift(left: Any, right: float) -> bool:
    """Return a strict relative-only redshift comparison."""
    try:
        return math.isclose(float(left), right, rel_tol=1.0e-9, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def _validate_snapshots(
    paths: list[Path],
    grid_maps: list[Any],
    fold: dict,
) -> bool:
    """Apply snapshot binding, congruence, and fold-consistency gate G4."""
    from hwoslaps.provenance import config_hash

    snapshot_paths = [path.parent.parent / "config_used.yaml" for path in paths]
    present = [path.is_file() for path in snapshot_paths]
    if any(present) and not all(present):
        raise ValueError("G4 partial snapshot presence across map ladder")
    embedded_present = [grid_map.config_hash is not None for grid_map in grid_maps]
    allow_unverified = fold["allow_unverified_maps"]
    if not any(present):
        if any(embedded_present):
            raise ValueError(
                "G4 embedded config_hash has no adjacent config_used.yaml"
            )
        if not allow_unverified:
            raise ValueError(
                "G4 unverified maps require allow_unverified_maps: true"
            )
        return False

    snapshots = []
    for snapshot_path in snapshot_paths:
        try:
            with snapshot_path.open("r", encoding="utf-8") as stream:
                snapshot = yaml.safe_load(stream)
        except Exception as exc:
            raise ValueError(
                f"G4 could not read snapshot {snapshot_path}: {exc}"
            ) from exc
        if not isinstance(snapshot, dict):
            raise ValueError(f"G4 snapshot {snapshot_path} must contain a mapping")
        snapshots.append(snapshot)

    for path, snapshot_path, grid_map, snapshot in zip(
        paths,
        snapshot_paths,
        grid_maps,
        snapshots,
    ):
        if grid_map.config_hash is None:
            continue
        actual = config_hash(snapshot)
        if grid_map.config_hash != actual:
            raise ValueError(
                f"G4 map {path} embedded config_hash {grid_map.config_hash} "
                f"does not match snapshot {snapshot_path} hash {actual}"
            )

    redacted_hashes = [
        config_hash(_redacted_snapshot(snapshot, path))
        for snapshot, path in zip(snapshots, snapshot_paths)
    ]
    if len(set(redacted_hashes)) != 1:
        details = ", ".join(
            f"{path}={digest}"
            for path, digest in zip(snapshot_paths, redacted_hashes)
        )
        raise ValueError(f"G4 snapshot congruence mismatch: {details}")

    lens_plane = fold["lens_plane"]
    host = fold["shmf"]["normalization"].get("from_f_sub")
    for snapshot_path, snapshot, grid_map in zip(
        snapshot_paths,
        snapshots,
        grid_maps,
    ):
        try:
            lensing = snapshot["lensing"]
            snapshot_cosmology = lensing["cosmology"]
            snapshot_lens_redshift = lensing["lens_galaxy"]["redshift"]
            snapshot_mass = lensing["subhalo"]["mass"]
            map_block = snapshot["modeling"]["fisher"]["map"]
            map_type = map_block["type"]
            snapshot_spacing = map_block["grid"]["spacing_arcsec"]
            snapshot_threshold = map_block["detection_q_threshold"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"G4 snapshot {snapshot_path} missing fold-consistency field {exc}"
            ) from exc
        try:
            mass_matches = math.isclose(
                float(snapshot_mass),
                grid_map.subhalo_mass,
                rel_tol=1.0e-9,
                abs_tol=0.0,
            )
        except (TypeError, ValueError):
            mass_matches = False
        if not mass_matches:
            raise ValueError(
                f"G4 snapshot {snapshot_path} subhalo mass {snapshot_mass!r} "
                f"does not match stored map mass {grid_map.subhalo_mass!r}"
            )
        if map_type != "grid":
            raise ValueError(
                f"G4 snapshot {snapshot_path} Fisher map type must be grid"
            )
        try:
            spacing_matches = math.isclose(
                float(snapshot_spacing),
                grid_map.spacing_arcsec,
                rel_tol=1.0e-9,
                abs_tol=0.0,
            )
        except (TypeError, ValueError):
            spacing_matches = False
        if not spacing_matches:
            raise ValueError(
                f"G4 snapshot {snapshot_path} map spacing_arcsec is "
                "inconsistent with the stored map"
            )
        try:
            threshold_matches = math.isclose(
                float(snapshot_threshold),
                grid_map.detection_q_threshold,
                rel_tol=1.0e-9,
                abs_tol=0.0,
            )
        except (TypeError, ValueError):
            threshold_matches = False
        if not threshold_matches:
            raise ValueError(
                f"G4 snapshot {snapshot_path} detection_q_threshold is "
                "inconsistent with the stored map"
            )
        if snapshot_cosmology != lens_plane["cosmology"]:
            raise ValueError(
                f"G4 snapshot {snapshot_path} cosmology is inconsistent "
                "with lens_plane.cosmology"
            )
        if not _close_redshift(
            snapshot_lens_redshift,
            lens_plane["lens_redshift"],
        ):
            raise ValueError(
                f"G4 snapshot {snapshot_path} lens_redshift is inconsistent "
                "with lens_plane.lens_redshift"
            )
        if host is not None:
            try:
                snapshot_source_redshift = lensing["source_galaxy"]["redshift"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"G4 snapshot {snapshot_path} missing source redshift {exc}"
                ) from exc
            if not _close_redshift(
                snapshot_source_redshift,
                host["source_redshift"],
            ):
                raise ValueError(
                    f"G4 snapshot {snapshot_path} source_redshift is "
                    "inconsistent with from_f_sub.source_redshift"
                )

    verified = (
        all(embedded_present)
        and len({grid_map.git_hash for grid_map in grid_maps}) == 1
    )
    if not verified and not allow_unverified:
        raise ValueError(
            "G4 maps without embedded hashes or from mixed code revisions "
            "require allow_unverified_maps: true; embedded hashes must be "
            "present and the ladder must come from one code revision"
        )
    return verified


def _threshold_area(grid_map: Any, statistic: str, threshold: float, path: Path) -> float:
    """Recompute one fold-time area and apply stored-threshold gate G5."""
    if statistic == "matched":
        mask = grid_map.evaluated_mask_2d & (grid_map.q_asimov_2d >= threshold)
        stored_mask = grid_map.detectable_mask_2d
        stored_area = grid_map.detectable_area_arcsec2
        mask_name = "detectable_mask_2d"
    else:
        mask = (
            grid_map.evaluated_mask_2d
            & (grid_map.amplitude_hat_2d > 0.0)
            & (grid_map.q_mismatch_2d >= threshold)
        )
        stored_mask = grid_map.mismatch_detectable_mask_2d
        stored_area = grid_map.mismatch_detectable_area_arcsec2
        mask_name = "mismatch_detectable_mask_2d"
    area = float(
        np.count_nonzero(mask)*(grid_map.spacing_arcsec*grid_map.spacing_arcsec)
    )
    if threshold == grid_map.detection_q_threshold:
        if stored_mask is None or not np.array_equal(mask, stored_mask):
            raise ValueError(f"G5 map {path} does not reproduce {mask_name}")
        if stored_area is None or area != stored_area:
            raise ValueError(f"G5 map {path} does not reproduce stored area")
    return area


def _load_and_validate_maps(fold: dict) -> tuple:
    """Load a map ladder and apply gates G0 through G5."""
    from hwoslaps.modeling.utils_fisher import load_fisher_grid_map_npz

    paths = [Path(item["path"]).expanduser().resolve() for item in fold["maps"]]
    if len(paths) != len(set(paths)):
        raise ValueError("G1 map resolved paths must be unique")
    for path in paths:
        _precheck_grid_members(path, fold["statistic"])
    file_hashes = [_file_sha256(path) for path in paths]
    if len(file_hashes) != len(set(file_hashes)):
        raise ValueError("G1 per-file sha256 values must be unique")

    grid_maps = []
    for path in paths:
        try:
            grid_map = load_fisher_grid_map_npz(path)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"G0 could not load map {path}: {exc}") from exc
        _validate_grid_map_structure(grid_map, path, fold["statistic"])
        grid_maps.append(grid_map)

    stored_masses = []
    for index, (grid_map, declared) in enumerate(zip(grid_maps, fold["maps"])):
        mass = grid_map.subhalo_mass
        if mass is None or not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"G1 map {paths[index]} stored subhalo_mass is invalid")
        if not math.isclose(
            declared["mass_msun"],
            mass,
            rel_tol=1.0e-6,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"G1 map {paths[index]} declared mass does not match stored mass"
            )
        stored_masses.append(float(mass))
    if not np.all(np.diff(stored_masses) > 0.0):
        raise ValueError("G1 stored subhalo masses must be strictly increasing")

    reference = grid_maps[0]
    for path, grid_map in zip(paths[1:], grid_maps[1:]):
        for name in ("x_coords", "y_coords", "evaluated_mask_2d"):
            if not _byte_identical(getattr(reference, name), getattr(grid_map, name)):
                raise ValueError(f"G2 map {path} {name} is not byte-identical")
        for name in (
            "spacing_arcsec",
            "centre_yx",
            "subhalo_model",
            "lens_einstein_radius",
            "source_image_asset_path",
            "source_image_asset_sha256_16",
        ):
            if getattr(reference, name) != getattr(grid_map, name):
                raise ValueError(f"G2 map {path} {name} is incompatible")
    host = fold["shmf"]["normalization"].get("from_f_sub")
    if host is not None:
        for path, grid_map in zip(paths, grid_maps):
            if grid_map.lens_einstein_radius is None or not math.isclose(
                host["einstein_radius_arcsec"],
                grid_map.lens_einstein_radius,
                rel_tol=1.0e-6,
                abs_tol=0.0,
            ):
                raise ValueError(
                    f"G2 map {path} from_f_sub.einstein_radius_arcsec "
                    "does not match lens_einstein_radius"
                )

    inputs_verified = _validate_snapshots(paths, grid_maps, fold)
    areas = np.asarray([
        _threshold_area(
            grid_map,
            fold["statistic"],
            fold["detection_q_threshold"],
            path,
        )
        for path, grid_map in zip(paths, grid_maps)
    ])
    q_arrays = [
        grid_map.q_asimov_2d
        if fold["statistic"] == "matched"
        else grid_map.q_mismatch_2d
        for grid_map in grid_maps
    ]
    manifest = [
        {
            "path": str(path),
            "sha256": file_hash,
            "q_grid_digest": _q_grid_digest(q_array),
            "stored_mass_msun": mass,
        }
        for path, file_hash, q_array, mass in zip(
            paths,
            file_hashes,
            q_arrays,
            stored_masses,
        )
    ]
    return (
        np.asarray(stored_masses),
        areas,
        manifest,
        inputs_verified,
    )


def _interpolate_area(
    masses_msun: np.ndarray,
    mass_lo: float,
    mass_hi: float,
    area_lo: float,
    area_hi: float,
) -> np.ndarray:
    """Evaluate the specified hybrid area interpolant on one interval.

    Positive endpoints are linear in ``(log10(m), log10(A))``. An interval
    with either endpoint equal to zero is linear in ``(log10(m), A)``.
    """
    masses = np.asarray(masses_msun, dtype=float)
    fraction = (
        (np.log10(masses) - math.log10(mass_lo))
        / (math.log10(mass_hi) - math.log10(mass_lo))
    )
    if area_lo > 0.0 and area_hi > 0.0:
        log_area = math.log10(area_lo) + fraction*(
            math.log10(area_hi) - math.log10(area_lo)
        )
        return np.power(10.0, log_area)
    return area_lo + fraction*(area_hi - area_lo)


def _trapezoid_uniform(values: np.ndarray, width: float) -> np.ndarray:
    """Integrate values sampled at uniform inclusive nodes."""
    return width*(
        0.5*values[0]
        + np.sum(values[1:-1], axis=0)
        + 0.5*values[-1]
    )/(values.shape[0] - 1)


def _integrate_mass_bins(
    masses_msun: np.ndarray,
    areas_kpc2: np.ndarray,
    sigma_sub_kpc2: float,
    slope: float,
    pivot_mass_msun: float,
    mhm_grid_msun: np.ndarray,
    suppression_abc: tuple[float, float, float],
    samples_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate CDM and WDM expectations over consecutive ladder bins.

    The integration variable is ``x=log10(m)``. At each inclusive uniform
    node the integrand is ``ln(10) m n(m) A(m)`` and each bin is integrated
    by the deterministic trapezoid rule.
    """
    masses = np.asarray(masses_msun, dtype=float)
    areas = np.asarray(areas_kpc2, dtype=float)
    mhm_grid = np.asarray(mhm_grid_msun, dtype=float)
    cdm_bins = np.empty(len(masses) - 1, dtype=float)
    wdm_bins = np.empty((len(masses) - 1, len(mhm_grid)), dtype=float)
    a, b, c = suppression_abc
    for index in range(len(masses) - 1):
        x_lo = math.log10(masses[index])
        x_hi = math.log10(masses[index + 1])
        x_nodes = np.linspace(x_lo, x_hi, samples_per_bin)
        mass_nodes = np.power(10.0, x_nodes)
        area_nodes = _interpolate_area(
            mass_nodes,
            masses[index],
            masses[index + 1],
            areas[index],
            areas[index + 1],
        )
        number_density = (
            sigma_sub_kpc2/pivot_mass_msun
            * np.power(mass_nodes/pivot_mass_msun, slope)
        )
        cdm_integrand = (
            math.log(10.0)*mass_nodes*number_density*area_nodes
        )
        cdm_bins[index] = _trapezoid_uniform(cdm_integrand, x_hi - x_lo)
        for mhm_index, half_mode_mass in enumerate(mhm_grid):
            suppression = wdm_suppression(
                mass_nodes,
                half_mode_mass,
                a,
                b,
                c,
            )
            wdm_bins[index, mhm_index] = _trapezoid_uniform(
                cdm_integrand*suppression,
                x_hi - x_lo,
            )
    return cdm_bins, wdm_bins


def _n_req_from_divergence(divergence: np.ndarray, threshold: float) -> np.ndarray:
    """Convert non-negative per-lens divergence to required-lens counts."""
    divergence = np.asarray(divergence, dtype=float)
    result = np.full(divergence.shape, np.inf, dtype=float)
    positive = divergence > 0.0
    result[positive] = threshold/divergence[positive]
    return result


def _ceil_finite(values: np.ndarray) -> np.ndarray:
    """Ceil finite values while preserving infinities."""
    result = np.array(values, copy=True, dtype=float)
    finite = np.isfinite(result)
    result[finite] = np.ceil(result[finite])
    return result


def _poisson_discrimination(
    cdm_bins: np.ndarray,
    wdm_bins: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mass-binned and total-count Poisson discrimination.

    The per-lens CDM-truth divergence is evaluated in the stable form
    ``sum(mu_cdm * (delta - log1p(delta)))``, where
    ``delta=mu_wdm/mu_cdm-1``, and ``N_req=threshold/D``. Bins with zero CDM
    expectation are handled by an explicit mask and contribute
    ``mu_wdm-mu_cdm``. No ``numpy.where`` evaluates the undefined logarithm.
    Merging mass bins cannot increase the Poisson KL, so
    ``N_req_single_bin >= N_req``.
    """
    cdm = np.asarray(cdm_bins, dtype=float)
    wdm = np.asarray(wdm_bins, dtype=float)
    terms = wdm - cdm[:, None]
    for index in np.flatnonzero(cdm > 0.0):
        positive_wdm = wdm[index] > 0.0
        delta = wdm[index, positive_wdm]/cdm[index] - 1.0
        terms[index, positive_wdm] = cdm[index]*np.maximum(
            delta - np.log1p(delta),
            0.0,
        )
        terms[index, ~positive_wdm] = np.inf
    divergence = np.maximum(np.sum(terms, axis=0), 0.0)
    n_req = _n_req_from_divergence(divergence, threshold)
    n_req_ceil = _ceil_finite(n_req)

    cdm_total = np.asarray([np.sum(cdm)])
    wdm_total = np.sum(wdm, axis=0, keepdims=True)
    total_terms = wdm_total - cdm_total[:, None]
    if cdm_total[0] > 0.0:
        positive_wdm = wdm_total[0] > 0.0
        delta = wdm_total[0, positive_wdm]/cdm_total[0] - 1.0
        total_terms[0, positive_wdm] = cdm_total[0]*np.maximum(
            delta - np.log1p(delta),
            0.0,
        )
        total_terms[0, ~positive_wdm] = np.inf
    total_divergence = np.maximum(total_terms[0], 0.0)
    n_req_single = _n_req_from_divergence(total_divergence, threshold)
    return divergence, n_req, n_req_ceil, n_req_single


def _compute_fold(
    masses_msun: np.ndarray,
    areas_kpc2: np.ndarray,
    sigma_sub_kpc2: float,
    slope: float,
    pivot_mass_msun: float,
    mhm_grid_msun: np.ndarray,
    suppression_abc: tuple[float, float, float],
    samples_per_bin: int,
    delta_logl_threshold: float,
) -> dict:
    """Compute the complete fold for one mass labeling."""
    cdm_bins, wdm_bins = _integrate_mass_bins(
        masses_msun,
        areas_kpc2,
        sigma_sub_kpc2,
        slope,
        pivot_mass_msun,
        mhm_grid_msun,
        suppression_abc,
        samples_per_bin,
    )
    divergence, n_req, n_req_ceil, n_req_single = _poisson_discrimination(
        cdm_bins,
        wdm_bins,
        delta_logl_threshold,
    )
    return {
        "mu_cdm": float(np.sum(cdm_bins)),
        "mu_per_bin_cdm": cdm_bins,
        "mu_wdm": np.sum(wdm_bins, axis=0),
        "mu_per_bin_wdm": wdm_bins,
        "D_per_lens": divergence,
        "N_req": n_req,
        "N_req_ceil": n_req_ceil,
        "N_req_single_bin": n_req_single,
    }


def _identity_config(config: dict) -> dict:
    """Remove map paths from the forecast-identity configuration copy."""
    identity = deepcopy(config)
    for item in identity["subhalo_forecast"]["maps"]:
        item.pop("path")
    return identity


def _forecast_id(
    schema_version: int,
    config: dict,
    manifest: list[dict],
    statistic: str,
    detection_q_threshold: float,
) -> str:
    """Hash the canonical relocatable forecast inputs."""
    identity_manifest = [
        {
            "sha256": item["sha256"],
            "q_grid_digest": item["q_grid_digest"],
            "stored_mass_msun": item["stored_mass_msun"],
        }
        for item in manifest
    ]
    payload = {
        "schema_version": schema_version,
        "config": _identity_config(config),
        "map_manifest": identity_manifest,
        "statistic": statistic,
        "detection_q_threshold": detection_q_threshold,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _source_digests() -> dict:
    """Hash analysis and plotting source files from runtime file locations."""
    analysis_path = Path(__file__).resolve()
    plotting_path = analysis_path.parents[1] / "plotting" / "subhalo_forecast.py"
    return {
        "analysis/subhalo_forecast.py": _file_sha256(analysis_path),
        "plotting/subhalo_forecast.py": _file_sha256(plotting_path),
    }


def run_subhalo_forecast(config: dict) -> SubhaloForecastData:
    """Fold a validated Fisher map ladder with CDM and WDM SHMFs.

    Parameters
    ----------
    config : `dict`
        Standalone fold configuration. It is validated and canonicalized on
        every call.

    Returns
    -------
    data : `SubhaloForecastData`
        Deterministic expectations, discrimination curves, input manifest,
        and provenance.

    Raises
    ------
    ValueError
        Raised by schema validation or gates G0--G5.
    """
    normalized = validate_subhalo_forecast_config(config)
    fold = normalized["subhalo_forecast"]
    masses, areas_arcsec2, manifest, inputs_verified = _load_and_validate_maps(fold)

    from hwoslaps.lensing.generator import _get_cosmology

    cosmology = _get_cosmology(fold["lens_plane"]["cosmology"])
    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(
        redshift=fold["lens_plane"]["lens_redshift"]
    )
    areas_kpc2 = areas_arcsec2*float(kpc_per_arcsec)**2

    shmf = fold["shmf"]
    normalization = shmf["normalization"]
    host = normalization.get("from_f_sub")
    if host is None:
        sigma_sub = normalization["sigma_sub_kpc2"]
        normalization_mode = "sigma_sub_kpc2"
        normalization_preset = None
        resolved_f_sub = None
    else:
        normalization_mode = "from_f_sub"
        normalization_preset = host.get("preset")
        resolved_f_sub = (
            _F_SUB_PRESETS[normalization_preset]
            if normalization_preset is not None
            else host["f_sub"]
        )
        sigma_sub = sigma_sub_from_f_sub(
            resolved_f_sub,
            host["mass_range_msun"],
            host["aperture_factor"],
            host["host_slope"],
            fold["lens_plane"]["lens_redshift"],
            host["source_redshift"],
            host["einstein_radius_arcsec"],
            fold["lens_plane"]["cosmology"],
            shmf["slope"],
            shmf["pivot_mass_msun"],
        )

    wdm = fold["wdm"]
    suppression = wdm["suppression"]
    suppression_abc = tuple(
        wdm["custom_abc"]
        if suppression == "custom"
        else _SUPPRESSION_PRESETS[suppression]
    )
    mhm_spec = wdm["half_mode_mass_grid"]
    mhm_grid = np.logspace(
        mhm_spec["log10_min_msun"],
        mhm_spec["log10_max_msun"],
        mhm_spec["num"],
    )
    thermal_grid = 3.3*np.power(mhm_grid/3.0e8, -1.0/3.33)
    result = _compute_fold(
        masses,
        areas_kpc2,
        sigma_sub,
        shmf["slope"],
        shmf["pivot_mass_msun"],
        mhm_grid,
        suppression_abc,
        fold["integration"]["samples_per_bin"],
        fold["discrimination"]["delta_logl_threshold"],
    )

    robustness = None
    shift = fold["robustness"]["mass_axis_shift_dex"]
    if shift > 0.0:
        masses_plus = masses*10.0**shift
        masses_minus = masses*10.0**-shift
        plus = _compute_fold(
            masses_plus,
            areas_kpc2,
            sigma_sub,
            shmf["slope"],
            shmf["pivot_mass_msun"],
            mhm_grid,
            suppression_abc,
            fold["integration"]["samples_per_bin"],
            fold["discrimination"]["delta_logl_threshold"],
        )
        minus = _compute_fold(
            masses_minus,
            areas_kpc2,
            sigma_sub,
            shmf["slope"],
            shmf["pivot_mass_msun"],
            mhm_grid,
            suppression_abc,
            fold["integration"]["samples_per_bin"],
            fold["discrimination"]["delta_logl_threshold"],
        )
        robustness = {
            "shift_dex": shift,
            "masses_shift_plus": masses_plus,
            "masses_shift_minus": masses_minus,
            "mass_range_folded_shift_plus": np.asarray(
                [masses_plus[0], masses_plus[-1]]
            ),
            "mass_range_folded_shift_minus": np.asarray(
                [masses_minus[0], masses_minus[-1]]
            ),
            "mu_cdm_shift_plus": plus["mu_cdm"],
            "mu_cdm_shift_minus": minus["mu_cdm"],
            "mu_per_bin_cdm_shift_plus": plus["mu_per_bin_cdm"],
            "mu_per_bin_cdm_shift_minus": minus["mu_per_bin_cdm"],
            "mu_wdm_shift_plus": plus["mu_wdm"],
            "mu_wdm_shift_minus": minus["mu_wdm"],
            "D_shift_plus": plus["D_per_lens"],
            "D_shift_minus": minus["D_per_lens"],
            "N_req_shift_plus": plus["N_req"],
            "N_req_shift_minus": minus["N_req"],
            "N_req_ceil_shift_plus": plus["N_req_ceil"],
            "N_req_ceil_shift_minus": minus["N_req_ceil"],
            "N_req_single_bin_shift_plus": plus["N_req_single_bin"],
            "N_req_single_bin_shift_minus": minus["N_req_single_bin"],
        }

    from hwoslaps.provenance import revision_provenance

    forecast_id = _forecast_id(
        _SCHEMA_VERSION,
        normalized,
        manifest,
        fold["statistic"],
        fold["detection_q_threshold"],
    )
    return SubhaloForecastData(
        schema_version=_SCHEMA_VERSION,
        forecast_id=forecast_id,
        ladder_masses_msun=masses,
        detectable_area_arcsec2=areas_arcsec2,
        detectable_area_kpc2=areas_kpc2,
        statistic=fold["statistic"],
        detection_q_threshold=fold["detection_q_threshold"],
        sigma_sub_kpc2=sigma_sub,
        normalization_mode=normalization_mode,
        normalization_preset=normalization_preset,
        resolved_f_sub=resolved_f_sub,
        from_f_sub=deepcopy(host),
        shmf_slope=shmf["slope"],
        pivot_mass_msun=shmf["pivot_mass_msun"],
        suppression=suppression,
        suppression_abc=suppression_abc,
        mhm_grid_msun=mhm_grid,
        m_wdm_kev=thermal_grid,
        mu_cdm=result["mu_cdm"],
        mu_per_bin_cdm=result["mu_per_bin_cdm"],
        mu_wdm=result["mu_wdm"],
        mu_per_bin_wdm=result["mu_per_bin_wdm"],
        D_per_lens=result["D_per_lens"],
        N_req=result["N_req"],
        N_req_ceil=result["N_req_ceil"],
        N_req_single_bin=result["N_req_single_bin"],
        mass_range_folded_msun=(float(masses[0]), float(masses[-1])),
        robustness=robustness,
        inputs_verified=inputs_verified,
        map_manifest=manifest,
        source_digests=_source_digests(),
        revision_provenance=revision_provenance(),
        config=normalized,
    )


def _compatibility_error(name: str) -> ValueError:
    """Build a path-specific ratio compatibility error."""
    return ValueError(f"forecast_ratio requires identical {name}")


def forecast_ratio(
    numerator: SubhaloForecastData,
    baseline: SubhaloForecastData,
) -> dict:
    """Return requirement ratios for two compatible PSF-state forecasts.

    Parameters
    ----------
    numerator : `SubhaloForecastData`
        Forecast for the PSF state being compared.
    baseline : `SubhaloForecastData`
        Compatible baseline PSF-state forecast.

    Returns
    -------
    ratios : `dict`
        Elementwise ``mu_ratio``, infinity-aware ``n_req_ratio``, and scalar
        ``mu_cdm_ratio``. Zero baseline mu values produce NaN. Two infinite
        required-lens counts have ratio one.

    Raises
    ------
    ValueError
        Raised when the physical forecast axes or fold settings differ.
    """
    if not np.array_equal(numerator.mhm_grid_msun, baseline.mhm_grid_msun):
        raise _compatibility_error("half-mode mass grid")
    if numerator.statistic != baseline.statistic:
        raise _compatibility_error("statistic")
    if numerator.detection_q_threshold != baseline.detection_q_threshold:
        raise _compatibility_error("detection threshold")
    shmf_numerator = (
        numerator.sigma_sub_kpc2,
        numerator.normalization_mode,
        numerator.normalization_preset,
        numerator.resolved_f_sub,
        numerator.from_f_sub,
        numerator.shmf_slope,
        numerator.pivot_mass_msun,
    )
    shmf_baseline = (
        baseline.sigma_sub_kpc2,
        baseline.normalization_mode,
        baseline.normalization_preset,
        baseline.resolved_f_sub,
        baseline.from_f_sub,
        baseline.shmf_slope,
        baseline.pivot_mass_msun,
    )
    if shmf_numerator != shmf_baseline:
        raise _compatibility_error("SHMF parameters")
    if (
        numerator.suppression != baseline.suppression
        or numerator.suppression_abc != baseline.suppression_abc
    ):
        raise _compatibility_error("WDM suppression")
    if not np.array_equal(
        numerator.ladder_masses_msun,
        baseline.ladder_masses_msun,
    ):
        raise _compatibility_error("ladder masses")
    numerator_config = deepcopy(numerator.config)
    baseline_config = deepcopy(baseline.config)
    for config in (numerator_config, baseline_config):
        fold = config["subhalo_forecast"]
        for item in fold["maps"]:
            item.pop("path")
        fold.pop("allow_unverified_maps")
    if numerator_config != baseline_config:
        raise ValueError(
            "forecast_ratio requires identical fold settings "
            "(only map paths and PSF state may differ)"
        )

    mu_ratio = np.full(numerator.mu_wdm.shape, np.nan, dtype=float)
    nonzero_mu = baseline.mu_wdm != 0.0
    mu_ratio[nonzero_mu] = numerator.mu_wdm[nonzero_mu]/baseline.mu_wdm[nonzero_mu]
    n_req_ratio = np.full(numerator.N_req.shape, np.nan, dtype=float)
    both_infinite = np.isinf(numerator.N_req) & np.isinf(baseline.N_req)
    n_req_ratio[both_infinite] = 1.0
    finite_baseline = np.isfinite(baseline.N_req) & (baseline.N_req != 0.0)
    n_req_ratio[finite_baseline] = (
        numerator.N_req[finite_baseline]/baseline.N_req[finite_baseline]
    )
    finite_over_infinite = np.isfinite(numerator.N_req) & np.isinf(baseline.N_req)
    n_req_ratio[finite_over_infinite] = 0.0
    both_zero = (numerator.N_req == 0.0) & (baseline.N_req == 0.0)
    n_req_ratio[both_zero] = 1.0
    positive_over_zero = (numerator.N_req > 0.0) & (baseline.N_req == 0.0)
    n_req_ratio[positive_over_zero] = np.inf
    mu_cdm_ratio = (
        float("nan")
        if baseline.mu_cdm == 0.0
        else numerator.mu_cdm/baseline.mu_cdm
    )
    return {
        "mu_ratio": mu_ratio,
        "n_req_ratio": n_req_ratio,
        "mu_cdm_ratio": mu_cdm_ratio,
    }


def _validate_manifest(manifest: Any) -> tuple[int, int]:
    """Validate loaded manifest structure and return map and bin counts."""
    if not isinstance(manifest, list) or len(manifest) < 3:
        raise ValueError("Subhalo forecast manifest must contain at least 3 maps")
    expected = {"path", "sha256", "q_grid_digest", "stored_mass_msun"}
    for index, item in enumerate(manifest):
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError(
                f"Subhalo forecast manifest entry {index} has invalid members"
            )
        if not isinstance(item["path"], str):
            raise ValueError(f"Subhalo forecast manifest entry {index} path is invalid")
        for name in ("sha256", "q_grid_digest"):
            digest = item[name]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(
                    f"Subhalo forecast manifest entry {index} {name} is invalid"
                )
        mass = item["stored_mass_msun"]
        if isinstance(mass, bool) or not isinstance(mass, (int, float)):
            raise ValueError(
                f"Subhalo forecast manifest entry {index} mass is invalid"
            )
    return len(manifest), len(manifest) - 1


def _shape(array: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    """Require one exact artifact-array shape."""
    if np.asarray(array).shape != expected:
        raise ValueError(
            f"Subhalo forecast member {name} shape {np.asarray(array).shape} "
            f"does not match {expected}"
        )


def _validate_forecast_shapes(data: SubhaloForecastData) -> None:
    """Validate every array shape against manifest dimensions."""
    n_maps, n_bins = _validate_manifest(data.map_manifest)
    n_mhm = len(data.mhm_grid_msun)
    if n_mhm < 2:
        raise ValueError("Subhalo forecast half-mode mass grid is too short")
    for name in (
        "ladder_masses_msun",
        "detectable_area_arcsec2",
        "detectable_area_kpc2",
    ):
        _shape(getattr(data, name), (n_maps,), name)
    _shape(data.suppression_abc, (3,), "suppression_abc")
    for name in (
        "mhm_grid_msun",
        "m_wdm_kev",
        "mu_wdm",
        "D_per_lens",
        "N_req",
        "N_req_ceil",
        "N_req_single_bin",
    ):
        _shape(getattr(data, name), (n_mhm,), name)
    _shape(data.mu_per_bin_cdm, (n_bins,), "mu_per_bin_cdm")
    _shape(data.mu_per_bin_wdm, (n_bins, n_mhm), "mu_per_bin_wdm")
    _shape(data.mass_range_folded_msun, (2,), "mass_range_folded_msun")
    manifest_masses = np.asarray([
        item["stored_mass_msun"] for item in data.map_manifest
    ])
    if not np.array_equal(data.ladder_masses_msun, manifest_masses):
        raise ValueError(
            "Subhalo forecast ladder masses do not match manifest masses"
        )
    if data.robustness is None:
        return
    if set(data.robustness) != set(_ROBUSTNESS_FIELDS):
        raise ValueError("Subhalo forecast robustness field set mismatch")
    shapes = {
        "masses_shift_plus": (n_maps,),
        "masses_shift_minus": (n_maps,),
        "mass_range_folded_shift_plus": (2,),
        "mass_range_folded_shift_minus": (2,),
        "mu_per_bin_cdm_shift_plus": (n_bins,),
        "mu_per_bin_cdm_shift_minus": (n_bins,),
        "mu_wdm_shift_plus": (n_mhm,),
        "mu_wdm_shift_minus": (n_mhm,),
        "D_shift_plus": (n_mhm,),
        "D_shift_minus": (n_mhm,),
        "N_req_shift_plus": (n_mhm,),
        "N_req_shift_minus": (n_mhm,),
        "N_req_ceil_shift_plus": (n_mhm,),
        "N_req_ceil_shift_minus": (n_mhm,),
        "N_req_single_bin_shift_plus": (n_mhm,),
        "N_req_single_bin_shift_minus": (n_mhm,),
    }
    for name, expected in shapes.items():
        _shape(data.robustness[name], expected, name)


def _content_digest(payload: dict[str, np.ndarray]) -> str:
    """Hash every named payload value with its dtype, shape, and bytes."""
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = np.asarray(payload[name])
        digest.update(name.encode())
        digest.update(b":")
        digest.update(value.dtype.str.encode())
        digest.update(b":")
        digest.update(str(value.shape).encode())
        digest.update(b":")
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def save_subhalo_forecast_npz(data, path) -> Path:
    """Save a deterministic subhalo forecast NPZ artifact.

    Parameters
    ----------
    data : `SubhaloForecastData`
        Forecast to persist.
    path : path-like
        Destination path.

    Returns
    -------
    path : `pathlib.Path`
        Written destination.

    Raises
    ------
    ValueError
        Raised when shapes or the forecast identity are inconsistent.
    """
    _validate_forecast_shapes(data)
    expected_id = _forecast_id(
        data.schema_version,
        data.config,
        data.map_manifest,
        data.statistic,
        data.detection_q_threshold,
    )
    if expected_id != data.forecast_id:
        raise ValueError(
            f"Subhalo forecast forecast_id mismatch: {data.forecast_id} != "
            f"{expected_id}"
        )
    payload = {
        "schema_version": np.asarray(data.schema_version, dtype=np.int64),
        "forecast_id": np.asarray(data.forecast_id),
        "ladder_masses_msun": np.asarray(data.ladder_masses_msun, dtype=float),
        "detectable_area_arcsec2": np.asarray(
            data.detectable_area_arcsec2,
            dtype=float,
        ),
        "detectable_area_kpc2": np.asarray(data.detectable_area_kpc2, dtype=float),
        "statistic": np.asarray(data.statistic),
        "detection_q_threshold": np.asarray(data.detection_q_threshold, dtype=float),
        "sigma_sub_kpc2": np.asarray(data.sigma_sub_kpc2, dtype=float),
        "normalization_mode": np.asarray(data.normalization_mode),
        "normalization_preset_json": np.asarray(
            _canonical_json(data.normalization_preset)
        ),
        "resolved_f_sub": np.asarray(
            np.nan if data.resolved_f_sub is None else data.resolved_f_sub,
            dtype=float,
        ),
        "from_f_sub_json": np.asarray(_canonical_json(data.from_f_sub)),
        "shmf_slope": np.asarray(data.shmf_slope, dtype=float),
        "pivot_mass_msun": np.asarray(data.pivot_mass_msun, dtype=float),
        "suppression": np.asarray(data.suppression),
        "suppression_abc": np.asarray(data.suppression_abc, dtype=float),
        "mhm_grid_msun": np.asarray(data.mhm_grid_msun, dtype=float),
        "m_wdm_kev": np.asarray(data.m_wdm_kev, dtype=float),
        "mu_cdm": np.asarray(data.mu_cdm, dtype=float),
        "mu_per_bin_cdm": np.asarray(data.mu_per_bin_cdm, dtype=float),
        "mu_wdm": np.asarray(data.mu_wdm, dtype=float),
        "mu_per_bin_wdm": np.asarray(data.mu_per_bin_wdm, dtype=float),
        "D_per_lens": np.asarray(data.D_per_lens, dtype=float),
        "N_req": np.asarray(data.N_req, dtype=float),
        "N_req_ceil": np.asarray(data.N_req_ceil, dtype=float),
        "N_req_single_bin": np.asarray(data.N_req_single_bin, dtype=float),
        "mass_range_folded_msun": np.asarray(
            data.mass_range_folded_msun,
            dtype=float,
        ),
        "robustness_present": np.asarray(data.robustness is not None, dtype=np.bool_),
        "inputs_verified": np.asarray(data.inputs_verified, dtype=np.bool_),
        "map_manifest_json": np.asarray(_canonical_json(data.map_manifest)),
        "source_digests_json": np.asarray(_canonical_json(data.source_digests)),
        "revision_provenance_json": np.asarray(
            _canonical_json(data.revision_provenance)
        ),
        "config_json": np.asarray(_canonical_json(data.config)),
    }
    if data.robustness is not None:
        for name in _ROBUSTNESS_FIELDS:
            payload[name] = np.asarray(data.robustness[name])
    payload["content_digest"] = np.asarray(_content_digest(payload))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez(stream, **payload)
    return destination


def _json_from_member(stored: Any, name: str) -> Any:
    """Decode one scalar JSON NPZ member."""
    raw = np.asarray(stored[name]).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _string_from_member(stored: Any, name: str) -> str:
    """Decode one scalar string NPZ member."""
    raw = np.asarray(stored[name]).item()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def load_subhalo_forecast_npz(path) -> SubhaloForecastData:
    """Load and fully verify a subhalo forecast NPZ artifact.

    Parameters
    ----------
    path : path-like
        Source NPZ path.

    Returns
    -------
    data : `SubhaloForecastData`
        Field-wise reconstructed forecast.

    Raises
    ------
    ValueError
        Raised for missing or unexpected members, invalid shapes, unsupported
        schema, or a recomputed identity mismatch.
    """
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as stored:
            members = set(stored.files)
            missing_base = _BASE_ARTIFACT_MEMBERS - members
            if missing_base:
                raise ValueError(
                    "Subhalo forecast artifact missing members: "
                    + ", ".join(sorted(missing_base))
                )
            for name in (
                "schema_version",
                "robustness_present",
                "content_digest",
            ):
                if np.asarray(stored[name]).shape != ():
                    raise ValueError(
                        f"Subhalo forecast scalar member {name} has invalid shape"
                    )
            if np.asarray(stored["robustness_present"]).dtype != np.bool_:
                raise ValueError(
                    "Subhalo forecast robustness_present must have boolean dtype"
                )
            robustness_present = bool(stored["robustness_present"])
            expected = _BASE_ARTIFACT_MEMBERS | (
                set(_ROBUSTNESS_FIELDS) if robustness_present else set()
            )
            missing = sorted(expected - members)
            unexpected = sorted(members - expected)
            if missing or unexpected:
                parts = []
                if missing:
                    parts.append("missing: " + ", ".join(missing))
                if unexpected:
                    parts.append("unexpected: " + ", ".join(unexpected))
                raise ValueError(
                    "Subhalo forecast artifact member mismatch; "
                    + "; ".join(parts)
                )
            digest_payload = {
                name: np.asarray(stored[name])
                for name in stored.files
                if name != "content_digest"
            }
            expected_digest = _content_digest(digest_payload)
            stored_digest = _string_from_member(stored, "content_digest")
            if stored_digest != expected_digest:
                raise ValueError(
                    f"Subhalo forecast content digest mismatch for {source}"
                )
            schema_version = int(stored["schema_version"])
            if schema_version != _SCHEMA_VERSION:
                raise ValueError("Unsupported subhalo forecast schema version")
            scalar_members = {
                "schema_version",
                "forecast_id",
                "statistic",
                "detection_q_threshold",
                "sigma_sub_kpc2",
                "normalization_mode",
                "normalization_preset_json",
                "resolved_f_sub",
                "from_f_sub_json",
                "shmf_slope",
                "pivot_mass_msun",
                "suppression",
                "mu_cdm",
                "robustness_present",
                "inputs_verified",
                "map_manifest_json",
                "source_digests_json",
                "revision_provenance_json",
                "config_json",
                "content_digest",
            }
            if robustness_present:
                scalar_members.update({
                    "shift_dex",
                    "mu_cdm_shift_plus",
                    "mu_cdm_shift_minus",
                })
            nonscalar = sorted(
                name
                for name in scalar_members
                if np.asarray(stored[name]).shape != ()
            )
            if nonscalar:
                raise ValueError(
                    "Subhalo forecast scalar members have invalid shapes: "
                    + ", ".join(nonscalar)
                )
            json_members = (
                "normalization_preset_json",
                "from_f_sub_json",
                "map_manifest_json",
                "source_digests_json",
                "revision_provenance_json",
                "config_json",
            )
            for name in json_members:
                raw = _string_from_member(stored, name)
                try:
                    parsed = json.loads(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Subhalo forecast JSON member {name} is invalid: {exc}"
                    ) from exc
                if raw != _canonical_json(parsed):
                    raise ValueError(
                        f"Subhalo forecast JSON member {name} is not canonical"
                    )
            resolved_f_sub_raw = float(stored["resolved_f_sub"])
            robustness = None
            if robustness_present:
                robustness = {
                    name: (
                        float(stored[name])
                        if name in {
                            "shift_dex",
                            "mu_cdm_shift_plus",
                            "mu_cdm_shift_minus",
                        }
                        else np.asarray(stored[name])
                    )
                    for name in _ROBUSTNESS_FIELDS
                }
            data = SubhaloForecastData(
                schema_version=schema_version,
                forecast_id=_string_from_member(stored, "forecast_id"),
                ladder_masses_msun=np.asarray(stored["ladder_masses_msun"]),
                detectable_area_arcsec2=np.asarray(
                    stored["detectable_area_arcsec2"]
                ),
                detectable_area_kpc2=np.asarray(stored["detectable_area_kpc2"]),
                statistic=_string_from_member(stored, "statistic"),
                detection_q_threshold=float(stored["detection_q_threshold"]),
                sigma_sub_kpc2=float(stored["sigma_sub_kpc2"]),
                normalization_mode=_string_from_member(
                    stored,
                    "normalization_mode",
                ),
                normalization_preset=_json_from_member(
                    stored,
                    "normalization_preset_json",
                ),
                resolved_f_sub=(
                    None if np.isnan(resolved_f_sub_raw) else resolved_f_sub_raw
                ),
                from_f_sub=_json_from_member(stored, "from_f_sub_json"),
                shmf_slope=float(stored["shmf_slope"]),
                pivot_mass_msun=float(stored["pivot_mass_msun"]),
                suppression=_string_from_member(stored, "suppression"),
                suppression_abc=tuple(
                    np.asarray(stored["suppression_abc"], dtype=float)
                ),
                mhm_grid_msun=np.asarray(stored["mhm_grid_msun"]),
                m_wdm_kev=np.asarray(stored["m_wdm_kev"]),
                mu_cdm=float(stored["mu_cdm"]),
                mu_per_bin_cdm=np.asarray(stored["mu_per_bin_cdm"]),
                mu_wdm=np.asarray(stored["mu_wdm"]),
                mu_per_bin_wdm=np.asarray(stored["mu_per_bin_wdm"]),
                D_per_lens=np.asarray(stored["D_per_lens"]),
                N_req=np.asarray(stored["N_req"]),
                N_req_ceil=np.asarray(stored["N_req_ceil"]),
                N_req_single_bin=np.asarray(stored["N_req_single_bin"]),
                mass_range_folded_msun=tuple(
                    np.asarray(stored["mass_range_folded_msun"], dtype=float)
                ),
                robustness=robustness,
                inputs_verified=bool(stored["inputs_verified"]),
                map_manifest=_json_from_member(stored, "map_manifest_json"),
                source_digests=_json_from_member(
                    stored,
                    "source_digests_json",
                ),
                revision_provenance=_json_from_member(
                    stored,
                    "revision_provenance_json",
                ),
                config=_json_from_member(stored, "config_json"),
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not load subhalo forecast artifact {source}: {exc}"
        ) from exc
    try:
        normalized_config = validate_subhalo_forecast_config(data.config)
    except ValueError as exc:
        raise ValueError(
            f"Subhalo forecast config echo is invalid: {exc}"
        ) from exc
    if normalized_config != data.config:
        raise ValueError("Subhalo forecast config echo is not canonical")
    robustness_config = normalized_config["subhalo_forecast"]["robustness"]
    configured_robustness = robustness_config["mass_axis_shift_dex"] > 0.0
    if (data.robustness is not None) != configured_robustness:
        raise ValueError(
            "Subhalo forecast robustness block inconsistent with configured "
            "mass_axis_shift_dex"
        )
    _validate_forecast_shapes(data)
    expected_id = _forecast_id(
        data.schema_version,
        data.config,
        data.map_manifest,
        data.statistic,
        data.detection_q_threshold,
    )
    if expected_id != data.forecast_id:
        raise ValueError(
            f"Subhalo forecast forecast_id mismatch: {data.forecast_id} != "
            f"{expected_id}"
        )
    return data
