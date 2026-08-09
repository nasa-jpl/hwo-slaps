"""Build PyAutoFit models for nonlinear validation."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np

from ...lensing.image_source import load_source_image_asset
from .clumpy_profiles import ClumpyTemplateContext
from .mass_mapping import MassMappingContext
from .model_specs import (
    GalaxySpec,
    ModelSpec,
    PriorSpec,
    ProfileSpec,
    fixed,
    linked,
    uniform,
)
from .trial import SubhaloTrial


DEFAULT_PRIOR_WIDTHS = {
    "lens_centre_sigma_arcsec": 0.005,
    "lens_einstein_radius_sigma": 0.01,
    "lens_ell_comps_sigma": 0.02,
    "source_centre_sigma_arcsec": 0.01,
    "source_ell_comps_sigma": 0.05,
    "source_intensity_frac_sigma": 0.5,
    "source_effective_radius_frac_sigma": 0.3,
    "subhalo_centre_window_arcsec": 0.03,
    "lens_slope_sigma": 0.05,
    "lens_multipole_comp_sigma": 0.01,
    "lens_shear_comp_sigma": 0.01,
    "source_sersic_index_sigma": 0.5,
    "image_flux_scale_frac_sigma": 0.5,
    "image_size_scale_frac_sigma": 0.3,
    "clumpy_flux_scale_frac_sigma": 0.5,
    "clumpy_size_scale_frac_sigma": 0.3,
    "clump_centre_sigma_arcsec": 0.01,
    "clump_intensity_frac_sigma": 0.5,
    "clump_effective_radius_frac_sigma": 0.3,
    "subhalo_freed_centre_window_arcsec": 0.15,
}
"""Default local prior widths for validation fits."""


def _widths(priors_config: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Merge user-supplied prior widths with defaults."""
    widths = dict(DEFAULT_PRIOR_WIDTHS)
    if priors_config:
        widths.update(priors_config)
    return widths


def _uniform_around(value: float, half_width: float) -> PriorSpec:
    """Return a uniform prior centered on a scalar value."""
    return uniform(
        float(value) - float(half_width),
        float(value) + float(half_width),
    )


def _uniform_fraction(value: float, frac_width: float) -> PriorSpec:
    """Return a positive uniform prior around a scalar value."""
    value = float(value)
    half_width = abs(value) * float(frac_width)
    lower = max(0.0, value - half_width)
    upper = value + half_width
    if lower == upper:
        upper = lower + 1.0e-12
    return uniform(lower, upper)


def _clipped_uniform(
    value: float,
    half_width: float,
    lower_bound: float,
    upper_bound: float,
    *,
    open_interval: bool = False,
) -> PriorSpec:
    """Return a locally centered prior clipped to physical bounds."""
    value = float(value)
    if open_interval:
        lower_bound = float(np.nextafter(lower_bound, upper_bound))
        upper_bound = float(np.nextafter(upper_bound, lower_bound))
    lower = max(value - float(half_width), lower_bound)
    upper = min(value + float(half_width), upper_bound)
    if lower > value or upper < value or lower >= upper:
        raise ValueError(
            f"Truth value {value} lies outside the clipped prior box "
            f"[{lower_bound}, {upper_bound}]"
        )
    return uniform(lower, upper)


def _ell_prior(value: float, half_width: float) -> PriorSpec:
    """Return an ellipticity-component prior inside its safe domain."""
    return _clipped_uniform(
        value,
        half_width,
        -0.9,
        0.9,
        open_interval=True,
    )


def _pin_local_prior(
    prior: PriorSpec,
    target_value: float,
    pin_to_targets: bool,
) -> PriorSpec:
    """Replace a validated local prior with its exact target when requested."""
    if pin_to_targets:
        return fixed(float(target_value))
    return prior


def _guard_supported_config(full_config: Dict[str, Any]) -> None:
    """Reject unresolved normalization or unsupported profile types."""
    throughput = full_config.get("observation", {}).get("throughput")
    if throughput is not None and float(throughput) != 1.0:
        raise ValueError(
            "Nonlinear source normalization for observation.throughput != "
            "1.0 is deliberately unresolved"
        )
    truth_lens = full_config["lensing"]["lens_galaxy"]
    mass_type = truth_lens["mass"].get("type")
    if mass_type not in {"Isothermal", "PowerLaw"}:
        raise ValueError(f"Unsupported truth mass profile type: {mass_type}")
    light_type = full_config["lensing"]["source_galaxy"]["light"].get(
        "type"
    )
    if light_type not in {"Exponential", "Sersic", "Clumpy", "Image"}:
        raise ValueError(f"Unsupported truth source light type: {light_type}")


def _fit_lens_config(full_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the matched or explicit fit-side macro configuration."""
    truth_lens = full_config["lensing"]["lens_galaxy"]
    fit_lens = full_config.get("modeling", {}).get("fit_lens")
    if (
        isinstance(fit_lens, dict)
        and str(fit_lens.get("mode", "")).lower() == "explicit"
    ):
        return fit_lens["lens_galaxy"]
    return truth_lens


def _macro_components(
    lens_config: Dict[str, Any],
    widths: Dict[str, float],
    *,
    pin_to_targets: bool = False,
) -> Dict[str, ProfileSpec]:
    """Build fit-side macro mass, multipole, and shear components."""
    mass_config = lens_config["mass"]
    mass_type = mass_config.get("type")
    if mass_type not in {"Isothermal", "PowerLaw"}:
        raise ValueError(f"Unsupported fit mass profile type: {mass_type}")
    centre = mass_config["centre"]
    ell_comps = mass_config["ell_comps"]
    parameters = {
        "centre_0": _pin_local_prior(
            _uniform_around(
                centre[0],
                widths["lens_centre_sigma_arcsec"],
            ),
            centre[0],
            pin_to_targets,
        ),
        "centre_1": _pin_local_prior(
            _uniform_around(
                centre[1],
                widths["lens_centre_sigma_arcsec"],
            ),
            centre[1],
            pin_to_targets,
        ),
        "einstein_radius": _pin_local_prior(
            _uniform_around(
                mass_config["einstein_radius"],
                widths["lens_einstein_radius_sigma"],
            ),
            mass_config["einstein_radius"],
            pin_to_targets,
        ),
        "ell_comps_0": _pin_local_prior(
            _ell_prior(
                ell_comps[0],
                widths["lens_ell_comps_sigma"],
            ),
            ell_comps[0],
            pin_to_targets,
        ),
        "ell_comps_1": _pin_local_prior(
            _ell_prior(
                ell_comps[1],
                widths["lens_ell_comps_sigma"],
            ),
            ell_comps[1],
            pin_to_targets,
        ),
    }
    if mass_type == "PowerLaw":
        parameters["slope"] = _pin_local_prior(
            _clipped_uniform(
                mass_config["slope"],
                widths["lens_slope_sigma"],
                1.2,
                2.8,
                open_interval=True,
            ),
            mass_config["slope"],
            pin_to_targets,
        )

    components = {
        "mass": ProfileSpec(class_name=mass_type, parameters=parameters)
    }
    if mass_type == "PowerLaw":
        for order_name in sorted(mass_config.get("multipoles", {})):
            order = int(order_name[1:])
            multipole_comps = mass_config["multipoles"][order_name]
            components[f"multipole_m{order}"] = ProfileSpec(
                class_name="PowerLawMultipole",
                parameters={
                    "m": fixed(order),
                    "multipole_comps_0": _pin_local_prior(
                        _uniform_around(
                            multipole_comps[0],
                            widths["lens_multipole_comp_sigma"],
                        ),
                        multipole_comps[0],
                        pin_to_targets,
                    ),
                    "multipole_comps_1": _pin_local_prior(
                        _uniform_around(
                            multipole_comps[1],
                            widths["lens_multipole_comp_sigma"],
                        ),
                        multipole_comps[1],
                        pin_to_targets,
                    ),
                    "centre_0": linked("mass", "centre_0"),
                    "centre_1": linked("mass", "centre_1"),
                    "einstein_radius": linked(
                        "mass",
                        "einstein_radius",
                    ),
                    "slope": linked("mass", "slope"),
                },
            )
    if "shear" in lens_config:
        shear = lens_config["shear"]
        components["shear"] = ProfileSpec(
            class_name="ExternalShear",
            parameters={
                "gamma_1": _pin_local_prior(
                    _uniform_around(
                        shear[0],
                        widths["lens_shear_comp_sigma"],
                    ),
                    shear[0],
                    pin_to_targets,
                ),
                "gamma_2": _pin_local_prior(
                    _uniform_around(
                        shear[1],
                        widths["lens_shear_comp_sigma"],
                    ),
                    shear[1],
                    pin_to_targets,
                ),
            },
        )
    return components


def _analytic_source_profile(
    light_config: Dict[str, Any],
    widths: Dict[str, float],
    *,
    pin_to_targets: bool = False,
) -> ProfileSpec:
    """Build an Exponential or Sersic source profile specification."""
    centre = light_config["centre"]
    ell_comps = light_config["ell_comps"]
    parameters = {
        "centre_0": _pin_local_prior(
            _uniform_around(
                centre[0],
                widths["source_centre_sigma_arcsec"],
            ),
            centre[0],
            pin_to_targets,
        ),
        "centre_1": _pin_local_prior(
            _uniform_around(
                centre[1],
                widths["source_centre_sigma_arcsec"],
            ),
            centre[1],
            pin_to_targets,
        ),
        "ell_comps_0": _pin_local_prior(
            _ell_prior(
                ell_comps[0],
                widths["source_ell_comps_sigma"],
            ),
            ell_comps[0],
            pin_to_targets,
        ),
        "ell_comps_1": _pin_local_prior(
            _ell_prior(
                ell_comps[1],
                widths["source_ell_comps_sigma"],
            ),
            ell_comps[1],
            pin_to_targets,
        ),
        "intensity": _pin_local_prior(
            _uniform_fraction(
                light_config["intensity"],
                widths["source_intensity_frac_sigma"],
            ),
            light_config["intensity"],
            pin_to_targets,
        ),
        "effective_radius": _pin_local_prior(
            _uniform_fraction(
                light_config["effective_radius"],
                widths["source_effective_radius_frac_sigma"],
            ),
            light_config["effective_radius"],
            pin_to_targets,
        ),
    }
    if light_config["type"] == "Sersic":
        parameters["sersic_index"] = _pin_local_prior(
            _clipped_uniform(
                light_config["sersic_index"],
                widths["source_sersic_index_sigma"],
                0.3,
                10.0,
            ),
            light_config["sersic_index"],
            pin_to_targets,
        )
    return ProfileSpec(
        class_name=light_config["type"],
        parameters=parameters,
    )


def _clumpy_template_context(
    light_config: Dict[str, Any],
) -> ClumpyTemplateContext:
    """Build the immutable clumpy-source template context."""
    host_config = light_config["host"]
    host_centre = tuple(float(value) for value in host_config["centre"])
    host = (
        float(host_config["ell_comps"][0]),
        float(host_config["ell_comps"][1]),
        float(host_config["intensity"]),
        float(host_config["effective_radius"]),
        float(host_config["sersic_index"]),
    )
    clumps = []
    for clump in light_config["clumps"]:
        clumps.append(
            (
                float(clump["centre"][0]) - host_centre[0],
                float(clump["centre"][1]) - host_centre[1],
                float(clump["ell_comps"][0]),
                float(clump["ell_comps"][1]),
                float(clump["intensity"]),
                float(clump["effective_radius"]),
                float(clump["sersic_index"]),
            )
        )
    canonical = (host, host_centre, tuple(clumps))
    context_hash = hashlib.sha256(
        repr(canonical).encode("utf-8")
    ).hexdigest()[:16]
    return ClumpyTemplateContext(
        host=host,
        host_centre=host_centre,
        clumps=tuple(clumps),
        context_hash=context_hash,
    )


def _clumpy_composite_profile(
    light_config: Dict[str, Any],
    widths: Dict[str, float],
    parameterization: str,
    *,
    pin_to_targets: bool = False,
) -> ProfileSpec:
    """Build a rigid or host-free transformed clumpy profile."""
    context = _clumpy_template_context(light_config)
    host_config = light_config["host"]
    host_ell = host_config["ell_comps"]
    parameters = {
        "centre_0": _pin_local_prior(
            _uniform_around(
                host_config["centre"][0],
                widths["source_centre_sigma_arcsec"],
            ),
            host_config["centre"][0],
            pin_to_targets,
        ),
        "centre_1": _pin_local_prior(
            _uniform_around(
                host_config["centre"][1],
                widths["source_centre_sigma_arcsec"],
            ),
            host_config["centre"][1],
            pin_to_targets,
        ),
        "flux_scale": _pin_local_prior(
            _uniform_fraction(
                light_config["flux_scale"],
                widths["clumpy_flux_scale_frac_sigma"],
            ),
            light_config["flux_scale"],
            pin_to_targets,
        ),
        "size_scale": _pin_local_prior(
            _uniform_fraction(
                light_config["size_scale"],
                widths["clumpy_size_scale_frac_sigma"],
            ),
            light_config["size_scale"],
            pin_to_targets,
        ),
    }
    if parameterization == "host_free":
        parameters.update(
            {
                "host_ell_comps_0": _pin_local_prior(
                    _ell_prior(
                        host_ell[0],
                        widths["source_ell_comps_sigma"],
                    ),
                    host_ell[0],
                    pin_to_targets,
                ),
                "host_ell_comps_1": _pin_local_prior(
                    _ell_prior(
                        host_ell[1],
                        widths["source_ell_comps_sigma"],
                    ),
                    host_ell[1],
                    pin_to_targets,
                ),
                "host_intensity": _pin_local_prior(
                    _uniform_fraction(
                        host_config["intensity"],
                        widths["source_intensity_frac_sigma"],
                    ),
                    host_config["intensity"],
                    pin_to_targets,
                ),
                "host_effective_radius": _pin_local_prior(
                    _uniform_fraction(
                        host_config["effective_radius"],
                        widths["source_effective_radius_frac_sigma"],
                    ),
                    host_config["effective_radius"],
                    pin_to_targets,
                ),
                "host_sersic_index": _pin_local_prior(
                    _clipped_uniform(
                        host_config["sersic_index"],
                        widths["source_sersic_index_sigma"],
                        0.3,
                        10.0,
                    ),
                    host_config["sersic_index"],
                    pin_to_targets,
                ),
            }
        )
    else:
        parameters.update(
            {
                "host_ell_comps": fixed(tuple(float(v) for v in host_ell)),
                "host_intensity": fixed(float(host_config["intensity"])),
                "host_effective_radius": fixed(
                    float(host_config["effective_radius"])
                ),
                "host_sersic_index": fixed(
                    float(host_config["sersic_index"])
                ),
            }
        )
    parameters["template_context"] = fixed(context)
    return ProfileSpec(
        class_name="ClumpyTransformedSource",
        parameters=parameters,
    )


def _sersic_component(
    component: Dict[str, Any],
    widths: Dict[str, float],
    *,
    clump: bool,
    centre: tuple,
    intensity: float,
    effective_radius: float,
    pin_to_targets: bool = False,
) -> ProfileSpec:
    """Build one fully-free-mode host or clump Sersic component."""
    centre_width = (
        widths["clump_centre_sigma_arcsec"]
        if clump
        else widths["source_centre_sigma_arcsec"]
    )
    parameters = {
        "centre_0": _pin_local_prior(
            _uniform_around(centre[0], centre_width),
            centre[0],
            pin_to_targets,
        ),
        "centre_1": _pin_local_prior(
            _uniform_around(centre[1], centre_width),
            centre[1],
            pin_to_targets,
        ),
    }
    if clump:
        parameters.update(
            {
                "ell_comps_0": fixed(float(component["ell_comps"][0])),
                "ell_comps_1": fixed(float(component["ell_comps"][1])),
                "intensity": _pin_local_prior(
                    _uniform_fraction(
                        intensity,
                        widths["clump_intensity_frac_sigma"],
                    ),
                    intensity,
                    pin_to_targets,
                ),
                "effective_radius": _pin_local_prior(
                    _uniform_fraction(
                        effective_radius,
                        widths["clump_effective_radius_frac_sigma"],
                    ),
                    effective_radius,
                    pin_to_targets,
                ),
                "sersic_index": fixed(float(component["sersic_index"])),
            }
        )
    else:
        parameters.update(
            {
                "ell_comps_0": _pin_local_prior(
                    _ell_prior(
                        component["ell_comps"][0],
                        widths["source_ell_comps_sigma"],
                    ),
                    component["ell_comps"][0],
                    pin_to_targets,
                ),
                "ell_comps_1": _pin_local_prior(
                    _ell_prior(
                        component["ell_comps"][1],
                        widths["source_ell_comps_sigma"],
                    ),
                    component["ell_comps"][1],
                    pin_to_targets,
                ),
                "intensity": _pin_local_prior(
                    _uniform_fraction(
                        intensity,
                        widths["source_intensity_frac_sigma"],
                    ),
                    intensity,
                    pin_to_targets,
                ),
                "effective_radius": _pin_local_prior(
                    _uniform_fraction(
                        effective_radius,
                        widths["source_effective_radius_frac_sigma"],
                    ),
                    effective_radius,
                    pin_to_targets,
                ),
                "sersic_index": _pin_local_prior(
                    _clipped_uniform(
                        component["sersic_index"],
                        widths["source_sersic_index_sigma"],
                        0.3,
                        10.0,
                    ),
                    component["sersic_index"],
                    pin_to_targets,
                ),
            }
        )
    return ProfileSpec(class_name="Sersic", parameters=parameters)


def _fully_free_clumpy_components(
    light_config: Dict[str, Any],
    widths: Dict[str, float],
    *,
    pin_to_targets: bool = False,
) -> Dict[str, ProfileSpec]:
    """Build independently parameterized as-built clumpy components."""
    flux_scale = float(light_config["flux_scale"])
    size_scale = float(light_config["size_scale"])
    host_config = light_config["host"]
    host_centre = tuple(float(value) for value in host_config["centre"])
    components = {
        "host": _sersic_component(
            host_config,
            widths,
            clump=False,
            centre=host_centre,
            intensity=float(host_config["intensity"]) * flux_scale,
            effective_radius=(
                float(host_config["effective_radius"]) * size_scale
            ),
            pin_to_targets=pin_to_targets,
        )
    }
    for index, clump in enumerate(light_config["clumps"]):
        offset = (
            float(clump["centre"][0]) - host_centre[0],
            float(clump["centre"][1]) - host_centre[1],
        )
        centre = (
            host_centre[0] + size_scale * offset[0],
            host_centre[1] + size_scale * offset[1],
        )
        components[f"clump_{index}"] = _sersic_component(
            clump,
            widths,
            clump=True,
            centre=centre,
            intensity=float(clump["intensity"]) * flux_scale,
            effective_radius=float(clump["effective_radius"]) * size_scale,
            pin_to_targets=pin_to_targets,
        )
    return components


def _source_components(
    light_config: Dict[str, Any],
    widths: Dict[str, float],
    clumpy_fit_parameterization: str,
    *,
    pin_to_targets: bool = False,
) -> tuple[Dict[str, ProfileSpec], Dict[str, Any]]:
    """Build source components and model metadata."""
    light_type = light_config["type"]
    if light_type in {"Exponential", "Sersic"}:
        return {
            "light": _analytic_source_profile(
                light_config,
                widths,
                pin_to_targets=pin_to_targets,
            )
        }, {}
    if light_type == "Image":
        asset = load_source_image_asset(light_config["asset_path"])
        profile = ProfileSpec(
            class_name="ImageSource",
            parameters={
                "centre_0": _pin_local_prior(
                    _uniform_around(
                        light_config["centre"][0],
                        widths["source_centre_sigma_arcsec"],
                    ),
                    light_config["centre"][0],
                    pin_to_targets,
                ),
                "centre_1": _pin_local_prior(
                    _uniform_around(
                        light_config["centre"][1],
                        widths["source_centre_sigma_arcsec"],
                    ),
                    light_config["centre"][1],
                    pin_to_targets,
                ),
                "flux_scale": _pin_local_prior(
                    _uniform_fraction(
                        light_config["flux_scale"],
                        widths["image_flux_scale_frac_sigma"],
                    ),
                    light_config["flux_scale"],
                    pin_to_targets,
                ),
                "size_scale": _pin_local_prior(
                    _uniform_fraction(
                        light_config["size_scale"],
                        widths["image_size_scale_frac_sigma"],
                    ),
                    light_config["size_scale"],
                    pin_to_targets,
                ),
                "rotation_deg": fixed(float(light_config["rotation_deg"])),
                "total_flux": fixed(float(light_config["total_flux"])),
                "asset": fixed(asset),
            },
        )
        return {"light": profile}, {
            "image_source_asset_hash": asset.sha256_16,
            "requires_cpu": True,
        }
    if clumpy_fit_parameterization not in {"rigid", "host_free", "fully_free"}:
        raise ValueError(
            "clumpy_fit_parameterization must be 'rigid', 'host_free', or "
            "'fully_free'"
        )
    metadata = {
        "clumpy_fit_parameterization": clumpy_fit_parameterization,
    }
    if clumpy_fit_parameterization == "fully_free":
        return _fully_free_clumpy_components(
            light_config,
            widths,
            pin_to_targets=pin_to_targets,
        ), metadata
    metadata["requires_cpu"] = True
    return {
        "light": _clumpy_composite_profile(
            light_config,
            widths,
            clumpy_fit_parameterization,
            pin_to_targets=pin_to_targets,
        )
    }, metadata


def smooth_model_spec_from_config(
    full_config: Dict[str, Any],
    priors_config: Optional[Dict[str, Any]] = None,
    clumpy_fit_parameterization: str = "host_free",
    *,
    pin_to_targets: bool = False,
) -> ModelSpec:
    """Build a source-neutral smooth-model specification from config.

    Parameters
    ----------
    full_config : `dict`
        Full HWO-SLAPS configuration.
    priors_config : `dict`, optional
        Prior-width overrides.
    clumpy_fit_parameterization : `str`, optional
        Clumpy-source fit mode: ``"rigid"``, ``"host_free"``, or
        ``"fully_free"``.
    pin_to_targets : `bool`, optional
        Internal switch replacing validated local priors with fixed targets.

    Returns
    -------
    spec : `ModelSpec`
        Smooth lens/source model specification.
    """
    _guard_supported_config(full_config)
    widths = _widths(priors_config)
    truth_lens = full_config["lensing"]["lens_galaxy"]
    source_config = full_config["lensing"]["source_galaxy"]
    fit_lens = _fit_lens_config(full_config)
    lens_components = _macro_components(
        fit_lens,
        widths,
        pin_to_targets=pin_to_targets,
    )
    source_components, source_metadata = _source_components(
        source_config["light"],
        widths,
        clumpy_fit_parameterization,
        pin_to_targets=pin_to_targets,
    )

    metadata = {"builder": "smooth_model_spec_from_config"}
    metadata.update(source_metadata)
    return ModelSpec(
        model_type="smooth",
        fit_mode="smooth",
        galaxies={
            "lens": GalaxySpec(
                name="lens",
                redshift=fixed(float(truth_lens["redshift"])),
                components=lens_components,
            ),
            "source": GalaxySpec(
                name="source",
                redshift=fixed(float(source_config["redshift"])),
                components=source_components,
            ),
        },
        metadata=metadata,
    )


def _validate_freed_context(
    full_config: Dict[str, Any],
    trial: SubhaloTrial,
    mass_context: Optional[MassMappingContext],
) -> MassMappingContext:
    """Validate the explicit mass context for one freed trial."""
    if mass_context is None:
        raise ValueError(
            "freed mode requires mass_context from "
            "build_mass_mapping_context or "
            "build_mass_mapping_context_explicit"
        )
    if trial.model != mass_context.subhalo_model:
        raise ValueError("trial model does not match mass_context.subhalo_model")
    if not np.isclose(
        trial.lens_redshift,
        mass_context.z_lens,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("trial lens redshift does not match mass_context")
    if not np.isclose(
        trial.source_redshift,
        mass_context.z_source,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("trial source redshift does not match mass_context")
    cosmology_name = full_config.get("lensing", {}).get("cosmology")
    if (
        cosmology_name is not None
        and cosmology_name != mass_context.cosmology_name
    ):
        raise ValueError("configuration cosmology does not match mass_context")
    return mass_context


def subhalo_model_spec_from_trial(
    full_config: Dict[str, Any],
    trial: SubhaloTrial,
    priors_config: Optional[Dict[str, Any]] = None,
    fit_mode: str = "fixed_template",
    *,
    mass_context: Optional[MassMappingContext] = None,
    clumpy_fit_parameterization: str = "host_free",
) -> ModelSpec:
    """Build a source-neutral subhalo-model specification.

    Parameters
    ----------
    full_config : `dict`
        Full HWO-SLAPS configuration.
    trial : `SubhaloTrial`
        Trial subhalo to include in the model.
    priors_config : `dict`, optional
        Prior-width overrides.
    fit_mode : `str`, optional
        ``"fixed_template"``, ``"local_search"``, or ``"freed"``.
    mass_context : `MassMappingContext`, optional
        Required only for freed fits.
    clumpy_fit_parameterization : `str`, optional
        Clumpy-source fit parameterization.

    Returns
    -------
    spec : `ModelSpec`
        Subhalo model specification.
    """
    if fit_mode not in {"fixed_template", "local_search", "freed"}:
        raise ValueError(
            "fit_mode must be 'fixed_template', 'local_search', or 'freed'"
        )
    if fit_mode != "freed" and mass_context is not None:
        raise ValueError("mass_context must be None for legacy fit modes")

    spec = smooth_model_spec_from_config(
        full_config,
        priors_config=priors_config,
        clumpy_fit_parameterization=clumpy_fit_parameterization,
    )
    galaxies = dict(spec.galaxies)
    y0, x0 = trial.position_yx_arcsec
    widths = _widths(priors_config)

    if fit_mode == "freed":
        mass_context = _validate_freed_context(
            full_config,
            trial,
            mass_context,
        )
        window = widths["subhalo_freed_centre_window_arcsec"]
        parameters = {
            "centre_0": _uniform_around(y0, window),
            "centre_1": _uniform_around(x0, window),
            "log10_m200": uniform(
                mass_context.log10_m200_lower,
                mass_context.log10_m200_upper,
            ),
            "mapping_context": fixed(mass_context),
        }
        class_name = {
            "NFW": "NFWMCRSubhaloSph",
            "SIS": "SISMCRSubhalo",
            "PointMass": "PointMassMCRSubhalo",
        }[trial.model]
    else:
        if fit_mode == "fixed_template":
            centre_0 = fixed(float(y0))
            centre_1 = fixed(float(x0))
        else:
            window = widths["subhalo_centre_window_arcsec"]
            centre_0 = _uniform_around(y0, window)
            centre_1 = _uniform_around(x0, window)
        parameters = {
            "centre_0": centre_0,
            "centre_1": centre_1,
        }
        class_name = trial.profile_class
        if trial.profile_class == "NFWSph":
            if trial.kappa_s is None or trial.scale_radius_arcsec is None:
                raise ValueError(
                    "NFWSph validation requires kappa_s and "
                    "scale_radius_arcsec"
                )
            parameters["kappa_s"] = fixed(float(trial.kappa_s))
            parameters["scale_radius"] = fixed(
                float(trial.scale_radius_arcsec)
            )
        elif trial.profile_class in {"PointMass", "IsothermalSph"}:
            if trial.einstein_radius_arcsec is None:
                raise ValueError(
                    f"{trial.profile_class} validation requires "
                    "einstein_radius_arcsec"
                )
            parameters["einstein_radius"] = fixed(
                float(trial.einstein_radius_arcsec)
            )
        else:
            raise ValueError(
                f"Unsupported subhalo profile class: {trial.profile_class}"
            )

    lens_galaxy = galaxies["lens"]
    lens_components = dict(lens_galaxy.components)
    lens_components["subhalo"] = ProfileSpec(
        class_name=class_name,
        parameters=parameters,
    )
    galaxies["lens"] = GalaxySpec(
        name=lens_galaxy.name,
        redshift=lens_galaxy.redshift,
        components=lens_components,
    )

    metadata = dict(spec.metadata)
    metadata.update({
        "builder": "subhalo_model_spec_from_trial",
        "trial_case_id": trial.case_id,
        "mass_profile_source": "hwo_slaps_forward_model",
    })
    if fit_mode == "freed":
        metadata.update(
            {
                "builder": "subhalo_model_spec_from_trial",
                "trial_case_id": trial.case_id,
                "mass_profile_source": "mass_mapping_context",
                "mass_context_hash": mass_context.context_hash,
                "requires_cpu": True,
            }
        )
    return ModelSpec(
        model_type="subhalo",
        fit_mode=fit_mode,
        galaxies=galaxies,
        metadata=metadata,
    )


def fixed_point_model_spec_from_trial(
    full_config: Dict[str, Any],
    trial: SubhaloTrial,
    priors_config: Optional[Dict[str, Any]] = None,
    clumpy_fit_parameterization: str = "host_free",
) -> ModelSpec:
    """Build a fixed-template-shaped model pinned to all target values.

    Parameters
    ----------
    full_config : `dict`
        Full HWO-SLAPS configuration.
    trial : `SubhaloTrial`
        Trial subhalo fixed at its configured mass-profile scales and centre.
    priors_config : `dict`, optional
        Prior-width overrides used to validate every target's physical box.
    clumpy_fit_parameterization : `str`, optional
        Clumpy-source fit parameterization.

    Returns
    -------
    spec : `ModelSpec`
        Fixed-template-shaped specification with no free parameters.
    """
    pinned_smooth = smooth_model_spec_from_config(
        full_config,
        priors_config=priors_config,
        clumpy_fit_parameterization=clumpy_fit_parameterization,
        pin_to_targets=True,
    )
    fixed_template = subhalo_model_spec_from_trial(
        full_config,
        trial,
        priors_config=priors_config,
        fit_mode="fixed_template",
        clumpy_fit_parameterization=clumpy_fit_parameterization,
    )
    galaxies = dict(pinned_smooth.galaxies)
    lens_galaxy = galaxies["lens"]
    lens_components = dict(lens_galaxy.components)
    lens_components["subhalo"] = fixed_template.galaxies[
        "lens"
    ].components["subhalo"]
    galaxies["lens"] = GalaxySpec(
        name=lens_galaxy.name,
        redshift=lens_galaxy.redshift,
        components=lens_components,
    )
    metadata = dict(fixed_template.metadata)
    metadata["builder"] = "fixed_point_model_spec_from_trial"
    return ModelSpec(
        model_type="subhalo",
        fit_mode="fixed_template",
        galaxies=galaxies,
        metadata=metadata,
    )


def _prior_to_autofit(prior: PriorSpec) -> Any:
    """Convert a source-neutral prior to a PyAutoFit prior or value."""
    import autofit as af

    if prior.kind == "fixed":
        return prior.value
    if prior.kind == "uniform":
        return af.UniformPrior(
            lower_limit=float(prior.lower),
            upper_limit=float(prior.upper),
        )
    if prior.kind == "log_uniform":
        return af.LogUniformPrior(
            lower_limit=float(prior.lower),
            upper_limit=float(prior.upper),
        )
    if prior.kind == "linked":
        raise ValueError("linked priors must be resolved after model creation")
    raise ValueError(f"Unsupported prior kind: {prior.kind}")


def _profile_class(class_name: str) -> Any:
    """Resolve a PyAutoLens profile class by supported name."""
    import autolens as al

    from ...lensing.image_source import ImageSource
    from .clumpy_profiles import ClumpyTransformedSource
    from .mass_mapping import (
        NFWMCRSubhaloSph,
        PointMassMCRSubhalo,
        SISMCRSubhalo,
    )

    classes = {
        "Isothermal": al.mp.Isothermal,
        "PowerLaw": al.mp.PowerLaw,
        "PowerLawMultipole": al.mp.PowerLawMultipole,
        "ExternalShear": al.mp.ExternalShear,
        "PointMass": al.mp.PointMass,
        "IsothermalSph": al.mp.IsothermalSph,
        "NFWSph": al.mp.NFWSph,
        "Exponential": al.lp.Exponential,
        "Sersic": al.lp.Sersic,
        "ImageSource": ImageSource,
        "ClumpyTransformedSource": ClumpyTransformedSource,
        "NFWMCRSubhaloSph": NFWMCRSubhaloSph,
        "SISMCRSubhalo": SISMCRSubhalo,
        "PointMassMCRSubhalo": PointMassMCRSubhalo,
    }
    if class_name not in classes:
        raise ValueError(
            f"Unsupported PyAutoLens profile class: {class_name}"
        )
    return classes[class_name]


def _assign_profile_value(profile_model: Any, name: str, value: Any) -> None:
    """Assign a scalar or tuple-child value to a PyAutoFit model."""
    if name == "centre_0":
        profile_model.centre.centre_0 = value
    elif name == "centre_1":
        profile_model.centre.centre_1 = value
    elif name == "ell_comps_0":
        profile_model.ell_comps.ell_comps_0 = value
    elif name == "ell_comps_1":
        profile_model.ell_comps.ell_comps_1 = value
    elif name == "multipole_comps_0":
        profile_model.multipole_comps.multipole_comps_0 = value
    elif name == "multipole_comps_1":
        profile_model.multipole_comps.multipole_comps_1 = value
    elif name == "host_ell_comps_0":
        profile_model.host_ell_comps.host_ell_comps_0 = value
    elif name == "host_ell_comps_1":
        profile_model.host_ell_comps.host_ell_comps_1 = value
    else:
        setattr(profile_model, name, value)


def _assign_profile_parameter(
    profile_model: Any,
    name: str,
    prior: PriorSpec,
) -> None:
    """Assign a scalar or tuple profile parameter to a PyAutoFit model."""
    _assign_profile_value(profile_model, name, _prior_to_autofit(prior))


def _profile_parameter_value(profile_model: Any, name: str) -> Any:
    """Return a scalar or tuple-child value from a PyAutoFit model."""
    if name == "centre_0":
        return profile_model.centre.centre_0
    if name == "centre_1":
        return profile_model.centre.centre_1
    if name == "ell_comps_0":
        return profile_model.ell_comps.ell_comps_0
    if name == "ell_comps_1":
        return profile_model.ell_comps.ell_comps_1
    if name == "multipole_comps_0":
        return profile_model.multipole_comps.multipole_comps_0
    if name == "multipole_comps_1":
        return profile_model.multipole_comps.multipole_comps_1
    if name == "host_ell_comps_0":
        return profile_model.host_ell_comps.host_ell_comps_0
    if name == "host_ell_comps_1":
        return profile_model.host_ell_comps.host_ell_comps_1
    return getattr(profile_model, name)


def _profile_model_from_spec(profile_spec: ProfileSpec) -> Any:
    """Build one profile model, including fixed image-asset arguments."""
    import autofit as af

    profile_class = _profile_class(profile_spec.class_name)
    if profile_spec.class_name == "ImageSource":
        asset_prior = profile_spec.parameters["asset"]
        if asset_prior.kind != "fixed":
            raise ValueError("ImageSource asset must be fixed")
        asset = asset_prior.value
        return af.Model(
            profile_class,
            pixel_scale_arcsec=float(asset.pixel_scale_arcsec),
            sb=asset.sb,
        )
    return af.Model(profile_class)


def autofit_model_from_spec(spec: ModelSpec) -> Any:
    """Convert a source-neutral model specification to a PyAutoFit model.

    Parameters
    ----------
    spec : `ModelSpec`
        Source-neutral model specification.

    Returns
    -------
    model : `autofit.Collection`
        PyAutoFit model collection with same-galaxy prior links resolved.
    """
    import autofit as af
    import autolens as al

    galaxy_models = {}
    for galaxy_name, galaxy_spec in spec.galaxies.items():
        component_models = {}
        for component_name, profile_spec in galaxy_spec.components.items():
            profile_model = _profile_model_from_spec(profile_spec)
            for parameter_name, prior in profile_spec.parameters.items():
                if prior.kind == "linked" or parameter_name == "asset":
                    continue
                _assign_profile_parameter(
                    profile_model,
                    parameter_name,
                    prior,
                )
            component_models[component_name] = profile_model

        for component_name, profile_spec in galaxy_spec.components.items():
            target_model = component_models[component_name]
            for parameter_name, prior in profile_spec.parameters.items():
                if prior.kind != "linked":
                    continue
                reference_component, reference_parameter = prior.value
                if reference_component not in component_models:
                    raise ValueError(
                        "linked prior must reference a component in the "
                        "same GalaxySpec"
                    )
                value = _profile_parameter_value(
                    component_models[reference_component],
                    reference_parameter,
                )
                _assign_profile_value(target_model, parameter_name, value)

        galaxy_models[galaxy_name] = af.Model(
            al.Galaxy,
            redshift=_prior_to_autofit(galaxy_spec.redshift),
            **component_models,
        )

    return af.Collection(galaxies=af.Collection(**galaxy_models))
