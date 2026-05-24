"""Build PyAutoFit models for nonlinear validation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .model_specs import (
    GalaxySpec,
    ModelSpec,
    PriorSpec,
    ProfileSpec,
    fixed,
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
    return uniform(float(value) - float(half_width), float(value) + float(half_width))


def _uniform_fraction(value: float, frac_width: float) -> PriorSpec:
    """Return a positive uniform prior around a scalar value."""
    value = float(value)
    half_width = abs(value)*float(frac_width)
    lower = max(0.0, value - half_width)
    upper = value + half_width
    if lower == upper:
        upper = lower + 1.0e-12
    return uniform(lower, upper)


def smooth_model_spec_from_config(
    full_config: Dict[str, Any],
    priors_config: Optional[Dict[str, Any]] = None,
) -> ModelSpec:
    """Build a source-neutral smooth-model specification from config.

    Parameters
    ----------
    full_config : `dict`
        Full HWO-SLAPS configuration.
    priors_config : `dict`, optional
        Prior-width overrides.

    Returns
    -------
    spec : `ModelSpec`
        Smooth lens/source model specification.
    """
    widths = _widths(priors_config)
    lens_config = full_config["lensing"]["lens_galaxy"]
    source_config = full_config["lensing"]["source_galaxy"]
    lens_mass_config = lens_config["mass"]
    source_light_config = source_config["light"]

    lens_centre = lens_mass_config["centre"]
    lens_ell_comps = lens_mass_config["ell_comps"]
    source_centre = source_light_config["centre"]
    source_ell_comps = source_light_config["ell_comps"]

    lens_mass = ProfileSpec(
        class_name="Isothermal",
        parameters={
            "centre_0": _uniform_around(
                lens_centre[0],
                widths["lens_centre_sigma_arcsec"],
            ),
            "centre_1": _uniform_around(
                lens_centre[1],
                widths["lens_centre_sigma_arcsec"],
            ),
            "einstein_radius": _uniform_around(
                lens_mass_config["einstein_radius"],
                widths["lens_einstein_radius_sigma"],
            ),
            "ell_comps_0": _uniform_around(
                lens_ell_comps[0],
                widths["lens_ell_comps_sigma"],
            ),
            "ell_comps_1": _uniform_around(
                lens_ell_comps[1],
                widths["lens_ell_comps_sigma"],
            ),
        },
    )
    source_light = ProfileSpec(
        class_name="Exponential",
        parameters={
            "centre_0": _uniform_around(
                source_centre[0],
                widths["source_centre_sigma_arcsec"],
            ),
            "centre_1": _uniform_around(
                source_centre[1],
                widths["source_centre_sigma_arcsec"],
            ),
            "ell_comps_0": _uniform_around(
                source_ell_comps[0],
                widths["source_ell_comps_sigma"],
            ),
            "ell_comps_1": _uniform_around(
                source_ell_comps[1],
                widths["source_ell_comps_sigma"],
            ),
            "intensity": _uniform_fraction(
                source_light_config["intensity"],
                widths["source_intensity_frac_sigma"],
            ),
            "effective_radius": _uniform_fraction(
                source_light_config["effective_radius"],
                widths["source_effective_radius_frac_sigma"],
            ),
        },
    )

    return ModelSpec(
        model_type="smooth",
        fit_mode="smooth",
        galaxies={
            "lens": GalaxySpec(
                name="lens",
                redshift=fixed(float(lens_config["redshift"])),
                components={"mass": lens_mass},
            ),
            "source": GalaxySpec(
                name="source",
                redshift=fixed(float(source_config["redshift"])),
                components={"light": source_light},
            ),
        },
        metadata={"builder": "smooth_model_spec_from_config"},
    )


def subhalo_model_spec_from_trial(
    full_config: Dict[str, Any],
    trial: SubhaloTrial,
    priors_config: Optional[Dict[str, Any]] = None,
    fit_mode: str = "fixed_template",
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
        Validation mode. Supported values are ``"fixed_template"`` and
        ``"local_search"``.

    Returns
    -------
    spec : `ModelSpec`
        Subhalo model specification.
    """
    if fit_mode not in {"fixed_template", "local_search"}:
        raise ValueError("fit_mode must be 'fixed_template' or 'local_search'")

    spec = smooth_model_spec_from_config(full_config, priors_config=priors_config)
    galaxies = dict(spec.galaxies)
    y0, x0 = trial.position_yx_arcsec
    widths = _widths(priors_config)

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
    if trial.profile_class == "NFWSph":
        if trial.kappa_s is None or trial.scale_radius_arcsec is None:
            raise ValueError("NFWSph validation requires kappa_s and scale_radius_arcsec")
        parameters["kappa_s"] = fixed(float(trial.kappa_s))
        parameters["scale_radius"] = fixed(float(trial.scale_radius_arcsec))
    elif trial.profile_class in {"PointMass", "IsothermalSph"}:
        if trial.einstein_radius_arcsec is None:
            raise ValueError(f"{trial.profile_class} validation requires einstein_radius_arcsec")
        parameters["einstein_radius"] = fixed(float(trial.einstein_radius_arcsec))
    else:
        raise ValueError(f"Unsupported subhalo profile class: {trial.profile_class}")

    lens_galaxy = galaxies["lens"]
    lens_components = dict(lens_galaxy.components)
    lens_components["subhalo"] = ProfileSpec(
        class_name=trial.profile_class,
        parameters=parameters,
    )
    galaxies["lens"] = GalaxySpec(
        name=lens_galaxy.name,
        redshift=lens_galaxy.redshift,
        components=lens_components,
    )

    return ModelSpec(
        model_type="subhalo",
        fit_mode=fit_mode,
        galaxies=galaxies,
        metadata={
            "builder": "subhalo_model_spec_from_trial",
            "trial_case_id": trial.case_id,
            "mass_profile_source": "hwo_slaps_forward_model",
        },
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
    raise ValueError(f"Unsupported prior kind: {prior.kind}")


def _profile_class(class_name: str) -> Any:
    """Resolve a PyAutoLens profile class by supported name."""
    import autolens as al

    classes = {
        "Isothermal": al.mp.Isothermal,
        "PointMass": al.mp.PointMass,
        "IsothermalSph": al.mp.IsothermalSph,
        "NFWSph": al.mp.NFWSph,
        "Exponential": al.lp.Exponential,
    }
    if class_name not in classes:
        raise ValueError(f"Unsupported PyAutoLens profile class: {class_name}")
    return classes[class_name]


def _assign_profile_parameter(profile_model: Any, name: str, prior: PriorSpec) -> None:
    """Assign a scalar or tuple profile parameter to a PyAutoFit model."""
    value = _prior_to_autofit(prior)
    if name == "centre_0":
        profile_model.centre.centre_0 = value
    elif name == "centre_1":
        profile_model.centre.centre_1 = value
    elif name == "ell_comps_0":
        profile_model.ell_comps.ell_comps_0 = value
    elif name == "ell_comps_1":
        profile_model.ell_comps.ell_comps_1 = value
    else:
        setattr(profile_model, name, value)


def autofit_model_from_spec(spec: ModelSpec) -> Any:
    """Convert a source-neutral model specification to a PyAutoFit model.

    Parameters
    ----------
    spec : `ModelSpec`
        Source-neutral model specification.

    Returns
    -------
    model : `autofit.Collection`
        PyAutoFit model collection.
    """
    import autofit as af
    import autolens as al

    galaxy_models = {}
    for galaxy_name, galaxy_spec in spec.galaxies.items():
        component_models = {}
        for component_name, profile_spec in galaxy_spec.components.items():
            profile_model = af.Model(_profile_class(profile_spec.class_name))
            for parameter_name, prior in profile_spec.parameters.items():
                _assign_profile_parameter(profile_model, parameter_name, prior)
            component_models[component_name] = profile_model

        galaxy_models[galaxy_name] = af.Model(
            al.Galaxy,
            redshift=_prior_to_autofit(galaxy_spec.redshift),
            **component_models,
        )

    return af.Collection(galaxies=af.Collection(**galaxy_models))
