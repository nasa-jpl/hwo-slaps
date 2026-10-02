"""Scalar nuisance specifications and selection independent of rendering.

A specification identifies a config path, finite-difference convention, and
optional Gaussian prior. Selection operates on any specification list, so a
caller can plan additional source or instrument parameters without creating a
detector or importing AutoLens. Image rendering remains the detector's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


# Reserved ``modeling.fisher.nuisance_subset`` words that select scalar
# nuisance directions by name prefix.  ``all`` and ``none`` are handled
# separately because they are not prefix selections.
_NUISANCE_SUBSET_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "lens_only": ("lens.",),
    "source_only": ("source.",),
    "lens_and_source": ("lens.", "source."),
}


@dataclass(frozen=True)
class ScalarNuisanceSpec:
    """Descriptor for one scalar nuisance parameter."""

    name: str
    path: Optional[Tuple[Any, ...]]
    step_mode: str
    step_key: Optional[str] = None
    prior_sigma: Optional[float] = None


def lookup_prior_sigma(prior_sigmas: Mapping[str, Any], name: str) -> Optional[float]:
    """Resolve and validate one optional scalar Gaussian prior sigma."""
    value = prior_sigmas.get(name)
    if value is None:
        return None
    sigma = float(value)
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError(f"Invalid prior sigma for {name}: {sigma}")
    return sigma


def build_scalar_nuisance_specs(
    source_light_config: Mapping[str, Any],
    prior_sigmas: Mapping[str, Any],
    *,
    include_background_offset: bool,
) -> List[ScalarNuisanceSpec]:
    """Describe the built-in scalar directions for the source-light schema.

    Image sources have no ellipticity parameters. Their ``source.intensity``
    and ``source.effective_radius`` labels perturb dimensionless flux/size
    scales, so the corresponding prior sigmas retain that fractional meaning.
    Parametric source directions retain their own parameter units.
    """

    def prior_sigma(name: str) -> Optional[float]:
        return lookup_prior_sigma(prior_sigmas, name)

    light_type = source_light_config["type"]
    light_root = ("lensing", "source_galaxy", "light")
    centre_root = light_root + ("centre",)
    ell_comps_root = light_root + ("ell_comps",)
    if light_type == "Image":
        intensity_path = light_root + ("flux_scale",)
        effective_radius_path = light_root + ("size_scale",)
    else:
        intensity_path = light_root + ("intensity",)
        effective_radius_path = light_root + ("effective_radius",)

    specs = [
        ScalarNuisanceSpec(
            name="lens.centre_y",
            path=("lensing", "lens_galaxy", "mass", "centre", 0),
            step_mode="additive",
            step_key="centre_arcsec",
            prior_sigma=prior_sigma("lens.centre_y"),
        ),
        ScalarNuisanceSpec(
            name="lens.centre_x",
            path=("lensing", "lens_galaxy", "mass", "centre", 1),
            step_mode="additive",
            step_key="centre_arcsec",
            prior_sigma=prior_sigma("lens.centre_x"),
        ),
        ScalarNuisanceSpec(
            name="lens.einstein_radius",
            path=("lensing", "lens_galaxy", "mass", "einstein_radius"),
            step_mode="additive",
            step_key="einstein_radius_arcsec",
            prior_sigma=prior_sigma("lens.einstein_radius"),
        ),
        ScalarNuisanceSpec(
            name="lens.ell_comp_1",
            path=("lensing", "lens_galaxy", "mass", "ell_comps", 0),
            step_mode="additive",
            step_key="ell_comp",
            prior_sigma=prior_sigma("lens.ell_comp_1"),
        ),
        ScalarNuisanceSpec(
            name="lens.ell_comp_2",
            path=("lensing", "lens_galaxy", "mass", "ell_comps", 1),
            step_mode="additive",
            step_key="ell_comp",
            prior_sigma=prior_sigma("lens.ell_comp_2"),
        ),
    ]
    specs.extend(
        [
            ScalarNuisanceSpec(
                name="source.centre_y",
                path=centre_root + (0,),
                step_mode="additive",
                step_key="centre_arcsec",
                prior_sigma=prior_sigma("source.centre_y"),
            ),
            ScalarNuisanceSpec(
                name="source.centre_x",
                path=centre_root + (1,),
                step_mode="additive",
                step_key="centre_arcsec",
                prior_sigma=prior_sigma("source.centre_x"),
            ),
        ]
    )
    if light_type != "Image":
        specs.extend(
            [
                ScalarNuisanceSpec(
                    name="source.ell_comp_1",
                    path=ell_comps_root + (0,),
                    step_mode="additive",
                    step_key="ell_comp",
                    prior_sigma=prior_sigma("source.ell_comp_1"),
                ),
                ScalarNuisanceSpec(
                    name="source.ell_comp_2",
                    path=ell_comps_root + (1,),
                    step_mode="additive",
                    step_key="ell_comp",
                    prior_sigma=prior_sigma("source.ell_comp_2"),
                ),
            ]
        )
    specs.extend(
        [
            ScalarNuisanceSpec(
                name="source.intensity",
                path=intensity_path,
                step_mode="multiplicative",
                step_key="source_intensity_frac",
                prior_sigma=prior_sigma("source.intensity"),
            ),
            ScalarNuisanceSpec(
                name="source.effective_radius",
                path=effective_radius_path,
                step_mode="multiplicative",
                step_key="source_reff_frac",
                prior_sigma=prior_sigma("source.effective_radius"),
            ),
        ]
    )
    if include_background_offset:
        specs.append(
            ScalarNuisanceSpec(
                name="observation.background_offset_adu",
                path=None,
                step_mode="additive",
                step_key=None,
                prior_sigma=prior_sigma("observation.background_offset_adu"),
            )
        )
    return specs


def select_scalar_nuisances(
    specs: List[ScalarNuisanceSpec], selector: Any
) -> Tuple[List[ScalarNuisanceSpec], str]:
    """Select scalar directions in canonical input order and label provenance.

    Reserved selectors choose lens/source prefixes, ``all``, or ``none``.
    Explicit names can select any available scalar specification. PSF modes
    retain their separate basis-selection contract and cannot be named here.
    """
    if selector is None:
        return specs, "all"

    reserved = sorted({"all", "none", *_NUISANCE_SUBSET_PREFIXES})
    if isinstance(selector, str):
        label = selector.strip().lower()
        if label == "all":
            return specs, label
        if label == "none":
            return [], label
        prefixes = _NUISANCE_SUBSET_PREFIXES.get(label)
        if prefixes is None:
            raise ValueError(
                "modeling.fisher.nuisance_subset must be one of "
                f"{reserved}, or a list of nuisance direction names; "
                f"got {selector!r}"
            )
        return [spec for spec in specs if spec.name.startswith(prefixes)], label

    if not isinstance(selector, (list, tuple)):
        raise ValueError(
            "modeling.fisher.nuisance_subset must be one of "
            f"{reserved}, or a list of nuisance direction names; "
            f"got {selector!r}"
        )
    if len(selector) == 0:
        raise ValueError(
            "modeling.fisher.nuisance_subset must be non-empty when given "
            "as a list; use 'none' to profile no nuisance directions."
        )

    available = {spec.name for spec in specs}
    requested = set()
    for entry in selector:
        if not isinstance(entry, str):
            raise ValueError(
                "modeling.fisher.nuisance_subset entries must be nuisance "
                f"direction names; got {entry!r}"
            )
        name = entry.strip()
        if name.startswith("psf."):
            raise ValueError(
                "modeling.fisher.nuisance_subset must not name PSF modes "
                f"({name!r}); PSF nuisance directions are governed by "
                "modeling.fisher.include_psf_nuisance and "
                "modeling.fisher.fit_psf_mode_selection."
            )
        if name not in available:
            raise ValueError(
                f"modeling.fisher.nuisance_subset names unknown direction "
                f"{name!r}. Valid directions for this scene are: "
                f"{sorted(available)}"
            )
        if name in requested:
            raise ValueError(
                "modeling.fisher.nuisance_subset contains duplicate "
                f"direction {name!r}."
            )
        requested.add(name)
    return [spec for spec in specs if spec.name in requested], "explicit"
