"""Subhalo-trial containers for nonlinear validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

PROFILE_CLASS_BY_MODEL = {
    "PointMass": "PointMass",
    "SIS": "IsothermalSph",
    "NFW": "NFWSph",
}
"""Mapping from HWO-SLAPS subhalo labels to PyAutoLens profile names."""


@dataclass(frozen=True)
class SubhaloTrial:
    """Physical subhalo trial being compared against Fisher.

    Parameters
    ----------
    case_id : `str`
        Stable identifier for this validation case.
    mass_msun : `float`
        Subhalo mass in solar masses.
    position_yx_arcsec : `tuple` [`float`, `float`]
        Subhalo center in PyAutoLens ``(y, x)`` arcsecond coordinates.
    model : `str`
        HWO-SLAPS subhalo model label.
    profile_class : `str`
        PyAutoLens mass-profile class name.
    lens_redshift : `float`
        Lens-plane redshift.
    source_redshift : `float`
        Source-plane redshift.
    einstein_radius_arcsec : `float`, optional
        Einstein radius for PointMass and SIS validation trials.
    kappa_s : `float`, optional
        NFW scale convergence for ``NFWSph`` validation trials.
    scale_radius_arcsec : `float`, optional
        NFW scale radius in arcseconds for ``NFWSph`` trials.
    concentration : `float`, optional
        NFW concentration used to generate the trial.
    concentration_model : `str`, optional
        Name of the concentration relation.
    fisher_q : `float`, optional
        Fisher local statistic for the same trial.
    fisher_z : `float`, optional
        Fisher local significance for the same trial.
    fisher_delta_log_l_equiv : `float`, optional
        Fisher-equivalent ``Delta log L = q_F/2``.
    metadata : `dict`, optional
        Additional provenance.
    """

    case_id: str
    mass_msun: float
    position_yx_arcsec: Tuple[float, float]
    model: str
    profile_class: str
    lens_redshift: float
    source_redshift: float
    einstein_radius_arcsec: Optional[float] = None
    kappa_s: Optional[float] = None
    scale_radius_arcsec: Optional[float] = None
    concentration: Optional[float] = None
    concentration_model: Optional[str] = None
    fisher_q: Optional[float] = None
    fisher_z: Optional[float] = None
    fisher_delta_log_l_equiv: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the trial to a JSON-compatible dictionary.

        Returns
        -------
        data : `dict`
            Dictionary representation of the trial.
        """
        return asdict(self)


def _case_id_from_trial(model: str, mass_msun: float, position_yx_arcsec: Tuple[float, float]) -> str:
    """Build a deterministic case identifier from trial coordinates."""
    y_arcsec, x_arcsec = position_yx_arcsec
    return f"{model}_m{mass_msun:.6e}_y{y_arcsec:+.5f}_x{x_arcsec:+.5f}"


def _profile_class_from_model(model: str) -> str:
    """Map a supported HWO-SLAPS subhalo model to a profile class."""
    if model not in PROFILE_CLASS_BY_MODEL:
        raise ValueError(f"Unsupported subhalo model for nonlinear validation: {model}")
    return PROFILE_CLASS_BY_MODEL[model]


def trial_from_lensing_truth(lensing_data: Any, case_id: Optional[str] = None) -> SubhaloTrial:
    """Create a nonlinear validation trial from generated lensing truth.

    Parameters
    ----------
    lensing_data : `object`
        Object with the HWO-SLAPS ``LensingData`` attributes.
    case_id : `str`, optional
        Explicit case identifier. If omitted, one is generated from the
        model, mass, and position.

    Returns
    -------
    trial : `SubhaloTrial`
        Validation trial matching the injected subhalo.

    Raises
    ------
    ValueError
        Raised when the lensing data do not contain a subhalo.
    """
    if not getattr(lensing_data, "has_subhalo", False):
        raise ValueError("Cannot build SubhaloTrial from lensing data without a subhalo")

    model = getattr(lensing_data, "subhalo_model")
    mass_msun = float(getattr(lensing_data, "subhalo_mass"))
    position_yx_arcsec = tuple(getattr(lensing_data, "subhalo_position"))
    if case_id is None:
        case_id = _case_id_from_trial(model, mass_msun, position_yx_arcsec)

    fisher_q = None
    fisher_z = None
    fisher_delta_log_l_equiv = None

    metadata = {
        "source": "lensing_truth",
        "concentration_source": getattr(lensing_data, "subhalo_concentration_source", None),
        "concentration_x_sub": getattr(lensing_data, "subhalo_concentration_x_sub", None),
        "concentration_h": getattr(lensing_data, "subhalo_concentration_h", None),
    }

    return SubhaloTrial(
        case_id=case_id,
        mass_msun=mass_msun,
        position_yx_arcsec=(float(position_yx_arcsec[0]), float(position_yx_arcsec[1])),
        model=model,
        profile_class=_profile_class_from_model(model),
        lens_redshift=float(getattr(lensing_data, "lens_redshift")),
        source_redshift=float(getattr(lensing_data, "source_redshift")),
        einstein_radius_arcsec=getattr(lensing_data, "subhalo_einstein_radius", None),
        kappa_s=getattr(lensing_data, "subhalo_kappa_s", None),
        scale_radius_arcsec=getattr(lensing_data, "subhalo_scale_radius_arcsec", None),
        concentration=getattr(lensing_data, "subhalo_concentration", None),
        concentration_model=getattr(lensing_data, "subhalo_concentration_model", None),
        fisher_q=fisher_q,
        fisher_z=fisher_z,
        fisher_delta_log_l_equiv=fisher_delta_log_l_equiv,
        metadata=metadata,
    )


def trial_from_fisher_map_position(
    full_config: Dict[str, Any],
    lensing_reference: Any,
    mass_msun: float,
    position_yx_arcsec: Tuple[float, float],
    fisher_q: Optional[float] = None,
    case_id: Optional[str] = None,
) -> SubhaloTrial:
    """Create a validation trial from a Fisher-map position.

    Parameters
    ----------
    full_config : `dict`
        Full HWO-SLAPS configuration containing the subhalo model label.
    lensing_reference : `object`
        Reference lensing data for redshifts and optional truth metadata.
    mass_msun : `float`
        Trial mass in solar masses.
    position_yx_arcsec : `tuple` [`float`, `float`]
        Trial position in arcseconds.
    fisher_q : `float`, optional
        Fisher statistic at this map position.
    case_id : `str`, optional
        Explicit case identifier.

    Returns
    -------
    trial : `SubhaloTrial`
        Validation trial. NFW scale parameters are populated only if they
        are available on ``lensing_reference``.
    """
    subhalo_config = full_config.get("lensing", {}).get("subhalo", {})
    model = subhalo_config.get("model", getattr(lensing_reference, "subhalo_model", None))
    if model is None:
        raise ValueError("Cannot infer subhalo model for Fisher-map trial")

    position_yx_arcsec = (float(position_yx_arcsec[0]), float(position_yx_arcsec[1]))
    mass_msun = float(mass_msun)
    if case_id is None:
        case_id = _case_id_from_trial(model, mass_msun, position_yx_arcsec)

    fisher_z = fisher_q**0.5 if fisher_q is not None and fisher_q >= 0.0 else None
    fisher_delta = 0.5*fisher_q if fisher_q is not None else None

    return SubhaloTrial(
        case_id=case_id,
        mass_msun=mass_msun,
        position_yx_arcsec=position_yx_arcsec,
        model=model,
        profile_class=_profile_class_from_model(model),
        lens_redshift=float(getattr(lensing_reference, "lens_redshift")),
        source_redshift=float(getattr(lensing_reference, "source_redshift")),
        einstein_radius_arcsec=getattr(lensing_reference, "subhalo_einstein_radius", None),
        kappa_s=getattr(lensing_reference, "subhalo_kappa_s", None),
        scale_radius_arcsec=getattr(lensing_reference, "subhalo_scale_radius_arcsec", None),
        concentration=getattr(lensing_reference, "subhalo_concentration", None),
        concentration_model=getattr(lensing_reference, "subhalo_concentration_model", None),
        fisher_q=fisher_q,
        fisher_z=fisher_z,
        fisher_delta_log_l_equiv=fisher_delta,
        metadata={"source": "fisher_map"},
    )
