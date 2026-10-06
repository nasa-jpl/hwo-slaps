"""Public engine values and operations, imported when requested."""
from importlib import import_module

_PUBLIC_API = {
    "summarize": ("reductions", "summarize"),
    "ForecastSummary": ("reductions", "ForecastSummary"),
    "aperture_selection": ("reductions", "aperture_selection"),
    "crossing": ("reach", "crossing"),
    "mass_reach": ("reach", "mass_reach"),
    "MassReach": ("reach", "MassReach"),
    "adaptive_mass_reach": ("reach", "adaptive_mass_reach"),
    "ElectronMaps": ("selection", "ElectronMaps"),
    "electron_maps": ("selection", "electron_maps"),
    "aperture_mask": ("selection", "aperture_mask"),
    "arc_snr": ("selection", "arc_snr"),
    "gradient_power": ("selection", "gradient_power"),
    "diffraction_scale_arcsec": ("selection", "diffraction_scale_arcsec"),
    "complexity": ("selection", "complexity"),
    "standardize": ("selection", "standardize"),
    "rank": ("selection", "rank"),
    "spearman_rank_correlation": ("selection", "spearman_rank_correlation"),
    "top_k_jaccard": ("selection", "top_k_jaccard"),
    "top_k_recovery": ("selection", "top_k_recovery"),
    "Cut": ("selection", "Cut"),
    "ScoreTerm": ("selection", "ScoreTerm"),
    "RankingPolicy": ("selection", "RankingPolicy"),
    "RankingResult": ("selection", "RankingResult"),
    "rank_pool": ("selection", "rank_pool"),
    "AgreementTable": ("nonlinear", "AgreementTable"),
    "AttemptSelection": ("nonlinear", "AttemptSelection"),
    "CaseClassification": ("nonlinear", "CaseClassification"),
    "CaseStatus": ("nonlinear", "CaseStatus"),
    "ClassificationRule": ("nonlinear", "ClassificationRule"),
    "RoleAcceptance": ("nonlinear", "RoleAcceptance"),
    "StatusResult": ("nonlinear", "StatusResult"),
    "case_status": ("nonlinear", "case_status"),
    "classify_case": ("nonlinear", "classify_case"),
    "detection_agreement": ("nonlinear", "detection_agreement"),
    "select_attempt": ("nonlinear", "select_attempt"),
}
__all__ = list(_PUBLIC_API)


def __getattr__(name):
    if name not in _PUBLIC_API:
        raise AttributeError(name)
    module, member = _PUBLIC_API[name]
    value = getattr(import_module(f".{module}", __name__), member)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
