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
    "BinomialCount": ("binomial", "BinomialCount"),
    "clopper_pearson": ("binomial", "clopper_pearson"),
    "knowledge_error_areas": ("knowledge_error", "knowledge_error_areas"),
    "KnowledgeErrorAreas": ("knowledge_error", "KnowledgeErrorAreas"),
    "knowledge_error_tolerance": ("knowledge_error", "knowledge_error_tolerance"),
    "ToleranceCriterion": ("knowledge_error", "ToleranceCriterion"),
    "ToleranceResult": ("knowledge_error", "ToleranceResult"),
    "first_separating_amplitude": ("knowledge_error", "first_separating_amplitude"),
    "SeparationResult": ("knowledge_error", "SeparationResult"),
    "RoleAcceptance": ("nonlinear", "RoleAcceptance"),
    "ClassificationRule": ("nonlinear", "ClassificationRule"),
    "case_status": ("nonlinear", "case_status"),
    "classify_case": ("nonlinear", "classify_case"),
    "select_attempt": ("nonlinear", "select_attempt"),
    "detection_agreement": ("nonlinear", "detection_agreement"),
    "CaseClassification": ("nonlinear", "CaseClassification"),
    "AttemptSelection": ("nonlinear", "AttemptSelection"),
    "AgreementTable": ("nonlinear", "AgreementTable"),
    "electron_maps": ("selection", "electron_maps"),
    "arc_snr": ("selection", "arc_snr"),
    "gradient_power": ("selection", "gradient_power"),
    "complexity": ("selection", "complexity"),
    "rank_pool": ("selection", "rank_pool"),
    "RankingPolicy": ("selection", "RankingPolicy"),
    "spearman_rank_correlation": ("selection", "spearman_rank_correlation"),
    "top_k_jaccard": ("selection", "top_k_jaccard"),
    "top_k_recovery": ("selection", "top_k_recovery"),
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
