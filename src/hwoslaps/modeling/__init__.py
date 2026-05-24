"""Fisher-based lens modeling module for HWO-SLAPS."""

from .generator_fisher import perform_fisher_detection
from .fisher_detector import FisherDetector
from .fisher_core import (
    AsimovAmplitudeResult,
    SignalBankResult,
    SpuriousAmplitudeResult,
    SystematicModeCoupling,
    SystematicModeScanResult,
    Whitener,
    ProfileLikelihoodWorkspace,
    compute_asimov_detectability,
    evaluate_signal_bank,
    compute_spurious_amplitude,
    scan_systematic_modes,
    sidak_local_p,
    sidak_local_z,
    bonferroni_local_p,
    global_p_from_local,
    detectable_area,
)
from .utils_fisher import (
    FisherDetectionData,
    FisherLocalData,
    FisherMapData,
    FisherModeCouplingData,
    FisherModeScanData,
    print_fisher_summary,
)

__all__ = [
    'perform_fisher_detection',
    'FisherDetector',
    'FisherDetectionData',
    'FisherLocalData',
    'FisherMapData',
    'FisherModeCouplingData',
    'FisherModeScanData',
    'print_fisher_summary',
    'AsimovAmplitudeResult',
    'SignalBankResult',
    'SpuriousAmplitudeResult',
    'SystematicModeCoupling',
    'SystematicModeScanResult',
    'Whitener',
    'ProfileLikelihoodWorkspace',
    'compute_asimov_detectability',
    'evaluate_signal_bank',
    'compute_spurious_amplitude',
    'scan_systematic_modes',
    'sidak_local_p',
    'sidak_local_z',
    'bonferroni_local_p',
    'global_p_from_local',
    'detectable_area',
]
