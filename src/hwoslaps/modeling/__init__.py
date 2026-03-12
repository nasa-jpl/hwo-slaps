"""Subhalo detection and lens modeling module for HWO-SLAPS."""

from .generator import perform_subhalo_detection
from .utils import DetectionData, print_detection_summary
from .chi_square_detector import ChiSquareSubhaloDetector, DetectionResult
from .generator_fisher import perform_fisher_detection
from .fisher_publication_detector import PublicationFisherDetector
from .fisher_publication_core import (
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
    'perform_subhalo_detection',
    'perform_fisher_detection',
    'PublicationFisherDetector',
    'DetectionData',
    'FisherDetectionData',
    'FisherLocalData',
    'FisherMapData',
    'FisherModeCouplingData',
    'FisherModeScanData',
    'ChiSquareSubhaloDetector',
    'DetectionResult',
    'print_detection_summary',
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
