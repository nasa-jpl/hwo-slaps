"""Tests for study-manifest expansion and PSF-ensemble seeding."""

import numpy as np
import pytest

from hwoslaps.analysis.manifest import expand_manifest


def _baseline_config():
    return {
        "run_name": "baseline",
        "global_seed": 0,
        "plotting": {"output_dir": "outputs/placeholder"},
        "lensing": {"subhalo": {"mass": 1.0e7}},
        "psf": {
            "telescope": {"num_rings": 2},
            "aberrations": {
                "enable_segment_pistons": False,
                "enable_segment_tiptilts": False,
                "enable_segment_hexikes": False,
                "enable_global_zernikes": False,
                "segment_pistons": {},
                "segment_tiptilts": {},
                "segment_hexikes": {},
                "global_zernikes": {},
            },
        },
        "modeling": {
            "fisher": {
                "mode": "local",
                "compute_psf_mode_scan": False,
                "map": {
                    "num_angles": 8,
                    "offset_pixels": 0.0,
                    "explicit_positions_yx": None,
                },
            }
        },
    }


def _ensemble_manifest(draws=2, amplitudes=(0.0, 5.0, 10.0), seed=1234):
    return {
        "study_name": "unit_test_study",
        "output_root": "outputs/unit_test_study",
        "map": {"num_angles": 8, "offset_pixels": 0.0},
        "psf_ensemble_sweep": {
            "enabled": True,
            "pivot_mass": {"label": "m1e7", "value": 1.0e7},
            "seed": seed,
            "units": "nm RMS",
            "families": ["segment_only", "global_only", "combined"],
            "amplitudes": list(amplitudes),
            "draws_per_amplitude": draws,
            "combined_rms_split": "equal_variance",
            "segment_hexikes": {"segments": "all", "mode_nolls": [2, 3, 4, 5, 6]},
            "global_zernikes": {"mode_nolls": "4-11"},
        },
    }


def test_ensemble_expansion_count_and_structure():
    runs = expand_manifest(_ensemble_manifest(), _baseline_config())
    # One perfect reference plus 3 families x 2 nonzero amplitudes x 2 draws.
    assert len(runs) == 1 + 3*2*2
    assert runs[0].sweep == "psf_ensemble_perfect_reference"
    assert runs[0].psf_case == "perfect"
    families = {run.psf_family for run in runs[1:]}
    assert families == {"segment_only", "global_only", "combined"}


def test_ensemble_draws_are_deterministic():
    first = expand_manifest(_ensemble_manifest(), _baseline_config())
    second = expand_manifest(_ensemble_manifest(), _baseline_config())
    for run_a, run_b in zip(first, second):
        assert run_a.run_name == run_b.run_name
        assert run_a.config == run_b.config


def test_ensemble_draws_differ_between_draws():
    runs = expand_manifest(_ensemble_manifest(), _baseline_config())
    segment_runs = [run for run in runs if run.psf_family == "segment_only"]
    coeffs_a = segment_runs[0].config["psf"]["aberrations"]["segment_hexikes"]
    coeffs_b = segment_runs[1].config["psf"]["aberrations"]["segment_hexikes"]
    assert coeffs_a != coeffs_b


def test_segment_rms_normalization():
    runs = expand_manifest(_ensemble_manifest(), _baseline_config())
    run = next(run for run in runs if run.psf_family == "segment_only" and run.psf_amplitude == 10.0)
    coeffs = run.config["psf"]["aberrations"]["segment_hexikes"]
    assert len(coeffs) == 19  # num_rings = 2 hexagonal aperture
    values = np.array([val for per_segment in coeffs.values() for val in per_segment.values()])
    aperture_rms = np.sqrt(np.sum(values**2)/len(coeffs))
    assert aperture_rms == pytest.approx(10.0)


def test_global_rms_normalization_and_mode_range():
    runs = expand_manifest(_ensemble_manifest(), _baseline_config())
    run = next(run for run in runs if run.psf_family == "global_only" and run.psf_amplitude == 5.0)
    coeffs = run.config["psf"]["aberrations"]["global_zernikes"]
    assert sorted(coeffs) == list(range(4, 12))
    assert np.linalg.norm(list(coeffs.values())) == pytest.approx(5.0)


def test_combined_family_splits_rms_budget_with_equal_variance():
    runs = expand_manifest(_ensemble_manifest(), _baseline_config())
    run = next(run for run in runs if run.psf_family == "combined" and run.psf_amplitude == 10.0)
    aberr = run.config["psf"]["aberrations"]
    segment_values = np.array(
        [val for per_segment in aberr["segment_hexikes"].values() for val in per_segment.values()]
    )
    aperture_rms = np.sqrt(np.sum(segment_values**2)/len(aberr["segment_hexikes"]))
    assert aperture_rms == pytest.approx(10.0/np.sqrt(2.0))
    assert np.linalg.norm(list(aberr["global_zernikes"].values())) == pytest.approx(10.0/np.sqrt(2.0))


def test_seed_depends_on_expansion_position():
    runs = expand_manifest(_ensemble_manifest(seed=1234), _baseline_config())
    seeds = [run.config["global_seed"] for run in runs]
    assert seeds[0] == 1234  # perfect reference records the base seed
    # Ensemble draws are seeded by base_seed + n_runs_so_far + 1.
    assert seeds[1:] == [1234 + idx + 1 for idx in range(1, len(runs))]


def test_mass_sweep_expansion_is_perfect_psf():
    manifest = {
        "study_name": "unit_test_study",
        "output_root": "outputs/unit_test_study",
        "mass_sweep": {
            "enabled": True,
            "masses": [
                {"label": "m1e7", "value": 1.0e7, "run_map": True},
                {"label": "m1e8", "value": 1.0e8},
            ],
        },
    }
    runs = expand_manifest(manifest, _baseline_config())
    assert [run.mass_msun for run in runs] == [1.0e7, 1.0e8]
    assert all(run.psf_case == "perfect" for run in runs)
    assert runs[0].config["modeling"]["fisher"]["mode"] == "both"
    assert runs[1].config["modeling"]["fisher"]["mode"] == "local"


def test_single_mode_psf_sweep_zero_amplitude_is_perfect():
    manifest = {
        "study_name": "unit_test_study",
        "output_root": "outputs/unit_test_study",
        "psf_sweep": {
            "enabled": True,
            "family": "segment_hexikes",
            "segment": 0,
            "mode_noll": 2,
            "pivot_mass": {"label": "m1e7", "value": 1.0e7},
            "amplitudes": [0.0, 100.0],
            "units": "nm RMS",
        },
    }
    runs = expand_manifest(manifest, _baseline_config())
    assert [run.psf_case for run in runs] == ["perfect", "segment_hexike"]
    aberr = runs[1].config["psf"]["aberrations"]
    assert aberr["segment_hexikes"] == {0: {2: 100.0}}
