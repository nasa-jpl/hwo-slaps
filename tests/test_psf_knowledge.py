"""Unit contracts for the v5 PSF knowledge-error block."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT/"scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from generate_psf_knowledge_campaign import (  # noqa: E402
    _map_queue_lines,
    _smoke_queue_lines,
)
from harvest_nonlinear_validation import (  # noqa: E402
    _verify_row,
    first_separating_delta,
)
from harvest_psf_knowledge import (  # noqa: E402
    _delta_star,
    _direction_estimand_rows,
    first_spurious_delta,
    matched_receipt_findings,
)
from run_nonlinear_validation import (  # noqa: E402
    derive_direction_seed,
    derive_sampler_seed,
)
from run_psf_knowledge_map import (  # noqa: E402
    map_artifact_name,
    select_knowledge_rungs,
    validate_job_coordinates,
    verify_ladder_artifact_identity,
    verify_truth_kernel_digest,
)
from hwoslaps.psf.mismatch import build_psf_mismatch_spec  # noqa: E402
from hwoslaps.psf.mismatch import _identity_from_payload  # noqa: E402
from test_fisher_grid_map import grid_setup  # noqa: E402,F401

ENTROPY = 20260823
PRIOR_SHA256 = (
    "bfbececdcbe5fb37a4abcb018b63544d47c4c37ecc1900a539d62755b740c488"
)


@pytest.fixture()
def compact_config(tmp_path):
    """Build the compact two-sided mismatch fixture used by pairing tests."""
    pytest.importorskip("autolens")
    pytest.importorskip("hcipy")
    with (PROJECT_ROOT/"configs"/"master_config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    prior_path = tmp_path/"prior.yaml"
    prior_path.write_text(
        yaml.safe_dump({
            "name": "tiny",
            "segment_variance_fraction": 0.4,
            "global_weights": {4: 1.0, 5: 0.5},
            "segment_weights": {1: 1.0, 2: 0.5},
        }),
        encoding="utf-8",
    )
    config["run_name"] = "psf-knowledge-pairing"
    config["plotting"] = {"enabled": False, "output_dir": str(tmp_path)}
    config["lensing"]["grid"] = {"shape": [15, 15], "pixel_scale": 0.1}
    config["lensing"]["lens_galaxy"]["mass"] = {
        "type": "Isothermal",
        "centre": [0.0, 0.0],
        "ell_comps": [0.05, -0.02],
        "einstein_radius": 0.5,
    }
    config["lensing"]["source_galaxy"]["light"].update({
        "centre": [0.02, 0.03],
        "ell_comps": [0.03, -0.01],
        "intensity": 4.0,
        "effective_radius": 0.16,
    })
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "model": "PointMass",
        "mass": 1.0e8,
        "position": {"type": "direct", "centre": [0.1, 0.0]},
    }
    config["psf"]["telescope"]["num_rings"] = 1
    config["psf"]["telescope"]["supersampling_factor"] = 1
    config["psf"]["hres_psf"].update({
        "num_pix": 64,
        "num_airy": 4,
        "sampling": 5,
        "save_highres_psf_npy": False,
    })
    config["psf"]["kernel"]["shape_native"] = [7, 7]
    config["psf"]["aberrations"] = {
        "enable_segment_pistons": True,
        "enable_segment_tiptilts": True,
        "enable_segment_hexikes": False,
        "enable_global_zernikes": True,
        "segment_pistons": {0: 0.5, 1: -0.5},
        "segment_tiptilts": {0: [0.01, -0.02]},
        "segment_hexikes": {0: {1: 3.0, 2: -1.0}},
        "global_zernikes": {4: 3.0, 6: -1.0},
    }
    config["modeling"]["fit_psf"] = {
        "mode": "delta",
        "delta": {
            "prior_table": str(prior_path),
            "amplitude_rms_nm": 2.0,
            "seed": 20260814,
            "family": "combined",
        },
    }
    from hwoslaps.config.validation import validate_or_raise

    validate_or_raise(config)
    return config


def _knowledge_fixture() -> dict:
    """Return the minimal knowledge block used by pure-function tests."""
    return {
        "member_set": {
            "name": "selected12",
            "source_campaign_uuid": "ladder-uuid",
        },
        "residual_model": {
            "amplitude_rms_nm_rungs": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0],
            "directions": 8,
        },
        "gates": {
            "retention_q10_min": 0.9,
            "spurious_q90_max": 0.1,
            "sensitivity": [[0.8, 0.2]],
        },
        "ratio_floor": {"cells": 33, "arcsec2": 0.0825},
    }


def test_direction_seed_values_are_pinned():
    """The four cross-check seeds use spawn key (7, direction, system)."""
    assert derive_direction_seed(ENTROPY, 1, 43) == 2917453207
    assert derive_direction_seed(ENTROPY, 8, 43) == 656673105
    assert derive_direction_seed(ENTROPY, 1, 728) == 2045173381
    assert derive_direction_seed(ENTROPY, 3, 813) == 1383433009


def test_direction_seeds_are_unique_and_disjoint_from_declared_streams():
    """Direction, sampler and null seed sets have no collisions."""
    direction = {
        derive_direction_seed(ENTROPY, d, index)
        for d in range(1, 9)
        for index in range(1000)
    }
    sampler = {
        derive_sampler_seed(ENTROPY, index, arm)
        for arm in range(24)
        for index in range(1000)
    }
    null = {
        int(np.random.SeedSequence(
            entropy=ENTROPY,
            spawn_key=(6, replicate, index),
        ).generate_state(1, dtype=np.uint32)[0])
        for replicate in range(1, 10)
        for index in range(1000)
    }
    assert len(direction) == 8000
    assert len(direction | sampler | null) == len(direction) + len(sampler) + len(null)


def _rung_artifact() -> dict:
    """Return a synthetic ladder artifact with three D-K5 crossings."""
    return {
        "rung_logm": np.asarray([6.0, 7.2, 8.0, 8.25]),
        "rung_q_max": np.asarray([1.0, 10.0, 20.0, 30.0]),
        "rung_detectable_area_arcsec2": np.asarray([0.0, 0.1, 0.2, 0.3]),
        "m_best": np.asarray(7.1),
        "m_best_bracket_logm": np.asarray([7.1, 7.2]),
        "m10": np.asarray(7.8),
        "m10_bracket_logm": np.asarray([7.7, 8.0]),
        "m50": np.asarray(8.1),
        "m50_bracket_logm": np.asarray([8.0, 8.25]),
    }


def test_d_k5_selects_upper_brackets_and_classes():
    """Three distinct upper brackets retain their class labels."""
    rungs = select_knowledge_rungs(_rung_artifact())
    assert [rung["logm"] for rung in rungs] == [7.2, 8.0, 8.25]
    assert [rung["classes"] for rung in rungs] == [
        ["m_best"], ["m10"], ["m50"]
    ]


def test_d_k5_collapses_coincident_upper_brackets():
    """Coincident M10 and M50 upper brackets become one rung."""
    artifact = _rung_artifact()
    artifact["m50_bracket_logm"] = np.asarray([7.9, 8.0])
    rungs = select_knowledge_rungs(artifact)
    assert [rung["logm"] for rung in rungs] == [7.2, 8.0]
    assert rungs[-1]["classes"] == ["m10", "m50"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda artifact: artifact.update({"m_best": np.asarray(np.nan)}),
            "m_best",
        ),
        (
            lambda artifact: artifact.update({
                "m50_bracket_logm": np.asarray([8.1, 9.0])
            }),
            "walked artifact rungs",
        ),
    ],
)
def test_d_k5_selection_fails_closed(mutation, match):
    """Non-finite crossings and unwalked brackets cannot be selected."""
    artifact = _rung_artifact()
    mutation(artifact)
    with pytest.raises(ValueError, match=match):
        select_knowledge_rungs(artifact)


def test_runner_coordinate_and_naming_guards():
    """Map coordinate guards and artifact names are frozen."""
    knowledge = _knowledge_fixture()
    with pytest.raises(ValueError, match="not a declared"):
        validate_job_coordinates(3.0, 1, knowledge)
    with pytest.raises(ValueError, match="nonzero"):
        validate_job_coordinates(5.0, 0, knowledge)
    with pytest.raises(ValueError, match="must be 0"):
        validate_job_coordinates(0.0, 1, knowledge)
    with pytest.raises(ValueError, match="outside"):
        validate_job_coordinates(5.0, 9, knowledge)
    assert map_artifact_name(7.2, 5.0, 1) == (
        "psf_knowledge_map_m7.20_delta5_dir1.npz"
    )
    assert map_artifact_name(8.0, 0.0, 0) == (
        "psf_knowledge_map_m8.00_delta0_dir0.npz"
    )


def test_runner_artifact_identity_guards():
    """System, UUID and truth-kernel mismatches fail closed."""
    config = {
        "run_name": "ladder_selected_sys0043",
        "ladder": {
            "aperture": {
                "stage0_aperture_sha256": "a"*64,
                "stage0_contour_sha256": "b"*64,
            }
        },
    }
    artifact = {
        "system_id": np.asarray(config["run_name"]),
        "tier": np.asarray("selected"),
        "psf_state": np.asarray("science35"),
        "psf_kernel_shape_native": np.asarray([999, 999]),
        "campaign_uuid": np.asarray("ladder-uuid"),
        "aperture_sha256": np.asarray("a"*64),
        "contour_sha256": np.asarray("b"*64),
    }
    with pytest.raises(ValueError, match="system id"):
        broken = copy.deepcopy(artifact)
        broken["system_id"] = np.asarray("wrong")
        verify_ladder_artifact_identity(broken, config, _knowledge_fixture())
    with pytest.raises(ValueError, match="campaign uuid"):
        broken = copy.deepcopy(artifact)
        broken["campaign_uuid"] = np.asarray("wrong")
        verify_ladder_artifact_identity(broken, config, _knowledge_fixture())
    with pytest.raises(ValueError, match="kernel digest"):
        verify_truth_kernel_digest("actual", "expected")


def test_estimand_arithmetic_enforces_ratio_floor_and_gates():
    """Retention, spurious fractions and delta-star use declared cells."""
    rows = [
        {
            "direction": 1, "seed": 1, "mismatch_cells": 90,
            "spurious_cells": 1, "mismatch_area_arcsec2": 0.225,
            "spurious_area_arcsec2": 0.0025,
        },
        {
            "direction": 2, "seed": 2, "mismatch_cells": 80,
            "spurious_cells": 2, "mismatch_area_arcsec2": 0.2,
            "spurious_area_arcsec2": 0.005,
        },
        {
            "direction": 3, "seed": 3, "mismatch_cells": 70,
            "spurious_cells": 3, "mismatch_area_arcsec2": 0.175,
            "spurious_area_arcsec2": 0.0075,
        },
    ]
    estimands, below, zero = _direction_estimand_rows(rows, 100, 33)
    assert below == 0
    assert zero == 0
    assert [entry["R"] for entry in estimands] == [0.9, 0.8, 0.7]
    assert [entry["F"] for entry in estimands] == [0.01, 0.02, 0.03]
    summary = {
        1.0: {
            "quantiles": {
                "R": {"q10": 0.9}, "F": {"q90": 0.03}
            }
        },
        5.0: {
            "quantiles": {
                "R": {"q10": 0.8}, "F": {"q90": 0.03}
            }
        },
        35.0: {
            "quantiles": {
                "R": {"q10": 0.99}, "F": {"q90": 0.0}
            }
        },
    }
    assert _delta_star(summary, 0.9, 0.1, 35.0)["delta_star"] == 1.0
    assert _delta_star(summary, 0.8, 0.2, 35.0)["delta_star"] == 5.0
    assert first_spurious_delta({
        1.0: {"directions": [{"spurious_cells": 0}]},
        5.0: {"directions": [{"spurious_cells": 1}]},
    }) == 5.0
    assert _delta_star(
        {1.0: {"quantiles": {"R": {"q10": 0.1}, "F": {"q90": 0.9}}}},
        0.9,
        0.1,
        35.0,
    )["none_passes"] is True
    _, below, zero = _direction_estimand_rows(rows, 20, 33)
    assert below == 3
    assert zero == 0
    _, below, zero = _direction_estimand_rows(rows, 0, 33)
    assert below == 0
    assert zero == 3


def test_receipt_cells_are_exact_and_q_max_uses_relative_tolerance():
    """A cell differs finding is exact while q-max follows the precedent."""
    artifact = {
        "matched_cells": np.asarray(33),
        "matched_q_max": np.asarray(10.0),
    }
    rung = {"production_cells": 33, "production_q_max": 10.0}
    assert matched_receipt_findings(artifact, rung) == []
    broken_cells = dict(artifact, matched_cells=np.asarray(34))
    assert any(
        "matched cells" in finding
        for finding in matched_receipt_findings(broken_cells, rung)
    )
    broken_q = dict(artifact, matched_q_max=np.asarray(10.0 + 1.0e-4))
    assert any(
        "matched q_max" in finding
        for finding in matched_receipt_findings(broken_q, rung)
    )


def _queue_job(run_name: str, image_side: int) -> dict:
    """Return a compact Fisher generator job fixture."""
    return {
        "run_name": run_name,
        "restamped_config": f"/campaign/configs/{run_name}.yaml",
        "ladder_artifact": f"/ladder/outputs/{run_name}/ladder_result.npz",
        "output_dir": f"/campaign/outputs/{run_name}",
        "image_side_px": image_side,
        "template": "template",
        "golden": image_side == 400,
        "rungs": [{"logm": 7.2}, {"logm": 8.0}],
    }


def test_fisher_queue_order_and_smoke_shape():
    """The largest system leads and each system owns 49 map jobs."""
    jobs = [_queue_job("large", 800), _queue_job("small", 400)]
    lines = _map_queue_lines(
        jobs,
        [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0],
        list(range(1, 9)),
    )
    assert len(lines) == 98
    assert lines[0].split()[0].endswith("large.yaml")
    assert lines[0].split()[2:4] == ["0", "0"]
    assert lines[49].split()[0].endswith("small.yaml")
    assert lines[49].split()[2:4] == ["0", "0"]
    smoke_lines, smoke_names = _smoke_queue_lines(
        jobs,
        {
            "deltas": [0.0, 5.0],
            "direction": 1,
            "members": ["smallest_image", "largest_image"],
        },
    )
    assert len(smoke_lines) == 4
    assert smoke_names == ["large", "small"]
    assert smoke_lines[0].split()[2:4] == ["0", "0"]
    assert smoke_lines[1].split()[2:4] == ["5", "1"]


def _flatten_draw(draw: dict) -> np.ndarray:
    """Flatten canonical segment and global draw coefficients in key order."""
    values = []
    for segment in sorted(draw["segment_hexikes"]):
        for mode in sorted(draw["segment_hexikes"][segment]):
            values.append(float(draw["segment_hexikes"][segment][mode]))
    for mode in sorted(draw["global_zernikes"]):
        values.append(float(draw["global_zernikes"][mode]))
    return np.asarray(values)


def test_paired_direction_draws_scale_linearly(compact_config):
    """One direction seed gives proportional fit-minus-truth draws."""
    low_config = copy.deepcopy(compact_config)
    high_config = copy.deepcopy(compact_config)
    low_config["modeling"]["fit_psf"]["delta"]["amplitude_rms_nm"] = 2.0
    high_config["modeling"]["fit_psf"]["delta"]["amplitude_rms_nm"] = 10.0
    low = build_psf_mismatch_spec(low_config)
    high = build_psf_mismatch_spec(high_config)
    np.testing.assert_allclose(
        _flatten_draw(high.draw_aberrations),
        5.0*_flatten_draw(low.draw_aberrations),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert low.measured_draw_rms_nm == pytest.approx(2.0, rel=1.0e-9)
    assert high.measured_draw_rms_nm == pytest.approx(10.0, rel=1.0e-9)


def test_delta_zero_grid_map_is_the_matched_limit(grid_setup):
    """A zero explicit delta matches the detector's production path."""
    from hwoslaps.modeling.fisher_detector import FisherDetector

    config = copy.deepcopy(grid_setup["config"])
    config["modeling"]["fit_psf"] = {
        "mode": "delta",
        "delta": {
            "prior_table": "configs/psf_priors/jwst_wss_drift_v1.yaml",
            "seed": 20260814,
            "family": "combined",
            "amplitude_rms_nm": 0.0,
        },
    }
    detector = FisherDetector(
        observation_baseline=grid_setup["observation_baseline"],
        lensing_baseline=grid_setup["lensing_baseline"],
        psf_data=grid_setup["psf_data"],
        full_config=config,
        fisher_config=copy.deepcopy(config["modeling"]["fisher"]),
    )
    delta_map = detector.compute_grid_map()
    matched_map = grid_setup["grid_map"]
    np.testing.assert_array_equal(
        delta_map.detectable_mask_2d,
        matched_map.detectable_mask_2d,
    )
    np.testing.assert_array_equal(
        delta_map.mismatch_detectable_mask_2d,
        delta_map.detectable_mask_2d,
    )
    np.testing.assert_allclose(
        delta_map.q_asimov_2d,
        matched_map.q_asimov_2d,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        delta_map.q_mismatch_2d,
        matched_map.q_asimov_2d,
        rtol=1.0e-10,
    )
    assert delta_map.num_false_positive == 0


def _delta_row_fixture() -> tuple[dict, dict, dict, dict]:
    """Return a consistent nonlinear delta-row verification fixture."""
    declaration = {
        "arm_index": 16,
        "dataset_kind": "noisy",
        "subhalo_in_truth": False,
        "fit_mode": "freed",
        "rung": "top",
        "sample": "all",
        "fit_psf_delta": {
            "amplitude_rms_nm": 5.0,
            "directions": [1, 2, 3],
        },
    }
    fit_block = {
        "kernel_shape_native": [51, 51],
        "fit_psf": {
            "mode": "delta",
            "prior_table": "configs/psf_priors/jwst_wss_drift_v1.yaml",
            "seed": 20260814,
            "family": "combined",
            "amplitude_rms_nm": 0.0,
        },
        "n_live_smooth": 100,
        "n_live_subhalo_search": 200,
        "n_live_subhalo_fixed": 100,
        "maxcall": 500000,
        "jax_n_batch": 32,
        "number_of_cores": 1,
        "log10_m200_range": [6.0, 9.7],
        "nautilus_training_workers": 4,
    }
    job = {
        "run_name": "ladder_selected_sys0043",
        "system_id": "ladder_selected_sys0043",
        "tier": "selected",
        "report_tiers": ["selected"],
        "template": "cosmos_48849",
        "golden": False,
        "censored": False,
        "restamped_config_hash": "staged-hash",
        "staged_global_seed": 12345,
        "ladder_campaign_uuid": "ladder-campaign",
        "ladder_config_hash": "ladder-config",
    }
    truth_hash = "truth-config-hash"
    lensing_scale = 0.00716
    delta_seed = derive_direction_seed(ENTROPY, 1, 43)
    delta_id = _identity_from_payload({
        "schema": "psf_mismatch_delta_v1",
        "prior_table_sha256": PRIOR_SHA256,
        "amplitude_rms_nm": 5.0,
        "seed": delta_seed,
        "family": "combined",
        "truth_psf_config_hash": truth_hash,
        "lensing_pixel_scale": lensing_scale,
    })
    payload = {
        "schema_version": 3,
        "system_id": job["run_name"],
        "arm": "noisy_control_d5",
        "arm_declaration": declaration,
        "sampler_seed": derive_sampler_seed(ENTROPY, 43, 16),
        "campaign_uuid": "campaign-uuid",
        "code_revision": {"sha256": "revision-sha"},
        "staged_config_hash": "staged-hash",
        "ladder_campaign_uuid": "ladder-campaign",
        "ladder_config_hash": "ladder-config",
        "censored": False,
        "tier": "selected",
        "rung": {"logm": 8.0},
        "noise_seed": 12345,
        "noise_replicate": 0,
        "noise_spawn_key": None,
        "fit_settings": fit_block,
        "kernel_sha256": "fit-kernel",
        "truth_kernel_sha256": "truth-kernel",
        "n_unmasked_pixels": 10,
        "smooth_status": "success",
        "subhalo_status": "success",
        "q_fit": 1.0,
        "delta_log_evidence": 0.0,
        "fit_psf_delta": {
            "amplitude_rms_nm": 5.0,
            "direction": 1,
            "seed": delta_seed,
            "seed_spawn_key": [7, 1, 43],
            "delta_id": delta_id,
            "requested_draw_rms_nm": 5.0,
            "measured_draw_rms_nm": 5.0,
            "fit_kernel_sha256": "fit-kernel",
            "truth_kernel_sha256": "truth-kernel",
            "fit_psf_config_hash": "fit-config-hash",
            "truth_psf_config_hash": truth_hash,
            "lensing_pixel_scale": lensing_scale,
            "prior_table_sha256": PRIOR_SHA256,
            "family": "combined",
        },
    }
    manifest = {
        "schema_version": 3,
        "design_freeze": {"version": 5},
        "name": "psf_knowledge_nonlinear_v1",
        "campaign_uuid": "campaign-uuid",
        "campaign": {
            "member_set": "selected12",
            "arms": ["noisy_control_d5"],
        },
        "code_revision": {"sha256": "revision-sha"},
    }
    protocol = {
        "arms": {"noisy_control_d5": declaration},
        "fit": fit_block,
        "seeds": {"entropy": ENTROPY},
        "psf_knowledge_error": {
            "residual_model": {"prior_table_sha256": PRIOR_SHA256}
        },
    }
    return job, payload, manifest, protocol


def test_delta_nonlinear_row_verification_and_findings():
    """Delta rows accept consistent provenance and flag each key mutation."""
    job, payload, manifest, protocol = _delta_row_fixture()
    knowledge = protocol["psf_knowledge_error"]
    assert _verify_row(
        job, "noisy_control_d5", payload, manifest, protocol, knowledge
    ) == []
    mutations = [
        (lambda row: row["fit_psf_delta"].update({"seed": 1}), "direction seed"),
        (
            lambda row: row.update({"kernel_sha256": "truth-kernel"}),
            "unexpectedly equals",
        ),
        (
            lambda row: row["fit_psf_delta"].update({
                "measured_draw_rms_nm": 5.0 + 1.0e-6
            }),
            "measured draw RMS",
        ),
        (
            lambda row: row["fit_psf_delta"].update({"delta_id": "wrong"}),
            "delta_id",
        ),
        (
            lambda row: row["fit_psf_delta"].update({"prior_table_sha256": "0"*64}),
            "prior table digest",
        ),
    ]
    for mutate, match in mutations:
        broken = copy.deepcopy(payload)
        mutate(broken)
        assert any(
            match in finding
            for finding in _verify_row(
                job,
                "noisy_control_d5",
                broken,
                manifest,
                protocol,
                knowledge,
            )
        )


def test_nonlinear_first_separating_delta_uses_cp_bounds():
    """The first control lower bound above the null upper bound is selected."""
    controls = {
        2.0: {"q_fit_ge_10": {"interval": [0.01, 0.20]}},
        5.0: {"q_fit_ge_10": {"interval": [0.21, 0.40]}},
        10.0: {"q_fit_ge_10": {"interval": [0.41, 0.60]}},
    }
    null = {"q_fit_ge_10": {"interval": [0.0, 0.20]}}
    assert first_separating_delta(controls, null) == 5.0
    assert first_separating_delta(
        {2.0: {"q_fit_ge_10": {"interval": [0.01, 0.20]}}},
        null,
    ) is None


def test_dispatcher_contains_map_phases_and_direction_artifact_rules():
    """The dispatcher exposes maps, maps_smokes and optional fit directions."""
    text = (PROJECT_ROOT/"scripts"/"nonlinear_validation_dispatch.sh").read_text()
    assert "maps) QUEUE" in text
    assert "maps_smokes) QUEUE" in text
    assert "run_psf_knowledge_map.py" in text
    assert "psf_knowledge_job_delta${FIELDS[2]}_dir${FIELDS[3]}.json" in text
    assert "nonlinear_validation_${FIELDS[2]}_dir${FIELDS[4]}.json" in text
    assert "--direction \"${FIELDS[4]}\"" in text
