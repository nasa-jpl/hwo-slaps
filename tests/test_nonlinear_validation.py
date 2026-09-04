"""Unit tests for the nonlinear-validation campaign scripts."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]/"scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from extract_injection_positions import (  # noqa: E402
    CENSORED_LOGM,
    argmax_inside_aperture,
    below_logm,
    injection_logm,
    support_half_widths,
)
from generate_nonlinear_validation_campaign import (  # noqa: E402
    _validate_source_files,
    _validate_reused_positions,
    eligible_arms,
    sample_members,
    smoke_jobs,
)
from harvest_nonlinear_validation import (  # noqa: E402
    _campaign_findings,
    _load_declared_source_rows,
    _science_v3,
    _verify_row,
    clopper_pearson,
    expected_provenance,
    spearman_rank_correlation,
)
from run_nonlinear_validation import (  # noqa: E402
    PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY,
    apply_direction_override,
    apply_noise_replicate,
    build_arm_config,
    derive_direction_seed,
    derive_noise_seed,
    derive_sampler_seed,
    system_index,
    validate_direction_argument,
)

ENTROPY = 20260823

ARMS_FIXTURE = {
    "asimov_injected": {
        "arm_index": 0, "dataset_kind": "asimov", "subhalo_in_truth": True,
        "fit_mode": "freed", "rung": "top", "sample": "all",
    },
    "noisy_control": {
        "arm_index": 2, "dataset_kind": "noisy", "subhalo_in_truth": False,
        "fit_mode": "freed", "rung": "top", "sample": "all",
    },
    "asimov_below": {
        "arm_index": 3, "dataset_kind": "asimov", "subhalo_in_truth": True,
        "fit_mode": "freed", "rung": "below", "sample": "non_censored",
    },
    "asimov_fixed_bridge": {
        "arm_index": 4, "dataset_kind": "asimov", "subhalo_in_truth": True,
        "fit_mode": "fixed_template", "rung": "top", "sample": "golden",
    },
}

DELTA_ARM_FIXTURE = {
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

FIT_BLOCK_FIXTURE = {
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


class TestInjectionLogm:
    def test_bracketed_member_takes_bracket_top(self):
        logm, censored = injection_logm(7.018, np.array([6.95, 7.05]))
        assert logm == 7.05
        assert censored is False

    def test_censored_member_takes_ceiling(self):
        logm, censored = injection_logm(
            float("nan"), np.array([float("nan"), float("nan")])
        )
        assert logm == CENSORED_LOGM
        assert censored is True

    def test_finite_m_best_with_nan_bracket_fails_closed(self):
        with pytest.raises(ValueError, match="inconsistent"):
            injection_logm(8.0, np.array([float("nan"), float("nan")]))

    def test_below_rung_is_the_bracket_bottom(self):
        assert below_logm(np.array([6.95, 7.05])) == 6.95
        with pytest.raises(ValueError, match="NaN"):
            below_logm(np.array([float("nan"), 7.05]))


class TestSupportHalfWidths:
    def test_hand_computed_box(self):
        # 400 px image, 51 px kernel: 400//2 - 25 - 1 = 174 pixels of
        # half-width at 0.01 arcsec per pixel.
        assert support_half_widths((400, 400), 0.01, (51, 51)) == (
            pytest.approx(1.74),
            pytest.approx(1.74),
        )

    def test_kernel_larger_than_image_fails_closed(self):
        with pytest.raises(ValueError, match="no .*valid pixels"):
            support_half_widths((354, 354), 0.00716, (999, 999))


class TestArgmaxInsideAperture:
    def test_peak_outside_aperture_is_ignored(self):
        y = np.array([-1.0, 0.0, 1.0, 2.0])
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        q = np.zeros((4, 4))
        q[3, 3] = 100.0
        q[1, 2] = 7.0
        position, q_max, indices, fraction = argmax_inside_aperture(
            y, x, q, (0.0, 0.0), 1.2
        )
        assert position == (0.0, 1.0)
        assert q_max == 7.0
        assert indices == (1, 2)
        assert fraction == 1.0

    def test_peak_outside_support_is_ignored(self):
        y = x = np.array([-1.0, 0.0, 1.0])
        q = np.zeros((3, 3))
        q[2, 2] = 50.0
        q[1, 1] = 5.0
        position, q_max, _, fraction = argmax_inside_aperture(
            y, x, q, (0.0, 0.0), 2.0, support_half_widths_arcsec=(0.5, 0.5)
        )
        assert position == (0.0, 0.0)
        assert q_max == 5.0
        assert fraction == pytest.approx(1.0/9.0)

    def test_no_node_inside_aperture_fails_closed(self):
        y = x = np.array([5.0, 6.0])
        with pytest.raises(ValueError, match="no node inside"):
            argmax_inside_aperture(y, x, np.ones((2, 2)), (0.0, 0.0), 1.0)

    def test_non_finite_inside_aperture_fails_closed(self):
        y = x = np.array([-1.0, 0.0, 1.0])
        q = np.ones((3, 3))
        q[1, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            argmax_inside_aperture(y, x, q, (0.0, 0.0), 1.5)


class TestSystemIndex:
    def test_parses_run_names_and_bare_ids(self):
        assert system_index("ladder_parent_sys0625") == 625
        assert system_index("ladder_selected_sys0728") == 728
        assert system_index("sys0007") == 7

    def test_rejects_identifiers_without_sys_block(self):
        with pytest.raises(ValueError, match="No 'sys' block"):
            system_index("member0625")
        with pytest.raises(ValueError, match="digits"):
            system_index("sys0625b")


class TestSamplerSeeds:
    def test_declared_rule_reference_values(self):
        # Locked reference values of the freeze v3 sampler stream:
        # SeedSequence(entropy=20260823, spawn_key=(5, i, arm)).
        assert derive_sampler_seed(ENTROPY, 625, 0) == 1844900749
        assert derive_sampler_seed(ENTROPY, 625, 1) == 2981609798
        assert derive_sampler_seed(ENTROPY, 625, 2) == 2855392809
        assert derive_sampler_seed(ENTROPY, 728, 0) == 167413308
        assert derive_sampler_seed(ENTROPY, 7, 2) == 3196177908

    def test_seeds_are_unique_across_the_campaign(self):
        seeds = {
            derive_sampler_seed(ENTROPY, index, arm_index)
            for index in range(1000)
            for arm_index in range(16)
        }
        assert len(seeds) == 16000


class TestNoiseSeeds:
    def test_declared_rule_reference_values(self):
        """Locked values of the v4 null-noise stream."""
        assert derive_noise_seed(ENTROPY, 1, 625) == 3904659053
        assert derive_noise_seed(ENTROPY, 9, 625) == 413005996
        assert derive_noise_seed(ENTROPY, 1, 728) == 4105259175
        assert derive_noise_seed(ENTROPY, 5, 7) == 4262057712

    def test_noise_seeds_are_unique_and_disjoint_from_sampler_seeds(self):
        """Noise and sampler streams have distinct seeds in the test grid."""
        noise = {
            derive_noise_seed(ENTROPY, replicate, index)
            for replicate in range(1, 10)
            for index in range(1000)
        }
        sampler = {
            derive_sampler_seed(ENTROPY, index, arm_index)
            for index in range(1000)
            for arm_index in range(16)
        }
        assert len(noise) == 9000
        assert len(noise | sampler) == len(noise) + len(sampler)


class TestDirectionSeeds:
    def test_declared_direction_reference_values(self):
        """Lock the spawn-key-7 direction stream to NumPy values."""
        assert PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY == 7
        assert derive_direction_seed(ENTROPY, 1, 43) == 2917453207
        assert derive_direction_seed(ENTROPY, 8, 43) == 656673105
        assert derive_direction_seed(ENTROPY, 1, 728) == 2045173381
        assert derive_direction_seed(ENTROPY, 3, 813) == 1383433009

    def test_direction_seeds_are_unique_and_disjoint(self):
        """The direction stream is disjoint from sampler and null streams."""
        direction = {
            derive_direction_seed(ENTROPY, d, index)
            for d in range(1, 9)
            for index in range(1000)
        }
        sampler = {
            derive_sampler_seed(ENTROPY, index, arm_index)
            for index in range(1000)
            for arm_index in range(24)
        }
        null = {
            derive_noise_seed(ENTROPY, replicate, index)
            for replicate in range(1, 10)
            for index in range(1000)
        }
        assert len(direction) == 8000
        assert len(sampler) == 24000
        assert len(null) == 9000
        assert len(direction | sampler | null) == (
            len(direction) + len(sampler) + len(null)
        )


def _staged_config() -> dict:
    return {
        "run_name": "ladder_parent_sys0625",
        "global_seed": 3792114890,
        "plotting": {"enabled": True, "output_dir": "/nfs/somewhere"},
        "psf": {"kernel": {"shape_native": [51, 51]}},
        "modeling": {"detection": "fisher", "enabled": False},
        "lensing": {
            "subhalo": {
                "enabled": False,
                "mass": 1.0e7,
                "model": "NFW",
                "position": {"type": "angle", "angle": 90.0},
            },
        },
        "observation": {"exposure_time": 2000.0},
    }


def _rung_payload() -> dict:
    return {
        "logm": 8.05,
        "mass_msun": 10.0**8.05,
        "position_yx_arcsec": [0.25, -0.85],
        "q_f_matched": 10.4,
        "q_f_production_at_position": 10.6,
    }


def _verification_fixture():
    """Return a minimal valid v4 row verification fixture."""
    declaration = deepcopy(ARMS_FIXTURE["noisy_control"])
    job = {
        "run_name": "ladder_parent_sys0625",
        "tier": "parent",
        "report_tiers": ["parent"],
        "template": "template_a",
        "golden": False,
        "censored": False,
        "restamped_config_hash": "staged-hash",
        "staged_global_seed": 3792114890,
        "ladder_campaign_uuid": "ladder-campaign",
        "ladder_config_hash": "ladder-config",
    }
    manifest = {
        "schema_version": 3,
        "design_freeze": {"version": 4},
        "name": "nonlinear_null_v1",
        "campaign_uuid": "campaign-uuid",
        "campaign": {
            "member_set": "production59",
            "arms": ["noisy_control"],
        },
        "code_revision": {"sha256": "revision-sha"},
        "amendments": [],
    }
    payload = {
        "schema_version": 3,
        "system_id": job["run_name"],
        "arm": "noisy_control",
        "arm_declaration": declaration,
        "sampler_seed": derive_sampler_seed(ENTROPY, 625, 2),
        "campaign_uuid": "campaign-uuid",
        "code_revision": {"sha256": "revision-sha"},
        "staged_config_hash": "staged-hash",
        "ladder_campaign_uuid": "ladder-campaign",
        "ladder_config_hash": "ladder-config",
        "censored": False,
        "tier": "parent",
        "rung": {"logm": 8.05},
        "noise_seed": 3792114890,
        "noise_replicate": 0,
        "noise_spawn_key": None,
        "fit_settings": deepcopy(FIT_BLOCK_FIXTURE),
        "kernel_sha256": "kernel-sha",
        "truth_kernel_sha256": "kernel-sha",
        "n_unmasked_pixels": 10,
        "smooth_status": "success",
        "subhalo_status": "success",
        "q_fit": 1.0,
    }
    return job, payload, manifest, {"arms": {"noisy_control": declaration},
                                    "fit": FIT_BLOCK_FIXTURE,
                                    "seeds": {"entropy": ENTROPY}}


class TestBuildArmConfig:
    def test_injected_arm_places_the_declared_subhalo(self):
        config = build_arm_config(
            _staged_config(),
            ARMS_FIXTURE["asimov_injected"],
            _rung_payload(),
            FIT_BLOCK_FIXTURE,
        )
        subhalo = config["lensing"]["subhalo"]
        assert subhalo["enabled"] is True
        assert subhalo["mass"] == pytest.approx(10.0**8.05)
        assert subhalo["position"] == {
            "type": "direct",
            "centre": [0.25, -0.85],
        }
        assert config["psf"]["kernel"]["shape_native"] == [51, 51]
        assert config["modeling"]["fit_psf"]["mode"] == "delta"
        assert (
            config["modeling"]["fit_psf"]["delta"]["amplitude_rms_nm"] == 0.0
        )
        assert config["plotting"] == {"enabled": False}

    def test_control_arm_keeps_the_subhalo_out_of_the_truth(self):
        config = build_arm_config(
            _staged_config(),
            ARMS_FIXTURE["noisy_control"],
            _rung_payload(),
            FIT_BLOCK_FIXTURE,
        )
        assert config["lensing"]["subhalo"]["enabled"] is False

    def test_kernel_disagreement_fails_closed(self):
        staged = _staged_config()
        staged["psf"]["kernel"]["shape_native"] = [999, 999]
        with pytest.raises(ValueError, match="declared fit kernel"):
            build_arm_config(
                staged,
                ARMS_FIXTURE["asimov_injected"],
                _rung_payload(),
                FIT_BLOCK_FIXTURE,
            )

    def test_staged_config_is_not_mutated(self):
        staged = _staged_config()
        reference = deepcopy(staged)
        build_arm_config(
            staged,
            ARMS_FIXTURE["asimov_injected"],
            _rung_payload(),
            FIT_BLOCK_FIXTURE,
        )
        assert staged == reference


class TestDirectionOverride:
    def test_override_sets_amplitude_and_seed_without_mutating_staged(self):
        """A direction is applied only to a private arm configuration."""
        staged = _staged_config()
        reference = deepcopy(staged)
        arm_config = build_arm_config(
            staged,
            DELTA_ARM_FIXTURE,
            _rung_payload(),
            FIT_BLOCK_FIXTURE,
        )
        original_arm = deepcopy(arm_config)
        configured, seed = apply_direction_override(
            arm_config,
            DELTA_ARM_FIXTURE,
            ENTROPY,
            625,
            2,
        )
        assert configured["modeling"]["fit_psf"]["delta"][
            "amplitude_rms_nm"
        ] == 5.0
        assert configured["modeling"]["fit_psf"]["delta"]["seed"] == seed
        assert seed == derive_direction_seed(ENTROPY, 2, 625)
        assert arm_config == original_arm
        assert staged == reference

    def test_missing_direction_for_delta_arm_fails_closed(self):
        with pytest.raises(ValueError, match="requires --direction"):
            validate_direction_argument(DELTA_ARM_FIXTURE, None)

    def test_direction_on_non_delta_arm_fails_closed(self):
        with pytest.raises(ValueError, match="only valid"):
            validate_direction_argument(ARMS_FIXTURE["noisy_control"], 1)

    def test_undeclared_direction_fails_closed(self):
        with pytest.raises(ValueError, match="not declared"):
            validate_direction_argument(DELTA_ARM_FIXTURE, 8)


class TestHarvestIntegrity:
    def test_ladder_campaign_binding_mismatch_is_reported(self):
        job, payload, manifest, protocol = _verification_fixture()
        payload["ladder_campaign_uuid"] = "wrong-campaign"
        findings = _verify_row(
            job, "noisy_control", payload, manifest, protocol
        )
        assert any("ladder_campaign_uuid" in finding for finding in findings)

    def test_censored_mismatch_is_reported(self):
        job, payload, manifest, protocol = _verification_fixture()
        job["censored"] = True
        findings = _verify_row(
            job, "noisy_control", payload, manifest, protocol
        )
        assert any("artifact censored" in finding for finding in findings)

    def test_successful_pair_with_no_q_fit_is_reported(self):
        job, payload, manifest, protocol = _verification_fixture()
        payload["q_fit"] = None
        findings = _verify_row(
            job, "noisy_control", payload, manifest, protocol
        )
        assert any("q_fit is None" in finding for finding in findings)

    def test_censored_top_rung_must_be_the_ceiling(self):
        job, payload, manifest, protocol = _verification_fixture()
        job["censored"] = True
        payload["censored"] = True
        payload["rung"]["logm"] = 9.4
        findings = _verify_row(
            job, "noisy_control", payload, manifest, protocol
        )
        assert any("logm" in finding and "9.5" in finding for finding in findings)

    def test_v4_job_arm_list_finding_path(self):
        job, payload, manifest, _ = _verification_fixture()
        protocol = {
            "arms": deepcopy(ARMS_FIXTURE),
            "member_sets": {"production59": {"n_systems": 1}},
            "campaigns": {
                "nonlinear_null_v1": {
                    "member_set": "production59",
                    "arms": ["noisy_control"],
                }
            },
        }
        manifest["n_systems"] = 1
        job["arms"] = {"asimov_injected": {"arm_index": 0}}
        manifest["jobs"] = [job]
        findings = _campaign_findings(manifest, protocol)
        assert any("job arms" in finding for finding in findings)

    def test_v4_missing_campaign_fails_closed_with_manifest_name(self):
        _, _, manifest, protocol = _verification_fixture()
        del manifest["campaign"]
        with pytest.raises(ValueError, match="manifest.json"):
            _campaign_findings(manifest, protocol, "manifest.json")

    def test_schema2_manifest_without_campaign_keeps_legacy_path(self):
        _, _, manifest, protocol = _verification_fixture()
        manifest["schema_version"] = 2
        manifest["design_freeze"]["version"] = 3
        del manifest["campaign"]
        assert _campaign_findings(manifest, protocol) == []


class TestScienceV3MissingValues:
    def test_injected_and_control_blocks_report_none_counts(self):
        """None likelihoods are visible in every relevant v3 arm block."""
        rows = [
            {
                "system_id": "sys0001",
                "arm": "asimov_injected",
                "censored": False,
                "golden": False,
                "template": "template_a",
                "q_fit": None,
                "q_f_matched": 10.2,
            },
            {
                "system_id": "sys0001",
                "arm": "noisy_injected",
                "censored": False,
                "golden": False,
                "template": "template_a",
                "q_fit": None,
                "q_f_matched": 10.2,
            },
            {
                "system_id": "sys0001",
                "arm": "noisy_control",
                "censored": False,
                "golden": False,
                "template": "template_a",
                "q_fit": None,
                "delta_log_evidence": None,
            },
        ]
        science = _science_v3(rows)
        assert science["asimov_injected"]["q_fit_none"] == 1
        assert science["noisy_injected"]["q_fit_none"] == 1
        assert science["noisy_control"]["q_fit_none"] == 1

    def test_below_arm_excludes_none_from_tested_denominator(self):
        rows = [
            {
                "system_id": "sys0001",
                "arm": "asimov_below",
                "censored": False,
                "q_fit": None,
            },
            {
                "system_id": "sys0002",
                "arm": "asimov_below",
                "censored": False,
                "q_fit": 5.0,
            },
            {
                "system_id": "sys0003",
                "arm": "asimov_below",
                "censored": False,
                "q_fit": 12.0,
            },
        ]
        below = _science_v3(rows)["asimov_below"]
        assert below["n"] == 3
        assert below["n_tested"] == 2
        assert below["q_fit_none"] == 1
        assert below["below_threshold"] == 1
        assert below["below_rung_consistency_fraction"] == pytest.approx(0.5)
        assert below["exceedances"] == ["sys0003 q_fit 12.0"]


class TestEligibleArms:
    def test_sample_rules(self):
        member = {"censored": False, "golden": False}
        assert eligible_arms(member, ARMS_FIXTURE) == [
            "asimov_injected", "noisy_control", "asimov_below",
        ]
        censored = {"censored": True, "golden": False}
        assert eligible_arms(censored, ARMS_FIXTURE) == [
            "asimov_injected", "noisy_control",
        ]
        golden = {"censored": False, "golden": True}
        assert eligible_arms(golden, ARMS_FIXTURE) == [
            "asimov_injected", "noisy_control", "asimov_below",
            "asimov_fixed_bridge",
        ]

    def test_campaign_arm_intersection(self):
        """Campaign membership removes otherwise eligible arms."""
        member = {"censored": False, "golden": True}
        assert eligible_arms(
            member,
            ARMS_FIXTURE,
            ["noisy_control", "asimov_fixed_bridge"],
        ) == ["noisy_control", "asimov_fixed_bridge"]

    def test_all_psf_knowledge_arms_are_eligible_for_selected_members(self):
        """The eight delta arms apply to every selected, non-censored member."""
        arms = {
            f"delta_{index}": {
                **DELTA_ARM_FIXTURE,
                "arm_index": index,
                "fit_psf_delta": {
                    "amplitude_rms_nm": float(index),
                    "directions": [1, 2, 3],
                },
            }
            for index in range(16, 24)
        }
        assert eligible_arms(
            {"censored": False, "golden": False},
            arms,
        ) == [f"delta_{index}" for index in range(16, 24)]


class TestSmokeJobs:
    def test_smallest_member_per_template(self):
        jobs = [
            {
                "template": "a", "image_side_px": 500, "run_name": "r1",
                "censored": False,
            },
            {
                "template": "a", "image_side_px": 400, "run_name": "r2",
                "censored": False,
            },
            {
                "template": "b", "image_side_px": 600, "run_name": "r3",
                "censored": False,
            },
        ]
        smokes = smoke_jobs(jobs, "smallest_image_per_template")
        assert [job["run_name"] for job in smokes] == ["r2", "r3"]

    def test_smallest_non_censored_member_per_template(self):
        jobs = [
            {
                "template": "a", "image_side_px": 300, "run_name": "r0",
                "censored": True,
            },
            {
                "template": "a", "image_side_px": 400, "run_name": "r1",
                "censored": False,
            },
            {
                "template": "b", "image_side_px": 500, "run_name": "r2",
                "censored": False,
            },
        ]
        smokes = smoke_jobs(
            jobs, "smallest_image_non_censored_per_template"
        )
        assert [job["run_name"] for job in smokes] == ["r1", "r2"]

    def test_smallest_image_golden_member(self):
        jobs = [
            {
                "template": "a", "image_side_px": 508, "run_name": "gold2",
                "golden": True,
            },
            {
                "template": "b", "image_side_px": 600, "run_name": "gold1",
                "golden": True,
            },
            {
                "template": "c", "image_side_px": 400, "run_name": "plain",
                "golden": False,
            },
        ]
        smokes = smoke_jobs(jobs, "smallest_image_golden")
        assert [job["run_name"] for job in smokes] == ["gold2"]


class TestSpearman:
    def test_perfect_monotone_relations(self):
        first = [1.0, 2.0, 5.0, 9.0]
        assert spearman_rank_correlation(first, [2.0, 4.0, 8.0, 16.0]) == 1.0
        assert spearman_rank_correlation(first, [8.0, 7.0, 3.0, 1.0]) == -1.0

    def test_hand_computed_tied_case(self):
        # Ranks: first (1, 2.5, 2.5, 4), second (2, 1, 3, 4). Centred,
        # sum(ab) = 3.0 and sqrt(sum(a^2) sum(b^2)) = sqrt(4.5 * 5.0),
        # so the rank Pearson correlation is 3/sqrt(22.5) = 2/sqrt(10).
        value = spearman_rank_correlation(
            [1.0, 2.0, 2.0, 3.0], [5.0, 4.0, 6.0, 7.0]
        )
        assert value == pytest.approx(2.0/np.sqrt(10.0))

    def test_constant_input_fails_closed(self):
        with pytest.raises(ValueError, match="constant"):
            spearman_rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


class TestExpectedProvenance:
    JOB = {
        "run_name": "ladder_selected_sys0069",
        "restamped_config_hash": "hash-original",
    }
    MANIFEST = {
        "code_revision": {"sha256": "rev-campaign"},
        "amendments": [{
            "reason": "fix",
            "code_revision": {"sha256": "rev-amended"},
            "jobs": {
                "ladder_selected_sys0069/asimov_fixed_bridge": {
                    "restamped_config_hash": "hash-amended",
                },
            },
        }],
    }

    def test_unamended_arm_keeps_campaign_provenance(self):
        assert expected_provenance(
            self.JOB, "asimov_injected", self.MANIFEST
        ) == ("rev-campaign", "hash-original")

    def test_amended_arm_resolves_to_amendment(self):
        assert expected_provenance(
            self.JOB, "asimov_fixed_bridge", self.MANIFEST
        ) == ("rev-amended", "hash-amended")

    def test_manifest_without_amendments(self):
        manifest = {"code_revision": {"sha256": "rev-campaign"}}
        assert expected_provenance(
            self.JOB, "asimov_fixed_bridge", manifest
        ) == ("rev-campaign", "hash-original")


class TestSampleMembers:
    def _stage_tier(self, run_dir, tier, indices, golden=(), censored=()):
        configs = run_dir/"configs"
        configs.mkdir(parents=True)
        for index in indices:
            run_name = f"ladder_{tier}_sys{index:04d}"
            (configs/f"{run_name}.yaml").write_text("run_name: x\n")
            outputs = run_dir/"outputs"/run_name
            outputs.mkdir(parents=True)
            np.savez(
                outputs/"ladder_result.npz",
                campaign_uuid=f"{tier}-campaign",
                config_hash=f"{tier}-config-{index}",
                golden=np.bool_(index in golden),
                m_best=np.float64(
                    float("nan") if index in censored else 8.0
                ),
            )

    def test_declared_sample_rule(self, tmp_path):
        parent_indices = list(range(48))
        selected_indices = [728] + list(range(100, 111))
        parent_indices[0] = 728
        self._stage_tier(
            tmp_path/"parent", "parent", parent_indices, censored={3, 4}
        )
        self._stage_tier(
            tmp_path/"selected", "selected", selected_indices,
            golden={728, 100},
        )
        members = sample_members(tmp_path/"parent", tmp_path/"selected")
        assert len(members) == 59
        overlap = [m for m in members if len(m["report_tiers"]) > 1]
        assert len(overlap) == 1
        assert overlap[0]["system_id"] == "sys0728"
        assert overlap[0]["tier"] == "parent"
        assert overlap[0]["report_tiers"] == ["parent", "selected"]
        # The overlap member's golden flag comes from its selected
        # artifact even though its parent artifact is not golden.
        assert overlap[0]["golden"] is True
        by_id = {member["system_id"]: member for member in members}
        assert by_id["sys0003"]["censored"] is True
        assert by_id["sys0005"]["censored"] is False
        assert by_id["sys0100"]["golden"] is True
        assert by_id["sys0001"]["ladder_campaign_uuid"] == "parent-campaign"
        assert by_id["sys0001"]["ladder_config_hash"] == "parent-config-1"
        assert by_id["sys0100"]["ladder_campaign_uuid"] == "selected-campaign"
        assert by_id["sys0100"]["ladder_config_hash"] == "selected-config-100"

    def test_selected12_keeps_selected_report_tier_and_parent_overlap(self, tmp_path):
        parent_indices = list(range(48))
        selected_indices = [728] + list(range(100, 111))
        parent_indices[0] = 728
        self._stage_tier(tmp_path/"parent", "parent", parent_indices)
        self._stage_tier(
            tmp_path/"selected", "selected", selected_indices, golden={728}
        )
        members = sample_members(
            tmp_path/"parent",
            tmp_path/"selected",
            mode="selected12",
        )
        assert len(members) == 12
        overlap = [
            member for member in members if member["system_id"] == "sys0728"
        ]
        assert len(overlap) == 1
        assert overlap[0]["run_name"] == "ladder_parent_sys0728"

    def test_wrong_tier_count_fails_closed(self, tmp_path):
        self._stage_tier(tmp_path/"parent", "parent", range(47))
        self._stage_tier(tmp_path/"selected", "selected", range(100, 112))
        with pytest.raises(ValueError, match="expected 48"):
            sample_members(tmp_path/"parent", tmp_path/"selected")

    def _stage_validation(self, root, censored=()):
        run_dir = root/"run"
        jobs = []
        for index in range(106):
            run_name = f"ladder_validation_sys{index:04d}"
            jobs.append({
                "job_id": run_name,
                "overrides": {
                    "ladder": {
                        "validation_sample_member": index < 100,
                        "snr_top12_member": index == 0 or index >= 100,
                    },
                    "stage0": {"system_id": f"sys{index:04d}"},
                },
                "scene": {},
            })
            config_dir = run_dir/"configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "run_name": run_name,
                "stage0": {
                    "source_template": f"template_{index % 5}",
                },
                "lensing": {"grid": {"shape": [400, 400]}},
            }
            (config_dir/f"{run_name}.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
            output_dir = run_dir/"outputs"/run_name
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                output_dir/"ladder_result.npz",
                campaign_uuid="validation-campaign",
                config_hash=f"validation-config-{index}",
                system_id=run_name,
                tier="validation",
                golden=np.bool_(False),
                m_best=np.float64(
                    float("nan") if index in censored else 8.0
                ),
            )
        manifest = {"campaign": {"jobs": jobs}}
        (root/"manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
        return run_dir

    def test_validation100_uses_validation_flag_and_keeps_overlap(self, tmp_path):
        run_dir = self._stage_validation(tmp_path, censored={3, 4})
        members = sample_members(
            validation_run=run_dir, mode="validation100"
        )
        assert len(members) == 100
        assert members[0]["system_id"] == "sys0000"
        assert all(member["tier"] == "validation" for member in members)
        assert all(member["golden"] is False for member in members)
        member_ids = {member["system_id"] for member in members}
        assert not member_ids.intersection(
            {f"sys{index:04d}" for index in range(100, 106)}
        )
        assert "sys0000" in member_ids
        assert members[0]["ladder_campaign_uuid"] == "validation-campaign"
        assert members[0]["ladder_config_hash"] == "validation-config-0"

    def test_validation100_wrong_count_fails_closed(self, tmp_path):
        run_dir = self._stage_validation(tmp_path)
        manifest_path = tmp_path/"manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["campaign"]["jobs"][0]["overrides"]["ladder"][
            "validation_sample_member"
        ] = False
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
        with pytest.raises(ValueError, match="expected 100"):
            sample_members(validation_run=run_dir, mode="validation100")

    def test_validation100_tier_mismatch_fails_closed(self, tmp_path):
        run_dir = self._stage_validation(tmp_path)
        artifact = (
            run_dir/"outputs"/"ladder_validation_sys0000"/
            "ladder_result.npz"
        )
        np.savez(
            artifact,
            campaign_uuid="validation-campaign",
            config_hash="validation-config-0",
            system_id="ladder_validation_sys0000",
            tier="parent",
            golden=np.bool_(False),
            m_best=np.float64(8.0),
        )
        with pytest.raises(ValueError, match="not 'validation'"):
            sample_members(validation_run=run_dir, mode="validation100")


class TestReusedPositions:
    def _stage_position_source(self, root, update=None):
        source_dir = root/"source"
        output_dir = source_dir/"outputs"/"ladder_parent_sys0001"
        output_dir.mkdir(parents=True)
        (source_dir/"manifest.json").write_text(
            json.dumps({
                "campaign_uuid": "source-campaign",
                "jobs": [{"run_name": "ladder_parent_sys0001"}],
            })
        )
        payload = {
            "system_id": "ladder_parent_sys0001",
            "fit_kernel_shape_native": [51, 51],
            "ladder_campaign_uuid": "ladder-campaign",
            "ladder_config_hash": "ladder-config",
            "censored": False,
            "rungs": {"top": {}, "below": {}},
        }
        if update:
            update(payload)
        (output_dir/"injection_position.json").write_text(
            json.dumps(payload)
        )
        member = {
            "run_name": "ladder_parent_sys0001",
            "ladder_campaign_uuid": "ladder-campaign",
            "ladder_config_hash": "ladder-config",
            "censored": False,
        }
        return source_dir, member

    @pytest.mark.parametrize(
        "update,match",
        [
            (lambda payload: payload.update(
                {"ladder_campaign_uuid": "wrong"}
            ), "ladder campaign uuid"),
            (lambda payload: payload.update(
                {"ladder_config_hash": "wrong"}
            ), "ladder config hash"),
            (lambda payload: payload.update({"censored": True}), "censored"),
            (lambda payload: payload["rungs"].pop("below"), "below rung"),
        ],
    )
    def test_reused_positions_require_member_identity_and_complete_rungs(
        self, tmp_path, update, match
    ):
        source_dir, member = self._stage_position_source(tmp_path, update)
        with pytest.raises(ValueError, match=match):
            _validate_reused_positions(
                source_dir, [member], "source-campaign", [51, 51], False
            )


class TestSourceBindings:
    def test_generator_and_harvest_require_hash_bound_clean_sources(self, tmp_path):
        """Both source consumers verify bytes and the CLEAN review."""
        source_dir = tmp_path/"source"
        harvest_dir = source_dir/"harvest"
        harvest_dir.mkdir(parents=True)
        uuid_value = "source-campaign"
        (source_dir/"manifest.json").write_text(
            json.dumps({"campaign_uuid": uuid_value})
        )
        harvest_path = harvest_dir/"harvest.json"
        review_path = harvest_dir/"review.json"
        harvest_path.write_text(
            json.dumps({"campaign_uuid": uuid_value, "rows": []})
        )
        review_path.write_text(json.dumps({"integrity": "CLEAN"}))

        def digest(path):
            """Return a temporary fixture file's SHA-256 digest."""
            return hashlib.sha256(path.read_bytes()).hexdigest()

        declaration = {
            "campaign": "nonlinear_validation_v1",
            "campaign_uuid": uuid_value,
            "harvest": "source/harvest/harvest.json",
            "harvest_sha256": digest(harvest_path),
            "review_sha256": digest(review_path),
        }
        echo = _validate_source_files(
            source_dir, declaration, "pooled_source"
        )
        assert echo["harvest"] == str(harvest_path.resolve())
        assert echo["review_integrity"] == "CLEAN"
        rows, resolved_harvest, resolved_review = _load_declared_source_rows(
            declaration, tmp_path/"current", "pooled_source"
        )
        assert rows == []
        assert resolved_harvest == harvest_path
        assert resolved_review == review_path

        review_path.write_text(json.dumps({"integrity": "FINDINGS"}))
        declaration["review_sha256"] = digest(review_path)
        with pytest.raises(ValueError, match="not CLEAN"):
            _validate_source_files(source_dir, declaration, "pooled_source")


class TestNoiseReplicate:
    def test_replicate_override_changes_only_private_config(self):
        staged = _staged_config()
        declaration = dict(ARMS_FIXTURE["noisy_control"])
        declaration["noise_replicate"] = 5
        config, seed, replicate, spawn_key = apply_noise_replicate(
            staged, declaration, ENTROPY, 625
        )
        assert staged["global_seed"] == 3792114890
        assert config["global_seed"] == seed == derive_noise_seed(
            ENTROPY, 5, 625
        )
        assert replicate == 5
        assert spawn_key == [6, 5, 625]

    def test_non_replicate_preserves_seed_and_returns_zero(self):
        staged = _staged_config()
        config, seed, replicate, spawn_key = apply_noise_replicate(
            staged, ARMS_FIXTURE["noisy_control"], ENTROPY, 625
        )
        assert config["global_seed"] == staged["global_seed"]
        assert seed == staged["global_seed"]
        assert replicate == 0
        assert spawn_key is None

    def test_asimov_replicate_fails_closed(self):
        declaration = dict(ARMS_FIXTURE["asimov_injected"])
        declaration["noise_replicate"] = 1
        with pytest.raises(ValueError, match="asimov"):
            apply_noise_replicate(
                _staged_config(), declaration, ENTROPY, 625
            )


class TestClopperPearson:
    def test_zero_and_full_counts_use_exact_limits(self):
        lower, upper = clopper_pearson(0, 590)
        assert lower == 0.0
        assert upper == pytest.approx(1.0 - 0.025**(1.0/590.0), abs=1e-9)
        lower, upper = clopper_pearson(590, 590)
        assert lower == pytest.approx(0.025**(1.0/590.0), abs=1e-9)
        assert upper == 1.0
