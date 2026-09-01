"""Unit tests for the nonlinear-validation campaign scripts."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

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
    eligible_arms,
    sample_members,
    smoke_jobs,
)
from harvest_nonlinear_validation import (  # noqa: E402
    expected_provenance,
    spearman_rank_correlation,
)
from run_nonlinear_validation import (  # noqa: E402
    build_arm_config,
    derive_sampler_seed,
    system_index,
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
            for arm_index in range(7)
        }
        assert len(seeds) == 7000


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


class TestSmokeJobs:
    def test_smallest_member_per_template(self):
        jobs = [
            {"template": "a", "image_side_px": 500, "run_name": "r1"},
            {"template": "a", "image_side_px": 400, "run_name": "r2"},
            {"template": "b", "image_side_px": 600, "run_name": "r3"},
        ]
        smokes = smoke_jobs(jobs)
        assert [job["run_name"] for job in smokes] == ["r2", "r3"]


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

    def test_wrong_tier_count_fails_closed(self, tmp_path):
        self._stage_tier(tmp_path/"parent", "parent", range(47))
        self._stage_tier(tmp_path/"selected", "selected", range(100, 112))
        with pytest.raises(ValueError, match="expected 48"):
            sample_members(tmp_path/"parent", tmp_path/"selected")
