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
    injection_logm,
)
from generate_nonlinear_validation_campaign import sample_members  # noqa: E402
from run_nonlinear_validation import (  # noqa: E402
    ARMS,
    build_arm_config,
    derive_sampler_seed,
    system_index,
)


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


class TestArgmaxInsideAperture:
    def test_peak_outside_aperture_is_ignored(self):
        y = np.array([-1.0, 0.0, 1.0, 2.0])
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        q = np.zeros((4, 4))
        q[3, 3] = 100.0
        q[1, 2] = 7.0
        position, q_max = argmax_inside_aperture(y, x, q, (0.0, 0.0), 1.2)
        assert position == (0.0, 1.0)
        assert q_max == 7.0

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
        assert derive_sampler_seed(625, 0) == 1844900749
        assert derive_sampler_seed(625, 1) == 2981609798
        assert derive_sampler_seed(625, 2) == 2855392809
        assert derive_sampler_seed(728, 0) == 167413308
        assert derive_sampler_seed(7, 2) == 3196177908

    def test_seeds_are_unique_across_the_campaign(self):
        seeds = {
            derive_sampler_seed(index, arm["arm_index"])
            for index in range(1000)
            for arm in ARMS.values()
        }
        assert len(seeds) == 3000


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


def _injection() -> dict:
    return {
        "system_id": "ladder_parent_sys0625",
        "injection_logm": 8.05,
        "injection_mass_msun": 10.0**8.05,
        "position_yx_arcsec": [0.25, -0.85],
        "q_at_position": 10.4,
        "censored": False,
    }


class TestBuildArmConfig:
    def test_injected_arm_places_the_declared_subhalo(self):
        staged = _staged_config()
        config = build_arm_config(staged, "asimov_injected", _injection())
        subhalo = config["lensing"]["subhalo"]
        assert subhalo["enabled"] is True
        assert subhalo["mass"] == pytest.approx(10.0**8.05)
        assert subhalo["position"] == {
            "type": "direct",
            "centre": [0.25, -0.85],
        }
        assert config["psf"]["kernel"]["shape_native"] == [999, 999]
        assert config["modeling"]["fit_psf"]["mode"] == "delta"
        assert (
            config["modeling"]["fit_psf"]["delta"]["amplitude_rms_nm"] == 0.0
        )
        assert config["plotting"] == {"enabled": False}

    def test_control_arm_keeps_the_subhalo_out_of_the_truth(self):
        config = build_arm_config(
            _staged_config(), "noisy_control", _injection()
        )
        assert config["lensing"]["subhalo"]["enabled"] is False

    def test_staged_config_is_not_mutated(self):
        staged = _staged_config()
        reference = deepcopy(staged)
        build_arm_config(staged, "noisy_injected", _injection())
        assert staged == reference


class TestSampleMembers:
    def _stage_tier(self, run_dir, tier, indices):
        configs = run_dir/"configs"
        configs.mkdir(parents=True)
        for index in indices:
            run_name = f"ladder_{tier}_sys{index:04d}"
            (configs/f"{run_name}.yaml").write_text("run_name: x\n")
            outputs = run_dir/"outputs"/run_name
            outputs.mkdir(parents=True)
            (outputs/"ladder_result.npz").write_bytes(b"")

    def test_declared_sample_rule(self, tmp_path):
        parent_indices = list(range(48))
        selected_indices = [728] + list(range(100, 111))
        parent_indices[0] = 728
        self._stage_tier(tmp_path/"parent", "parent", parent_indices)
        self._stage_tier(tmp_path/"selected", "selected", selected_indices)
        members = sample_members(tmp_path/"parent", tmp_path/"selected")
        assert len(members) == 59
        overlap = [m for m in members if len(m["report_tiers"]) > 1]
        assert len(overlap) == 1
        assert overlap[0]["system_id"] == "sys0728"
        assert overlap[0]["tier"] == "parent"
        assert overlap[0]["report_tiers"] == ["parent", "selected"]

    def test_wrong_tier_count_fails_closed(self, tmp_path):
        self._stage_tier(tmp_path/"parent", "parent", range(47))
        self._stage_tier(tmp_path/"selected", "selected", range(100, 112))
        with pytest.raises(ValueError, match="expected 48"):
            sample_members(tmp_path/"parent", tmp_path/"selected")
