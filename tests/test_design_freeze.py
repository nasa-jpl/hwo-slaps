"""Contracts for the canonical DesignFreeze artifact and its loader.

The freeze file's own digest is deliberately not pinned here. The freeze
is expected to be amended at ratification, and a self-pinned hash would
turn every legitimate amendment into a test failure. What is pinned is
the rule that makes the digest meaningful: the required blocks exist,
every bound artifact still carries the digest the freeze records, and
every constant pinned in the freeze equals the module constant it
governs.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from hwoslaps.campaign import design_freeze as df


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT/"configs"/"design"/"design_freeze_v1.yaml"
COMMITTED_PRE_REGISTRATION = "configs/design/selection_rule_v2.md"


@pytest.fixture(scope="module")
def freeze():
    """Load the committed design freeze once."""
    return df.load_design_freeze(FREEZE_PATH)


def _write_freeze(directory, document):
    """Write one freeze document to a temporary file and return its path."""
    path = Path(directory)/"design_freeze.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=True)
    return path


def test_committed_freeze_loads_and_validates(freeze):
    """The committed artifact passes its own validation."""
    assert freeze["schema_version"] == df.DESIGN_FREEZE_SCHEMA_VERSION
    assert freeze["freeze"]["name"] == "design_freeze_v1"
    assert freeze["freeze"]["status"] == "ratified"
    assert freeze["freeze"]["version"] == 5


def test_required_blocks_are_all_present(freeze):
    """Every block the campaign consumes exists in the artifact."""
    for block in df.REQUIRED_BLOCKS:
        assert block in freeze, block


def test_counts_and_strata_are_the_ruled_ones(freeze):
    """Stage 0 is 1000 with 48/12/5 strata and a representative parent."""
    assert freeze["stage0"]["n_systems"] == 1000
    assert freeze["strata"]["parent"]["size"] == 48
    assert freeze["strata"]["selected"]["size"] == 12
    assert freeze["strata"]["golden"]["size"] == 5
    assert (
        freeze["strata"]["parent"]["mode"] == "stratified_representative_subsample"
    )
    assert freeze["strata"]["parent"]["provisional"] is True


def test_detector_and_throughput_carry_the_p0_1_ruling(freeze):
    """0.2 e- per read is central and 0.21 is the ruled baseline."""
    detector = freeze["observing"]["detector"]
    assert detector["read_noise_per_read_e"] == 0.2
    assert detector["effective_read_noise_e"] == pytest.approx(
        0.2*2.0**0.5, rel=1e-12
    )
    assert freeze["observing"]["throughput"]["baseline"] == 0.21
    assert "2.5" not in str(detector["read_noise_per_read_e"])


def test_r_arms_are_labelled_with_their_meanings(freeze):
    """Each R arm declares a label, a meaning and its two levers."""
    arms = freeze["observing"]["r_arms"]["arms"]
    assert set(arms) >= {"R0", "R1", "R2", "R3"}
    assert arms["R0"]["throughput"] == 0.21
    assert arms["R1"]["throughput"] == 0.504
    assert arms["R2"]["source_magnitude_ab"] == 23.795
    assert arms["R3"]["source_magnitude_ab"] == 23.795
    assert "silver" in arms["R1"]["meaning"].lower() or "silver" in str(
        freeze["observing"]["throughput"]["optimistic_arm_caveat"]
    ).lower()
    assert "brightest-decile" in arms["R2"]["meaning"]
    assert "caveat" not in arms["R2"]


def test_aperture_factor_is_pinned_once_and_matches_both_modules(freeze):
    """The freeze is the single source of truth for the factor 2."""
    from hwoslaps.analysis import selection_score
    from hwoslaps.lensing import critical_curve

    assert freeze["aperture"]["theta_e_factor"] == 2.0
    assert critical_curve.DEFAULT_APERTURE_THETA_E_FACTOR == 2.0
    assert selection_score.APERTURE_THETA_E_MULTIPLE == 2.0
    assert freeze["aperture"]["computational_margin_fraction"] == 0.1
    assert (
        critical_curve.DEFAULT_COMPUTATIONAL_MARGIN_FRACTION
        == freeze["aperture"]["computational_margin_fraction"]
    )
    algorithm = freeze["aperture"]["theta_e_algorithm"]
    assert algorithm["algorithm_id"] == critical_curve.ALGORITHM_ID
    assert algorithm["choice_rule_id"] == critical_curve.CHOICE_RULE_ID


def test_selection_block_restates_the_frozen_score(freeze):
    """The score, cuts and tier sizes match the T4 module."""
    from hwoslaps.analysis import selection_score

    selection = freeze["selection"]
    assert selection["score"]["expression"] == "score = z(log S) + z(log C)"
    assert selection["observables"]["complexity"].startswith(
        "C = theta_res^2 * G / S^2"
    )
    assert selection["floor_cuts"]["theta_e_arcsec_min"] == (
        selection_score.FLOOR_THETA_E_ARCSEC
    )
    assert selection["floor_cuts"]["arc_snr_min"] == selection_score.FLOOR_ARC_SNR
    assert selection["tier_sizes"]["selected"] == selection_score.SELECTED_TIER_SIZE
    assert selection["tier_sizes"]["golden"] == selection_score.GOLDEN_TIER_SIZE
    assert set(
        variant["id"] for variant in selection["score"]["variants"]
    ) == {"s_only", "s_plus_c", "oracle_by_sensitivity"}


def test_mass_ladder_policy_is_the_ruled_one(freeze):
    """Coarse 0.25 dex over 6.0-9.5, refine 0.1, two zero rungs down."""
    ladder = freeze["mass_ladder"]
    assert ladder["coarse"] == {"step_dex": 0.25, "low": 6.0, "high": 9.5}
    assert ladder["refine"]["step_dex"] == 0.1
    assert ladder["extend_down"]["zero_rungs"] == 2
    assert "M50" in ladder["extend_up"]["stop"]


def test_seed_streams_are_declared_in_full(freeze):
    """Every stream, the replicate list and the engine seed rule exist."""
    streams = freeze["seeds"]["streams"]
    assert freeze["seeds"]["entropy"] == 20260823
    assert streams["parent_design"]["spawn_key"] == [0]
    assert streams["primary_noise"]["spawn_key"] == [1]
    assert streams["rank_stability_noise"]["spawn_key"] == [2]
    assert streams["null_noise"]["spawn_key"] == [6]
    assert streams["template_permutation"]["spawn_key"] == [3]
    assert streams["bootstrap"]["spawn_key"] == [4]
    assert streams["rank_stability_noise"]["replicates"] == 20
    assert streams["rank_stability_noise"]["replicate_indices"] == list(range(20))
    assert streams["null_noise"]["replicates"] == 9
    assert streams["null_noise"]["replicate_indices"] == list(range(1, 10))
    assert "generate_state" in streams["primary_noise"]["engine_seed_rule"]
    assert len(freeze["seeds"]["draw_order"]) == 10


def test_psf_knowledge_v5_block_is_fully_declared(freeze):
    """The v5 PSF knowledge block pins its rungs, gates and campaign."""
    knowledge = freeze["psf_knowledge_error"]
    residual = knowledge["residual_model"]
    assert knowledge["declared_v5"] == "2026-09-04"
    assert residual["amplitude_rms_nm_rungs"] == [
        0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0
    ]
    assert residual["endpoint_anchor_nm"] == 35.0
    assert residual["directions"] == 8
    assert residual["direction_indices"] == list(range(1, 9))
    assert residual["prior_table_sha256"] == (
        "bfbececdcbe5fb37a4abcb018b63544d47c4c37ecc1900a539d62755b740c488"
    )
    assert knowledge["gates"]["retention_q10_min"] == 0.9
    assert knowledge["gates"]["spurious_q90_max"] == 0.1
    assert knowledge["ratio_floor"]["cells"] == 33
    assert knowledge["ratio_floor"]["arcsec2"] == pytest.approx(33*0.0025)
    assert knowledge["member_set"] == {
        "name": "selected12",
        "source": "ladder_selected_v1/run",
        "source_campaign_uuid": "e1144d71-d6dd-4789-84c1-c37a9e045ea0",
        "n_systems": 12,
        "tier": "selected",
    }
    assert knowledge["campaigns"]["psf_knowledge_fisher_v1"]["phases"] == [
        "maps"
    ]


def test_psf_knowledge_direction_stream_and_arms_are_declared(freeze):
    """The direction stream and all eight delta arms are frozen."""
    stream = freeze["seeds"]["streams"]["psf_knowledge_direction"]
    assert stream["spawn_key"] == [7]
    assert stream["directions"] == 8
    assert stream["direction_indices"] == list(range(1, 9))
    arms = freeze["nonlinear_validation"]["arms"]
    delta_arms = [
        name for name, declaration in arms.items()
        if "fit_psf_delta" in declaration
    ]
    assert [arms[name]["arm_index"] for name in delta_arms] == list(range(16, 24))
    assert all(
        arms[name]["fit_psf_delta"]["directions"] == [1, 2, 3]
        for name in delta_arms
    )
    nonlinear = freeze["nonlinear_validation"]
    assert nonlinear["member_sets"]["selected12"]["n_systems"] == 12
    assert "reference_source" in nonlinear["campaigns"][
        "psf_knowledge_nonlinear_v1"
    ]
    assert "null_source" in nonlinear["campaigns"][
        "psf_knowledge_nonlinear_v1"
    ]


def test_nonlinear_extension_sets_and_campaigns_are_declared(freeze):
    """The v4 member sets and campaign arm lists are frozen."""
    nonlinear = freeze["nonlinear_validation"]
    assert set(nonlinear["member_sets"]) == {
        "production59", "validation100", "selected12"
    }
    assert set(nonlinear["campaigns"]) == {
        "nonlinear_null_v1",
        "nonlinear_validation100_v1",
        "psf_knowledge_nonlinear_v1",
    }
    assert nonlinear["campaigns"]["nonlinear_null_v1"]["arms"] == [
        f"noisy_control_r{index}" for index in range(1, 10)
    ]
    assert nonlinear["campaigns"]["nonlinear_validation100_v1"]["arms"] == [
        "asimov_injected",
        "noisy_injected",
        "noisy_control",
        "asimov_below",
    ]
    source = nonlinear["campaigns"]["nonlinear_null_v1"][
        "replicate_zero_source"
    ]
    assert source["harvest_sha256"] == (
        "a8f8fa33d53ee1ab32b88b08ae3d95ceaf51a0896283f017525472bb2359c993"
    )
    assert source["review_sha256"] == (
        "88042d9403aa49bfc4ea464ece3347a0ea2bf1c6938a73b7988b59212e49895c"
    )
    pooled = nonlinear["campaigns"]["nonlinear_validation100_v1"][
        "pooled_source"
    ]
    assert pooled["campaign"] == "nonlinear_validation_v1"
    assert pooled["campaign_uuid"] == source["campaign_uuid"]
    assert pooled["harvest_sha256"] == source["harvest_sha256"]
    assert pooled["review_sha256"] == source["review_sha256"]


def test_claim_labels_and_ceiling_flag(freeze):
    """Every central claim label is present and the ceiling flag is set."""
    labels = freeze["claim_labels"]
    assert freeze["foreground_free_ceiling"] is True
    assert "source-only information ceiling" in labels["central_result"]
    assert "idealized upper-bound" in labels["counts"]
    assert "reference ensemble" in labels["ensemble"]
    assert "templates" in labels["templates"]
    assert "survey population" in labels["forbidden_phrases"]


def test_nothing_remains_provisional_and_rulings_are_recorded(freeze):
    """The 2026-08-23 ratification emptied the provisional list."""
    assert freeze["provisional_items"] == []
    assert list(df.REQUIRED_PROVISIONAL_ITEMS) == []
    rulings = {item["id"] for item in freeze["ratifications"]}
    assert {
        "parent_design",
        "representative_48_semantics",
        "t4_noisy_g_labelling",
        "template_83935_resolution_caveat",
        "spacing_systematic",
        "throughput_bracket",
        "golden_magnitude_anchor",
    } <= rulings
    for item in freeze["ratifications"]:
        assert item["ruled"] == "2026-08-23"
        assert item["ruling"].strip()
    systematic = freeze["declared_systematics"]["spatial_sampling_qmax"]
    assert systematic["value_dex"] == -0.004


def test_parent_design_is_embedded_and_hash_referenced(freeze):
    """The whole B8 design travels inside the freeze with its digest."""
    parent = freeze["parent_design"]
    assert parent["schema"] == "hwoslaps.parent_design"
    assert parent["status"] == "provisional"
    assert parent["stage0"]["n_systems"] == freeze["stage0"]["n_systems"]
    assert len(parent["distributions"]) >= 12
    source = freeze["parent_design_source"]
    assert source["embedded"] is True
    assert len(source["sha256"]) == 64


def test_bound_artifact_hashes_match_the_files_on_disk(freeze):
    """Every committed artifact the freeze pins still hashes the same."""
    report = df.verify_bound_artifacts(freeze, root=REPO_ROOT)
    verified = report["verified"]
    assert verified["observing_reference"] == (
        freeze["observing"]["reference"]["sha256"]
    )
    for level in freeze["templates"]["levels"]:
        assert verified[f"template_{level['id']}"] == level["sha256"]
    assert len([name for name in verified if name.startswith("template_")]) == 5


def test_template_assets_hash_to_the_frozen_values(freeze):
    """The five template digests are recomputed from the asset bytes."""
    for level in freeze["templates"]["levels"]:
        path = REPO_ROOT/level["asset_path"]
        assert path.is_file(), path
        assert df.file_sha256(path) == level["sha256"]


def test_template_bank_fills_the_pool_exactly(freeze):
    """Five levels at 200 each fill the 1000-system pool."""
    templates = freeze["templates"]
    assert templates["count"] == 5
    assert templates["per_level"] == 200
    assert templates["per_level"]*templates["count"] == freeze["stage0"]["n_systems"]


def test_loader_rejects_a_wrong_schema_version(freeze):
    """An unsupported schema version fails closed."""
    broken = copy.deepcopy(freeze)
    broken["schema_version"] = 99
    with pytest.raises(df.DesignFreezeError, match="schema_version"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_missing_block(freeze):
    """Dropping a required block fails closed."""
    broken = copy.deepcopy(freeze)
    del broken["mass_ladder"]
    with pytest.raises(df.DesignFreezeError, match="missing required blocks"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_detuned_aperture_factor(freeze):
    """A freeze factor that leaves the modules behind fails closed."""
    broken = copy.deepcopy(freeze)
    broken["aperture"]["theta_e_factor"] = 1.5
    with pytest.raises(df.DesignFreezeError, match="single source of truth"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_reinstated_conservative_read_noise(freeze):
    """The retired 2.5 e- per read cannot re-enter the design."""
    broken = copy.deepcopy(freeze)
    broken["observing"]["detector"]["read_noise_per_read_e"] = 2.5
    with pytest.raises(df.DesignFreezeError, match="P0-1"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_top_48_parent_tier(freeze):
    """Turning the parent tier into a score selection fails closed."""
    broken = copy.deepcopy(freeze)
    broken["strata"]["parent"]["mode"] = "top_by_score"
    with pytest.raises(df.DesignFreezeError, match="representative"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_cleared_ceiling_flag(freeze):
    """The foreground-free ceiling flag cannot be turned off silently."""
    broken = copy.deepcopy(freeze)
    broken["foreground_free_ceiling"] = False
    with pytest.raises(df.DesignFreezeError, match="ceiling-only"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_changed_provisional_item_list(freeze):
    """Reintroducing a provisional item after ratification fails closed."""
    broken = copy.deepcopy(freeze)
    broken["provisional_items"] = [{"id": "resurrected_item"}]
    with pytest.raises(df.DesignFreezeError, match="provisional_items"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_reordered_draw_order(freeze):
    """A draw order that leaves the parent design behind fails closed."""
    broken = copy.deepcopy(freeze)
    order = list(broken["seeds"]["draw_order"])
    broken["seeds"]["draw_order"] = order[1:] + order[:1]
    with pytest.raises(df.DesignFreezeError, match="draw_order"):
        df.validate_design_freeze(broken)


def test_verification_rejects_a_changed_asset(freeze, tmp_path):
    """A template asset whose bytes moved fails verification."""
    broken = copy.deepcopy(freeze)
    broken["templates"]["levels"][0]["sha256"] = "0"*64
    with pytest.raises(df.DesignFreezeError, match="does not match the frozen"):
        df.verify_bound_artifacts(broken, root=REPO_ROOT)


def test_loading_verifies_changed_template_bytes(freeze, tmp_path):
    """A freeze whose template digest moved cannot be loaded at all."""
    broken = copy.deepcopy(freeze)
    broken["templates"]["levels"][0]["sha256"] = "0"*64
    path = _write_freeze(tmp_path, broken)
    with pytest.raises(df.DesignFreezeError, match="does not match the frozen"):
        df.load_design_freeze(path)


def test_loading_verifies_the_observing_reference(freeze, tmp_path):
    """A freeze whose observing reference moved cannot be loaded at all."""
    broken = copy.deepcopy(freeze)
    broken["observing"]["reference"]["sha256"] = "0"*64
    path = _write_freeze(tmp_path, broken)
    with pytest.raises(df.DesignFreezeError, match="observing_reference"):
        df.load_design_freeze(path)


def test_loading_rejects_a_missing_committed_artifact(freeze, tmp_path):
    """A committed artifact that is not on disk is never optional."""
    broken = copy.deepcopy(freeze)
    broken["observing"]["reference"]["path"] = "configs/observing/absent.yaml"
    path = _write_freeze(tmp_path, broken)
    with pytest.raises(df.DesignFreezeError, match="does not exist"):
        df.load_design_freeze(path)


def test_declared_committed_pre_registration_is_verified(freeze, tmp_path):
    """A declared committed copy of the signed rule is re-hashed on load."""
    bound = copy.deepcopy(freeze)
    bound["selection"]["pre_registration"]["committed_path"] = (
        COMMITTED_PRE_REGISTRATION
    )
    loaded = df.load_design_freeze(_write_freeze(tmp_path, bound))
    report = df.verify_bound_artifacts(loaded, root=REPO_ROOT)
    assert report["verified"]["selection_pre_registration_committed"] == (
        freeze["selection"]["pre_registration"]["sha256"]
    )


def test_golden_anchor_is_bound_and_verified(freeze):
    """The A6-2 anchor document is a required committed binding."""
    anchor = freeze["observing"]["golden_anchor"]
    assert anchor["path"] == "configs/design/golden_magnitude_anchor.md"
    report = df.verify_bound_artifacts(freeze, root=REPO_ROOT)
    assert report["verified"]["golden_magnitude_anchor"] == anchor["sha256"]


def test_golden_anchor_mismatch_fails_closed(freeze):
    """A golden-anchor document with a different digest fails."""
    broken = copy.deepcopy(freeze)
    broken["observing"]["golden_anchor"]["sha256"] = "0" * 64
    with pytest.raises(df.DesignFreezeError, match="golden_magnitude_anchor"):
        df.verify_bound_artifacts(broken, root=REPO_ROOT)


def test_golden_anchor_missing_file_fails_closed(freeze):
    """A golden-anchor path that does not exist fails as committed."""
    broken = copy.deepcopy(freeze)
    broken["observing"]["golden_anchor"]["path"] = (
        "configs/design/no_such_anchor.md"
    )
    with pytest.raises(df.DesignFreezeError, match="does not exist"):
        df.verify_bound_artifacts(broken, root=REPO_ROOT)


def test_loader_rejects_a_malformed_golden_anchor_path(freeze):
    """A non-string golden-anchor path is a DesignFreezeError."""
    broken = copy.deepcopy(freeze)
    broken["observing"]["golden_anchor"]["path"] = 7
    with pytest.raises(df.DesignFreezeError, match="golden_anchor"):
        df.validate_design_freeze(broken)


def test_committed_pre_registration_mismatch_fails_closed(freeze, tmp_path):
    """A committed copy that does not carry the frozen digest fails."""
    broken = copy.deepcopy(freeze)
    broken["selection"]["pre_registration"]["committed_path"] = (
        "configs/design/design_freeze_v1.yaml"
    )
    path = _write_freeze(tmp_path, broken)
    with pytest.raises(df.DesignFreezeError) as excinfo:
        df.load_design_freeze(path)
    message = str(excinfo.value)
    assert "selection_pre_registration_committed" in message
    assert "design_freeze_v1.yaml" in message
    assert freeze["selection"]["pre_registration"]["sha256"] in message


def test_committed_pre_registration_missing_file_fails_closed(freeze, tmp_path):
    """A declared committed copy is required, never optional."""
    broken = copy.deepcopy(freeze)
    broken["selection"]["pre_registration"]["committed_path"] = (
        "configs/design/absent_selection_rule.md"
    )
    path = _write_freeze(tmp_path, broken)
    with pytest.raises(df.DesignFreezeError, match="does not exist"):
        df.load_design_freeze(path)


def test_absent_committed_path_leaves_verification_unchanged(freeze):
    """A freeze that declares no committed copy is verified without it."""
    without = copy.deepcopy(freeze)
    del without["selection"]["pre_registration"]["committed_path"]
    df.validate_design_freeze(without)
    report = df.verify_bound_artifacts(without, root=REPO_ROOT)
    assert "selection_pre_registration_committed" not in report["verified"]
    assert "selection_pre_registration_committed" not in report["absent"]


def test_loader_rejects_a_malformed_committed_path(freeze):
    """A committed path that is not a repo-relative string fails closed."""
    broken = copy.deepcopy(freeze)
    broken["selection"]["pre_registration"]["committed_path"] = 7
    with pytest.raises(df.DesignFreezeError, match="committed_path"):
        df.validate_design_freeze(broken)


def test_the_verification_opt_out_is_explicit(freeze, tmp_path):
    """A hash-only caller must ask for the unverified load by name."""
    broken = copy.deepcopy(freeze)
    broken["templates"]["levels"][0]["sha256"] = "0"*64
    path = _write_freeze(tmp_path, broken)
    loaded = df.load_design_freeze(path, skip_bound_artifact_verification=True)
    assert loaded["templates"]["levels"][0]["sha256"] == "0"*64


def test_extraction_settings_are_declared_in_full(freeze):
    """The grid and the guards the runner consumes are part of the design."""
    algorithm = freeze["aperture"]["theta_e_algorithm"]
    assert algorithm["extraction_grid"]["pixel_scale_arcsec"] == 0.01
    assert algorithm["extraction_grid"]["half_width_factor"] == 4.0
    assert algorithm["guards"]["closure_tolerance_pixels"] == 0.5
    assert algorithm["guards"]["border_margin_pixels"] == 2.0
    assert algorithm["guards"]["min_contour_vertices"] == 32


def test_loader_rejects_a_missing_extraction_grid(freeze):
    """Dropping the extraction grid fails closed."""
    broken = copy.deepcopy(freeze)
    del broken["aperture"]["theta_e_algorithm"]["extraction_grid"]
    with pytest.raises(df.DesignFreezeError, match="extraction_grid"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_non_positive_extraction_pixel_scale(freeze):
    """An extraction grid the extraction cannot use fails closed."""
    broken = copy.deepcopy(freeze)
    broken["aperture"]["theta_e_algorithm"]["extraction_grid"][
        "pixel_scale_arcsec"
    ] = 0.0
    with pytest.raises(df.DesignFreezeError, match="pixel_scale_arcsec"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_negative_border_margin(freeze):
    """A negative contour guard fails closed."""
    broken = copy.deepcopy(freeze)
    broken["aperture"]["theta_e_algorithm"]["guards"][
        "border_margin_pixels"
    ] = -1.0
    with pytest.raises(df.DesignFreezeError, match="border_margin_pixels"):
        df.validate_design_freeze(broken)


def test_loader_rejects_an_unusable_min_contour_vertices(freeze):
    """A vertex floor below the extraction's own minimum fails closed."""
    broken = copy.deepcopy(freeze)
    broken["aperture"]["theta_e_algorithm"]["guards"][
        "min_contour_vertices"
    ] = 3
    with pytest.raises(df.DesignFreezeError, match="min_contour_vertices"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_missing_null_noise_stream(freeze):
    """The v4 operational null stream is required."""
    broken = copy.deepcopy(freeze)
    del broken["seeds"]["streams"]["null_noise"]
    with pytest.raises(df.DesignFreezeError, match="null_noise"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_replicate_asimov_arm(freeze):
    """Noise replicates cannot be attached to Asimov data."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["arms"]["noisy_control_r1"][
        "dataset_kind"
    ] = "asimov"
    with pytest.raises(df.DesignFreezeError, match="dataset_kind"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_replicate_truth_subhalo(freeze):
    """Noise replicates cannot contain a truth subhalo."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["arms"]["noisy_control_r1"][
        "subhalo_in_truth"
    ] = True
    with pytest.raises(df.DesignFreezeError, match="subhalo_in_truth"):
        df.validate_design_freeze(broken)


def test_loader_rejects_an_undeclared_noise_replicate(freeze):
    """Every replicate index must belong to the frozen stream."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["arms"]["noisy_control_r1"][
        "noise_replicate"
    ] = 10
    with pytest.raises(df.DesignFreezeError, match="replicate_indices"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_campaign_undeclared_arm(freeze):
    """A campaign cannot name an arm absent from the protocol."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["campaigns"]["nonlinear_null_v1"][
        "arms"
    ].append("not_declared")
    with pytest.raises(df.DesignFreezeError, match="undeclared arm"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_campaign_undeclared_member_set(freeze):
    """A campaign must use a declared member set."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["campaigns"]["nonlinear_null_v1"][
        "member_set"
    ] = "not_declared"
    with pytest.raises(df.DesignFreezeError, match="member set"):
        df.validate_design_freeze(broken)


def test_loader_rejects_an_unknown_smoke_member_rule(freeze):
    """Smoke selection uses a closed enum."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["campaigns"][
        "nonlinear_null_v1"
    ]["smoke_rule"]["member"] = "unknown_member_rule"
    with pytest.raises(df.DesignFreezeError, match="unknown rule"):
        df.validate_design_freeze(broken)


def test_loader_rejects_nonconsecutive_null_replicates(freeze):
    """Null replicate indices must be exactly 1 through 9."""
    broken = copy.deepcopy(freeze)
    broken["seeds"]["streams"]["null_noise"]["replicate_indices"] = list(
        range(9)
    )
    with pytest.raises(df.DesignFreezeError, match="null_noise"):
        df.validate_design_freeze(broken)


def test_loader_rejects_missing_declared_replicate_arm(freeze):
    """Every declared null replicate arm is required by the stream rule."""
    broken = copy.deepcopy(freeze)
    del broken["nonlinear_validation"]["arms"]["noisy_control_r9"]
    broken["nonlinear_validation"]["campaigns"]["nonlinear_null_v1"][
        "arms"
    ].remove("noisy_control_r9")
    with pytest.raises(df.DesignFreezeError, match="noise_replicate"):
        df.validate_design_freeze(broken)


def test_loader_rejects_campaign_missing_one_replicate_arm(freeze):
    """A replicate campaign must carry the complete declared replicate set."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["campaigns"]["nonlinear_null_v1"][
        "arms"
    ].remove("noisy_control_r9")
    with pytest.raises(df.DesignFreezeError, match="noise_replicate"):
        df.validate_design_freeze(broken)


def test_loader_rejects_an_unbound_source_digest(freeze):
    """External nonlinear sources require both content digests."""
    broken = copy.deepcopy(freeze)
    del broken["nonlinear_validation"]["campaigns"]["nonlinear_null_v1"][
        "replicate_zero_source"
    ]["review_sha256"]
    with pytest.raises(df.DesignFreezeError, match="review_sha256"):
        df.validate_design_freeze(broken)


def test_loader_rejects_an_unbound_pooled_source(freeze):
    """The validation-100 pooled source carries the same binding contract."""
    broken = copy.deepcopy(freeze)
    del broken["nonlinear_validation"]["campaigns"][
        "nonlinear_validation100_v1"
    ]["pooled_source"]["harvest_sha256"]
    with pytest.raises(df.DesignFreezeError, match="harvest_sha256"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_missing_psf_knowledge_direction_stream(freeze):
    """The v5 direction stream cannot be removed from the freeze."""
    broken = copy.deepcopy(freeze)
    del broken["seeds"]["streams"]["psf_knowledge_direction"]
    with pytest.raises(df.DesignFreezeError, match="psf_knowledge_direction"):
        df.validate_design_freeze(broken)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda document: document["nonlinear_validation"]["arms"][
                "noisy_control_d2"
            ]["fit_psf_delta"].update({"amplitude_rms_nm": 3.0}),
            "amplitude_rms_nm",
        ),
        (
            lambda document: document["nonlinear_validation"]["arms"][
                "noisy_control_d2"
            ]["fit_psf_delta"].update({"directions": [9]}),
            "directions",
        ),
        (
            lambda document: document["nonlinear_validation"]["arms"][
                "asimov_injected"
            ].update({
                "fit_psf_delta": {
                    "amplitude_rms_nm": 2.0,
                    "directions": [1, 2, 3],
                }
            }),
            "dataset_kind",
        ),
        (
            lambda document: document["nonlinear_validation"]["arms"][
                "noisy_control_d2"
            ].update({
                "fit_mode": "fixed_template",
            }),
            "fit_mode",
        ),
        (
            lambda document: document["nonlinear_validation"]["arms"][
                "noisy_control_d2"
            ].update({
                "rung": "below",
            }),
            "rung",
        ),
        (
            lambda document: document["nonlinear_validation"]["arms"][
                "asimov_fixed_bridge"
            ].update({
                "fit_psf_delta": {
                    "amplitude_rms_nm": 2.0,
                    "directions": [1, 2, 3],
                }
            }),
            "dataset_kind",
        ),
    ],
)
def test_loader_rejects_invalid_psf_knowledge_delta_arms(
    freeze, mutation, match
):
    """Delta arms obey the frozen amplitude, direction and arm rules."""
    broken = copy.deepcopy(freeze)
    mutation(broken)
    with pytest.raises(df.DesignFreezeError, match=match):
        df.validate_design_freeze(broken)


def test_loader_rejects_delta_arm_noise_replicate(freeze):
    """Knowledge-error arms cannot consume the null-noise replicate stream."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["arms"]["noisy_control_d2"][
        "noise_replicate"
    ] = 1
    with pytest.raises(df.DesignFreezeError, match="noise_replicate"):
        df.validate_design_freeze(broken)


def test_loader_rejects_duplicate_psf_knowledge_pair(freeze):
    """The amplitude and truth-subhalo pair is unique across delta arms."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["arms"]["noisy_control_d5"][
        "fit_psf_delta"
    ]["amplitude_rms_nm"] = 2.0
    with pytest.raises(df.DesignFreezeError, match="duplicate"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_mixed_delta_campaign(freeze):
    """A nonlinear delta campaign cannot mix matched and delta arms."""
    broken = copy.deepcopy(freeze)
    broken["nonlinear_validation"]["campaigns"][
        "psf_knowledge_nonlinear_v1"
    ]["arms"].append("noisy_control")
    with pytest.raises(df.DesignFreezeError, match="mix"):
        df.validate_design_freeze(broken)


def test_loader_rejects_a_delta_campaign_without_null_source(freeze):
    """The nonlinear knowledge campaign must bind the matched null."""
    broken = copy.deepcopy(freeze)
    del broken["nonlinear_validation"]["campaigns"][
        "psf_knowledge_nonlinear_v1"
    ]["null_source"]
    with pytest.raises(df.DesignFreezeError, match="null_source"):
        df.validate_design_freeze(broken)


def test_loader_rejects_nonincreasing_psf_rungs(freeze):
    """Residual amplitude rungs are a strictly increasing sequence."""
    broken = copy.deepcopy(freeze)
    broken["psf_knowledge_error"]["residual_model"][
        "amplitude_rms_nm_rungs"
    ] = [0.0, 2.0, 1.0]
    with pytest.raises(df.DesignFreezeError, match="strictly increasing"):
        df.validate_design_freeze(broken)


def test_loader_rejects_endpoint_not_at_last_rung(freeze):
    """The endpoint anchor is the last declared residual rung."""
    broken = copy.deepcopy(freeze)
    broken["psf_knowledge_error"]["residual_model"]["endpoint_anchor_nm"] = 20.0
    with pytest.raises(df.DesignFreezeError, match="last"):
        df.validate_design_freeze(broken)


def test_loader_rejects_psf_knowledge_gate_outside_unit_interval(freeze):
    """Knowledge-error gates are strict unit-interval probabilities."""
    broken = copy.deepcopy(freeze)
    broken["psf_knowledge_error"]["gates"]["retention_q10_min"] = 1.1
    with pytest.raises(df.DesignFreezeError, match="between 0 and 1"):
        df.validate_design_freeze(broken)


def test_loader_rejects_wrong_psf_knowledge_ratio_floor_area(freeze):
    """The ratio-floor area is pinned to its integer cell count."""
    broken = copy.deepcopy(freeze)
    broken["psf_knowledge_error"]["ratio_floor"]["arcsec2"] = 0.08
    with pytest.raises(df.DesignFreezeError, match="ratio_floor.arcsec2"):
        df.validate_design_freeze(broken)


def test_loader_rejects_wrong_psf_knowledge_prior_digest(freeze):
    """The declared drift-prior bytes are hash-bound."""
    broken = copy.deepcopy(freeze)
    broken["psf_knowledge_error"]["residual_model"][
        "prior_table_sha256"
    ] = "0"*64
    with pytest.raises(df.DesignFreezeError, match="prior table sha256"):
        df.validate_design_freeze(broken)


def test_loader_rejects_unknown_psf_knowledge_smoke_member(freeze):
    """The Fisher smoke member selection is a closed enum."""
    broken = copy.deepcopy(freeze)
    broken["psf_knowledge_error"]["campaigns"][
        "psf_knowledge_fisher_v1"
    ]["smoke_rule"]["members"] = ["unknown"]
    with pytest.raises(df.DesignFreezeError, match="smoke_rule.members"):
        df.validate_design_freeze(broken)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda document: document["freeze"].pop("amendment_v5"),
            "amendment_v5",
        ),
        (
            lambda document: document["psf_knowledge_error"].update(
                {"rulings_v5": None}
            ),
            "rulings_v5",
        ),
        (
            lambda document: document["psf_knowledge_error"].update(
                {"success_criteria": None}
            ),
            "success_criteria",
        ),
        (
            lambda document: document["nonlinear_validation"].update(
                {"rulings_v5": {}}
            ),
            "nonlinear_validation.rulings_v5",
        ),
        (
            lambda document: document["nonlinear_validation"].update(
                {"declared_v5": None}
            ),
            "nonlinear_validation.declared_v5",
        ),
        (
            lambda document: document["nonlinear_validation"].update(
                {"declared_v5": "   "}
            ),
            "nonlinear_validation.declared_v5",
        ),
        (
            lambda document: document["nonlinear_validation"].update(
                {"success_criteria": [None]}
            ),
            "nonlinear_validation.success_criteria",
        ),
        (
            lambda document: document["psf_knowledge_error"].update(
                {"declared_v5": "yesterday"}
            ),
            "psf_knowledge_error.declared_v5",
        ),
    ],
)
def test_loader_rejects_a_semantically_hollow_v5_block(freeze, mutation, match):
    """A version-5 document must carry its amendment, rulings and criteria."""
    broken = copy.deepcopy(freeze)
    mutation(broken)
    with pytest.raises(df.DesignFreezeError, match=match):
        df.validate_design_freeze(broken)
