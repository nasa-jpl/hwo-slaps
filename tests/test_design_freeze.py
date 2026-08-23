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

from hwoslaps.campaign import design_freeze as df


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT/"configs"/"design"/"design_freeze_v1.yaml"


@pytest.fixture(scope="module")
def freeze():
    """Load the committed design freeze once."""
    return df.load_design_freeze(FREEZE_PATH)


def test_committed_freeze_loads_and_validates(freeze):
    """The committed artifact passes its own validation."""
    assert freeze["schema_version"] == df.DESIGN_FREEZE_SCHEMA_VERSION
    assert freeze["freeze"]["name"] == "design_freeze_v1"
    assert freeze["freeze"]["status"] == "provisional"


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
    assert arms["R2"]["source_magnitude_ab"] == 23.345
    assert arms["R3"]["source_magnitude_ab"] == 23.345
    assert "silver" in arms["R1"]["meaning"].lower() or "silver" in str(
        freeze["observing"]["throughput"]["optimistic_arm_caveat"]
    ).lower()
    assert "outside the declared parent" in arms["R2"]["caveat"]


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
    assert streams["template_permutation"]["spawn_key"] == [3]
    assert streams["bootstrap"]["spawn_key"] == [4]
    assert streams["rank_stability_noise"]["replicates"] == 20
    assert streams["rank_stability_noise"]["replicate_indices"] == list(range(20))
    assert "generate_state" in streams["primary_noise"]["engine_seed_rule"]
    assert len(freeze["seeds"]["draw_order"]) == 10


def test_claim_labels_and_ceiling_flag(freeze):
    """Every central claim label is present and the ceiling flag is set."""
    labels = freeze["claim_labels"]
    assert freeze["foreground_free_ceiling"] is True
    assert "source-only information ceiling" in labels["central_result"]
    assert "idealized upper-bound" in labels["counts"]
    assert "reference ensemble" in labels["ensemble"]
    assert "templates" in labels["templates"]
    assert "survey population" in labels["forbidden_phrases"]


def test_provisional_items_are_exactly_the_declared_four(freeze):
    """The morning ratification list names exactly four open items."""
    identifiers = [item["id"] for item in freeze["provisional_items"]]
    assert identifiers == list(df.REQUIRED_PROVISIONAL_ITEMS)
    for item in freeze["provisional_items"]:
        assert item["summary"].strip()
        assert item["ratify"].strip()


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
    """Silently dropping an unratified item fails closed."""
    broken = copy.deepcopy(freeze)
    broken["provisional_items"] = broken["provisional_items"][:2]
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
