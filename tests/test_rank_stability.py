"""Contracts for the pre-registered T4 rank-stability harness.

The seed contract, the freeze bindings, the member loader and the
metrics are pure and run without the observation engine. The end-to-end
run draws detector-noise realizations with the production noise model,
so it is guarded on that import and runs on a deliberately small pool
under a freeze whose replicate count has been cut down.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.analysis import rank_stability as rst
from hwoslaps.analysis import selection_score as ss
from hwoslaps.campaign import design_freeze as df
from hwoslaps.campaign import stage0


FREEZE_PATH = PROJECT_ROOT / "configs" / "design" / "design_freeze_v1.yaml"

PRE_REGISTRATION_PATH = PROJECT_ROOT / "configs" / "design" / "selection_rule_v2.md"
"""Committed copy of the signed selection pre-registration."""

POOL_SIZE = 16
"""Member count of the synthetic pool, above the frozen tier of 12."""

TEST_REPLICATES = 3
"""Replicate count the end-to-end test cuts the freeze down to."""

MEMBER_SHAPE = (64, 64)
MEMBER_PIXEL_SCALE = 0.05
MEMBER_EXPOSURE_S = 2000.0
MEMBER_SKY_E_S = 0.00251028
MEMBER_DARK_E_S = 0.002
MEMBER_READ_NOISE_E = 0.2828427
MEMBER_GAIN = 1.0
MEMBER_WAVELENGTH_M = 5.5e-7
MEMBER_DIAMETER_M = 6.0

CAMPAIGN_SYSTEM_0_SEEDS = (
    1642941589,
    2042547359,
    2195805282,
    1276981447,
    508097036,
    789937718,
    1082302538,
    3773433871,
    1949521194,
    842236747,
    1015075890,
    267636301,
    2790530779,
    3180518699,
    2140174103,
    614850073,
    166618009,
    2353544223,
    3585936888,
    4020097432,
)
"""Engine noise seed of ``sys0000`` in each declared replicate (`tuple`).

The values the ``stage0_pool_v1`` campaign actually drew. They pin the
narrowing of the declared ``(2, k, i)`` spawn key to the 32-bit engine
input, so a change anywhere in that chain fails here rather than
silently producing a different set of realizations.
"""


@pytest.fixture(scope="module")
def freeze():
    """Load the committed design freeze once."""
    return df.load_design_freeze(FREEZE_PATH)


@pytest.fixture(scope="module")
def small_freeze(freeze):
    """Return the freeze with its replicate count cut down.

    The harness never takes a replicate override: the count is frozen
    exactly so it cannot be chosen after the stability numbers are seen.
    A test that wants a shorter run therefore has to hand the harness a
    different declaration, which is what this does.
    """
    reduced = copy.deepcopy(freeze)
    stream = reduced["seeds"]["streams"][rst.NOISE_STREAM]
    stream["replicates"] = TEST_REPLICATES
    stream["replicate_indices"] = list(range(TEST_REPLICATES))
    return reduced


def _source_map(theta_e, clumps, total_rate, rng):
    """Render one toy arc: a ring plus a declared clump count."""
    rows = (np.arange(MEMBER_SHAPE[0]) - 0.5 * (MEMBER_SHAPE[0] - 1)) * MEMBER_PIXEL_SCALE
    cols = (np.arange(MEMBER_SHAPE[1]) - 0.5 * (MEMBER_SHAPE[1] - 1)) * MEMBER_PIXEL_SCALE
    y_arcsec, x_arcsec = np.meshgrid(-rows, cols, indexing="ij")
    width = 0.25 * theta_e
    image = np.exp(-0.5 * ((np.hypot(y_arcsec, x_arcsec) - theta_e) / width) ** 2)
    for index in range(clumps):
        angle = 2.0 * np.pi * (index + 0.5) / clumps
        offset = np.hypot(
            y_arcsec - theta_e * np.sin(angle), x_arcsec - theta_e * np.cos(angle)
        )
        image = image + 1.5 * np.exp(-0.5 * (offset / (0.35 * width)) ** 2)
    image = np.clip(image * (1.0 + 0.02 * rng.standard_normal(MEMBER_SHAPE)), 0.0, None)
    return total_rate * image / float(np.sum(image))


def _member_grid():
    """Return the ray-traced grid the synthetic members share."""
    rows = (np.arange(MEMBER_SHAPE[0]) - 0.5 * (MEMBER_SHAPE[0] - 1)) * MEMBER_PIXEL_SCALE
    cols = (np.arange(MEMBER_SHAPE[1]) - 0.5 * (MEMBER_SHAPE[1] - 1)) * MEMBER_PIXEL_SCALE
    y_arcsec, x_arcsec = np.meshgrid(-rows, cols, indexing="ij")
    return np.stack([y_arcsec, x_arcsec], axis=-1)


def _member_payload(index, rng, **overrides):
    """Return the ``.npz`` payload of one synthetic member."""
    theta_e = 0.55 + 0.01 * index
    payload = {
        "system_id": np.asarray(stage0.system_id(index, POOL_SIZE)),
        "source_eps": _source_map(theta_e, 1 + (index % 5), 4.0 * 1.3 ** (index % 6), rng),
        "grid_arcsec": _member_grid(),
        "pixel_scale_arcsec": np.asarray(MEMBER_PIXEL_SCALE),
        "theta_e_arcsec": np.asarray(theta_e),
        "exposure_time_s": np.asarray(MEMBER_EXPOSURE_S),
        "sky_background_e_s": np.asarray(MEMBER_SKY_E_S),
        "dark_current_e_s": np.asarray(MEMBER_DARK_E_S),
        "read_noise_e": np.asarray(MEMBER_READ_NOISE_E),
        "gain_e_adu": np.asarray(MEMBER_GAIN),
        "wavelength_m": np.asarray(MEMBER_WAVELENGTH_M),
        "diameter_m": np.asarray(MEMBER_DIAMETER_M),
    }
    payload.update(overrides)
    return payload


def _write_pool(directory, size=POOL_SIZE):
    """Write one synthetic member pool and return its directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260823)
    for index in range(size):
        np.savez(directory / f"member_{index:03d}.npz", **_member_payload(index, rng))
    return directory


@pytest.fixture(scope="module")
def pool_dir(tmp_path_factory):
    """Write the synthetic pool once for the whole module."""
    return _write_pool(tmp_path_factory.mktemp("rank_stability_pool"))


def test_the_signed_pre_registration_is_committed_at_the_frozen_digest(freeze):
    """The freeze binds the pre-registration and the copy matches it.

    The document is the signed authority for the whole selection rule.
    A clean clone must be able to read it and to confirm that what it
    reads is what the freeze was signed against.
    """
    assert PRE_REGISTRATION_PATH.is_file()
    digest = hashlib.sha256(PRE_REGISTRATION_PATH.read_bytes()).hexdigest()
    assert digest == freeze["selection"]["pre_registration"]["sha256"]
    assert freeze["selection"]["pre_registration"]["version"] == 2


def test_replicate_seeds_reproduce_the_campaign_binding(freeze):
    """Every declared replicate seed of ``sys0000`` is regenerated."""
    measured = tuple(
        rst.replicate_noise_seed(freeze, replicate, 0)
        for replicate in rst.replicate_indices(freeze)
    )
    assert measured == CAMPAIGN_SYSTEM_0_SEEDS


def test_replicate_seeds_follow_the_declared_spawn_key(freeze):
    """The seed is the declared spawn key narrowed to 32 bits."""
    stream = tuple(freeze["seeds"]["streams"][rst.NOISE_STREAM]["spawn_key"])
    expected = np.random.SeedSequence(
        entropy=int(freeze["seeds"]["entropy"]), spawn_key=stream + (7, 42)
    ).generate_state(1, dtype=np.uint32)[0]
    assert rst.replicate_noise_seed(freeze, 7, 42) == int(expected)


def test_replicate_seeds_separate_replicates_and_systems(freeze):
    """No two declared draws share a seed."""
    seeds = [
        rst.replicate_noise_seed(freeze, replicate, index)
        for replicate in rst.replicate_indices(freeze)
        for index in range(20)
    ]
    assert len(set(seeds)) == len(seeds)


def test_the_freeze_declares_twenty_replicates(freeze):
    """The pre-registered replicate count is read, never assumed."""
    assert rst.replicate_indices(freeze) == tuple(range(20))
    assert rst.stability_tier_size(freeze) == 12


def test_replicate_indices_reject_a_count_that_disagrees(freeze):
    """A declaration that lists the wrong indices fails closed."""
    broken = copy.deepcopy(freeze)
    broken["seeds"]["streams"][rst.NOISE_STREAM]["replicates"] = 5
    with pytest.raises(ValueError, match="replicate_indices"):
        rst.replicate_indices(broken)


def test_tier_size_rejects_a_disagreement_with_the_module(freeze):
    """A ``k`` that leaves the frozen selected tier fails closed."""
    broken = copy.deepcopy(freeze)
    broken["selection"]["rank_stability"]["k"] = 11
    with pytest.raises(ValueError, match="selected tier size"):
        rst.stability_tier_size(broken)


def test_seed_binding_records_every_system_of_every_replicate(freeze):
    """The report states the seeds it drew rather than a rule alone."""
    ids = [stage0.system_id(index, POOL_SIZE) for index in range(POOL_SIZE)]
    binding = rst.seed_binding(freeze, ids)
    assert binding["entropy"] == int(freeze["seeds"]["entropy"])
    assert binding["spawn_key_root"] == [2]
    assert len(binding["seeds"]) == 20
    assert sorted(binding["seeds"]["0"]) == sorted(ids)
    assert binding["seeds"]["0"]["sys0000"] == CAMPAIGN_SYSTEM_0_SEEDS[0]


def test_system_index_round_trips_the_stage0_identifier():
    """The seed binding recovers the index the campaign wrote."""
    for index in (0, 7, 42, 999):
        assert rst.system_index(stage0.system_id(index, 1000)) == index


@pytest.mark.parametrize(
    "bad",
    ["0042", "sys", "sysx042", "sys 42", "sys42x", "sys042.0", 42],
)
def test_system_index_rejects_a_non_canonical_identifier(bad):
    """An id whose index cannot be recovered is a fail-closed error."""
    with pytest.raises(ValueError):
        rst.system_index(bad)


def test_system_index_rejects_a_look_alike_identifier():
    """``str.isdigit`` admits digits that ``sys`` never wrote.

    Re-encoding at the same width is what rejects them, so a member file
    carrying a look-alike identifier cannot quietly borrow the noise
    seed of the system whose index its digits happen to parse to.
    """
    assert "٤٢".isdigit()
    with pytest.raises(ValueError, match="canonical"):
        rst.system_index("sys٤٢")

    assert rst.system_index("sys42") == 42


def test_load_member_reads_the_declared_contract(pool_dir):
    """Every declared field arrives, with the file digest beside it."""
    path = sorted(pool_dir.glob("*.npz"))[0]
    member = rst.load_member(path)
    assert member["system_id"] == "sys0000"
    assert member["system_index"] == 0
    assert member["source_eps"].shape == MEMBER_SHAPE
    assert member["grid_arcsec"].shape == MEMBER_SHAPE + (2,)
    assert member["lens_centre_arcsec"] == (0.0, 0.0)
    assert member["m_lim_log10_msun"] is None
    assert member["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key in rst.MEMBER_SCALARS:
        assert isinstance(member[key], float)


def test_load_pool_orders_by_file_name(pool_dir):
    """The pool loads in sorted file order, which is index order here."""
    members = rst.load_pool(pool_dir)
    assert len(members) == POOL_SIZE
    assert [member["system_index"] for member in members] == list(range(POOL_SIZE))


def test_load_pool_rejects_an_empty_directory(tmp_path):
    """A pool with no members is a configuration error."""
    with pytest.raises(ValueError, match="No member"):
        rst.load_pool(tmp_path)


def test_load_pool_rejects_duplicate_system_ids(tmp_path):
    """One system cannot enter the pool twice."""
    rng = np.random.default_rng(1)
    for name in ("a.npz", "b.npz"):
        np.savez(tmp_path / name, **_member_payload(0, rng))
    with pytest.raises(ValueError, match="Duplicate system ids"):
        rst.load_pool(tmp_path)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"source_eps": np.zeros(MEMBER_SHAPE) - 1.0e-9}, "non-negative"),
        ({"source_eps": np.zeros((4, 4, 4))}, "two-dimensional"),
        ({"grid_arcsec": np.zeros((8, 8, 2))}, "does not match"),
        ({"theta_e_arcsec": np.asarray(np.nan)}, "not finite"),
        ({"lens_centre_arcsec": np.zeros(3)}, "two entries"),
    ],
)
def test_load_member_rejects_a_broken_record(tmp_path, overrides, match):
    """A malformed member raises instead of reaching the estimators."""
    rng = np.random.default_rng(2)
    np.savez(tmp_path / "member.npz", **_member_payload(0, rng, **overrides))
    with pytest.raises(ValueError, match=match):
        rst.load_member(tmp_path / "member.npz")


def test_load_member_rejects_a_missing_scalar(tmp_path):
    """Every declared scalar is required, none is defaulted."""
    rng = np.random.default_rng(3)
    payload = _member_payload(0, rng)
    del payload["read_noise_e"]
    np.savez(tmp_path / "member.npz", **payload)
    with pytest.raises(ValueError, match="read_noise_e"):
        rst.load_member(tmp_path / "member.npz")


def test_noiseless_observables_are_the_committed_estimators(pool_dir):
    """The harness adds no estimator of its own."""
    member = rst.load_member(sorted(pool_dir.glob("*.npz"))[0])
    row = rst.noiseless_observables(member)

    mask, radius, theta_res = rst.member_geometry(member)
    source_e = member["source_eps"] * member["exposure_time_s"]
    blank = ss.blank_variance_e2(
        member["sky_background_e_s"],
        member["dark_current_e_s"],
        member["read_noise_e"],
        member["exposure_time_s"],
    )
    variance = ss.expected_variance_e2(source_e, blank)

    assert radius == pytest.approx(
        ss.APERTURE_THETA_E_MULTIPLE * member["theta_e_arcsec"]
    )
    assert row["arc_snr"] == ss.arc_snr(source_e, variance, mask=mask)
    assert row["gradient_power_arcsec2"] == ss.gradient_power(
        source_e, variance, member["pixel_scale_arcsec"], mask=mask
    )
    assert row["complexity"] == ss.complexity(
        row["gradient_power_arcsec2"], row["arc_snr"], theta_res
    )
    assert row["aperture_pixels"] == int(np.count_nonzero(mask))
    assert row["blank_variance_e2"] == blank


def test_the_synthetic_pool_clears_the_floor_cuts(pool_dir):
    """The test pool exercises the ranking, not the cuts."""
    rows = [rst.noiseless_observables(member) for member in rst.load_pool(pool_dir)]
    passed = ss.apply_floor_cuts(
        [row["theta_e_arcsec"] for row in rows], [row["arc_snr"] for row in rows]
    )
    assert bool(np.all(passed))


def test_rank_measured_pool_refuses_a_pool_too_small_for_the_tier(pool_dir):
    """Fail closed rather than shrink the frozen tier."""
    rows = [rst.noiseless_observables(member) for member in rst.load_pool(pool_dir)]
    with pytest.raises(ValueError, match="too few"):
        rst.rank_measured_pool(rows, "s_plus_c", len(rows) + 1)


def test_rank_measured_pool_reports_the_floor_cut_casualties(pool_dir):
    """A member that leaves the pool under a cut is named."""
    rows = [rst.noiseless_observables(member) for member in rst.load_pool(pool_dir)]
    rows[3]["theta_e_arcsec"] = 0.4
    curve = rst.rank_measured_pool(rows, "s_plus_c", 12)
    assert curve["failed_ids"] == ["sys0003"]
    assert "sys0003" not in curve["ranking"]
    assert len(curve["survivor_ids"]) == POOL_SIZE - 1


def test_compare_rankings_does_not_depend_on_the_argument_order(pool_dir):
    """Spearman pairs on ascending shared ids, not on either ranking."""
    rows = [rst.noiseless_observables(member) for member in rst.load_pool(pool_dir)]
    first = rst.rank_measured_pool(rows, "s_only", 12)["ranking"]
    second = rst.rank_measured_pool(rows, "s_plus_c", 12)["ranking"]
    forward = rst.compare_rankings(first, second, 12)
    reverse = rst.compare_rankings(second, first, 12)
    assert forward["spearman"] == reverse["spearman"]
    assert forward["top_k_jaccard"] == reverse["top_k_jaccard"]
    assert forward["shared_members"] == POOL_SIZE


def test_compare_rankings_rejects_rankings_that_barely_overlap():
    """Two rankings must share members to be comparable."""
    with pytest.raises(ValueError, match="at least two members"):
        rst.compare_rankings(("a", "b", "c"), ("c", "d", "e"), 3)


def test_estimator_ratios_are_one_against_the_noiseless_rows(pool_dir):
    """The diagnostic reduces to unity when nothing was perturbed."""
    rows = [rst.noiseless_observables(member) for member in rst.load_pool(pool_dir)]
    ratios = rst.estimator_ratios(rows, rows)
    assert ratios["arc_snr_ratio_median"] == pytest.approx(1.0)
    assert ratios["gradient_power_ratio_median"] == pytest.approx(1.0)


def test_definitions_block_quotes_the_freeze(freeze):
    """The report cannot drift from the rule it claims to follow."""
    definitions = rst.definitions_block(freeze)
    selection = freeze["selection"]
    assert definitions["score"] == selection["score"]["expression"]
    assert definitions["noisy_variance"] == selection["rank_stability"][
        "noisy_variance_rule"
    ]
    assert definitions["statistics"] == list(selection["rank_stability"]["statistics"])
    assert definitions["pre_registration"]["sha256"] == selection["pre_registration"][
        "sha256"
    ]
    assert definitions["tier_sizes"] == {"selected": 12, "golden": 5}


def test_the_freeze_statistics_are_the_ones_the_harness_reports(freeze):
    """Every declared statistic is a function this harness calls."""
    declared = set(freeze["selection"]["rank_stability"]["statistics"])
    assert declared <= set(ss.__all__)


def test_run_rank_stability_is_deterministic(pool_dir, small_freeze):
    """Two runs of one pool under one freeze agree in every number."""
    pytest.importorskip("hwoslaps.observation.noise_models")

    first = rst.run_rank_stability(pool_dir, small_freeze, "determinism")
    second = rst.run_rank_stability(pool_dir, small_freeze, "determinism")
    assert first == second


def test_run_rank_stability_follows_the_declared_replicates(pool_dir, small_freeze):
    """The run draws exactly the declared replicates and no others."""
    pytest.importorskip("hwoslaps.observation.noise_models")

    report = rst.run_rank_stability(pool_dir, small_freeze, "replicates")
    stability = report["stability"]
    assert stability["replicates"] == list(range(TEST_REPLICATES))
    assert sorted(stability["estimator_ratios"]) == sorted(
        str(replicate) for replicate in range(TEST_REPLICATES)
    )
    for variant in ss.SCORE_VARIANTS:
        assert sorted(stability["per_replicate"][variant]) == sorted(
            str(replicate) for replicate in range(TEST_REPLICATES)
        )
        pairs = TEST_REPLICATES * (TEST_REPLICATES - 1) // 2
        assert stability["pairwise"][variant]["pairs"] == pairs


def test_run_rank_stability_reports_the_frozen_selection(pool_dir, small_freeze):
    """The frozen rule runs on the noiseless truth, tiers and all."""
    pytest.importorskip("hwoslaps.observation.noise_models")

    report = rst.run_rank_stability(pool_dir, small_freeze, "selection")
    frozen = report["frozen_selection"]
    assert frozen["variant"] == "s_plus_c"
    assert frozen["selected_ids"] == frozen["ranking"][:12]
    assert frozen["golden_ids"] == frozen["ranking"][:5]
    assert frozen["ranking"] == report["curves"]["s_plus_c"]["ranking"]
    assert report["oracle_available"] is False
    assert report["oracle_ranking"] is None
    assert report["tier_size"] == 12
    assert report["pool_size"] == POOL_SIZE
    assert len(report["members"]) == POOL_SIZE


def test_run_rank_stability_uses_the_seeds_it_reports(pool_dir, small_freeze):
    """Replaying the reported seeds reproduces the reported diagnostics.

    The binding in the report is not a restatement of the rule, it is
    the list of realizations the numbers came from, so replaying it has
    to land on the same estimator ratios.
    """
    pytest.importorskip("hwoslaps.observation.noise_models")

    report = rst.run_rank_stability(pool_dir, small_freeze, "seeds")
    seeds = report["seed_binding"]["seeds"]["1"]
    members = rst.load_pool(pool_dir)
    replayed = [
        rst.noisy_observables(member, seeds[member["system_id"]]) for member in members
    ]

    assert rst.estimator_ratios(report["observables"], replayed) == (
        report["stability"]["estimator_ratios"]["1"]
    )
    for member in members:
        assert seeds[member["system_id"]] == rst.replicate_noise_seed(
            small_freeze, 1, member["system_index"]
        )


def test_noisy_observables_use_no_truth(pool_dir, small_freeze):
    """Two seeds give two realizations; the estimator sees neither truth."""
    pytest.importorskip("hwoslaps.observation.noise_models")

    member = rst.load_member(sorted(pool_dir.glob("*.npz"))[0])
    first = rst.noisy_observables(
        member, rst.replicate_noise_seed(small_freeze, 0, member["system_index"])
    )
    again = rst.noisy_observables(
        member, rst.replicate_noise_seed(small_freeze, 0, member["system_index"])
    )
    other = rst.noisy_observables(
        member, rst.replicate_noise_seed(small_freeze, 1, member["system_index"])
    )
    assert first == again
    assert first["arc_snr"] != other["arc_snr"]


def test_analysis_package_exports_the_rank_stability_api():
    """The lazy package attribute resolves the new module."""
    import hwoslaps.analysis as analysis

    assert analysis.NOISE_STREAM == "rank_stability_noise"
    assert analysis.run_rank_stability is rst.run_rank_stability
    assert "replicate_noise_seed" in dir(analysis)
