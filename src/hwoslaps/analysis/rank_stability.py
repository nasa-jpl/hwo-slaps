"""The pre-registered T4 noise-seed rank-stability harness.

Section 4 of the signed selection pre-registration,
``configs/design/selection_rule_v2.md``, executed against the frozen
declarations of ``configs/design/design_freeze_v1.yaml``. The finding it
answers is P0-6, verbatim: "The current S/G statistics are computed from
noiseless source-only truth. Unless rank stability is demonstrated using
noisy/reconstructed observables, call this an idealized no-subhalo proxy
selection, not an operational Roman/Euclid/HWO target selector."

Everything the run consumes is declared before it starts. The replicate
count, the replicate indices and the per-system noise seeds come from
the freeze's ``seeds.streams.rank_stability_noise`` stream, the tier
size and the reported statistics from its ``selection.rank_stability``
block, and the estimators, cuts, score and tie rule from
`hwoslaps.analysis.selection_score`. Nothing here restates a frozen
number: a value that is not in the freeze is not in this module.

What one run produces:

1. the per-member observables ``S``, ``G`` and ``C`` on the noiseless
   lensed-source truth, inside the ``R <= 2 theta_E`` aperture;
2. the pre-registered curve comparison, ``s_only`` against
   ``s_plus_c``, plus the oracle ranking by measured sensitivity when
   and only when every member carries an ``M_lim`` (none do at Stage 0:
   the score is frozen before any of them is measured);
3. the rank-stability test itself, which recomputes the identical
   estimators on detector-noise realizations of every member and
   reports Spearman correlation of ranking positions, top-K Jaccard,
   and the oracle-recovered fraction where an oracle exists;
4. the estimator ratios, which record how far the noisy ``S`` and ``G``
   sit from their noiseless values. The raw-pixel gradient estimator
   picks up the pixel noise itself, so ``G`` carries a noise floor on
   the noisy path. This is a diagnostic column, never part of the rule.

Member contract, one ``.npz`` per system, every array required unless
marked optional::

    system_id             0-d str, ``sys`` and the zero-padded pool
                          index, as `hwoslaps.campaign.stage0.system_id`
                          writes it
    source_eps            (ny, nx) float, noiseless lensed and
                          PSF-convolved source in e-/s, no subhalo
    grid_arcsec           (ny, nx, 2) float, the ray-traced grid (y, x)
    pixel_scale_arcsec    scalar, arcsec per pixel, both axes
    theta_e_arcsec        scalar, the D-F7 aperture radius is twice this
    exposure_time_s       scalar
    sky_background_e_s    scalar, e-/pixel/s
    dark_current_e_s      scalar, e-/pixel/s
    read_noise_e          scalar, effective combined-image e-/pixel
    gain_e_adu            scalar, e-/ADU
    wavelength_m          scalar, the band centre setting theta_res
    diameter_m            scalar, the aperture diameter setting theta_res
    lens_centre_arcsec    optional (2,) float, defaults to the origin
    m_lim_log10_msun      optional scalar, POST-CAMPAIGN ONLY

Stage 0 stores no electron maps, by the freeze's own
``stage0.artifact_contents`` ruling, so the member records are re-rendered
from the byte-frozen staged configurations of the campaign that produced
them.

Declared choices, all recorded into the report:

- The noiseless path weights by the expected variance ``s + B``. The
  noisy path uses no truth at all: the signal is the realization minus
  the known mean background and the variance is ``max(signal, 0) + B``,
  which is the freeze's ``noisy_variance_rule``, so the test measures
  what an observer could actually rank on.
- Floor cuts are re-applied per replicate on that replicate's own
  ``S``, so a member can leave the pool under noise. Rankings are
  compared on the intersection of their survivor sets and the
  intersection size is reported.
- The tier size is the frozen ``k``. A ranking shorter than ``k`` is a
  failure, not a reason to shrink the tier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import selection_score as ss


__all__ = [
    "MEMBER_SCALARS",
    "NOISE_STREAM",
    "compare_rankings",
    "curve_comparison",
    "definitions_block",
    "estimator_ratios",
    "load_member",
    "load_pool",
    "member_geometry",
    "noiseless_observables",
    "noisy_observables",
    "rank_measured_pool",
    "replicate_indices",
    "replicate_noise_seed",
    "replicate_stability",
    "run_rank_stability",
    "seed_binding",
    "stability_tier_size",
    "system_index",
]


NOISE_STREAM = "rank_stability_noise"
"""Freeze seed stream this harness draws from (`str`)."""

MEMBER_SCALARS = (
    "pixel_scale_arcsec",
    "theta_e_arcsec",
    "exposure_time_s",
    "sky_background_e_s",
    "dark_current_e_s",
    "read_noise_e",
    "gain_e_adu",
    "wavelength_m",
    "diameter_m",
)
"""Required scalar fields of one member record (`tuple` of `str`)."""

SYSTEM_ID_PREFIX = "sys"
"""Prefix `hwoslaps.campaign.stage0.system_id` writes (`str`)."""


def _sha256(path) -> str:
    """Return the sha256 hex digest of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def system_index(system_id: str) -> int:
    """Return the zero-based pool index encoded in one system id.

    The declared ``rank_stability_noise`` spawn key is ``(2, k, i)`` for
    replicate ``k`` and system index ``i``, so the seed of a member is
    fixed by its identity and not by the order the member files happen
    to be loaded in. The index is recovered from the identifier
    `hwoslaps.campaign.stage0.system_id` wrote, and re-encoding it at
    the same width must reproduce that identifier exactly.

    Parameters
    ----------
    system_id : `str`
        Stage 0 identifier such as ``sys0042``.

    Returns
    -------
    index : `int`
        Zero-based pool index.

    Raises
    ------
    ValueError
        Raised when the identifier does not carry a recoverable index.
    """
    if not isinstance(system_id, str) or not system_id.startswith(SYSTEM_ID_PREFIX):
        raise ValueError(
            f"system_id must start with {SYSTEM_ID_PREFIX!r}, got {system_id!r}."
        )
    digits = system_id[len(SYSTEM_ID_PREFIX):]
    if not digits.isdigit():
        raise ValueError(f"system_id {system_id!r} carries no pool index.")
    index = int(digits)
    if f"{SYSTEM_ID_PREFIX}{index:0{len(digits)}d}" != system_id:
        raise ValueError(f"system_id {system_id!r} is not a canonical Stage 0 id.")
    return index


def _noise_stream(freeze: dict) -> dict:
    """Return the declared rank-stability noise stream of one freeze."""
    return freeze["seeds"]["streams"][NOISE_STREAM]


def replicate_indices(freeze: dict) -> tuple[int, ...]:
    """Return the declared replicate indices of one freeze.

    The freeze lists the indices explicitly and states the count beside
    them, precisely so the replicate count cannot be chosen after the
    stability numbers are seen. Both are read and they must agree.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.

    Returns
    -------
    replicates : `tuple` [`int`]
        Declared replicate indices, in declared order.

    Raises
    ------
    ValueError
        Raised when the declared indices are not ``0 .. replicates - 1``.
    """
    stream = _noise_stream(freeze)
    declared = tuple(int(entry) for entry in stream["replicate_indices"])
    count = int(stream["replicates"])
    if declared != tuple(range(count)):
        raise ValueError(
            f"seeds.streams.{NOISE_STREAM}.replicate_indices {list(declared)} are not "
            f"0 .. {count - 1}."
        )
    return declared


def replicate_noise_seed(freeze: dict, replicate: int, index: int) -> int:
    """Return the engine noise seed of one system in one replicate.

    The engine takes a single integer noise seed, so the declared
    ``(2, k, i)`` spawn key is narrowed to 32 bits exactly once, here,
    by the rule `hwoslaps.campaign.stage0.engine_noise_seed` applies to
    the primary stream.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    replicate : `int`
        Declared replicate index ``k``.
    index : `int`
        Zero-based system index ``i``.

    Returns
    -------
    seed : `int`
        Non-negative 32-bit engine noise seed.
    """
    seeds = freeze["seeds"]
    stream = tuple(_noise_stream(freeze)["spawn_key"])
    sequence = np.random.SeedSequence(
        entropy=int(seeds["entropy"]),
        spawn_key=stream + (int(replicate), int(index)),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def seed_binding(freeze: dict, system_ids) -> dict:
    """Return the complete seed binding one run consumed.

    Every seed the run drew is written out, so the report states which
    realizations produced its numbers rather than asserting that they
    could be re-derived.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    system_ids : sequence of `str`
        Pool member identifiers.

    Returns
    -------
    binding : `dict`
        Entropy, spawn key, the rule, and the seed of every system in
        every declared replicate.
    """
    ids = tuple(system_ids)
    indices = {system_id: system_index(system_id) for system_id in ids}
    replicates = replicate_indices(freeze)
    return {
        "entropy": int(freeze["seeds"]["entropy"]),
        "spawn_key_root": list(_noise_stream(freeze)["spawn_key"]),
        "rule": (
            "SeedSequence(entropy, spawn_key=root + (k, i)).generate_state(1, uint32)[0] "
            "for replicate k and system index i"
        ),
        "replicates": list(replicates),
        "seeds": {
            str(replicate): {
                system_id: replicate_noise_seed(freeze, replicate, indices[system_id])
                for system_id in ids
            }
            for replicate in replicates
        },
    }


def stability_tier_size(freeze: dict) -> int:
    """Return the frozen tier size the stability metrics report on.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.

    Returns
    -------
    k : `int`
        Declared ``selection.rank_stability.k``.

    Raises
    ------
    ValueError
        Raised when the declared tier size is not a positive integer, or
        when it disagrees with the frozen selected-tier size.
    """
    k = freeze["selection"]["rank_stability"]["k"]
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"selection.rank_stability.k must be a positive integer, got {k!r}.")
    if k != ss.SELECTED_TIER_SIZE:
        raise ValueError(
            f"selection.rank_stability.k {k} does not match the frozen selected tier "
            f"size {ss.SELECTED_TIER_SIZE}."
        )
    return k


def _scalar(payload, key: str, path) -> float:
    """Read one required finite scalar from a member payload."""
    if key not in payload:
        raise ValueError(f"{path}: member is missing {key}.")
    value = float(np.asarray(payload[key], dtype=float).reshape(()))
    if not np.isfinite(value):
        raise ValueError(f"{path}: {key} is not finite.")
    return value


def load_member(path) -> dict:
    """Load and validate one member record.

    Parameters
    ----------
    path : `pathlib.Path` or `str`
        Member ``.npz`` following the contract in the module docstring.

    Returns
    -------
    member : `dict`
        Validated member fields, the source path and its file digest.

    Raises
    ------
    ValueError
        Raised for a missing field, a non-finite scalar, a source map
        that is not a non-negative two-dimensional electron rate, or a
        grid that does not cover that map.
    """
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        for key in ("system_id", "source_eps", "grid_arcsec"):
            if key not in keys:
                raise ValueError(f"{path}: member is missing {key}.")
        system_id = str(np.asarray(payload["system_id"]).reshape(()))
        source_eps = np.asarray(payload["source_eps"], dtype=float)
        grid_arcsec = np.asarray(payload["grid_arcsec"], dtype=float)
        member = {key: _scalar(payload, key, path) for key in MEMBER_SCALARS}
        if "lens_centre_arcsec" in keys:
            centre = np.asarray(payload["lens_centre_arcsec"], dtype=float).reshape(-1)
            if centre.size != 2:
                raise ValueError(f"{path}: lens_centre_arcsec must hold two entries.")
            member["lens_centre_arcsec"] = (float(centre[0]), float(centre[1]))
        else:
            member["lens_centre_arcsec"] = (0.0, 0.0)
        member["m_lim_log10_msun"] = (
            _scalar(payload, "m_lim_log10_msun", path)
            if "m_lim_log10_msun" in keys
            else None
        )

    if source_eps.ndim != 2:
        raise ValueError(f"{path}: source_eps must be two-dimensional.")
    if not np.all(np.isfinite(source_eps)) or np.any(source_eps < 0.0):
        raise ValueError(f"{path}: source_eps must be finite and non-negative.")
    if grid_arcsec.shape != source_eps.shape + (2,):
        raise ValueError(
            f"{path}: grid_arcsec shape {grid_arcsec.shape} does not match "
            f"source_eps shape {source_eps.shape}."
        )
    member["system_id"] = system_id
    member["system_index"] = system_index(system_id)
    member["source_eps"] = source_eps
    member["grid_arcsec"] = grid_arcsec
    member["path"] = str(path)
    member["sha256"] = _sha256(path)
    return member


def load_pool(members_dir) -> tuple[dict, ...]:
    """Load every member of one pool in sorted file order.

    Parameters
    ----------
    members_dir : `pathlib.Path` or `str`
        Directory of member ``.npz`` records.

    Returns
    -------
    members : `tuple` [`dict`]
        Validated members, ordered by file name.

    Raises
    ------
    ValueError
        Raised for an empty directory or a duplicated system id.
    """
    paths = sorted(Path(members_dir).glob("*.npz"))
    if not paths:
        raise ValueError(f"No member .npz files under {members_dir}.")
    members = tuple(load_member(path) for path in paths)
    ids = [member["system_id"] for member in members]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate system ids in {members_dir}.")
    return members


def member_geometry(member: dict):
    """Return the aperture mask, its radius and the diffraction scale.

    Parameters
    ----------
    member : `dict`
        Validated member record.

    Returns
    -------
    mask : `numpy.ndarray`
        Boolean D-F7 aperture over the member grid.
    radius : `float`
        Aperture radius in arcseconds.
    theta_res : `float`
        Diffraction scale in arcseconds.
    """
    radius = ss.APERTURE_THETA_E_MULTIPLE * member["theta_e_arcsec"]
    mask = ss.aperture_mask(
        member["grid_arcsec"][..., 0],
        member["grid_arcsec"][..., 1],
        radius,
        centre_arcsec=member["lens_centre_arcsec"],
    )
    theta_res = ss.diffraction_scale_arcsec(member["wavelength_m"], member["diameter_m"])
    return mask, radius, theta_res


def _observables(signal_e, variance_e2, member, mask, theta_res):
    """Return ``S``, ``G`` and ``C`` for one signal and variance map."""
    snr = ss.arc_snr(signal_e, variance_e2, mask=mask)
    power = ss.gradient_power(
        signal_e, variance_e2, member["pixel_scale_arcsec"], mask=mask
    )
    return snr, power, ss.complexity(power, snr, theta_res)


def _blank_variance(member: dict) -> float:
    """Return the blank-pixel variance of one member's observing setup."""
    return ss.blank_variance_e2(
        member["sky_background_e_s"],
        member["dark_current_e_s"],
        member["read_noise_e"],
        member["exposure_time_s"],
    )


def noiseless_observables(member: dict) -> dict:
    """Measure one member on its noiseless lensed-source truth.

    Parameters
    ----------
    member : `dict`
        Validated member record.

    Returns
    -------
    row : `dict`
        Observables, aperture census and the member's ``M_lim`` if it
        carries one.
    """
    mask, radius, theta_res = member_geometry(member)
    source_e = member["source_eps"] * member["exposure_time_s"]
    blank = _blank_variance(member)
    variance = ss.expected_variance_e2(source_e, blank)
    snr, power, complexity = _observables(source_e, variance, member, mask, theta_res)
    return {
        "system_id": member["system_id"],
        "theta_e_arcsec": member["theta_e_arcsec"],
        "arc_snr": snr,
        "gradient_power_arcsec2": power,
        "complexity": complexity,
        "theta_res_arcsec": theta_res,
        "aperture_radius_arcsec": radius,
        "aperture_pixels": int(np.count_nonzero(mask)),
        "blank_variance_e2": blank,
        "m_lim_log10_msun": member["m_lim_log10_msun"],
    }


def noisy_observables(member: dict, seed: int) -> dict:
    """Measure one member on a background-subtracted noise realization.

    The realization is the production detector model, so the test
    inherits the campaign's noise physics rather than a restatement of
    it. No truth enters the estimator: the mean background is known from
    the declared exposure and detector, and the variance follows the
    realized signal under the freeze's ``noisy_variance_rule``.

    Parameters
    ----------
    member : `dict`
        Validated member record.
    seed : `int`
        Engine noise seed, from `replicate_noise_seed`.

    Returns
    -------
    row : `dict`
        Observables measured on the realization.
    """
    from ..observation.noise_models import apply_detector_noise

    mask, _, theta_res = member_geometry(member)
    detector = {
        "gain": member["gain_e_adu"],
        "read_noise": member["read_noise_e"],
        "dark_current": member["dark_current_e_s"],
        "sky_background": member["sky_background_e_s"],
    }
    _, components = apply_detector_noise(
        member["source_eps"], member["exposure_time_s"], detector, seed=int(seed)
    )
    signal = components["final_e"] - (components["sky_e"] + components["dark_e"])
    variance = ss.expected_variance_e2(np.maximum(signal, 0.0), _blank_variance(member))
    snr, power, complexity = _observables(signal, variance, member, mask, theta_res)
    return {
        "system_id": member["system_id"],
        "theta_e_arcsec": member["theta_e_arcsec"],
        "arc_snr": snr,
        "gradient_power_arcsec2": power,
        "complexity": complexity,
    }


def rank_measured_pool(rows, variant: str, k: int) -> dict:
    """Cut, score and rank one measured pool under one score variant.

    Parameters
    ----------
    rows : sequence of `dict`
        Per-member observables carrying ``system_id``,
        ``theta_e_arcsec``, ``arc_snr`` and ``complexity``.
    variant : `str`
        Member of `hwoslaps.analysis.selection_score.SCORE_VARIANTS`.
    k : `int`
        Frozen tier size the survivors must be able to fill.

    Returns
    -------
    curve : `dict`
        Survivor ids, their scores, the ranking best first, and the ids
        the floor cuts removed.

    Raises
    ------
    ValueError
        Raised when fewer than ``k`` members survive the floor cuts.
    """
    ids = [row["system_id"] for row in rows]
    theta_e = [row["theta_e_arcsec"] for row in rows]
    snr = [row["arc_snr"] for row in rows]
    complexity = [row["complexity"] for row in rows]
    passed = ss.apply_floor_cuts(theta_e, snr)
    survivors = np.flatnonzero(passed)
    if survivors.size < k:
        raise ValueError(
            f"{survivors.size} of {len(ids)} members survive the floor cuts, too few to "
            f"fill the frozen tier of {k}."
        )
    survivor_ids = tuple(ids[index] for index in survivors)
    scores = ss.selection_scores(
        np.asarray(snr, dtype=float)[survivors],
        np.asarray(complexity, dtype=float)[survivors],
        variant=variant,
    )
    return {
        "variant": variant,
        "survivor_ids": list(survivor_ids),
        "scores": [float(value) for value in scores],
        "ranking": list(ss.rank_by_score(survivor_ids, scores)),
        "failed_ids": [ids[index] for index in np.flatnonzero(~passed)],
    }


def compare_rankings(first, second, k: int) -> dict:
    """Compare two rankings on their shared members.

    Spearman runs over the positions of the ids both rankings hold, in
    ascending id order so the pairing does not depend on either
    ranking's order. The Jaccard index runs over the leading ``k`` of
    each full ranking.

    Parameters
    ----------
    first, second : sequence of `str`
        System ids, best first.
    k : `int`
        Frozen tier size.

    Returns
    -------
    comparison : `dict`
        Shared-member count, tier size, Spearman correlation and top-``k``
        Jaccard index.

    Raises
    ------
    ValueError
        Raised when the rankings share fewer than two members.
    """
    shared = sorted(set(first) & set(second))
    if len(shared) < 2:
        raise ValueError("Two rankings must share at least two members to be compared.")
    positions_first = ss.ranking_positions(first)
    positions_second = ss.ranking_positions(second)
    return {
        "shared_members": len(shared),
        "tier_size": int(k),
        "spearman": ss.spearman_rank_correlation(
            [positions_first[system_id] for system_id in shared],
            [positions_second[system_id] for system_id in shared],
        ),
        "top_k_jaccard": ss.top_k_jaccard(first, second, k),
    }


def _summary(values) -> dict:
    """Return the min, median and mean of one metric across pairs."""
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
    }


def estimator_ratios(reference_rows, rows) -> dict:
    """Report how far the noisy estimators sit from the noiseless ones.

    The raw-pixel gradient estimator picks up the pixel noise itself, so
    ``G`` carries a noise floor on the noisy path. Recording the median
    ratio turns an otherwise puzzling stability number into a diagnosed
    one; it is a diagnostic column, not part of the frozen rule.

    Parameters
    ----------
    reference_rows : sequence of `dict`
        Noiseless observables.
    rows : sequence of `dict`
        Observables of one noise realization of the same members.

    Returns
    -------
    ratios : `dict`
        Pool-median noisy-over-noiseless ratio of ``S`` and of ``G``.
    """
    reference = {row["system_id"]: row for row in reference_rows}
    snr_ratio = [row["arc_snr"] / reference[row["system_id"]]["arc_snr"] for row in rows]
    power_ratio = [
        row["gradient_power_arcsec2"]
        / reference[row["system_id"]]["gradient_power_arcsec2"]
        for row in rows
    ]
    return {
        "arc_snr_ratio_median": float(np.median(snr_ratio)),
        "gradient_power_ratio_median": float(np.median(power_ratio)),
    }


def curve_comparison(curves, oracle_ranking, observables, k: int) -> dict:
    """Compare the pre-registered curves against each other and the oracle.

    Parameters
    ----------
    curves : `dict`
        Ranked curve per score variant, from `rank_measured_pool`.
    oracle_ranking : sequence of `str` or `None`
        Ranking by measured sensitivity, or `None` before the ladders
        exist.
    observables : sequence of `dict`
        Noiseless observables carrying ``m_lim_log10_msun``.
    k : `int`
        Frozen tier size.

    Returns
    -------
    comparison : `dict`
        The curve-against-curve comparison and, when an oracle exists,
        each curve against it.
    """
    comparison = {
        "s_only_vs_s_plus_c": compare_rankings(
            curves["s_only"]["ranking"], curves["s_plus_c"]["ranking"], k
        ),
        "oracle": None,
    }
    if oracle_ranking is None:
        return comparison
    m_lim = {row["system_id"]: row["m_lim_log10_msun"] for row in observables}
    against_oracle = {}
    for variant, curve in curves.items():
        ranking = curve["ranking"]
        against_oracle[variant] = {
            "tier_size": int(k),
            "oracle_recovered_fraction": ss.oracle_recovered_fraction(
                ranking, oracle_ranking, k
            ),
            "spearman_score_vs_m_lim": ss.spearman_rank_correlation(
                curve["scores"],
                [m_lim[system_id] for system_id in curve["survivor_ids"]],
            ),
            "ranking_vs_oracle": compare_rankings(ranking, oracle_ranking, k),
        }
    comparison["oracle"] = against_oracle
    return comparison


def replicate_stability(
    freeze: dict,
    members,
    reference_rows,
    reference_curves,
    oracle_ranking,
    k: int,
    progress=None,
) -> dict:
    """Recompute the curves on every declared replicate and compare.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    members : sequence of `dict`
        Validated member records.
    reference_rows : sequence of `dict`
        Noiseless observables of the same members.
    reference_curves : `dict`
        Noiseless ranked curve per score variant.
    oracle_ranking : sequence of `str` or `None`
        Ranking by measured sensitivity, or `None`.
    k : `int`
        Frozen tier size.
    progress : callable, optional
        Called as ``progress(done, total)`` after every replicate.

    Returns
    -------
    stability : `dict`
        Per-replicate and pairwise comparisons, their summaries, and the
        estimator ratios.
    """
    replicates = replicate_indices(freeze)
    per_replicate = {variant: {} for variant in ss.SCORE_VARIANTS}
    rankings = {variant: {} for variant in ss.SCORE_VARIANTS}
    ratios = {}
    for done, replicate in enumerate(replicates, start=1):
        rows = [
            noisy_observables(
                member, replicate_noise_seed(freeze, replicate, member["system_index"])
            )
            for member in members
        ]
        ratios[str(replicate)] = estimator_ratios(reference_rows, rows)
        for variant in ss.SCORE_VARIANTS:
            curve = rank_measured_pool(rows, variant, k)
            ranking = curve["ranking"]
            rankings[variant][replicate] = ranking
            entry = compare_rankings(reference_curves[variant]["ranking"], ranking, k)
            entry["survivors"] = len(curve["survivor_ids"])
            entry["failed_ids"] = curve["failed_ids"]
            if oracle_ranking is not None:
                entry["oracle_recovered_fraction"] = ss.oracle_recovered_fraction(
                    ranking, oracle_ranking, k
                )
            per_replicate[variant][str(replicate)] = entry
        if progress is not None:
            progress(done, len(replicates))

    pairwise = {}
    for variant in ss.SCORE_VARIANTS:
        spearman = []
        jaccard = []
        for position, first in enumerate(replicates):
            for second in replicates[position + 1:]:
                entry = compare_rankings(
                    rankings[variant][first], rankings[variant][second], k
                )
                spearman.append(entry["spearman"])
                jaccard.append(entry["top_k_jaccard"])
        pairwise[variant] = {
            "pairs": len(spearman),
            "spearman": _summary(spearman),
            "top_k_jaccard": _summary(jaccard),
        }

    summary = {
        variant: {
            "spearman_vs_noiseless": _summary(
                [entry["spearman"] for entry in per_replicate[variant].values()]
            ),
            "top_k_jaccard_vs_noiseless": _summary(
                [entry["top_k_jaccard"] for entry in per_replicate[variant].values()]
            ),
        }
        for variant in ss.SCORE_VARIANTS
    }
    if oracle_ranking is not None:
        for variant in ss.SCORE_VARIANTS:
            summary[variant]["oracle_recovered_fraction"] = _summary(
                [
                    entry["oracle_recovered_fraction"]
                    for entry in per_replicate[variant].values()
                ]
            )
    return {
        "replicates": [int(replicate) for replicate in replicates],
        "per_replicate": per_replicate,
        "pairwise": pairwise,
        "summary": summary,
        "estimator_ratios": ratios,
    }


def definitions_block(freeze: dict) -> dict:
    """Return the frozen definitions one run applied, for the record.

    Every entry is read from the freeze or from the module constants the
    freeze cross-checks, so the report cannot drift from the rule it
    claims to have followed.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.

    Returns
    -------
    definitions : `dict`
        The pre-registration binding, the observables, the score, the
        cuts, the tie rule and the tier sizes.
    """
    selection = freeze["selection"]
    return {
        "pre_registration": dict(selection["pre_registration"]),
        "implementation": selection["module"],
        "harness": "src/hwoslaps/analysis/rank_stability.py",
        "floor_cuts": {
            "theta_e_arcsec_greater_than": ss.FLOOR_THETA_E_ARCSEC,
            "arc_snr_greater_than": ss.FLOOR_ARC_SNR,
            "strict": bool(selection["floor_cuts"]["strict"]),
        },
        "aperture": selection["observables"]["aperture"],
        "arc_snr": selection["observables"]["arc_snr"],
        "gradient_power": selection["observables"]["gradient_power"],
        "complexity": selection["observables"]["complexity"],
        "score": selection["score"]["expression"],
        "standardization": selection["score"]["standardization"],
        "tie_rule": selection["score"]["tie_rule"],
        "noiseless_variance": selection["observables"]["pixel_variance_e2"],
        "noisy_variance": selection["rank_stability"]["noisy_variance_rule"],
        "statistics": list(selection["rank_stability"]["statistics"]),
        "tier_sizes": {
            "selected": ss.SELECTED_TIER_SIZE,
            "golden": ss.GOLDEN_TIER_SIZE,
        },
        "spearman_score_vs_m_lim_sign": (
            "a working score correlates NEGATIVELY with log10 M_lim: higher score, lower "
            "detectable mass"
        ),
    }


def _oracle_ranking(observables, survivor_ids):
    """Return the sensitivity ranking of the survivors, or `None`."""
    have = [row["m_lim_log10_msun"] is not None for row in observables]
    if not all(have):
        if any(have):
            raise ValueError(
                "m_lim_log10_msun must be present for every member or for none."
            )
        return None
    by_id = {row["system_id"]: row["m_lim_log10_msun"] for row in observables}
    return list(
        ss.rank_by_sensitivity(survivor_ids, [by_id[entry] for entry in survivor_ids])
    )


def run_rank_stability(members_dir, freeze: dict, label: str, progress=None) -> dict:
    """Run the whole pre-registered T4 harness over one pool.

    Parameters
    ----------
    members_dir : `pathlib.Path` or `str`
        Directory of member ``.npz`` records.
    freeze : `dict`
        Validated design freeze, from
        `hwoslaps.campaign.design_freeze.load_design_freeze`.
    label : `str`
        Name of this run, carried into the report.
    progress : callable, optional
        Called as ``progress(done, total)`` after every replicate.

    Returns
    -------
    report : `dict`
        Definitions, seed binding, member digests, observables, curves,
        the frozen selection and the rank-stability result.

    Raises
    ------
    ValueError
        Raised for a malformed pool, a member whose statistics are
        inadmissible, or a survivor set too small for the frozen tier.
    """
    members = load_pool(members_dir)
    k = stability_tier_size(freeze)
    observables = [noiseless_observables(member) for member in members]
    curves = {
        variant: rank_measured_pool(observables, variant, k)
        for variant in ss.SCORE_VARIANTS
    }
    frozen = ss.rank_pool(
        [row["system_id"] for row in observables],
        [row["theta_e_arcsec"] for row in observables],
        [row["arc_snr"] for row in observables],
        [row["complexity"] for row in observables],
        variant="s_plus_c",
    )
    oracle_ranking = _oracle_ranking(observables, curves["s_plus_c"]["survivor_ids"])
    return {
        "label": label,
        "members_dir": str(members_dir),
        "pool_size": len(members),
        "tier_size": k,
        "definitions": definitions_block(freeze),
        "seed_binding": seed_binding(
            freeze, [member["system_id"] for member in members]
        ),
        "members": [
            {
                "system_id": member["system_id"],
                "file": Path(member["path"]).name,
                "sha256": member["sha256"],
            }
            for member in members
        ],
        "oracle_available": oracle_ranking is not None,
        "observables": observables,
        "curves": curves,
        "frozen_selection": {
            "variant": frozen.variant,
            "survivors": len(frozen.survivor_ids),
            "ranking": list(frozen.ranking),
            "selected_ids": list(frozen.selected_ids),
            "golden_ids": list(frozen.golden_ids),
        },
        "oracle_ranking": oracle_ranking,
        "curve_comparison": curve_comparison(curves, oracle_ranking, observables, k),
        "stability": replicate_stability(
            freeze,
            members,
            observables,
            curves,
            oracle_ranking,
            k,
            progress=progress,
        ),
    }
