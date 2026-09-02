"""Adaptive Fisher ladder campaign generator for layers 3 and 4.

The code path that turns a harvested Stage 0 campaign and the frozen
layer 2 selection artifact into a runnable S1-lite ladder campaign. One
job is one tier member's complete adaptive mass ladder, so the detector
setup is amortized across every rung of that member and the adaptivity
lives inside the job while the manifest stays static.

The two tiers share the machinery and differ only by argument. The
parent tier emits the 48 stratified representative members of layer 3;
the selected tier emits the 12 top-by-score members of layer 4, with the
golden 5 flagged inside it. A selected member that also sits in the
parent 48 is emitted again rather than deduplicated, because each
campaign must stand alone, and the overlap is recorded per job.

Every job configuration starts from the member's own Stage 0 staged
configuration, which already carries the frozen scene overrides, the
template asset digest, the extraction settings, the source revision and
the ``stage0`` block. The ladder adds one top-level ``ladder`` block that
the engine ignores and the staged configuration hash covers, resizes
the production grid from the member's realized aperture, stages the
committed science35 truth state into ``psf.aberrations``, and re-stamps
``stage0.code_revision`` with this campaign's own generation revision:
a ladder job runs at the revision its own campaign was generated at,
which is what the runner's moved-code gate and the executor's artifact
binding both enforce, and the Stage 0 generation revision remains
recoverable through the bound Stage 0 frozen manifest.

The integrity chain is closed at every link before a byte is written.
The freeze mapping is proved equal to the freeze file whose digest is
recorded, and a freeze that is not ratified is refused outright because
the freeze_order clause forbids emitting an injected-subhalo job before
the selection is frozen and hashed. The selection artifact must record
the same design freeze digest and the same campaign UUID as the Stage 0
campaign it selects from, and must carry the frozen tier sizes. Every
member's staged configuration is re-hashed against the Stage 0 frozen
manifest, every member's Stage 0 artifact must carry the campaign UUID
and staged configuration hash of that member, and the aperture
recomputed from the realized ``theta_E_eff`` must hash to the aperture
digest the Stage 0 job recorded, so the ladder is bound to exactly the
aperture the selection was computed in.

Ladder jobs consume no random stream. Fisher ladders are deterministic,
the manifest says so in its ``seed_policy``, and adding a stream would
require a freeze amendment rather than a code change.

Selection artifact schema
-------------------------
The consumed layer 2 artifact is the document written by the t12
selection driver, ``schema: stage0_selection_freeze_provisional_v1``.
Only the members this writer depends on are required, and each one is
checked rather than assumed:

``schema``
    Identifier string, recorded in the manifest's ``seed_policy``.
``campaign.campaign_uuid``
    UUID of the Stage 0 campaign the selection was computed on. It must
    equal the campaign UUID of the frozen Stage 0 manifest at
    ``stage0_root``.
``design_freeze.sha256``
    Digest of the design freeze the selection was computed under. It
    must equal the digest of the freeze this campaign is generated from.
``representative_48``, ``selected_12``, ``golden_5``
    One block per tier, each holding ``system_ids`` and a ``members``
    list in the same order. Every member carries ``system_id``, the
    realized ``theta_e_eff_arcsec`` and the score rank
    ``rank_s_plus_c``. The block lengths must equal the frozen
    ``strata`` sizes and the golden ids must be a subset of the
    selected ids.
``rule``
    Optional. When present its ``parent_size``, ``selected_size`` and
    ``golden_size`` must agree with the frozen ``strata`` sizes.

Outputs
-------
``manifest.yaml``
    A valid S1-lite campaign manifest. Its ``seed_policy`` binds the
    design freeze, the Stage 0 frozen manifest and the selection
    artifact by digest, records the per-job staged configuration hash
    the freeze step will reproduce, declares that no random stream is
    consumed, and carries the tier summary including any perimeter
    capped member.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from hwoslaps.provenance import config_hash

from . import _common
from .design_freeze import (
    file_sha256,
    load_design_freeze,
    repo_root,
)


__all__ = [
    "ESTIMAND_CONVENTIONS",
    "LADDER_ARTIFACT",
    "LADDER_ENGINE",
    "LADDER_KERNEL",
    "LADDER_MASK_MODE",
    "LADDER_PSF_STATE",
    "LADDER_RUNNER",
    "LADDER_TIERS",
    "NODE_SPACING_ARCSEC",
    "SATURATION_FRACTION",
    "SCIENCE35_PSF_STATE_PATH",
    "SCIENCE35_PSF_STATE_SHA256",
    "LadderError",
    "aperture_plan",
    "build_ladder_campaign",
    "ladder_job_id",
    "load_selection_artifact",
    "mass_ladder_policy",
    "validate_ladder_manifest",
    "write_ladder_campaign",
]


LADDER_RUNNER = "scripts/run_ladder.py"
"""Per-job runner every ladder manifest must invoke (`str`)."""

LADDER_ARTIFACT = "ladder_result.npz"
"""Artifact every ladder job declares and writes (`str`)."""

LADDER_TIERS = ("parent", "selected")
"""Tiers this generator emits, in freeze order (`tuple` of `str`)."""

LADDER_PSF_STATE = "science35"
"""Standing science-quality truth PSF state label (`str`)."""

SCIENCE35_PSF_STATE_PATH = "configs/psf_states/science_hwo35.yaml"
"""Repo-relative committed science35 state file (`str`).

The standing science-quality truth state: the ``jwst_wss_static_v1``
combined global and segment draw at 35.0 nm measured piston-removed
RMS under seed 20260835, realized in the orthonormal aperture basis
with exact renormalization. The writer stages this file's
``psf.aberrations`` block into every job configuration, because the
runner reconstructs the state from the staged configuration alone.
"""

SCIENCE35_PSF_STATE_SHA256 = (
    "2ed99a3f125735abece890be7a083483faeb3b42e7ac6e03ffbecfdd37d4258c"
)
"""Pinned digest of the committed science35 state file (`str`).

The state's identity is its bytes. A regenerated or edited file cannot
travel under the frozen ``science35`` label, even with a matching
structure, because both the writer and the validator refuse any file
that does not hash to this value.
"""

LADDER_KERNEL = "k999"
"""Frozen Fisher kernel label of every ladder rung (`str`)."""

LADDER_ENGINE = "jax"
"""Fisher engine every ladder rung runs on (`str`)."""

LADDER_MASK_MODE = "all_pixels"
"""Mask mode inside the D-F7 aperture (`str`)."""

NODE_SPACING_ARCSEC = 0.05
"""Fisher grid-map node spacing under the A2 ruling (`float`).

The A2 ruling keeps production maps at this spacing and carries the
declared one-signed systematic recorded in the freeze under
``declared_systematics.spatial_sampling_qmax`` on every quoted
``M_lim``.
"""

SATURATION_FRACTION = 0.99
"""Aperture fraction at which the coarse ascent stops (`float`)."""

CROSSING_CONVENTIONS = {
    "q_max_threshold": 10.0,
    "m10_aperture_fraction": 0.1,
    "m50_aperture_fraction": 0.5,
    "direction": "first upward crossing",
    "m_best_interpolation": (
        "log-linear in q_max through the threshold, on the refined ladder"
    ),
    "m10_m50_interpolation": (
        "linear in log10 M200 of aperture_fraction, on the coarse rungs"
    ),
}
"""Crossing constants the walk implements (`dict`)."""

ESTIMAND_CONVENTIONS = {
    "m_best": (
        "M_best: log-linear interpolation of q_max through 10 on the "
        "refined ladder (the t9 crossing convention). If the ladder never "
        "crosses 10, M_best is null and the curve stands as measured (a "
        "finding, never an extrapolation)."
    ),
    "m10_m50": (
        "M10, M50: linear interpolation in logm of aperture_fraction "
        "through 0.10 / 0.50 on the coarse rung sequence (panel_driver "
        "_crossing convention, first upward crossing). Null when never "
        "crossed."
    ),
    "a_of_m": (
        "A(M): the per-rung detectable_area_arcsec2 sequence itself."
    ),
}
"""Estimand definitions every ladder job carries (`dict`)."""

RANDOM_STREAM_POLICY = (
    "Fisher ladders are deterministic. A ladder job consumes no random "
    "stream and the runner must not construct a numpy Generator. Adding a "
    "stream requires a design freeze amendment, not a code change."
)
"""The manifest's no-RNG declaration (`str`)."""

_MANIFEST_NAME = "manifest.yaml"
_STAGE0_ARTIFACT_NAME = "stage0_observation.npz"
_FROZEN_MANIFEST_NAME = "manifest.frozen.yaml"

_SELECTION_TIER_BLOCKS = {
    "parent": "representative_48",
    "selected": "selected_12",
    "golden": "golden_5",
}

_SELECTION_RULE_SIZES = {
    "parent": "parent_size",
    "selected": "selected_size",
    "golden": "golden_size",
}

_HEX_DIGITS = frozenset("0123456789abcdef")

_THETA_E_MATCH_TOLERANCE = 1.0e-12


class LadderError(ValueError):
    """Raised for any ladder campaign generation or validation failure."""


def ladder_job_id(tier: str, system_id: str) -> str:
    """Return the campaign job identifier of one ladder member.

    Parameters
    ----------
    tier : `str`
        Member of `LADDER_TIERS`.
    system_id : `str`
        Stage 0 system identifier of the member.

    Returns
    -------
    job_id : `str`
        Identifier such as ``ladder_parent_sys0042``, which matches the
        S1-lite ``[a-z0-9_]+`` identifier pattern.

    Raises
    ------
    LadderError
        Raised for an unknown tier or a system id that would produce an
        identifier the S1-lite schema rejects.
    """
    if tier not in LADDER_TIERS:
        raise LadderError(
            f"Tier {tier!r} is not one of the declared ladder tiers "
            f"{list(LADDER_TIERS)}"
        )
    if not isinstance(system_id, str) or not system_id:
        raise LadderError(f"System id {system_id!r} must be a non-empty string")
    job_id = f"ladder_{tier}_{system_id}"
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
           for character in job_id):
        raise LadderError(
            f"System id {system_id!r} yields job id {job_id!r}, which does not "
            "match the S1-lite [a-z0-9_]+ identifier pattern"
        )
    return job_id


def _verified_freeze_artifact(freeze: dict, freeze_path) -> Path:
    """Prove the consumed freeze mapping is the artifact on disk.

    Every ladder artifact records the digest of the freeze *file*, so a
    mapping that has drifted from the file would travel under a digest
    describing a different design. The file is re-read through
    `hwoslaps.campaign.design_freeze.load_design_freeze`, which also
    verifies the artifacts the freeze binds by hash, and compared with
    the consumed mapping for exact structural equality.

    Parameters
    ----------
    freeze : `dict`
        Freeze mapping handed to the builder.
    freeze_path : path-like or `None`
        Freeze artifact whose digest is recorded. `None` means the
        committed freeze.

    Returns
    -------
    resolved : `pathlib.Path`
        The freeze artifact the campaign is built from.

    Raises
    ------
    LadderError
        Raised when the mapping does not equal the file, which includes
        every case where it never went through the verifying loader.
    """
    resolved = _common._freeze_artifact_path(freeze_path)
    if freeze != load_design_freeze(resolved):
        raise LadderError(
            "The design freeze mapping handed to the ladder campaign builder "
            f"is not the content of {resolved}; the recorded design freeze "
            "digest would describe a different design than the one being built"
        )
    return resolved


def _verified_ratified_freeze(freeze: dict) -> str:
    """Return the freeze status, refusing anything but a ratified one.

    The strata freeze_order clause admits an injected-subhalo job only
    after the selection is frozen and hashed, which the design records by
    ratifying the freeze. A ladder campaign generated under a
    provisional freeze would be exactly the ordering the clause forbids.
    """
    status = str(freeze["freeze"]["status"])
    if status != "ratified":
        raise LadderError(
            f"The design freeze status is {status!r}, not 'ratified'; the "
            "strata freeze_order clause admits an injected-subhalo job only "
            "after the selection is frozen and hashed"
        )
    return status


def _verified_runner_command(runner_command) -> list:
    """Return the runner command, proved to be the ladder runner.

    One ladder job is one member's whole adaptive walk, so a manifest
    that points at any other runner would silently change what a job
    means. The S1-lite schema separately requires the ``{config}``
    placeholder.
    """
    parts = [str(part) for part in runner_command]
    if not any(
        part == LADDER_RUNNER or part.endswith("/" + LADDER_RUNNER)
        for part in parts
    ):
        raise LadderError(
            f"The runner command {parts} does not invoke the ladder runner "
            f"{LADDER_RUNNER!r}; one ladder job is one member's complete "
            "adaptive walk and no driver may substitute another runner"
        )
    return parts


def _require_mapping(value: Any, description: str) -> dict:
    """Return a mapping value or raise naming what was expected."""
    if not isinstance(value, dict):
        raise LadderError(f"{description} must be a mapping, got {type(value)}")
    return value


def _require_member(mapping: dict, key: str, description: str) -> Any:
    """Return a required mapping member or raise naming its path."""
    if key not in mapping:
        raise LadderError(f"{description} is missing required key '{key}'")
    return mapping[key]


def _require_digest(value: Any, description: str) -> str:
    """Return a full lowercase SHA-256 digest string or raise."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise LadderError(
            f"{description} must be a 64-character lowercase sha256 digest, "
            f"got {value!r}"
        )
    return value


def _verify_bound_file(path: Path, expected: str, description: str) -> str:
    """Hash one bound file and compare it with the recorded digest."""
    if not path.is_file():
        raise LadderError(
            f"The manifest binds {description} {path}, which does not exist"
        )
    digest = file_sha256(path)
    if digest != expected:
        raise LadderError(
            f"The manifest binds {description} {path} at {expected} but its "
            f"bytes hash to {digest}"
        )
    return digest


def _stage0_campaign(stage0_root: Path) -> dict:
    """Load and verify the frozen Stage 0 campaign at one root.

    The S1-lite loader is used rather than a private re-read so the
    frozen manifest is checked against its freeze-time digest and its
    structural contract before any number is taken out of it.
    """
    from . import s1_lite

    if not stage0_root.is_dir():
        raise LadderError(
            f"Stage 0 campaign root {stage0_root} is not a directory"
        )
    try:
        return _common.load_frozen_manifest(stage0_root)
    except s1_lite.CampaignError as exc:
        raise LadderError(
            f"Stage 0 campaign root {stage0_root} is not a frozen campaign: "
            f"{exc}"
        ) from exc


def _selection_tier_block(artifact: dict, tier: str, size: int) -> tuple:
    """Return one validated tier block of the selection artifact."""
    name = _SELECTION_TIER_BLOCKS[tier]
    block = _require_mapping(
        _require_member(artifact, name, "The selection artifact"),
        f"The selection artifact '{name}' block",
    )
    ids = _require_member(block, "system_ids", f"selection artifact '{name}'")
    if not isinstance(ids, list) or not all(
        isinstance(entry, str) and entry for entry in ids
    ):
        raise LadderError(
            f"selection artifact '{name}'.system_ids must list non-empty "
            f"strings, got {ids!r}"
        )
    if len(set(ids)) != len(ids):
        raise LadderError(
            f"selection artifact '{name}'.system_ids repeats a member"
        )
    if len(ids) != size:
        raise LadderError(
            f"selection artifact '{name}' holds {len(ids)} members but the "
            f"freeze declares strata.{tier}.size {size}"
        )
    members = _require_member(block, "members", f"selection artifact '{name}'")
    if not isinstance(members, list) or len(members) != len(ids):
        found = len(members) if isinstance(members, list) else members
        raise LadderError(
            f"selection artifact '{name}'.members must hold one record per "
            f"system id, got {found!r}"
        )
    records = {}
    for index, raw in enumerate(members):
        path = f"selection artifact '{name}'.members[{index}]"
        record = _require_mapping(raw, path)
        system_id = _require_member(record, "system_id", path)
        if system_id != ids[index]:
            raise LadderError(
                f"{path}.system_id {system_id!r} is not the corresponding "
                f"system id {ids[index]!r}"
            )
        theta_e = _require_member(record, "theta_e_eff_arcsec", path)
        if isinstance(theta_e, bool) or not isinstance(theta_e, (int, float)):
            raise LadderError(f"{path}.theta_e_eff_arcsec must be numeric")
        if not math.isfinite(float(theta_e)) or float(theta_e) <= 0.0:
            raise LadderError(
                f"{path}.theta_e_eff_arcsec must be positive and finite, got "
                f"{theta_e!r}"
            )
        rank = _require_member(record, "rank_s_plus_c", path)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise LadderError(
                f"{path}.rank_s_plus_c must be a positive integer score rank, "
                f"got {rank!r}"
            )
        records[str(system_id)] = {
            "theta_e_eff_arcsec": float(theta_e),
            "rank_s_plus_c": int(rank),
        }
    return tuple(str(entry) for entry in ids), records


def load_selection_artifact(path) -> dict:
    """Load one layer 2 selection artifact without interpreting it.

    Parameters
    ----------
    path : path-like
        Selection freeze artifact written by the t12 selection driver.
        JSON and YAML renderings are both accepted, because both parse
        as one YAML mapping.

    Returns
    -------
    artifact : `dict`
        Parsed document.

    Raises
    ------
    LadderError
        Raised when the file is missing or does not hold a mapping.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LadderError(f"Selection artifact {resolved} does not exist")
    with resolved.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return _require_mapping(document, f"Selection artifact {resolved}")


def _verified_selection(
    freeze: dict,
    freeze_digest: str,
    stage0_campaign: dict,
    artifact: dict,
) -> dict:
    """Verify one selection artifact against the freeze and the pool.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    freeze_digest : `str`
        Digest of the freeze artifact this campaign is generated from.
    stage0_campaign : `dict`
        Frozen Stage 0 campaign the selection was computed on.
    artifact : `dict`
        Parsed selection artifact.

    Returns
    -------
    selection : `dict`
        Schema label and the per-tier id tuples and member records.

    Raises
    ------
    LadderError
        Raised when the artifact records another design freeze digest,
        another campaign UUID, a tier size the freeze does not declare,
        or a golden tier that is not a subset of the selected tier.
    """
    schema = _require_member(artifact, "schema", "The selection artifact")
    if not isinstance(schema, str) or not schema:
        raise LadderError(
            f"The selection artifact schema must be a non-empty string, got "
            f"{schema!r}"
        )
    design = _require_mapping(
        _require_member(artifact, "design_freeze", "The selection artifact"),
        "The selection artifact 'design_freeze' block",
    )
    recorded = _require_digest(
        _require_member(design, "sha256", "selection artifact design_freeze"),
        "selection artifact design_freeze.sha256",
    )
    if recorded != freeze_digest:
        raise LadderError(
            f"The selection artifact was computed under design freeze "
            f"{recorded} but this campaign is generated from {freeze_digest}; "
            "the selection and the ladder must share one design"
        )
    campaign = _require_mapping(
        _require_member(artifact, "campaign", "The selection artifact"),
        "The selection artifact 'campaign' block",
    )
    campaign_uuid = _require_member(
        campaign, "campaign_uuid", "selection artifact campaign"
    )
    if str(campaign_uuid) != str(stage0_campaign["campaign_uuid"]):
        raise LadderError(
            f"The selection artifact selects from campaign {campaign_uuid!r} "
            f"but the Stage 0 campaign is {stage0_campaign['campaign_uuid']!r}"
        )

    sizes = {
        tier: int(freeze["strata"][tier]["size"])
        for tier in ("parent", "selected", "golden")
    }
    rule = artifact.get("rule")
    if rule is not None:
        rule_block = _require_mapping(rule, "The selection artifact 'rule' block")
        for tier, key in _SELECTION_RULE_SIZES.items():
            if key in rule_block and int(rule_block[key]) != sizes[tier]:
                raise LadderError(
                    f"selection artifact rule.{key} is {rule_block[key]} but "
                    f"the freeze declares strata.{tier}.size {sizes[tier]}"
                )

    tiers = {}
    for tier in ("parent", "selected", "golden"):
        ids, records = _selection_tier_block(artifact, tier, sizes[tier])
        tiers[tier] = {"system_ids": ids, "members": records}
    missing = sorted(
        set(tiers["golden"]["system_ids"]) - set(tiers["selected"]["system_ids"])
    )
    if missing:
        raise LadderError(
            "The selection artifact golden tier is not a subset of the "
            f"selected tier; {missing} are golden but not selected"
        )
    return {"schema": str(schema), "tiers": tiers}


def _scalar_text(value: Any) -> str:
    """Return one stored artifact member as text."""
    array = np.asarray(value)
    if array.size != 1:
        raise LadderError("A Stage 0 identity member must hold a single value")
    return str(array.reshape(-1)[0])


def _stage0_member(
    stage0_root: Path, stage0_campaign: dict, job: dict
) -> dict:
    """Read and verify one Stage 0 member before consuming its numbers.

    The staged configuration is re-hashed against the frozen manifest and
    the artifact must carry the campaign UUID and the staged
    configuration hash of exactly this job, so a ladder can never be
    built on an artifact from another campaign, another member, or a
    staged configuration that has moved since the Stage 0 freeze.

    Parameters
    ----------
    stage0_root : `pathlib.Path`
        Frozen Stage 0 campaign root.
    stage0_campaign : `dict`
        Frozen Stage 0 campaign block.
    job : `dict`
        Frozen Stage 0 job of the member.

    Returns
    -------
    member : `dict`
        Staged configuration, its digests, the member overrides, the
        realized ``theta_E_eff`` and the aperture provenance hashes.

    Raises
    ------
    LadderError
        Raised for a missing staged configuration or artifact, a digest
        that no longer matches the frozen record, an identity member the
        artifact does not carry or does not match, or an aperture hash
        that disagrees with the staged configuration.
    """
    job_id = str(job["job_id"])
    staged_path = stage0_root/"configs"/f"{job_id}.yaml"
    if not staged_path.is_file():
        raise LadderError(
            f"Stage 0 member '{job_id}' has no staged config {staged_path}"
        )
    digest = file_sha256(staged_path)
    if digest != job["staged_config_sha256"]:
        raise LadderError(
            f"Stage 0 staged config {staged_path} sha256 {digest} does not "
            f"match the frozen manifest record {job['staged_config_sha256']}"
        )
    with staged_path.open("r", encoding="utf-8") as stream:
        staged = yaml.safe_load(stream)
    staged = _require_mapping(staged, f"Stage 0 staged config {staged_path}")
    recomputed = config_hash(staged)
    if recomputed != job["config_hash"]:
        raise LadderError(
            f"Stage 0 staged config {staged_path} hashes to {recomputed} but "
            f"the frozen manifest records {job['config_hash']}"
        )
    stage0_block = _require_mapping(
        _require_member(staged, "stage0", f"Stage 0 staged config {staged_path}"),
        f"Stage 0 staged config {staged_path} 'stage0' block",
    )

    artifact_path = stage0_root/"outputs"/job_id/_STAGE0_ARTIFACT_NAME
    if not artifact_path.is_file():
        raise LadderError(
            f"Stage 0 member '{job_id}' has no harvested artifact "
            f"{artifact_path}; the ladder consumes a harvested campaign"
        )
    try:
        with np.load(artifact_path, allow_pickle=False) as stored:
            members = {name: stored[name] for name in stored.files}
    except Exception as exc:
        raise LadderError(
            f"Stage 0 artifact {artifact_path} does not load: {exc}"
        ) from exc
    required = (
        "campaign_uuid",
        "config_hash",
        "system_id",
        "theta_e_eff_arcsec",
        "aperture_radius_arcsec",
        "contour_sha256",
        "aperture_sha256",
    )
    absent = [name for name in required if name not in members]
    if absent:
        raise LadderError(
            f"Stage 0 artifact {artifact_path} does not carry "
            + ", ".join(absent)
        )
    for name, expected in (
        ("campaign_uuid", str(stage0_campaign["campaign_uuid"])),
        ("config_hash", str(job["config_hash"])),
        ("system_id", job_id),
    ):
        found = _scalar_text(members[name])
        if found != expected:
            raise LadderError(
                f"Stage 0 artifact {artifact_path} {name} {found!r} does not "
                f"match the frozen campaign value {expected!r}"
            )
    for name, declared in (
        ("contour_sha256", stage0_block["theta_e_contour_sha256"]),
        ("aperture_sha256", stage0_block["theta_e_aperture_sha256"]),
    ):
        found = _scalar_text(members[name])
        if found != str(declared):
            raise LadderError(
                f"Stage 0 artifact {artifact_path} {name} {found} does not "
                f"match the staged configuration value {declared}"
            )
    theta_e_eff = float(np.asarray(members["theta_e_eff_arcsec"]).reshape(-1)[0])
    if not math.isfinite(theta_e_eff) or theta_e_eff <= 0.0:
        raise LadderError(
            f"Stage 0 artifact {artifact_path} theta_e_eff_arcsec "
            f"{theta_e_eff} is not positive and finite"
        )
    return {
        "system_id": job_id,
        "overrides": deepcopy(job["overrides"]),
        "staged_config_sha256": digest,
        "stage0_config_hash": str(job["config_hash"]),
        "scene": str(job["scene"]),
        "theta_e_eff_arcsec": theta_e_eff,
        "aperture_radius_arcsec": float(
            np.asarray(members["aperture_radius_arcsec"]).reshape(-1)[0]
        ),
        "contour_sha256": _scalar_text(members["contour_sha256"]),
        "aperture_sha256": _scalar_text(members["aperture_sha256"]),
    }


def aperture_plan(freeze: dict, theta_e_eff_arcsec: float) -> dict:
    """Size one member's ladder aperture and production grid.

    The aperture is the frozen D-F7 one: radius ``theta_e_factor``
    times the realized ``theta_E_eff``, with the map extent taken from
    the aperture's own computational margin so the mask machinery can
    evaluate the rim. The grid follows the freeze's ``grid_sizing`` rule
    and a member whose required side exceeds the declared maximum is
    capped and flagged rather than silently truncated.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    theta_e_eff_arcsec : `float`
        Realized effective Einstein radius from the member's Stage 0
        artifact.

    Returns
    -------
    plan : `dict`
        Aperture declaration, its canonical digest, the grid shape and
        the perimeter cap record.
    """
    from hwoslaps.lensing.critical_curve import ApertureDefinition

    factor = float(freeze["aperture"]["theta_e_factor"])
    margin = float(freeze["aperture"]["computational_margin_fraction"])
    pixel_scale = float(freeze["grid_sizing"]["pixel_scale_arcsec"])
    maximum = int(freeze["grid_sizing"]["max_side_px"])
    aperture = ApertureDefinition(
        centre_arcsec=(0.0, 0.0),
        theta_e_eff_arcsec=float(theta_e_eff_arcsec),
        theta_e_factor=factor,
        computational_margin_fraction=margin,
    )
    extent = float(aperture.required_map_extent_arcsec)
    required = int(math.ceil(extent/pixel_scale))
    required += required % 2
    side = min(required, maximum)
    side -= side % 2
    half_width = 0.5*side*pixel_scale
    return {
        "theta_e_factor": factor,
        "theta_e_eff_arcsec": float(theta_e_eff_arcsec),
        "radius_arcsec": float(aperture.radius_arcsec),
        "computational_margin_fraction": margin,
        "required_map_half_width_arcsec": float(
            aperture.required_map_half_width_arcsec
        ),
        "required_map_extent_arcsec": extent,
        "aperture_sha256": aperture.sha256,
        "pixel_scale_arcsec": pixel_scale,
        "grid_shape": [side, side],
        "required_side_px": required,
        "max_side_px": maximum,
        "realized_half_width_arcsec": half_width,
        "perimeter_cap_flag": bool(required > maximum),
    }


def mass_ladder_policy(freeze: dict) -> dict:
    """Return the frozen mass-ladder policy plus the walk's constants.

    The freeze's policy travels verbatim so a reader of one staged
    configuration sees the whole rule without opening the freeze, and
    the two constants the walk implements are added beside it: the
    aperture fraction that counts as saturation, and the crossing
    conventions the estimands are read under.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.

    Returns
    -------
    policy : `dict`
        The freeze ``mass_ladder`` block with ``saturation_fraction``
        and ``crossing_conventions`` added.

    Raises
    ------
    LadderError
        Raised when the freeze already declares either constant, which
        would make the staged configuration and the freeze disagree
        about which document owns it.
    """
    policy = deepcopy(freeze["mass_ladder"])
    for key in ("saturation_fraction", "crossing_conventions"):
        if key in policy:
            raise LadderError(
                f"The freeze mass_ladder block already declares '{key}'; the "
                "implementation constants and the frozen policy must not "
                "collide"
            )
    policy["saturation_fraction"] = SATURATION_FRACTION
    policy["crossing_conventions"] = deepcopy(CROSSING_CONVENTIONS)
    return policy


def _science35_state() -> dict:
    """Load and verify the committed science35 truth-state file.

    Returns
    -------
    state : `dict`
        The resolved path, its digest and the ``psf.aberrations`` block.

    Raises
    ------
    LadderError
        Raised when the committed file is absent, does not hash to the
        pinned `SCIENCE35_PSF_STATE_SHA256`, or does not carry the
        combined global and segment structure the science35 label
        declares.
    """
    path = repo_root()/SCIENCE35_PSF_STATE_PATH
    if not path.is_file():
        raise LadderError(
            f"The committed science35 state file {path} does not exist; the "
            "ladder stages its psf.aberrations into every job and cannot run "
            "without it"
        )
    digest = file_sha256(path)
    if digest != SCIENCE35_PSF_STATE_SHA256:
        raise LadderError(
            f"The science35 state file {path} hashes to {digest}, not the "
            f"pinned {SCIENCE35_PSF_STATE_SHA256}; the state's identity is "
            "its bytes and a moved state cannot travel under the frozen label"
        )
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    aberrations = _require_mapping(
        _require_member(
            _require_mapping(
                _require_member(
                    _require_mapping(document, f"science35 state {path}"),
                    "psf",
                    f"science35 state {path}",
                ),
                f"science35 state {path} 'psf' block",
            ),
            "aberrations",
            f"science35 state {path} 'psf' block",
        ),
        f"science35 state {path} 'psf.aberrations' block",
    )
    for name, expected in (
        ("enable_segment_pistons", False),
        ("enable_segment_tiptilts", False),
        ("enable_segment_hexikes", True),
        ("enable_global_zernikes", True),
    ):
        if bool(aberrations.get(name, False)) is not expected:
            raise LadderError(
                f"science35 state {path} sets {name} to "
                f"{aberrations.get(name)!r}, not {expected}; the label "
                "declares the combined global and segment state"
            )
    for name in ("segment_hexikes", "global_zernikes"):
        if not aberrations.get(name):
            raise LadderError(
                f"science35 state {path} carries no {name} coefficients"
            )
    return {"path": path, "sha256": digest, "aberrations": aberrations}


def _ladder_block(
    freeze: dict,
    tier: str,
    golden: bool,
    parent_overlap: bool,
    plan: dict,
    member: dict,
) -> dict:
    """Build the top-level ``ladder`` block of one staged job config."""
    return {
        "tier": tier,
        "golden": bool(golden),
        "parent_overlap": bool(parent_overlap),
        "psf_state": LADDER_PSF_STATE,
        "kernel": LADDER_KERNEL,
        "engine": LADDER_ENGINE,
        "mask_mode": LADDER_MASK_MODE,
        "node_spacing_arcsec": NODE_SPACING_ARCSEC,
        "threshold": str(freeze["mass_ladder"]["threshold"]),
        "aperture": {
            "theta_e_factor": plan["theta_e_factor"],
            "theta_e_eff_arcsec": plan["theta_e_eff_arcsec"],
            "radius_arcsec": plan["radius_arcsec"],
            "computational_margin_fraction": plan[
                "computational_margin_fraction"
            ],
            "required_map_half_width_arcsec": plan[
                "required_map_half_width_arcsec"
            ],
            "pixel_scale_arcsec": plan["pixel_scale_arcsec"],
            "grid_shape": list(plan["grid_shape"]),
            "required_side_px": plan["required_side_px"],
            "max_side_px": plan["max_side_px"],
            "perimeter_cap_flag": plan["perimeter_cap_flag"],
            "stage0_contour_sha256": member["contour_sha256"],
            "stage0_aperture_sha256": member["aperture_sha256"],
        },
        "mass_ladder": mass_ladder_policy(freeze),
        "estimand_conventions": deepcopy(ESTIMAND_CONVENTIONS),
    }


def _declared_spacing_systematic(freeze: dict) -> dict:
    """Return the A2 spacing systematic the ladder's maps carry.

    Every ``M_lim`` this campaign produces comes from a 0.05 arcsec node
    spacing map, so the declared systematic must be in force in the
    freeze the campaign is generated from. A freeze that no longer
    declares it is refused rather than generating numbers whose label
    has quietly disappeared.
    """
    systematics = freeze.get("declared_systematics")
    block = None
    if isinstance(systematics, dict):
        block = systematics.get("spatial_sampling_qmax")
    if not isinstance(block, dict) or "value_dex" not in block:
        raise LadderError(
            "The design freeze does not declare "
            "declared_systematics.spatial_sampling_qmax; the A2 ruling keeps "
            f"production maps at {NODE_SPACING_ARCSEC} arcsec node spacing "
            "and every quoted M_lim carries that systematic"
        )
    return {"value_dex": float(block["value_dex"])}


def _staged_config_hashes(
    scene_paths: dict,
    observing_reference: Optional[str],
    jobs: list,
    output_root: Path,
) -> dict:
    """Return the staged configuration hash the freeze step will produce.

    The S1-lite staging helpers are called rather than reimplemented, so
    the recorded hash is by construction the one ``freeze_campaign``
    computes for the same manifest. Recording it is what lets
    `validate_ladder_manifest` re-hash every job before the campaign is
    ever frozen.

    Parameters
    ----------
    scene_paths : `dict`
        Scene label to resolved base scene configuration path.
    observing_reference : `str` or `None`
        Resolved observing reference path.
    jobs : `list`
        Manifest jobs, each with ``job_id``, ``scene`` and ``overrides``.
    output_root : `pathlib.Path`
        Resolved campaign output root.

    Returns
    -------
    hashes : `dict`
        Job id to staged configuration hash.
    """
    scene_configs = {
        label: _common.load_yaml_mapping(
            path, f"Campaign base scene config '{label}'"
        )
        for label, path in scene_paths.items()
    }
    observation = None
    source_patches: dict = {}
    if observing_reference is not None:
        observation, source_patches = _common.load_observing_reference(
            Path(observing_reference), sorted(scene_paths)
        )
    hashes = {}
    for job in jobs:
        merged, _ = _common.stage_job_config(
            scene_configs[job["scene"]],
            observation,
            source_patches.get(job["scene"]),
            job["overrides"],
            job["job_id"],
            output_root,
        )
        hashes[job["job_id"]] = config_hash(merged)
    return hashes


def build_ladder_campaign(
    freeze: dict,
    *,
    tier: str,
    stage0_root,
    selection_artifact,
    output_root: str,
    runner_command,
    directory,
    freeze_path=None,
    campaign_name: Optional[str] = None,
    campaign_uuid: Optional[str] = None,
) -> dict:
    """Build one tier's ladder manifest from a harvested Stage 0 campaign.

    Every gate of section 1 of the ladder spec is closed before a job is
    built: the freeze mapping is proved to be its file, the freeze must
    be ratified, the runner must be the ladder runner, the selection
    artifact must record this design and this campaign and carry the
    frozen tier sizes, and each member's staged configuration and
    artifact must reconcile with the Stage 0 frozen manifest.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze, which must equal the content of
        ``freeze_path``.
    tier : `str`
        Member of `LADDER_TIERS`.
    stage0_root : path-like
        Harvested Stage 0 campaign root.
    selection_artifact : path-like
        Layer 2 selection freeze artifact of the same campaign.
    output_root : `str`
        Campaign output root written into the manifest verbatim.
    runner_command : sequence of `str`
        S1-lite runner command, which must carry the ``{config}``
        placeholder and must invoke `LADDER_RUNNER`.
    directory : path-like
        Directory the manifest will be written to. Relative manifest
        paths, including ``output_root``, resolve against it exactly as
        S1-lite resolves them at freeze time.
    freeze_path : path-like, optional
        Freeze artifact whose digest is recorded. Defaults to the
        committed freeze.
    campaign_name : `str`, optional
        S1-lite campaign name. Defaults to ``ladder_<tier>``.
    campaign_uuid : `str`, optional
        Pinned campaign UUID. Left unset the S1-lite freeze step
        generates one.

    Returns
    -------
    built : `dict`
        ``manifest``, ``summary`` and the per-member records.

    Raises
    ------
    LadderError
        Raised for any gate of section 1 that does not close. Nothing is
        written and no partial campaign is returned.
    """
    if tier not in LADDER_TIERS:
        raise LadderError(
            f"Tier {tier!r} is not one of the declared ladder tiers "
            f"{list(LADDER_TIERS)}"
        )
    freeze_artifact = _verified_freeze_artifact(freeze, freeze_path)
    status = _verified_ratified_freeze(freeze)
    command = _verified_runner_command(runner_command)
    systematic = _declared_spacing_systematic(freeze)
    code_revision = _common._code_revision_record()

    target = Path(directory).expanduser().resolve()
    root = Path(stage0_root).expanduser().resolve()
    stage0_campaign = _stage0_campaign(root)
    artifact_path = Path(selection_artifact).expanduser().resolve()
    selection = _verified_selection(
        freeze,
        file_sha256(freeze_artifact),
        stage0_campaign,
        load_selection_artifact(artifact_path),
    )

    psf_state = _science35_state()
    stage0_jobs = {str(job["job_id"]): job for job in stage0_campaign["jobs"]}
    parent_ids = set(selection["tiers"]["parent"]["system_ids"])
    golden_ids = set(selection["tiers"]["golden"]["system_ids"])
    member_ids = sorted(selection["tiers"][tier]["system_ids"])
    selected_records = selection["tiers"][tier]["members"]

    scenes: dict = {}
    jobs = []
    members = []
    for system_id in member_ids:
        job = stage0_jobs.get(system_id)
        if job is None:
            raise LadderError(
                f"Selection tier '{tier}' names member {system_id!r}, which is "
                f"not a job of the Stage 0 campaign at {root}"
            )
        member = _stage0_member(root, stage0_campaign, job)
        declared = selected_records[system_id]["theta_e_eff_arcsec"]
        realized = member["theta_e_eff_arcsec"]
        if abs(realized - declared) > _THETA_E_MATCH_TOLERANCE*max(
            abs(realized), 1.0
        ):
            raise LadderError(
                f"Member {system_id} realizes theta_E_eff {realized} in its "
                f"Stage 0 artifact but the selection artifact records "
                f"{declared}; the ladder aperture must be the one the "
                "selection was computed in"
            )
        plan = aperture_plan(freeze, realized)
        if plan["aperture_sha256"] != member["aperture_sha256"]:
            raise LadderError(
                f"Member {system_id} aperture recomputed from the realized "
                f"theta_E_eff hashes to {plan['aperture_sha256']} but its "
                f"Stage 0 artifact records {member['aperture_sha256']}"
            )
        overrides = deepcopy(member["overrides"])
        overrides["ladder"] = _ladder_block(
            freeze,
            tier,
            golden=tier == "selected" and system_id in golden_ids,
            parent_overlap=system_id in parent_ids,
            plan=plan,
            member=member,
        )
        overrides["lensing"]["grid"] = {
            "shape": list(plan["grid_shape"]),
            "pixel_scale": plan["pixel_scale_arcsec"],
        }
        overrides.setdefault("psf", {})["aberrations"] = deepcopy(
            psf_state["aberrations"]
        )
        overrides["stage0"]["code_revision"] = deepcopy(code_revision)
        scenes[member["scene"]] = stage0_campaign["base_scene_configs"][
            member["scene"]
        ]
        jobs.append({
            "job_id": ladder_job_id(tier, system_id),
            "scene": member["scene"],
            "overrides": overrides,
        })
        record = dict(member)
        record["job_id"] = jobs[-1]["job_id"]
        record["grid_shape"] = list(plan["grid_shape"])
        record["perimeter_cap_flag"] = plan["perimeter_cap_flag"]
        record["golden"] = overrides["ladder"]["golden"]
        record["parent_overlap"] = overrides["ladder"]["parent_overlap"]
        record["rank_s_plus_c"] = selected_records[system_id]["rank_s_plus_c"]
        members.append(record)

    scene_paths = {}
    for label, bound in scenes.items():
        path = Path(str(bound["path"]))
        _verify_bound_file(
            path, str(bound["sha256"]), f"the base scene config '{label}'"
        )
        scene_paths[label] = path
    reference = stage0_campaign["observing_reference"]
    reference_path = None
    if reference is not None:
        reference_path = Path(str(reference["path"]))
        _verify_bound_file(
            reference_path, str(reference["sha256"]), "the observing reference"
        )

    resolved_output_root = _common.resolve_path(str(output_root), target)
    job_config_hashes = _staged_config_hashes(
        scene_paths,
        None if reference_path is None else str(reference_path),
        jobs,
        resolved_output_root,
    )

    capped = sorted(
        record["system_id"] for record in members if record["perimeter_cap_flag"]
    )
    summary = {
        "tier": tier,
        "n_jobs": len(jobs),
        "system_ids": list(member_ids),
        "golden_system_ids": sorted(
            record["system_id"] for record in members if record["golden"]
        ),
        "parent_overlap_system_ids": sorted(
            record["system_id"] for record in members
            if record["parent_overlap"]
        ),
        "perimeter_capped_system_ids": capped,
        "grid_side_px_min": min(record["grid_shape"][0] for record in members),
        "grid_side_px_max": max(record["grid_shape"][0] for record in members),
        "declared_max_side_px": int(freeze["grid_sizing"]["max_side_px"]),
        "theta_e_eff_arcsec_min": min(
            record["theta_e_eff_arcsec"] for record in members
        ),
        "theta_e_eff_arcsec_max": max(
            record["theta_e_eff_arcsec"] for record in members
        ),
    }

    manifest = {
        "campaign": {
            "name": str(campaign_name or f"ladder_{tier}"),
            "output_root": str(output_root),
            "runner_command": command,
            "base_scene_configs": {
                label: str(path) for label, path in scene_paths.items()
            },
            "expected_artifacts": [LADDER_ARTIFACT],
            "expected_job_count": len(jobs),
            "seed_policy": {
                "design_freeze_path": str(freeze_artifact),
                "design_freeze_sha256": file_sha256(freeze_artifact),
                "design_freeze_status": status,
                "code_revision_sha256": str(code_revision["sha256"]),
                "tier": tier,
                "tier_size": len(jobs),
                "tier_size_frozen": int(freeze["strata"][tier]["size"]),
                "golden_size_frozen": int(freeze["strata"]["golden"]["size"]),
                "stage0_root": str(root),
                "stage0_campaign_uuid": str(stage0_campaign["campaign_uuid"]),
                "stage0_frozen_manifest_sha256": file_sha256(
                    root/_FROZEN_MANIFEST_NAME
                ),
                "selection_artifact_path": str(artifact_path),
                "selection_artifact_sha256": file_sha256(artifact_path),
                "selection_artifact_schema": selection["schema"],
                "psf_state": LADDER_PSF_STATE,
                "psf_state_path": str(psf_state["path"]),
                "psf_state_sha256": psf_state["sha256"],
                "spatial_sampling_qmax_dex": systematic["value_dex"],
                "consumes_random_stream": False,
                "random_stream_policy": RANDOM_STREAM_POLICY,
                "job_config_hashes": job_config_hashes,
                "foreground_free_ceiling": bool(freeze["foreground_free_ceiling"]),
                "summary": summary,
            },
            "jobs": jobs,
        }
    }
    if reference_path is not None:
        manifest["campaign"]["observing_reference"] = str(reference_path)
    if campaign_uuid is not None:
        manifest["campaign"]["campaign_uuid"] = str(campaign_uuid)
    return {"manifest": manifest, "summary": summary, "members": members}


def write_ladder_campaign(
    directory,
    freeze: dict,
    *,
    tier: str,
    stage0_root,
    selection_artifact,
    output_root: str,
    runner_command,
    freeze_path=None,
    campaign_name: Optional[str] = None,
    campaign_uuid: Optional[str] = None,
) -> dict:
    """Write one tier's ladder manifest into a directory.

    Parameters
    ----------
    directory : path-like
        Destination directory, created if absent.
    freeze : `dict`
        Validated design freeze.
    tier : `str`
        Member of `LADDER_TIERS`.
    stage0_root : path-like
        Harvested Stage 0 campaign root.
    selection_artifact : path-like
        Layer 2 selection freeze artifact of the same campaign.
    output_root : `str`
        Campaign output root written into the manifest verbatim.
    runner_command : sequence of `str`
        S1-lite runner command invoking `LADDER_RUNNER`.
    freeze_path : path-like, optional
        Freeze artifact whose digest is recorded.
    campaign_name : `str`, optional
        S1-lite campaign name. Defaults to ``ladder_<tier>``.
    campaign_uuid : `str`, optional
        Pinned campaign UUID.

    Returns
    -------
    written : `dict`
        Manifest path, its digest, the tier summary and the job count.
    """
    target = Path(directory).expanduser().resolve()
    built = build_ladder_campaign(
        freeze,
        tier=tier,
        stage0_root=stage0_root,
        selection_artifact=selection_artifact,
        output_root=output_root,
        runner_command=runner_command,
        directory=target,
        freeze_path=freeze_path,
        campaign_name=campaign_name,
        campaign_uuid=campaign_uuid,
    )
    target.mkdir(parents=True, exist_ok=True)
    payload = _common._manifest_bytes(built["manifest"])
    manifest_path = target/_MANIFEST_NAME
    manifest_path.write_bytes(payload)
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "summary": built["summary"],
        "n_jobs": len(built["manifest"]["campaign"]["jobs"]),
    }


def _policy_member(policy: dict, key: str) -> Any:
    """Return one required ``seed_policy`` member or raise."""
    if key not in policy:
        raise LadderError(
            f"campaign.seed_policy is missing '{key}'; a ladder manifest binds "
            "its design freeze, its Stage 0 campaign and its selection "
            "artifact by digest and declares its random-stream policy"
        )
    return policy[key]


def _validate_no_rng_declaration(policy: dict) -> None:
    """Require the manifest's explicit no-random-stream declaration."""
    declared = _policy_member(policy, "consumes_random_stream")
    if declared is not False:
        raise LadderError(
            "campaign.seed_policy.consumes_random_stream must be false; "
            "Fisher ladders are deterministic and adding a stream requires a "
            "design freeze amendment"
        )
    statement = _policy_member(policy, "random_stream_policy")
    if statement != RANDOM_STREAM_POLICY:
        raise LadderError(
            "campaign.seed_policy.random_stream_policy does not restate the "
            "frozen no-random-stream declaration"
        )


def _validate_tier_sizes(freeze: dict, policy: dict, jobs: list) -> str:
    """Check the manifest's tier against the freeze's declared strata."""
    tier = _policy_member(policy, "tier")
    if tier not in LADDER_TIERS:
        raise LadderError(
            f"campaign.seed_policy.tier {tier!r} is not one of "
            f"{list(LADDER_TIERS)}"
        )
    size = int(freeze["strata"][tier]["size"])
    if len(jobs) != size:
        raise LadderError(
            f"The manifest holds {len(jobs)} jobs but the freeze declares "
            f"strata.{tier}.size {size}"
        )
    if int(_policy_member(policy, "tier_size_frozen")) != size:
        raise LadderError(
            "campaign.seed_policy.tier_size_frozen does not equal the freeze's "
            f"strata.{tier}.size {size}"
        )
    golden_size = int(freeze["strata"]["golden"]["size"])
    if int(_policy_member(policy, "golden_size_frozen")) != golden_size:
        raise LadderError(
            "campaign.seed_policy.golden_size_frozen does not equal the "
            f"freeze's strata.golden.size {golden_size}"
        )
    flagged = sum(
        1 for job in jobs if job["overrides"]["ladder"]["golden"] is True
    )
    if tier == "selected" and flagged != golden_size:
        raise LadderError(
            f"The selected tier flags {flagged} golden members but the freeze "
            f"declares strata.golden.size {golden_size}"
        )
    if tier == "parent" and flagged:
        raise LadderError(
            f"The parent tier flags {flagged} golden members; the golden flag "
            "belongs to the selected tier"
        )
    return tier


def _validate_ladder_block(freeze: dict, tier: str, job: dict) -> dict:
    """Check one job's ladder block against the freeze and the constants."""
    job_id = job["job_id"]
    overrides = job["overrides"]
    block = overrides.get("ladder")
    if not isinstance(block, dict):
        raise LadderError(
            f"Job '{job_id}' carries no top-level 'ladder' block"
        )
    expected_keys = {
        "tier",
        "golden",
        "parent_overlap",
        "psf_state",
        "kernel",
        "engine",
        "mask_mode",
        "node_spacing_arcsec",
        "threshold",
        "aperture",
        "mass_ladder",
        "estimand_conventions",
    }
    if set(block) != expected_keys:
        raise LadderError(
            f"Job '{job_id}' ladder block members {sorted(block)} are not the "
            f"declared set {sorted(expected_keys)}"
        )
    for key, expected in (
        ("tier", tier),
        ("psf_state", LADDER_PSF_STATE),
        ("kernel", LADDER_KERNEL),
        ("engine", LADDER_ENGINE),
        ("mask_mode", LADDER_MASK_MODE),
        ("node_spacing_arcsec", NODE_SPACING_ARCSEC),
        ("threshold", str(freeze["mass_ladder"]["threshold"])),
    ):
        if block[key] != expected:
            raise LadderError(
                f"Job '{job_id}' ladder block {key} is {block[key]!r}, not the "
                f"frozen value {expected!r}"
            )
    for key in ("golden", "parent_overlap"):
        if not isinstance(block[key], bool):
            raise LadderError(
                f"Job '{job_id}' ladder block {key} must be boolean"
            )
    if block["mass_ladder"] != mass_ladder_policy(freeze):
        raise LadderError(
            f"Job '{job_id}' ladder mass_ladder policy is not the frozen "
            "policy plus the declared implementation constants"
        )
    if block["estimand_conventions"] != ESTIMAND_CONVENTIONS:
        raise LadderError(
            f"Job '{job_id}' ladder estimand_conventions are not the declared "
            "panel conventions"
        )
    return block


def _validate_aperture_arithmetic(freeze: dict, job: dict, block: dict) -> None:
    """Re-derive one job's aperture and grid from its own staged inputs."""
    job_id = job["job_id"]
    aperture = block["aperture"]
    if not isinstance(aperture, dict):
        raise LadderError(f"Job '{job_id}' ladder aperture must be a mapping")
    for key in ("stage0_contour_sha256", "stage0_aperture_sha256"):
        _require_digest(
            aperture.get(key), f"Job '{job_id}' ladder aperture.{key}"
        )
    plan = aperture_plan(freeze, float(aperture["theta_e_eff_arcsec"]))
    for key in (
        "theta_e_factor",
        "radius_arcsec",
        "computational_margin_fraction",
        "required_map_half_width_arcsec",
        "pixel_scale_arcsec",
        "required_side_px",
        "max_side_px",
        "perimeter_cap_flag",
    ):
        if aperture[key] != plan[key]:
            raise LadderError(
                f"Job '{job_id}' ladder aperture.{key} is {aperture[key]!r} but "
                f"the freeze's rule re-derives {plan[key]!r}"
            )
    if list(aperture["grid_shape"]) != list(plan["grid_shape"]):
        raise LadderError(
            f"Job '{job_id}' ladder aperture.grid_shape "
            f"{aperture['grid_shape']} is not the re-derived "
            f"{plan['grid_shape']}"
        )
    if aperture["stage0_aperture_sha256"] != plan["aperture_sha256"]:
        raise LadderError(
            f"Job '{job_id}' ladder aperture is bound to Stage 0 aperture "
            f"{aperture['stage0_aperture_sha256']} but the aperture "
            f"re-derived from its theta_E_eff hashes to "
            f"{plan['aperture_sha256']}"
        )
    grid = job["overrides"]["lensing"]["grid"]
    if list(grid["shape"]) != list(plan["grid_shape"]):
        raise LadderError(
            f"Job '{job_id}' lensing.grid.shape {grid['shape']} is not the "
            f"grid the aperture rule sizes, {plan['grid_shape']}"
        )
    if float(grid["pixel_scale"]) != plan["pixel_scale_arcsec"]:
        raise LadderError(
            f"Job '{job_id}' lensing.grid.pixel_scale {grid['pixel_scale']} is "
            f"not the frozen {plan['pixel_scale_arcsec']}"
        )


def validate_ladder_manifest(manifest_path) -> dict:
    """Validate one written ladder manifest and everything it binds.

    The S1-lite schema check is the first pass. The rest re-opens every
    file the manifest binds by digest, re-hashes each job's staged
    configuration exactly as the freeze step will render it, re-derives
    every aperture and grid from the staged inputs, and holds each ladder
    block against the freeze's own policy and the declared walk
    constants. The no-random-stream declaration must be present and must
    say no.

    Parameters
    ----------
    manifest_path : path-like
        Manifest written by `write_ladder_campaign`.

    Returns
    -------
    normalized : `dict`
        The S1-lite normalized manifest.

    Raises
    ------
    LadderError
        Raised when a bound digest is malformed, its file is missing or
        has moved, a staged configuration no longer hashes to the
        recorded value, a tier size disagrees with the freeze, an
        aperture or grid is not re-derivable, a ladder block is not
        policy-identical to the freeze, or the no-random-stream
        declaration is absent.
    hwoslaps.campaign.s1_lite.CampaignError
        Raised for any schema violation. Nothing is repaired.
    """
    from . import s1_lite

    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        normalized = s1_lite.validate_campaign_manifest(yaml.safe_load(stream))
    campaign = normalized["campaign"]
    policy = campaign["seed_policy"]

    if campaign["expected_artifacts"] != [LADDER_ARTIFACT]:
        raise LadderError(
            f"campaign.expected_artifacts {campaign['expected_artifacts']} is "
            f"not the declared ladder artifact [{LADDER_ARTIFACT!r}]"
        )
    _verified_runner_command(campaign["runner_command"])
    _validate_no_rng_declaration(policy)

    psf_state = _science35_state()
    if str(_policy_member(policy, "psf_state")) != LADDER_PSF_STATE:
        raise LadderError(
            f"campaign.seed_policy.psf_state {policy['psf_state']!r} is not "
            f"the frozen ladder state {LADDER_PSF_STATE!r}"
        )
    _verify_bound_file(
        Path(str(_policy_member(policy, "psf_state_path"))).expanduser(),
        _require_digest(
            _policy_member(policy, "psf_state_sha256"),
            "campaign.seed_policy.psf_state_sha256",
        ),
        "the science35 state file",
    )
    if _policy_member(policy, "psf_state_sha256") != psf_state["sha256"]:
        raise LadderError(
            "campaign.seed_policy.psf_state_sha256 "
            f"{policy['psf_state_sha256']} is not the pinned science35 state "
            f"digest {psf_state['sha256']}"
        )

    freeze_digest = _require_digest(
        _policy_member(policy, "design_freeze_sha256"),
        "campaign.seed_policy.design_freeze_sha256",
    )
    freeze_artifact = Path(
        str(_policy_member(policy, "design_freeze_path"))
    ).expanduser()
    _verify_bound_file(freeze_artifact, freeze_digest, "the design freeze")
    freeze = load_design_freeze(freeze_artifact)
    _verified_ratified_freeze(freeze)

    stage0_root = Path(str(_policy_member(policy, "stage0_root"))).expanduser()
    _verify_bound_file(
        stage0_root/_FROZEN_MANIFEST_NAME,
        _require_digest(
            _policy_member(policy, "stage0_frozen_manifest_sha256"),
            "campaign.seed_policy.stage0_frozen_manifest_sha256",
        ),
        "the Stage 0 frozen manifest",
    )
    _verify_bound_file(
        Path(str(_policy_member(policy, "selection_artifact_path"))).expanduser(),
        _require_digest(
            _policy_member(policy, "selection_artifact_sha256"),
            "campaign.seed_policy.selection_artifact_sha256",
        ),
        "the selection artifact",
    )

    tier = _validate_tier_sizes(freeze, policy, campaign["jobs"])
    for job in campaign["jobs"]:
        block = _validate_ladder_block(freeze, tier, job)
        _validate_aperture_arithmetic(freeze, job, block)
        staged_aberrations = job["overrides"].get("psf", {}).get("aberrations")
        if staged_aberrations != psf_state["aberrations"]:
            raise LadderError(
                f"Job '{job['job_id']}' does not stage the committed "
                "science35 psf.aberrations block; the runner reconstructs "
                "the truth state from the staged configuration alone"
            )
        declared_revision = job["overrides"]["stage0"].get("code_revision")
        if not isinstance(declared_revision, dict) or str(
            declared_revision.get("sha256")
        ) != str(_policy_member(policy, "code_revision_sha256")):
            raise LadderError(
                f"Job '{job['job_id']}' declares code revision "
                f"{declared_revision!r}, not this campaign's generation "
                "revision; a ladder job runs at the revision its own "
                "campaign was generated at"
            )

    scene_paths = {
        label: _common.resolve_path(value, path.parent)
        for label, value in campaign["base_scene_configs"].items()
    }
    reference = campaign["observing_reference"]
    recorded = _require_mapping(
        _policy_member(policy, "job_config_hashes"),
        "campaign.seed_policy.job_config_hashes",
    )
    hashes = _staged_config_hashes(
        scene_paths,
        None if reference is None
        else str(_common.resolve_path(reference, path.parent)),
        campaign["jobs"],
        _common.resolve_path(campaign["output_root"], path.parent),
    )
    if recorded != hashes:
        differing = sorted(
            job_id for job_id in set(recorded) | set(hashes)
            if recorded.get(job_id) != hashes.get(job_id)
        )
        raise LadderError(
            "campaign.seed_policy.job_config_hashes does not reproduce the "
            f"staged configurations of {differing}; the manifest and the "
            "inputs it stages from have diverged"
        )
    return normalized
