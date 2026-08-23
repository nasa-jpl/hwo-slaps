"""Contracts for the ladder campaign writer and its manifest validator.

Everything here runs CPU-only on a synthetic Stage 0 campaign. The
fixture freezes a small S1-lite pool through the real S1-lite layer and
writes the identity members every Stage 0 artifact carries, so the
writer meets exactly the inputs it meets in production without the
lensing engine ever being imported. The design freeze itself is the
committed one, because the tier sizes, the aperture rule and the ladder
policy the writer is held to are the frozen values.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.campaign import design_freeze as df
from hwoslaps.campaign import ladder
from hwoslaps.campaign import s1_lite
from hwoslaps.lensing.critical_curve import ApertureDefinition


FREEZE_PATH = PROJECT_ROOT/"configs"/"design"/"design_freeze_v1.yaml"

POOL_SIZE = 60
"""Synthetic Stage 0 pool size, large enough for the parent 48 (`int`)."""

STAGE0_UUID = "11111111-2222-3333-4444-555555555555"
LADDER_UUID = "66666666-7777-8888-9999-aaaaaaaaaaaa"

SCENE_LABEL = "scene4_cosmos"

HAND_CHECKED_INDEX = 0
"""Member whose grid arithmetic the tests compute by hand (`int`)."""

CAPPED_INDEX = 55
"""Member whose required grid exceeds the declared maximum (`int`).

It sits outside the parent 48 and inside the selected 12, so one tier
reports a capped member and the other reports none.
"""

PARENT_IDS = tuple(f"sys{index:04d}" for index in range(48))
SELECTED_IDS = (
    "sys0000",
    "sys0001",
    "sys0002",
    "sys0003",
    "sys0004",
    "sys0005",
    "sys0048",
    "sys0049",
    "sys0050",
    "sys0051",
    "sys0052",
    "sys0055",
)
GOLDEN_IDS = SELECTED_IDS[:5]

RUNNER_COMMAND = ["python", "scripts/run_ladder.py", "{config}"]


@pytest.fixture(scope="module")
def freeze():
    """Load the committed design freeze once."""
    return df.load_design_freeze(FREEZE_PATH)


def _system_id(index):
    """Return the synthetic Stage 0 system id of one index."""
    return f"sys{index:04d}"


def _theta_e_eff(index):
    """Return the realized theta_E_eff of one synthetic member."""
    if index == HAND_CHECKED_INDEX:
        return 1.0
    if index == CAPPED_INDEX:
        return 20.0
    return 0.6 + 0.01*index


def _aperture_sha256(freeze, theta_e_eff):
    """Return the aperture digest a Stage 0 job would have recorded."""
    return ApertureDefinition(
        centre_arcsec=(0.0, 0.0),
        theta_e_eff_arcsec=float(theta_e_eff),
        theta_e_factor=float(freeze["aperture"]["theta_e_factor"]),
        computational_margin_fraction=float(
            freeze["aperture"]["computational_margin_fraction"]
        ),
    ).sha256


def _contour_sha256(system_id):
    """Return a deterministic stand-in for one extracted contour digest."""
    return hashlib.sha256(f"contour-{system_id}".encode("utf-8")).hexdigest()


def _scene_config():
    """Return the stub base scene configuration the pool is staged from."""
    return {
        "lensing": {
            "grid": {"shape": [64, 64], "pixel_scale": 0.00716},
            "lens_galaxy": {
                "redshift": 0.3,
                "mass": {
                    "type": "Isothermal",
                    "einstein_radius": 1.0,
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.0, 0.0],
                },
            },
            "source_galaxy": {"redshift": 0.9, "light": {"type": "Image"}},
            "subhalo": {"enabled": False},
            "cosmology": "Planck15",
        },
        "psf": {"kernel": {"shape_native": [51, 51]}},
        "observation": {"exposure_time": 2000.0},
        "modeling": {"enabled": False},
    }


def _observing_reference():
    """Return the stub observing reference the pool is staged from."""
    return {
        "observation": {"exposure_time": 2000.0, "sky_background_e_s": 0.0},
        "source_normalization": {
            SCENE_LABEL: {
                "lensing": {"source_galaxy": {"light": {"flux_scale": 1.0}}}
            }
        },
    }


def _stage0_overrides(freeze, index):
    """Return the Stage 0 job overrides of one synthetic member."""
    system_id = _system_id(index)
    theta_e_eff = _theta_e_eff(index)
    plan = ladder.aperture_plan(freeze, theta_e_eff)
    return {
        "global_seed": 1000 + index,
        "stage0": {
            "system_id": system_id,
            "source_template": "cosmos_48849",
            "source_asset_path": "configs/source_assets/cosmos_48849_hlr011.npz",
            "source_asset_sha256": "a"*64,
            "theta_e_contour_sha256": _contour_sha256(system_id),
            "theta_e_aperture_sha256": _aperture_sha256(freeze, theta_e_eff),
            "theta_e_eff_arcsec": theta_e_eff,
            "code_revision": {
                "git_hash": "0"*40,
                "git_dirty": False,
                "sha256": "b"*64,
            },
        },
        "lensing": {
            "grid": {
                "shape": list(plan["grid_shape"]),
                "pixel_scale": plan["pixel_scale_arcsec"],
            },
            "subhalo": {"enabled": False},
        },
        "psf": {"kernel": {"shape_native": [51, 51]}},
        "observation": {"exposure_time": 2000.0},
        "modeling": {"enabled": False},
    }


def _write_yaml(path, document):
    """Write one mapping as YAML and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=True)
    return path


def _write_stage0_campaign(freeze, directory):
    """Freeze a synthetic Stage 0 campaign and write its artifacts."""
    directory = Path(directory)
    scene_path = _write_yaml(directory/"scene.yaml", _scene_config())
    reference_path = _write_yaml(
        directory/"observing_reference.yaml", _observing_reference()
    )
    root = directory/"run"
    manifest = {
        "campaign": {
            "name": "stage0_pool",
            "output_root": str(root),
            "runner_command": [
                "python", "scripts/run_stage0_observation.py", "{config}"
            ],
            "base_scene_configs": {SCENE_LABEL: str(scene_path)},
            "observing_reference": str(reference_path),
            "expected_artifacts": ["stage0_observation.npz"],
            "expected_job_count": POOL_SIZE,
            "campaign_uuid": STAGE0_UUID,
            "seed_policy": {"entropy": 20260823},
            "jobs": [
                {
                    "job_id": _system_id(index),
                    "scene": SCENE_LABEL,
                    "overrides": _stage0_overrides(freeze, index),
                }
                for index in range(POOL_SIZE)
            ],
        }
    }
    manifest_path = _write_yaml(directory/"manifest.yaml", manifest)
    s1_lite.freeze_campaign(manifest_path)
    frozen = s1_lite._load_frozen_manifest(root)
    for job in frozen["jobs"]:
        index = int(job["job_id"][3:])
        theta_e_eff = _theta_e_eff(index)
        output_dir = root/"outputs"/job["job_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir/"stage0_observation.npz",
            campaign_uuid=np.asarray(STAGE0_UUID),
            config_hash=np.asarray(job["config_hash"]),
            system_id=np.asarray(job["job_id"]),
            theta_e_eff_arcsec=np.asarray(theta_e_eff),
            aperture_radius_arcsec=np.asarray(
                theta_e_eff*float(freeze["aperture"]["theta_e_factor"])
            ),
            contour_sha256=np.asarray(_contour_sha256(job["job_id"])),
            aperture_sha256=np.asarray(_aperture_sha256(freeze, theta_e_eff)),
        )
    return root


def _member_record(system_id, rank):
    """Return one selection artifact member record."""
    return {
        "system_id": system_id,
        "theta_e_eff_arcsec": _theta_e_eff(int(system_id[3:])),
        "rank_s_plus_c": rank,
        "score_s_plus_c": 3.5 - 0.1*rank,
        "source_template": "cosmos_48849",
    }


def _selection_document(freeze_digest):
    """Return the synthetic layer 2 selection freeze document."""
    ranks = {
        system_id: index + 1 for index, system_id in enumerate(SELECTED_IDS)
    }
    for index, system_id in enumerate(PARENT_IDS):
        ranks.setdefault(system_id, len(SELECTED_IDS) + index + 1)
    return {
        "schema": "stage0_selection_freeze_provisional_v1",
        "campaign": {
            "name": "stage0_pool",
            "campaign_uuid": STAGE0_UUID,
            "jobs": POOL_SIZE,
        },
        "design_freeze": {"sha256": freeze_digest, "status": "ratified"},
        "rule": {"parent_size": 48, "selected_size": 12, "golden_size": 5},
        "representative_48": {
            "system_ids": list(PARENT_IDS),
            "members": [
                _member_record(system_id, ranks[system_id])
                for system_id in PARENT_IDS
            ],
        },
        "selected_12": {
            "system_ids": list(SELECTED_IDS),
            "members": [
                _member_record(system_id, ranks[system_id])
                for system_id in SELECTED_IDS
            ],
        },
        "golden_5": {
            "system_ids": list(GOLDEN_IDS),
            "members": [
                _member_record(system_id, ranks[system_id])
                for system_id in GOLDEN_IDS
            ],
        },
    }


@pytest.fixture(scope="module")
def stage0_root(freeze, tmp_path_factory):
    """Freeze one synthetic Stage 0 campaign for the whole module."""
    return _write_stage0_campaign(freeze, tmp_path_factory.mktemp("stage0"))


@pytest.fixture(scope="module")
def selection_path(tmp_path_factory):
    """Write the synthetic selection artifact once."""
    return _write_yaml(
        tmp_path_factory.mktemp("selection")/"selection_freeze.yaml",
        _selection_document(df.design_freeze_digest(FREEZE_PATH)),
    )


def _write_campaign(freeze, directory, stage0_root, selection_path, tier):
    """Write one ladder campaign into a directory."""
    return ladder.write_ladder_campaign(
        directory,
        freeze,
        tier=tier,
        stage0_root=stage0_root,
        selection_artifact=selection_path,
        output_root=str(Path(directory)/"run"),
        runner_command=RUNNER_COMMAND,
        freeze_path=FREEZE_PATH,
        campaign_uuid=LADDER_UUID,
    )


@pytest.fixture(scope="module")
def parent_campaign(freeze, stage0_root, selection_path, tmp_path_factory):
    """Write the parent-tier campaign once."""
    directory = tmp_path_factory.mktemp("parent")/"campaign"
    written = _write_campaign(
        freeze, directory, stage0_root, selection_path, "parent"
    )
    with written["manifest_path"].open("r", encoding="utf-8") as stream:
        written["manifest"] = yaml.safe_load(stream)
    return written


@pytest.fixture(scope="module")
def selected_campaign(freeze, stage0_root, selection_path, tmp_path_factory):
    """Write the selected-tier campaign once."""
    directory = tmp_path_factory.mktemp("selected")/"campaign"
    written = _write_campaign(
        freeze, directory, stage0_root, selection_path, "selected"
    )
    with written["manifest_path"].open("r", encoding="utf-8") as stream:
        written["manifest"] = yaml.safe_load(stream)
    return written


@pytest.fixture
def selection_copy(selection_path, tmp_path):
    """Return a mutable copy of the selection artifact."""

    def write(mutate):
        """Write one mutated selection artifact and return its path."""
        with selection_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        mutate(document)
        return _write_yaml(tmp_path/"selection_freeze.yaml", document)

    return write


# ---------------------------------------------------------------------------
# Tier membership and job identity
# ---------------------------------------------------------------------------


def test_parent_tier_emits_every_representative_member(freeze, parent_campaign):
    """Layer 3 emits one job per stratified representative member."""
    jobs = parent_campaign["manifest"]["campaign"]["jobs"]
    assert len(jobs) == freeze["strata"]["parent"]["size"]
    assert [job["job_id"] for job in jobs] == [
        f"ladder_parent_{system_id}" for system_id in sorted(PARENT_IDS)
    ]
    assert parent_campaign["manifest"]["campaign"]["expected_job_count"] == 48


def test_selected_tier_emits_every_selected_member(freeze, selected_campaign):
    """Layer 4 emits all 12 selected members, overlap included."""
    jobs = selected_campaign["manifest"]["campaign"]["jobs"]
    assert len(jobs) == freeze["strata"]["selected"]["size"]
    assert [job["job_id"] for job in jobs] == [
        f"ladder_selected_{system_id}" for system_id in sorted(SELECTED_IDS)
    ]


def test_job_ids_sort_within_the_tier(selected_campaign):
    """Jobs are emitted sorted by system id within the tier."""
    ids = [job["job_id"] for job in selected_campaign["manifest"]["campaign"]["jobs"]]
    assert ids == sorted(ids)
    assert all(
        character in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for job_id in ids
        for character in job_id
    )


def test_overlap_flag_records_membership_of_the_parent_tier(selected_campaign):
    """A selected member inside the 48 is flagged rather than deduplicated."""
    flags = {
        job["overrides"]["ladder"]["tier"]: None
        for job in selected_campaign["manifest"]["campaign"]["jobs"]
    }
    assert set(flags) == {"selected"}
    overlap = {
        job["job_id"].removeprefix("ladder_selected_"):
        job["overrides"]["ladder"]["parent_overlap"]
        for job in selected_campaign["manifest"]["campaign"]["jobs"]
    }
    assert overlap == {
        system_id: system_id in PARENT_IDS for system_id in SELECTED_IDS
    }
    assert sum(overlap.values()) == 6


def test_golden_flag_is_the_selection_artifact_subset(selected_campaign):
    """The golden 5 are flagged inside the selected tier, and only there."""
    golden = {
        job["job_id"].removeprefix("ladder_selected_")
        for job in selected_campaign["manifest"]["campaign"]["jobs"]
        if job["overrides"]["ladder"]["golden"]
    }
    assert golden == set(GOLDEN_IDS)
    assert selected_campaign["summary"]["golden_system_ids"] == sorted(GOLDEN_IDS)


def test_parent_tier_flags_no_golden_member(parent_campaign):
    """The golden flag belongs to the selected tier alone."""
    assert all(
        job["overrides"]["ladder"]["golden"] is False
        for job in parent_campaign["manifest"]["campaign"]["jobs"]
    )
    assert parent_campaign["summary"]["golden_system_ids"] == []


# ---------------------------------------------------------------------------
# The staged ladder block
# ---------------------------------------------------------------------------


def test_ladder_block_carries_exactly_the_declared_fields(freeze, parent_campaign):
    """The block the engine ignores holds the spec's field set verbatim."""
    for job in parent_campaign["manifest"]["campaign"]["jobs"]:
        block = job["overrides"]["ladder"]
        assert set(block) == {
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
        assert block["psf_state"] == "science35"
        assert block["kernel"] == "k999"
        assert block["engine"] == "jax"
        assert block["mask_mode"] == "all_pixels"
        assert block["node_spacing_arcsec"] == 0.05
        assert block["threshold"] == freeze["mass_ladder"]["threshold"]


def test_ladder_block_echoes_the_frozen_mass_ladder_policy(freeze, parent_campaign):
    """The whole frozen policy travels with the two walk constants."""
    for job in parent_campaign["manifest"]["campaign"]["jobs"]:
        policy = job["overrides"]["ladder"]["mass_ladder"]
        for key, value in freeze["mass_ladder"].items():
            assert policy[key] == value, key
        assert policy["coarse"]["step_dex"] == 0.25
        assert policy["coarse"]["low"] == 6.0
        assert policy["coarse"]["high"] == 9.5
        assert policy["refine"]["step_dex"] == 0.1
        assert policy["extend_down"]["zero_rungs"] == 2
        assert policy["saturation_fraction"] == 0.99
        assert policy["crossing_conventions"]["q_max_threshold"] == 10.0
        assert policy["crossing_conventions"]["m10_aperture_fraction"] == 0.1
        assert policy["crossing_conventions"]["m50_aperture_fraction"] == 0.5


def test_ladder_block_carries_the_estimand_conventions(parent_campaign):
    """Every job restates how M_best, M10, M50 and A(M) are read."""
    for job in parent_campaign["manifest"]["campaign"]["jobs"]:
        conventions = job["overrides"]["ladder"]["estimand_conventions"]
        assert conventions == ladder.ESTIMAND_CONVENTIONS
        assert "never an extrapolation" in conventions["m_best"]
        assert "first upward crossing" in conventions["m10_m50"]
        assert "detectable_area_arcsec2" in conventions["a_of_m"]


def test_ladder_is_bound_to_the_stage0_aperture(freeze, parent_campaign):
    """The contour and aperture digests come from the Stage 0 job itself."""
    for job in parent_campaign["manifest"]["campaign"]["jobs"]:
        system_id = job["job_id"].removeprefix("ladder_parent_")
        aperture = job["overrides"]["ladder"]["aperture"]
        assert aperture["stage0_contour_sha256"] == _contour_sha256(system_id)
        assert aperture["stage0_aperture_sha256"] == _aperture_sha256(
            freeze, _theta_e_eff(int(system_id[3:]))
        )
        assert aperture["theta_e_factor"] == 2.0
        assert aperture["computational_margin_fraction"] == 0.1
        assert aperture["radius_arcsec"] == pytest.approx(
            2.0*aperture["theta_e_eff_arcsec"], rel=1e-12
        )
        assert aperture["required_map_half_width_arcsec"] == pytest.approx(
            1.1*aperture["radius_arcsec"], rel=1e-12
        )


def test_staged_configs_keep_the_stage0_block_verbatim(
    freeze, stage0_root, parent_campaign
):
    """The ladder job inherits the Stage 0 provenance block unchanged."""
    frozen = s1_lite._load_frozen_manifest(Path(stage0_root))
    stage0_jobs = {job["job_id"]: job for job in frozen["jobs"]}
    for job in parent_campaign["manifest"]["campaign"]["jobs"]:
        system_id = job["job_id"].removeprefix("ladder_parent_")
        expected = stage0_jobs[system_id]["overrides"]["stage0"]
        assert job["overrides"]["stage0"] == expected
        assert job["overrides"]["psf"]["kernel"]["shape_native"] == [51, 51]


def test_staged_configs_carry_the_committed_science35_state(parent_campaign):
    """Every ladder job stages the committed science35 aberration block."""
    state = ladder._science35_state()
    for job in parent_campaign["manifest"]["campaign"]["jobs"]:
        assert job["overrides"]["psf"]["aberrations"] == state["aberrations"]


def test_the_science35_state_is_refused_off_its_pinned_digest(monkeypatch):
    """A regenerated state file cannot travel under the frozen label."""
    monkeypatch.setattr(ladder, "SCIENCE35_PSF_STATE_SHA256", "0"*64)
    with pytest.raises(ladder.LadderError, match="pinned"):
        ladder._science35_state()


# ---------------------------------------------------------------------------
# Grid sizing
# ---------------------------------------------------------------------------


def test_grid_sizing_matches_the_hand_computed_arithmetic(parent_campaign):
    """A one-arcsecond member sizes to a hand-checked 616 pixel grid."""
    job = next(
        job for job in parent_campaign["manifest"]["campaign"]["jobs"]
        if job["job_id"].endswith(_system_id(HAND_CHECKED_INDEX))
    )
    aperture = job["overrides"]["ladder"]["aperture"]
    assert aperture["theta_e_eff_arcsec"] == 1.0
    assert aperture["radius_arcsec"] == 2.0
    assert aperture["required_map_half_width_arcsec"] == pytest.approx(2.2)
    assert aperture["required_side_px"] == 616
    assert aperture["grid_shape"] == [616, 616]
    assert aperture["perimeter_cap_flag"] is False
    assert job["overrides"]["lensing"]["grid"]["shape"] == [616, 616]
    assert job["overrides"]["lensing"]["grid"]["pixel_scale"] == 0.00716


def test_every_grid_follows_the_frozen_sizing_rule(freeze, parent_campaign):
    """Each side is the even-rounded ceiling of the margined extent."""
    pixel_scale = freeze["grid_sizing"]["pixel_scale_arcsec"]
    maximum = freeze["grid_sizing"]["max_side_px"]
    for job in parent_campaign["manifest"]["campaign"]["jobs"]:
        aperture = job["overrides"]["ladder"]["aperture"]
        extent = 2.0*aperture["required_map_half_width_arcsec"]
        side = aperture["grid_shape"][0]
        assert side % 2 == 0
        assert side <= maximum
        assert aperture["grid_shape"] == [side, side]
        if not aperture["perimeter_cap_flag"]:
            assert side*pixel_scale >= extent
            assert (side - 2)*pixel_scale < extent


def test_capped_member_is_flagged_and_reported(freeze, selected_campaign):
    """A member past the declared maximum is flagged, never truncated."""
    capped = _system_id(CAPPED_INDEX)
    assert selected_campaign["summary"]["perimeter_capped_system_ids"] == [capped]
    job = next(
        job for job in selected_campaign["manifest"]["campaign"]["jobs"]
        if job["job_id"].endswith(capped)
    )
    aperture = job["overrides"]["ladder"]["aperture"]
    assert aperture["perimeter_cap_flag"] is True
    assert aperture["grid_shape"] == [freeze["grid_sizing"]["max_side_px"]]*2
    assert aperture["required_side_px"] > freeze["grid_sizing"]["max_side_px"]


def test_uncapped_tier_reports_no_capped_member(parent_campaign):
    """The parent tier's cap list is empty, and says so explicitly."""
    assert parent_campaign["summary"]["perimeter_capped_system_ids"] == []
    assert parent_campaign["summary"]["declared_max_side_px"] == 2048


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_manifest_validates_against_the_s1_lite_schema(parent_campaign):
    """The generated manifest is a valid S1-lite campaign manifest."""
    normalized = s1_lite.validate_campaign_manifest(
        copy.deepcopy(parent_campaign["manifest"])
    )["campaign"]
    assert normalized["name"] == "ladder_parent"
    assert normalized["expected_job_count"] == 48
    assert normalized["expected_artifacts"] == ["ladder_result.npz"]
    assert normalized["campaign_uuid"] == LADDER_UUID
    assert "{config}" in " ".join(normalized["runner_command"])


def test_manifest_declares_that_no_random_stream_is_consumed(parent_campaign):
    """Fisher ladders are deterministic and the manifest says so."""
    policy = parent_campaign["manifest"]["campaign"]["seed_policy"]
    assert policy["consumes_random_stream"] is False
    assert "must not construct a numpy Generator" in policy[
        "random_stream_policy"
    ]


def test_manifest_binds_its_inputs_by_digest(
    parent_campaign, stage0_root, selection_path
):
    """The freeze, the pool and the selection all travel by digest."""
    policy = parent_campaign["manifest"]["campaign"]["seed_policy"]
    assert policy["design_freeze_sha256"] == df.design_freeze_digest(FREEZE_PATH)
    assert policy["design_freeze_status"] == "ratified"
    assert policy["stage0_campaign_uuid"] == STAGE0_UUID
    assert policy["stage0_frozen_manifest_sha256"] == df.file_sha256(
        Path(stage0_root)/"manifest.frozen.yaml"
    )
    assert policy["selection_artifact_sha256"] == df.file_sha256(selection_path)
    assert policy["selection_artifact_schema"] == (
        "stage0_selection_freeze_provisional_v1"
    )
    assert policy["spatial_sampling_qmax_dex"] == -0.004
    assert policy["tier_size_frozen"] == 48
    assert policy["golden_size_frozen"] == 5


def test_manifest_records_the_staged_config_hash_of_every_job(parent_campaign):
    """Each job's staged configuration hash is bound before any freeze."""
    campaign = parent_campaign["manifest"]["campaign"]
    hashes = campaign["seed_policy"]["job_config_hashes"]
    assert sorted(hashes) == sorted(job["job_id"] for job in campaign["jobs"])
    assert all(len(value) == 16 for value in hashes.values())
    assert len(set(hashes.values())) == len(hashes)


def test_written_manifest_is_byte_identical_on_a_rerun(
    freeze, stage0_root, selection_path, tmp_path
):
    """Same inputs and same output root give byte-identical manifests."""
    first = ladder.write_ladder_campaign(
        tmp_path/"a",
        freeze,
        tier="selected",
        stage0_root=stage0_root,
        selection_artifact=selection_path,
        output_root=str(tmp_path/"run"),
        runner_command=RUNNER_COMMAND,
        freeze_path=FREEZE_PATH,
        campaign_uuid=LADDER_UUID,
    )
    second = ladder.write_ladder_campaign(
        tmp_path/"b",
        freeze,
        tier="selected",
        stage0_root=stage0_root,
        selection_artifact=selection_path,
        output_root=str(tmp_path/"run"),
        runner_command=RUNNER_COMMAND,
        freeze_path=FREEZE_PATH,
        campaign_uuid=LADDER_UUID,
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (
        first["manifest_path"].read_bytes() == second["manifest_path"].read_bytes()
    )


# ---------------------------------------------------------------------------
# Fail-closed gates
# ---------------------------------------------------------------------------


def test_writer_rejects_an_unknown_tier(
    freeze, stage0_root, selection_path, tmp_path
):
    """Only the two declared tiers may be emitted."""
    with pytest.raises(ladder.LadderError, match="declared ladder tiers"):
        _write_campaign(
            freeze, tmp_path/"campaign", stage0_root, selection_path, "golden"
        )


def test_writer_rejects_a_non_ratified_freeze(
    freeze, stage0_root, selection_path, tmp_path
):
    """The freeze_order clause forbids a ladder under an open freeze."""
    provisional = copy.deepcopy(freeze)
    provisional["freeze"]["status"] = "provisional"
    freeze_path = _write_yaml(tmp_path/"design_freeze.yaml", provisional)
    with pytest.raises(ladder.LadderError, match="not 'ratified'"):
        ladder.write_ladder_campaign(
            tmp_path/"campaign",
            provisional,
            tier="parent",
            stage0_root=stage0_root,
            selection_artifact=selection_path,
            output_root=str(tmp_path/"run"),
            runner_command=RUNNER_COMMAND,
            freeze_path=freeze_path,
        )


def test_writer_rejects_a_freeze_that_is_not_the_file_it_names(
    freeze, stage0_root, selection_path, tmp_path
):
    """A mutated in-memory freeze cannot travel under the committed digest."""
    mutated = copy.deepcopy(freeze)
    mutated["grid_sizing"]["max_side_px"] = 1024
    with pytest.raises(ladder.LadderError, match="is not the content of"):
        _write_campaign(
            freeze=mutated,
            directory=tmp_path/"campaign",
            stage0_root=stage0_root,
            selection_path=selection_path,
            tier="parent",
        )


def test_writer_rejects_a_runner_that_is_not_the_ladder_runner(
    freeze, stage0_root, selection_path, tmp_path
):
    """One ladder job is one member's whole walk, run by one runner."""
    with pytest.raises(ladder.LadderError, match="ladder runner"):
        ladder.write_ladder_campaign(
            tmp_path/"campaign",
            freeze,
            tier="parent",
            stage0_root=stage0_root,
            selection_artifact=selection_path,
            output_root=str(tmp_path/"run"),
            runner_command=["python", "scripts/run_something_else.py", "{config}"],
            freeze_path=FREEZE_PATH,
        )


def test_writer_rejects_a_selection_under_another_design_freeze(
    freeze, stage0_root, selection_copy, tmp_path
):
    """The selection and the ladder must share one design."""
    path = selection_copy(
        lambda document: document["design_freeze"].update({"sha256": "0"*64})
    )
    with pytest.raises(ladder.LadderError, match="computed under design freeze"):
        _write_campaign(freeze, tmp_path/"campaign", stage0_root, path, "parent")


def test_writer_rejects_a_selection_of_another_campaign(
    freeze, stage0_root, selection_copy, tmp_path
):
    """A selection computed on another pool cannot seed this ladder."""
    path = selection_copy(
        lambda document: document["campaign"].update(
            {"campaign_uuid": "99999999-8888-7777-6666-555555555555"}
        )
    )
    with pytest.raises(ladder.LadderError, match="selects from campaign"):
        _write_campaign(freeze, tmp_path/"campaign", stage0_root, path, "parent")


def test_writer_rejects_a_tier_that_is_not_the_frozen_size(
    freeze, stage0_root, selection_copy, tmp_path
):
    """A short selected tier fails against the freeze's strata sizes."""

    def shrink(document):
        """Drop one member from the selected tier."""
        document["selected_12"]["system_ids"].pop()
        document["selected_12"]["members"].pop()

    path = selection_copy(shrink)
    with pytest.raises(ladder.LadderError, match="strata.selected.size"):
        _write_campaign(freeze, tmp_path/"campaign", stage0_root, path, "selected")


def test_writer_rejects_a_golden_tier_outside_the_selected_tier(
    freeze, stage0_root, selection_copy, tmp_path
):
    """The golden 5 must be a subset of the selected 12."""

    def stray(document):
        """Replace one golden member with an unselected system."""
        document["golden_5"]["system_ids"][0] = "sys0059"
        document["golden_5"]["members"][0] = _member_record("sys0059", 60)

    path = selection_copy(stray)
    with pytest.raises(ladder.LadderError, match="not a subset"):
        _write_campaign(freeze, tmp_path/"campaign", stage0_root, path, "selected")


def test_writer_rejects_a_selection_rule_that_contradicts_the_freeze(
    freeze, stage0_root, selection_copy, tmp_path
):
    """A recorded tier size that disagrees with the freeze fails closed."""
    path = selection_copy(
        lambda document: document["rule"].update({"golden_size": 6})
    )
    with pytest.raises(ladder.LadderError, match="rule.golden_size"):
        _write_campaign(freeze, tmp_path/"campaign", stage0_root, path, "selected")


def test_writer_rejects_a_member_missing_from_the_pool(
    freeze, stage0_root, selection_copy, tmp_path
):
    """A selected id the Stage 0 campaign never ran fails closed."""

    def rename(document):
        """Point one selected member at a system outside the pool."""
        document["selected_12"]["system_ids"][-1] = "sys0999"
        document["selected_12"]["members"][-1] = _member_record("sys0055", 12)
        document["selected_12"]["members"][-1]["system_id"] = "sys0999"

    path = selection_copy(rename)
    with pytest.raises(ladder.LadderError, match="not a job of the Stage 0"):
        _write_campaign(freeze, tmp_path/"campaign", stage0_root, path, "selected")


def test_writer_rejects_a_selection_theta_e_that_is_not_the_realized_one(
    freeze, stage0_root, selection_copy, tmp_path
):
    """The ladder aperture must be the one the selection was computed in."""

    def drift(document):
        """Move one recorded theta_E_eff off the realized value."""
        document["selected_12"]["members"][0]["theta_e_eff_arcsec"] = 1.5

    path = selection_copy(drift)
    with pytest.raises(ladder.LadderError, match="selection artifact records"):
        _write_campaign(freeze, tmp_path/"campaign", stage0_root, path, "selected")


def test_writer_rejects_a_pool_artifact_from_another_campaign(
    freeze, selection_path, tmp_path
):
    """An artifact that does not carry this campaign's UUID is refused."""
    root = _write_stage0_campaign(freeze, tmp_path/"stage0")
    artifact = root/"outputs"/"sys0000"/"stage0_observation.npz"
    with np.load(artifact, allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    payload["campaign_uuid"] = np.asarray("00000000-0000-0000-0000-000000000000")
    np.savez(artifact, **payload)
    with pytest.raises(ladder.LadderError, match="campaign_uuid"):
        _write_campaign(freeze, tmp_path/"campaign", root, selection_path, "parent")


def test_writer_rejects_a_pool_member_without_a_harvested_artifact(
    freeze, selection_path, tmp_path
):
    """The ladder consumes a harvested campaign, not a staged one."""
    root = _write_stage0_campaign(freeze, tmp_path/"stage0")
    (root/"outputs"/"sys0000"/"stage0_observation.npz").unlink()
    with pytest.raises(ladder.LadderError, match="no harvested artifact"):
        _write_campaign(freeze, tmp_path/"campaign", root, selection_path, "parent")


def test_writer_rejects_a_staged_config_that_has_moved(
    freeze, selection_path, tmp_path
):
    """A staged configuration edited after the Stage 0 freeze fails closed."""
    root = _write_stage0_campaign(freeze, tmp_path/"stage0")
    staged = root/"configs"/"sys0000.yaml"
    staged.write_bytes(staged.read_bytes() + b"\nextra_key: 1\n")
    with pytest.raises(ladder.LadderError, match="does not match the frozen"):
        _write_campaign(freeze, tmp_path/"campaign", root, selection_path, "parent")


def test_writer_rejects_an_aperture_hash_that_is_not_the_stage0_one(
    freeze, selection_path, tmp_path
):
    """The recomputed aperture must hash to the Stage 0 aperture digest."""
    root = _write_stage0_campaign(freeze, tmp_path/"stage0")
    artifact = root/"outputs"/"sys0000"/"stage0_observation.npz"
    with np.load(artifact, allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    payload["aperture_sha256"] = np.asarray("c"*64)
    np.savez(artifact, **payload)
    with pytest.raises(ladder.LadderError, match="staged configuration value"):
        _write_campaign(freeze, tmp_path/"campaign", root, selection_path, "parent")


def test_writer_rejects_a_stage0_root_that_is_not_frozen(
    freeze, selection_path, tmp_path
):
    """A directory that holds no frozen manifest is not a Stage 0 campaign."""
    empty = tmp_path/"empty"
    empty.mkdir()
    with pytest.raises(ladder.LadderError, match="not a frozen campaign"):
        _write_campaign(freeze, tmp_path/"campaign", empty, selection_path, "parent")


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_validation_accepts_the_written_manifests(
    parent_campaign, selected_campaign
):
    """Both tiers pass the full ladder validation as written."""
    for campaign in (parent_campaign, selected_campaign):
        normalized = ladder.validate_ladder_manifest(campaign["manifest_path"])
        assert normalized["campaign"]["expected_artifacts"] == [
            "ladder_result.npz"
        ]


@pytest.fixture
def manifest_copy(freeze, stage0_root, selection_path, tmp_path):
    """Write one ladder campaign a test may tamper with freely."""
    written = _write_campaign(
        freeze, tmp_path/"campaign", stage0_root, selection_path, "selected"
    )
    return written["manifest_path"]


def _rewrite_manifest(manifest_path, mutate):
    """Rewrite one manifest through a mutation of its campaign block."""
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    mutate(manifest["campaign"])
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=True)
    return manifest


def test_validation_rejects_a_dropped_no_rng_declaration(manifest_copy):
    """A manifest that stops declaring the RNG policy fails closed."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["seed_policy"].pop("consumes_random_stream"),
    )
    with pytest.raises(ladder.LadderError, match="consumes_random_stream"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_claimed_random_stream(manifest_copy):
    """Adding a stream requires a freeze amendment, not a manifest edit."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["seed_policy"].update(
            {"consumes_random_stream": True}
        ),
    )
    with pytest.raises(ladder.LadderError, match="must be false"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_tier_short_of_the_frozen_size(manifest_copy):
    """A dropped job no longer fills the frozen selected tier."""

    def drop(campaign):
        """Remove one job and keep the declared count consistent."""
        campaign["jobs"].pop()
        campaign["expected_job_count"] -= 1
        campaign["seed_policy"]["job_config_hashes"].popitem()

    _rewrite_manifest(manifest_copy, drop)
    with pytest.raises(ladder.LadderError, match="strata.selected.size"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_ladder_block_off_the_frozen_policy(manifest_copy):
    """A hand-edited mass ladder is not the frozen policy."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["jobs"][0]["overrides"]["ladder"][
            "mass_ladder"
        ]["coarse"].update({"high": 10.5}),
    )
    with pytest.raises(ladder.LadderError, match="not the frozen"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_job_without_the_science35_state(manifest_copy):
    """A job whose staged aberrations are not science35 fails closed."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["jobs"][0]["overrides"]["psf"][
            "aberrations"
        ].update({"enable_global_zernikes": False}),
    )
    with pytest.raises(ladder.LadderError, match="science35"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_moved_node_spacing(manifest_copy):
    """The A2 ruling pins the node spacing every quoted M_lim carries."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["jobs"][0]["overrides"]["ladder"].update(
            {"node_spacing_arcsec": 0.02}
        ),
    )
    with pytest.raises(ladder.LadderError, match="node_spacing_arcsec"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_an_extra_ladder_field(manifest_copy):
    """The ladder block holds the declared field set and nothing else."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["jobs"][0]["overrides"]["ladder"].update(
            {"extra": True}
        ),
    )
    with pytest.raises(ladder.LadderError, match="not the declared set"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_grid_that_the_aperture_rule_does_not_size(
    manifest_copy,
):
    """A hand-edited grid no longer follows from the member's aperture."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["jobs"][0]["overrides"]["lensing"][
            "grid"
        ].update({"shape": [128, 128]}),
    )
    with pytest.raises(ladder.LadderError, match="lensing.grid.shape"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_an_aperture_off_its_own_theta_e(manifest_copy):
    """The aperture arithmetic must re-derive from the staged inputs."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["jobs"][0]["overrides"]["ladder"][
            "aperture"
        ].update({"radius_arcsec": 3.0}),
    )
    with pytest.raises(ladder.LadderError, match="radius_arcsec"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_stale_staged_config_hash(manifest_copy):
    """An overrides edit that the recorded hashes do not cover fails closed."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["jobs"][0]["overrides"].update(
            {"global_seed": 424242}
        ),
    )
    with pytest.raises(ladder.LadderError, match="job_config_hashes"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_moved_selection_artifact(
    manifest_copy, selection_path, tmp_path
):
    """A selection artifact whose bytes changed after generation fails."""
    moved = tmp_path/"moved_selection.yaml"
    moved.write_bytes(selection_path.read_bytes() + b"\n# amended\n")
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["seed_policy"].update(
            {"selection_artifact_path": str(moved)}
        ),
    )
    with pytest.raises(ladder.LadderError, match="hash to"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_a_missing_stage0_frozen_manifest(
    manifest_copy, tmp_path
):
    """The pool the ladder was built from must still be on disk."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign["seed_policy"].update(
            {"stage0_root": str(tmp_path/"absent")}
        ),
    )
    with pytest.raises(ladder.LadderError, match="does not exist"):
        ladder.validate_ladder_manifest(manifest_copy)


def test_validation_rejects_another_expected_artifact(manifest_copy):
    """A ladder job writes the ladder artifact and declares only it."""
    _rewrite_manifest(
        manifest_copy,
        lambda campaign: campaign.update(
            {"expected_artifacts": ["stage0_observation.npz"]}
        ),
    )
    with pytest.raises(ladder.LadderError, match="declared ladder artifact"):
        ladder.validate_ladder_manifest(manifest_copy)
