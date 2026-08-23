"""Contracts for the Stage 0 sampler and S1-lite campaign generator.

The sampler tests are pure and run without the lensing engine. The
manifest tests need the engine because every system's grid is sized from
its extracted ``theta_E_eff``, so they run on a deliberately small pool.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from hwoslaps.campaign import design_freeze as df
from hwoslaps.campaign import stage0


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT/"configs"/"design"/"design_freeze_v1.yaml"

SMALL_POOL = 10
"""Pool size for the engine-backed manifest tests (`int`)."""


@pytest.fixture(scope="module")
def freeze():
    """Load the committed design freeze once."""
    return df.load_design_freeze(FREEZE_PATH)


def _write_freeze(directory, document):
    """Write one freeze document to a temporary file and return its path."""
    path = Path(directory)/"design_freeze.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=True)
    return path


def _rewrite_manifest(manifest_path, mutate):
    """Rewrite one manifest through a mutation of its campaign block."""
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    mutate(manifest["campaign"])
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=True)
    return manifest


@pytest.fixture(scope="module")
def pool(freeze):
    """Sample the full declared Stage 0 pool once."""
    return stage0.sample_pool(freeze)


@pytest.fixture(scope="module")
def runner_command():
    """Return a runner command carrying the mandatory placeholder."""
    return ["python", "scripts/run_stage0_observation.py", "{config}"]


def test_system_ids_are_padded_and_unique(freeze, pool):
    """Job ids sort in index order and match the S1-lite pattern."""
    ids = [record["system_id"] for record in pool]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "sys0000"
    assert ids[-1] == "sys0999"
    assert ids == sorted(ids)
    assert all(character in "abcdefghijklmnopqrstuvwxyz0123456789_"
               for identifier in ids for character in identifier)


def test_sampling_is_deterministic_for_a_repeated_index(freeze):
    """The same index redraws bit-identically."""
    first = stage0.sample_system(freeze, 137)
    second = stage0.sample_system(freeze, 137)
    assert first == second


def test_sampling_does_not_depend_on_pool_size(freeze):
    """A system regenerated inside a smaller pool is unchanged."""
    small = stage0.sample_pool(freeze, SMALL_POOL)
    for record in small:
        standalone = stage0.sample_system(freeze, record["index"])
        for key, value in standalone.items():
            assert record[key] == value, key


def test_different_indices_draw_different_systems(freeze):
    """Neighbouring indices are independent draws, not a shared stream."""
    first = stage0.sample_system(freeze, 0)
    second = stage0.sample_system(freeze, 1)
    assert first["z_lens"] != second["z_lens"]
    assert first["sigma_v_km_s"] != second["sigma_v_km_s"]


def test_every_sampled_variable_respects_its_declared_bounds(freeze, pool):
    """No draw leaves the support the design declares."""
    design = freeze["parent_design"]["distributions"]
    for record in pool:
        assert design["z_lens"]["low"] <= record["z_lens"] <= design["z_lens"]["high"]
        source_low = max(
            design["z_source"]["low_floor"],
            record["z_lens"] + design["z_source"]["min_separation"],
        )
        assert source_low <= record["z_source"] <= design["z_source"]["high"]
        assert record["z_source"] - record["z_lens"] >= (
            design["z_source"]["min_separation"] - 1e-12
        )
        assert (
            design["sigma_v"]["low"]
            <= record["sigma_v_km_s"]
            <= design["sigma_v"]["high"]
        )
        assert (
            design["lens_axis_ratio"]["low"]
            <= record["lens_axis_ratio"]
            <= design["lens_axis_ratio"]["high"]
        )
        assert 0.0 <= record["lens_position_angle_deg"] <= 180.0
        assert (
            design["caustic_offset_fraction"]["low"]
            <= record["caustic_offset_fraction"]
            <= design["caustic_offset_fraction"]["high"]
        )
        assert 0.0 <= record["caustic_offset_azimuth_deg"] <= 360.0
        assert (
            design["source_magnitude_ab"]["low"]
            <= record["source_magnitude_ab"]
            <= design["source_magnitude_ab"]["high"]
        )
        assert (
            design["source_half_light_radius_arcsec"]["low"]
            <= record["source_half_light_radius_arcsec"]
            <= design["source_half_light_radius_arcsec"]["high"]
        )
        assert 0.0 <= record["source_rotation_deg"] <= 360.0


def test_template_allocation_is_exactly_balanced(freeze, pool):
    """Each of the five levels receives exactly 200 systems."""
    counts: dict = {}
    for record in pool:
        counts[record["source_template"]] = counts.get(record["source_template"], 0) + 1
    levels = [level["id"] for level in freeze["templates"]["levels"]]
    assert sorted(counts) == sorted(levels)
    assert set(counts.values()) == {freeze["templates"]["per_level"]}


def test_template_allocation_is_seeded_and_repeatable(freeze):
    """The balanced permutation redraws identically."""
    assert stage0.assign_templates(freeze) == stage0.assign_templates(freeze)


def test_template_allocation_rejects_an_unbalanced_pool(freeze):
    """A pool size the levels cannot divide fails closed."""
    with pytest.raises(stage0.Stage0Error, match="divisible"):
        stage0.assign_templates(freeze, 7)


def test_theta_e_floor_survival_matches_the_design_sanity_check(freeze, pool):
    """About 96.6 per cent of the parent clears the 0.5 arcsec floor."""
    summary = stage0.pool_summary(freeze, pool)
    fraction = summary["theta_e_floor_survival_fraction"]
    assert 0.93 <= fraction <= 0.99
    assert fraction == pytest.approx(0.966, abs=0.02)
    assert summary["template_balance"] == {
        level["id"]: freeze["templates"]["per_level"]
        for level in freeze["templates"]["levels"]
    }


def test_induced_theta_e_reproduces_the_design_anchor(freeze, pool):
    """The induced mean theta_E sits at the SLACS-calibrated anchor."""
    summary = stage0.pool_summary(freeze, pool)
    assert summary["theta_e_design_arcsec_mean"] == pytest.approx(1.15, abs=0.06)
    quantiles = summary["quantiles"]["theta_e_design_arcsec"]
    assert quantiles["p50"] == pytest.approx(1.075, abs=0.06)
    assert quantiles["p99"] == pytest.approx(2.42, abs=0.15)
    assert summary["theta_e_design_arcsec_max"] < 4.0


def test_einstein_radius_scales_as_sigma_squared(freeze):
    """theta_E is quadratic in the velocity dispersion at fixed geometry."""
    base = stage0.einstein_radius_arcsec(200.0, 0.2, 0.6)
    doubled = stage0.einstein_radius_arcsec(400.0, 0.2, 0.6)
    assert doubled/base == pytest.approx(4.0, rel=1e-9)


def test_einstein_radius_rejects_an_unordered_pair(freeze):
    """A source in front of the lens fails closed."""
    with pytest.raises(stage0.Stage0Error, match="does not exceed"):
        stage0.einstein_radius_arcsec(250.0, 0.6, 0.2)


def test_area_equivalent_factor_is_the_analytic_one():
    """The isothermal factor is 2 sqrt(q) / (1 + q), unity when circular."""
    assert stage0.sie_area_equivalent_factor(1.0) == pytest.approx(1.0)
    for axis_ratio in (0.4, 0.5, 0.6, 0.75, 0.95):
        assert stage0.sie_area_equivalent_factor(axis_ratio) == pytest.approx(
            2.0*math.sqrt(axis_ratio)/(1.0 + axis_ratio)
        )


def test_macro_parameter_inverts_the_area_factor():
    """Solving and re-applying the factor round-trips to theta_E."""
    for axis_ratio in (0.4, 0.75, 1.0):
        parameter = stage0.macro_einstein_radius_arcsec(1.3, axis_ratio)
        realized = parameter*stage0.sie_area_equivalent_factor(axis_ratio)
        assert realized == pytest.approx(1.3, rel=1e-12)


def test_engine_noise_seed_is_deterministic_and_in_range(freeze):
    """The 32-bit engine seed is reproducible and non-negative."""
    seed = stage0.engine_noise_seed(freeze, 5)
    assert seed == stage0.engine_noise_seed(freeze, 5)
    assert seed != stage0.engine_noise_seed(freeze, 6)
    assert 0 <= seed < 2**32


def test_selection_observable_plan_declares_what_is_computed(freeze):
    """The plan names the module, the plane and the frozen statistics."""
    plan = stage0.selection_observable_plan(freeze)
    assert plan["module"].endswith("selection_score.py")
    assert plan["computed_by"].endswith("run_stage0_observation.py")
    assert "noiseless" in plan["plane"]
    assert plan["score"] == "score = z(log S) + z(log C)"
    assert plan["floor_cuts"]["theta_e_arcsec_min"] == 0.5
    assert plan["floor_cuts"]["arc_snr_min"] == 20.0


# ---------------------------------------------------------------------------
# Engine-backed contracts
# ---------------------------------------------------------------------------

autolens = pytest.importorskip("autolens")


@pytest.fixture(scope="module")
def small_campaign(freeze, runner_command, tmp_path_factory):
    """Build one small Stage 0 campaign in memory."""
    return stage0.build_stage0_campaign(
        freeze,
        output_root=str(tmp_path_factory.mktemp("root")),
        runner_command=runner_command,
        freeze_path=FREEZE_PATH,
        n_systems=SMALL_POOL,
        allow_unfrozen_pool=True,
        campaign_uuid="11111111-2222-3333-4444-555555555555",
    )


def test_extracted_theta_e_matches_the_design_within_tolerance(small_campaign):
    """The T7 extraction confirms the solved macro parameter."""
    tolerance = 0.02
    for system in small_campaign["catalogue"]["systems"]:
        ratio = system["theta_e_realized_over_design"]
        assert abs(ratio - 1.0) <= tolerance, system["system_id"]
        assert system["theta_e_extraction"]["algorithm_id"] == (
            "tangential_critical_curve_marching_squares_v1"
        )
        assert len(system["theta_e_extraction"]["contour_sha256"]) == 64
        assert len(system["theta_e_extraction"]["aperture_sha256"]) == 64


def test_extraction_confirms_the_analytic_area_factor(freeze):
    """The isothermal area factor is measured, not assumed."""
    import autogalaxy as ag

    from hwoslaps.lensing import critical_curve as cc

    for axis_ratio in (0.5, 0.75):
        ell_comps = ag.convert.ell_comps_from(axis_ratio=axis_ratio, angle=30.0)
        galaxy = autolens.Galaxy(
            redshift=0.2,
            mass=autolens.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.0,
                ell_comps=tuple(ell_comps),
            ),
        )
        extraction = cc.extract_theta_e(
            galaxy,
            lens_centre_arcsec=(0.0, 0.0),
            grid=cc.CriticalCurveGrid(
                requested_half_width_arcsec=4.0, pixel_scale_arcsec=0.005
            ),
        )
        assert extraction.theta_e_eff_arcsec == pytest.approx(
            stage0.sie_area_equivalent_factor(axis_ratio), abs=2e-4
        )


def test_grid_sizing_follows_the_aperture_margin_rule(freeze, small_campaign):
    """Every grid covers 2 theta_E plus the declared margin, rounded even."""
    pixel_scale = freeze["grid_sizing"]["pixel_scale_arcsec"]
    factor = freeze["aperture"]["theta_e_factor"]
    margin = freeze["aperture"]["computational_margin_fraction"]
    for system in small_campaign["catalogue"]["systems"]:
        grid = system["grid"]
        extent = 2.0*factor*system["theta_e_eff_arcsec"]*(1.0 + margin)
        assert grid["required_map_extent_arcsec"] == pytest.approx(extent, rel=1e-12)
        assert grid["shape"][0] == grid["shape"][1]
        assert grid["shape"][0] % 2 == 0
        if not grid["grid_capped"]:
            assert grid["shape"][0]*pixel_scale >= extent
            assert (grid["shape"][0] - 2)*pixel_scale < extent
            assert grid["realized_coverage_theta_e"] >= factor


def test_grid_cap_flags_rather_than_silently_truncating(freeze):
    """A grid past the declared maximum is capped and flagged."""
    from hwoslaps.lensing.critical_curve import ApertureDefinition

    aperture = ApertureDefinition(
        centre_arcsec=(0.0, 0.0),
        theta_e_eff_arcsec=20.0,
        theta_e_factor=freeze["aperture"]["theta_e_factor"],
        computational_margin_fraction=freeze["aperture"][
            "computational_margin_fraction"
        ],
    )
    plan = stage0.grid_plan(freeze, aperture)
    assert plan["grid_capped"] is True
    assert plan["shape"][0] == freeze["grid_sizing"]["max_side_px"]
    assert plan["required_side_px"] > plan["shape"][0]
    assert plan["realized_coverage_theta_e"] < plan["requested_coverage_theta_e"]


def test_manifest_validates_against_the_s1_lite_schema(small_campaign):
    """The generated manifest is a valid S1-lite campaign manifest."""
    from hwoslaps.campaign.s1_lite import validate_campaign_manifest

    normalized = validate_campaign_manifest(
        copy.deepcopy(small_campaign["manifest"])
    )["campaign"]
    assert normalized["expected_job_count"] == SMALL_POOL
    assert normalized["expected_artifacts"] == ["stage0_observation.npz"]
    assert normalized["observing_reference"].endswith(
        "hwo_eac1_hri_reference_v1.yaml"
    )
    assert len(normalized["jobs"]) == SMALL_POOL


def test_every_job_disables_the_subhalo_and_pins_the_psf_kernel(small_campaign):
    """Stage 0 renders no-subhalo observations at the declared kernel."""
    for job in small_campaign["manifest"]["campaign"]["jobs"]:
        overrides = job["overrides"]
        assert overrides["lensing"]["subhalo"]["enabled"] is False
        assert overrides["psf"]["kernel"]["shape_native"] == [51, 51]
        assert overrides["modeling"]["enabled"] is False
        assert overrides["observation"]["exposure_time"] == 2000.0
        assert overrides["lensing"]["source_galaxy"]["light"]["type"] == "Image"


def test_source_flux_carries_the_magnitude_at_the_sampled_size(
    freeze, small_campaign
):
    """total_flux divides out the size factor the Image source applies."""
    templates = {
        level["id"]: level for level in freeze["templates"]["levels"]
    }
    for system, job in zip(
        small_campaign["catalogue"]["systems"],
        small_campaign["manifest"]["campaign"]["jobs"],
    ):
        template = templates[system["source_template"]]
        magnitude_scale = 10.0**(-0.4*(system["source_magnitude_ab"] - 24.845))
        expected = (
            template["canonical_total_flux"]
            * magnitude_scale
            / system["source_size_scale"]**2
        )
        light = job["overrides"]["lensing"]["source_galaxy"]["light"]
        assert light["total_flux"] == pytest.approx(expected, rel=1e-12)
        assert light["asset_path"] == template["asset_path"]
        assert light["size_scale"] == pytest.approx(
            system["source_half_light_radius_arcsec"]/0.11, rel=1e-12
        )
        assert job["overrides"]["stage0"][
            "target_unlensed_rate_e_per_s"
        ] == pytest.approx(8.951505744562876*magnitude_scale, rel=1e-12)


def test_each_job_binds_its_template_asset_by_digest(freeze, small_campaign):
    """The asset digest travels inside the job, not only in the catalogue."""
    templates = {level["id"]: level for level in freeze["templates"]["levels"]}
    for job in small_campaign["manifest"]["campaign"]["jobs"]:
        block = job["overrides"]["stage0"]
        template = templates[block["source_template"]]
        assert block["source_asset_path"] == template["asset_path"]
        assert block["source_asset_sha256"] == template["sha256"]
        assert block["source_asset_path"] == job["overrides"]["lensing"][
            "source_galaxy"
        ]["light"]["asset_path"]


def test_each_job_carries_the_generating_source_revision(small_campaign):
    """A resume under moved code is detectable from the job alone."""
    from hwoslaps.provenance import revision_digest, revision_provenance

    expected = revision_digest(revision_provenance())
    jobs = small_campaign["manifest"]["campaign"]["jobs"]
    for job in jobs:
        revision = job["overrides"]["stage0"]["code_revision"]
        assert revision["sha256"] == expected
        assert set(revision) == {"git_hash", "git_dirty", "sha256"}
    assert small_campaign["catalogue"]["code_revision"]["sha256"] == expected


def test_each_job_carries_the_frozen_extraction_settings(freeze, small_campaign):
    """The runner is handed the extraction the design declares."""
    algorithm = freeze["aperture"]["theta_e_algorithm"]
    for system, job in zip(
        small_campaign["catalogue"]["systems"],
        small_campaign["manifest"]["campaign"]["jobs"],
    ):
        block = job["overrides"]["stage0"]
        settings = block["theta_e_extraction"]
        assert settings["algorithm_id"] == algorithm["algorithm_id"]
        assert settings["choice_rule_id"] == algorithm["choice_rule_id"]
        assert settings["extraction_grid"] == {
            "pixel_scale_arcsec": algorithm["extraction_grid"][
                "pixel_scale_arcsec"
            ],
            "half_width_factor": algorithm["extraction_grid"][
                "half_width_factor"
            ],
        }
        assert settings["guards"] == {
            "closure_tolerance_pixels": algorithm["guards"][
                "closure_tolerance_pixels"
            ],
            "border_margin_pixels": algorithm["guards"]["border_margin_pixels"],
            "min_contour_vertices": algorithm["guards"]["min_contour_vertices"],
        }
        assert settings["theta_e_factor"] == freeze["aperture"]["theta_e_factor"]
        assert settings["computational_margin_fraction"] == freeze["aperture"][
            "computational_margin_fraction"
        ]
        assert block["theta_e_eff_arcsec"] == pytest.approx(
            system["theta_e_eff_arcsec"], rel=1e-12
        )
        assert block["theta_e_eff_tolerance_fractional"] == freeze["derived"][
            "einstein_radius"
        ]["verification"]["tolerance_fractional"]


def test_non_default_extraction_settings_reach_every_job(
    freeze, runner_command, tmp_path
):
    """A freeze that moves the extraction moves what the runner is given."""
    amended = copy.deepcopy(freeze)
    algorithm = amended["aperture"]["theta_e_algorithm"]
    algorithm["extraction_grid"]["half_width_factor"] = 4.5
    algorithm["guards"]["closure_tolerance_pixels"] = 0.75
    algorithm["guards"]["border_margin_pixels"] = 3.0
    algorithm["guards"]["min_contour_vertices"] = 40
    amended_path = _write_freeze(tmp_path/"design", amended)
    built = stage0.build_stage0_campaign(
        amended,
        output_root=str(tmp_path/"root"),
        runner_command=runner_command,
        freeze_path=amended_path,
        n_systems=5,
        allow_unfrozen_pool=True,
    )
    for job in built["manifest"]["campaign"]["jobs"]:
        settings = job["overrides"]["stage0"]["theta_e_extraction"]
        assert settings["extraction_grid"]["half_width_factor"] == 4.5
        assert settings["guards"] == {
            "closure_tolerance_pixels": 0.75,
            "border_margin_pixels": 3.0,
            "min_contour_vertices": 40,
        }
    for system in built["catalogue"]["systems"]:
        grid = system["theta_e_extraction"]["grid"]
        assert grid["requested_half_width_arcsec"] == pytest.approx(
            4.5*system["macro_einstein_radius_arcsec"], rel=1e-12
        )


def test_pool_size_deviation_is_recorded_in_the_manifest(small_campaign):
    """A campaign that is not the frozen pool says so in its seed policy."""
    policy = small_campaign["manifest"]["campaign"]["seed_policy"]
    assert policy["stage0_n_systems"] == SMALL_POOL
    assert policy["stage0_n_systems_frozen"] == 1000


def test_each_job_carries_its_rate_contract_target(freeze, small_campaign):
    """The P0-3 target rate travels inside the job that renders it."""
    tolerance = freeze["templates"]["rate_contract_production_tolerance"]
    for job in small_campaign["manifest"]["campaign"]["jobs"]:
        block = job["overrides"]["stage0"]
        assert block["system_id"] == job["job_id"]
        assert block["rate_contract_tolerance"] == tolerance
        assert block["target_unlensed_rate_e_per_s"] > 0.0


def test_source_centre_follows_the_declared_offset_geometry(small_campaign):
    """The source centre is the sampled offset at the sampled azimuth."""
    for system in small_campaign["catalogue"]["systems"]:
        beta = system["caustic_offset_fraction"]*system["theta_e_design_arcsec"]
        azimuth = math.radians(system["caustic_offset_azimuth_deg"])
        assert system["source_centre_arcsec"] == pytest.approx(
            [beta*math.sin(azimuth), beta*math.cos(azimuth)], rel=1e-12
        )
        assert system["source_offset_arcsec"] == pytest.approx(beta, rel=1e-12)


def test_written_campaign_is_byte_identical_on_a_rerun(
    freeze, runner_command, tmp_path
):
    """Same freeze and same output root give byte-identical artifacts."""
    output_root = str(tmp_path/"campaign")
    first = stage0.write_stage0_campaign(
        tmp_path/"a",
        freeze,
        output_root=output_root,
        runner_command=runner_command,
        freeze_path=FREEZE_PATH,
        n_systems=SMALL_POOL,
        allow_unfrozen_pool=True,
        campaign_uuid="11111111-2222-3333-4444-555555555555",
    )
    second = stage0.write_stage0_campaign(
        tmp_path/"b",
        freeze,
        output_root=output_root,
        runner_command=runner_command,
        freeze_path=FREEZE_PATH,
        n_systems=SMALL_POOL,
        allow_unfrozen_pool=True,
        campaign_uuid="11111111-2222-3333-4444-555555555555",
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["catalogue_sha256"] == second["catalogue_sha256"]
    assert (
        first["manifest_path"].read_bytes() == second["manifest_path"].read_bytes()
    )
    assert (
        first["catalogue_path"].read_bytes() == second["catalogue_path"].read_bytes()
    )


@pytest.fixture(scope="module")
def written_campaign(freeze, runner_command, tmp_path_factory):
    """Write one small Stage 0 campaign to disk once."""
    directory = tmp_path_factory.mktemp("written")
    return stage0.write_stage0_campaign(
        directory/"campaign",
        freeze,
        output_root=str(directory/"root"),
        runner_command=runner_command,
        freeze_path=FREEZE_PATH,
        n_systems=SMALL_POOL,
        allow_unfrozen_pool=True,
    )


@pytest.fixture
def campaign_copy(written_campaign, tmp_path):
    """Copy the written campaign so one test can tamper with it freely."""
    destination = tmp_path/"campaign"
    destination.mkdir(parents=True)
    for source in (
        written_campaign["manifest_path"],
        written_campaign["catalogue_path"],
    ):
        (destination/source.name).write_bytes(source.read_bytes())
    return destination/written_campaign["manifest_path"].name


def test_manifest_binds_the_catalogue_and_the_freeze_by_digest(written_campaign):
    """The seed policy carries both digests and validation re-hashes them."""
    with written_campaign["manifest_path"].open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    policy = manifest["campaign"]["seed_policy"]
    assert policy["design_freeze_sha256"] == df.design_freeze_digest(FREEZE_PATH)
    assert policy["design_freeze_path"] == str(FREEZE_PATH)
    assert policy["catalogue_sha256"] == hashlib.sha256(
        written_campaign["catalogue_path"].read_bytes()
    ).hexdigest()
    assert policy["entropy"] == 20260823
    assert policy["foreground_free_ceiling"] is True
    assert stage0.validate_stage0_manifest(written_campaign["manifest_path"])


def test_manifest_validation_rejects_a_byte_flipped_catalogue(campaign_copy):
    """One changed catalogue byte breaks the manifest's binding."""
    catalogue_path = campaign_copy.parent/"stage0_catalogue.json"
    payload = bytearray(catalogue_path.read_bytes())
    payload[-2] ^= 0x20
    catalogue_path.write_bytes(bytes(payload))
    with pytest.raises(stage0.Stage0Error, match="hash to"):
        stage0.validate_stage0_manifest(campaign_copy)


def test_manifest_validation_rejects_a_missing_catalogue(campaign_copy):
    """A manifest cannot carry a digest for a catalogue that is not there."""
    (campaign_copy.parent/"stage0_catalogue.json").unlink()
    with pytest.raises(stage0.Stage0Error, match="does not exist"):
        stage0.validate_stage0_manifest(campaign_copy)


def test_manifest_validation_rejects_a_garbage_catalogue_digest(campaign_copy):
    """A digest string that is not a sha256 fails closed, not silently."""
    _rewrite_manifest(
        campaign_copy,
        lambda campaign: campaign["seed_policy"].update(
            {"catalogue_sha256": "not-a-digest"}
        ),
    )
    with pytest.raises(stage0.Stage0Error, match="lowercase sha256 digest"):
        stage0.validate_stage0_manifest(campaign_copy)


def test_manifest_validation_rejects_an_all_zero_catalogue_digest(campaign_copy):
    """The digest a stub would carry no longer passes validation."""
    _rewrite_manifest(
        campaign_copy,
        lambda campaign: campaign["seed_policy"].update(
            {"catalogue_sha256": "0"*64}
        ),
    )
    with pytest.raises(stage0.Stage0Error, match="hash to"):
        stage0.validate_stage0_manifest(campaign_copy)


def test_manifest_validation_rejects_a_missing_design_freeze(campaign_copy):
    """The freeze the manifest names must be on disk."""
    _rewrite_manifest(
        campaign_copy,
        lambda campaign: campaign["seed_policy"].update(
            {"design_freeze_path": str(campaign_copy.parent/"absent.yaml")}
        ),
    )
    with pytest.raises(stage0.Stage0Error, match="does not exist"):
        stage0.validate_stage0_manifest(campaign_copy)


def test_manifest_validation_rejects_a_moved_design_freeze(campaign_copy):
    """A freeze whose bytes changed after generation fails closed."""
    amended = campaign_copy.parent/"amended_freeze.yaml"
    amended.write_bytes(FREEZE_PATH.read_bytes() + b"\n# amended\n")
    _rewrite_manifest(
        campaign_copy,
        lambda campaign: campaign["seed_policy"].update(
            {"design_freeze_path": str(amended)}
        ),
    )
    with pytest.raises(stage0.Stage0Error, match="hash to"):
        stage0.validate_stage0_manifest(campaign_copy)


def test_manifest_validation_rejects_a_catalogue_naming_another_freeze(
    campaign_copy,
):
    """The catalogue and the manifest must agree on which design ran."""
    catalogue_path = campaign_copy.parent/"stage0_catalogue.json"
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    catalogue["design_freeze"]["sha256"] = "1"*64
    payload = (json.dumps(catalogue, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    catalogue_path.write_bytes(payload)
    _rewrite_manifest(
        campaign_copy,
        lambda campaign: campaign["seed_policy"].update(
            {"catalogue_sha256": hashlib.sha256(payload).hexdigest()}
        ),
    )
    with pytest.raises(stage0.Stage0Error, match="records design freeze"):
        stage0.validate_stage0_manifest(campaign_copy)


def test_catalogue_records_the_provisional_status_and_claim_labels(
    freeze, runner_command, tmp_path
):
    """A downstream reader learns the design is unratified from the file."""
    written = stage0.write_stage0_campaign(
        tmp_path/"campaign",
        freeze,
        output_root=str(tmp_path/"root"),
        runner_command=runner_command,
        freeze_path=FREEZE_PATH,
        n_systems=SMALL_POOL,
        allow_unfrozen_pool=True,
    )
    catalogue = json.loads(written["catalogue_path"].read_text(encoding="utf-8"))
    assert catalogue["design_freeze"]["status"] == "provisional"
    assert catalogue["design_freeze"]["provisional_items"] == list(
        df.REQUIRED_PROVISIONAL_ITEMS
    )
    assert catalogue["foreground_free_ceiling"] is True
    assert "source-only information ceiling" in catalogue["claim_labels"][
        "central_result"
    ]
    assert len(catalogue["systems"]) == SMALL_POOL
    assert catalogue["summary"]["grid"]["capped_systems"] == []


def test_generator_rejects_a_system_whose_extraction_disagrees(
    freeze, runner_command, tmp_path
):
    """A macro that does not realize its design theta_E fails closed."""
    broken = copy.deepcopy(freeze)
    broken["derived"]["einstein_radius"]["verification"][
        "tolerance_fractional"
    ] = 1.0e-9
    broken_path = _write_freeze(tmp_path/"design", broken)
    with pytest.raises(stage0.Stage0Error, match="outside the declared"):
        stage0.build_stage0_campaign(
            broken,
            output_root=str(tmp_path/"root"),
            runner_command=runner_command,
            freeze_path=broken_path,
            n_systems=5,
            allow_unfrozen_pool=True,
        )


def test_builder_rejects_a_freeze_that_is_not_the_file_it_names(
    freeze, runner_command, tmp_path
):
    """A mutated in-memory freeze cannot travel under the committed digest."""
    mutated = copy.deepcopy(freeze)
    mutated["stage0"]["exposure_time_s"] = 1000.0
    with pytest.raises(stage0.Stage0Error, match="is not the content of"):
        stage0.build_stage0_campaign(
            mutated,
            output_root=str(tmp_path/"root"),
            runner_command=runner_command,
            freeze_path=FREEZE_PATH,
            n_systems=5,
            allow_unfrozen_pool=True,
        )


def test_builder_rejects_a_freeze_whose_bound_asset_moved(
    freeze, runner_command, tmp_path
):
    """A freeze that no longer matches its assets never reaches a build."""
    broken = copy.deepcopy(freeze)
    broken["templates"]["levels"][0]["sha256"] = "0"*64
    broken_path = _write_freeze(tmp_path/"design", broken)
    with pytest.raises(df.DesignFreezeError, match="does not match the frozen"):
        stage0.build_stage0_campaign(
            broken,
            output_root=str(tmp_path/"root"),
            runner_command=runner_command,
            freeze_path=broken_path,
            n_systems=5,
            allow_unfrozen_pool=True,
        )


def test_builder_rejects_an_unfrozen_pool_size_without_the_flag(
    freeze, runner_command, tmp_path
):
    """The freeze declares the pool; a deviation must be asked for by name."""
    with pytest.raises(stage0.Stage0Error, match="allow_unfrozen_pool"):
        stage0.build_stage0_campaign(
            freeze,
            output_root=str(tmp_path/"root"),
            runner_command=runner_command,
            freeze_path=FREEZE_PATH,
            n_systems=5,
        )


def test_each_job_binds_the_extracted_contour_by_digest(small_campaign):
    """The generator's exact curve travels inside the job it sized."""
    for system, job in zip(
        small_campaign["catalogue"]["systems"],
        small_campaign["manifest"]["campaign"]["jobs"],
    ):
        block = job["overrides"]["stage0"]
        extraction = system["theta_e_extraction"]
        assert block["theta_e_contour_sha256"] == extraction["contour_sha256"]
        assert block["theta_e_aperture_sha256"] == extraction["aperture_sha256"]


def test_builder_rejects_a_runner_that_is_not_the_frozen_one(freeze, tmp_path):
    """The freeze declares the runner and no driver may substitute another."""
    with pytest.raises(stage0.Stage0Error, match="frozen Stage 0 runner"):
        stage0.build_stage0_campaign(
            freeze,
            output_root=str(tmp_path/"root"),
            runner_command=["python", "scripts/run_something_else.py", "{config}"],
            freeze_path=FREEZE_PATH,
            n_systems=5,
            allow_unfrozen_pool=True,
        )


def test_pool_summary_reports_grid_and_ratio_ranges(small_campaign):
    """The summary carries the numbers the morning brief quotes."""
    summary = small_campaign["summary"]
    assert summary["grid"]["min_side_px"] <= summary["grid"]["max_side_px"]
    assert summary["grid"]["declared_max_side_px"] == 2048
    assert 0.99 < summary["theta_e_eff"]["min_realized_over_design"] < 1.01
    assert 0.99 < summary["theta_e_eff"]["max_realized_over_design"] < 1.01
    assert np.isfinite(summary["theta_e_design_arcsec_mean"])
