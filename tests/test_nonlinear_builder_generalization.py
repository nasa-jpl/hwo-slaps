"""Tests for generalized nonlinear model construction and runner identity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json
from types import SimpleNamespace

import autofit as af
import autolens as al
import numpy as np
import pytest
import yaml

import hwoslaps.modeling.nonlinear as nonlinear
from hwoslaps.lensing.generator import (
    _create_lens_galaxy,
    _create_source_galaxy,
)
from hwoslaps.lensing.mass_models import (
    concentration_mass_relation,
    einstein_radius_point_mass,
    einstein_radius_sis_m200,
    nfw_lensing_parameters,
)
from hwoslaps.modeling.nonlinear.autolens_model_builder import (
    autofit_model_from_spec,
    smooth_model_spec_from_config,
    subhalo_model_spec_from_trial,
)
from hwoslaps.modeling.nonlinear.autolens_runner import (
    AutoLensFitRunner,
    NonlinearSearchSettings,
    _n_like_max_reached,
    analysis_key_from,
)
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial
from hwoslaps.modeling.nonlinear.trial import (
    trial_from_fisher_map_position,
)


SCENE_NAMES = (
    "scene1_smooth_ring.yaml",
    "scene2_clumpy.yaml",
    "scene3_bow_dot.yaml",
    "scene4_cosmos.yaml",
    "scene5_ablation_sie_fit.yaml",
    "scene5_flex_macro.yaml",
)


def _scene(name):
    """Load one scene configuration."""
    with open(f"configs/scenes/{name}", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _simple_config():
    """Return the canonical legacy nonlinear-builder configuration."""
    return {
        "lensing": {
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "centre": [0.0, 0.0],
                    "einstein_radius": 1.0,
                    "ell_comps": [0.1, 0.0],
                },
            },
            "source_galaxy": {
                "redshift": 0.6,
                "light": {
                    "type": "Exponential",
                    "centre": [-0.03, 0.08],
                    "ell_comps": [0.1, 0.2],
                    "intensity": 2.0,
                    "effective_radius": 0.11,
                },
            },
        },
    }


def _legacy_trial():
    """Return a fixed-profile trial for legacy subhalo builders."""
    return SubhaloTrial(
        case_id="legacy",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.1, -0.1),
        model="NFW",
        profile_class="NFWSph",
        lens_redshift=0.2,
        source_redshift=0.6,
        kappa_s=0.01,
        scale_radius_arcsec=0.2,
    )


def test_builder_guards_throughput_and_unknown_profiles():
    """Accept unit throughput and reject unresolved or unknown models."""
    config = _simple_config()
    config["observation"] = {"throughput": 1.0}
    smooth_model_spec_from_config(config)
    config["observation"]["throughput"] = 0.5
    with pytest.raises(ValueError, match="normalization"):
        smooth_model_spec_from_config(config)

    unknown_mass = _simple_config()
    unknown_mass["lensing"]["lens_galaxy"]["mass"]["type"] = "BadMass"
    with pytest.raises(ValueError, match="truth mass"):
        smooth_model_spec_from_config(unknown_mass)
    unknown_light = _simple_config()
    unknown_light["lensing"]["source_galaxy"]["light"]["type"] = "BadLight"
    with pytest.raises(ValueError, match="source light"):
        smooth_model_spec_from_config(unknown_light)


def test_flexible_guard_is_removed_and_power_law_builds():
    """Expose no legacy flexible-lens guard and build PowerLaw truth."""
    assert not hasattr(nonlinear, "guard_flexible_lens_nonlinear")
    spec = smooth_model_spec_from_config(_scene("scene5_flex_macro.yaml"))
    assert spec.galaxies["lens"].components["mass"].class_name == "PowerLaw"


def test_fit_lens_explicit_routes_macro_but_inherits_truth_redshift():
    """Use explicit SIE parameters while retaining the truth lens plane."""
    explicit = _scene("scene5_ablation_sie_fit.yaml")
    explicit_spec = smooth_model_spec_from_config(explicit)
    lens = explicit_spec.galaxies["lens"]
    assert lens.redshift.value == pytest.approx(
        explicit["lensing"]["lens_galaxy"]["redshift"]
    )
    assert lens.components["mass"].class_name == "Isothermal"
    assert set(lens.components) == {"mass"}
    assert lens.components["mass"].parameters["einstein_radius"].lower == pytest.approx(
        0.99
    )

    matched = _scene("scene5_flex_macro.yaml")
    matched_spec = smooth_model_spec_from_config(matched)
    assert matched_spec.galaxies["lens"].components["mass"].class_name == "PowerLaw"
    absent = _simple_config()
    assert smooth_model_spec_from_config(absent).galaxies["lens"].components[
        "mass"
    ].class_name == "Isothermal"


def test_make_analysis_rejects_jax_for_custom_profiles(tmp_path):
    """Reject JAX before constructing analysis for a CPU-only model."""
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True),
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="requires CPU"):
        runner.make_analysis(None, model_metadata={"requires_cpu": True})


@pytest.mark.parametrize("scene_name", ["scene2_clumpy.yaml", "scene4_cosmos.yaml"])
@pytest.mark.parametrize("fit_mode", ["fixed_template", "local_search"])
def test_legacy_subhalo_specs_preserve_custom_source_metadata(
    scene_name,
    fit_mode,
):
    """Carry CPU and image provenance through both legacy fit modes."""
    spec = subhalo_model_spec_from_trial(
        _scene(scene_name),
        _legacy_trial(),
        fit_mode=fit_mode,
    )
    assert spec.metadata["requires_cpu"] is True
    if scene_name == "scene4_cosmos.yaml":
        assert spec.metadata["image_source_asset_hash"]


def test_builder_metadata_drives_legacy_jax_guard(tmp_path):
    """Reject JAX using metadata from a legacy custom-source builder."""
    spec = subhalo_model_spec_from_trial(
        _scene("scene2_clumpy.yaml"),
        _legacy_trial(),
        fit_mode="fixed_template",
    )
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True),
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="requires CPU"):
        runner.make_analysis(None, model_metadata=spec.metadata)


def test_scene5_macro_count_and_link_identity():
    """Build 12 macro freedoms with shared PowerLaw multipole priors."""
    spec = smooth_model_spec_from_config(_scene("scene5_flex_macro.yaml"))
    model = autofit_model_from_spec(spec)
    assert model.galaxies.lens.prior_count == 12
    mass = model.galaxies.lens.mass
    for name in ("multipole_m3", "multipole_m4"):
        multipole = getattr(model.galaxies.lens, name)
        assert multipole.centre.centre_0 is mass.centre.centre_0
        assert multipole.centre.centre_1 is mass.centre.centre_1
        assert multipole.einstein_radius is mass.einstein_radius
        assert multipole.slope is mass.slope


def test_single_multipole_plus_shear_has_ten_macro_parameters():
    """Count six PowerLaw, two multipole, and two shear freedoms."""
    config = _scene("scene5_flex_macro.yaml")
    config["lensing"]["lens_galaxy"]["mass"]["multipoles"].pop("m4")
    model = autofit_model_from_spec(smooth_model_spec_from_config(config))
    assert model.galaxies.lens.prior_count == 10


@pytest.mark.parametrize("scene_name", SCENE_NAMES)
def test_all_scene_truth_point_tracer_images_match(scene_name):
    """Match every scene's intended fit-side truth tracer image."""
    config = _scene(scene_name)
    spec = smooth_model_spec_from_config(config)
    instance = autofit_model_from_spec(spec).instance_from_prior_medians()
    truth_lens_config = config["lensing"]["lens_galaxy"]
    fit_lens = config.get("modeling", {}).get("fit_lens")
    if isinstance(fit_lens, dict) and fit_lens.get("mode") == "explicit":
        truth_lens_config = {
            "redshift": config["lensing"]["lens_galaxy"]["redshift"],
            **fit_lens["lens_galaxy"],
        }
    truth_lens = _create_lens_galaxy(truth_lens_config)
    truth_source = _create_source_galaxy(
        config["lensing"]["source_galaxy"]
    )
    truth = al.Tracer(galaxies=[truth_lens, truth_source])
    fitted = al.Tracer(
        galaxies=[
            instance.galaxies.lens,
            instance.galaxies.source,
        ]
    )
    grid = al.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.04)
    np.testing.assert_allclose(
        np.asarray(fitted.image_2d_from(grid)),
        np.asarray(truth.image_2d_from(grid)),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_multipole_truth_convention_matches_generator_deflections():
    """Pin fit-side multipole component orientation to generator truth."""
    config = _scene("scene5_flex_macro.yaml")
    truth = _create_lens_galaxy(config["lensing"]["lens_galaxy"])
    fitted = autofit_model_from_spec(
        smooth_model_spec_from_config(config)
    ).instance_from_prior_medians().galaxies.lens
    grid = al.Grid2D.uniform(shape_native=(9, 9), pixel_scales=0.08)
    for name in ("multipole_m3", "multipole_m4"):
        np.testing.assert_allclose(
            np.asarray(getattr(fitted, name).deflections_yx_2d_from(grid)),
            np.asarray(getattr(truth, name).deflections_yx_2d_from(grid)),
            rtol=1.0e-12,
            atol=1.0e-14,
        )


def test_slope_sersic_and_ellipticity_prior_clipping():
    """Clip physical boxes and reject truth outside their safe domains."""
    power_law = _scene("scene5_flex_macro.yaml")
    power_law["lensing"]["lens_galaxy"]["mass"]["slope"] = 1.21
    spec = smooth_model_spec_from_config(
        power_law,
        priors_config={"lens_slope_sigma": 0.2},
    )
    slope = spec.galaxies["lens"].components["mass"].parameters["slope"]
    assert 1.2 < slope.lower < 1.21 < slope.upper

    bad_slope = deepcopy(power_law)
    bad_slope["lensing"]["lens_galaxy"]["mass"]["slope"] = 1.2
    with pytest.raises(ValueError, match="outside"):
        smooth_model_spec_from_config(bad_slope)

    sersic = _simple_config()
    light = sersic["lensing"]["source_galaxy"]["light"]
    light["type"] = "Sersic"
    light["sersic_index"] = 0.4
    spec = smooth_model_spec_from_config(
        sersic,
        priors_config={"source_sersic_index_sigma": 1.0},
    )
    index = spec.galaxies["source"].components["light"].parameters[
        "sersic_index"
    ]
    assert index.lower == pytest.approx(0.3)

    bad_ell = _simple_config()
    bad_ell["lensing"]["lens_galaxy"]["mass"]["ell_comps"][0] = 0.9
    with pytest.raises(ValueError, match="outside"):
        smooth_model_spec_from_config(bad_ell)

    clipped_ell = _simple_config()
    clipped_ell["lensing"]["lens_galaxy"]["mass"]["ell_comps"][0] = 0.89
    spec = smooth_model_spec_from_config(
        clipped_ell,
        priors_config={"lens_ell_comps_sigma": 0.05},
    )
    ell_prior = spec.galaxies["lens"].components["mass"].parameters[
        "ell_comps_0"
    ]
    assert ell_prior.upper < 0.9
    assert ell_prior.lower < 0.89 < ell_prior.upper


def test_image_source_spec_is_compact_and_has_four_free_parameters():
    """Build the image asset once and keep its array out of payloads."""
    config = _scene("scene4_cosmos.yaml")
    spec = smooth_model_spec_from_config(config)
    source = spec.galaxies["source"].components["light"]
    model = autofit_model_from_spec(spec)
    assert model.galaxies.source.prior_count == 4
    assert source.parameters["rotation_deg"].value == pytest.approx(0.0)
    assert source.parameters["total_flux"].value == pytest.approx(
        config["lensing"]["source_galaxy"]["light"]["total_flux"]
    )
    payload = spec.to_dict()
    encoded = json.dumps(payload)
    asset_hash = source.parameters["asset"].value.sha256_16
    assert asset_hash in encoded
    assert "[[" not in encoded
    assert len(encoded) < 10000


class _Native:
    """Minimal object exposing a native array."""

    def __init__(self, values):
        self.native = np.asarray(values, dtype=float)


def _identity_inputs():
    """Return a minimal dataset, metadata, and model metadata triple."""
    dataset = SimpleNamespace(
        data=_Native([[1.0, 2.0]]),
        noise_map=_Native([[0.1, 0.2]]),
        psf=_Native([[0.0, 1.0, 0.0]]),
    )
    metadata = {
        "dataset_kind": "asimov",
        "background_treatment": "subtract_known",
        "psf_truth_label": "truth",
        "psf_fit_label": "fit",
    }
    model_metadata = {
        "fit_mode": "freed",
        "clumpy_fit_parameterization": "host_free",
        "mass_context_hash": "mass-a",
        "image_source_asset_hash": "asset-a",
        "resolved_prior_widths": {"width": 0.1},
    }
    return dataset, metadata, model_metadata


def test_analysis_key_covers_dataset_and_model_identity():
    """Change every required identity field and reproduce equal inputs."""
    dataset, metadata, model_metadata = _identity_inputs()
    baseline = analysis_key_from(dataset, metadata, model_metadata)
    assert analysis_key_from(dataset, metadata, model_metadata) == baseline

    variants = []
    changed = deepcopy(metadata)
    changed["dataset_kind"] = "noisy"
    variants.append((dataset, changed, model_metadata))
    for attribute in ("data", "psf"):
        changed_dataset = deepcopy(dataset)
        setattr(changed_dataset, attribute, _Native([[3.0, 4.0]]))
        variants.append((changed_dataset, metadata, model_metadata))
    for key, value in (
        ("fit_mode", "fixed_template"),
        ("clumpy_fit_parameterization", "rigid"),
        ("mass_context_hash", "mass-b"),
        ("resolved_prior_widths", {"width": 0.2}),
    ):
        changed_model = deepcopy(model_metadata)
        changed_model[key] = value
        variants.append((dataset, metadata, changed_model))
    for values in variants:
        assert analysis_key_from(*values) != baseline


def test_nautilus_settings_and_search_name_are_exact(monkeypatch, tmp_path):
    """Forward current update fields and no stale Nautilus kwargs."""
    captured = {}

    class FakeNautilus:
        """Capture constructor keyword arguments."""

        def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag=None,
            n_live=10,
            number_of_cores=1,
            iterations_per_quick_update=None,
            iterations_per_full_update=None,
            n_like_max=None,
            seed=None,
            **kwargs,
        ):
            captured.update(locals())

    monkeypatch.setattr(af, "Nautilus", FakeNautilus)
    settings = NonlinearSearchSettings(
        iterations_per_quick_update=7,
        iterations_per_full_update=11,
        maxcall=13,
        seed=17,
    )
    runner = AutoLensFitRunner(settings, output_dir=tmp_path)
    runner._make_search("case", "smooth", 5, "abc123")
    assert captured["name"] == "case_smooth_abc123"
    assert captured["iterations_per_quick_update"] == 7
    assert captured["iterations_per_full_update"] == 11
    assert captured["seed"] == 17
    assert "iterations_per_update" not in captured
    assert "resume" not in {item.name for item in fields(settings)}
    assert captured["kwargs"] == {}


def test_nautilus_kwargs_only_constructor_receives_all_settings(
    monkeypatch,
    tmp_path,
):
    """Pass every non-null allowlisted setting without signature filtering."""
    captured = {}

    class KwargsOnlyNautilus:
        """Capture every keyword through a kwargs-only constructor."""

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(af, "Nautilus", KwargsOnlyNautilus)
    settings = NonlinearSearchSettings(
        number_of_cores=2,
        iterations_per_quick_update=7,
        iterations_per_full_update=11,
        maxcall=13,
        seed=17,
        path_prefix="prefix",
        unique_tag="tag",
    )
    runner = AutoLensFitRunner(settings, output_dir=tmp_path)
    runner._make_search("case", "subhalo", 5, "abc123")
    assert captured == {
        "path_prefix": str(tmp_path / "prefix"),
        "name": "case_subhalo_abc123",
        "unique_tag": "tag",
        "n_live": 5,
        "number_of_cores": 2,
        "iterations_per_quick_update": 7,
        "iterations_per_full_update": 11,
        "n_like_max": 13,
        "seed": 17,
    }


def test_n_like_max_uses_nautilus_samples_info_only():
    """Read the real call counter and report absent counters as unknown."""
    reached = SimpleNamespace(
        samples=SimpleNamespace(samples_info={"total_samples": 100})
    )
    not_reached = SimpleNamespace(
        samples=SimpleNamespace(samples_info={"total_samples": 79})
    )
    absent = SimpleNamespace(
        samples=SimpleNamespace(total_samples=100)
    )
    assert _n_like_max_reached(reached, 80) is True
    assert _n_like_max_reached(not_reached, 80) is False
    assert _n_like_max_reached(absent, 80) is None


def test_result_callback_success_and_warning_paths(monkeypatch, tmp_path):
    """Invoke callbacks on success without converting callback errors."""
    sample = SimpleNamespace(log_likelihood=-5.0)
    result = SimpleNamespace(
        samples=SimpleNamespace(
            max_log_likelihood=lambda: sample,
            samples_info={"total_samples": 10},
        )
    )

    class Search:
        """Return one synthetic successful result."""

        def fit(self, model, analysis):
            return result

    runner = AutoLensFitRunner(
        NonlinearSearchSettings(maxcall=10),
        output_dir=tmp_path,
    )
    monkeypatch.setattr(runner, "_make_search", lambda **kwargs: Search())
    received = []
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=3),
        analysis=object(),
        role="smooth",
        fit_mode="freed",
        case_id="case",
        n_live=5,
        analysis_key="key",
        result_callback=lambda raw, model: received.append((raw, model)),
    )
    assert summary.status == "success"
    assert summary.n_like_max_reached is True
    assert received[0][0] is result

    def raising_callback(result, model):
        raise RuntimeError("callback boom")

    warned = runner.run_model(
        model=SimpleNamespace(total_free_parameters=3),
        analysis=object(),
        role="smooth",
        fit_mode="freed",
        case_id="case",
        n_live=5,
        analysis_key="key",
        result_callback=raising_callback,
    )
    assert warned.status == "success"
    assert "callback boom" in warned.warnings[0]


def test_fisher_map_trial_recomputes_mismatched_mass_scales():
    """Recompute NFW scales for new masses and preserve equal-mass truth."""
    config = _scene("scene1_smooth_ring.yaml")
    reference = SimpleNamespace(
        subhalo_model="NFW",
        subhalo_mass=1.0e7,
        lens_redshift=0.2,
        source_redshift=0.6,
        subhalo_kappa_s=0.01,
        subhalo_scale_radius_arcsec=0.2,
        subhalo_concentration=20.0,
        subhalo_concentration_model="moline2017_eq7",
        subhalo_einstein_radius=None,
    )
    changed = trial_from_fisher_map_position(
        config,
        reference,
        mass_msun=1.0e8,
        position_yx_arcsec=(0.1, -0.1),
    )
    concentration = concentration_mass_relation(
        1.0e8,
        model="moline2017_eq7",
        x_sub=1.0,
        h=0.6774,
    )
    expected = nfw_lensing_parameters(
        1.0e8,
        concentration,
        0.2,
        0.6,
        al.cosmo.Planck15(),
    )
    assert changed.kappa_s == pytest.approx(expected[0], rel=1.0e-10)
    assert changed.scale_radius_arcsec == pytest.approx(
        expected[1], rel=1.0e-10
    )
    assert changed.metadata["profile_scales_source"] == "recomputed"

    equal = trial_from_fisher_map_position(
        config,
        reference,
        mass_msun=1.0e7,
        position_yx_arcsec=(0.1, -0.1),
    )
    assert equal.kappa_s == pytest.approx(0.01)
    assert equal.scale_radius_arcsec == pytest.approx(0.2)
    assert equal.metadata["profile_scales_source"] == "reference"


@pytest.mark.parametrize(
    "model,helper",
    [
        ("SIS", einstein_radius_sis_m200),
        ("PointMass", einstein_radius_point_mass),
    ],
)
def test_fisher_map_trial_recomputes_non_nfw_mass_scales(model, helper):
    """Recompute SIS and point-mass radii for mismatched trial masses."""
    config = _scene("scene1_smooth_ring.yaml")
    config["lensing"]["subhalo"]["model"] = model
    reference = SimpleNamespace(
        subhalo_model=model,
        subhalo_mass=1.0e7,
        lens_redshift=0.2,
        source_redshift=0.6,
        subhalo_einstein_radius=0.01,
    )
    trial = trial_from_fisher_map_position(
        config,
        reference,
        mass_msun=2.0e7,
        position_yx_arcsec=(0.1, -0.1),
    )
    expected = helper(
        2.0e7,
        0.2,
        0.6,
        al.cosmo.Planck15(),
    )
    assert trial.einstein_radius_arcsec == pytest.approx(expected)
    assert trial.kappa_s is None
    assert trial.scale_radius_arcsec is None
    assert trial.concentration is None
    assert trial.metadata["profile_scales_source"] == "recomputed"
