"""Prepared-case preflight, masks and real search/refinement/session transport."""

import dataclasses
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.inference.api import prepare_case, validate_nonlinear
from hwoslaps.inference.result import CaseResult, ForecastReference, RoleStatus
from hwoslaps.inference.settings import FitSpec, MassSupport, RefineSettings, SamplerSettings

pytestmark = pytest.mark.backend
POWER_LAW = {"kind": "power_law", "c0": 20.0, "mass_pivot_msun": 1.0e8,
             "mass_slope": -0.1, "redshift_slope": -1.0}


@pytest.mark.parametrize("defect", ["config", "truth_kernel", "injection", "redshift", "support", "noisy_anchor",
                                  "identity_kernel", "covariance", "model_sis", "model_concentration", "refine_cpu"])
def test_prepare_case_refuses_incoherent_inputs_before_any_backend_object(defect, prepared_forecast_factory,
                                                                         tiny_gaussian_kernel, tmp_path, monkeypatch):
    from hwoslaps.inference import api
    from hwoslaps.optics.kernels import DetectorPSF, KernelBinding
    from hwoslaps.scene.halos import HaloModel, FixedConcentration

    prepared = prepared_forecast_factory()
    trial = prepared.hypothesis(1.0e8, (0.4, -0.6))
    observation, fit = prepared.observation, FitSpec(mode="fixed_template")
    messages = {"config": "configuration digests differ", "truth_kernel": "truth kernel binding differs",
                "injection": "injects another hypothesis", "redshift": "configured hypothesis redshift",
                "support": "outside fit support", "noisy_anchor": "requires an expected observation",
                "identity_kernel": "1x1 model kernel", "covariance": "dense forecast noise covariance",
                "model_sis": "trial halo model recipe differs", "model_concentration": "trial halo model recipe differs"}
    if defect == "config":
        observation = dataclasses.replace(observation, config_digest="ff" * 32)
    elif defect == "truth_kernel":
        shifted = np.roll(tiny_gaussian_kernel, 1, axis=0)
        kernel = DetectorPSF.from_array(shifted, 0.05, normalize=True)
        observation = dataclasses.replace(observation, psfs=KernelBinding.uniform(kernel, ["source"]))
    elif defect == "injection":
        observation = dataclasses.replace(observation, subhalo=prepared.hypothesis(2.0e8, (0.4, -0.6)))
    elif defect == "redshift":
        trial = dataclasses.replace(trial, redshift=0.3)
    elif defect == "support":
        trial = dataclasses.replace(trial, position_yx_arcsec=(1.0, 0.0))
    elif defect == "noisy_anchor":
        observation, fit = observation.draw(11), FitSpec(mode="fixed_template", h1="truth_anchor")
    elif defect == "identity_kernel":
        path = tmp_path / "identity.npy"
        np.save(path, [[1.0]])
        prepared = prepared_forecast_factory({"psf": {"model": {"kind": "kernel", "path": str(path),
                                                               "pixel_scale_arcsec": 0.05}}})
        observation = prepared.observation
    elif defect == "model_sis":
        trial = dataclasses.replace(trial, model=HaloModel("SIS", None, None))
    elif defect == "model_concentration":
        trial = dataclasses.replace(trial, model=HaloModel("NFW", FixedConcentration(5.0), None))
    elif defect == "covariance":
        path = tmp_path / "covariance.npy"
        small = prepared_forecast_factory({"scene": {"grid": {"shape": [15, 15]}}})
        np.save(path, np.diag(small.sigma_adu.reshape(-1) ** 2))
        prepared = prepared_forecast_factory({"scene": {"grid": {"shape": [15, 15]}},
                                              "forecast": {"noise_covariance": str(path)}})
        observation = prepared.observation
        trial = prepared.hypothesis(1.0e8, (0.0, 0.0))
    built = []
    original = api.build_fit_data

    def recording_build(*args, **kwargs):
        built.append(True)
        if defect == "refine_cpu":
            pytest.fail("CPU refinement reached fit-data construction before its public entry refusal")
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "build_fit_data", recording_build)
    if defect == "refine_cpu":
        output = tmp_path / "refine_cpu"
        with pytest.raises(ValueError, match=r"refinement requires sampler\.use_jax"):
            validate_nonlinear(prepared, trial, observation, fit=fit, sampler=SamplerSettings(use_jax=False),
                               sampler_seed=7, refine=RefineSettings(), output_dir=output)
        assert not output.exists()
    else:
        with pytest.raises(ValueError, match=messages[defect]):
            prepare_case(prepared, trial, observation, fit=fit, use_jax=False)
    assert built == []


def test_default_mask_is_all_pixels_minus_the_psf_border(prepared_forecast_factory):
    prepared = prepared_forecast_factory({"forecast": {"mask": {"kind": "annulus", "inner_arcsec": 0.3,
                                                                "outer_arcsec": 0.8}}})
    trial = prepared.hypothesis(1.0e8, (0.4, -0.6))
    default = prepare_case(prepared, trial, prepared.observation, fit=FitSpec(mode="fixed_template"), use_jax=False)
    selected = prepare_case(prepared, trial, prepared.observation,
                            fit=FitSpec(mode="fixed_template", mask="forecast_mask_minus_psf_border"), use_jax=False)
    assert default.data.record["mask"]["name"] == "all_pixels_minus_psf_border"
    assert default.data.pixel_count == 1156
    y, x = np.mgrid[:40, :40].astype(float)
    y, x = (19.5 - y) * 0.05, (x - 19.5) * 0.05
    radial = np.hypot(y, x)
    expected = (radial >= 0.3) & (radial <= 0.8)
    expected[:3] = expected[-3:] = False
    expected[:, :3] = expected[:, -3:] = False
    np.testing.assert_array_equal(selected.data.mask, expected)
    assert selected.data.pixel_count < default.data.pixel_count


@pytest.mark.parametrize("defect", ["directory", "reference_mass", "reference_position"])
def test_case_output_and_reference_node_are_checked(defect, prepared_forecast, tmp_path):
    from hwoslaps.fisher.api import forecast

    trial = prepared_forecast.hypothesis(1.0e8, (0.4, -0.6))
    reference = ForecastReference.from_result(forecast(prepared_forecast, masses_msun=[1.0e8],
                                                     positions=[[0.4, -0.6]]), mass_index=0, position_index=0)
    if defect == "directory":
        directory = tmp_path / "outputs" / "case"
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_text("keep")
        message = "must be absent or empty"
    else:
        reference = dataclasses.replace(reference, **({"mass_msun": 2.0e8} if defect == "reference_mass"
                                                       else {"position_yx_arcsec": (0.3, -0.6)}))
        message = "differs from trial node"
    with pytest.raises(ValueError, match=message):
        validate_nonlinear(prepared_forecast, trial, prepared_forecast.observation, fit=FitSpec(mode="fixed_template"),
                           sampler=SamplerSettings(), sampler_seed=7, output_dir=tmp_path / "outputs",
                           case_id="case", forecast_reference=reference)
    if defect == "directory":
        assert (directory / "keep.txt").read_text() == "keep"
    else:
        assert not (tmp_path / "outputs").exists()


def test_expected_fixed_template_case_with_truth_anchor(prepared_forecast, tmp_path, monkeypatch):
    from hwoslaps.simulation import simulate

    trial = prepared_forecast.hypothesis(1.0e8, (0.4, -0.6))
    observation = simulate(prepared_forecast, subhalo=trial, noise_seed=None)
    outside = tmp_path / "workers"
    outside.mkdir()
    monkeypatch.chdir(outside)
    result = validate_nonlinear(prepared_forecast, trial, observation,
                                fit=FitSpec(mode="fixed_template", h1="truth_anchor"),
                                sampler=SamplerSettings(use_jax=True, n_live_smooth=20, n_eff=200.0,
                                                        n_like_max=2000, jax_n_batch=100),
                                sampler_seed=7, refine=RefineSettings(maxiter=50, repeat_maxiter=50),
                                output_dir=tmp_path / "results")
    case = prepare_case(prepared_forecast, trial, observation, fit=result.fit, use_jax=False)
    normalization = float(case.fit_at("subhalo", case.truth_vector("subhalo")).noise_normalization)
    assert result.subhalo.acceptance_status is RoleStatus.ZERO_RESIDUAL_ANCHOR
    assert result.subhalo.anchor_chi2 <= 1e-8
    assert result.subhalo.log_likelihood == pytest.approx(-0.5 * normalization, abs=1e-8)
    assert result.smooth.refinement is not None and result.smooth.refinement.best_vector is not None
    assert result.q_signed == 2.0 * (result.subhalo.log_likelihood - result.smooth.refinement.best_log_likelihood)
    assert result.delta_log_evidence is None
    mapping = result.to_mapping()
    assert CaseResult.from_mapping(json.loads(json.dumps(mapping))).to_mapping() == mapping
    assert (tmp_path / "results" / result.case_id / result.smooth.sampler.output_path / "files").is_dir()


def test_freed_noisy_case_on_two_cores_reports_recovery(prepared_forecast, tmp_path):
    from hwoslaps.simulation import simulate

    trial = prepared_forecast.hypothesis(1.0e8, (0.4, -0.6))
    result = validate_nonlinear(prepared_forecast, trial, simulate(prepared_forecast, subhalo=trial, noise_seed=11),
                                fit=FitSpec(mode="freed", mass_support=MassSupport(6.0, 9.7)),
                                sampler=SamplerSettings(number_of_cores=2, n_live_smooth=10, n_live_subhalo_search=10,
                                                        n_like_max=80),
                                sampler_seed=7, output_dir=tmp_path)
    assert result.smooth.status == result.subhalo.status == "success"
    assert result.recovery is not None and result.recovery.refined is None
    assert 6.0 <= result.recovery.sampler.log10_mass <= 9.7
    assert result.recovery.sampler.margin_to_lower_dex == result.recovery.sampler.log10_mass - 6.0
    if result.recovery.log10_mass_quantiles is not None:
        assert list(result.recovery.log10_mass_quantiles) == sorted(result.recovery.log10_mass_quantiles)


def test_freed_case_with_refinement_reports_sampler_and_refined_estimates(prepared_forecast, tmp_path):
    from hwoslaps.inference.hypotheses import build_role_models
    from hwoslaps.simulation import simulate

    trial = prepared_forecast.hypothesis(1.0e8, (0.4, -0.6))
    observation = simulate(prepared_forecast, subhalo=trial, noise_seed=11)
    result = validate_nonlinear(prepared_forecast, trial, observation,
                                fit=FitSpec(mode="freed", mass_support=MassSupport(6.0, 9.7)),
                                sampler=SamplerSettings(use_jax=True, n_live_smooth=20, n_live_subhalo_search=20, n_eff=200,
                                                        n_like_max=2000, jax_n_batch=100), sampler_seed=7,
                                refine=RefineSettings(maxiter=50, repeat_maxiter=50), output_dir=tmp_path)
    assert result.recovery is not None, (result.smooth.error, result.subhalo.error)
    assert result.recovery.sampler != result.recovery.refined
    names = result.subhalo.parameter_names
    vector = result.subhalo.refinement.best_vector
    mass = vector[names.index("galaxies.lens.subhalo.log10_m200")]
    centre = tuple(vector[names.index(f"galaxies.lens.subhalo.centre.centre_{index}")] for index in (0, 1))
    assert result.recovery.refined.log10_mass == mass and result.recovery.refined.centre_yx == centre
    free = [parameter.name for parameter in prepared_forecast.nuisances.parameters if parameter.kind == "scene"]
    mapping = build_role_models(prepared_forecast.scene, trial, free, result.fit, use_jax=True).mass_mapping
    assert result.recovery.refined.profile_scales == mapping.profile_scales(mass)
    assert result.q_signed == 2.0 * (result.subhalo.refinement.best_log_likelihood
                                   - result.smooth.refinement.best_log_likelihood)
    # The sampler estimate is tied to the retained AutoFit summary rather than the refined vector.
    summary = json.loads((tmp_path / result.case_id / result.subhalo.sampler.output_path / "files/samples_summary.json").read_text())
    saved = summary["arguments"]["max_log_likelihood_sample"]["arguments"]["kwargs"]["arguments"]
    assert result.recovery.sampler.log10_mass == saved["galaxies.lens.subhalo.log10_m200"]


def test_supplied_session_is_borrowed_and_left_active(prepared_forecast, tmp_path):
    from hwoslaps.inference.backend import BackendSession

    trial = prepared_forecast.hypothesis(1.0e8, (0.4, -0.6))
    fit = FitSpec(mode="fixed_template", h1="truth_anchor")
    # Null expected data has no truth subhalo, so use a zero-residual injected observation.
    from hwoslaps.simulation import simulate
    observation = simulate(prepared_forecast, subhalo=trial, noise_seed=None)
    sampler = SamplerSettings(use_jax=True, n_live_smooth=10, n_like_max=80, jax_n_batch=100)
    with pytest.raises(RuntimeError, match="must be active"):
        validate_nonlinear(prepared_forecast, trial, observation, fit=fit, sampler=sampler, sampler_seed=7,
                           session=BackendSession(), output_dir=tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()
    with BackendSession(training_workers=2) as session:
        children = {child.pid for child in multiprocessing.active_children()}
        validate_nonlinear(prepared_forecast, trial, observation, fit=fit, sampler=sampler, sampler_seed=7,
                           session=session, output_dir=tmp_path / "outputs", case_id="first")
        assert session.active and {child.pid for child in multiprocessing.active_children()} == children
        with pytest.raises(ValueError, match="separated samples.*refinement needs"):
            validate_nonlinear(prepared_forecast, trial, observation, fit=fit, sampler=dataclasses.replace(sampler, n_live_smooth=20, n_eff=200, n_like_max=2000), sampler_seed=8,
                               refine=RefineSettings(original_start_count=1000000),
                               session=session, output_dir=tmp_path / "outputs", case_id="second")
        assert session.active and {child.pid for child in multiprocessing.active_children()} == children


@pytest.mark.xtx_gpu
@pytest.mark.parametrize(("redshift", "mode"), [(z, mode) for z in (0.15, 0.4)
                                               for mode in ("fixed_template", "freed")])
def test_off_plane_hypothesis_jax_likelihood_and_gradient(redshift, mode, prepared_forecast_factory):
    from hwoslaps.simulation import simulate

    prepared = prepared_forecast_factory({"scene": {"subhalo": {"redshift": redshift, "concentration": POWER_LAW}}})
    trial = prepared.hypothesis(1.0e8, (0.4, -0.6))
    observation = simulate(prepared, subhalo=trial, noise_seed=None)
    fit = FitSpec(mode=mode, mass_support=MassSupport(6.0, 9.7) if mode == "freed" else None)
    jax_case = prepare_case(prepared, trial, observation, fit=fit, use_jax=True)
    numpy_case = prepare_case(prepared, trial, observation, fit=fit, use_jax=False)
    model = jax_case.model("subhalo")
    vectors = [model.truth, *(model.lower + np.random.default_rng(20261005).random((2, len(model.parameter_names)))
                              * (model.upper - model.lower))]
    for vector in vectors:
        value, gradient = jax_case.objective("subhalo").value_and_gradient((vector - model.lower) / (model.upper - model.lower))
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        assert value == pytest.approx(0.5 * numpy_case.chi_squared("subhalo", vector), rel=1e-10, abs=1e-10)


def test_round_exponential_gradient_matches_numpy_objective_differences(prepared_forecast_factory):
    from hwoslaps.simulation import simulate

    prepared = prepared_forecast_factory({"scene": {"source": {"light": {"light": {"ell_comps": [0.0, 0.0]}}}}})
    trial = prepared.hypothesis(1e8, (0.4, -0.6))
    observation = simulate(prepared, subhalo=trial, noise_seed=None)
    fit = FitSpec(mode="fixed_template")
    case = prepare_case(prepared, trial, observation, fit=fit, use_jax=True)
    numpy_case = prepare_case(prepared, trial, observation, fit=fit, use_jax=False)
    for role in ("smooth", "subhalo"):
        model = case.model(role)
        point = (model.truth - model.lower) / (model.upper - model.lower)
        if role == "subhalo":
            point[model.parameter_names.index("galaxies.source.light.intensity")] += 0.2
        value, gradient = case.objective(role).value_and_gradient(point)
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        for index, name in enumerate(model.parameter_names):
            if "source.light.ell_comps" not in name:
                continue
            plus, minus = point.copy(), point.copy()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            oracle = (numpy_case.chi_squared(role, model.lower + plus * (model.upper - model.lower))
                      - numpy_case.chi_squared(role, model.lower + minus * (model.upper - model.lower))) / 4e-4
            assert abs(oracle) > 0.1
            assert gradient[index] == pytest.approx(oracle, rel=2e-6, abs=1e-6)
    assert case.models.smooth.galaxies[-1].components[0][1].profile_class == "hwoslaps.inference.light_profiles:Exponential"
    elliptical = prepared_forecast_factory()
    original = prepare_case(elliptical, elliptical.hypothesis(1e8, (0.4, -0.6)), elliptical.observation,
                            fit=fit, use_jax=True)
    assert original.models.smooth.galaxies[-1].components[0][1].profile_class == "autolens:lp.Exponential"


def test_coincident_light_cusp_is_recorded_by_refinement(prepared_forecast_factory):
    from hwoslaps.inference.refine import refine
    from hwoslaps.inference.starts import RefineStart

    prepared = prepared_forecast_factory({"scene": {"grid": {"shape": [41, 41], "over_sample_size": 1},
        "lens": {"light": {"bulge": {"type": "Exponential", "centre": [0.0, 0.0], "ell_comps": [0.0, 0.0],
                                      "intensity": 0.2, "effective_radius": 0.4}}}}})
    case = prepare_case(prepared, prepared.hypothesis(1e8, (0.4, -0.6)), prepared.observation,
                        fit=FitSpec(mode="fixed_template"), use_jax=True)
    model = case.model("smooth")
    likelihood = case.log_likelihood("smooth", model.truth)
    assert np.isfinite(likelihood)
    starts = tuple(RefineStart.from_physical(index=index, physical=model.truth, lower=model.lower, upper=model.upper,
                   source="sampler_ml" if index == 0 else "sampler_sample", original=index != 0,
                   origin={"saved_log_likelihood": likelihood} if index == 0 else {}) for index in range(3))
    outcome = refine(starts, case.objective("smooth"), RefineSettings(original_start_count=2))
    assert outcome.acceptance_status is RoleStatus.UNRESOLVED
    assert outcome.best_log_likelihood is None
    assert all("Exponential gradient is undefined at an exactly zero-radius" in row["message"]
               for row in outcome.to_mapping()["record"]["runs"])


def test_custom_mask_case_records_round_trip(prepared_forecast, tmp_path):
    from hwoslaps.inference.settings import PixelMask
    from hwoslaps.simulation import simulate

    mask = np.ones((40, 40), dtype=bool)
    mask[12:15, 17:23] = False
    trial = prepared_forecast.hypothesis(1e8, (0.4, -0.6))
    result = validate_nonlinear(prepared_forecast, trial, simulate(prepared_forecast, subhalo=trial, noise_seed=None),
                                fit=FitSpec(mode="fixed_template", mask=PixelMask(mask), h1="truth_anchor"),
                                sampler=SamplerSettings(n_live_smooth=20, n_like_max=80),
                                sampler_seed=7, output_dir=tmp_path)
    record = json.loads(json.dumps(result.to_mapping()))
    restored = CaseResult.from_mapping(record)
    np.testing.assert_array_equal(restored.fit.mask.values, mask)
    assert restored.to_mapping() == record
    assert restored.fit.mask.digest == result.fit.mask.digest
    assert not restored.fit.mask.values.flags.writeable
    # A valid mask codec for another detector shape is still an invalid case record.
    record["data"]["shape"] = [41, 40]
    with pytest.raises(ValueError, match="custom mask shape differs from the recorded case data shape"):
        CaseResult.from_mapping(record)


@pytest.mark.parametrize("fixed_shape", [[], ["lens.mass.mass.ell_comp_1"], ["lens.mass.mass.ell_comp_1", "lens.mass.mass.ell_comp_2"]],
                         ids=["both-free", "one-free", "both-fixed"])
def test_free_circular_isothermal_refuses_only_the_origin_gradient(fixed_shape, prepared_forecast_factory):
    prepared = prepared_forecast_factory({"scene": {"lens": {"mass": {"mass": {"ell_comps": [0.0, 0.0]}}}},
                                           "forecast": {"nuisances": {"fixed": fixed_shape}}})
    case = prepare_case(prepared, prepared.hypothesis(1e8, (0.4, -0.6)), prepared.observation,
                        fit=FitSpec(mode="fixed_template"), use_jax=True)
    for role in ("smooth", "subhalo"):
        model = case.model(role)
        point = (model.truth - model.lower) / (model.upper - model.lower)
        objective = case.objective(role)
        assert np.isfinite(case.log_likelihood(role, model.truth))
        assert np.all(np.isfinite(objective.residual(point)))
        if len(fixed_shape) == 2:
            value, gradient = objective.value_and_gradient(point)
            assert np.isfinite(value) and np.all(np.isfinite(gradient))
            continue
        with pytest.raises(ValueError, match="Isothermal gradient is undefined at exactly ell_comps"):
            objective.value_and_gradient(point)
        free_shape = next(index for index, name in enumerate(model.parameter_names) if "lens.mass.ell_comps" in name)
        for physical_shape in (1e-8, 0.02):
            permitted = point.copy()
            permitted[free_shape] = (physical_shape - model.lower[free_shape]) / (model.upper[free_shape] - model.lower[free_shape])
            value, gradient = objective.value_and_gradient(permitted)
            assert np.isfinite(value) and np.all(np.isfinite(gradient))
