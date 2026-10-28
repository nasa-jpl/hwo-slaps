"""Independent sampling-completion controls and artifact quality contracts."""

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from hwoslaps.inference.result import SamplerDiagnostics, SamplerRecord, RoleFit, RoleStatus
from hwoslaps.inference.sampler import stopping_diagnostics


def _record(diagnostics):
    return SamplerRecord("search", "search/id", "a" * 64, 100, {}, {}, 7, 1,
                         -1., -2., 100, None, .1, diagnostics)


def _role(diagnostics):
    return RoleFit("smooth", "search", "success", -1., -1., ("x", "y"), 2,
                   _record(diagnostics), None, {}, None, None, .1)


def test_quality_roundtrip_and_legacy_records_do_not_infer_completion():
    diagnostic = SamplerDiagnostics({"n_eff": 500}, {"n_eff": 100}, False, False,
                                    "budget_exhausted", True)
    role = _role(diagnostic)
    restored = RoleFit.from_mapping(json.loads(json.dumps(role.to_mapping(), allow_nan=False)))
    assert restored == role
    assert role.sampling_status == "incomplete"
    assert role.posterior_quality == role.evidence_quality == "provisional"
    assert role.pdf_usable is True
    assert role.acceptance_status is RoleStatus.SAMPLER_ONLY
    complete = replace(role, sampler=replace(role.sampler, diagnostics=replace(
        diagnostic, exploration_complete=True, criteria_satisfied=True, stop_reason="criteria_satisfied")))
    assert complete.posterior_quality == complete.evidence_quality == "complete"
    assert complete.acceptance_status is RoleStatus.SAMPLER_ONLY
    legacy = role.to_mapping()
    del legacy["sampler"]["diagnostics"]
    del legacy["strategy_record"]
    del legacy["quality"]
    restored = RoleFit.from_mapping(legacy)
    assert restored.sampling_status == "unknown" and restored.evidence_quality == "provisional"


@pytest.mark.backend
@pytest.mark.parametrize("capped", [True, False], ids=["budget_limited", "criterion_complete"])
def test_independent_gaussian_sampler_completion_and_evidence(capped):
    """A normalized bounded 2-D Gaussian has an independent analytic evidence.

    The capped run often has usable weights; that does not satisfy exploration
    or ESS. The completed control tests evidence and marginals separately from MLE.
    """
    from nautilus import Sampler

    sampler = Sampler(lambda unit: 10. * np.asarray(unit) - 5.,
                      lambda point: -.5 * np.sum(np.asarray(point) ** 2),
                      n_dim=2, n_live=100, seed=23, n_batch=100)
    settings = {"n_eff": 1000, "n_shell": 1, "f_live": .01,
                "n_like_max": 100 if capped else 20000, "discard_exploration": False}
    success = sampler.run(**settings, verbose=False)
    diagnostics = stopping_diagnostics(sampler, settings)
    assert diagnostics.criteria_satisfied == bool(success) == (not capped)
    assert diagnostics.stop_reason == ("budget_exhausted" if capped else "criteria_satisfied")
    assert diagnostics.achieved["n_like"] == sampler.n_like
    if capped:
        _, log_weights, _ = sampler.posterior()
        assert np.exp(log_weights).max() < .99
        assert diagnostics.exploration_complete is False
        assert diagnostics.achieved["n_eff"] < settings["n_eff"]
    else:
        expected = math.log(2. * math.pi * math.erf(5. / math.sqrt(2.)) ** 2 / 100.)
        assert sampler.log_z == pytest.approx(expected, abs=.15)
        samples, log_weights, _ = sampler.posterior()
        weights = np.exp(log_weights)
        means = np.sum(samples * weights[:, None], axis=0)
        variances = np.sum((samples - means) ** 2 * weights[:, None], axis=0)
        assert np.max(np.abs(means)) < .15
        assert np.max(np.abs(variances - 1.)) < .2


@pytest.mark.backend
def test_public_capped_fit_retains_provisional_values_and_roundtrips(prepared_forecast_factory, tmp_path):
    from hwoslaps.analysis.nonlinear import scientific_outputs
    from hwoslaps.artifacts import save_case, load_case
    from hwoslaps.inference.api import validate_nonlinear
    from hwoslaps.inference.settings import BoxRule, FitSpec, MassSupport, PriorWidths, SamplerSettings
    from hwoslaps.scene.parameters import scene_parameters
    from hwoslaps.simulation import simulate

    initial = prepared_forecast_factory()
    free = {"source.light.light.intensity", "source.light.light.effective_radius"}
    fixed = [parameter.name for parameter in scene_parameters(initial.scene.spec) if parameter.name not in free]
    prepared = prepared_forecast_factory({"forecast": {"nuisances": {"fixed": fixed, "background_offset": False}}})
    trial = prepared.hypothesis(1.e8, (.4, -.6))
    fit = FitSpec(mode="freed", mass_support=MassSupport(7.999, 8.001), prior_widths=PriorWidths(
        rules=(("source.amplitude", BoxRule(.0001)), ("source.size", BoxRule(.00001))),
        subhalo_freed_window_arcsec=.0001))
    result = validate_nonlinear(prepared, trial, simulate(prepared, subhalo=trial, noise_seed=None), fit=fit,
        sampler=SamplerSettings(n_live_smooth=10, n_live_subhalo_search=10, n_eff=500, n_like_max=100),
        sampler_seed=7, output_dir=tmp_path / "fit")
    assert result.recovery.pdf_usable is True
    assert result.recovery.log10_mass_quantiles is not None
    assert result.recovery.posterior_quality == result.evidence_quality == "provisional"
    for role in (result.smooth, result.subhalo):
        assert role.status == "success" and role.sampling_status == "incomplete"
        assert role.sampler.diagnostics.stop_reason == "budget_exhausted"
        assert role.sampler.diagnostics.exploration_complete is False
    path = save_case(result, tmp_path / "case.json")
    restored = load_case(path)
    assert restored.to_mapping() == result.to_mapping()
    assert scientific_outputs([restored]).evidence_differences == ()
    assert scientific_outputs([restored]).recoveries == ()
    explicit = scientific_outputs([restored], include_provisional=True)
    assert len(explicit.evidence_differences) == len(explicit.recoveries) == 1
