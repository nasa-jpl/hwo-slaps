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
