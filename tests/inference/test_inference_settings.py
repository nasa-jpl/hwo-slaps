"""Inference settings: strict mappings, coherent combinations, paper box arithmetic and defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.inference.settings import (
    DEFAULT_BOX_RULES, BoxRule, FitSpec, MassSupport, PriorWidths, RefineSettings, SamplerSettings,
)

N1_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity" / "n1_nonlinear_likelihood.npz"


@dataclass(frozen=True)
class Domain:
    """The four fields BoxRule.box reads from a registry parameter domain (scene.profiles.Interval)."""

    lower: float
    upper: float
    open_lower: bool = True
    open_upper: bool = True


UNBOUNDED = Domain(-math.inf, math.inf)
POSITIVE = Domain(0.0, math.inf)
ELLIPTICITY = Domain(-1.0, 1.0)


@pytest.mark.parametrize(("read", "mapping", "path"), [
    (RefineSettings.from_mapping, {"maxiterr": 3}, "refine.maxiterr"),
    (SamplerSettings.from_mapping, {"n_live_smoth": 50}, "sampler.n_live_smoth"),
    (MassSupport.from_mapping, {"log10_mass_min": 6.0, "log10_mass_max": 9.7, "log10_mass_mid": 8.0},
     "fit.mass_support.log10_mass_mid"),
    (PriorWidths.from_mapping, {"subhalo_window_arcsec": 0.03}, "fit.prior_widths.subhalo_window_arcsec"),
    (FitSpec.from_mapping, {"mode": "fixed_template", "maks": "all_pixels_minus_psf_border"}, "fit.maks"),
    (FitSpec.from_mapping, {"mode": "freed", "mass_support": {"log10_mass_min": 6.0, "log10_mass_maximum": 9.7}},
     "fit.mass_support.log10_mass_maximum"),
    (FitSpec.from_mapping, {"mode": "fixed_template", "prior_widths": {"rules": {"lens.einstien_radius": {}}}},
     "fit.prior_widths.rules.lens.einstien_radius"),
    (FitSpec.from_mapping, {"mode": "fixed_template", "prior_widths": {"rules": {"lens.position": {"halfwidth": 1}}}},
     "fit.prior_widths.rules.lens.position.halfwidth"),
], ids=["refine", "sampler", "mass-support", "prior-widths", "fit", "fit-mass-support", "rule-key", "rule-field"])
def test_settings_mappings_reject_unknown_keys(read, mapping, path):
    with pytest.raises(ConfigError) as caught:
        read(mapping)
    assert caught.value.path == path
    assert "unknown key" in caught.value.message


@pytest.mark.parametrize(("build", "path"), [
    (lambda: FitSpec(mode="freed"), "mass_support"),
    (lambda: FitSpec(mode="fixed_template", mass_support=MassSupport(6.0, 9.7)), "mass_support"),
    (lambda: FitSpec(mode="local_search", anchor_chi2_tolerance=0.0), "anchor_chi2_tolerance"),
    (lambda: MassSupport(9.0, 9.0), "log10_mass_max"),
    (lambda: SamplerSettings(use_jax=True, number_of_cores=2), "number_of_cores"),
    (lambda: BoxRule(0.02, clip=(0.9, -0.9)), "clip"),
    (lambda: FitSpec.from_mapping({"mode": "freed"}), "fit.mass_support"),
    (lambda: FitSpec.from_mapping({"mode": "freed", "mass_support": {"log10_mass_min": 9.7, "log10_mass_max": 6.0}}),
     "fit.mass_support.log10_mass_max"),
    (lambda: FitSpec.from_mapping({"mode": "fixed_template", "mask": [[True, False]]}), "fit.mask"),
    (lambda: SamplerSettings.from_mapping({"use_jax": True, "number_of_cores": 4}), "sampler.number_of_cores"),
], ids=["freed-without-support", "fixed-with-support", "zero-anchor-tolerance", "empty-support",
        "jax-with-cores", "reversed-clip", "mapping-freed-without-support",
        "mapping-reversed-support", "mapping-array-mask", "mapping-jax-with-cores"])
def test_settings_reject_incoherent_combinations(build, path):
    with pytest.raises(ConfigError) as caught:
        build()
    assert caught.value.path == path


@pytest.mark.parametrize(("cls", "field", "value"), [
    (SamplerSettings, "n_eff", True), (SamplerSettings, "n_eff", 0.0), (SamplerSettings, "n_eff", -5.0),
    (SamplerSettings, "n_eff", math.nan), (SamplerSettings, "n_eff", math.inf), (SamplerSettings, "n_shell", True),
    (SamplerSettings, "n_shell", 0), (SamplerSettings, "n_shell", 1.5), (SamplerSettings, "f_live", 0.0),
    (SamplerSettings, "f_live", 1.5), (SamplerSettings, "discard_exploration", 1),
    (SamplerSettings, "discard_exploration", "yes"), (SamplerSettings, "retain_search_internal", None),
    (SamplerSettings, "retain_search_internal", 1), (SamplerSettings, "jax_n_batch", 0),
    (SamplerSettings, "jax_n_batch", -1), (SamplerSettings, "n_like_max", 0), (SamplerSettings, "number_of_cores", 0),
    (SamplerSettings, "n_live_smooth", 0), (RefineSettings, "maxiter", 0), (RefineSettings, "ftol", -1.0e-3),
    (RefineSettings, "start_separation_posterior_sigma", 0.0), (RefineSettings, "scalar_residual_tolerance", 0.0),
    (RefineSettings, "original_start_count", 0),
], ids=lambda value: value.__name__ if isinstance(value, type) else str(value))
def test_settings_refuse_values_outside_their_domains(cls, field, value):
    """Booleans are not numbers, counts are positive integers, tolerances finite and in range."""
    with pytest.raises(ConfigError) as caught:
        cls(**{field: value})
    assert caught.value.path == field
    with pytest.raises(ConfigError) as caught:
        cls.from_mapping({field: value})
    assert caught.value.path.endswith(f".{field}")


def test_live_points_follow_the_role_and_mode():
    """H0 always uses n_live_smooth; H1 the fixed-template or the search count (I-8)."""
    settings = SamplerSettings(n_live_smooth=11, n_live_subhalo_fixed=22, n_live_subhalo_search=33)
    assert [settings.n_live("smooth", mode) for mode in ("fixed_template", "local_search", "freed")] == [11] * 3
    assert [settings.n_live("subhalo", mode) for mode in ("fixed_template", "local_search", "freed")] == [22, 33, 33]
    with pytest.raises(ValueError, match="role must be one of"):
        settings.n_live("lens", "freed")


def test_box_rules_reproduce_paper_bounds():
    """The N1 smooth boxes of the paper code, bitwise, plus the clip and domain branches."""
    n1 = np.load(N1_FIXTURE, allow_pickle=False)
    truth = n1["smooth_vectors"][0]
    widths = PriorWidths()
    rows = [("lens", "position", UNBOUNDED)] * 2 + [("lens", "einstein_radius", POSITIVE)] + \
        [("lens", "ellipticity", ELLIPTICITY)] * 2 + [("source", "position", UNBOUNDED)] * 2 + \
        [("source", "ellipticity", ELLIPTICITY)] * 2 + [("source", "amplitude", POSITIVE),
                                                        ("source", "size", POSITIVE)]
    assert list(n1["smooth_prior_paths"])[9:] == ["galaxies.source.light.intensity",
                                                  "galaxies.source.light.effective_radius"]
    boxes = [widths.rule(galaxy, kind).box(value, domain) for (galaxy, kind, domain), value in zip(rows, truth)]
    assert [lower for lower, _ in boxes] == n1["smooth_prior_lower"].tolist()
    assert [upper for _, upper in boxes] == n1["smooth_prior_upper"].tolist()

    assert BoxRule(0.3, fractional=True).box(0.11, POSITIVE) == (0.11 - 0.3 * 0.11, 0.11 + 0.3 * 0.11)
    assert BoxRule(0.02, clip=(-0.9, 0.9)).box(0.89, ELLIPTICITY) == (0.89 - 0.02, float(np.nextafter(0.9, -0.9)))
    assert BoxRule(0.02, clip=(-0.9, 0.9)).box(-0.89, ELLIPTICITY) == (float(np.nextafter(-0.9, 0.9)), -0.89 + 0.02)
    assert BoxRule(0.5).box(0.2, POSITIVE) == (float(np.nextafter(0.0, 1.0)), 0.7)
    assert BoxRule(0.5).box(0.8, Domain(0.0, 1.0, open_lower=False, open_upper=False)) == (0.8 - 0.5, 1.0)
    with pytest.raises(ConfigError, match="does not lie inside the box"):
        BoxRule(0.02, clip=(-0.9, 0.9)).box(0.95, ELLIPTICITY)
    with pytest.raises(ConfigError, match="zero width at truth 0.0"):
        BoxRule(0.5, fractional=True).box(0.0, UNBOUNDED)


def test_default_settings_equal_the_paper_procedure():
    """Literals of 41621de fresh_profile.py:62-76 and autolens_model_builder.py:22-34."""
    assert RefineSettings().to_mapping() == {
        "original_start_count": 8, "start_separation_normalized_l2": 0.05, "start_separation_posterior_sigma": 1.0,
        "maxiter": 500, "ftol": 0.0, "gtol": 1.0e-10, "maxls": 50, "repeat_maxiter": 1000, "repeat_ftol": 0.0,
        "repeat_gtol": 1.0e-12, "support_log_likelihood_tolerance": 0.1, "repeat_log_likelihood_tolerance": 0.1,
        "minimum_distinct_original_start_support": 2, "scalar_residual_tolerance": 1.0e-4}
    assert {name: rule.to_mapping() for name, rule in DEFAULT_BOX_RULES} == {
        "lens.position": {"half_width": 0.005, "fractional": False, "clip": None},
        "lens.einstein_radius": {"half_width": 0.01, "fractional": False, "clip": None},
        "lens.ellipticity": {"half_width": 0.02, "fractional": False, "clip": [-0.9, 0.9]},
        "lens.amplitude": {"half_width": 0.5, "fractional": True, "clip": None},
        "lens.size": {"half_width": 0.3, "fractional": True, "clip": None},
        "source.position": {"half_width": 0.01, "fractional": False, "clip": None},
        "source.ellipticity": {"half_width": 0.05, "fractional": False, "clip": [-0.9, 0.9]},
        "source.amplitude": {"half_width": 0.5, "fractional": True, "clip": None},
        "source.size": {"half_width": 0.3, "fractional": True, "clip": None}}
    widths = PriorWidths()
    assert (widths.subhalo_local_window_arcsec, widths.subhalo_freed_window_arcsec) == (0.03, 0.15)
    assert FitSpec(mode="fixed_template").to_mapping()["mask"] == "all_pixels_minus_psf_border"


def test_settings_round_trip_and_rule_overrides_merge_field_by_field():
    """A record reads back to an equal value; a rule entry overrides only the fields it names."""
    fit = FitSpec.from_mapping({"mode": "freed", "mass_support": {"log10_mass_min": 6, "log10_mass_max": 9.7},
                                "prior_widths": {"rules": {"lens.ellipticity": {"half_width": 0.05},
                                                           "lens.einstein_radius": {"half_width": 0.1,
                                                                                    "fractional": True}}}})
    assert fit.prior_widths.rule("lens", "ellipticity") == BoxRule(0.05, clip=(-0.9, 0.9))
    assert fit.prior_widths.rule("lens", "einstein_radius") == BoxRule(0.1, fractional=True)
    assert fit.mass_support == MassSupport(6.0, 9.7)
    assert fit.mass_support.contains(9.7) and not fit.mass_support.contains(9.700000000000001)
    assert FitSpec.from_mapping(fit.to_mapping()) == fit
    sampler = SamplerSettings(n_eff=200, n_shell=1, f_live=0.01, discard_exploration=False, use_jax=True,
                              jax_n_batch=50, retain_search_internal=True)
    assert SamplerSettings.from_mapping(sampler.to_mapping()) == sampler
    assert sampler.to_mapping()["n_eff"] == 200.0 and isinstance(sampler.to_mapping()["n_eff"], float)
    refine = RefineSettings(original_start_count=2, maxiter=50, start_separation_posterior_sigma=2.0)
    assert RefineSettings.from_mapping(refine.to_mapping()) == refine
