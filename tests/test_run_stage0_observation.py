"""Contracts for the Stage 0 observation runner's pure helpers.

The rendering path itself is exercised by the campaign smoke rather than
by a unit test. What is pinned here is the numerical guard that stands
between the PSF convolution and the pre-registered expected-variance
map, because that guard is the only place the runner is allowed to
change a pixel value, and the three fail-closed identity checks that
stand between a staged configuration and a rendered artifact: the
template asset bytes, the source revision, and the frozen theta_E
extraction settings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT/"scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_stage0_observation as runner  # noqa: E402
from run_stage0_observation import (  # noqa: E402
    ARTIFACT_NAME,
    CONVOLUTION_ROUNDOFF_TOLERANCE,
    _clip_convolution_roundoff,
    _verify_code_revision,
    _verify_extraction_settings,
    _verify_source_asset,
)


def test_artifact_name_matches_the_frozen_declaration():
    """The runner writes the artifact the design freeze declares."""
    from hwoslaps.campaign.design_freeze import load_design_freeze

    freeze = load_design_freeze(
        PROJECT_ROOT/"configs"/"design"/"design_freeze_v1.yaml"
    )
    assert freeze["stage0"]["artifact"] == ARTIFACT_NAME
    assert freeze["stage0"]["runner"].endswith("run_stage0_observation.py")


def test_a_non_negative_map_passes_through_unchanged():
    """Nothing is touched when the convolution left no negatives."""
    values = np.array([[0.0, 1.0], [2.5, 4.0]])
    clipped, minimum = _clip_convolution_roundoff(values)
    assert np.array_equal(clipped, values)
    assert minimum == 0.0


def test_round_off_negatives_are_clipped_and_reported():
    """A round-off excursion is clipped to zero and its size recorded."""
    values = np.array([[-2.8e-15, 1.0], [100.0, 0.0]])
    clipped, minimum = _clip_convolution_roundoff(values)
    assert minimum == pytest.approx(-2.8e-15)
    assert clipped.min() == 0.0
    assert clipped[0, 1] == 1.0
    assert clipped[1, 0] == 100.0


def test_a_real_negative_excursion_fails_closed():
    """A negative far beyond round-off is a fault, not a rounding artifact."""
    peak = 100.0
    values = np.array([[-0.5*peak, 1.0], [peak, 0.0]])
    with pytest.raises(ValueError, match="not round-off"):
        _clip_convolution_roundoff(values)


def test_the_tolerance_is_a_declared_fraction_of_the_peak():
    """The guard scales with the map, and sits at the round-off level."""
    assert CONVOLUTION_ROUNDOFF_TOLERANCE == 1.0e-9
    peak = 1.0e6
    just_inside = np.array([[-0.5*CONVOLUTION_ROUNDOFF_TOLERANCE*peak, peak]])
    just_outside = np.array([[-2.0*CONVOLUTION_ROUNDOFF_TOLERANCE*peak, peak]])
    assert _clip_convolution_roundoff(just_inside)[0].min() == 0.0
    with pytest.raises(ValueError):
        _clip_convolution_roundoff(just_outside)


# ---------------------------------------------------------------------------
# Identity checks between the staged configuration and the render
# ---------------------------------------------------------------------------


def _asset_config(path, digest, declared_path=None):
    """Build the fragment of a staged config the asset check reads."""
    return {
        "lensing": {
            "source_galaxy": {
                "light": {"asset_path": str(declared_path or path)},
            },
        },
        "stage0": {
            "source_asset_path": str(declared_path or path),
            "source_asset_sha256": digest,
        },
    }


def _written_asset(tmp_path):
    """Write one stand-in asset file and return its path and digest."""
    path = tmp_path/"template.npz"
    path.write_bytes(b"prepared source asset bytes")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_matching_asset_digest_passes(tmp_path):
    """The declared digest of the bytes on disk is returned unchanged."""
    path, digest = _written_asset(tmp_path)
    assert _verify_source_asset(_asset_config(path, digest)) == digest


def test_changed_asset_bytes_fail_closed(tmp_path):
    """A template whose bytes moved under the design refuses to render."""
    path, digest = _written_asset(tmp_path)
    config = _asset_config(path, digest)
    path.write_bytes(b"prepared source asset bytes, edited")
    with pytest.raises(ValueError, match="asset bytes moved"):
        _verify_source_asset(config)


def test_a_missing_asset_fails_closed(tmp_path):
    """A declared template that is not on disk refuses to render."""
    path, digest = _written_asset(tmp_path)
    config = _asset_config(path, digest)
    path.unlink()
    with pytest.raises(ValueError, match="does not exist"):
        _verify_source_asset(config)


def test_a_swapped_asset_path_fails_closed(tmp_path):
    """The rendered asset and the bound asset must be the same file."""
    path, digest = _written_asset(tmp_path)
    other = tmp_path/"other.npz"
    other.write_bytes(b"prepared source asset bytes")
    config = _asset_config(path, digest)
    config["lensing"]["source_galaxy"]["light"]["asset_path"] = str(other)
    with pytest.raises(ValueError, match="but the campaign declares"):
        _verify_source_asset(config)


def _revision_config(declared):
    """Build the fragment of a staged config the revision check reads."""
    return {"stage0": {"code_revision": declared}}


def test_the_generating_revision_is_accepted():
    """A job rendered at the revision it was generated at is accepted."""
    from hwoslaps.provenance import revision_digest, revision_provenance

    revision = revision_provenance()
    declared = {
        "git_hash": revision["git_hash"],
        "git_dirty": revision["git_dirty"],
        "sha256": revision_digest(revision),
    }
    verified = _verify_code_revision(_revision_config(declared))
    assert verified["sha256"] == declared["sha256"]
    assert verified["git_hash"] == revision["git_hash"]


def test_a_different_source_revision_fails_closed():
    """A resume under moved code refuses to render rather than mix states."""
    declared = {"git_hash": "a"*40, "git_dirty": False, "sha256": "b"*64}
    with pytest.raises(ValueError, match="check out the recorded revision"):
        _verify_code_revision(_revision_config(declared))


def _extraction(pixel_scale=0.01, half_width=4.0, factor=2.0, margin=0.1):
    """Build one extraction result carrying declared settings."""
    from hwoslaps.lensing import critical_curve as cc

    return cc.ThetaEExtraction(
        contour_arcsec=np.zeros((4, 2)),
        area_arcsec2=float(np.pi),
        theta_e_eff_arcsec=1.0,
        aperture=cc.ApertureDefinition(
            centre_arcsec=(0.0, 0.0),
            theta_e_eff_arcsec=1.0,
            theta_e_factor=factor,
            computational_margin_fraction=margin,
        ),
        grid=cc.CriticalCurveGrid(
            requested_half_width_arcsec=half_width,
            pixel_scale_arcsec=pixel_scale,
        ),
        lens_centre_arcsec=(0.0, 0.0),
        curve_counts={"extracted": 1, "closed": 1, "enclosing": 1},
    )


def _frozen_settings():
    """Return the staged extraction settings the committed freeze declares."""
    return {
        "algorithm_id": "tangential_critical_curve_marching_squares_v1",
        "choice_rule_id": "largest_area_closed_curve_enclosing_lens_centre",
        "extraction_grid": {
            "pixel_scale_arcsec": 0.01,
            "half_width_factor": 4.0,
        },
        "theta_e_factor": 2.0,
        "computational_margin_fraction": 0.1,
        "guards": {
            "closure_tolerance_pixels": 0.5,
            "border_margin_pixels": 2.0,
            "min_contour_vertices": 32,
        },
    }


def test_an_extraction_that_honours_the_settings_passes():
    """The settings the campaign froze are the settings that were used."""
    _verify_extraction_settings(_extraction(), _frozen_settings(), 1.0)


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"pixel_scale": 0.005}, "grid pixel scale"),
        ({"half_width": 6.0}, "grid requested half width"),
        ({"factor": 1.5}, "aperture theta_E factor"),
        ({"margin": 0.2}, "computational margin fraction"),
    ],
)
def test_a_runner_side_settings_mismatch_is_detected(changes, expected):
    """An extraction that left the frozen settings fails closed."""
    with pytest.raises(ValueError, match=expected):
        _verify_extraction_settings(
            _extraction(**changes), _frozen_settings(), 1.0
        )


def test_the_frozen_settings_match_the_committed_freeze():
    """The settings pinned in this test are the ones the freeze declares."""
    from hwoslaps.campaign.design_freeze import load_design_freeze

    freeze = load_design_freeze(
        PROJECT_ROOT/"configs"/"design"/"design_freeze_v1.yaml"
    )
    algorithm = freeze["aperture"]["theta_e_algorithm"]
    settings = _frozen_settings()
    assert settings["algorithm_id"] == algorithm["algorithm_id"]
    assert settings["choice_rule_id"] == algorithm["choice_rule_id"]
    for key, value in settings["extraction_grid"].items():
        assert value == algorithm["extraction_grid"][key], key
    for key, value in settings["guards"].items():
        assert value == algorithm["guards"][key], key
    assert settings["theta_e_factor"] == freeze["aperture"]["theta_e_factor"]
    assert settings["computational_margin_fraction"] == freeze["aperture"][
        "computational_margin_fraction"
    ]


def test_a_contour_that_leaves_the_generator_curve_fails_closed():
    """A re-extraction that no longer reproduces the bound curve refuses.

    The declared ``theta_E_eff`` is within tolerance, so only the exact
    contour digest separates this job from an accepted one: that is the
    guard against an engine or dependency drift too small to move the
    scalar but large enough to move the curve.
    """
    pytest.importorskip("autolens")

    settings = _frozen_settings()
    config = {
        "lensing": {
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.0, 0.0],
                    "einstein_radius": 1.0,
                },
            },
        },
        "stage0": {
            "system_id": "sys0000",
            "theta_e_extraction": settings,
            "theta_e_eff_arcsec": 1.0,
            "theta_e_eff_tolerance_fractional": 0.02,
            "theta_e_contour_sha256": "0"*64,
            "theta_e_aperture_sha256": "1"*64,
        },
    }
    with pytest.raises(ValueError, match="no longer reproduces"):
        runner._extract_theta_e_eff(config)


def test_an_algorithm_the_checkout_does_not_implement_fails_closed():
    """A campaign frozen on another extraction algorithm refuses to render."""
    settings = _frozen_settings()
    settings["algorithm_id"] = "tangential_critical_curve_marching_squares_v0"
    config = {
        "lensing": {"lens_galaxy": {"mass": {"einstein_radius": 1.0}}},
        "stage0": {
            "system_id": "sys0000",
            "theta_e_extraction": settings,
            "theta_e_eff_arcsec": 1.0,
            "theta_e_eff_tolerance_fractional": 0.02,
        },
    }
    with pytest.raises(ValueError, match="this checkout implements"):
        runner._extract_theta_e_eff(config)
