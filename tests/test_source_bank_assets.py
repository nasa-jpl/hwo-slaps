"""Canaries for the committed D-F3 five-template source bank.

Every anchor is pinned by content hash, checked against the prepared-asset
conventions of the Item 4 tool, and held to the Sol Pro P0-3 discrete
detected-rate contract: the unlensed render on the exact production grid
sums to the committed detected electron rate, and the lensed pipeline
applies magnification exactly once on top of that.
"""

from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from hwoslaps.lensing import generate_lensing_system, load_source_image_asset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_source_image import (  # noqa: E402
    OBSERVING_REFERENCE_RELPATH,
    PRODUCTION_SCENE_RELPATH,
    detected_rate_reference,
    production_render_config,
    render_unlensed_asset,
    solve_detected_rate_normalization,
    verify_asset_rate_contract,
)

ASSET_DIR = PROJECT_ROOT / "configs" / "source_assets"

TARGET_HALF_LIGHT_ARCSEC = 0.11
"""Half-light radius every bank anchor is matched to, in arcseconds."""

PRODUCTION_GRID_SHAPE = (500, 500)
PRODUCTION_PIXEL_SCALE_ARCSEC = 0.00716

QUALIFICATION_PROFILE_ANGULAR_INTEGRAL = 0.289151264
"""Internal preparation convention; never a detected electron rate."""

LEGACY_ANCHOR = {
    "ident": 48849,
    "morphology_class": "clumpy_s_bar",
    "filename": "cosmos_48849_hlr011.npz",
    "sha256": (
        "c67570fdc56068fc900e3428f6b6ff2d5ffce8f459858c302c6de1461d24ed21"
    ),
    "script_version": 1,
    "pixel_scale_arcsec": 0.002732128852944742,
    "total_flux": 0.00045889467368609476,
}
"""The anchor prepared before the contract solve existed.

Every bank anchor is held to the contract, so this entry is
parametrized alongside the four prepared with it.
"""

NEW_ANCHORS = (
    {
        "ident": 62410,
        "morphology_class": "smooth_disk",
        "filename": "cosmos_62410_hlr011.npz",
        "sha256": (
            "5b716e7bb61b36e385394be32faf3e64fbcfd9a7f543779b491ed8603ddc2b4d"
        ),
        "script_version": 2,
        "pixel_scale_arcsec": 0.005553115716269291,
        "total_flux": 0.0004589089094626745,
    },
    {
        "ident": 159916,
        "morphology_class": "clumpy",
        "filename": "cosmos_159916_hlr011.npz",
        "sha256": (
            "6aa84a2874aef1d30d1292f3ec29163f70f6dd78b155f45f43db044a31ae6222"
        ),
        "script_version": 2,
        "pixel_scale_arcsec": 0.005965587590013045,
        "total_flux": 0.000458924098862195,
    },
    {
        "ident": 162893,
        "morphology_class": "irregular_merger",
        "filename": "cosmos_162893_hlr011.npz",
        "sha256": (
            "76e61166e4f44d946a122131533a3520b135a5704938931e3f2397437616f18e"
        ),
        "script_version": 2,
        "pixel_scale_arcsec": 0.005584403908234906,
        "total_flux": 0.0004589439758914851,
    },
    {
        "ident": 127283,
        "morphology_class": "compact",
        "filename": "cosmos_127283_hlr011.npz",
        "sha256": (
            "686e17abd6643219231a56249eb86a67ab940164c4521da1f55d87e4255dee5b"
        ),
        "script_version": 2,
        "pixel_scale_arcsec": 0.005185449728701348,
        "total_flux": 0.00045889704537767584,
    },
)

BANK = (LEGACY_ANCHOR,) + NEW_ANCHORS

ANALYTIC_CANARY_ANCHOR = NEW_ANCHORS[0]
"""Bank anchor the analytic magnification canary is rendered from.

The canary needs a template whose whole lit footprint fits inside the
two-image regime of one circular lens, which the two widest templates do
not at a tractable grid size.
"""

ANALYTIC_CANARY_EINSTEIN_RADIUS_ARCSEC = 2.0
ANALYTIC_CANARY_SOURCE_CENTRE_ARCSEC = (0.0, 1.0)
ANALYTIC_CANARY_GRID_SHAPE = (400, 400)
ANALYTIC_CANARY_PIXEL_SCALE_ARCSEC = 0.02
"""Circular-lens canary geometry.

The Einstein radius and the source offset put the whole lit footprint of
the anchor inside the two-image regime, and the grid half width of 3.99
arcsec contains both images of every lit sample, so the canary compares
total fluxes rather than truncated ones.
"""

ANALYTIC_CANARY_TOLERANCE = 1.0e-2
"""Accepted departure from the analytic canary prediction.

The engine sums the source brightness at image-plane pixel centres while
the prediction sums it at asset sample centres, and the two quadratures of
the same integral part company at the few parts in a thousand for a
template sampled more finely than the canary grid. The canary is a guard
against a magnification factor of about four, not a convergence test.
"""


def _sha256(path):
    """Return the full SHA-256 hex digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_path(anchor):
    """Return the committed asset path of one bank anchor."""
    return ASSET_DIR / anchor["filename"]


def _ids(anchors):
    """Return readable parametrization identifiers."""
    return [f"{anchor['ident']}_{anchor['morphology_class']}" for anchor in anchors]


@pytest.fixture(scope="module")
def reference():
    """Return the committed observing reference record."""
    return detected_rate_reference(PROJECT_ROOT / OBSERVING_REFERENCE_RELPATH)


@pytest.fixture(scope="module")
def production_render(reference):
    """Return the production grid and Image-source geometry."""
    return production_render_config(
        PROJECT_ROOT / PRODUCTION_SCENE_RELPATH, reference["pixel_scale_arcsec"]
    )


@pytest.mark.parametrize("anchor", BANK, ids=_ids(BANK))
def test_bank_asset_content_hash_is_pinned(anchor):
    """Pin every bank anchor to its prepared content."""
    path = _asset_path(anchor)
    assert path.exists(), f"missing bank asset {path}"
    assert _sha256(path) == anchor["sha256"]


@pytest.mark.parametrize("anchor", BANK, ids=_ids(BANK))
def test_bank_asset_loads_with_prepared_conventions(anchor):
    """Load every anchor as a square unit-integral half-light asset."""
    asset = load_source_image_asset(_asset_path(anchor))

    assert asset.sb.ndim == 2
    assert asset.sb.shape[0] == asset.sb.shape[1]
    assert np.all(np.isfinite(asset.sb))
    assert np.all(asset.sb >= 0.0)
    assert asset.pixel_scale_arcsec > 0.0
    assert asset.pixel_scale_arcsec**2 * float(asset.sb.sum()) == pytest.approx(
        1.0, rel=1.0e-12
    )

    assert asset.metadata["format_version"] == 1
    provenance = asset.metadata["provenance"]
    assert provenance["target_half_light_arcsec"] == TARGET_HALF_LIGHT_ARCSEC
    assert provenance["catalog_id"] == f"COSMOS 23.5 ident {anchor['ident']}"
    assert provenance["script_version"] == anchor["script_version"]
    assert provenance["bin"] == 1
    assert provenance["flip_y"] is False
    assert provenance["pixel_scale_arcsec"] == pytest.approx(
        asset.pixel_scale_arcsec, rel=0.0, abs=0.0
    )


@pytest.mark.parametrize("anchor", BANK, ids=_ids(BANK))
def test_bank_asset_stores_the_detected_rate_contract(anchor, reference):
    """The stored contract is the committed detected rate, not a convention."""
    asset = load_source_image_asset(_asset_path(anchor))
    contract = asset.metadata["provenance"]["rate_contract"]

    target = contract["target_rate_e_per_s"]
    assert target == reference["target_rate_e_per_s"]
    assert target != QUALIFICATION_PROFILE_ANGULAR_INTEGRAL
    assert contract["units"].startswith("detected electrons per second")
    assert contract["realized_rate_e_per_s"] == pytest.approx(target, rel=1.0e-12)

    assert contract["total_flux"] == pytest.approx(anchor["total_flux"], rel=1.0e-12)
    assert tuple(contract["grid_shape"]) == PRODUCTION_GRID_SHAPE
    assert contract["pixel_scale_arcsec"] == PRODUCTION_PIXEL_SCALE_ARCSEC
    assert contract["pixel_scale_arcsec"] == reference["pixel_scale_arcsec"]
    assert contract["discrete_mapping_ratio"] == pytest.approx(1.0, abs=1.0e-3)
    assert asset.pixel_scale_arcsec == pytest.approx(
        anchor["pixel_scale_arcsec"], rel=1.0e-12
    )

    geometry = contract["render_geometry"]
    assert geometry["flux_scale"] == 1.0
    assert geometry["size_scale"] == 1.0


@pytest.mark.parametrize("anchor", BANK, ids=_ids(BANK))
def test_bank_asset_re_renders_to_its_target_rate(anchor):
    """Re-rendering the committed asset reproduces the stored contract."""
    contract = verify_asset_rate_contract(_asset_path(anchor))
    assert contract["realized_rate_e_per_s"] == pytest.approx(
        contract["target_rate_e_per_s"], rel=1.0e-12
    )


def test_legacy_anchor_contract_matches_the_committed_reference(
    reference, production_render
):
    """The 48849 solve reproduces the committed scene4 normalization."""
    grid_config, source_config = production_render
    contract = solve_detected_rate_normalization(
        _asset_path(LEGACY_ANCHOR),
        reference,
        grid_config,
        source_config,
        PROJECT_ROOT / PRODUCTION_SCENE_RELPATH,
    )
    document = yaml.safe_load(
        (PROJECT_ROOT / OBSERVING_REFERENCE_RELPATH).read_text(encoding="utf-8")
    )
    committed = document["source_normalization"]["scene4_cosmos"]["lensing"][
        "source_galaxy"
    ]["light"]["total_flux"]

    assert contract["total_flux"] == pytest.approx(committed, rel=1.0e-12)
    assert contract["realized_rate_e_per_s"] == pytest.approx(
        reference["target_rate_e_per_s"], rel=1.0e-12
    )


def _lensed_scene(anchor, total_flux):
    """Return the subhalo-free lensed system of one anchor at one flux."""
    scene = yaml.safe_load(
        (PROJECT_ROOT / PRODUCTION_SCENE_RELPATH).read_text(encoding="utf-8")
    )
    scene = deepcopy(scene)
    light = scene["lensing"]["source_galaxy"]["light"]
    light["asset_path"] = str(_asset_path(anchor))
    light["total_flux"] = float(total_flux)
    scene["lensing"]["subhalo"]["enabled"] = False
    return generate_lensing_system(scene["lensing"], scene)


@pytest.mark.parametrize("anchor", BANK, ids=_ids(BANK))
def test_lensed_render_applies_magnification_exactly_once(
    anchor, production_render
):
    """The lensed image is the contract source sampled at traced positions.

    Magnification is emergent: no factor multiplies the normalization, so
    the lensed render must equal the same surface brightness evaluated on
    the ray-traced source-plane grid, and the lensed total must exceed the
    unlensed contract rate only by that emergent magnification.
    """
    asset = load_source_image_asset(_asset_path(anchor))
    contract = asset.metadata["provenance"]["rate_contract"]
    total_flux = contract["total_flux"]

    grid_config, source_config = production_render
    unlensed = render_unlensed_asset(
        _asset_path(anchor), grid_config, source_config, total_flux
    )
    assert float(unlensed.sum()) == pytest.approx(
        contract["target_rate_e_per_s"], rel=1.0e-12
    )

    system = _lensed_scene(anchor, total_flux)
    lensed = np.asarray(system.image, dtype=float)
    traced = system.tracer.traced_grid_2d_list_from(grid=system.grid)[-1]
    source_light = system.tracer.galaxies[-1].light
    manual = np.asarray(source_light.image_2d_from(grid=traced).native, dtype=float)

    np.testing.assert_array_equal(lensed, manual)
    magnification = float(lensed.sum()) / float(unlensed.sum())
    assert magnification > 1.0
    assert np.isfinite(magnification)


def test_lensed_render_is_exactly_linear_in_the_contract_normalization():
    """Doubling the contract flux doubles the lensed image exactly.

    Linearity in the configured normalization is necessary but not
    sufficient: a constant spurious magnification factor would survive it
    unchanged. The absolute anchor that excludes such a factor is
    :func:`test_lensed_flux_matches_the_analytic_circular_lens_anchor`.
    """
    anchor = ANALYTIC_CANARY_ANCHOR
    asset = load_source_image_asset(_asset_path(anchor))
    total_flux = asset.metadata["provenance"]["rate_contract"]["total_flux"]

    single = np.asarray(_lensed_scene(anchor, total_flux).image, dtype=float)
    doubled = np.asarray(_lensed_scene(anchor, 2.0 * total_flux).image, dtype=float)

    np.testing.assert_array_equal(doubled, 2.0 * single)


def _source_plane_radii(asset):
    """Return every asset sample's distance from the canary lens centre.

    The ``Image`` profile lays sample ``(row, col)`` at the sky offset
    ``(row - row_c, col - col_c) * pixel_scale_arcsec`` from its centre at
    unit size scale and zero rotation, so the source-plane radius of each
    sample follows from the asset geometry alone.
    """
    rows, cols = np.indices(asset.sb.shape, dtype=float)
    offset_y = (rows - (asset.sb.shape[0] - 1) / 2.0) * asset.pixel_scale_arcsec
    offset_x = (cols - (asset.sb.shape[1] - 1) / 2.0) * asset.pixel_scale_arcsec
    return np.hypot(
        ANALYTIC_CANARY_SOURCE_CENTRE_ARCSEC[0] + offset_y,
        ANALYTIC_CANARY_SOURCE_CENTRE_ARCSEC[1] + offset_x,
    )


def _analytic_canary_render_config(anchor):
    """Return the canary grid and Image-source blocks of one anchor."""
    grid_config = {
        "shape": list(ANALYTIC_CANARY_GRID_SHAPE),
        "pixel_scale": ANALYTIC_CANARY_PIXEL_SCALE_ARCSEC,
    }
    source_config = {
        "redshift": 0.6,
        "light": {
            "type": "Image",
            "asset_path": str(_asset_path(anchor)),
            "centre": list(ANALYTIC_CANARY_SOURCE_CENTRE_ARCSEC),
            "rotation_deg": 0.0,
            "total_flux": 1.0,
            "flux_scale": 1.0,
            "size_scale": 1.0,
        },
    }
    return grid_config, source_config


def _analytic_canary_system(anchor, total_flux):
    """Return the circular-lens canary system of one anchor at one flux."""
    grid_config, source_config = _analytic_canary_render_config(anchor)
    source_config["light"]["total_flux"] = float(total_flux)
    lensing = {
        "grid": grid_config,
        "lens_galaxy": {
            "redshift": 0.2,
            "mass": {
                "type": "Isothermal",
                "centre": [0.0, 0.0],
                "ell_comps": [0.0, 0.0],
                "einstein_radius": ANALYTIC_CANARY_EINSTEIN_RADIUS_ARCSEC,
            },
        },
        "source_galaxy": source_config,
        "subhalo": {"enabled": False},
        "cosmology": "Planck15",
    }
    full_config = {
        "run_name": "analytic-magnification-canary",
        "global_seed": 5,
        "lensing": lensing,
    }
    return generate_lensing_system(lensing, full_config)


def test_lensed_flux_matches_the_analytic_circular_lens_anchor():
    """The lensed flux is the contract flux magnified exactly once.

    A circular isothermal lens magnifies a source at ``0 < |beta| <
    theta_E`` by ``mu(beta) = 2 theta_E / |beta|``, and the lensed flux of
    an extended source is that magnification integrated against its own
    surface brightness. The prediction is therefore an absolute flux built
    from the asset samples and the stored contract normalization alone,
    with no ratio of two pipeline renders in it: a pipeline that applied
    magnification a second time would land on the ``mu**2`` prediction,
    four times the right answer, and a pipeline that applied none would
    land on the unlensed flux.
    """
    anchor = ANALYTIC_CANARY_ANCHOR
    asset = load_source_image_asset(_asset_path(anchor))
    total_flux = asset.metadata["provenance"]["rate_contract"]["total_flux"]
    pixel_area = ANALYTIC_CANARY_PIXEL_SCALE_ARCSEC**2

    radius = _source_plane_radii(asset)
    lit = asset.sb > 0.0
    assert float(radius[lit].min()) > 0.0
    assert float(radius[lit].max()) < ANALYTIC_CANARY_EINSTEIN_RADIUS_ARCSEC

    magnification = 2.0 * ANALYTIC_CANARY_EINSTEIN_RADIUS_ARCSEC / radius
    weight = asset.sb * asset.pixel_scale_arcsec**2
    once = total_flux * float((weight * magnification).sum())
    twice = total_flux * float((weight * magnification**2).sum())
    assert twice / once > 3.0

    grid_config, source_config = _analytic_canary_render_config(anchor)
    unlensed = render_unlensed_asset(
        _asset_path(anchor), grid_config, source_config, total_flux
    )
    assert float(unlensed.sum()) * pixel_area == pytest.approx(
        total_flux, rel=ANALYTIC_CANARY_TOLERANCE
    )

    lensed = np.asarray(_analytic_canary_system(anchor, total_flux).image, dtype=float)
    for border in (lensed[0], lensed[-1], lensed[:, 0], lensed[:, -1]):
        assert float(np.abs(border).max()) == 0.0

    assert float(lensed.sum()) * pixel_area == pytest.approx(
        once, rel=ANALYTIC_CANARY_TOLERANCE
    )
    assert float(lensed.sum()) * pixel_area != pytest.approx(twice, rel=0.5)
