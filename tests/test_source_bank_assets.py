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
        "fb51b801b98653f6a263016fa55ae0e1194734c197081d652ee00c51077ccd60"
    ),
    "script_version": 1,
}

NEW_ANCHORS = (
    {
        "ident": 62410,
        "morphology_class": "smooth_disk",
        "filename": "cosmos_62410_hlr011.npz",
        "sha256": (
            "b3fdc541c3e31013d5bb37c9e948fa74a9ce485cbb6dd1d959d3c4f0a5392376"
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
            "1d601a6c3e969d558102c72b25307b2328a044396adcafbb63f3721aae435aa0"
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
            "795d2f5eb636cbb59440547950260b70032f6e52540ac1c582e58ad0daf8ee4b"
        ),
        "script_version": 2,
        "pixel_scale_arcsec": 0.005584403908234906,
        "total_flux": 0.0004589439758914851,
    },
    {
        "ident": 83935,
        "morphology_class": "compact",
        "filename": "cosmos_83935_hlr011.npz",
        "sha256": (
            "50ae26b0bb66d75e0147b669f40028f5333e665ff16e5444326fa682e6723ef9"
        ),
        "script_version": 2,
        "pixel_scale_arcsec": 0.012148903205522409,
        "total_flux": 0.0004588806668010262,
    },
)

BANK = (LEGACY_ANCHOR,) + NEW_ANCHORS


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


@pytest.mark.parametrize("anchor", NEW_ANCHORS, ids=_ids(NEW_ANCHORS))
def test_new_bank_asset_stores_the_detected_rate_contract(anchor, reference):
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


@pytest.mark.parametrize("anchor", NEW_ANCHORS, ids=_ids(NEW_ANCHORS))
def test_new_bank_asset_re_renders_to_its_target_rate(anchor):
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


@pytest.mark.parametrize("anchor", NEW_ANCHORS, ids=_ids(NEW_ANCHORS))
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

    A magnification applied a second time anywhere in the pipeline would
    break this identity, because it would multiply a quantity that is not
    the configured normalization.
    """
    anchor = NEW_ANCHORS[0]
    asset = load_source_image_asset(_asset_path(anchor))
    total_flux = asset.metadata["provenance"]["rate_contract"]["total_flux"]

    single = np.asarray(_lensed_scene(anchor, total_flux).image, dtype=float)
    doubled = np.asarray(_lensed_scene(anchor, 2.0 * total_flux).image, dtype=float)

    np.testing.assert_array_equal(doubled, 2.0 * single)
