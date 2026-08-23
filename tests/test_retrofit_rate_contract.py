"""Tests for the rate-contract retrofit of already-prepared assets.

The bank anchor prepared before the contract solve existed is contracted in
place by this tool, so the tests hold it to the two properties that make
the retrofit safe: the embedded contract is the one the preparation solve
would have produced, and nothing else in the asset moves.
"""

from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from hwoslaps.lensing import load_source_image_asset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_source_image import (  # noqa: E402
    OBSERVING_REFERENCE_RELPATH,
    PRODUCTION_SCENE_RELPATH,
    detected_rate_reference,
    verify_asset_rate_contract,
)
from prepare_source_image import main as prepare_main  # noqa: E402
from retrofit_rate_contract import (  # noqa: E402
    main,
    read_prepared_asset,
    retrofit_rate_contract,
)


def _resolved_source(shape=(96, 96), background=2.0):
    """Return a background-plus-Gaussian source resolved over many pixels."""
    rows, cols = np.indices(shape, dtype=float)
    radius = np.hypot(rows - 47.0, cols - 47.0)
    source = np.where(radius <= 30.0, 50.0 * np.exp(-0.5 * (radius / 10.0) ** 2), 0.0)
    return background + source


def _prepared_asset(tmp_path, name, rate_contract=False):
    """Prepare one synthetic asset with or without the contract solve."""
    input_path = tmp_path / "resolved.npy"
    if not input_path.exists():
        np.save(input_path, _resolved_source())
    output_path = tmp_path / name
    argv = [
        str(input_path),
        str(output_path),
        "--target-half-light-arcsec",
        "0.11",
    ]
    if rate_contract:
        argv.append("--rate-contract")
    assert prepare_main(argv) == 0
    return output_path


def test_retrofit_embeds_the_contract_and_preserves_the_asset(tmp_path):
    """The retrofit adds the contract block and moves nothing else."""
    asset_path = _prepared_asset(tmp_path, "legacy.npz")
    original = read_prepared_asset(asset_path)
    assert "rate_contract" not in original["metadata"]["provenance"]

    contract = retrofit_rate_contract(asset_path)

    reference = detected_rate_reference(PROJECT_ROOT / OBSERVING_REFERENCE_RELPATH)
    assert contract["target_rate_e_per_s"] == reference["target_rate_e_per_s"]
    assert contract["realized_rate_e_per_s"] == pytest.approx(
        reference["target_rate_e_per_s"], rel=1.0e-12
    )

    retrofitted = read_prepared_asset(asset_path)
    np.testing.assert_array_equal(retrofitted["sb"], original["sb"])
    assert retrofitted["pixel_scale_arcsec"] == original["pixel_scale_arcsec"]
    assert retrofitted["metadata"]["format_version"] == 1
    provenance = deepcopy(retrofitted["metadata"]["provenance"])
    assert provenance.pop("rate_contract") == contract
    assert provenance == original["metadata"]["provenance"]
    assert verify_asset_rate_contract(asset_path) == contract


def test_retrofit_reproduces_the_preparation_contract(tmp_path):
    """The retrofit solve is the one the preparation tool would have run."""
    prepared_path = _prepared_asset(tmp_path, "prepared.npz", rate_contract=True)
    retrofitted_path = _prepared_asset(tmp_path, "retrofitted.npz")

    contract = retrofit_rate_contract(retrofitted_path)

    expected = load_source_image_asset(prepared_path).metadata["provenance"][
        "rate_contract"
    ]
    assert contract == expected


def test_retrofit_refuses_an_asset_that_already_carries_a_contract(tmp_path):
    """A solved contract is never silently re-solved and overwritten."""
    asset_path = _prepared_asset(tmp_path, "contracted.npz", rate_contract=True)

    with pytest.raises(ValueError, match="already carries"):
        retrofit_rate_contract(asset_path)


def test_retrofit_leaves_the_asset_untouched_when_the_solve_fails(tmp_path):
    """A refused contract render neither rewrites nor litters the asset."""
    asset_path = _prepared_asset(tmp_path, "legacy.npz")
    before = asset_path.read_bytes()
    scene = yaml.safe_load(
        (PROJECT_ROOT / PRODUCTION_SCENE_RELPATH).read_text(encoding="utf-8")
    )
    scene["lensing"]["grid"]["pixel_scale"] = (
        2.0 * scene["lensing"]["grid"]["pixel_scale"]
    )
    scene_path = tmp_path / "foreign_scene.yaml"
    scene_path.write_text(yaml.safe_dump(scene), encoding="utf-8")

    with pytest.raises(ValueError, match="observing reference declares"):
        retrofit_rate_contract(asset_path, scene_path)

    assert asset_path.read_bytes() == before
    assert list(tmp_path.glob("*.retrofit.*")) == []


def test_refresh_re_solves_a_contract_against_a_regenerated_reference(tmp_path):
    """A stale stored reference digest is replaced by a re-solve.

    The regenerated reference carries the same target rate in different
    bytes, which is exactly the case the verification gate fails closed
    on: the refresh must re-bind the contract to the live bytes while
    holding the samples and the solved numbers fixed.
    """
    asset_path = _prepared_asset(tmp_path, "contracted.npz", rate_contract=True)
    original = read_prepared_asset(asset_path)
    stale = original["metadata"]["provenance"]["rate_contract"]
    reference_path = PROJECT_ROOT / OBSERVING_REFERENCE_RELPATH
    regenerated_path = tmp_path / "regenerated_reference.yaml"
    regenerated_path.write_text(
        reference_path.read_text(encoding="utf-8") + "\n# regenerated\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="observing reference.*now hashes to"):
        verify_asset_rate_contract(
            asset_path, PROJECT_ROOT / PRODUCTION_SCENE_RELPATH, regenerated_path
        )

    contract = retrofit_rate_contract(
        asset_path, reference_path=regenerated_path, refresh=True
    )

    assert contract["target_rate_e_per_s"] == stale["target_rate_e_per_s"]
    assert contract["total_flux"] == stale["total_flux"]
    assert contract["observing_reference"]["sha256"] == hashlib.sha256(
        regenerated_path.read_bytes()
    ).hexdigest()
    refreshed = read_prepared_asset(asset_path)
    np.testing.assert_array_equal(refreshed["sb"], original["sb"])
    assert verify_asset_rate_contract(
        asset_path, PROJECT_ROOT / PRODUCTION_SCENE_RELPATH, regenerated_path
    ) == contract
    with pytest.raises(ValueError, match="observing reference.*now hashes to"):
        verify_asset_rate_contract(asset_path)


def test_refresh_refuses_an_asset_without_a_contract(tmp_path):
    """A refresh of nothing is a wrong invocation, not a silent embed."""
    asset_path = _prepared_asset(tmp_path, "legacy.npz")

    with pytest.raises(ValueError, match="no provenance.rate_contract block"):
        retrofit_rate_contract(asset_path, refresh=True)


def test_read_prepared_asset_rejects_a_foreign_archive(tmp_path):
    """An archive that is not a version-one asset fails loudly."""
    path = tmp_path / "foreign.npz"
    np.savez(path, sb=np.ones((8, 8)))

    with pytest.raises(ValueError, match="must contain exactly"):
        read_prepared_asset(path)


def test_retrofit_cli_reports_the_rewritten_asset_identity(tmp_path, capsys):
    """The CLI prints the new content hash the pins have to follow."""
    asset_path = _prepared_asset(tmp_path, "legacy.npz")

    assert main([str(asset_path)]) == 0

    printed = capsys.readouterr().out
    contract = load_source_image_asset(asset_path).metadata["provenance"][
        "rate_contract"
    ]
    assert hashlib.sha256(asset_path.read_bytes()).hexdigest() in printed
    assert f"{contract['target_rate_e_per_s']:.12g}" in printed
