#!/usr/bin/env python
"""Embed the detected-rate contract into an already-prepared source asset.

The Sol Pro P0-3 contract is solved by ``prepare_source_image`` while an
asset is being prepared. The legacy bank anchor predates that step, so it
carries no ``provenance.rate_contract`` block and cannot be verified. This
tool solves the contract for an existing asset through the exact same code
path and rewrites the asset with the block added, leaving the samples, the
pixel scale, and every other provenance key untouched.

``--refresh`` instead re-solves the contract of an asset that already
carries one, for when a contract input the stored digests bind, such as
the observing reference, was regenerated. The verification gate fails
closed on the stale digests, so the contract must be re-solved against
the live files; the samples and every other provenance key are held
byte-identical exactly as in the embedding case.

Rewriting the asset changes its file bytes, so after a run the printed
sha256 must replace the old pin in every place that binds it:

  * ``configs/design/design_freeze_v1.yaml`` source bank level ``sha256``,
    together with its ``canonical_total_flux_source`` text, which stops
    being the observing-reference patch and becomes
    ``provenance.rate_contract.total_flux``
  * the anchor ``sha256`` in ``tests/test_source_bank_assets.py``
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_source_image import (  # noqa: E402
    OBSERVING_REFERENCE_RELPATH,
    PRODUCTION_SCENE_RELPATH,
    PROJECT_ROOT,
    _clear_asset_loader_cache,
    _sha256,
    detected_rate_reference,
    production_render_config,
    solve_detected_rate_normalization,
    verify_asset_rate_contract,
    write_asset,
)

ASSET_KEYS = {'sb', 'pixel_scale_arcsec', 'metadata_json'}
"""Exact array set of a version-one source-image asset."""


def read_prepared_asset(path):
    """Read one prepared asset without the production loader.

    The retrofit has to prove that it preserved the stored bytes, so the
    arrays are read raw rather than through the validating loader, whose
    cached view is shared with the rendering path.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Prepared ``.npz`` source-image asset.

    Returns
    -------
    asset : `dict`
        Samples, pixel scale, and parsed metadata of the asset.
    """
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != ASSET_KEYS:
            raise ValueError(
                f'Asset {path} must contain exactly {sorted(ASSET_KEYS)}, '
                f'found {sorted(keys)}'
            )
        sb = np.asarray(archive['sb'], dtype=np.float64)
        pixel_scale = np.asarray(archive['pixel_scale_arcsec'])
        encoded = np.asarray(archive['metadata_json']).item()
    if isinstance(encoded, bytes):
        encoded = encoded.decode('utf-8')
    metadata = json.loads(encoded)
    if pixel_scale.ndim != 0 or pixel_scale.dtype != np.dtype(np.float64):
        raise ValueError(
            f'Asset {path} pixel_scale_arcsec must be a float64 scalar'
        )
    if not isinstance(metadata, dict) or metadata.get('format_version') != 1:
        raise ValueError(f'Asset {path} metadata format_version must be 1')
    if not isinstance(metadata.get('provenance'), dict):
        raise ValueError(f'Asset {path} metadata provenance must be a dict')
    return {
        'sb': sb,
        'pixel_scale_arcsec': float(pixel_scale),
        'metadata': metadata,
    }


def _assert_only_the_contract_changed(original, retrofitted, path):
    """Fail unless a rewritten asset changed the contract and nothing else."""
    if retrofitted['sb'].shape != original['sb'].shape:
        raise ValueError(f'Retrofit changed the sb shape of {path}')
    if retrofitted['sb'].tobytes() != original['sb'].tobytes():
        raise ValueError(f'Retrofit changed the sb samples of {path}')
    if (
        np.float64(retrofitted['pixel_scale_arcsec']).tobytes()
        != np.float64(original['pixel_scale_arcsec']).tobytes()
    ):
        raise ValueError(f'Retrofit changed the pixel scale of {path}')
    provenance = dict(retrofitted['metadata']['provenance'])
    if 'rate_contract' not in provenance:
        raise ValueError(f'Retrofit wrote no rate_contract into {path}')
    provenance.pop('rate_contract')
    base = dict(original['metadata']['provenance'])
    base.pop('rate_contract', None)
    if provenance != base:
        raise ValueError(f'Retrofit changed the stored provenance of {path}')
    expected = dict(original['metadata'])
    expected['provenance'] = retrofitted['metadata']['provenance']
    if retrofitted['metadata'] != expected:
        raise ValueError(f'Retrofit changed the stored metadata of {path}')


def retrofit_rate_contract(asset_path, scene_path=None, reference_path=None,
                           refresh=False):
    """Solve and embed the rate contract of one prepared asset in place.

    The contract is solved against the asset exactly as committed, the
    rewritten asset is staged beside it, and the staged file is proved to
    preserve the samples, the pixel scale, and the stored provenance and
    to verify against the reference before it replaces the original.

    Parameters
    ----------
    asset_path : `str` or `pathlib.Path`
        Prepared asset carrying no ``provenance.rate_contract`` block,
        or, under ``refresh``, exactly one.
    scene_path : `str` or `pathlib.Path`, optional
        Production scene supplying the contract render geometry. Defaults
        to the committed production Image scene.
    reference_path : `str` or `pathlib.Path`, optional
        Observing reference supplying the target detected rate. Defaults
        to the committed observing reference.
    refresh : `bool`, optional
        Re-solve and replace an existing contract instead of embedding a
        first one. This exists for a regenerated contract input, whose
        stored digest fails the verification gate until the contract is
        re-solved against the live bytes; it never changes which files
        the contract is solved against, only when a stored block may be
        replaced.

    Returns
    -------
    contract : `dict`
        The embedded and verified rate contract.
    """
    asset_path = Path(asset_path).expanduser().resolve()
    scene_path = Path(
        scene_path
        if scene_path is not None
        else PROJECT_ROOT / PRODUCTION_SCENE_RELPATH
    ).expanduser().resolve()
    reference_path = Path(
        reference_path
        if reference_path is not None
        else PROJECT_ROOT / OBSERVING_REFERENCE_RELPATH
    ).expanduser().resolve()

    original = read_prepared_asset(asset_path)
    contracted = 'rate_contract' in original['metadata']['provenance']
    if contracted and not refresh:
        raise ValueError(
            f'Asset {asset_path} already carries a provenance.rate_contract '
            'block; retrofitting would overwrite a solved contract. Pass '
            'refresh to re-solve it against regenerated contract inputs'
        )
    if refresh and not contracted:
        raise ValueError(
            f'Asset {asset_path} carries no provenance.rate_contract block '
            'to refresh; run the plain retrofit to embed a first contract'
        )

    reference = detected_rate_reference(reference_path)
    grid_config, source_config = production_render_config(
        scene_path, reference['pixel_scale_arcsec']
    )
    _clear_asset_loader_cache()
    contract = solve_detected_rate_normalization(
        asset_path, reference, grid_config, source_config, scene_path
    )

    provenance = dict(original['metadata']['provenance'])
    provenance['rate_contract'] = contract
    handle, staged_name = tempfile.mkstemp(
        prefix=f'{asset_path.stem}.retrofit.', suffix='.npz',
        dir=str(asset_path.parent),
    )
    os.close(handle)
    staged_path = Path(staged_name)
    try:
        write_asset(
            staged_path,
            original['sb'],
            original['pixel_scale_arcsec'],
            provenance,
        )
        _assert_only_the_contract_changed(
            original, read_prepared_asset(staged_path), asset_path
        )
        _clear_asset_loader_cache()
        verify_asset_rate_contract(staged_path, scene_path, reference_path)
        os.chmod(staged_path, stat.S_IMODE(asset_path.stat().st_mode))
        os.replace(staged_path, asset_path)
    finally:
        if staged_path.exists():
            staged_path.unlink()
        _clear_asset_loader_cache()
    return contract


def _argument_parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset')
    parser.add_argument(
        '--reference',
        default=str(PROJECT_ROOT / OBSERVING_REFERENCE_RELPATH),
        help='observing reference supplying the target detected rate',
    )
    parser.add_argument(
        '--scene',
        default=str(PROJECT_ROOT / PRODUCTION_SCENE_RELPATH),
        help='production Image-source scene supplying the contract render',
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='re-solve and replace an existing contract whose recorded '
             'input digests went stale under a regenerated input',
    )
    return parser


def main(argv=None):
    """Run the rate-contract retrofit CLI.

    Parameters
    ----------
    argv : sequence of `str`, optional
        Arguments excluding the program name. Defaults to ``sys.argv``.

    Returns
    -------
    status : `int`
        Zero on success.
    """
    args = _argument_parser().parse_args(argv)
    asset_path = Path(args.asset).expanduser().resolve()
    contract = retrofit_rate_contract(
        asset_path, args.scene, args.reference, refresh=args.refresh
    )
    print(f"{'asset':<24} {asset_path}")
    print(f"{'sha256':<24} {_sha256(asset_path)}")
    print(f"{'target_rate_e_per_s':<24} {contract['target_rate_e_per_s']:.12g}")
    print(
        f"{'realized_rate_e_per_s':<24} "
        f"{contract['realized_rate_e_per_s']:.12g}"
    )
    print(f"{'contract_total_flux':<24} {contract['total_flux']:.17g}")
    print(
        f"{'discrete_mapping_ratio':<24} "
        f"{contract['discrete_mapping_ratio']:.12g}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
