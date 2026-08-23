"""Offline tests for the HWO EAC1 HRI reference derivation script."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
SCRIPT_PATH = PROJECT_ROOT / 'scripts' / 'derive_hwo_eac1_hri_reference.py'
SPEC = importlib.util.spec_from_file_location('derive_hwo_reference', SCRIPT_PATH)
DERIVATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DERIVATION)

QUALIFICATION_TOTAL_FLUX = 0.289151264
EXPONENTIAL_SCENES = (
    'scene1_smooth_ring',
    'scene3_bow_dot',
    'scene5_flex_macro',
    'scene5_ablation_sie_fit',
)
NUMBER_OF_CLUMPS = 3
CLUMP_FLUX_FRACTION = 0.1
STUB_AREA_M2 = 33.6
PIXEL_SCALE_ARCSEC = 0.00716
PIXEL_AREA_ARCSEC2 = PIXEL_SCALE_ARCSEC ** 2

PHYSICAL_TARGET_RATE_E_PER_S = 14.78754553
CORRECTED_EXPONENTIAL_INTENSITY = 0.00524356964
CORRECTED_CLUMPY_FLUX_SCALE = 0.00262178482
CORRECTED_IMAGE_TOTAL_FLUX = 0.00075809239
ANCHOR_RELATIVE_TOLERANCE = 1.0e-2
"""Window around the audit's provisional corrected normalizations.

The audit derived those three values as the continuous-convention
values divided by the detector pixel area. The discrete render on the
production grid reproduces them to well inside a percent; a wider miss
means the render, the grid, or the target rate moved.
"""

COMMITTED_REFERENCE_PATH = (
    PROJECT_ROOT / 'configs' / 'observing' / 'hwo_eac1_hri_reference_v1.yaml'
)
COMMITTED_BLANK_PIXEL_VARIANCE_E2 = 9.10056
"""Blank-pixel variance of the reference, in electrons squared.

``2000 * (0.00251028 + 0.002) + 0.28284 ** 2``, quoted to five
decimals, so the blank-pixel sigma is 3.01671 e-. The retired
LUVOIR-HDI read noise of 2.5 e- per read gave 21.52056 e^2 and a sigma
of 4.63903 e-; both are provenance history since the SEI ruling of
2026-08-23.
"""

AUDIT_ARC_SNR = 303.94
"""Integrated source-only arc S/N the audit recovered.

That value came from the depth-ladder run, which observed scene 1
through an aberrated science PSF state and a 999 pixel kernel. A render
through the scene's own perfect PSF concentrates the same electrons
more tightly and therefore lands above it, so the production check
brackets the value widely rather than pinning it.
"""

SYNTHETIC_PROFILE_SIDE = 24
SYNTHETIC_MAGNIFICATIONS = (18.7, 11.3, 25.9, 14.1, 21.4, 9.6)


def _committed_reference():
    """Load the committed reference artifact."""
    with COMMITTED_REFERENCE_PATH.open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def _committed_baseline_sums():
    """Map every scene config path onto its committed discrete sum."""
    details = _committed_reference()['metadata'][
        'source_normalization_details'
    ]['scene_details']
    return {
        detail['scene_config']: detail['baseline_discrete_sum']
        for detail in details.values()
    }


def _freeze_baseline_sums(monkeypatch):
    """Serve the committed discrete render sums without the engine.

    The unlensed render is the only part of the normalization solve that
    needs autolens. Replaying the sums the committed artifact recorded
    keeps every downstream contract testable on a bare interpreter and
    ties the replay to the artifact rather than to a copied literal.
    """
    sums = _committed_baseline_sums()

    def frozen_sum(scene_config_relpath):
        if scene_config_relpath not in sums:
            raise AssertionError(
                f'No committed discrete sum for {scene_config_relpath}.'
            )
        return sums[scene_config_relpath]

    monkeypatch.setattr(DERIVATION, 'unlensed_discrete_sum', frozen_sum)
    return sums


def _synthetic_source_profile():
    """Return a compact non-negative profile normalized to unit sum."""
    axis = np.linspace(-3.0, 3.0, SYNTHETIC_PROFILE_SIDE)
    rows, columns = np.meshgrid(axis, axis, indexing='ij')
    profile = np.exp(-((rows - 0.4) ** 2 + (columns + 0.6) ** 2))
    return profile / float(np.sum(profile))


def _synthetic_forward_model(monkeypatch, baseline_sums):
    """Replace the production forward model with a linear stand-in.

    Each scene keeps its own magnification, so the scenes reach one arc
    S/N at genuinely different intrinsic rates, which is the property
    the arc_snr mode exists to produce.
    """
    profile = _synthetic_source_profile()
    magnifications = dict(
        zip(sorted(baseline_sums), SYNTHETIC_MAGNIFICATIONS)
    )
    baselines = {
        baseline['scene_config']: baseline
        for baseline in DERIVATION.SCENE_LIGHT_BASELINES.values()
    }

    def forward(scene_config_relpath, observation, light_patch):
        light = light_patch['lensing']['source_galaxy']['light']
        field = DERIVATION.SCENE_NORMALIZED_FIELD[
            baselines[scene_config_relpath]['light_type']
        ]
        scene = DERIVATION.load_scene_config(scene_config_relpath)
        baseline_value = scene['lensing']['source_galaxy']['light'][field]
        scale = light[field] / baseline_value
        rate = baseline_sums[scene_config_relpath] * scale
        return (
            profile
            * magnifications[scene_config_relpath]
            * rate
            * observation['exposure_time']
        )

    monkeypatch.setattr(DERIVATION, 'scene_source_electrons', forward)
    return magnifications


def _stub_area_report():
    """Return a minimal collecting-area report for offline document tests.

    Only the area itself is a stub. The geometry entries come from the
    committed artifact, because the SEI instrument-reference block checks
    them against the vendored EAC1 prescription and a hand-written
    geometry would drift from the production pupil.
    """
    committed = _committed_reference()['metadata']['collecting_area']
    return {
        'value_m2': STUB_AREA_M2,
        'method': 'offline test stub',
        'num_segments': committed['num_segments'],
        'geometry': committed['geometry'],
    }


def _require_vendored_sei():
    """Skip unless the vendored SEI data files are present.

    The SEI files live under the gitignored ``scratch`` tree, so a
    checkout without them cannot build a document at all. Every honesty
    check on the SEI read itself runs against synthetic files in a
    temporary directory and needs no skip.
    """
    for filename in (DERIVATION.SEI_HRI_FILENAME, DERIVATION.SEI_EAC1_FILENAME):
        if not (DERIVATION.SEI_VENDOR_PATH / filename).is_file():
            pytest.skip(
                f'Vendored SEI file {filename} is absent from '
                f'{DERIVATION.SEI_VENDOR_RELPATH}.'
            )


def _sei_hri_document(read_noise_e=0.2, dark_current=0.002,
                      plate_scale_mas=7.16):
    """Build the ``HRI.yaml`` subset the derivation reads."""
    return {
        'UVIS': {
            'plate_scale': [plate_scale_mas, 'mas'],
            'detector': {
                'detector_RN': [read_noise_e, 'electrons'],
                'detector_DC': [dark_current, 'electrons/pixel/second'],
            },
        },
    }


def _sei_eac1_document(segment_size=1.65, optical_gap=0.006, num_rings=2,
                       num_segments=19, circumscribing_diameter=7.225765):
    """Build the ``EAC1.yaml`` subset the derivation reads."""
    return {
        'PM': {
            'circumscribing_diameter': [circumscribing_diameter, 'meters'],
            'segmentation_parameters': {
                'segment_size': [segment_size, 'meters'],
                'optical_gap': [optical_gap, 'meters'],
                'number_rings': [num_rings, 'unitless'],
                'number_segments': [num_segments, 'unitless'],
            },
        },
    }


def _write_sei_vendor(path, hri=None, eac1=None):
    """Write one SEI vendor directory and return its path.

    A ``None`` document leaves that file out, which is the removed-file
    case; a raw string is written verbatim, which is the unparsable case.
    """
    path.mkdir(parents=True, exist_ok=True)
    for filename, document in (
        (DERIVATION.SEI_HRI_FILENAME, hri),
        (DERIVATION.SEI_EAC1_FILENAME, eac1),
    ):
        if document is None:
            continue
        with (path / filename).open('w', encoding='utf-8') as stream:
            if isinstance(document, str):
                stream.write(document)
            else:
                yaml.safe_dump(document, stream, sort_keys=False)
    return path


def _synthetic_sei_vendor(tmp_path, hri=None, eac1=None):
    """Write a faithful synthetic SEI vendor directory, edits applied."""
    return _write_sei_vendor(
        tmp_path / 'sei',
        hri=_sei_hri_document() if hri is None else hri,
        eac1=_sei_eac1_document() if eac1 is None else eac1,
    )


def _document(**overrides):
    """Build one reference document without touching the pupil model."""
    pytest.importorskip('autolens')
    _require_vendored_sei()

    parameters = {
        'area_report': _stub_area_report(),
        'pixel_scale_arcsec': PIXEL_SCALE_ARCSEC,
        'sed_mode': 'flat_fnu',
        'color_ab': None,
        'mag_vband': None,
        'exposure_s': 2000.0,
        'system_qe': 0.21,
        'sky_mag': 23.0,
    }
    parameters.update(overrides)
    return DERIVATION.build_reference_document(**parameters)


def _scene_light(label):
    """Return one scene's configured source-light block."""
    relpath = DERIVATION.SCENE_LIGHT_BASELINES[label]['scene_config']
    scene = DERIVATION.load_scene_config(relpath)
    return scene['lensing']['source_galaxy']['light']


def _patched_render_sum(label, leaf):
    """Sum one scene's unlensed render after applying a patch leaf."""
    relpath = DERIVATION.SCENE_LIGHT_BASELINES[label]['scene_config']
    lensing = DERIVATION.load_scene_config(relpath)['lensing']
    lensing['source_galaxy']['light'].update(leaf)
    image = DERIVATION.render_unlensed_source(
        lensing['source_galaxy'], lensing['grid']
    )
    return float(np.sum(image))


def test_ab_magnitude_converts_to_the_hand_computed_flux_density():
    """Reproduce the AB zero-point conversion independently of the module."""
    expected = 3631.0 * 10.0 ** (-0.4 * 24.3)

    assert DERIVATION.ab_mag_to_fnu_jy(24.3) == pytest.approx(expected, rel=1.0e-12)
    assert DERIVATION.AB_ZERO_POINT_JY == 3631.0
    assert DERIVATION.ab_mag_to_fnu_jy(0.0) == pytest.approx(3631.0, rel=1.0e-12)


def test_photon_integral_matches_the_hand_computed_flat_fnu_band():
    """Reproduce the flat-f_nu photon integral over the 450-550 nm band."""
    expected = 1.0e-26 / 6.62607015e-34 * math.log(550.0 / 450.0)

    rate = DERIVATION.photon_rate_per_m2(1.0, 450.0e-9, 550.0e-9)

    assert rate == pytest.approx(expected, rel=1.0e-12)
    assert 1.0e-26 / DERIVATION.PLANCK_H == pytest.approx(1.50919e7, rel=1.0e-5)
    with pytest.raises(ValueError, match='must exceed'):
        DERIVATION.photon_rate_per_m2(1.0, 550.0e-9, 450.0e-9)


def test_sky_chain_matches_an_independent_computation_and_stays_physical():
    """Reproduce the sky electron rate and bracket it physically."""
    flux_density = 3631.0 * 10.0 ** (-0.4 * 23.0)
    photons = flux_density * 1.0e-26 / 6.62607015e-34 * math.log(550.0 / 450.0)
    expected = photons * 33.6 * 0.21 * 0.00716 ** 2

    rate = DERIVATION.sky_rate_e_per_pix_s(23.0, 33.6, 0.21, 0.00716)

    assert rate == pytest.approx(expected, rel=1.0e-12)
    assert 2.0e-3 < rate < 3.0e-3


def test_effective_read_noise_is_the_combined_image_value():
    """Pin the combined-image read noise and its provenance pair."""
    assert DERIVATION.effective_read_noise(0.2, 2) == 0.2 * math.sqrt(2.0)
    assert DERIVATION.effective_read_noise(0.2, 2) == 0.28284271247461906

    reference = _document()
    detector = reference['observation']['detector']
    provenance = reference['metadata']['detector']

    assert detector['read_noise'] == 0.28284271247461906
    assert provenance['read_noise_per_read_e'] == 0.2
    assert provenance['n_reads'] == 2
    assert detector['read_noise'] != provenance['read_noise_per_read_e']
    assert provenance['effective_read_noise_formula'] == 'per_read_e * sqrt(n_reads)'
    assert provenance['read_noise_citation'] == 'SEI'
    assert '2.5 e- per read' in provenance['read_noise_retired_provenance']


def test_sei_is_the_named_instrument_reference_for_the_read_noise():
    """Pin the SEI provenance block and keep 2.5 e- out of the values."""
    metadata = _document()['metadata']
    instrument = metadata['instrument_reference']

    assert instrument['package'] == 'hwo_sci_eng'
    assert instrument['version'] == '0.1.9'
    assert instrument['status'] == (
        'current public HWO working engineering reference adopted for this '
        'study'
    )
    assert instrument['vendored_path'] == (
        'scratch/q1_observing_conditions/sei_v0.1.9'
    )
    assert instrument['citation'] == 'SEI'
    assert 'hwo_sci_eng v0.1.9' in metadata['citations']['SEI']

    adopted = instrument['adopted_parameters']['read_noise_per_read_e']
    assert adopted['value'] == 0.2
    assert '2.5 e- per read' in adopted['replaces']

    corroborated = instrument['corroborated_parameters']
    assert corroborated['pixel_scale_arcsec']['sei_value'] == 0.00716
    assert corroborated['pixel_scale_arcsec']['value'] == PIXEL_SCALE_ARCSEC
    assert corroborated['dark_current_e_per_pix_s']['sei_value'] == 0.002
    geometry = corroborated['telescope_geometry']['value']
    assert geometry['segment_point_to_point_m'] == 1.65
    assert geometry['gap_size_m'] == 0.006
    assert geometry['pupil_diameter_m'] == 7.225765
    assert geometry['num_rings'] == 2
    assert geometry['num_segments'] == 19

    assert DERIVATION.READ_NOISE_PER_READ_E == 0.2
    assert 2.5 not in (
        metadata['detector']['read_noise_per_read_e'],
        metadata['detector']['effective_read_noise_e'],
        metadata['detector']['dark_current_e_per_pix_s'],
    )


def test_the_sei_values_come_out_of_the_vendored_files():
    """Read the real vendored files and pin what they supply.

    This is the check that the declared constants are the SEI's own
    numbers rather than a restatement of them: every value the artifact
    attributes to the SEI is parsed back out of the vendored bytes here,
    and the recorded digests are recomputed independently.
    """
    _require_vendored_sei()

    sei = DERIVATION.read_sei_instrument_parameters()

    assert sei['read_noise_per_read_e'] == DERIVATION.READ_NOISE_PER_READ_E
    assert sei['dark_current_e_per_pix_s'] == (
        DERIVATION.DARK_CURRENT_E_PER_PIX_S
    )
    assert sei['plate_scale_arcsec'] == pytest.approx(
        DERIVATION.SEI_PLATE_SCALE_ARCSEC, rel=1.0e-12
    )
    assert sei['segment_point_to_point_m'] == 1.65
    assert sei['gap_size_m'] == 0.006
    assert sei['num_rings'] == 2
    assert sei['num_segments'] == 19
    assert sei['pupil_diameter_m'] == 7.225765

    expected_digests = {
        filename: hashlib.sha256(
            (DERIVATION.SEI_VENDOR_PATH / filename).read_bytes()
        ).hexdigest()
        for filename in (
            DERIVATION.SEI_HRI_FILENAME, DERIVATION.SEI_EAC1_FILENAME
        )
    }
    instrument = DERIVATION._sei_provenance(
        PIXEL_SCALE_ARCSEC, _stub_area_report()
    )

    assert sei['sha256'] == expected_digests
    assert instrument['data_file_sha256'] == expected_digests
    assert instrument['data_files_read'] == list(expected_digests)
    assert 'fails the run' in instrument['data_file_note']
    assert instrument == _committed_reference()['metadata'][
        'instrument_reference'
    ]


def test_a_missing_vendored_sei_file_fails_the_derivation(tmp_path):
    """Refuse to claim SEI provenance without the SEI files."""
    area_report = _stub_area_report()
    empty = _write_sei_vendor(tmp_path / 'empty')
    hri_only = _write_sei_vendor(
        tmp_path / 'hri_only', hri=_sei_hri_document()
    )

    with pytest.raises(ValueError, match=r'HRI\.yaml does not exist'):
        DERIVATION._sei_provenance(
            PIXEL_SCALE_ARCSEC, area_report, vendor_path=empty
        )
    with pytest.raises(ValueError, match=r'EAC1\.yaml does not exist'):
        DERIVATION._sei_provenance(
            PIXEL_SCALE_ARCSEC, area_report, vendor_path=hri_only
        )


def test_an_edited_vendored_sei_value_fails_the_derivation(tmp_path):
    """Every declared constant must still match its edited source."""
    area_report = _stub_area_report()
    faithful = _synthetic_sei_vendor(tmp_path / 'faithful')

    assert DERIVATION._sei_provenance(
        PIXEL_SCALE_ARCSEC, area_report, vendor_path=faithful
    )['adopted_parameters']['read_noise_per_read_e']['value'] == 0.2

    edits = {
        'read_noise_per_read_e': {'hri': _sei_hri_document(read_noise_e=2.5)},
        'dark_current_e_per_pix_s': {
            'hri': _sei_hri_document(dark_current=0.02)
        },
        'plate_scale_arcsec': {
            'hri': _sei_hri_document(plate_scale_mas=14.32)
        },
        'segment_point_to_point_m': {
            'eac1': _sei_eac1_document(segment_size=1.7)
        },
        'gap_size_m': {'eac1': _sei_eac1_document(optical_gap=0.004)},
        'pupil_diameter_m': {
            'eac1': _sei_eac1_document(circumscribing_diameter=5.976938)
        },
        'num_rings': {'eac1': _sei_eac1_document(num_rings=3)},
        'num_segments': {'eac1': _sei_eac1_document(num_segments=18)},
    }
    for name, edit in edits.items():
        vendor = _synthetic_sei_vendor(tmp_path / name, **edit)
        with pytest.raises(ValueError, match=f'{name} is declared as'):
            DERIVATION._sei_provenance(
                PIXEL_SCALE_ARCSEC, area_report, vendor_path=vendor
            )


def test_a_malformed_vendored_sei_leaf_fails_the_derivation(tmp_path):
    """Reject a missing key, a wrong unit, or a non-numeric value."""
    area_report = _stub_area_report()
    without_read_noise = _sei_hri_document()
    del without_read_noise['UVIS']['detector']['detector_RN']
    wrong_unit = _sei_hri_document()
    wrong_unit['UVIS']['detector']['detector_RN'] = [0.2, 'ADU']
    not_a_pair = _sei_hri_document()
    not_a_pair['UVIS']['detector']['detector_RN'] = 0.2
    not_a_number = _sei_hri_document()
    not_a_number['UVIS']['detector']['detector_RN'] = ['0.2', 'electrons']

    cases = (
        (without_read_noise, r'missing UVIS\.detector\.detector_RN'),
        (wrong_unit, r"quoted in 'ADU', expected 'electrons'"),
        (not_a_pair, r'is not a \[value, unit\] pair'),
        (not_a_number, r'is not a number'),
        ('- a bare sequence\n', 'does not parse as a YAML mapping'),
    )
    for index, (hri, message) in enumerate(cases):
        vendor = _synthetic_sei_vendor(tmp_path / f'case{index}', hri=hri)
        with pytest.raises(ValueError, match=message):
            DERIVATION._sei_provenance(
                PIXEL_SCALE_ARCSEC, area_report, vendor_path=vendor
            )


def test_the_sei_check_guards_both_sides_of_every_corroboration(tmp_path):
    """Fail when the derivation moves away from the unchanged SEI files."""
    vendor = _synthetic_sei_vendor(tmp_path)
    area_report = _stub_area_report()
    moved_geometry = _stub_area_report()
    moved_geometry['geometry'] = dict(moved_geometry['geometry'])
    moved_geometry['geometry']['gap_size_m'] = 0.004

    with pytest.raises(ValueError, match='gap_size_m is declared as'):
        DERIVATION._sei_provenance(
            PIXEL_SCALE_ARCSEC, moved_geometry, vendor_path=vendor
        )
    with pytest.raises(
        ValueError, match='derived pixel_scale_arcsec is declared as'
    ):
        DERIVATION._sei_provenance(
            2.0 * PIXEL_SCALE_ARCSEC, area_report, vendor_path=vendor
        )
    with pytest.raises(ValueError, match='must carry the geometry'):
        DERIVATION._sei_provenance(
            PIXEL_SCALE_ARCSEC,
            {'value_m2': STUB_AREA_M2},
            vendor_path=vendor,
        )


def test_scene_one_closed_form_reproduces_the_qualification_flux():
    """Check the frozen qualification flux against the n=1 closed form."""
    closed_form = DERIVATION.sersic_n1_total_flux(2.0, 0.11)

    assert closed_form == pytest.approx(QUALIFICATION_TOTAL_FLUX, abs=1.0e-9)
    assert DERIVATION.verify_qualification_total_flux() == closed_form


def test_every_source_family_shares_one_continuous_normalization():
    """Pin the closed-form angular integral of all three families.

    The three families configure the same continuous angular integral,
    which is the quantity the profile normalizations control and which
    the qualification scenes froze at 0.289151264.
    """
    smooth = _scene_light('scene1_smooth_ring')
    clumpy = _scene_light('scene2_clumpy')
    cosmos = _scene_light('scene4_cosmos')

    smooth_integral = DERIVATION.source_profile_angular_integral(smooth)
    clumpy_integral = DERIVATION.source_profile_angular_integral(clumpy)
    cosmos_integral = DERIVATION.source_profile_angular_integral(cosmos)

    assert smooth_integral == pytest.approx(QUALIFICATION_TOTAL_FLUX, abs=1.0e-9)
    assert clumpy_integral == pytest.approx(QUALIFICATION_TOTAL_FLUX, rel=1.0e-7)
    assert cosmos_integral == QUALIFICATION_TOTAL_FLUX

    host_only = {**clumpy, 'clumps': []}
    host_integral = DERIVATION.source_profile_angular_integral(host_only)
    assert len(clumpy['clumps']) == NUMBER_OF_CLUMPS
    assert 1.0 - host_integral / clumpy_integral == pytest.approx(
        CLUMP_FLUX_FRACTION, rel=1.0e-7
    )

    with pytest.raises(ValueError, match='Unsupported source light type'):
        DERIVATION.source_profile_angular_integral({'type': 'Guessed'})


def _light_patch(patches, name):
    """Return one patch's light-level leaf mapping after shape checks."""
    patch = patches[name]
    assert set(patch) == {'lensing'}
    assert set(patch['lensing']) == {'source_galaxy'}
    assert set(patch['lensing']['source_galaxy']) == {'light'}
    return patch['lensing']['source_galaxy']['light']


def test_scene_patches_scale_one_leaf_per_family_and_record_both_rates():
    """Solve every scene to one detected rate and record both conventions.

    Each patch is a deep-merge config fragment touching only the single
    light-level leaf its family is linear in, so the clumpy 90/10 split
    is preserved structurally: ``flux_scale`` multiplies the host and
    every clump uniformly and the clump list is never replaced.
    """
    pytest.importorskip('autolens')

    target = PHYSICAL_TARGET_RATE_E_PER_S
    patches, details = DERIVATION.scene_flux_patches(target, PIXEL_SCALE_ARCSEC)

    assert set(patches) == set(DERIVATION.SCENE_LIGHT_BASELINES)
    assert set(details) == set(DERIVATION.SCENE_LIGHT_BASELINES)

    for name, detail in details.items():
        field = detail['normalized_field']
        light = _light_patch(patches, name)
        assert set(light) == {field}
        assert light[field] == detail[field]
        assert detail['target_rate_e_per_s'] == target
        assert detail['realized_rate_e_per_s'] == pytest.approx(
            target, rel=1.0e-12
        )
        assert detail['pixel_scale_arcsec'] == PIXEL_SCALE_ARCSEC
        assert detail['grid_shape'] == [500, 500]
        assert light[field] == pytest.approx(
            detail['baseline_value'] * detail['scale_factor'], rel=1.0e-15
        )
        assert detail['profile_angular_integral'] == pytest.approx(
            detail['baseline_profile_angular_integral']
            * detail['scale_factor'],
            rel=1.0e-15,
        )
        assert detail['profile_angular_integral'] == pytest.approx(
            PIXEL_AREA_ARCSEC2 * detail['realized_rate_e_per_s'],
            rel=DERIVATION.DISCRETE_MAPPING_TOLERANCE,
        )
        assert abs(detail['baseline_discrete_mapping_ratio'] - 1.0) < (
            DERIVATION.DISCRETE_MAPPING_TOLERANCE
        )

    for name in EXPONENTIAL_SCENES:
        assert details[name]['normalized_field'] == 'intensity'
        assert _light_patch(patches, name)['intensity'] == pytest.approx(
            CORRECTED_EXPONENTIAL_INTENSITY, rel=ANCHOR_RELATIVE_TOLERANCE
        )

    clumpy = _light_patch(patches, 'scene2_clumpy')
    assert set(clumpy) == {'flux_scale'}
    assert clumpy['flux_scale'] == pytest.approx(
        CORRECTED_CLUMPY_FLUX_SCALE, rel=ANCHOR_RELATIVE_TOLERANCE
    )

    cosmos = _light_patch(patches, 'scene4_cosmos')
    assert set(cosmos) == {'total_flux'}
    assert cosmos['total_flux'] == pytest.approx(
        CORRECTED_IMAGE_TOTAL_FLUX, rel=ANCHOR_RELATIVE_TOLERANCE
    )


def test_unlensed_render_sums_to_the_target_rate_for_every_family():
    """Render each patched family and recover the target detected rate.

    This is the source-unit contract: the observation layer reads
    rendered samples as per-pixel e-/s, so the discrete pixel sum of the
    patched unlensed source must be the derived detected rate.
    """
    pytest.importorskip('autolens')

    target = PHYSICAL_TARGET_RATE_E_PER_S
    patches, details = DERIVATION.scene_flux_patches(target, PIXEL_SCALE_ARCSEC)

    for label in ('scene1_smooth_ring', 'scene2_clumpy', 'scene4_cosmos'):
        leaf = _light_patch(patches, label)
        realized = _patched_render_sum(label, leaf)

        assert realized == pytest.approx(target, rel=1.0e-10)
        assert realized == pytest.approx(
            details[label]['realized_rate_e_per_s'], rel=1.0e-10
        )
        assert PIXEL_AREA_ARCSEC2 * realized == pytest.approx(
            details[label]['profile_angular_integral'],
            rel=DERIVATION.DISCRETE_MAPPING_TOLERANCE,
        )


def test_normalization_scaling_conserves_the_lensed_to_unlensed_ratio():
    """Hold magnification invariant under the normalization scaling.

    Every source family is linear in its normalized field, so scaling
    that leaf must move the lensed and the unlensed discrete sums by the
    same factor and leave their ratio, the magnification, untouched.
    """
    pytest.importorskip('autolens')
    from hwoslaps.lensing.generator import generate_lensing_system

    relpath = DERIVATION.SCENE_LIGHT_BASELINES['scene1_smooth_ring'][
        'scene_config'
    ]
    coarse_grid = {'shape': [200, 200], 'pixel_scale': 0.0179}
    scale = 1.0e-3
    sums = []
    for intensity in (2.0, 2.0 * scale):
        config = DERIVATION.load_scene_config(relpath)
        config['lensing']['grid'] = dict(coarse_grid)
        config['lensing']['source_galaxy']['light']['intensity'] = intensity
        lensed = float(
            np.sum(generate_lensing_system(config['lensing'], config).image)
        )
        unlensed = float(
            np.sum(
                DERIVATION.render_unlensed_source(
                    config['lensing']['source_galaxy'], config['lensing']['grid']
                )
            )
        )
        sums.append((lensed, unlensed))

    (base_lensed, base_unlensed), (scaled_lensed, scaled_unlensed) = sums

    assert scaled_unlensed / base_unlensed == pytest.approx(scale, rel=1.0e-12)
    assert scaled_lensed / base_lensed == pytest.approx(scale, rel=1.0e-12)
    assert scaled_lensed / scaled_unlensed == pytest.approx(
        base_lensed / base_unlensed, rel=1.0e-10
    )
    assert 10.0 < base_lensed / base_unlensed < 30.0


def test_pixel_area_maps_the_discrete_sum_onto_the_continuous_integral():
    """Refine the sampling and converge onto the closed-form integral.

    The configured normalization controls a continuous angular integral
    while the engine reads a discrete pixel sum. The two conventions
    differ by exactly the pixel area, so refining the grid drives
    ``pixel_area * discrete_sum`` onto the closed form while the raw sum
    itself stays near ``closed_form / pixel_area``.
    """
    pytest.importorskip('autolens')

    relpath = DERIVATION.SCENE_LIGHT_BASELINES['scene1_smooth_ring'][
        'scene_config'
    ]
    source_config = DERIVATION.load_scene_config(relpath)['lensing'][
        'source_galaxy'
    ]
    closed_form = DERIVATION.source_profile_angular_integral(
        source_config['light']
    )
    errors = []
    for factor in (4, 2, 1):
        grid_config = {
            'shape': [500 // factor, 500 // factor],
            'pixel_scale': PIXEL_SCALE_ARCSEC * factor,
        }
        pixel_area = grid_config['pixel_scale'] ** 2
        discrete_sum = float(
            np.sum(DERIVATION.render_unlensed_source(source_config, grid_config))
        )
        errors.append(abs(pixel_area * discrete_sum / closed_form - 1.0))
        if factor == 1:
            assert discrete_sum == pytest.approx(
                closed_form / PIXEL_AREA_ARCSEC2, rel=1.0e-3
            )

    assert errors[0] < 5.0e-3
    assert errors[1] <= errors[0] + 1.0e-12
    assert errors[2] <= errors[1] + 1.0e-12
    assert errors[2] < 1.0e-3


def test_scene_flux_patches_reject_a_pixel_scale_the_scenes_do_not_sample():
    """Fail closed when the reference pixel scale leaves the scene grid."""
    with pytest.raises(ValueError, match='per-pixel rates in this artifact'):
        DERIVATION.scene_flux_patches(PHYSICAL_TARGET_RATE_E_PER_S, 0.01)

    with pytest.raises(ValueError, match='must be a positive finite number'):
        DERIVATION.scene_flux_patches(0.0, PIXEL_SCALE_ARCSEC)


def test_throughput_is_applied_exactly_once():
    """Scale the system QE and see every derived rate move once."""
    flux_density = 3631.0 * 10.0 ** (-0.4 * 24.3)
    photons = flux_density * 1.0e-26 / 6.62607015e-34 * math.log(550.0 / 450.0)
    expected_rate = photons * STUB_AREA_M2 * 0.21

    baseline = _document()
    doubled = _document(system_qe=0.42)

    photometry = baseline['metadata']['source_photometry']
    assert photometry['detected_rate_e_per_s'] == pytest.approx(
        expected_rate, rel=1.0e-12
    )
    assert baseline['observation']['throughput'] == 1.0
    assert doubled['observation']['throughput'] == 1.0
    assert baseline['metadata']['throughput_chain']['value'] == 0.21
    assert baseline['metadata']['throughput_chain']['applied_to'] == (
        'the derived source and sky rates in this artifact only'
    )

    baseline_normalization = baseline['metadata']['source_normalization_details']
    doubled_normalization = doubled['metadata']['source_normalization_details']
    assert doubled_normalization['target_rate_e_per_s'] == pytest.approx(
        2.0 * baseline_normalization['target_rate_e_per_s'], rel=1.0e-12
    )
    assert doubled['observation']['detector']['sky_background'] == pytest.approx(
        2.0 * baseline['observation']['detector']['sky_background'], rel=1.0e-12
    )
    for label, detail in baseline_normalization['scene_details'].items():
        field = detail['normalized_field']
        doubled_detail = doubled_normalization['scene_details'][label]
        assert doubled_detail[field] == pytest.approx(
            2.0 * detail[field], rel=1.0e-12
        )


def test_pupil_mask_area_brackets_the_gapless_hexagon_area():
    """Integrate the real pupil mask and cross-check the hexagon formula."""
    pytest.importorskip('hcipy')

    geometry = DERIVATION.load_pupil_geometry()
    area = DERIVATION.collecting_area_m2(geometry)
    report = DERIVATION.collecting_area_report(area, geometry)
    hexagon = 19 * (3.0 * math.sqrt(3.0) / 8.0) * 1.65 ** 2

    low, high = DERIVATION.AREA_RATIO_BOUNDS
    assert report['num_segments'] == 19
    assert report['gapless_hexagon_area_m2'] == pytest.approx(hexagon, rel=1.0e-12)
    assert low <= area / hexagon <= high
    assert area == pytest.approx(33.6, abs=0.05)
    assert report['gap_loss_fraction'] == pytest.approx(1.0 - area / hexagon, rel=1.0e-12)

    reference = _document(area_report=report)

    assert reference['metadata']['collecting_area']['value_m2'] == area


def test_sed_modes_record_both_magnitudes_and_the_mode():
    """Exercise every SED mode and its provenance record."""
    flat = _document()['metadata']['source_photometry']
    assert flat['sed_mode'] == 'flat_fnu'
    assert flat['input_magnitude_ab'] == 24.3
    assert flat['derived_magnitude_ab'] == 24.3
    assert flat['sed_mode_description'] == 'flat-f_nu reference assumption'

    colored = _document(sed_mode='declared_color', color_ab=0.4)
    colored_photometry = colored['metadata']['source_photometry']
    assert colored_photometry['sed_mode'] == 'declared_color'
    assert colored_photometry['color_ab'] == 0.4
    assert colored_photometry['input_magnitude_ab'] == 24.3
    assert colored_photometry['derived_magnitude_ab'] == pytest.approx(
        24.7, rel=1.0e-12
    )

    vband = _document(sed_mode='vband_photometry', mag_vband=23.9)
    vband = vband['metadata']['source_photometry']
    assert vband['sed_mode'] == 'vband_photometry'
    assert vband['measured_vband_magnitude_ab'] == 23.9
    assert vband['input_magnitude_ab'] == 24.3
    assert vband['derived_magnitude_ab'] == 23.9

    assert DERIVATION.source_mag_hri_v(24.3, 'flat_fnu') == 24.3
    with pytest.raises(ValueError, match='--color-ab'):
        DERIVATION.source_mag_hri_v(24.3, 'declared_color')
    with pytest.raises(ValueError, match='--source-mag-vband'):
        DERIVATION.source_mag_hri_v(24.3, 'vband_photometry')
    with pytest.raises(ValueError, match='Unknown sed-mode'):
        DERIVATION.source_mag_hri_v(24.3, 'guessed_color')


def test_declared_color_flows_through_the_corrected_scene_mapping():
    """Route a fainter SED mode through the same render-based mapping."""
    baseline = _document()['metadata']['source_normalization_details']
    colored = _document(sed_mode='declared_color', color_ab=0.5)
    colored = colored['metadata']['source_normalization_details']

    ratio = 10.0 ** (-0.4 * 0.5)
    assert colored['target_rate_e_per_s'] == pytest.approx(
        ratio * baseline['target_rate_e_per_s'], rel=1.0e-12
    )
    for label, detail in colored['scene_details'].items():
        field = detail['normalized_field']
        baseline_detail = baseline['scene_details'][label]
        assert detail['baseline_discrete_sum'] == (
            baseline_detail['baseline_discrete_sum']
        )
        assert detail[field] == pytest.approx(
            ratio * baseline_detail[field], rel=1.0e-12
        )
        assert detail['realized_rate_e_per_s'] == pytest.approx(
            colored['target_rate_e_per_s'], rel=1.0e-12
        )


def test_artifact_round_trips_and_validates_as_an_observation_block(tmp_path):
    """Write, reload, and validate the artifact against the engine schema."""
    validation = pytest.importorskip('hwoslaps.config.validation')

    document = _document()
    path = tmp_path / 'hwo_eac1_hri_reference_v1.yaml'
    DERIVATION.write_reference_document(path, document)

    with path.open('r', encoding='utf-8') as stream:
        reloaded = yaml.safe_load(stream)

    assert set(reloaded) == {'observation', 'source_normalization', 'metadata'}
    assert reloaded == document
    reference = reloaded
    observation = reference['observation']
    validation.validate_observation_config(observation)
    assert set(reference['source_normalization']) == set(
        DERIVATION.SCENE_LIGHT_BASELINES
    )

    assert observation['exposure_time'] == 2000.0
    assert observation['throughput'] == 1.0
    assert observation['detector']['gain'] == 1.0
    assert observation['detector']['dark_current'] == 0.002
    assert set(reference['metadata']['citations']) == {
        'S1', 'S3', 'S4', 'S6', 'L1', 'SEI'
    }

    with pytest.raises(ValueError, match='Refusing to overwrite'):
        DERIVATION.write_reference_document(path, document)
    DERIVATION.write_reference_document(path, document, force=True)


def test_main_refuses_to_overwrite_an_existing_artifact(tmp_path):
    """Refuse the CLI write before any derivation work happens."""
    path = tmp_path / 'hwo_eac1_hri_reference_v1.yaml'
    path.write_text('observing_reference: {}\n', encoding='utf-8')

    with pytest.raises(ValueError, match='Refusing to overwrite'):
        DERIVATION.main(['--output', str(path)])


def test_metadata_declares_the_unlensed_normalization_plane():
    """Pin the magnification-plane hygiene declarations."""
    photometry = _document()['metadata']['source_photometry']

    assert 'unlensed' in photometry['normalization_plane']
    assert 'magnification' in photometry['normalization_plane']
    assert 'magnification-corrected' in photometry['input_magnitude_note']
    assert photometry['input_band'] == 'F814W'
    assert photometry['input_citation'] == 'L1'


def test_metadata_separates_the_profile_and_detector_conventions():
    """Keep both normalization concepts named and mapped in provenance."""
    metadata = _document()['metadata']
    normalization = metadata['source_normalization_details']

    assert normalization['pixel_area_arcsec2'] == pytest.approx(
        PIXEL_AREA_ARCSEC2, rel=1.0e-15
    )
    assert normalization['pixel_scale_arcsec'] == PIXEL_SCALE_ARCSEC
    assert 'never detected totals' in normalization['convention']
    assert 'not itself a detected rate' in normalization['convention']
    assert 'discrete pixel sum' in normalization['convention']
    assert normalization['qualification_profile_angular_integral'] == (
        QUALIFICATION_TOTAL_FLUX
    )
    assert 'not a detected electron rate' in normalization['qualification_note']
    assert normalization['sersic_b1'] == 1.6783886549215685
    assert 'production render' in metadata['observation_model_semantics']

    for detail in normalization['scene_details'].values():
        assert detail['target_rate_e_per_s'] == (
            normalization['target_rate_e_per_s']
        )
        assert detail['realized_rate_e_per_s'] > 0.0
        assert detail['profile_angular_integral'] == pytest.approx(
            detail['baseline_profile_angular_integral']
            * detail['scale_factor'],
            rel=1.0e-15,
        )
        assert detail['profile_angular_integral'] == pytest.approx(
            PIXEL_AREA_ARCSEC2 * detail['realized_rate_e_per_s'],
            rel=DERIVATION.DISCRETE_MAPPING_TOLERANCE,
        )
        assert detail['baseline_profile_angular_integral'] == pytest.approx(
            QUALIFICATION_TOTAL_FLUX, rel=1.0e-7
        )


def test_artifact_feeds_an_s1_lite_freeze_against_the_real_scenes(tmp_path):
    """Freeze an S1-lite campaign from the artifact and the real scenes.

    This is the cross-contract guard: the artifact's top-level shape and
    per-scene patch fragments must stay exactly what the S1-lite
    observing-reference loader deep-merges into staged job configs.
    """
    campaign = pytest.importorskip('hwoslaps.campaign')

    document = _document()
    reference_path = tmp_path / 'hwo_eac1_hri_reference_v1.yaml'
    DERIVATION.write_reference_document(reference_path, document)

    scene_labels = sorted(DERIVATION.SCENE_LIGHT_BASELINES)
    manifest = {
        'campaign': {
            'name': 'contract_probe',
            'output_root': str(tmp_path / 'campaign_root'),
            'runner_command': ['python', 'runner.py', '-c', '{config}'],
            'base_scene_configs': {
                label: str(
                    PROJECT_ROOT
                    / DERIVATION.SCENE_LIGHT_BASELINES[label]['scene_config']
                )
                for label in scene_labels
            },
            'observing_reference': str(reference_path),
            'seed_policy': {'note': 'contract probe, no draws'},
            'expected_job_count': len(scene_labels),
            'jobs': [
                {
                    'job_id': f'probe_{label}',
                    'scene': label,
                    'overrides': {
                        'psf': {'kernel': {'shape_native': [999, 999]}}
                    },
                }
                for label in scene_labels
            ],
        }
    }
    manifest_path = tmp_path / 'manifest.yaml'
    with manifest_path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)

    campaign.freeze_campaign(manifest_path)

    details = document['metadata']['source_normalization_details']
    for label in scene_labels:
        staged_path = (
            tmp_path / 'campaign_root' / 'configs' / f'probe_{label}.yaml'
        )
        with staged_path.open('r', encoding='utf-8') as stream:
            staged = yaml.safe_load(stream)
        staged_observation = staged['observation']
        for key, value in document['observation'].items():
            if key == 'detector':
                for name, rate in value.items():
                    assert staged_observation['detector'][name] == rate
            else:
                assert staged_observation[key] == value
        assert staged['psf']['kernel']['shape_native'] == [999, 999]
        light = staged['lensing']['source_galaxy']['light']
        detail = details['scene_details'][label]
        field = detail['normalized_field']
        assert light[field] == detail[field]
        if detail['light_type'] == 'Clumpy':
            assert light['host']['intensity'] == 1.8
            assert [clump['intensity'] for clump in light['clumps']] == (
                [2.0166667] * NUMBER_OF_CLUMPS
            )
        elif detail['light_type'] == 'Image':
            assert light['flux_scale'] == 1.0


def test_default_normalization_mode_rebuilds_the_committed_reference(
    monkeypatch,
):
    """Rebuild the committed artifact and demand an exact match.

    This is the value-for-value guard on the frozen default: every value
    the committed reference carries, and every key it does not, must come
    back unchanged from the current code. It is deliberately NOT a
    byte-for-byte guard, because ``generation_date`` stamps the current
    date and so a rebuild on any later day differs from the committed
    artifact in that one field alone.
    """
    _require_vendored_sei()
    committed = _committed_reference()
    _freeze_baseline_sums(monkeypatch)
    photometry = committed['metadata']['source_photometry']
    detector = committed['metadata']['detector']

    rebuilt = DERIVATION.build_reference_document(
        area_report=committed['metadata']['collecting_area'],
        pixel_scale_arcsec=detector['pixel_scale_arcsec'],
        sed_mode=photometry['sed_mode'],
        color_ab=photometry['color_ab'],
        mag_vband=photometry['measured_vband_magnitude_ab'],
        exposure_s=committed['observation']['exposure_time'],
        system_qe=committed['metadata']['throughput_chain']['value'],
        sky_mag=detector['sky_surface_brightness_ab_per_arcsec2'],
    )

    assert rebuilt['observation'] == committed['observation']
    assert rebuilt['source_normalization'] == committed['source_normalization']
    rebuilt['metadata'].pop('generation_date')
    committed['metadata'].pop('generation_date')
    assert rebuilt['metadata'] == committed['metadata']
    assert DERIVATION.SCRIPT_VERSION == '3'


def test_default_normalization_mode_is_the_explicit_intrinsic_rate_mode(
    monkeypatch,
):
    """Pin the default mode and its identity with the explicit request."""
    _require_vendored_sei()
    _freeze_baseline_sums(monkeypatch)

    implicit = DERIVATION.build_reference_document(
        area_report=_stub_area_report(),
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        sed_mode='flat_fnu',
        color_ab=None,
        mag_vband=None,
        exposure_s=2000.0,
        system_qe=0.21,
        sky_mag=23.0,
    )
    explicit = DERIVATION.build_reference_document(
        area_report=_stub_area_report(),
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        sed_mode='flat_fnu',
        color_ab=None,
        mag_vband=None,
        exposure_s=2000.0,
        system_qe=0.21,
        sky_mag=23.0,
        normalization_mode='intrinsic_rate',
    )

    implicit['metadata'].pop('generation_date')
    explicit['metadata'].pop('generation_date')
    assert implicit == explicit
    assert DERIVATION.NORMALIZATION_MODES == ('intrinsic_rate', 'arc_snr')
    parser = DERIVATION._build_parser()
    assert parser.parse_args([]).normalization_mode == 'intrinsic_rate'
    assert parser.parse_args([]).target_arc_snr is None

    normalization = implicit['metadata']['source_normalization_details']
    assert 'normalization_mode' not in normalization
    assert 'requested_arc_snr' not in normalization
    for detail in normalization['scene_details'].values():
        assert 'arc_snr_solution' not in detail


def test_blank_pixel_variance_reproduces_the_units_audit():
    """Reproduce the audit's blank-pixel variance from the artifact."""
    observation = _committed_reference()['observation']
    detector = observation['detector']
    expected = (
        (detector['sky_background'] + detector['dark_current'])
        * observation['exposure_time']
        + detector['read_noise'] ** 2
    )

    variance = DERIVATION.blank_pixel_variance_e2(observation)

    assert variance == pytest.approx(expected, rel=1.0e-15)
    assert variance == pytest.approx(
        COMMITTED_BLANK_PIXEL_VARIANCE_E2, rel=1.0e-5
    )
    assert math.sqrt(variance) == pytest.approx(3.01671, rel=1.0e-5)

    with pytest.raises(ValueError, match='must be a positive finite number'):
        DERIVATION.blank_pixel_variance_e2(
            {**observation, 'exposure_time': 0.0}
        )
    with pytest.raises(ValueError, match='must be a non-negative finite'):
        DERIVATION.blank_pixel_variance_e2(
            {**observation, 'detector': {**detector, 'read_noise': -1.0}}
        )
    with pytest.raises(ValueError, match='detector block'):
        DERIVATION.blank_pixel_variance_e2({'exposure_time': 2000.0})


def test_integrated_source_snr_follows_the_per_pixel_convention():
    """Match the per-pixel source S/N definition and reject bad maps."""
    variance = COMMITTED_BLANK_PIXEL_VARIANCE_E2
    electrons = _synthetic_source_profile() * 5.0e4
    per_pixel = electrons / np.sqrt(electrons + variance)

    arc_snr = DERIVATION.integrated_source_snr(electrons, variance)

    assert arc_snr == pytest.approx(
        float(np.sqrt(np.sum(per_pixel ** 2))), rel=1.0e-14
    )
    assert arc_snr > 0.0

    with pytest.raises(ValueError, match='must not be empty'):
        DERIVATION.integrated_source_snr(np.array([]), variance)
    with pytest.raises(ValueError, match='must be finite'):
        DERIVATION.integrated_source_snr(np.array([np.nan]), variance)
    with pytest.raises(ValueError, match='must be a positive finite number'):
        DERIVATION.integrated_source_snr(electrons, 0.0)
    with pytest.raises(ValueError, match='not a physical electron map'):
        DERIVATION.integrated_source_snr(np.array([-2.0 * variance]), variance)


def test_achieved_arc_snr_increases_monotonically_with_the_scale_factor():
    """Hold the monotonicity the arc S/N solve depends on.

    The achieved value rises linearly while the blank-pixel variance
    dominates and as the square root once source shot noise does, so it
    is strictly increasing across the whole range and both limits are
    recovered.
    """
    variance = COMMITTED_BLANK_PIXEL_VARIANCE_E2
    profile = _synthetic_source_profile()
    scales = np.logspace(-6.0, 6.0, 49)

    achieved = [
        DERIVATION.integrated_source_snr(profile * scale * 1.0e4, variance)
        for scale in scales
    ]

    assert all(
        later > earlier for earlier, later in zip(achieved, achieved[1:])
    )
    faint = DERIVATION.integrated_source_snr(profile * 1.0e-4, variance)
    fainter = DERIVATION.integrated_source_snr(profile * 1.0e-5, variance)
    assert faint / fainter == pytest.approx(10.0, rel=1.0e-6)
    bright = DERIVATION.integrated_source_snr(profile * 1.0e10, variance)
    brighter = DERIVATION.integrated_source_snr(profile * 1.0e12, variance)
    assert brighter / bright == pytest.approx(10.0, rel=1.0e-4)


def test_arc_snr_solver_hits_its_target_and_records_its_effort():
    """Solve a monotone response and pin the recorded provenance."""
    variance = COMMITTED_BLANK_PIXEL_VARIANCE_E2
    profile = _synthetic_source_profile() * 4.0e4

    def response(scale):
        return DERIVATION.integrated_source_snr(profile * scale, variance)

    target = 2.5 * response(1.0)

    scale, record = DERIVATION.solve_arc_snr_scale(response, 1.0, target)

    assert response(scale) == pytest.approx(target, rel=1.0e-6)
    assert record['requested_arc_snr'] == target
    assert record['achieved_arc_snr'] == pytest.approx(target, rel=1.0e-6)
    assert record['relative_residual'] <= 1.0e-6
    assert record['relative_tolerance'] == 1.0e-6
    assert record['initial_scale_factor'] == 1.0
    assert record['bracket_low_scale_factor'] <= scale
    assert scale <= record['bracket_high_scale_factor']
    assert record['bracket_steps'] >= 1
    assert record['solver_iterations'] >= 1
    assert record['forward_model_evaluations'] >= record['solver_iterations']
    assert record['solver'].startswith('scipy.optimize.brentq')

    exact_scale, exact_record = DERIVATION.solve_arc_snr_scale(
        lambda _: 7.0, 0.25, 7.0
    )
    assert exact_scale == pytest.approx(0.25, rel=1.0e-12)
    assert exact_record['bracket_steps'] == 0
    assert exact_record['solver_iterations'] == 0
    assert exact_record['relative_residual'] == 0.0
    assert exact_record['forward_model_evaluations'] == 2


def test_arc_snr_solver_raises_rather_than_returning_a_failed_bracket():
    """Fail loudly when no scale factor reaches the requested value."""
    def saturating(scale):
        return 12.0 - 1.0 / (1.0 + float(scale))

    with pytest.raises(ValueError, match='is not bracketed by scale'):
        DERIVATION.solve_arc_snr_scale(saturating, 1.0, 25.0)

    def vanishing(scale):
        return 1.0e-3 + 1.0 / (1.0 + 1.0 / float(scale))

    with pytest.raises(ValueError, match='is not bracketed by scale'):
        DERIVATION.solve_arc_snr_scale(vanishing, 1.0, 1.0e-6)

    def unphysical(scale):
        return -1.0 * float(scale)

    with pytest.raises(ValueError, match='must be positive and finite'):
        DERIVATION.solve_arc_snr_scale(unphysical, 1.0, 10.0)

    with pytest.raises(ValueError, match='target_arc_snr must be a positive'):
        DERIVATION.solve_arc_snr_scale(saturating, 1.0, 0.0)
    with pytest.raises(ValueError, match='initial_scale must be a positive'):
        DERIVATION.solve_arc_snr_scale(saturating, 0.0, 5.0)


def test_normalization_mode_validation_rejects_mismatched_arguments():
    """Reject every mode and argument pairing that cannot be honoured."""
    assert DERIVATION.validate_normalization_mode('arc_snr', 300.0) == (
        'arc_snr'
    )
    assert DERIVATION.validate_normalization_mode(
        'intrinsic_rate', None
    ) == 'intrinsic_rate'

    with pytest.raises(ValueError, match='Unknown normalization-mode'):
        DERIVATION.validate_normalization_mode('equal_flux', None)
    with pytest.raises(ValueError, match='does not accept --target-arc-snr'):
        DERIVATION.validate_normalization_mode('intrinsic_rate', 300.0)
    with pytest.raises(ValueError, match='requires --target-arc-snr'):
        DERIVATION.validate_normalization_mode('arc_snr', None)
    with pytest.raises(ValueError, match=r'--target-arc-snr must be a pos'):
        DERIVATION.validate_normalization_mode('arc_snr', 0.0)
    with pytest.raises(ValueError, match=r'--target-arc-snr must be a pos'):
        DERIVATION.validate_normalization_mode('arc_snr', float('nan'))
    with pytest.raises(ValueError, match=r'--target-arc-snr must be a pos'):
        DERIVATION.validate_normalization_mode('arc_snr', -12.0)

    with pytest.raises(ValueError, match='does not accept an observation'):
        DERIVATION.scene_flux_patches(
            PHYSICAL_TARGET_RATE_E_PER_S,
            PIXEL_SCALE_ARCSEC,
            observation=_committed_reference()['observation'],
        )
    with pytest.raises(ValueError, match='arc_snr requires an observation'):
        DERIVATION.scene_flux_patches(
            PHYSICAL_TARGET_RATE_E_PER_S,
            PIXEL_SCALE_ARCSEC,
            normalization_mode='arc_snr',
            target_arc_snr=300.0,
        )
    with pytest.raises(ValueError, match='Unknown normalization-mode'):
        DERIVATION.build_reference_document(
            area_report=_stub_area_report(),
            pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
            sed_mode='flat_fnu',
            color_ab=None,
            mag_vband=None,
            exposure_s=2000.0,
            system_qe=0.21,
            sky_mag=23.0,
            normalization_mode='equal_flux',
        )


def test_arc_snr_mode_equalizes_the_arc_and_records_both_conventions(
    monkeypatch, tmp_path
):
    """Solve every scene to one arc S/N and keep both conventions.

    The stand-in forward model gives each scene its own magnification,
    so equal arc S/N means unequal intrinsic rates. Both numbers, the
    request, and the solver effort must survive into the artifact.
    """
    _require_vendored_sei()
    baseline_sums = _freeze_baseline_sums(monkeypatch)
    magnifications = _synthetic_forward_model(monkeypatch, baseline_sums)
    target = 250.0

    document = DERIVATION.build_reference_document(
        area_report=_stub_area_report(),
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        sed_mode='flat_fnu',
        color_ab=None,
        mag_vband=None,
        exposure_s=2000.0,
        system_qe=0.21,
        sky_mag=23.0,
        normalization_mode='arc_snr',
        target_arc_snr=target,
    )

    normalization = document['metadata']['source_normalization_details']
    assert normalization['normalization_mode'] == 'arc_snr'
    assert normalization['requested_arc_snr'] == target
    assert normalization['arc_snr_blank_pixel_variance_e2'] == (
        DERIVATION.blank_pixel_variance_e2(document['observation'])
    )
    assert 'SNR_arc' in normalization['arc_snr_formula']
    assert 'subhalo is disabled' in normalization['arc_snr_forward_model']
    assert 'photometric anchor' in normalization['target_rate_note']

    realized = {}
    for label, detail in normalization['scene_details'].items():
        solution = detail['arc_snr_solution']
        field = detail['normalized_field']
        assert detail['normalization_mode'] == 'arc_snr'
        assert solution['requested_arc_snr'] == target
        assert solution['achieved_arc_snr'] == pytest.approx(
            target, rel=1.0e-6
        )
        assert solution['relative_residual'] <= 1.0e-6
        assert solution['solver_iterations'] >= 1
        assert solution['forward_model_evaluations'] >= 1
        assert detail['realized_rate_e_per_s'] == pytest.approx(
            detail['baseline_discrete_sum'] * detail['scale_factor'],
            rel=1.0e-15,
        )
        patch = document['source_normalization'][label]
        light = patch['lensing']['source_galaxy']['light']
        assert light[field] == detail[field]
        realized[detail['scene_config']] = detail['realized_rate_e_per_s']

    assert len(set(realized.values())) == len(realized)
    ranked = sorted(realized, key=lambda path: magnifications[path])
    rates = [realized[path] for path in ranked]
    assert rates == sorted(rates, reverse=True)

    path = tmp_path / 'arc_snr_reference.yaml'
    DERIVATION.write_reference_document(path, document)
    with path.open('r', encoding='utf-8') as stream:
        assert yaml.safe_load(stream) == document
    DERIVATION._print_arc_snr_summary(normalization)


def test_arc_snr_mode_leaves_the_observation_block_untouched(monkeypatch):
    """Change only the source normalization, never the detector."""
    _require_vendored_sei()
    baseline_sums = _freeze_baseline_sums(monkeypatch)
    _synthetic_forward_model(monkeypatch, baseline_sums)
    parameters = {
        'area_report': _stub_area_report(),
        'pixel_scale_arcsec': PIXEL_SCALE_ARCSEC,
        'sed_mode': 'flat_fnu',
        'color_ab': None,
        'mag_vband': None,
        'exposure_s': 2000.0,
        'system_qe': 0.21,
        'sky_mag': 23.0,
    }

    intrinsic = DERIVATION.build_reference_document(**parameters)
    arc = DERIVATION.build_reference_document(
        normalization_mode='arc_snr', target_arc_snr=250.0, **parameters
    )

    assert arc['observation'] == intrinsic['observation']
    assert arc['metadata']['source_photometry'] == (
        intrinsic['metadata']['source_photometry']
    )
    assert set(arc['source_normalization']) == set(
        intrinsic['source_normalization']
    )
    for label, patch in arc['source_normalization'].items():
        baseline_patch = intrinsic['source_normalization'][label]
        assert set(patch['lensing']['source_galaxy']['light']) == set(
            baseline_patch['lensing']['source_galaxy']['light']
        )
        assert patch != baseline_patch


def test_arc_snr_cli_requires_its_target_before_any_derivation(tmp_path):
    """Refuse an arc_snr run without a target ahead of the pupil work."""
    path = tmp_path / 'hwo_eac1_hri_reference_v1.yaml'

    with pytest.raises(ValueError, match='requires --target-arc-snr'):
        DERIVATION.main(
            ['--output', str(path), '--normalization-mode', 'arc_snr']
        )
    with pytest.raises(ValueError, match='does not accept --target-arc-snr'):
        DERIVATION.main(['--output', str(path), '--target-arc-snr', '300.0'])

    assert not path.exists()


def test_production_forward_model_matches_the_engine_noise_map():
    """Observe a real scene and reproduce the engine's own source S/N.

    This is the arc S/N unit contract against the production pipeline:
    the electrons this module hands the solver, divided by the engine
    noise map, must integrate to exactly the value the closed formula
    gives. The committed artifact supplies both the observation block
    and the scene normalization, so the check runs on the frozen
    reference rather than on invented numbers.
    """
    pytest.importorskip('autolens')
    pytest.importorskip('hcipy')
    noise_models = pytest.importorskip('hwoslaps.observation.noise_models')

    reference = _committed_reference()
    observation = reference['observation']
    label = 'scene1_smooth_ring'
    relpath = DERIVATION.SCENE_LIGHT_BASELINES[label]['scene_config']
    patch = reference['source_normalization'][label]
    detail = reference['metadata']['source_normalization_details'][
        'scene_details'
    ][label]
    variance = DERIVATION.blank_pixel_variance_e2(observation)

    electrons = DERIVATION.scene_source_electrons(relpath, observation, patch)

    assert electrons.shape == tuple(detail['grid_shape'])
    assert np.all(np.isfinite(electrons))
    noise_adu = noise_models.create_noise_map(
        source_eps=np.maximum(electrons, 0.0) / observation['exposure_time'],
        exposure_time=observation['exposure_time'],
        detector_config=observation['detector'],
    )
    engine_snr = electrons / observation['detector']['gain'] / noise_adu
    arc_snr = DERIVATION.integrated_source_snr(electrons, variance)

    assert arc_snr == pytest.approx(
        float(np.sqrt(np.sum(engine_snr ** 2))), rel=1.0e-12
    )
    magnification = float(np.sum(electrons)) / (
        detail['realized_rate_e_per_s'] * observation['exposure_time']
    )
    assert 10.0 < magnification < 30.0
    assert AUDIT_ARC_SNR / 3.0 < arc_snr < AUDIT_ARC_SNR * 6.0


def test_arc_snr_solve_converges_through_the_production_forward_model():
    """Solve one real scene end to end and hit the target to 1e-6.

    The target is set from the achieved value at the intrinsic-rate
    scale factor, so the bracket search closes in one step and the solve
    stays affordable while still exercising the real lensing render, the
    real PSF convolution, and Brent's method on the log scale.
    """
    pytest.importorskip('autolens')
    pytest.importorskip('hcipy')

    reference = _committed_reference()
    observation = reference['observation']
    label = 'scene1_smooth_ring'
    baseline = DERIVATION.SCENE_LIGHT_BASELINES[label]
    relpath = baseline['scene_config']
    field = DERIVATION.SCENE_NORMALIZED_FIELD[baseline['light_type']]
    light = DERIVATION.load_scene_config(relpath)['lensing'][
        'source_galaxy'
    ]['light']
    detail = reference['metadata']['source_normalization_details'][
        'scene_details'
    ][label]

    response = DERIVATION.scene_arc_snr_response(
        relpath, field, light[field], observation
    )
    initial_scale = detail['scale_factor']
    target = 1.5 * response(initial_scale)

    scale, record = DERIVATION.solve_arc_snr_scale(
        response, initial_scale, target
    )

    assert scale > initial_scale
    assert record['achieved_arc_snr'] == pytest.approx(target, rel=1.0e-6)
    assert record['relative_residual'] <= 1.0e-6
    assert record['bracket_steps'] == 1
    assert record['solver_iterations'] >= 1
    assert record['bracket_low_scale_factor'] == pytest.approx(
        initial_scale, rel=1.0e-14
    )
    assert record['bracket_high_scale_factor'] == pytest.approx(
        initial_scale * DERIVATION.ARC_SNR_BRACKET_FACTOR, rel=1.0e-14
    )
