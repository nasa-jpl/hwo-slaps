"""Offline tests for the HWO EAC1 HRI reference derivation script."""

from __future__ import annotations

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


def _stub_area_report():
    """Return a minimal collecting-area report for offline document tests."""
    return {'value_m2': STUB_AREA_M2, 'method': 'offline test stub'}


def _document(**overrides):
    """Build one reference document without touching the pupil model."""
    pytest.importorskip('autolens')

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
    assert DERIVATION.effective_read_noise(2.5, 2) == 2.5 * math.sqrt(2.0)
    assert DERIVATION.effective_read_noise(2.5, 2) == 3.5355339059327378

    reference = _document()
    detector = reference['observation']['detector']
    provenance = reference['metadata']['detector']

    assert detector['read_noise'] == 3.5355339059327378
    assert provenance['read_noise_per_read_e'] == 2.5
    assert provenance['n_reads'] == 2
    assert detector['read_noise'] != provenance['read_noise_per_read_e']
    assert provenance['effective_read_noise_formula'] == 'per_read_e * sqrt(n_reads)'


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
    assert set(reference['metadata']['citations']) == {'S1', 'S3', 'S4', 'S6', 'L1'}

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
