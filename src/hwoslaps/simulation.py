"""Expected and noisy observations with separate scene and detector seeds."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

from .config.schema import ConfigSource, resolve_config
from .identity import validate_file_manifest, validate_loaded_file
from .observation.normalization import resolve_observing
from .observation.observation import Observation, observe
from .scene.builder import build_scene, native_sampling_variation
from .scene.cosmology import Cosmology
from .scene.halos import Halo
from .scene.image_source import load_image_asset
from .scene.perturbers import realize_perturbers
from .spectra.bandpass import build_bandpass
from .fisher.psf_pair import bind_truth, truth_provider, validate_loaded_psf_files

if TYPE_CHECKING:
    from .fisher.api import PreparedForecast

__all__ = ["simulate"]


def _validate_subhalo(subhalo: Halo | None, scene, cosmology: Cosmology) -> None:
    if subhalo is None:
        return
    if not isinstance(subhalo, Halo):
        raise TypeError("subhalo must be a Halo or None")
    expected_redshift = scene.lens.redshift if scene.subhalo_redshift is None else scene.subhalo_redshift
    if subhalo.redshift != expected_redshift:
        raise ValueError(f"subhalo redshift {subhalo.redshift} differs from the hypothesis redshift {expected_redshift}")
    if subhalo.source_redshift != scene.source.redshift or subhalo.cosmology != cosmology:
        raise ValueError("subhalo source redshift and cosmology must match the scene")


def simulate(source: ConfigSource | PreparedForecast, *, subhalo: Halo | None, noise_seed: int | None,
             base_dir: PathLike[str] | None = None) -> Observation:
    """Observe a supplied halo or a smooth control; ``None`` seed keeps the expectation."""
    from .fisher.api import PreparedForecast

    if isinstance(source, PreparedForecast):
        source.validate_identity()
        _validate_subhalo(subhalo, source.scene.spec, source.scene.cosmology)
        if subhalo is None:
            expected = source.observation
        else:
            scene = source.renderer.scene(subhalo=subhalo)
            expected = observe(scene, source.psfs.truth_kernels, source.observation.exposure,
                               config_digest=source.record["config_digest"], photometry=source.observation.photometry,
                               sampling=source.observation.sampling)
    else:
        config = resolve_config(source, base_dir=base_dir)
        identity = config.capture_identity()
        config_digest = identity["config_digest"]
        manifest = identity["file_digests"]
        bandpass = None if config.instrument.bandpass is None else build_bandpass(config.instrument.bandpass)
        if bandpass is not None:
            for path, digest in bandpass.file_digests.items():
                validate_loaded_file(path, digest, manifest)
        paths = dict.fromkeys(str(component.values["asset_path"]) for galaxy in (config.scene.lens, config.scene.source)
                              for component in galaxy.light if component.type == "Image")
        assets = {path: load_image_asset(path) for path in paths}
        for path, asset in assets.items():
            validate_loaded_file(path, asset.digest, manifest)
        cosmology = Cosmology(config.cosmology)
        provider = truth_provider(config, bandpass=bandpass)
        setup = resolve_observing(config.scene, config.instrument, config.observation, truth=provider,
                                  bandpass=bandpass, assets=assets, expected_file_digests=manifest)
        for path, digest in setup.file_digests.items():
            validate_loaded_file(path, digest, manifest)
        _validate_subhalo(subhalo, setup.scene, cosmology)
        kernels, _ = bind_truth(provider, setup.scene, setup.instrument, loaded_seds=setup.loaded_seds)
        perturbers = realize_perturbers(setup.scene, cosmology, seed=config.seed)
        validate_loaded_psf_files(provider, kernels, manifest)
        smooth = build_scene(setup.scene, cosmology, subhalo=None, perturbers=perturbers, assets=assets, loaded_seds=setup.loaded_seds)
        sampling = native_sampling_variation(smooth)
        scene = smooth if subhalo is None else build_scene(setup.scene, cosmology, subhalo=subhalo,
                                                         perturbers=perturbers, assets=assets, loaded_seds=setup.loaded_seds)
        expected = observe(scene, kernels, setup.exposure, config_digest=config_digest,
                           photometry=setup.photometry, sampling=sampling)
        validate_file_manifest(manifest)
    return expected if noise_seed is None else expected.draw(noise_seed)
