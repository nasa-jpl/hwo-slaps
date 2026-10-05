"""Expected and noisy observations with separate scene and detector seeds."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

from .config.schema import ConfigSource, resolve_config
from .observation.normalization import resolve_observing
from .observation.observation import Observation, observe
from .scene.builder import build_scene, native_sampling_variation
from .scene.cosmology import Cosmology
from .scene.halos import Halo
from .scene.image_source import load_image_asset
from .scene.perturbers import realize_perturbers
from .fisher.psf_pair import bind_truth, truth_provider

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
        config_digest = config.digest()
        cosmology = Cosmology(config.cosmology)
        provider = truth_provider(config)
        setup = resolve_observing(config.scene, config.instrument, config.observation, truth=provider)
        _validate_subhalo(subhalo, setup.scene, cosmology)
        kernels, _ = bind_truth(provider, setup.scene, setup.instrument)
        perturbers = realize_perturbers(setup.scene, cosmology, seed=config.seed)
        paths = dict.fromkeys(str(component.values["asset_path"]) for galaxy in (setup.scene.lens, setup.scene.source)
                              for component in galaxy.light if component.type == "Image")
        assets = {path: load_image_asset(path) for path in paths}
        smooth = build_scene(setup.scene, cosmology, subhalo=None, perturbers=perturbers, assets=assets)
        sampling = native_sampling_variation(smooth)
        scene = smooth if subhalo is None else build_scene(setup.scene, cosmology, subhalo=subhalo,
                                                         perturbers=perturbers, assets=assets)
        expected = observe(scene, kernels, setup.exposure, config_digest=config_digest,
                           photometry=setup.photometry, sampling=sampling)
    return expected if noise_seed is None else expected.draw(noise_seed)
