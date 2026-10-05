"""Expected detector images of a scene, using the observation's convolution route."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
import types

import numpy as np

from ..observation.expected import Exposure, convolve_light
from ..optics.kernels import KernelBinding, convolve_real_space
from ..scene.builder import Scene, build_scene
from ..scene.cosmology import Cosmology
from ..scene.halos import Halo
from ..scene.image_source import ImageAsset, frozen_value
from ..scene.spec import SceneSpec


@dataclass(frozen=True)
class SceneRenderer:
    """Render nodes and nuisance scenes with the preparation's already loaded assets."""

    spec: SceneSpec
    cosmology: Cosmology
    perturbers: tuple[Halo, ...]
    exposure: Exposure
    assets: Mapping[str, ImageAsset] = field(kw_only=True)

    def scene(self, *, subhalo: Halo | None = None, spec: SceneSpec | None = None) -> Scene:
        return build_scene(self.spec if spec is None else spec, self.cosmology, subhalo=subhalo,
                           perturbers=self.perturbers, assets=self.assets)

    def light_rate(self, scene: Scene, binding: KernelBinding,
                   planes: Collection[str] | None = None) -> np.ndarray:
        rates = convolve_light(scene.light_images, scene.light_groups, binding, scene.pixel_scale_arcsec)
        selected = [rates[plane] for plane in ("lens", "source")
                    if plane in rates and (planes is None or plane in planes)]
        if not selected:
            return np.zeros(scene.spec.grid.shape, dtype=float)
        total = selected[0]
        for rate in selected[1:]:
            total = total + rate
        return total

    def mean_adu(self, scene: Scene, binding: KernelBinding) -> np.ndarray:
        return self.exposure.mean_adu(self.light_rate(scene, binding))

    def derivative_adu(self, scene: Scene, binding: KernelBinding,
                       derivatives: Sequence[np.ndarray]) -> np.ndarray:
        if len(derivatives) != len(binding.kernels):
            raise ValueError("derivatives must be aligned with binding.kernels")
        if set(scene.light_groups) != set(binding.group_index):
            raise ValueError("derivative binding must cover the scene's light groups")
        partitions: dict[tuple[str, int], list[str]] = {}
        for group, definition in scene.light_groups.items():
            partitions.setdefault((definition.plane, binding.group_index[group]), []).append(group)
        terms = []
        for (_, index), members in partitions.items():
            light = scene.light_images[members[0]]
            for member in members[1:]:
                light = light + scene.light_images[member]
            terms.append(convolve_real_space(light, derivatives[index], scene.pixel_scale_arcsec))
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return self.exposure.signal_adu(total)

    def __reduce__(self):
        # Scene values and asset metadata are mapping proxies; workers receive their
        # values, and restore the same frozen types without reading an asset file.
        assets = {path: (asset.sb, asset.pixel_scale_arcsec, _transport_value(asset.metadata, frozen=False), asset.digest)
                  for path, asset in self.assets.items()}
        return (_restore_renderer, (_transport_value(self.spec, frozen=False), self.cosmology, self.perturbers, self.exposure, assets))


def _transport_value(value, *, frozen):
    if is_dataclass(value):
        return replace(value, **{item.name: _transport_value(getattr(value, item.name), frozen=frozen)
                                 for item in fields(value)})
    if isinstance(value, Mapping):
        copied = {key: _transport_value(item, frozen=frozen) for key, item in value.items()}
        return types.MappingProxyType(copied) if frozen else copied
    if isinstance(value, (list, tuple)):
        return tuple(_transport_value(item, frozen=frozen) for item in value)
    return value


def _restore_renderer(spec, cosmology, perturbers, exposure, records):
    assets = {path: ImageAsset(samples, scale, frozen_value(metadata), digest)
              for path, (samples, scale, metadata, digest) in records.items()}
    for asset in assets.values():
        asset.sb.setflags(write=False)
    return SceneRenderer(_transport_value(spec, frozen=True), cosmology, perturbers, exposure, assets=assets)
