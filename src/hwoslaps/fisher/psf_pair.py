"""Truth and model PSFs bound once to the resolved scene's light groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ..optics.kernels import KernelBinding
from ..optics.providers import ModelPSF, PSFProvider, build_model_psf, build_psf_provider
from ..optics.wavefront import WavefrontMode

if TYPE_CHECKING:
    from ..config.schema import EngineConfig
    from ..instrument import Instrument
    from ..scene.spec import SceneSpec

__all__ = ["PsfPair", "bind_psfs", "bind_truth", "truth_provider"]


@dataclass(frozen=True)
class PsfPair:
    truth: PSFProvider
    model: ModelPSF
    truth_kernels: KernelBinding
    model_kernels: KernelBinding
    spectral: Mapping[str, Any] | None

    @property
    def relation(self) -> str:
        return self.model.relation

    @property
    def mismatched(self) -> bool:
        return self.relation != "matched"

    def model_kernel_derivatives(self, mode: WavefrontMode, step_nm: float) -> tuple[np.ndarray, ...]:
        provider = self.model.provider
        if provider.basis is None or provider.coefficients is None:
            raise ValueError("wavefront nuisances need a model PSF with a wavefront basis")
        coefficients = provider.coefficients
        value = coefficients.value(mode)
        nodes = provider.wavelengths_m
        if nodes is None or len(nodes) != 1:
            raise ValueError("wavefront derivatives require a monochromatic optical provider")
        plus = provider.kernel(nodes[0], coefficients=coefficients.replace(mode, value + step_nm))
        minus = provider.kernel(nodes[0], coefficients=coefficients.replace(mode, value - step_nm))
        derivative = (plus.kernel - minus.kernel) / (2.0 * step_nm)
        return (derivative,)

    def to_mapping(self) -> dict[str, Any]:
        draw = self.model.knowledge_error
        return {"truth": self.truth.to_mapping(), "model": self.model.provider.to_mapping(),
                "truth_kernels": self.truth_kernels.to_mapping(), "model_kernels": self.model_kernels.to_mapping(),
                "psf_relation": self.relation, "knowledge_error": None if draw is None else draw.to_mapping(),
                "spectral": self.spectral}


def truth_provider(config: EngineConfig) -> PSFProvider:
    return build_psf_provider(config.psf.truth, pixel_scale_arcsec=config.scene.grid.pixel_scale_arcsec)


def bind_truth(provider: PSFProvider, scene: SceneSpec,
               instrument: Instrument) -> tuple[KernelBinding, Mapping[str, Any] | None]:
    nodes = provider.wavelengths_m
    if nodes is None:
        kernel = provider.kernel()
    elif len(nodes) == 1:
        kernel = provider.kernel(nodes[0])
    else:
        raise ValueError("several wavelength nodes require chromatic PSF binding")
    return KernelBinding.uniform(kernel, tuple(scene.light_groups())), None


def bind_psfs(config: EngineConfig, scene: SceneSpec, instrument: Instrument, *, truth: PSFProvider) -> PsfPair:
    truth_kernels, spectral = bind_truth(truth, scene, instrument)
    model = build_model_psf(config.psf.model, truth, pixel_scale_arcsec=scene.grid.pixel_scale_arcsec)
    model_kernels = truth_kernels if model.relation == "matched" else bind_truth(model.provider, scene, instrument)[0]
    return PsfPair(truth, model, truth_kernels, model_kernels, spectral)
