"""Truth and model PSFs bound once to the resolved scene's light groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ..identity import validate_loaded_file
from ..optics.kernels import KernelBinding
from ..optics.optical_psf import OpticalPSF, OpticalSpec
from ..optics.providers import KernelFileSpec, ModelPSF, PSFProvider, build_model_psf, build_psf_provider
from ..spectra.bandpass import Bandpass
from ..spectra.sed import SED
from ..optics.wavefront import WavefrontMode

if TYPE_CHECKING:
    from ..config.schema import EngineConfig
    from ..instrument import Instrument
    from ..scene.spec import SceneSpec

__all__ = ["PsfPair", "bind_psfs", "bind_truth", "truth_provider", "validate_loaded_psf_files"]


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


def truth_provider(config: EngineConfig, *, bandpass: Bandpass | None = None) -> PSFProvider:
    spec = config.psf.truth
    keywords = {}
    if isinstance(spec, OpticalSpec) and spec.wavelength_m is None:
        if bandpass is None:
            raise ValueError("sampled optical wavelengths require the primary bandpass")
        keywords["wavelengths_m"] = bandpass.nodes(spec.wavelength_samples)
    elif not isinstance(spec, (OpticalSpec, KernelFileSpec)) and bandpass is not None:
        keywords["bandpass_support_m"] = bandpass.support_m
    return build_psf_provider(spec, pixel_scale_arcsec=config.scene.grid.pixel_scale_arcsec, **keywords)


def validate_loaded_psf_files(provider: PSFProvider, binding: KernelBinding, manifest: Mapping[str, str]) -> None:
    """The provider's loaded kernel and truth-draw files must match the input manifest."""
    for path, digest in provider.file_digests.items():
        validate_loaded_file(path, digest, manifest)
    for kernel in binding.kernels:
        if kernel.source["kind"] == "file":
            validate_loaded_file(kernel.source["path"], kernel.source["file_sha256"], manifest)
    if isinstance(provider, OpticalPSF) and provider.draw is not None:
        draw = provider.draw
        if draw.spec.prior.kind == "path":
            validate_loaded_file(draw.spec.prior.path, draw.prior_digest, manifest)


def bind_truth(provider: PSFProvider, scene: SceneSpec,
               instrument: Instrument, *, loaded_seds: Mapping[str, SED] | None = None) -> tuple[KernelBinding, Mapping[str, Any] | None]:
    nodes = provider.wavelengths_m
    if nodes is None:
        kernel = provider.kernel()
    elif len(nodes) == 1:
        kernel = provider.kernel(nodes[0])
    else:
        raise ValueError("several wavelength nodes require chromatic PSF binding")
    return KernelBinding.uniform(kernel, tuple(scene.light_groups(loaded_seds=loaded_seds))), None


def bind_psfs(config: EngineConfig, scene: SceneSpec, instrument: Instrument, *, truth: PSFProvider,
              loaded_seds: Mapping[str, SED] | None = None) -> PsfPair:
    truth_kernels, spectral = bind_truth(truth, scene, instrument, loaded_seds=loaded_seds)
    model = build_model_psf(config.psf.model, truth, pixel_scale_arcsec=scene.grid.pixel_scale_arcsec)
    model_kernels = truth_kernels if model.relation == "matched" else bind_truth(
        model.provider, scene, instrument, loaded_seds=loaded_seds)[0]
    return PsfPair(truth, model, truth_kernels, model_kernels, spectral)
