"""Truth and model PSFs bound once to the resolved scene's light groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ..identity import json_ready, validate_loaded_file
from ..scene.image_source import frozen_value
from ..optics.kernels import KernelBinding
from ..optics.chromatic import NoSpectralResponse, SpectralWeights, effective_kernel, sed_weights
from ..optics.optical_psf import OpticalPSF, OpticalSpec
from ..optics.providers import KernelFileSpec, ModelPSF, PSFProvider, build_model_psf, build_psf_provider
from ..spectra.bandpass import Bandpass
from ..spectra.sed import SED, build_sed
from ..spectra.photometry import effective_wavelength_m
from ..optics.wavefront import WavefrontMode

if TYPE_CHECKING:
    from ..config.schema import EngineConfig
    from ..instrument import Instrument
    from ..scene.spec import SceneSpec

__all__ = ["PsfPair", "bind_psfs", "bind_truth", "truth_provider", "validate_loaded_psf_files"]


@dataclass(frozen=True)
class _BoundPSF:
    binding: KernelBinding
    record: Mapping[str, Any] | None


@dataclass(frozen=True)
class PsfPair:
    truth: PSFProvider
    model: ModelPSF
    truth_kernels: KernelBinding
    model_kernels: KernelBinding
    spectral: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "spectral", frozen_value(self.spectral))

    @property
    def relation(self) -> str:
        return self.model.relation

    @property
    def mismatched(self) -> bool:
        return self.relation != "matched"

    def model_kernel_derivatives(self, mode: WavefrontMode, step_nm: float) -> tuple[np.ndarray, ...] | Mapping[str, np.ndarray]:
        provider = self.model.provider
        if provider.basis is None or provider.coefficients is None:
            raise ValueError("wavefront nuisances need a model PSF with a wavefront basis")
        coefficients = provider.coefficients
        value = coefficients.value(mode)
        upper, lower = coefficients.replace(mode, value + step_nm), coefficients.replace(mode, value - step_nm)
        nodes = provider.wavelengths_m
        if nodes is None:
            raise ValueError("wavefront derivatives need optical wavelength nodes")
        if len(nodes) == 1:
            plus = provider.kernel(nodes[0], coefficients=upper)
            minus = provider.kernel(nodes[0], coefficients=lower)
            return ((plus.kernel - minus.kernel) / (2.0 * step_nm),)
        if self.spectral is None or self.spectral["model"] is None:
            raise ValueError("chromatic derivatives require the bound model spectral record")
        groups = self.spectral["model"]["groups"]
        if self.relation == "monochromatic":
            return {group: (provider.kernel(record["kernel_wavelength_m"], coefficients=upper).kernel
                            - provider.kernel(record["kernel_wavelength_m"], coefficients=lower).kernel)
                           / (2.0 * step_nm) for group, record in groups.items()}
        plus, minus = provider.kernels(coefficients=upper), provider.kernels(coefficients=lower)
        derivatives = {}
        for group, record in groups.items():
            values = record["weights"]
            weights = SpectralWeights(values["wavelengths_m"], values["bin_edges_m"],
                                      values["rates"], values["log_rate_scale"])
            derivatives[group] = (effective_kernel(plus, weights, provider.pixel_scale_arcsec, source={}).kernel
                                  - effective_kernel(minus, weights, provider.pixel_scale_arcsec, source={}).kernel
                                  ) / (2.0 * step_nm)
        return derivatives

    def to_mapping(self) -> dict[str, Any]:
        draw = self.model.knowledge_error
        return {"truth": self.truth.to_mapping(), "model": self.model.provider.to_mapping(),
                "truth_kernels": self.truth_kernels.to_mapping(), "model_kernels": self.model_kernels.to_mapping(),
                "psf_relation": self.relation, "knowledge_error": None if draw is None else draw.to_mapping(),
                "spectral": json_ready(self.spectral)}


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
    bound = _bind_provider(provider, scene, instrument, loaded_seds=loaded_seds)
    return bound.binding, bound.record


def _captured_seds(scene: SceneSpec, supplied: Mapping[str, SED] | None) -> Mapping[str, SED]:
    if supplied is not None:
        return supplied
    return {f"{galaxy.plane}.{component.name}": build_sed(component.sed, redshift=galaxy.redshift)
            for galaxy in (scene.lens, scene.source) for component in galaxy.light if component.sed is not None}


def _bind_provider(provider: PSFProvider, scene: SceneSpec, instrument: Instrument, *,
                   loaded_seds: Mapping[str, SED] | None = None,
                   monochromatic: bool = False, wavelength_nm: float | None = None) -> _BoundPSF:
    seds = _captured_seds(scene, loaded_seds)
    groups = scene.light_groups(loaded_seds=seds)
    nodes = provider.wavelengths_m
    band = instrument.bandpass
    if nodes is None or (len(nodes) == 1 and not monochromatic):
        # Original one-node operations: no weighted arithmetic or renormalization.
        kernel = provider.kernel() if nodes is None else provider.kernel(nodes[0])
        binding = KernelBinding.uniform(kernel, tuple(groups))
        record = None
        if nodes is not None and band is not None:
            records = {}
            for key, group in groups.items():
                entry = {"sed_digest": None, "weights": None, "effective_wavelength_m": None,
                         "kernel_wavelength_m": nodes[0], "status": "no_sed"}
                if group.sed is not None:
                    sed = seds[f"{group.plane}.{group.components[0]}"]
                    entry["sed_digest"] = sed.digest()
                    try:
                        weights = sed_weights(band, sed, nodes)
                    except NoSpectralResponse:
                        entry["status"] = "zero_response"
                    else:
                        entry.update(weights=weights.to_mapping(), effective_wavelength_m=effective_wavelength_m(sed, band),
                                     status="response")
                records[key] = entry
            record = {"wavelengths_m": list(nodes), "kernel_wavelengths_m": list(nodes),
                      "captured_power_fractions": [kernel.source["captured_power_fraction"]],
                      "node_kernels": [{"identity": kernel.kernel_identity().to_mapping(), "source": dict(kernel.source)}],
                      "groups": records}
        return _BoundPSF(binding, record)
    if band is None:
        raise ValueError("chromatic PSF binding needs the captured instrument bandpass")
    group_seds = {}
    for key, group in groups.items():
        if group.sed is None:
            raise ValueError(f"chromatic light group {key} needs an SED")
        group_seds[key] = seds[f"{group.plane}.{group.components[0]}"]
    weights = {key: sed_weights(band, sed, nodes) for key, sed in group_seds.items()}
    means = {key: effective_wavelength_m(sed, band) for key, sed in group_seds.items()}
    if monochromatic:
        wavelengths = {key: mean if wavelength_nm is None else wavelength_nm / 1.0e9
                       for key, mean in means.items()}
        cache = {wavelength: provider.kernel(wavelength) for wavelength in dict.fromkeys(wavelengths.values())}
        kernels = {key: cache[wavelength] for key, wavelength in wavelengths.items()}
        node_kernels = tuple(cache.values())
    else:
        wavelengths = None
        node_kernels = provider.kernels()
        kernels = {key: effective_kernel(node_kernels, weight, scene.grid.pixel_scale_arcsec,
                     source={"provider": provider.to_mapping(), "weights": weight.to_mapping()})
                   for key, weight in weights.items()}
    binding = KernelBinding.from_groups(kernels)
    record = {"wavelengths_m": list(nodes),
              "kernel_wavelengths_m": [node.source.get("wavelength_m") for node in node_kernels],
              "captured_power_fractions": [node.source["captured_power_fraction"] for node in node_kernels],
              "node_kernels": [{"identity": node.kernel_identity().to_mapping(), "source": dict(node.source)}
                               for node in node_kernels],
              "groups": {key: {"sed_digest": group_seds[key].digest(), "weights": weights[key].to_mapping(),
                                "effective_wavelength_m": means[key],
                                "kernel_wavelength_m": None if wavelengths is None else wavelengths[key],
                                "kernel_identity": kernels[key].kernel_identity().to_mapping()} for key in groups}}
    return _BoundPSF(binding, record)


def bind_psfs(config: EngineConfig, scene: SceneSpec, instrument: Instrument, *, truth: PSFProvider,
              loaded_seds: Mapping[str, SED] | None = None) -> PsfPair:
    seds = _captured_seds(scene, loaded_seds)
    bound_truth = _bind_provider(truth, scene, instrument, loaded_seds=seds)
    model = build_model_psf(config.psf.model, truth, pixel_scale_arcsec=scene.grid.pixel_scale_arcsec)
    bound_model = bound_truth if model.relation == "matched" else _bind_provider(
        model.provider, scene, instrument, loaded_seds=seds,
        monochromatic=model.relation == "monochromatic", wavelength_nm=model.wavelength_nm)
    spectral = None if bound_truth.record is None and bound_model.record is None else {
        "truth": bound_truth.record, "model": bound_model.record}
    return PsfPair(truth, model, bound_truth.binding, bound_model.binding, spectral)
