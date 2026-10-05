"""Resolve photometric inputs once, retaining the exact loaded spectral identities."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

from ..config.checks import ConfigError, Rule
from ..constants import JANSKY_SI, PLANCK_J_S
from ..identity import json_ready, validate_loaded_file
from ..instrument import Instrument, InstrumentSpec, build_instrument, check_finite_number
from ..scene.image_source import ImageAsset, frozen_value, load_image_asset
from ..scene.profiles import PROFILE_TYPES
from ..spectra.bandpass import Bandpass, bandpass_spec_mapping, build_bandpass, integrate_dlnlambda
from ..spectra.photometry import band_mean_throughput, fnu_jy_to_ab, rate_from_ab, sky_rate_e_per_s_per_pixel
from ..spectra.sed import SED, build_sed, sed_spec_mapping
from .expected import Exposure
from .observation import ObservationSpec

if TYPE_CHECKING:
    from ..optics.providers import PSFProvider
    from ..scene.spec import SceneSpec

__all__ = ["CROSS_RULES", "ObservingSetup", "PhotometryRecord", "resolve_observing"]


def _ab_inputs(root: Mapping[str, Any], path: str) -> None:
    magnitude = root["observation"]["sky"]["ab_mag_per_arcsec2"] is not None
    for plane in ("lens", "source"):
        magnitude |= any(component["flux"] is not None and component["flux"]["ab_mag"] is not None
                         for component in root["scene"][plane]["light"].values())
    if not magnitude:
        return
    instrument = root["instrument"]
    if instrument["bandpass"] is None:
        raise ConfigError("instrument.bandpass", "AB photometry requires an instrument bandpass")
    if instrument["collecting_area_m2"] is None and root["psf"]["truth"]["kind"] != "optical":
        raise ConfigError("instrument.collecting_area_m2", "AB photometry needs a collecting area or an optical truth pupil")


CROSS_RULES = (Rule("AB source or sky inputs require a bandpass and a collecting area (X6)", _ab_inputs),)


@dataclass(frozen=True)
class PhotometryRecord:
    collecting_area_m2: float | None
    collecting_area_source: Literal["config", "optical_pupil"] | None
    bandpass: Mapping[str, Any] | None
    sky: Mapping[str, Any]
    components: Mapping[str, Mapping[str, Any]]
    file_digests: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("bandpass", "sky", "components", "file_digests"):
            object.__setattr__(self, name, frozen_value(getattr(self, name)))

    def to_mapping(self) -> dict[str, Any]:
        return {"collecting_area_m2": self.collecting_area_m2, "collecting_area_source": self.collecting_area_source,
                "bandpass": json_ready(self.bandpass), "sky": json_ready(self.sky),
                "components": json_ready(self.components), "file_digests": dict(self.file_digests)}


@dataclass(frozen=True)
class ObservingSetup:
    scene: SceneSpec
    instrument: Instrument
    exposure: Exposure
    photometry: PhotometryRecord | None
    file_digests: Mapping[str, str] = field(default_factory=dict)
    loaded_seds: Mapping[str, SED] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_digests", MappingProxyType(dict(self.file_digests)))
        object.__setattr__(self, "loaded_seds", MappingProxyType(dict(self.loaded_seds)))


def resolve_observing(scene: SceneSpec, instrument: InstrumentSpec, observation: ObservationSpec, *,
                      truth: PSFProvider, bandpass: Bandpass | None = None,
                      assets: Mapping[str, ImageAsset] | None = None,
                      expected_file_digests: Mapping[str, str] | None = None) -> ObservingSetup:
    """Resolved amplitudes, sky rate and captured spectra for every later render.

    An amplitude-given component is retained as the same object. Intrinsic flux inputs
    set the analytic profile integral; the finite-grid mapping ratio is recorded without
    rescaling. Only AB inputs with no supplied collecting area evaluate the truth pupil area.
    """
    files: dict[str, str] = {}
    def merge(values: Mapping[str, str]) -> None:
        for path, digest in values.items():
            if path in files and files[path] != digest:
                raise ValueError(f"{path}: spectral input changed between reads")
            if expected_file_digests is not None:
                validate_loaded_file(path, digest, expected_file_digests)
            files[path] = digest

    components = tuple(component for galaxy in (scene.lens, scene.source) for component in galaxy.light)
    area_required = observation.sky.ab_mag_per_arcsec2 is not None or any(
        component.flux is not None and component.flux.ab_mag is not None for component in components)
    provider_area = truth.collecting_area_m2 if area_required and instrument.collecting_area_m2 is None else None
    built = build_instrument(instrument, provider_area_m2=provider_area, area_required=area_required, bandpass=bandpass)
    if built.bandpass is not None:
        merge(built.bandpass.file_digests)
    loaded_seds: dict[str, SED] = {}
    for galaxy in (scene.lens, scene.source):
        for component in galaxy.light:
            if component.sed is not None:
                sed = build_sed(component.sed, redshift=galaxy.redshift)
                loaded_seds[f"{galaxy.plane}.{component.name}"] = sed
                merge(sed.file_digests)

    def reference(spec):
        if spec is None:
            return None
        if spec == instrument.bandpass:
            return built.bandpass
        band = build_bandpass(spec)
        merge(band.file_digests)
        return band

    sky = observation.sky
    if sky.rate_e_per_s is not None:
        sky_rate = sky.rate_e_per_s
    else:
        if built.bandpass is None or built.collecting_area_m2 is None:
            raise ValueError("AB sky photometry requires the instrument bandpass and collecting area")
        sky_sed = None if sky.sed is None else build_sed(sky.sed, redshift=0.0)
        if sky_sed is not None:
            merge(sky_sed.file_digests)
        sky_rate = sky_rate_e_per_s_per_pixel(sky.ab_mag_per_arcsec2, built.bandpass, built.collecting_area_m2,
                                            scene.grid.pixel_scale_arcsec, sed=sky_sed, reference_band=reference(sky.reference_band))
    sky_record = {"input": {"rate_e_per_s": sky.rate_e_per_s, "ab_mag_per_arcsec2": sky.ab_mag_per_arcsec2,
                           "reference_band": None if sky.reference_band is None else bandpass_spec_mapping(sky.reference_band),
                           "sed": None if sky.sed is None else sed_spec_mapping(sky.sed)}, "rate_e_per_s": sky_rate}
    record_needed = (area_required or built.bandpass is not None or built.collecting_area_m2 is not None
                     or any(component.flux is not None or component.sed is not None for component in components))
    if not record_needed:
        exposure = Exposure(built.detector, observation.exposure_time_s, sky_rate, observation.exposure_count)
        return ObservingSetup(scene, built, exposure, None, files, loaded_seds)
    loaded_assets = {} if assets is None else dict(assets)
    for component in components:
        if component.type == "Image" and component.flux is not None:
            path = str(component.values["asset_path"])
            if path not in loaded_assets:
                loaded_assets[path] = load_image_asset(path)
            merge({path: loaded_assets[path].digest})
    component_records = {}
    resolved = scene
    omega_pixel = scene.grid.pixel_scale_arcsec**2
    for galaxy in (scene.lens, scene.source):
        light = []
        for component in galaxy.light:
            profile = PROFILE_TYPES[component.type]
            values = dict(component.values)
            amplitude_key = profile.amplitude_key
            unit_integral = check_finite_number(f"{galaxy.plane}.{component.name}.unit_integral_arcsec2",
                                                profile.unit_integral(values), positive=True)
            name = f"{galaxy.plane}.{component.name}"
            sed = loaded_seds.get(name)
            flux = component.flux
            if flux is None:
                amplitude = values[amplitude_key]
                rate = amplitude * unit_integral / omega_pixel if record_needed else None
                updated = component
                ratio = None
            else:
                if flux.rate_e_per_s is not None:
                    rate = flux.rate_e_per_s
                else:
                    if built.bandpass is None or built.collecting_area_m2 is None:
                        raise ValueError(f"{name}: AB flux requires the instrument bandpass and collecting area")
                    rate = rate_from_ab(flux.ab_mag, built.bandpass, built.collecting_area_m2,
                                        sed=sed, reference_band=reference(flux.reference_band))
                amplitude = check_finite_number(f"{name}.{amplitude_key}", rate * omega_pixel / unit_integral, positive=True)
                values[amplitude_key] = amplitude
                updated = replace(component, values=frozen_value(values), flux=None)
                from ..scene.builder import render_component_unlensed
                ratio = float(render_component_unlensed(updated, scene.grid, assets=loaded_assets).sum()) * omega_pixel / (amplitude * unit_integral)
            light.append(updated)
            if record_needed:
                magnitude = None
                if built.bandpass is not None and built.collecting_area_m2 is not None:
                    response = integrate_dlnlambda(built.bandpass.throughput, built.bandpass.wavelengths_m)
                    magnitude = fnu_jy_to_ab(rate * PLANCK_J_S / (built.collecting_area_m2 * JANSKY_SI * response))
                component_records[name] = {"flux_input": None if flux is None else {
                        "rate_e_per_s": flux.rate_e_per_s, "ab_mag": flux.ab_mag,
                        "reference_band": None if flux.reference_band is None else bandpass_spec_mapping(flux.reference_band)},
                    "sed": None if component.sed is None else sed_spec_mapping(component.sed),
                    "rate_e_per_s": rate, "amplitude_key": amplitude_key, "amplitude": amplitude,
                    "unit_integral_arcsec2": unit_integral, "mapping_ratio": ratio, "ab_mag_in_band": magnitude,
                    "band_mean_throughput": None if sed is None or built.bandpass is None else band_mean_throughput(built.bandpass, sed)}
        if any(new is not old for new, old in zip(light, galaxy.light)):
            resolved = replace(resolved, **{galaxy.plane: replace(galaxy, light=tuple(light))})
    exposure = Exposure(built.detector, observation.exposure_time_s, sky_rate, observation.exposure_count)
    record = None if not record_needed else PhotometryRecord(built.collecting_area_m2, built.collecting_area_source,
             None if built.bandpass is None else built.bandpass.to_mapping(), sky_record, component_records, files)
    return ObservingSetup(resolved, built, exposure, record, files, loaded_seds)
