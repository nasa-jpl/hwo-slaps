"""The observation section, the ``Observation`` product and ``observe``.

``observe`` turns a rendered scene into its expected detector image: the light convolved
with the truth kernels, the detector mean and the noise map. ``Observation.draw`` makes one
noisy realization of that expectation; every realization comes from the expected
observation, so the noise map always describes the expectation.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Literal, Mapping

import numpy as np

from ..config.checks import ConfigError, Integer, Key, Nullable, Real, Rule, Table
from ..identity import array_digest
from ..instrument import check_finite_number
from ..spectra.bandpass import BANDPASS_TABLE, BandpassSpec, parse_bandpass
from ..spectra.sed import SED_TABLE, SEDSpec, parse_sed
from .expected import Exposure, Plane, convolve_light
from .noise import draw_noisy_adu

if TYPE_CHECKING:
    from ..optics.kernels import KernelBinding
    from ..scene.builder import Scene
    from ..scene.halos import Halo
    from ..scene.spec import GridSpec
    from .normalization import PhotometryRecord

__all__ = [
    "MAX_NATIVE_SAMPLING_VARIATION", "OBSERVATION_TABLE", "Observation", "ObservationSpec", "SKY_TABLE", "SkySpec", "observe",
    "parse_observation",
]

# Measured by the native-pixel oracle in test_observation_sampling.py.
MAX_NATIVE_SAMPLING_VARIATION: Final[float] = 0.063

def _sky_reference(values: Mapping[str, Any], path: str) -> None:
    if values["reference_band"] is not None and values["ab_mag_per_arcsec2"] is None:
        raise ConfigError(f"{path}.reference_band", "a sky reference band requires an AB surface brightness")
    if (values["reference_band"] is None) != (values["sed"] is None):
        raise ConfigError(f"{path}.sed", "a sky SED is required exactly when a reference band is given")


SKY_TABLE = Table((
    Key("rate_e_per_s", Nullable(Real(min=0.0)), "detected sky rate of one pixel", None, unit="e-/s per pixel"),
    Key("ab_mag_per_arcsec2", Nullable(Real()), "AB sky surface brightness", None, unit="mag/arcsec^2"),
    Key("reference_band", Nullable(BANDPASS_TABLE), "band of the supplied sky magnitude", None),
    Key("sed", Nullable(SED_TABLE), "sky spectral shape for a reference-band magnitude", None),
), exactly_one=(("rate_e_per_s", "ab_mag_per_arcsec2"),),
   rules=(Rule("reference sky magnitude and SED belong together", _sky_reference),), doc="the sky background")

OBSERVATION_TABLE = Table(
    keys=(
        Key("exposure_time_s", Real(min=0.0, min_open=True),
            "total exposure time of the summed exposures", unit="s"),
        Key("exposure_count", Integer(min=1),
            "number of equal exposures summed into the image; read noise enters once per exposure",
            default=1),
        Key("sky", SKY_TABLE, "the sky background"),
    ),
    doc="the exposure",
)


@dataclass(frozen=True)
class SkySpec:
    rate_e_per_s: float | None = None
    ab_mag_per_arcsec2: float | None = None
    reference_band: BandpassSpec | None = None
    sed: SEDSpec | None = None


@dataclass(frozen=True)
class ObservationSpec:
    """The parsed ``observation`` section."""

    exposure_time_s: float
    exposure_count: int
    sky: SkySpec

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> ObservationSpec:
        """The spec of values already read by ``OBSERVATION_TABLE``."""
        sky = values["sky"]
        reference, sed = sky["reference_band"], sky["sed"]
        return cls(exposure_time_s=values["exposure_time_s"], exposure_count=values["exposure_count"],
                   sky=SkySpec(sky["rate_e_per_s"], sky["ab_mag_per_arcsec2"],
                       None if reference is None else parse_bandpass(reference, "observation.sky.reference_band"),
                       None if sed is None else parse_sed(sed, "observation.sky.sed")))


def parse_observation(mapping: Mapping[str, Any], path: str = "observation") -> ObservationSpec:
    """Read the ``observation`` section strictly; errors are ConfigErrors at their dotted path."""
    return ObservationSpec.from_values(OBSERVATION_TABLE.read(mapping, path))


@dataclass(frozen=True, eq=False)
class Observation:
    """An expected or noisy detector image of a scene, with the light it was made from.

    Arrays and mappings are read-only, and arrays are shaped like the grid, so the expected
    observation and its draws can share them. ``light_rate_e_per_s`` is the convolved light of
    every plane in detected e-/s per pixel (round-off negatives kept);
    ``light_rate_by_plane_e_per_s`` holds it per plane, ``"source"`` always and ``"lens"``
    with lens light, and without lens light its ``"source"`` entry is the total itself.
    ``noise_map_adu`` comes from the expected image for both kinds. ``sampling`` is the
    within-pixel variation of each fiducial smooth-scene light group, measured once by the
    preparation caller and retained for injected observations and noise draws.
    """

    kind: Literal["expected", "noisy"]
    data_adu: np.ndarray
    expected_adu: np.ndarray
    noise_map_adu: np.ndarray
    light_rate_e_per_s: np.ndarray
    light_rate_by_plane_e_per_s: Mapping[Plane, np.ndarray]
    grid: GridSpec
    exposure: Exposure
    psfs: KernelBinding
    noise_seed: int | None
    subhalo: Halo | None
    config_digest: str | None
    photometry: PhotometryRecord | None
    sampling: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.kind not in ("expected", "noisy"):
            raise ValueError(f"kind must be 'expected' or 'noisy', got {self.kind!r}")
        if (self.kind == "expected") != (self.noise_seed is None):
            raise ValueError("an expected observation has no noise seed and a noisy one has its seed")
        if self.kind == "expected" and self.data_adu is not self.expected_adu:
            raise ValueError("the data of an expected observation is its expected image")
        planes = set(self.light_rate_by_plane_e_per_s)
        if "source" not in planes or not planes <= {"lens", "source"}:
            raise ValueError(f"light planes must be 'source' and optionally 'lens', got {sorted(planes)}")
        shape = tuple(self.grid.shape)
        arrays = {"data_adu": self.data_adu, "expected_adu": self.expected_adu,
                  "noise_map_adu": self.noise_map_adu, "light_rate_e_per_s": self.light_rate_e_per_s,
                  **{f"light_rate_by_plane_e_per_s[{plane}]": rate
                     for plane, rate in self.light_rate_by_plane_e_per_s.items()}}
        for name, array in arrays.items():
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}; the grid is {shape}")
        if set(self.sampling) != set(self.psfs.group_index):
            raise ValueError(f"sampling covers {sorted(self.sampling)}; the light groups are "
                             f"{sorted(self.psfs.group_index)}")
        object.__setattr__(self, "light_rate_by_plane_e_per_s",
                           types.MappingProxyType(dict(self.light_rate_by_plane_e_per_s)))
        sampling = {key: check_finite_number(f"sampling[{key!r}]", value, positive=False)
                    for key, value in self.sampling.items()}
        object.__setattr__(self, "sampling", types.MappingProxyType(sampling))

    @property
    def pixel_scale_arcsec(self) -> float:
        return self.grid.pixel_scale_arcsec

    def counts_e(self) -> np.ndarray:
        """Expected electron counts, the Poisson mean of a draw."""
        return self.exposure.counts_e(self.light_rate_e_per_s)

    def variance_e2(self) -> np.ndarray:
        return self.exposure.variance_e2(self.light_rate_e_per_s)

    def draw(self, seed: int) -> Observation:
        """One noisy realization from ``root_rng(seed)``; only an expected observation draws."""
        if self.kind != "expected":
            raise ValueError("a noisy observation cannot be drawn again; draw from the expected observation")
        data = draw_noisy_adu(self.counts_e(), self.exposure, seed)
        data.flags.writeable = False
        return replace(self, kind="noisy", data_adu=data, noise_seed=int(seed))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "noise_seed": self.noise_seed,
            "data_digest": array_digest(self.data_adu),
            "expected_digest": array_digest(self.expected_adu),
            "noise_map_digest": array_digest(self.noise_map_adu),
            "grid": {"shape": list(self.grid.shape), "pixel_scale_arcsec": self.grid.pixel_scale_arcsec,
                     "over_sample_size": self.grid.over_sample_size},
            "exposure": self.exposure.to_mapping(),
            "kernels": self.psfs.to_mapping(),
            "subhalo": None if self.subhalo is None else self.subhalo.to_mapping(),
            "config_digest": self.config_digest,
            "photometry": None if self.photometry is None else self.photometry.to_mapping(),
            "sampling": dict(self.sampling),
        }


def observe(scene: Scene, kernels: KernelBinding, exposure: Exposure, *, config_digest: str | None,
            photometry: PhotometryRecord | None, sampling: Mapping[str, float]) -> Observation:
    """The expected observation of ``scene`` through the truth kernels ``kernels``.

    No random generator is used. The caller supplies ``sampling`` measured on the fiducial
    smooth scene once during preparation. It is recorded without another light evaluation,
    including when ``scene`` contains an injected subhalo.
    """
    by_plane = convolve_light(scene.light_images, scene.light_groups, kernels, scene.pixel_scale_arcsec)
    total = by_plane["lens"] + by_plane["source"] if "lens" in by_plane else by_plane["source"]
    expected = exposure.mean_adu(total)
    noise_map = exposure.noise_map_adu(total)
    for array in (*by_plane.values(), total, expected, noise_map):
        array.flags.writeable = False
    return Observation(
        kind="expected", data_adu=expected, expected_adu=expected, noise_map_adu=noise_map,
        light_rate_e_per_s=total, light_rate_by_plane_e_per_s=by_plane, grid=scene.spec.grid,
        exposure=exposure, psfs=kernels, noise_seed=None, subhalo=scene.subhalo,
        config_digest=config_digest, photometry=photometry, sampling=sampling,
    )
