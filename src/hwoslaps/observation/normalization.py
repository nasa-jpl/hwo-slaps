"""Resolution of the configured instrument and observation into forward-model inputs.

``resolve_observing`` runs once per preparation or simulation, before the scene is
rendered: it builds the instrument, resolves the sky rate and the light amplitudes, and
records what it resolved. Detected-rate inputs pass through unchanged and make no
photometry record.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

from ..instrument import Instrument, InstrumentSpec, build_instrument
from .expected import Exposure
from .observation import ObservationSpec

if TYPE_CHECKING:
    from ..optics.providers import PSFProvider
    from ..scene.spec import SceneSpec

__all__ = ["ObservingSetup", "PhotometryRecord", "resolve_observing"]


@dataclass(frozen=True)
class PhotometryRecord:
    """What photometric inputs resolved to: collecting area, bandpass, sky and components.

    ``components`` maps ``"<galaxy>.<component>"`` to its input, resolved rate and amplitude.
    """

    collecting_area_m2: float | None
    collecting_area_source: Literal["config", "optical_pupil"] | None
    bandpass: Mapping[str, Any] | None
    sky: Mapping[str, Any]
    components: Mapping[str, Mapping[str, Any]]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "collecting_area_m2": self.collecting_area_m2,
            "collecting_area_source": self.collecting_area_source,
            "bandpass": deepcopy(self.bandpass),
            "sky": deepcopy(self.sky),
            "components": deepcopy(self.components),
        }


@dataclass(frozen=True)
class ObservingSetup:
    """The resolved scene, the built instrument, the exposure and the photometry record."""

    scene: SceneSpec
    instrument: Instrument
    exposure: Exposure
    photometry: PhotometryRecord | None


def resolve_observing(scene: SceneSpec, instrument: InstrumentSpec, observation: ObservationSpec, *,
                      truth: PSFProvider) -> ObservingSetup:
    """The forward-model inputs of one configuration.

    ``truth`` is the truth PSF provider, the source of the collecting area that magnitude
    inputs need; detected-rate inputs need none, so no kernel or pupil is evaluated. An
    exposure whose blank pixels would have zero variance raises ValueError.
    """
    built = build_instrument(instrument)
    exposure = Exposure(built.detector, observation.exposure_time_s, observation.sky.rate_e_per_s,
                        observation.exposure_count)
    return ObservingSetup(scene=scene, instrument=built, exposure=exposure, photometry=None)
