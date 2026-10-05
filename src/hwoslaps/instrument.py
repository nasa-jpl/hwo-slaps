"""The instrument section: the detector noise parameters, parsed and built.

The detector carries the noise model only (gain, read noise, dark current). Spectral
response, including detector quantum efficiency, belongs to the bandpass, so the detected
rates the observation consumes already include it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real as _Number
from typing import Any, Mapping

import numpy as np

from .config.checks import ConfigError, Key, Nullable, Real, Rule, Table, Text

__all__ = [
    "CROSS_RULES", "DETECTOR_TABLE", "Detector", "INSTRUMENT_TABLE", "Instrument", "InstrumentSpec",
    "build_instrument", "check_finite_number", "parse_instrument",
]


def check_finite_number(name: str, value: Any, *, positive: bool) -> float:
    """``value`` as a float when it is a finite number, > 0 if ``positive`` and >= 0 otherwise.

    Booleans are refused, an integer too large for a float is not finite, and the bound applies
    to the float returned. A value outside the domain raises a ValueError whose message starts
    with ``name``; ``Detector``, ``Exposure`` and ``Observation.sampling`` check their numbers
    with it.
    """
    number = math.nan
    if isinstance(value, _Number) and not isinstance(value, (bool, np.bool_)):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
    if not (math.isfinite(number) and (number > 0 if positive else number >= 0)):
        raise ValueError(f"{name} must be a finite number {'> 0' if positive else '>= 0'}, got {value!r}")
    return number


@dataclass(frozen=True)
class Detector:
    """Noise parameters of one detector pixel; values are normalized to float.

    Construction refuses values outside the physical domain (gain > 0, read noise >= 0,
    dark current >= 0, all finite numbers, booleans refused) with a ValueError whose
    message starts with the field name.
    """

    gain_e_per_adu: float
    read_noise_e: float
    dark_current_e_per_s: float

    def __post_init__(self) -> None:
        for name, positive in (("gain_e_per_adu", True), ("read_noise_e", False),
                               ("dark_current_e_per_s", False)):
            object.__setattr__(self, name, check_finite_number(name, getattr(self, name), positive=positive))

    def to_mapping(self) -> dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _detector_domain(values: Mapping[str, Any], path: str) -> None:
    try:
        Detector(**values)
    except ValueError as error:
        raise ConfigError(path, str(error)) from None


DETECTOR_TABLE = Table(
    keys=(
        Key("gain_e_per_adu", Real(), "detector gain; must be > 0", unit="e-/ADU"),
        Key("read_noise_e", Real(), "read noise of one pixel in one exposure; must be >= 0",
            unit="e- per pixel per exposure"),
        Key("dark_current_e_per_s", Real(), "dark current of one pixel; must be >= 0",
            unit="e-/s per pixel"),
    ),
    rules=(Rule("gain > 0, read noise >= 0 and dark current >= 0 (the Detector domain)",
                _detector_domain),),
    doc="detector noise parameters",
)

INSTRUMENT_TABLE = Table(
    keys=(
        Key("name", Nullable(Text()), "instrument label, recorded only", default=None),
        Key("detector", DETECTOR_TABLE, "detector noise parameters"),
    ),
    doc="the instrument",
)

# Rules over the whole configuration that config/schema.py runs after the section reads (SPEC_CORE 4.8).
CROSS_RULES: tuple[Rule, ...] = ()


@dataclass(frozen=True)
class InstrumentSpec:
    """The parsed ``instrument`` section."""

    name: str | None
    detector: Detector

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> InstrumentSpec:
        """The spec of values already read by ``INSTRUMENT_TABLE``."""
        return cls(name=values["name"], detector=Detector(**values["detector"]))


def parse_instrument(mapping: Mapping[str, Any], path: str = "instrument") -> InstrumentSpec:
    """Read the ``instrument`` section strictly; errors are ConfigErrors at their dotted path."""
    return InstrumentSpec.from_values(INSTRUMENT_TABLE.read(mapping, path))


@dataclass(frozen=True)
class Instrument:
    """The instrument an observation is made with."""

    name: str | None
    detector: Detector


def build_instrument(spec: InstrumentSpec) -> Instrument:
    return Instrument(name=spec.name, detector=spec.detector)
