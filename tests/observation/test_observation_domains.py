"""Physical domains of the detector and observation values, configured and built in Python."""

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.instrument import Detector, InstrumentSpec, parse_instrument
from hwoslaps.observation.expected import Exposure
from hwoslaps.observation.observation import ObservationSpec, SkySpec, parse_observation

DETECTOR = {"gain_e_per_adu": 1.0, "read_noise_e": 0.2, "dark_current_e_per_s": 0.002}
OBSERVATION = {"exposure_time_s": 900.0, "sky": {"rate_e_per_s": 1.0}}


def _instrument(**detector):
    return lambda: parse_instrument({"detector": {**DETECTOR, **detector}})


def _observation(**values):
    return lambda: parse_observation({**OBSERVATION, **values})


def _sky(**sky):
    return lambda: parse_observation({**OBSERVATION, "sky": sky})


def _exposure(**values):
    arguments = {"exposure_time_s": 900.0, "sky_rate_e_per_s": 1.0, **values}
    return lambda: Exposure(Detector(**DETECTOR), **arguments)


# (build, exception type, message prefix, key the message names)
REJECTED = [
    pytest.param(_instrument(gain_e_per_adu=0.0), ConfigError, "instrument.detector", "gain_e_per_adu", id="gain-0"),
    pytest.param(_instrument(gain_e_per_adu=-1), ConfigError, "instrument.detector", "gain_e_per_adu", id="gain-negative"),
    pytest.param(_instrument(read_noise_e=-0.1), ConfigError, "instrument.detector", "read_noise_e", id="read-noise-negative"),
    pytest.param(_instrument(dark_current_e_per_s=-0.01), ConfigError, "instrument.detector", "dark_current_e_per_s", id="dark-negative"),
    pytest.param(_instrument(gain_e_per_adu=float("nan")), ConfigError, "instrument.detector", "gain_e_per_adu", id="gain-nan"),
    pytest.param(_instrument(read_noise_e=float("inf")), ConfigError, "instrument.detector", "read_noise_e", id="read-noise-inf"),
    pytest.param(_instrument(dark_current_e_per_s=True), ConfigError, "instrument.detector", "dark_current_e_per_s", id="dark-bool"),
    pytest.param(_instrument(gain_e_per_adu="1"), ConfigError, "instrument.detector", "gain_e_per_adu", id="gain-text"),
    pytest.param(_observation(exposure_time_s=0.0), ConfigError, "observation", "exposure_time_s", id="time-0"),
    pytest.param(_observation(exposure_count=0), ConfigError, "observation", "exposure_count", id="count-0"),
    pytest.param(_observation(exposure_count=1.5), ConfigError, "observation", "exposure_count", id="count-fraction"),
    pytest.param(_observation(exposure_count=True), ConfigError, "observation", "exposure_count", id="count-bool"),
    pytest.param(_sky(rate_e_per_s=-1.0), ConfigError, "observation", "rate_e_per_s", id="sky-negative"),
    pytest.param(_sky(), ConfigError, "observation", "rate_e_per_s", id="sky-missing"),
    pytest.param(lambda: Detector(gain_e_per_adu=-1.0, read_noise_e=0.2, dark_current_e_per_s=0.002),
                 ValueError, "gain_e_per_adu", "gain_e_per_adu", id="direct-gain-negative"),
    pytest.param(lambda: Detector(gain_e_per_adu=True, read_noise_e=0.2, dark_current_e_per_s=0.002),
                 ValueError, "gain_e_per_adu", "gain_e_per_adu", id="direct-gain-bool"),
    pytest.param(lambda: Detector(gain_e_per_adu=10**400, read_noise_e=0.2, dark_current_e_per_s=0.002),
                 ValueError, "gain_e_per_adu", "gain_e_per_adu", id="direct-gain-overflow"),
    pytest.param(lambda: Detector(gain_e_per_adu=1.0, read_noise_e=float("nan"), dark_current_e_per_s=0.002),
                 ValueError, "read_noise_e", "read_noise_e", id="direct-read-noise-nan"),
    pytest.param(_exposure(exposure_count=1.5), ValueError, "exposure_count", "exposure_count", id="direct-count-fraction"),
    pytest.param(_exposure(exposure_count=True), ValueError, "exposure_count", "exposure_count", id="direct-count-bool"),
    pytest.param(_exposure(exposure_time_s=0.0), ValueError, "exposure_time_s", "exposure_time_s", id="direct-time-0"),
    pytest.param(_exposure(sky_rate_e_per_s=-1.0), ValueError, "sky_rate_e_per_s", "sky_rate_e_per_s", id="direct-sky-negative"),
    pytest.param(lambda: Exposure(DETECTOR, 900.0, 1.0), ValueError, "detector", "Detector", id="direct-detector-mapping"),
]


@pytest.mark.parametrize("build, error, prefix, key", REJECTED)
def test_detector_and_observation_values_out_of_domain_are_rejected(build, error, prefix, key):
    with pytest.raises(error) as caught:
        build()
    assert type(caught.value) is error
    assert str(caught.value).startswith(prefix)
    assert key in str(caught.value)


def test_physical_boundary_values_are_accepted_and_normalized():
    zero_noise = parse_instrument({"name": "zero read noise", "detector": {**DETECTOR, "read_noise_e": 0}})
    assert zero_noise == InstrumentSpec(name="zero read noise", detector=Detector(1.0, 0.0, 0.002))
    assert parse_instrument({"detector": {**DETECTOR, "dark_current_e_per_s": 0.0}}).detector.dark_current_e_per_s == 0.0
    assert parse_observation({"exposure_time_s": 900, "sky": {"rate_e_per_s": 0}}) == ObservationSpec(
        exposure_time_s=900.0, exposure_count=1, sky=SkySpec(rate_e_per_s=0.0))
    exposure = Exposure(Detector(2, 0, 0), exposure_time_s=1, sky_rate_e_per_s=1, exposure_count=np.int64(2))
    mapping = exposure.to_mapping()
    assert mapping == {
        "detector": {"gain_e_per_adu": 2.0, "read_noise_e": 0.0, "dark_current_e_per_s": 0.0},
        "exposure_time_s": 1.0, "exposure_count": 2, "sky_rate_e_per_s": 1.0,
    }
    assert [type(mapping[key]) for key in ("exposure_time_s", "sky_rate_e_per_s", "exposure_count")] == [
        float, float, int]
    assert all(type(value) is float for value in mapping["detector"].values())
