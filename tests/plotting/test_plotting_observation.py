"""Current expected/noisy Observation artists and expected source S/N units."""

import numpy as np
import pytest

from hwoslaps.plotting import plot_observation

pytestmark = pytest.mark.backend


def test_observation_plots_draw_current_arrays_and_expected_source_snr(plt, observation_product):
    observation = observation_product
    for current in (observation, observation.draw(71)):
        expected_snr = (current.light_rate_by_plane_e_per_s["source"] * 17. / 2.5) / current.noise_map_adu
        expected = {"data": current.data_adu, "expected": current.expected_adu,
                    "noise": current.noise_map_adu, "snr": expected_snr}
        for quantity, values in expected.items():
            _, ax = plt.subplots()
            assert plot_observation(current, quantity, ax=ax) is ax
            np.testing.assert_array_equal(ax.images[0].get_array(), values)
            assert ax.images[0].origin == "upper"
            assert ax.images[0].get_extent() == pytest.approx((-1.025, 1.025, -1.025, 1.025))
            if quantity == "snr":
                assert ax.get_title() == "Expected source S/N"
        with pytest.raises(ValueError, match="quantity"):
            plot_observation(current, "absent")
