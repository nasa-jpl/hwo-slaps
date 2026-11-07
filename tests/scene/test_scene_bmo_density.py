"""The documented BMO n=1 family agrees with an independent density integral."""

import numpy as np
import pytest
from scipy.integrate import quad

from hwoslaps.scene.halos import bmo_mass_fraction


@pytest.mark.parametrize("tau", [0.1, 1.0, 3.0, 40.0, 1000.0])
def test_bmo_n1_total_mass_is_integral_of_named_density(tau):
    # x^2 rho(x), relative to rho_s, for rho_NFW(x) * tau^2/(tau^2+x^2).
    integral, error = quad(lambda x: x / (1 + x)**2 * tau**2 / (tau**2 + x**2),
                           0.0, np.inf, epsabs=1e-12, epsrel=1e-12)
    assert error < 1e-10
    assert bmo_mass_fraction(tau) == pytest.approx(integral, rel=2e-12)
