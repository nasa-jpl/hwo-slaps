"""Shared physical constants and unit conversions.

This module centralizes constants used across the package to avoid
hardcoded numeric values and ensure consistency. Where possible, values
are derived from `astropy.units` to follow standardized definitions.

Notes
-----
The megaparsec-to-meter conversion has previously been inconsistently
hardcoded across the codebase (e.g., 3.086e22, 3.0857e22). This module
exposes a single authoritative value derived via `astropy.units`, which
is approximately 3.08567758e22 meters per megaparsec.
"""

from astropy import constants as const
from astropy import units as u

# Physical constants (SI)
PLANCK_J_S: float = float(const.h.value)
"""Planck constant h in J s (exact SI value, 6.62607015e-34)."""

C_M_S: float = float(const.c.value)
"""Speed of light in vacuum in m/s."""

C_KM_S: float = float(const.c.to(u.km / u.s).value)
"""Speed of light in vacuum in km/s."""

G_SI: float = float(const.G.value)
"""Newtonian gravitational constant in m^3 kg^-1 s^-2."""

MSUN_KG: float = float((1 * u.Msun).to(u.kg).value)
"""Nominal solar mass in kg."""

# Spectral flux density
JANSKY_SI: float = float((1 * u.Jy).to(u.W / (u.m**2 * u.Hz)).value)
"""One jansky in W m^-2 Hz^-1 (1e-26)."""

AB_ZERO_POINT_JY: float = 3631.0
"""AB magnitude zero point in janskys: m_AB = -2.5 log10(f_nu / 3631 Jy).

The rounded Oke and Gunn value used by the HWO reference photometry; astropy's
``u.ABflux`` (3630.78 Jy, from -48.60 in cgs) is a different number.
"""

# Distance conversions
PC_TO_M: float = float((1 * u.pc).to(u.m).value)
"""Meters per parsec (pc → m)."""

KPC_TO_M: float = float((1 * u.kpc).to(u.m).value)
"""Meters per kiloparsec (kpc → m)."""

MPC_TO_M: float = float((1 * u.Mpc).to(u.m).value)
"""Meters per megaparsec (Mpc → m)."""

M_TO_KPC: float = float((1 * u.m).to(u.kpc).value)
"""Kiloparsecs per meter (m → kpc), the scale astropy applies in ``.to(u.kpc)``."""


# Miscellaneous helpers used in multiple modules
KM_TO_M: float = 1000.0
"""Meters per kilometer (km → m)."""

ARCSEC_PER_RAD: float = float((1 * u.rad).to(u.arcsec).value)
"""Arcseconds per radian (rad → arcsec)."""
