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

from astropy import units as u


# Distance conversions
PC_TO_M: float = float((1 * u.pc).to(u.m).value)
"""Meters per parsec (pc → m)."""

KPC_TO_M: float = float((1 * u.kpc).to(u.m).value)
"""Meters per kiloparsec (kpc → m)."""

MPC_TO_M: float = float((1 * u.Mpc).to(u.m).value)
"""Meters per megaparsec (Mpc → m)."""


# Miscellaneous helpers used in multiple modules
KM_TO_M: float = 1000.0
"""Meters per kilometer (km → m)."""

ARCSEC_PER_RAD: float = float((1 * u.rad).to(u.arcsec).value)
"""Arcseconds per radian (rad → arcsec)."""


