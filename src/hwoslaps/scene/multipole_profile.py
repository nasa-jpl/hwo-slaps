"""Cartesian multipole deflections, including finite autodiff at zero components.

Only the deflection method differs from AutoGalaxy: C and S are linear in the two
Cartesian components, avoiding the amplitude/angle singularity at (0, 0).
"""

import autoarray as aa
import autolens as al
import numpy as np
from autogalaxy.profiles.mass.total.power_law_multipole import radial_and_angle_grid_from

__all__ = ["CartesianPowerLawMultipole"]


class CartesianPowerLawMultipole(al.mp.PowerLawMultipole):
    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid, xp=np, **kwargs):
        r, phi = radial_and_angle_grid_from(grid=grid, xp=xp)
        c1, c2 = self.multipole_comps
        m, slope = self.m, self.slope
        a = self.einstein_radius ** (slope - 1.0) / ((3.0 - slope) ** 2 - m ** 2)
        cm, sm = xp.cos(m * phi), xp.sin(m * phi)
        cosine = c2 * cm + c1 * sm
        sine = c2 * sm - c1 * cm
        radial = (3.0 - slope) * a * r ** (2.0 - slope) * cosine
        angular = -m * a * r ** (2.0 - slope) * sine
        return xp.stack(self.jacobian(a_r=radial, a_angle=angular, polar_angle_grid=phi, xp=xp), axis=-1)
