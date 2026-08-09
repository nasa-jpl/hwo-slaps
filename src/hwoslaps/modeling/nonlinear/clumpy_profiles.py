"""Composite fit-side light profile for transformed clumpy sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    ClumpyTransformedSource: Any


def _as_float(value: Any) -> Any:
    """Convert concrete scalars to float while preserving traced values."""
    if isinstance(value, (int, float, np.generic)):
        return float(value)
    return value


@dataclass(frozen=True)
class ClumpyTemplateContext:
    """Immutable internal structure for a transformed clumpy source.

    Parameters
    ----------
    host : `tuple` [`float`, ...]
        Host ``(ell_0, ell_1, intensity, reff, sersic_index)`` values.
    host_centre : `tuple` [`float`, `float`]
        Truth host centre in ``(y, x)`` coordinates.
    clumps : `tuple` [`tuple`, ...]
        Per-clump relative centre, shape, intensity, size, and index values.
    context_hash : `str`
        First 16 hexadecimal characters of the canonical context SHA-256.
    """

    host: Tuple[float, ...]
    host_centre: Tuple[float, float]
    clumps: Tuple[Tuple[float, ...], ...]
    context_hash: str


def _build_profile_class() -> None:
    """Build and publish the lazy PyAutoLens composite profile class."""
    if "ClumpyTransformedSource" in globals():
        return

    import autogalaxy as ag

    class ClumpyTransformedSource(ag.LightProfile):
        """Jointly transformed host-and-clump Sersic light profile.

        Parameters
        ----------
        centre : `tuple` [`float`, `float`], optional
            Shared host centre in ``(y, x)`` coordinates.
        flux_scale : `float`, optional
            Joint intensity multiplier.
        size_scale : `float`, optional
            Joint size and clump-offset multiplier.
        host_ell_comps : `tuple` [`float`, `float`], optional
            Host ellipticity components.
        host_intensity : `float`, optional
            Unscaled host intensity.
        host_effective_radius : `float`, optional
            Unscaled host effective radius.
        host_sersic_index : `float`, optional
            Host Sersic index.
        template_context : `ClumpyTemplateContext`, optional
            Fixed clump structure and truth provenance.
        """

        def __init__(
            self,
            centre=(0.0, 0.0),
            flux_scale=1.0,
            size_scale=1.0,
            host_ell_comps=(0.0, 0.0),
            host_intensity=1.0,
            host_effective_radius=0.1,
            host_sersic_index=1.0,
            template_context=None,
        ):
            if template_context is None:
                raise ValueError(
                    "template_context is required for a clumpy source"
                )
            centre = tuple(_as_float(value) for value in centre)
            host_ell_comps = tuple(
                _as_float(value) for value in host_ell_comps
            )
            flux_scale = _as_float(flux_scale)
            size_scale = _as_float(size_scale)
            host_intensity = _as_float(host_intensity)
            host_effective_radius = _as_float(host_effective_radius)
            host_sersic_index = _as_float(host_sersic_index)
            super().__init__(
                centre=centre,
                ell_comps=host_ell_comps,
                intensity=host_intensity * flux_scale,
            )
            self.flux_scale = flux_scale
            self.size_scale = size_scale
            self.host_ell_comps = host_ell_comps
            self.host_intensity = host_intensity
            self.host_effective_radius = host_effective_radius
            self.host_sersic_index = host_sersic_index
            self.template_context = template_context

            host = ag.lp.Sersic(
                centre=centre,
                ell_comps=host_ell_comps,
                intensity=host_intensity * flux_scale,
                effective_radius=host_effective_radius * size_scale,
                sersic_index=host_sersic_index,
            )
            clumps = []
            for values in template_context.clumps:
                offset_y, offset_x, ell_0, ell_1, intensity, reff, index = (
                    values
                )
                clumps.append(
                    ag.lp.Sersic(
                        centre=(
                            centre[0] + size_scale * offset_y,
                            centre[1] + size_scale * offset_x,
                        ),
                        ell_comps=(ell_0, ell_1),
                        intensity=intensity * flux_scale,
                        effective_radius=reff * size_scale,
                        sersic_index=index,
                    )
                )
            self.host_profile = host
            self.clump_profiles = tuple(clumps)
            self.components = (host, *clumps)

        def image_2d_from(
            self,
            grid,
            xp=np,
            operated_only=None,
            **kwargs,
        ):
            """Evaluate and sum all internal Sersic components.

            Parameters
            ----------
            grid : `autoarray.type.Grid2DLike`
                Sky coordinates for image evaluation.
            xp : `module`, optional
                Array namespace accepted for profile API compatibility.
            operated_only : `bool`, optional
                Operated-profile selector passed through to each component.

            Returns
            -------
            image : `autoarray.Array2D`
                Summed surface brightness of the host and clumps.
            """
            images = [
                component.image_2d_from(
                    grid=grid,
                    xp=xp,
                    operated_only=operated_only,
                    **kwargs,
                )
                for component in self.components
            ]
            image = images[0]
            for component_image in images[1:]:
                image = image + component_image
            return image

        def image_2d_via_radii_from(self, grid_radii, xp=np, **kwargs):
            """Reject radial evaluation of a non-radial composite.

            Parameters
            ----------
            grid_radii : `object`
                Radial coordinates, which are unsupported.
            xp : `module`, optional
                Array namespace accepted for profile API compatibility.

            Raises
            ------
            NotImplementedError
                Always raised because the composite is not radial.
            """
            raise NotImplementedError(
                "ClumpyTransformedSource is not radially symmetric"
            )

    ClumpyTransformedSource.__module__ = __name__
    ClumpyTransformedSource.__qualname__ = "ClumpyTransformedSource"
    globals()["ClumpyTransformedSource"] = ClumpyTransformedSource


def __getattr__(name: str) -> Any:
    """Resolve the lazy module-level composite profile class by name."""
    if name == "ClumpyTransformedSource":
        _build_profile_class()
        return globals()[name]
    raise AttributeError(name)


__all__ = ["ClumpyTemplateContext", "ClumpyTransformedSource"]
