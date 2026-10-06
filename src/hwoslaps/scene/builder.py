"""Scene builder: the AutoLens tracer of a scene and its light images per light group.

Plane assembly rule, shared by the truth scene and the fit models (``inference.hypotheses``):
1. The galaxy ``lens`` holds, in order, the lens mass components (each with its layout
   companions), the lens light components, the perturbers at the lens redshift as
   ``perturber_<index>``, then the subhalo as ``subhalo`` when it is at the lens redshift.
   AutoLens sums a galaxy's deflections in attribute order, so this is the summation order.
2. Halos at another redshift follow: one galaxy ``perturbers_<k>`` per distinct perturber
   redshift in ascending redshift (components ``perturber_<index>``), then an off-plane
   subhalo as component ``subhalo`` of the galaxy ``subhalo_plane``. No galaxy is named
   ``subhalo``: AutoLens re-traces the centre of a galaxy of that name from the image plane.
3. The galaxy ``source`` is last.
The tracer receives the galaxies in this order and groups them into planes by redshift.

Every light image is the block mean of the light on the over-sampled grid (detected e-/s
per pixel sample). A scene with one source light group and no lens light is rendered by
``tracer.image_2d_from``, the route of the RASTI-26-183 paper code that the paper anchors pin.
"""

from __future__ import annotations

import types
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cosmology import Cosmology
from .halos import Halo
from .image_source import ImageAsset
from .profiles import PROFILE_TYPES, instantiate
from ..spectra.sed import SED
from .spec import ComponentSpec, GridSpec, LightGroup, SceneSpec

__all__ = ["Scene", "build_scene", "native_sampling_variation", "render_component_unlensed"]

_GRID_TEMPLATES: OrderedDict[tuple[tuple[int, int], float, int], Any] = OrderedDict()
_GRID_TEMPLATE_LIMIT = 4


@dataclass(frozen=True, eq=False)
class Scene:
    """A built scene: its spec, halos, AutoLens grid and tracer, and the light of each group.

    Scenes compare and hash by identity.
    """

    spec: SceneSpec
    cosmology: Cosmology
    subhalo: Halo | None
    perturbers: tuple[Halo, ...]
    grid: Any
    tracer: Any
    light_groups: Mapping[str, LightGroup]
    light_images: Mapping[str, np.ndarray]
    light_profiles: Mapping[str, tuple[Any, ...]]

    @property
    def plane_count(self) -> int:
        """Distinct redshifts among the lens, the source and every halo."""
        halos = self.perturbers + (() if self.subhalo is None else (self.subhalo,))
        return len({self.spec.lens.redshift, self.spec.source.redshift, *(halo.redshift for halo in halos)})

    @property
    def pixel_scale_arcsec(self) -> float:
        return self.spec.grid.pixel_scale_arcsec

    def einstein_radius(self) -> float:
        """The Einstein radius of the single lens mass component that has one."""
        return self.spec.einstein_radius()


def _set_writeable(value: Any, writeable: bool, seen: set[int]) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        value.flags.writeable = writeable
    elif isinstance(value, dict):
        for item in value.values():
            _set_writeable(item, writeable, seen)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _set_writeable(item, writeable, seen)
    elif hasattr(value, "__dict__"):
        for item in vars(value).values():
            _set_writeable(item, writeable, seen)


def _over_sampled_grid(grid: GridSpec) -> Any:
    """A detached AutoLens ``Grid2D`` of the grid spec.

    ``Grid2D.uniform`` builds the uniform sub-pixel grid with a Python loop over pixels, which
    takes seconds on a 500 x 500 image; one read-only template per geometry (a bounded memo of
    four) is copied for every caller.
    """
    import autolens as al

    key = (tuple(grid.shape), float(grid.pixel_scale_arcsec), int(grid.over_sample_size))
    template = _GRID_TEMPLATES.pop(key, None)
    if template is None:
        template = al.Grid2D.uniform(shape_native=key[0], pixel_scales=key[1], over_sample_size=key[2])
        _set_writeable(template, False, set())
    _GRID_TEMPLATES[key] = template
    while len(_GRID_TEMPLATES) > _GRID_TEMPLATE_LIMIT:
        _GRID_TEMPLATES.popitem(last=False)
    mask = deepcopy(template.mask)
    over_sampler = deepcopy(template.over_sampler)
    over_sampler.mask = mask
    if hasattr(over_sampler.sub_size, "mask"):
        over_sampler.sub_size.mask = mask
    copy = al.Grid2D(values=np.array(template.array, dtype=float, copy=True), mask=mask,
                     over_sample_size=np.array(template.over_sample_size, copy=True),
                     over_sampled=deepcopy(template.over_sampled), over_sampler=over_sampler)
    _set_writeable(copy, True, set())
    return copy


def _check_halo(halo: Halo, spec: SceneSpec, cosmology: Cosmology, label: str) -> None:
    if halo.source_redshift != spec.source.redshift or halo.cosmology != cosmology:
        raise ValueError(f"{label} lenses a source at z = {halo.source_redshift} in {halo.cosmology!r}; the scene's "
                         f"source is at z = {spec.source.redshift} in {cosmology!r}")


def _read_only(image: Any) -> np.ndarray:
    array = np.array(image.native.array, dtype=float)
    array.flags.writeable = False
    return array


def build_scene(spec: SceneSpec, cosmology: Cosmology, *, subhalo: Halo | None, perturbers: Sequence[Halo] = (),
                assets: Mapping[str, ImageAsset] | None = None, loaded_seds: Mapping[str, SED] | None = None) -> Scene:
    """Build the tracer and light images of ``spec`` with ``subhalo`` (or none) and the realized perturbers.

    ``assets`` maps an absolute asset path to its loaded ``ImageAsset``; when given, no file is read.
    """
    import autofit as af
    import autolens as al

    perturbers = tuple(perturbers)
    for index, halo in enumerate(perturbers):
        _check_halo(halo, spec, cosmology, f"perturber {index}")
    if subhalo is not None:
        _check_halo(subhalo, spec, cosmology, "the subhalo")
    # Profiles become attributes of the truth galaxy and of the fit's galaxy model by keyword: a name
    # that shadows a galaxy method breaks the truth galaxy, one that shadows a model attribute drops
    # the component from the fit model without an error.
    owners = {"al.Galaxy": al.Galaxy(redshift=spec.lens.redshift),
              "af.Model(al.Galaxy)": af.Model(al.Galaxy, redshift=spec.lens.redshift)}

    def profiles_of(components: Sequence[ComponentSpec]) -> dict[str, Any]:
        profiles: dict[str, Any] = {}
        for component in components:
            for attribute, profile in instantiate(component, assets=assets).items():
                for label, owner in owners.items():
                    if hasattr(owner, attribute):
                        raise ValueError(f"scene.{component.plane}.{component.role}.{component.name}: the galaxy "
                                         f"attribute {attribute!r} shadows an attribute of {label}; rename it")
                profiles[attribute] = profile
        return profiles

    lens_redshift = spec.lens.redshift
    lens_light, source_light = profiles_of(spec.lens.light), profiles_of(spec.source.light)
    attributes: dict[str, dict[str, Any]] = {"lens": {**profiles_of(spec.lens.mass), **lens_light}}
    redshifts = {"lens": lens_redshift}
    off_plane = sorted({halo.redshift for halo in perturbers} - {lens_redshift})
    for k, redshift in enumerate(off_plane):
        attributes[f"perturbers_{k}"], redshifts[f"perturbers_{k}"] = {}, redshift
    for index, halo in enumerate(perturbers):
        galaxy = "lens" if halo.redshift == lens_redshift else f"perturbers_{off_plane.index(halo.redshift)}"
        attributes[galaxy][f"perturber_{index}"] = halo.autolens_profile()
    if subhalo is not None and subhalo.redshift == lens_redshift:
        attributes["lens"]["subhalo"] = subhalo.autolens_profile()
    elif subhalo is not None:
        attributes["subhalo_plane"] = {"subhalo": subhalo.autolens_profile()}
        redshifts["subhalo_plane"] = subhalo.redshift
    attributes["source"], redshifts["source"] = source_light, spec.source.redshift
    galaxies = {name: al.Galaxy(redshift=redshifts[name], **profiles) for name, profiles in attributes.items()}

    grid = _over_sampled_grid(spec.grid)
    tracer = al.Tracer(galaxies=list(galaxies.values()), cosmology=cosmology.autogalaxy())
    groups = spec.light_groups(loaded_seds=loaded_seds)
    if set(groups) <= {"lens", "source"}:
        light_profiles = {"lens": tuple(lens_light.values())} if lens_light else {}
        light_profiles["source"] = tuple(source_light.values())
        if list(groups) == ["source"]:
            images = {"source": _read_only(tracer.image_2d_from(grid=grid))}
        else:
            traced = tracer.traced_grid_2d_list_from(grid=grid)
            images = {plane: _read_only(galaxies[plane].image_2d_from(
                grid=traced[tracer.plane_redshifts.index(redshifts[plane])])) for plane in ("lens", "source")}
    else:
        traced = tracer.traced_grid_2d_list_from(grid=grid)
        light_profiles, images = {}, {}
        for key, group in groups.items():
            galaxy_spec = spec.lens if group.plane == "lens" else spec.source
            profiles = lens_light if group.plane == "lens" else source_light
            selected = tuple(profiles[component.name + layout.suffix]
                             for component in galaxy_spec.light if component.name in group.components
                             for layout in PROFILE_TYPES[component.type].layout(component.values))
            light_profiles[key] = selected
            plane_grid = traced[tracer.plane_redshifts.index(redshifts[group.plane])]
            image = _read_only(selected[0].image_2d_from(grid=plane_grid))
            for profile in selected[1:]:
                image = image + _read_only(profile.image_2d_from(grid=plane_grid))
            image.setflags(write=False)
            images[key] = image
    for key, image in images.items():
        if not np.all(np.isfinite(image)):
            raise ValueError(f"the {key} light image has non-finite values")
    return Scene(spec=spec, cosmology=cosmology, subhalo=subhalo, perturbers=perturbers, grid=grid, tracer=tracer,
                 light_groups=groups, light_images=types.MappingProxyType(images),
                 light_profiles=types.MappingProxyType(light_profiles))


def native_sampling_variation(scene: Scene) -> Mapping[str, float]:
    """Relative within-pixel variation of each light group's lensed light.

    For the over-sampled samples ``s_ju`` of pixel ``j`` (block mean ``m_j``, the native image),
    ``sqrt(sum_j sum_u (s_ju - m_j)^2) / sqrt(sum_j sum_u m_j^2)``. It is zero by construction
    at ``over_sample_size: 1`` and cannot see structure within a pixel there.
    """
    traced = scene.tracer.traced_grid_2d_list_from(grid=scene.grid)
    samples_per_pixel = scene.spec.grid.over_sample_size**2
    redshifts = {"lens": scene.spec.lens.redshift, "source": scene.spec.source.redshift}
    variations = {}
    for key, group in scene.light_groups.items():
        sub_grid = traced[scene.tracer.plane_redshifts.index(redshifts[group.plane])].over_sampled
        samples = sum(np.asarray(profile.image_2d_from(grid=sub_grid), dtype=float)
                      for profile in scene.light_profiles[key]).reshape(-1, samples_per_pixel)
        means = samples.mean(axis=1, keepdims=True)
        power = float(np.sum(means**2)) * samples_per_pixel
        if not power > 0.0:
            raise ValueError(f"light group {key!r} has no light on the grid")
        variations[key] = float(np.sqrt(np.sum((samples - means) ** 2) / power))
    return types.MappingProxyType(variations)


def render_component_unlensed(component: ComponentSpec, grid: GridSpec, *, assets: Mapping[str, ImageAsset] | None = None) -> np.ndarray:
    """A light component on the over-sampled grid without lensing, block-meaned: e-/s per pixel."""
    if component.role != "light":
        raise ValueError(f"component {component.name!r} is a {component.role} component; only light renders")
    (profile,) = instantiate(component, assets=assets).values()
    return _read_only(profile.image_2d_from(grid=_over_sampled_grid(grid)))
