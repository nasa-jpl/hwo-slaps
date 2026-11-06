"""Value/geometry-derivative boundary for removable NFW-type central cusps."""
from dataclasses import replace
import numpy as np

_HALO_CUSPS = {'hwoslaps.scene.halo_profiles:NFWSph', 'hwoslaps.scene.halo_profiles:TruncatedNFWSph',
               'hwoslaps.inference.subhalo_classes:NFWM200SubhaloSph',
               'hwoslaps.inference.subhalo_classes:TNFWM200SubhaloSph'}


def guard_halo_cusp_gradient(objective, model, integration_yx):
    """Reject free-centre derivatives at an exact cusp, retaining finite values."""
    coordinates = np.asarray(integration_yx, dtype=float)
    centres = []
    next_index = 0
    for galaxy in model.galaxies:
        elements = {}
        for name, component in galaxy.components:
            for argument in component.arguments:
                for index, prior in enumerate(argument.elements):
                    key = (name, argument.name, index if argument.pair else None)
                    if prior.kind == 'linked':
                        elements[key] = elements[prior.link]
                    else:
                        elements[key] = (next_index if prior.kind == 'uniform' else None, prior)
                        if prior.kind == 'uniform':
                            next_index += 1
            if component.profile_class in _HALO_CUSPS:
                pair = tuple(elements[(name, 'centre', index)] for index in (0, 1))
                if any(index is not None for index, _ in pair):
                    centres.append((f'galaxies.{galaxy.name}.{name}', pair))
    if not centres:
        return objective
    def value_and_gradient(z):
        physical = objective.to_physical(z)
        for name, pair in centres:
            centre = np.array([physical[index] if index is not None else prior.value for index, prior in pair])
            if np.any(np.all(coordinates == centre[None,:], axis=1)):
                raise ValueError(f'NFW/TNFW halo geometry derivative is undefined at the central cusp of {name}; '
                                 'use value-only sampling, a fixed centre, or a noncoincident position')
        return objective.value_and_gradient(z)
    return replace(objective, value_and_gradient=value_and_gradient)
