"""Finite central values do not authorize a free-centre NFW cusp derivative."""
import numpy as np
import pytest
from hwoslaps.inference.fit_model import FitArgument, FitComponent, FitGalaxy, FitModel, fixed, uniform
from hwoslaps.inference.halo_gradients import guard_halo_cusp_gradient


def _model(free_centre):
    centre = (uniform(-1.,1.,truth=0.), uniform(-1.,1.,truth=0.)) if free_centre else (fixed(0.),fixed(0.))
    profile = FitComponent('hwoslaps.scene.halo_profiles:NFWSph', (
        FitArgument('centre',centre,pair=True),
        FitArgument('kappa_s',(uniform(.01,.1,truth=.05),),pair=False),
        FitArgument('scale_radius',(fixed(1.),),pair=False)))
    return FitModel('subhalo',(FitGalaxy('lens',.2,(('subhalo',profile),)),))


def test_exact_cusp_rejects_only_geometry_gradient(box_objective):
    model = _model(True)
    objective = box_objective(model.lower, model.upper, lambda x:float(x@x), lambda x:2*x)
    guarded = guard_halo_cusp_gradient(objective, model, [[0.,0.],[.1,.2]])
    exact = (model.truth-model.lower)/(model.upper-model.lower)
    assert np.all(np.isfinite(guarded.residual(exact)))
    with pytest.raises(ValueError, match='geometry derivative is undefined at the central cusp'):
        guarded.value_and_gradient(exact)
    nearby = exact.copy(); nearby[0] += 1.e-10
    assert np.all(np.isfinite(guarded.value_and_gradient(nearby)[1]))
    model = _model(False)
    objective = box_objective(model.lower,model.upper,lambda x:float(x@x),lambda x:2*x)
    guarded = guard_halo_cusp_gradient(objective, model, [[0.,0.]])
    assert guarded is objective
    assert np.all(np.isfinite(guarded.value_and_gradient(np.array([.5]))[1]))
