"""Registry model transport, physical fit boxes and real persistent JAX fitness."""

import math

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.inference.fit_model import autofit_model
from hwoslaps.inference.hypotheses import build_role_models
from hwoslaps.inference.settings import FitSpec, PriorWidths
from hwoslaps.scene.parameters import scene_parameters

pytestmark = pytest.mark.backend


def _mass(kind="PowerLaw", **values):
    return {"type":kind,"centre":[0.,0.],"einstein_radius":.8,"ell_comps":[.05,.0],
            **({"slope":2.08} if kind=="PowerLaw" else {}),**values}


def _models(prepared, use_jax=False, prior=None, free=None):
    if free is None:
        free=[p.name for p in prepared.nuisances.parameters if p.kind=="scene"]
    return build_role_models(prepared.scene,prepared.hypothesis(1.e8,(.4,-.6)),free,
                             FitSpec(mode="fixed_template",prior_widths=prior or PriorWidths()),use_jax=use_jax)


@pytest.mark.parametrize("kind",["PowerLaw","Isothermal"])
def test_new_lens_models_at_truth_render_truth_and_share_priors(kind,prepared_forecast_factory):
    import autolens as al
    from hwoslaps.scene.builder import build_scene

    overrides={"scene":{"lens":{"mass":{"mass":_mass(kind,multipoles={"m3":[.02,-.01],"m4":[-.03,.01]}),
                  "shear":{"type":"ExternalShear","gamma_1":.08,"gamma_2":-.05}}},
        "source":{"light":{"light":{"type":"Sersic","centre":[-.03,.08],"ell_comps":[.14516129,.25142673],
                  "intensity":1.,"effective_radius":.12,"sersic_index":2.5},
                  "second":{"type":"Exponential","centre":[.04,-.06],"ell_comps":[.08,.03],"intensity":.3,"effective_radius":.18}}}}}
    prepared=prepared_forecast_factory(overrides)
    models=_models(prepared)
    expected_mass_count=12 if kind=="PowerLaw" else 11
    for role in ("smooth","subhalo"):
        model=getattr(models,role); converted=autofit_model(model)
        assert converted.prior_count == len(scene_parameters(prepared.scene.spec))
        assert len([name for name in model.parameter_names if ".lens." in name]) == expected_mass_count
        main=converted.galaxies.lens.mass
        for order in ("m3","m4"):
            companion=getattr(converted.galaxies.lens,"mass_multipole_"+order)
            assert companion.centre.centre_0 is main.centre.centre_0
            assert companion.centre.centre_1 is main.centre.centre_1
            assert companion.einstein_radius is main.einstein_radius
            if kind=="PowerLaw": assert companion.slope is main.slope
            else: assert companion.slope == 2.
        instance=converted.instance_from_vector(vector=model.truth.tolist())
        tracer=al.Tracer(galaxies=list(instance.galaxies),cosmology=prepared.scene.cosmology.autogalaxy())
        truth=build_scene(prepared.scene.spec,prepared.scene.cosmology,
                          subhalo=None if role=="smooth" else prepared.hypothesis(1.e8,(.4,-.6)))
        np.testing.assert_array_equal(tracer.image_2d_from(grid=truth.grid).native,
                                      truth.tracer.image_2d_from(grid=truth.grid).native)
        median=converted.instance_from_prior_medians()
        median_tracer=al.Tracer(galaxies=list(median.galaxies),cosmology=prepared.scene.cosmology.autogalaxy())
        np.testing.assert_allclose(median_tracer.image_2d_from(grid=truth.grid).native,
                                   truth.tracer.image_2d_from(grid=truth.grid).native,rtol=1e-12,atol=0.)
    bare_free=[p.name for p in scene_parameters(prepared.scene.spec) if p.name.startswith("lens.mass.mass.")]
    assert autofit_model(_models(prepared,free=bare_free).smooth).prior_count == (10 if kind=="PowerLaw" else 9)


@pytest.mark.parametrize("kind,q,accepted",[("PowerLaw",.6,True),("PowerLaw",.52,False),("Isothermal",.3,True)])
def test_jax_power_law_box_floor_is_specific_to_series(kind,q,accepted,prepared_forecast_factory):
    prepared=prepared_forecast_factory({"scene":{"lens":{"mass":{"mass":_mass(kind,ell_comps=[0.,(1-q)/(1+q)])}}}})
    if accepted: _models(prepared,use_jax=True)
    else:
        with pytest.raises(ConfigError,match="JAX PowerLaw box minimum axis ratio"):
            _models(prepared,use_jax=True)
        _models(prepared,use_jax=False)


@pytest.mark.parametrize("kind,q,pair,width,accepted,maximum",[
    ("Isothermal",1.,[.64,.64],.01,True,None),
    ("Isothermal",1.,[.66,.66],.01,False,None),
    ("PowerLaw",1.,[.64,.64],.01,False,None),
    ("Isothermal",.8,[0.,-.7],.01,True,None),
    ("Isothermal",.8,[0.,-.7],.1,False,.063193),
    ("Isothermal",.8,[0.,-.8],.01,False,0.),
])
def test_multipole_boxes_keep_every_corner_positive(kind,q,pair,width,accepted,maximum,prepared_forecast_factory):
    mass=_mass(kind,ell_comps=[0.,(1-q)/(1+q)],multipoles={"m4":pair})
    if kind=="PowerLaw":mass["slope"]=2.
    prepared=prepared_forecast_factory({"scene":{"lens":{"mass":{"mass":mass}}}})
    prior=PriorWidths.from_mapping({"rules":{"lens.multipole":{"half_width":width}}})
    if accepted:
        model=_models(prepared,prior=prior).smooth
        # Independent angular minima at all 16 corners of ell and m4; include slope endpoints if free.
        from itertools import product
        phi=np.arange(7200)*2*np.pi/7200
        vals=[p for p in scene_parameters(prepared.scene.spec) if p.name.startswith("lens.mass.mass.")]
        defs={p.definition.name:p for p in vals}
        axes=[prior.rule("lens",defs[name].definition.kind).box(defs[name].value,defs[name].definition.domain)
              for name in ("ell_comp_1","ell_comp_2","multipole_m4_1","multipole_m4_2")]
        slopes=prior.rule("lens","slope").box(2.,defs["slope"].definition.domain) if kind=="PowerLaw" else (2.,)
        for e1,e2,c1,c2 in product(*axes):
            e=math.hypot(e1,e2);qq=(1-e)/(1+e)
            if kind=="Isothermal":qq=min(qq,.99999)
            angle=.5*math.atan2(e1,e2)
            eta=np.sqrt(np.cos(phi-angle)**2+np.sin(phi-angle)**2/qq**2)
            for slope in slopes:
                kappa=(3-slope)/(1+qq)*(.8/eta)**(slope-1)+.5*.8**(slope-1)*(c2*np.cos(4*phi)+c1*np.sin(4*phi))
                assert np.min(kappa)>0
    else:
        with pytest.raises(ConfigError) as error:_models(prepared,prior=prior)
        assert error.value.path=="scene.lens.mass.mass"
        if maximum==0.:
            assert "no symmetric multipole half width fits" in str(error.value)
            assert "ellipticity" in str(error.value)
        elif maximum is not None:
            import re
            value=float(re.search(r"half width is ([0-9.e+-]+)",str(error.value)).group(1))
            # Solve independent quadratic w^2 + (.7+w)^2 = q_min^2.
            e=math.hypot(.02,1/9+.02);qmin=(1-e)/(1+e)
            expected=(-.7+math.sqrt(2*qmin*qmin-.7*.7))/2
            assert value==pytest.approx(expected,rel=1e-6,abs=0.)


def test_shear_corner_names_the_largest_fitting_width(prepared_forecast_factory):
    prepared=prepared_forecast_factory({"scene":{"lens":{"mass":{"shear":{"type":"ExternalShear","gamma_1":.7,"gamma_2":.7}}}}})
    with pytest.raises(ConfigError,match="largest symmetric shear half width is 0.00710678"):
        _models(prepared)


def _family_overrides(family,image_asset):
    overrides={"scene":{"grid":{"shape":[20,20]}}}
    if family in {"power_law","multipoles","shear","all"}:
        mass=_mass()
        if family in {"multipoles","all"}:mass["multipoles"]={"m3":[.02,-.01],"m4":[-.03,.01]}
        overrides["scene"]["lens"]={"mass":{"mass":mass}}
        if family in {"shear","all"}:overrides["scene"]["lens"]["mass"]["shear"]={"type":"ExternalShear","gamma_1":.08,"gamma_2":-.05}
    if family in {"sersic","all"}:overrides["scene"]["source"]={"light":{"light":{"type":"Sersic","centre":[-.03,.08],
        "ell_comps":[.14516129,.25142673],"intensity":1.,"effective_radius":.12,"sersic_index":4.}}}
    elif family=="image":overrides["scene"]["source"]={"light":{"light":{"type":"Image","centre":[-.03,.08],
        "asset_path":str(image_asset),"rotation_deg":12.,"total_flux":.2}}}
    return overrides


def _fitness_pair(prepared):
    from hwoslaps.inference.api import prepare_case
    from autofit.non_linear.fitness import Fitness

    halo=prepared.hypothesis(1.e8,(.1,-.15));fit=FitSpec(mode="fixed_template")
    numpy_case=prepare_case(prepared,halo,prepared.observation,fit=fit,use_jax=False)
    jax_case=prepare_case(prepared,halo,prepared.observation,fit=fit,use_jax=True)
    model=jax_case.autofit_models["smooth"]
    merit=-1.23456789e99
    jfit=Fitness(model=model,analysis=jax_case.analysis,paths=None,fom_is_log_likelihood=True,
                 resample_figure_of_merit=merit,use_jax_vmap=True,batch_size=3)
    nfit=Fitness(model=numpy_case.autofit_models["smooth"],analysis=numpy_case.analysis,paths=None,
                 fom_is_log_likelihood=True,resample_figure_of_merit=merit,use_jax_vmap=False)
    columns=np.linspace(-.03,.03,model.prior_count)
    rows=np.linspace(-.04,.04,3)[:,None]
    a=np.asarray([model.vector_from_unit_vector(row) for row in .40+rows+columns])
    b=np.asarray([model.vector_from_unit_vector(row) for row in .60-rows-columns])
    assert np.all(np.any(a!=b,axis=0))
    return jfit,nfit,merit,a,b


@pytest.mark.parametrize("gpu",[False,pytest.param(True,marks=pytest.mark.xtx_gpu)],ids=["cpu","gpu"])
@pytest.mark.parametrize("family",["power_law","multipoles","shear","sersic","all","image"])
def test_real_persistent_fitness_matches_numpy_for_each_family(gpu,family,prepared_forecast_factory,image_asset):
    import jax

    prepared=prepared_forecast_factory(_family_overrides(family,image_asset))
    jfit,nfit,_,a,b=_fitness_pair(prepared)
    assert jax.default_backend()==("gpu" if gpu else "cpu")
    for batch in (a,b):
        actual=np.asarray(jax.block_until_ready(jfit.call_wrap(batch)))
        expected=np.asarray([nfit.call_wrap(row) for row in batch])
        assert actual.dtype==np.float64 and np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-5)


@pytest.mark.parametrize("gpu",[False,pytest.param(True,marks=pytest.mark.xtx_gpu)],ids=["cpu","gpu"])
def test_nonfinite_vector_resamples_without_poisoning_fitness(gpu,prepared_forecast_factory,image_asset):
    import jax

    prepared=prepared_forecast_factory(_family_overrides("all",image_asset))
    jfit,nfit,merit,a,b=_fitness_pair(prepared)
    assert jax.default_backend()==("gpu" if gpu else "cpu")
    first=np.asarray(jax.block_until_ready(jfit.call_wrap(a)))
    invalid=a.copy();invalid[1,0]=np.nan
    result=np.asarray(jax.block_until_ready(jfit.call_wrap(invalid)))
    assert result[1]==merit
    np.testing.assert_array_equal(result[[0,2]],first[[0,2]])
    recovered=np.asarray(jax.block_until_ready(jfit.call_wrap(b)))
    assert recovered.dtype==np.float64 and np.all(np.isfinite(recovered))
    np.testing.assert_allclose(recovered,[nfit.call_wrap(row) for row in b],rtol=1e-10,atol=1e-5)


@pytest.mark.parametrize("gpu",[False,pytest.param(True,marks=pytest.mark.xtx_gpu)],ids=["cpu","gpu"])
@pytest.mark.parametrize("comps",[(0.,0.),(1.e-12,0.),(1.e-3,2.e-3)])
def test_cartesian_multipole_objective_gradient_matches_independent_fd(gpu,comps,prepared_forecast_factory):
    import jax
    from hwoslaps.inference.api import prepare_case

    prepared=prepared_forecast_factory({"scene":{"grid":{"shape":[20,20]},"lens":{"mass":{"mass":_mass(
        multipoles={"m3":list(comps),"m4":list(comps)})}}}})
    halo=prepared.hypothesis(1.e8,(.1,-.15))
    case=prepare_case(prepared,halo,prepared.observation.draw(7),fit=FitSpec(mode="fixed_template"),use_jax=True)
    assert jax.default_backend()==("gpu" if gpu else "cpu")
    objective=case.objective("smooth");model=case.model("smooth")
    z=(model.truth-model.lower)/(model.upper-model.lower)
    value,gradient=objective.value_and_gradient(z)
    assert math.isfinite(value) and np.all(np.isfinite(gradient))
    # FD values from the independent NumPy likelihood, not the JAX scalar under test.
    ncase=prepare_case(prepared,halo,case.observation,fit=FitSpec(mode="fixed_template"),use_jax=False)
    indices=[i for i,name in enumerate(model.parameter_names) if ".multipole_comps." in name]
    for i in indices:
        step=np.zeros_like(z);step[i]=1.e-4
        upper=ncase.log_likelihood("smooth",objective.to_physical(z+step))
        lower=ncase.log_likelihood("smooth",objective.to_physical(z-step))
        expected=-(upper-lower)/(2.e-4)
        assert gradient[i]==pytest.approx(expected,rel=1e-6,abs=0.)


@pytest.mark.parametrize("n",[.36,.5,.75,1.,2.5,4.,8.])
@pytest.mark.parametrize("ell",[(0.,0.),(.05,.02)])
def test_sersic_primal_and_differential_follow_backend_and_independent_fd(n,ell):
    import autolens as al
    import autogalaxy as ag
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.light_profiles import Sersic

    ensure_jax_x64()
    grid=al.Grid2DIrregular(values=np.array([[.1,.2],[-.15,.08],[.3,-.12],[.05,-.3]]))
    params=np.array([-.03,.08,*ell,1.,.12,n])
    weights=np.array([-2.,1.,.5,3.])
    def image(cls,p,xp):
        profile=cls(centre=(p[0],p[1]),ell_comps=(p[2],p[3]),intensity=p[4],effective_radius=p[5],sersic_index=p[6])
        return xp.asarray(profile.image_2d_from(grid=grid,xp=xp).array)
    np.testing.assert_array_equal(image(Sersic,params,np),image(ag.lp.Sersic,params,np))
    np.testing.assert_array_equal(jax.jit(lambda p:image(Sersic,p,jnp))(params),
                                  jax.jit(lambda p:image(ag.lp.Sersic,p,jnp))(params))
    error,gradient=jax.jit(checkify.checkify(jax.grad(lambda p:jnp.asarray(weights)@image(Sersic,p,jnp))))(params)
    error.throw();assert np.all(np.isfinite(gradient))
    expected=[]
    for i in range(len(params)):
        step=np.zeros_like(params);step[i]=1.e-6
        expected.append(weights@(image(ag.lp.Sersic,params+step,np)-image(ag.lp.Sersic,params-step,np))/(2.e-6))
    np.testing.assert_allclose(gradient,expected,rtol=1e-6,atol=1e-8)
    if ell!=(0.,0.):
        parent=jax.jit(jax.grad(lambda p:jnp.asarray(weights)@image(ag.lp.Sersic,p,jnp)))(params)
        np.testing.assert_allclose(gradient,parent,rtol=2e-14,atol=1e-13)


@pytest.mark.parametrize("n",[.36,.5,.75,1.,2.5])
@pytest.mark.parametrize("ell",[(0.,0.),(.05,.02)])
def test_sersic_centre_gradient_uses_the_actual_index_domain(n,ell):
    import autolens as al
    import autogalaxy as ag
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.light_profiles import Sersic
    from hwoslaps.scene.profiles import sersic_constant

    ensure_jax_x64()
    grid=al.Grid2DIrregular(values=np.array([[-.03,.08]]))
    params=np.array([-.03,.08,*ell,1.,.12,n])
    def value(p,cls=Sersic,xp=jnp):
        profile=cls(centre=(p[0],p[1]),ell_comps=(p[2],p[3]),intensity=p[4],effective_radius=p[5],sersic_index=p[6])
        return xp.sum(profile.image_2d_from(grid=grid,xp=xp).array)
    primal=float(jax.jit(value)(params))
    assert primal==float(jax.jit(lambda p:value(p,ag.lp.Sersic))(params))
    error,gradient=jax.jit(checkify.checkify(jax.grad(value)))(params)
    if n>=1:
        with pytest.raises(Exception,match="zero-radius source-centre sample for n >= 1"):error.throw()
    else:
        error.throw();gradient=np.asarray(gradient)
        assert np.all(np.isfinite(gradient))
        np.testing.assert_array_equal(gradient[:4],np.zeros(4))
        assert gradient[5]==0.
        b=sersic_constant(n)
        derivative_b=2.-4/(405*n*n)-92/(25515*n**3)-393/(1148175*n**4)+4*2194697/(30690717750*n**5)
        assert gradient[4]==pytest.approx(math.exp(b),rel=1e-12,abs=0.)
        assert gradient[6]==pytest.approx(math.exp(b)*derivative_b,rel=1e-12,abs=0.)
        for i in (0,1,6):
            step=np.zeros_like(params);step[i]=1.e-6
            fd=(float(value(params+step,ag.lp.Sersic,np))-float(value(params-step,ag.lp.Sersic,np)))/(2.e-6)
            assert gradient[i]==pytest.approx(fd,rel=1e-6,abs=1e-8)


def _image_samples():
    rows,cols=np.indices((8,10),dtype=float)
    sb=np.exp(-.5*(((rows-2.7)/1.1)**2+((cols-5.6)/1.4)**2))
    return sb/(.2**2*sb.sum())


def _image_profile(params):
    from hwoslaps.scene.image_profile import ImageLightProfile
    return ImageLightProfile(centre=(params[0],params[1]),rotation_deg=params[4],pixel_scale_arcsec=.2,
                             sb=_image_samples(),total_flux=1.7,flux_scale=params[2],size_scale=params[3])


@pytest.mark.parametrize("gpu",[False,pytest.param(True,marks=pytest.mark.xtx_gpu)],ids=["cpu","gpu"])
@pytest.mark.parametrize("size",[.85,1.2])
def test_image_jax_retains_the_complete_zero_pad_rotation_matrix(gpu,size):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from hwoslaps.inference.backend import ensure_jax_x64

    ensure_jax_x64()
    assert jax.default_backend()==("gpu" if gpu else "cpu")
    rows=np.array([2.25,0.,7.,-.5,7.5,-1.,8.,-1.01,8.01,3.3])
    cols=np.array([4.6,0.,9.,4.2,5.1,2.,7.,3.,6.,10.01])
    theta=np.deg2rad(37.3);u=(cols-4.5)*.2*size;v=(rows-3.5)*.2*size
    points=np.column_stack((.13+u*np.sin(theta)+v*np.cos(theta),-.21+u*np.cos(theta)-v*np.sin(theta)))
    profile=_image_profile(np.array([.13,-.21,1.15,size,37.3]))
    grid=al.Grid2DIrregular(values=points)
    actual=np.asarray(profile.image_2d_from(grid=grid,xp=jnp).array)
    expected=[];sb=_image_samples()
    for row,col in zip(rows,cols):
        r0,c0=math.floor(row),math.floor(col);wr,wc=row-r0,col-c0
        def pixel(r,c):return sb[r,c] if 0<=r<8 and 0<=c<10 else 0.
        expected.append(1.7*1.15*((1-wr)*(1-wc)*pixel(r0,c0)+(1-wr)*wc*pixel(r0,c0+1)
                                +wr*(1-wc)*pixel(r0+1,c0)+wr*wc*pixel(r0+1,c0+1)))
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(actual[7:],np.zeros(3))


@pytest.mark.parametrize("gpu",[False,pytest.param(True,marks=pytest.mark.xtx_gpu)],ids=["cpu","gpu"])
def test_image_persistent_jit_rotation_gradient_and_warm_transfer_contract(gpu):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from hwoslaps.inference.backend import ensure_jax_x64

    ensure_jax_x64()
    assert jax.default_backend()==("gpu" if gpu else "cpu")
    params=np.array([.13,-.21,1.15,.92,37.3])
    grid=al.Grid2D.uniform(shape_native=(3,3),pixel_scales=.11,origin=(.13,-.21))
    def image(p):return _image_profile(p).image_2d_from(grid=grid,xp=jnp).array
    persistent=jax.jit(image);device=jax.device_put(params)
    first=np.asarray(persistent(device).block_until_ready())
    changed=params+np.array([.006,-.004,.05,.03,2.])
    second=np.asarray(persistent(jax.device_put(changed)).block_until_ready())
    assert not np.array_equal(first,second)
    with jax.transfer_guard("disallow"):persistent(device).block_until_ready()
    gradient=np.asarray(jax.grad(lambda p:jnp.sum(image(p)))(device))
    expected=[]
    for i in range(5):
        step=np.zeros(5);step[i]=1.e-6
        upper=np.asarray(_image_profile(params+step).image_2d_from(grid=grid,xp=np).array).sum()
        lower=np.asarray(_image_profile(params-step).image_2d_from(grid=grid,xp=np).array).sum()
        expected.append((upper-lower)/(2.e-6))
    assert np.all(np.isfinite(gradient)) and abs(gradient[4])>0.
    np.testing.assert_allclose(gradient,expected,rtol=1e-6,atol=1e-7)



def test_isothermal_single_m4_has_seven_mass_priors(prepared_forecast_factory):
    prepared=prepared_forecast_factory({"scene":{"lens":{"mass":{"mass":_mass("Isothermal",multipoles={"m4":[.02,-.01]})}}}})
    free=[p.name for p in scene_parameters(prepared.scene.spec) if p.name.startswith("lens.mass.")]
    converted=autofit_model(_models(prepared,free=free).smooth)
    assert converted.prior_count==7
    assert converted.galaxies.lens.mass_multipole_m4.slope==2.


@pytest.mark.parametrize("n",[.36,.5,.75,1.,2.5,4.,8.])
def test_centred_sersic_fixed_geometry_keeps_finite_index_derivative(n):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.light_profiles import Sersic
    from hwoslaps.scene.profiles import sersic_constant

    ensure_jax_x64()
    grid=al.Grid2DIrregular(values=np.array([[-.03,.08]]))
    def value(index):
        profile=Sersic(centre=(-.03,.08),ell_comps=(0.,0.),intensity=1.,effective_radius=.12,sersic_index=index)
        return jnp.sum(profile.image_2d_from(grid=grid,xp=jnp).array)
    error,derivative=jax.jit(checkify.checkify(jax.grad(value)))(n)
    error.throw()
    db=2-4/(405*n*n)-92/(25515*n**3)-393/(1148175*n**4)+4*2194697/(30690717750*n**5)
    assert float(derivative)==pytest.approx(math.exp(sersic_constant(n))*db,rel=1e-12,abs=0.)


@pytest.mark.parametrize("n",[.5,2.5])
def test_fixed_sersic_source_geometry_still_checks_moving_traced_rays(n):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.light_profiles import Sersic

    ensure_jax_x64()
    source=Sersic(centre=(-.03,.08),ell_comps=(0.,0.),intensity=1.,effective_radius=.12,sersic_index=n)
    def value(ray):
        grid=al.Grid2DIrregular(values=ray[None,:],xp=jnp)
        return jnp.sum(source.image_2d_from(grid=grid,xp=jnp).array)
    error,gradient=jax.jit(checkify.checkify(jax.grad(value)))(jnp.array([-.03,.08]))
    if n>=1:
        with pytest.raises(Exception,match="zero-radius source-centre sample for n >= 1"):error.throw()
    else:
        error.throw();np.testing.assert_array_equal(gradient,np.zeros(2))



def _powerlaw_boundary_case(mapping, tmp_path, slope, *, free_slope=False, fixed_shape=False):
    from hwoslaps.config.schema import parse_config
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.inference.api import prepare_case
    from hwoslaps.simulation import simulate

    kernel=np.array([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])/16.
    path=tmp_path/"boundary_kernel.npy";np.save(path,kernel)
    mapping["psf"]["truth"].update(path=str(path))
    mapping["scene"]["grid"]["shape"]=[20,20]
    mapping["scene"]["lens"]["mass"]["mass"].update(type="PowerLaw",slope=slope,ell_comps=[0.,0.])
    mapping["scene"]["source"]["light"]["light"]["ell_comps"]=[.14516129,.25142673]
    fixed=["lens.mass.mass.centre_*","source.light.*"]
    if not free_slope:fixed.append("lens.mass.mass.slope")
    if fixed_shape:fixed.append("lens.mass.mass.ell_comp_*")
    mapping["forecast"]["nuisances"]={"fixed":fixed,"background_offset":False}
    prepared=prepare_forecast(parse_config(mapping))
    try:
        observation=simulate(prepared,subhalo=None,noise_seed=7)
        case=prepare_case(prepared,prepared.hypothesis(1.e8,(.1,-.15)),observation,
                          fit=FitSpec(mode="fixed_template"),use_jax=True)
    except BaseException:
        prepared.close();raise
    return prepared,case


@pytest.mark.parametrize("free_slope",[False,True],ids=["fixed_slope","actual_uniform_slope"])
def test_circular_nonisothermal_powerlaw_refuses_only_gradient_at_actual_slope(free_slope,minimal_mapping,tmp_path):
    prepared,case=_powerlaw_boundary_case(minimal_mapping,tmp_path,2. if free_slope else 2.08,free_slope=free_slope)
    try:
        model=case.model("smooth");objective=case.objective("smooth")
        physical=model.truth.copy()
        if free_slope:
            index=next(i for i,name in enumerate(model.parameter_names) if name.endswith(".slope"))
            physical[index]=2.02
        z=(physical-model.lower)/(model.upper-model.lower)
        assert math.isfinite(case.log_likelihood("smooth",physical))
        assert np.all(np.isfinite(objective.residual(z)))
        with pytest.raises(ValueError,match=r"PowerLaw.*normalization cusp"):
            objective.value_and_gradient(z)
        # The same supported vectors with a small noncircular shape still refine.
        index=next(i for i,name in enumerate(model.parameter_names) if name.endswith("ell_comps_0"))
        physical[index]=.005;z=(physical-model.lower)/(model.upper-model.lower)
        value,gradient=objective.value_and_gradient(z)
        assert math.isfinite(value) and np.all(np.isfinite(gradient))
    finally:prepared.close()


def test_circular_powerlaw_fixed_ellipse_keeps_other_parameter_gradients(minimal_mapping,tmp_path):
    prepared,case=_powerlaw_boundary_case(minimal_mapping,tmp_path,2.08,fixed_shape=True)
    try:
        model=case.model("smooth");objective=case.objective("smooth")
        assert len(model.parameter_names)==1 and model.parameter_names[0].endswith("einstein_radius")
        z=(model.truth-model.lower)/(model.upper-model.lower)
        value,gradient=objective.value_and_gradient(z)
        assert math.isfinite(value) and np.all(np.isfinite(gradient))
        step=np.ones_like(z)*1e-4
        expected=-(case.log_likelihood("smooth",objective.to_physical(z+step))-
                   case.log_likelihood("smooth",objective.to_physical(z-step)))/(2e-4)
        assert gradient[0]==pytest.approx(expected,rel=1e-6,abs=0.)
    finally:prepared.close()


@pytest.mark.parametrize("slope",[2.,2.08],ids=["regular_isothermal_powerlaw","normalization_cusp"])
def test_circular_powerlaw_one_sided_values_classify_the_shape_differential(slope,minimal_mapping,tmp_path):
    from hwoslaps.inference.api import prepare_case

    prepared,case=_powerlaw_boundary_case(minimal_mapping,tmp_path,slope)
    try:
        numpy_case=prepare_case(prepared,case.hypothesis,case.observation,fit=case.fit,use_jax=False)
        model=case.model("smooth");physical=model.truth.copy()
        axes=[i for i,name in enumerate(model.parameter_names) if "ell_comps_" in name]
        f0=.5*numpy_case.chi_squared("smooth",physical)
        differences=[]
        for h in (1.e-4,1.e-5,1.e-6):
            right=[];left=[]
            for i in axes:
                step=np.zeros_like(physical);step[i]=h
                right.append((.5*numpy_case.chi_squared("smooth",physical+step)-f0)/h)
                left.append((f0-.5*numpy_case.chi_squared("smooth",physical-step))/h)
            differences.append(np.asarray(right)-np.asarray(left))
            print(f"PowerLaw gamma={slope} h={h} right={right} left={left} gap={differences[-1]}")
        z=(physical-model.lower)/(model.upper-model.lower)
        if slope!=2.:
            assert np.linalg.norm(differences[-1])>.5*np.linalg.norm(differences[0])
            with pytest.raises(ValueError,match="normalization cusp"):
                case.objective("smooth").value_and_gradient(z)
        else:
            assert np.linalg.norm(differences[-1])<.05*np.linalg.norm(differences[0])
            value,gradient=case.objective("smooth").value_and_gradient(z)
            print(f"PowerLaw gamma2 primitive value={value} gradient={gradient}")
            assert math.isfinite(value) and np.all(np.isfinite(gradient))
    finally:prepared.close()


@pytest.mark.parametrize("ell,slope",[((0.,0.),2.),((.05,.02),2.),((-.02,.04),2.08)])
def test_powerlaw_adapter_preserves_parent_bits_and_all_regular_direction_derivatives(ell,slope):
    import autolens as al
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.mass_profiles import PowerLaw

    ensure_jax_x64()
    points=np.array([[.13,.21],[-.27,.15],[.41,-.31],[-.11,-.38]])
    grid=al.Grid2DIrregular(values=points)
    params=np.array([.03,-.02,*ell,.8,slope])
    weights=np.array([[.7,-.3],[-1.,.2],[.4,.9],[-.2,.5]])
    def values(cls,p,xp):
        profile=cls(centre=(p[0],p[1]),ell_comps=(p[2],p[3]),einstein_radius=p[4],slope=p[5])
        return xp.asarray(profile.deflections_yx_2d_from(grid=grid,xp=xp).array)
    np.testing.assert_array_equal(values(PowerLaw,params,np),values(al.mp.PowerLaw,params,np))
    np.testing.assert_array_equal(jax.jit(lambda p:values(PowerLaw,p,jnp))(params),
                                  jax.jit(lambda p:values(al.mp.PowerLaw,p,jnp))(params))
    error,gradient=jax.jit(checkify.checkify(jax.grad(lambda p:jnp.sum(jnp.asarray(weights)*values(PowerLaw,p,jnp)))))(params)
    error.throw();assert np.all(np.isfinite(gradient))
    expected=[]
    for i in range(6):
        step=np.zeros(6);step[i]=1e-6
        expected.append(np.sum(weights*(values(al.mp.PowerLaw,params+step,np)-values(al.mp.PowerLaw,params-step,np)))/(2e-6))
    np.testing.assert_allclose(gradient,expected,rtol=1e-6,atol=1e-8)
    if ell!=(0.,0.):
        parent=jax.jit(jax.grad(lambda p:jnp.sum(jnp.asarray(weights)*values(al.mp.PowerLaw,p,jnp))))(params)
        np.testing.assert_array_equal(gradient,parent)
    else:
        # Independent m=2 potential from linearizing A2kappa at gamma2:
        # psi2=-theta/3*r*(e2*cos2phi+e1*sin2phi).
        y,x=(points-params[:2]).T;phi=np.arctan2(y,x);a=-params[4]/3
        harmonic=[]
        for e1,e2 in ((1.,0.),(0.,1.)):
            ar=a*(e2*np.cos(2*phi)+e1*np.sin(2*phi))
            ap=-2*a*(e2*np.sin(2*phi)-e1*np.cos(2*phi))
            harmonic.append(np.sum(weights*np.column_stack((ar*np.sin(phi)+ap*np.cos(phi),
                                                             ar*np.cos(phi)-ap*np.sin(phi)))))
        np.testing.assert_allclose(np.asarray(gradient)[2:4],harmonic,rtol=1e-12,atol=0.)
        # Source/ray-coordinate directions also remain the parent derivative.
        profile=PowerLaw(centre=tuple(params[:2]),ell_comps=ell,einstein_radius=params[4],slope=slope)
        parent_profile=al.mp.PowerLaw(centre=tuple(params[:2]),ell_comps=ell,einstein_radius=params[4],slope=slope)
        def ray_value(cls_profile,ray):
            moving=al.Grid2DIrregular(values=ray,xp=jnp)
            return jnp.sum(jnp.asarray(weights)*cls_profile.deflections_yx_2d_from(grid=moving,xp=jnp).array)
        error,actual=jax.jit(checkify.checkify(jax.grad(lambda ray:ray_value(profile,ray))))(points)
        error.throw()
        expected=jax.jit(jax.grad(lambda ray:ray_value(parent_profile,ray)))(points)
        np.testing.assert_array_equal(actual,expected)


def test_powerlaw_regular_circle_refuses_singular_mass_centre_geometry_gradient():
    import autolens as al
    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify
    from hwoslaps.inference.backend import ensure_jax_x64
    from hwoslaps.inference.mass_profiles import PowerLaw

    ensure_jax_x64()
    grid=al.Grid2DIrregular(values=np.array([[.03,-.02]]))
    def value(ell,cls=PowerLaw):
        profile=cls(centre=(.03,-.02),ell_comps=(ell[0],ell[1]),einstein_radius=.8,slope=2.)
        return profile.deflections_yx_2d_from(grid=grid,xp=jnp).array
    ell=jnp.zeros(2)
    np.testing.assert_array_equal(jax.jit(value)(ell),jax.jit(lambda e:value(e,al.mp.PowerLaw))(ell))
    error,derivative=jax.jit(checkify.checkify(jax.jacrev(value)))(ell)
    with pytest.raises(Exception,match="exactly coincident mass-centre sample with active geometry"):
        error.throw()
