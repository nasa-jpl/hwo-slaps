"""New macro columns equal the lens/source chain rule on independent point formulae."""

import math

import numpy as np
import pytest

pytestmark=pytest.mark.backend


@pytest.mark.parametrize("name",["gamma_1","gamma_2","multipole_m3_1","multipole_m4_2","slope"])
def test_macro_columns_equal_the_lens_source_chain_rule(name,minimal_mapping,tmp_path):
    import autolens as al
    from hwoslaps.fisher.api import prepare_forecast

    delta=np.zeros((3,3));delta[1,1]=1.;path=tmp_path/"delta.npy";np.save(path,delta)
    minimal_mapping["psf"]["truth"].update(path=str(path))
    minimal_mapping["scene"]["lens"]["mass"]={
        "mass":{"type":"PowerLaw","centre":[.01,-.02],"ell_comps":[0.,0.],"einstein_radius":.8,"slope":2.08,
                "multipoles":{"m3":[.02,-.01],"m4":[-.03,.01]}},
        "shear":{"type":"ExternalShear","gamma_1":.08,"gamma_2":-.05}}
    # Slope's simple radial oracle needs the spherical base alone.
    if name=="slope":del minimal_mapping["scene"]["lens"]["mass"]["mass"]["multipoles"]
    minimal_mapping["scene"]["source"]["light"]["light"].update(type="Sersic",sersic_index=.5)
    minimal_mapping["forecast"]["nuisances"]={"steps":{kind:1e-6 for kind in ("slope","shear","multipole")}}
    with prepare_forecast(minimal_mapping) as prepared:
        scene=prepared.scene;points=np.asarray(scene.grid.over_sampled)
        beta=np.asarray(scene.tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(values=points))[-1])
        source=scene.light_profiles["source"][0]
        n=.5;b=2*n-1/3+4/(405*n)+46/(25515*n*n)+131/(1148175*n**3)-2194697/(30690717750*n**4)
        def brightness(p):
            r2=np.sum((p-source.centre)**2,axis=1)
            return source.intensity*np.exp(-b*(r2/source.effective_radius**2-1))
        gradient=[]
        for axis in (0,1):
            step=np.zeros_like(beta);step[:,axis]=1e-7
            gradient.append((brightness(beta+step)-brightness(beta-step))/(2e-7))
        gradient=np.column_stack(gradient)
        y,x=points.T
        if name=="gamma_1":alpha=np.column_stack((-y,x));component="shear"
        elif name=="gamma_2":alpha=np.column_stack((x,y));component="shear"
        else:
            y,x=(points-[.01,-.02]).T;r=np.hypot(y,x);phi=np.arctan2(y,x);slope=2.08
            if name=="slope":
                alpha=.8**(slope-1)*r[:,None]**(1-slope)*np.column_stack((y,x))*np.log(.8/r)[:,None]
            else:
                m=3 if "m3" in name else 4
                angle=math.pi/(2*m) if name.endswith("_1") else 0.
                a=.8**(slope-1)/((3-slope)**2-m*m)
                ar=(3-slope)*a*r**(2-slope)*np.cos(m*(phi-angle))
                ap=-m*a*r**(2-slope)*np.sin(m*(phi-angle))
                alpha=np.column_stack((ar*np.sin(phi)+ap*np.cos(phi),ar*np.cos(phi)-ap*np.sin(phi)))
            component="mass"
        derivative=-np.sum(gradient*alpha,axis=1)
        expected=np.asarray(scene.grid.over_sampler.binned_array_2d_from(array=derivative).native)
        exposure=prepared.observation.exposure;expected*=exposure.exposure_time_s/exposure.detector.gain_e_per_adu
        actual=prepared.nuisances.images[prepared.nuisances.names.index(f"lens.mass.{component}.{name}")]
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<=1e-6
