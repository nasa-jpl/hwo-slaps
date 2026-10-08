"""New registry families through the actual prepared detector/Fisher engines."""

import numpy as np
import pytest

pytestmark=pytest.mark.backend


@pytest.mark.parametrize("gpu",[False,pytest.param(True,marks=pytest.mark.xtx_gpu)],ids=["cpu","gpu"])
@pytest.mark.parametrize("family",["power_law","multipoles","shear","sersic","multi_light"])
def test_new_model_fisher_engines_have_reference_statistics(gpu,family,minimal_mapping,image_asset):
    import jax
    from hwoslaps.fisher.api import Execution,forecast,prepare_forecast

    if family in {"power_law","multipoles"}:
        mass=minimal_mapping["scene"]["lens"]["mass"]["mass"]
        mass.update(type="PowerLaw",slope=2.08)
        if family=="multipoles":mass["multipoles"]={"m3":[.02,-.01],"m4":[-.03,.01]}
    elif family=="shear":minimal_mapping["scene"]["lens"]["mass"]["shear"]={"type":"ExternalShear","gamma_1":.08,"gamma_2":-.05}
    else:
        minimal_mapping["scene"]["source"]["light"]["light"].update(type="Sersic",sersic_index=4.)
        if family=="multi_light":
            minimal_mapping["scene"]["lens"]["light"]={"bulge":{"type":"Sersic","centre":[0.,0.],"ell_comps":[.07,.03],"intensity":.1,"effective_radius":.4,"sersic_index":3.}}
            minimal_mapping["scene"]["source"]["light"]["disk"]={"type":"Exponential","centre":[.04,-.06],"ell_comps":[.08,.03],"intensity":.3,"effective_radius":.18}
            minimal_mapping["scene"]["source"]["light"]["clumps"]={"type":"Image","centre":[-.02,.05],"asset_path":str(image_asset),"rotation_deg":30.,"total_flux":.1}
    assert jax.default_backend()==("gpu" if gpu else "cpu")
    with prepare_forecast(minimal_mapping) as reference,prepare_forecast(minimal_mapping,execution=Execution(engine="jax")) as candidate:
        expected=forecast(reference,masses_msun=[1.e8,3.e8]);actual=forecast(candidate,masses_msun=[1.e8,3.e8])
        for field in ("fisher_raw","fisher_profiled","sigma_amplitude"):
            np.testing.assert_allclose(getattr(actual,field),getattr(expected,field),rtol=1e-6,atol=0.)
