"""Photon-bin and optical combination oracles at the chromatic owner boundary."""

import math

import numpy as np
import pytest

from hwoslaps.optics.chromatic import effective_kernel, sed_weights
from hwoslaps.optics.kernels import DetectorPSF
from hwoslaps.spectra.bandpass import build_bandpass, parse_bandpass
from hwoslaps.spectra.photometry import detected_flux_per_m2
from hwoslaps.spectra.sed import build_sed, parse_sed


def top_hat(throughput=.21):
    return build_bandpass(parse_bandpass({"kind":"top_hat","min_nm":450.,"max_nm":550.,
                                         "throughput":throughput},"band"))


@pytest.mark.parametrize("kind",["flat_fnu","flat_flambda"])
@pytest.mark.parametrize("nodes_nm",[[500.],[470.,490.,530.],[300.,470.,490.,530.,700.]])
def test_sed_bin_weights_follow_independent_photon_equations(kind,nodes_nm):
    band=top_hat();sed=build_sed(parse_sed({"kind":kind},"sed"),redshift=.6)
    weights=sed_weights(band,sed,np.array(nodes_nm)/1.e9)
    # Voronoi edges by hand, clipped to the declared support.
    interior=np.clip((np.array(nodes_nm[1:])+np.array(nodes_nm[:-1]))/2,450.,550.)
    edges=np.array([450.,*interior,550.])/1.e9
    expected=np.log(edges[1:]/edges[:-1]) if kind=="flat_fnu" else edges[1:]**2-edges[:-1]**2
    expected/=expected.sum()
    np.testing.assert_allclose(weights.normalized,expected,rtol=1.e-8,atol=0.)
    np.testing.assert_array_equal(weights.bin_edges_m,edges)
    # Mathematical photon measure includes the single recorded common factor.
    measured=weights.rates.sum()*math.exp(weights.log_rate_scale)
    expected_integral=detected_flux_per_m2(sed,1.,band)*6.62607015e-34/1.e-26
    assert measured==pytest.approx(expected_integral,rel=1.e-12,abs=0.)
    with pytest.raises(ValueError):weights.rates[0]=0.


@pytest.mark.parametrize("frame,redshift",[("observed",0.),("rest",.5)])
def test_narrow_table_line_is_assigned_to_exactly_one_provider_bin(frame,redshift,tmp_path):
    band=top_hat();factor=1+redshift if frame=="rest" else 1.
    grid=band.wavelengths_m*1.e9;index=np.searchsorted(grid,500.);left,right=grid[index-1:index+1]
    knots=[400.,left+.2*(right-left),left+.5*(right-left),left+.8*(right-left),600.]
    path=tmp_path/"line.npz";np.savez(path,wave=np.array(knots)/factor,value=[0.,0.,1.,0.,0.])
    sed=build_sed(parse_sed({"kind":"table","path":str(path),"wavelength_key":"wave","value_key":"value",
        "wavelength_unit":"nm","quantity":"fnu","frame":frame},"sed"),redshift=redshift)
    weights=sed_weights(band,sed,np.array([455.,475.,500.,525.,545.])/1.e9)
    np.testing.assert_array_equal(weights.normalized,[0.,0.,1.,0.,0.])
    assert weights.rates.sum()*math.exp(weights.log_rate_scale)==pytest.approx(
        detected_flux_per_m2(sed,1.,band)*6.62607015e-34/1.e-26,rel=1.e-12,abs=0.)


@pytest.mark.parametrize("quantity",["fnu","flambda"])
@pytest.mark.parametrize("normalization",[1.e308,1.e-310])
def test_arbitrary_table_normalization_preserves_relative_weights_and_effective_kernel(quantity,normalization,tmp_path):
    band=top_hat(.5);nodes=np.array([460.,500.,540.])/1.e9
    kernels=[]
    for i in range(3):
        values=np.zeros((3,3));values[1,1]=.6;values[i,0]=.4
        kernels.append(DetectorPSF.from_array(values,.03,normalize=False))
    spectra=[]
    for name,scale in (("ordinary",1.),("rescaled",normalization)):
        path=tmp_path/f"{name}.npz";np.savez(path,wave=[400.,600.],value=[scale,scale])
        spectra.append(build_sed(parse_sed({"kind":"table","path":str(path),"wavelength_key":"wave",
            "value_key":"value","wavelength_unit":"nm","quantity":quantity},"sed"),redshift=0.))
    ordinary,rescaled=[sed_weights(band,sed,nodes) for sed in spectra]
    np.testing.assert_allclose(rescaled.normalized,ordinary.normalized,rtol=1.e-12,atol=0.)
    actual=effective_kernel(kernels,rescaled,.03,source={})
    expected=effective_kernel(kernels,ordinary,.03,source={})
    np.testing.assert_allclose(actual.kernel,expected.kernel,rtol=1.e-12,atol=0.)
    assert rescaled.to_mapping()["log_rate_scale"]==rescaled.log_rate_scale


def test_effective_kernel_keeps_node_order_and_the_original_one_node_object():
    band=top_hat();sed=build_sed(parse_sed({"kind":"flat_fnu"},"sed"),redshift=0.)
    values=[np.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]),np.full((3,3),1/9)]
    kernels=[DetectorPSF.from_array(value,.03,normalize=False) for value in values]
    singleton=sed_weights(band,sed,[500.e-9])
    assert effective_kernel(kernels[:1],singleton,.03,source={"unused":True}) is kernels[0]
    weights=sed_weights(band,sed,[460.e-9,530.e-9])
    edge=495.e-9;first=math.log(edge/450.e-9)/math.log(550./450.)
    expected=first*values[0]+(1-first)*values[1]
    combined=effective_kernel(kernels,weights,.03,source={"label":"two nodes"})
    np.testing.assert_allclose(combined.kernel,expected,rtol=1.e-12,atol=0.)
    assert combined.source["captured_power_fraction"] is None
