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
    np.testing.assert_allclose(weights.normalized,expected,rtol=1.e-10,atol=0.)
    np.testing.assert_allclose(weights.bin_edges_m,edges,rtol=1.e-15,atol=0.)
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


def circular_provider(band, count):
    from hwoslaps.constants import ARCSEC_PER_RAD
    from hwoslaps.optics.providers import build_psf_provider, parse_psf
    pitch=500.e-9/(2*2.)*ARCSEC_PER_RAD
    spec=parse_psf({"truth":{"kind":"optical","pupil":{"kind":"circular","diameter_m":2.,
        "pixels":256,"supersampling":4},"focal_length_m":20.,"wavelength_samples":count,
        "detector_oversampling":3,"kernel_shape":[31,31]}}).truth
    return build_psf_provider(spec,pixel_scale_arcsec=pitch,wavelengths_m=band.nodes(count)),pitch


@pytest.mark.backend
def test_broadband_circular_kernel_matches_weighted_analytic_airy():
    from scipy.special import j1
    from hwoslaps.constants import ARCSEC_PER_RAD
    band=top_hat();sed=build_sed(parse_sed({"kind":"flat_fnu"},"sed"),redshift=0.)
    provider,pitch=circular_provider(band,8)
    actual=effective_kernel(provider.kernels(),sed_weights(band,sed,provider.wavelengths_m),pitch,source={}).kernel
    # Independent Airy probability density, pixel-integrated on the specified 3x3 centres.
    # 64 wavelength bins carry exact flat-fnu dln(lambda) weights, without the spectral helper.
    # Normalize each independent analytic node on the same support, as the detector kernel requires.
    edges=np.linspace(450.e-9,550.e-9,65);nodes=(edges[:-1]+edges[1:])/2
    weights=np.log(edges[1:]/edges[:-1])/math.log(550./450.)
    yy,xx=np.mgrid[-15:16,-15:16];expected=np.zeros((31,31))
    omega=(pitch/ARCSEC_PER_RAD/3)**2
    for wavelength,weight in zip(nodes,weights):
        node=np.zeros_like(expected)
        for oy in (-1/3,0.,1/3):
            for ox in (-1/3,0.,1/3):
                angle=np.hypot(yy+oy,xx+ox)*pitch/ARCSEC_PER_RAD
                v=math.pi*2.*angle/wavelength
                airy=np.ones_like(v);nonzero=v!=0
                airy[nonzero]=(2*j1(v[nonzero])/v[nonzero])**2
                node+=math.pi*2.**2/(4*wavelength**2)*airy*omega
        expected+=weight*node/node.sum()
    assert np.max(np.abs(actual-expected))/expected.max()<5.e-4


def ramp_band(tmp_path,eps):
    width=500.*eps;low,high=500.-width/2,500.+width/2
    path=tmp_path/f"ramp-{eps}.npz";np.savez(path,wave=[low,high],value=[.1,.7])
    band=build_bandpass(parse_bandpass({"kind":"table","path":str(path),"wavelength_key":"wave",
        "value_key":"value","wavelength_unit":"nm","support_nm":[low,high]},"band"))
    return band,width,low,high


@pytest.mark.backend
def test_photon_weighted_mean_wavelength_is_second_order_accurate(tmp_path):
    from hwoslaps.spectra.photometry import effective_wavelength_m
    sed=build_sed(parse_sed({"kind":"flat_fnu"},"sed"),redshift=0.)
    mean_errors=[];centre_errors=[]
    for eps in (.001,.04,.08,.16):
        band,width,low,high=ramp_band(tmp_path,eps)
        # Integral(T dlambda) / integral(T dlambda/lambda), with the throughput factor cancelled.
        expected_nm=width/((1-1.5*500./width)*math.log(high/low)+1.5)
        mean=effective_wavelength_m(sed,band)
        assert mean*1.e9==pytest.approx(expected_nm,rel=1.e-9,abs=0.)
        provider,pitch=circular_provider(band,24)
        combined=effective_kernel(provider.kernels(),sed_weights(band,sed,provider.wavelengths_m),pitch,source={}).kernel
        at_mean=provider.kernel(mean).kernel;at_centre=provider.kernel(500.e-9).kernel
        residual=np.linalg.norm(combined-at_mean)/np.linalg.norm(at_mean)
        if eps==.001:
            assert residual<1.e-6
        else:
            mean_errors.append(residual)
            centre_errors.append(np.linalg.norm(combined-at_centre)/np.linalg.norm(at_centre))
    assert mean_errors[0]<5.e-4
    assert all(3.5<ratio<4.5 for ratio in np.array(mean_errors[1:])/mean_errors[:-1])
    assert all(1.7<ratio<2.3 for ratio in np.array(centre_errors[1:])/centre_errors[:-1])


def test_unrepresentable_log_measure_is_a_numeric_error_instead_of_zero_response():
    from decimal import Decimal,localcontext
    from hwoslaps.optics.chromatic import NoSpectralResponse
    # The requested common log scale itself exceeds float64's largest finite scalar.
    # These photons are positive, so the one-node metadata consumer must not label them absent.
    with localcontext() as context:
        context.prec=60
        assert Decimal("1e308")*(Decimal(1000)/2).ln()>Decimal(str(np.finfo(float).max))
    band=build_bandpass(parse_bandpass({"kind":"top_hat","min_nm":1.,"max_nm":2.,"throughput":.5},"band"))
    sed=build_sed(parse_sed({"kind":"power_law","index":1.e308},"sed"),redshift=0.)
    with np.errstate(over="ignore"),pytest.raises(ValueError) as caught:
        sed_weights(band,sed,band.nodes(1))
    assert not isinstance(caught.value,NoSpectralResponse)
