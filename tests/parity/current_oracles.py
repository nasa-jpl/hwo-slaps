"""Independent current-policy checks beside immutable historical fixtures.

The profile oracle uses LAPACK pivoted Householder QR, not the engine's SVD,
normal equations, residual-basis implementation, or stored forecast outputs.
Only the physical forward images, nuisance derivatives and noise are inputs.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from scipy.linalg.lapack import get_lapack_funcs


class AugmentedQrOracle:
    def __init__(self, design, precision):
        self.n, self.p = design.shape
        augmented = np.vstack((design, np.diag(np.sqrt(precision))))
        # Extended-precision norms independently normalize the physical columns.
        scale = np.sqrt(np.sum(augmented.astype(np.longdouble)**2, axis=0))
        assert np.all(scale > 0), "this representative cohort must have no zero nuisance columns"
        normalized = np.asarray(augmented / scale, dtype=float)
        singular = np.linalg.svd(normalized, compute_uv=False)
        # QR and a truncated SVD need not select the same near-cutoff subspace.
        # These controls deliberately require an independently resolved full rank.
        assert singular[-1] > 1.e-10 * singular[0], singular
        (self.reflectors, self.tau), triangular, self.pivots = qr(normalized, mode="raw", pivoting=True)
        self.smallest_relative_singular_value = float(singular[-1] / singular[0])
        assert np.all(np.abs(np.diag(triangular)) > 0)

    def residual_coordinates(self, values):
        values = np.asarray(values, dtype=float)
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        augmented = np.asfortranarray(np.vstack((values, np.zeros((self.p, values.shape[1])))))
        transform = get_lapack_funcs("ormqr", (self.reflectors, augmented))
        _, work, info = transform("L", "T", self.reflectors, self.tau, augmented, lwork=-1)
        assert info == 0
        transformed, _, info = transform("L", "T", self.reflectors, self.tau, augmented, lwork=int(work[0]))
        assert info == 0
        residual = transformed[self.p:]
        return residual[:, 0] if vector else residual


def assert_detector_response(preparation):
    observation = preparation.observation
    exposure = observation.exposure
    detector = exposure.detector
    rate = np.asarray(observation.light_rate_e_per_s)
    time, gain = exposure.exposure_time_s, detector.gain_e_per_adu
    sky, dark = exposure.sky_rate_e_per_s*time, detector.dark_current_e_per_s*time
    mean = ((rate*time + sky) + dark)/gain
    variance = ((np.maximum(rate, 0)*time + dark) + sky) + exposure.exposure_count*detector.read_noise_e**2
    np.testing.assert_array_equal(preparation.mean_truth_adu, mean)
    np.testing.assert_array_equal(preparation.sigma_adu, np.sqrt(variance)/gain)
    assert mean.shape == preparation.mask.shape == preparation.scene.spec.grid.shape
    assert np.all(np.isfinite(mean)) and np.all(preparation.sigma_adu > 0)
    support = tuple(max(kernel.shape[axis]//2 for binding in
                        (preparation.psfs.truth_kernels, preparation.psfs.model_kernels)
                        for kernel in binding.kernels) for axis in (0, 1))
    geometry = preparation.record["rendering"]
    assert geometry["padding_yx"] == support or tuple(geometry["padding_yx"]) == support
    assert tuple(geometry["render_shape"]) == tuple(n+2*p for n, p in zip(mean.shape, support))
    assert geometry["boundary"] == "render_exterior_light_then_crop"


def forecast_oracle(preparation, masses, positions):
    mask, sigma = preparation.mask, preparation.sigma_adu[preparation.mask]
    design = np.column_stack([image[mask]/sigma for image in preparation.nuisances.images])
    oracle = AugmentedQrOracle(design, preparation.nuisances.prior_precision)
    baseline = preparation.mean_model_adu
    bias = (preparation.mean_truth_adu-baseline)[mask]/sigma
    bias_residual = oracle.residual_coordinates(bias)
    fields = {name: [] for name in ("fisher_raw", "fisher_profiled", "sigma_amplitude", "degradation")}
    if preparation.psfs.mismatched:
        fields.update({name: [] for name in ("amplitude_hat", "q_mismatch", "z_mismatch",
                                           "amplitude_spurious", "q_spurious", "z_spurious")})
    for mass in masses:
        rows = {name: [] for name in fields}
        for position in positions:
            scene = preparation.renderer.scene(subhalo=preparation.hypothesis(float(mass), tuple(position)))
            model = preparation.renderer.mean_adu(scene, preparation.psfs.model_kernels)
            signal = (model-baseline)[mask]/sigma
            residual = oracle.residual_coordinates(signal)
            raw, information = float(signal@signal), float(residual@residual)
            assert information > 0 and np.isfinite(information)
            rows["fisher_raw"].append(raw)
            rows["fisher_profiled"].append(information)
            rows["sigma_amplitude"].append(1/np.sqrt(information))
            rows["degradation"].append(information/raw)
            if preparation.psfs.mismatched:
                truth = preparation.renderer.mean_adu(scene, preparation.psfs.truth_kernels)
                data_residual = oracle.residual_coordinates((truth-baseline)[mask]/sigma)
                amplitude = float(residual@data_residual/information)
                spurious = float(residual@bias_residual/information)
                for prefix, value in (("mismatch", amplitude), ("spurious", spurious)):
                    rows["amplitude_hat" if prefix == "mismatch" else "amplitude_spurious"].append(value)
                    z = value*np.sqrt(information)
                    rows["z_"+prefix].append(z)
                    rows["q_"+prefix].append(z*z)
        for name in fields:
            fields[name].append(rows[name])
    output = {name: np.asarray(values) for name, values in fields.items()}
    output["q_asimov"] = output["fisher_profiled"]
    return output, oracle.smallest_relative_singular_value


def assert_forecast_statistics(result, expected):
    assert np.all(result.template_status == "valid"), result.template_status
    errors = {}
    for name, values in expected.items():
        actual = np.asarray(getattr(result, name))
        # Two independently evaluated algorithms each target 1e-8 precision.
        # This is a current QR oracle comparison, never a relaxed archive test.
        np.testing.assert_allclose(actual, values, rtol=2.e-8, atol=2.e-10, err_msg=name)
        errors[name] = float(np.max(np.abs(actual-values)))
    return errors


def historical_changes(result, expected, indices, fields):
    changes = {}
    for name in fields:
        old = expected["reference__"+name][:, indices]
        new = np.asarray(getattr(result, name))
        finite = np.isfinite(old) & np.isfinite(new)
        fractional = np.divide(new-old, old, out=np.full_like(new, np.nan), where=finite & (old != 0))
        changes[name] = {"historical": old.tolist(), "current": new.tolist(),
                         "max_absolute_change": float(np.max(np.abs(new[finite]-old[finite]))),
                         "max_fractional_change": float(np.nanmax(np.abs(fractional)))}
    return changes


def write_receipt(tmp_path, name, record, record_property):
    path = Path(tmp_path)/(name+".json")
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    record_property("numerical_policy_receipt", str(path))
    record_property("numerical_policy_summary", json.dumps(record, allow_nan=False))


def gaussian_likelihood_terms(case, role, vector):
    fit = case.fit_at(role, vector)
    data = np.asarray(case.data.imaging.data, dtype=float)
    noise = np.asarray(case.data.imaging.noise_map, dtype=float)
    residual = (data-np.asarray(fit.model_data, dtype=float))/noise
    chi2 = float(residual@residual)
    normalization = float(np.sum(np.log(2*np.pi*noise**2)))
    return {"chi_squared": chi2, "noise_normalization": normalization,
            "log_likelihood": -.5*(chi2+normalization), "figure_of_merit": -.5*(chi2+normalization)}
