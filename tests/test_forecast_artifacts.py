"""The real forecast NPZ format, including absent and undefined diagnostics."""

import numpy as np
import pytest

from hwoslaps.modeling.forecast_results import ForecastResult


def _result(*, mismatch=False):
    fields = dict(
        masses_msun=np.array([1.0e7, 1.0e8]),
        positions_yx=np.array([[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4]]),
        q_asimov=np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]),
        fisher_raw=np.array([[2.0, 5.0, 10.0], [17.0, 26.0, 37.0]]),
        fisher_profiled=np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]),
        sigma_amplitude=np.array([[1.0, 0.5, 1.0 / 3.0], [0.25, 0.2, 1.0 / 6.0]]),
        degradation=np.array([[2.0, 1.25, 10.0 / 9.0], [17.0 / 16.0, 1.04, 37.0 / 36.0]]),
        runtime_provenance={"engine": "reference", "workers": 1, "inputs": {"source": "abc"}},
    )
    if mismatch:
        fields.update(
            amplitude_hat=np.array([[1.0, -2.0, np.nan], [0.0, 0.5, 1.0]]),
            q_mismatch=np.array([[2.0, 3.0, np.nan], [0.0, 1.0, 4.0]]),
            z_mismatch=np.array([[2.0 ** 0.5, -(3.0 ** 0.5), np.nan], [0.0, 1.0, 2.0]]),
            amplitude_spurious=np.array([[0.2, -0.4, np.nan], [0.0, 0.1, 0.2]]),
            q_spurious=np.array([[0.1, 0.2, np.nan], [0.0, 0.4, np.inf]]),
            z_spurious=np.array([[0.1 ** 0.5, -(0.2 ** 0.5), np.nan], [0.0, 0.4 ** 0.5, np.inf]]),
        )
    return ForecastResult(**fields)


@pytest.mark.parametrize("mismatch", [False, True])
def test_forecast_npz_roundtrip_preserves_scientific_arrays_and_explicit_metadata(tmp_path, mismatch):
    original = _result(mismatch=mismatch)
    path = original.save_npz(tmp_path / "nested" / "forecast.npz")
    with np.load(path, allow_pickle=False) as stored:
        assert int(stored["schema_version"]) == 1
        assert stored["q_asimov"].shape == (2, 3)
        assert stored["q_asimov"].dtype == np.float64
        assert ("q_mismatch" in stored.files) is mismatch
    loaded = ForecastResult.load_npz(path)
    for name in (
        "masses_msun", "positions_yx", "q_asimov", "fisher_raw", "fisher_profiled",
        "sigma_amplitude", "degradation", "amplitude_hat", "q_mismatch", "z_mismatch",
        "amplitude_spurious", "q_spurious", "z_spurious",
    ):
        expected = getattr(original, name)
        actual = getattr(loaded, name)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)
    assert loaded.runtime_provenance == {"engine": "reference", "workers": 1, "inputs": {"source": "abc"}}
    np.testing.assert_array_equal(loaded.z_asimov, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_forecast_npz_refuses_overwrite_without_altering_existing_bytes(tmp_path):
    path = _result().save_npz(tmp_path / "forecast.npz")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        _result(mismatch=True).save_npz(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".forecast-*"))


@pytest.mark.parametrize("defect", ["schema", "shape", "object", "metadata", "missing"])
def test_forecast_npz_rejects_unknown_or_malformed_public_artifacts(tmp_path, defect):
    source = _result().save_npz(tmp_path / "source.npz")
    with np.load(source, allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    if defect == "schema":
        payload["schema_version"] = np.asarray(99)
    elif defect == "shape":
        payload["q_asimov"] = np.ones((3, 2))
    elif defect == "object":
        payload["q_asimov"] = np.asarray([[object()]], dtype=object)
    elif defect == "metadata":
        payload["runtime_provenance_json"] = np.asarray("[]")
    else:
        del payload["q_asimov"]
    path = tmp_path / "invalid.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        ForecastResult.load_npz(path)
