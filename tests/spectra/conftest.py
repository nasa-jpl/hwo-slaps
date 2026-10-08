"""Synthetic spectral files shared by the spectral owner tests."""

import numpy as np
import pytest
import yaml


@pytest.fixture
def spectral_file(tmp_path):
    def write(wavelengths_nm, values, *, suffix="npz", unit="nm", name="curve"):
        factor = {"nm": 1.0, "angstrom": 10.0, "um": 1.0e-3, "m": 1.0e-9}[unit]
        wavelengths = np.asarray(wavelengths_nm, dtype=float) * factor
        response = np.asarray(values, dtype=float)
        path = tmp_path / f"{name}.{suffix}"
        if suffix in ("yaml", "yml"):
            path.write_text(yaml.safe_dump({"wave": wavelengths.tolist(), "response": response.tolist()}), encoding="utf-8")
        elif suffix == "csv":
            np.savetxt(path, np.column_stack((wavelengths, response)), delimiter=",", header="wave,response", comments="")
        else:
            np.savez(path, wave=wavelengths, response=response)
        return {"path": str(path), "wavelength_key": "wave", "value_key": "response", "wavelength_unit": unit}
    return write
