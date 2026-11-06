"""Named spectral columns, unit conversion and actual-byte identity."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.spectra.tables import parse_table, read_table


@pytest.mark.parametrize("suffix", ["yaml", "yml", "csv", "npz"])
@pytest.mark.parametrize("unit", ["nm", "angstrom", "um", "m"])
def test_table_formats_and_units_preserve_named_curve(spectral_file, suffix, unit):
    mapping = spectral_file([450.0, 500.0, 550.0], [0.2, 0.4, 0.7], suffix=suffix, unit=unit)
    table = read_table(parse_table(mapping, "table"))
    np.testing.assert_allclose(table.wavelengths_m, np.array([450.0, 500.0, 550.0]) / 1.0e9, rtol=1.0e-15)
    np.testing.assert_array_equal(table.values, [0.2, 0.4, 0.7])
    assert table.digest == hashlib.sha256(Path(mapping["path"]).read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        table.values[0] = 1.0


@pytest.mark.parametrize("wavelengths,values", [([500.0], [0.2]), ([500.0, 500.0], [0.2, 0.3]),
    ([550.0, 450.0], [0.2, 0.3]), ([0.0, 550.0], [0.2, 0.3]), ([450.0, float("nan")], [0.2, 0.3]),
    ([450.0, 550.0], [0.2, float("inf")]), ([450.0, 550.0], [0.2])])
def test_invalid_table_arrays_name_file_and_columns(spectral_file, wavelengths, values):
    mapping = spectral_file(wavelengths, values)
    with pytest.raises(ValueError, match="curve.npz.*wave.*response"):
        read_table(parse_table(mapping, "table"))


def test_missing_column_is_refused(spectral_file):
    mapping = spectral_file([450.0, 550.0], [0.2, 0.3])
    mapping["value_key"] = "absent"
    with pytest.raises(ValueError, match="missing column.*absent"):
        read_table(parse_table(mapping, "table"))
