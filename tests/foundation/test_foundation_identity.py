"""The one digest format: arrays, canonical mappings, text, files and kernel identities.

Expected digests are SHA-256 of bytes assembled here from the documented format
(``hashlib`` on a header string written out by hand plus C-order little-endian
values), plus the two literals pinned by the specification.
"""

import hashlib
import json
from pathlib import PurePosixPath

import numpy as np
import pytest

from hwoslaps.identity import (
    KernelIdentity, array_digest, canonical_json, file_digest, mapping_digest, native_ready, text_digest,
)

BASE = np.arange(6, dtype="<f8").reshape(2, 3)
WIDE = np.arange(12, dtype="<f8").reshape(2, 6)


def documented(header, values):
    return hashlib.sha256(header.encode("ascii") + np.ascontiguousarray(values).tobytes()).hexdigest()


@pytest.mark.parametrize("values, expected", [
    pytest.param(BASE, "3d347f279a31a36e0afba171c10e70ac0d61f8b2e0c29d4586bc27fdccd167aa", id="pinned-literal"),
    pytest.param(BASE, documented("dtype=<f8;shape=2,3;", BASE), id="c-order"),
    pytest.param(np.asfortranarray(BASE), documented("dtype=<f8;shape=2,3;", BASE), id="fortran-copy"),
    pytest.param(WIDE[:, ::2], documented("dtype=<f8;shape=2,3;", WIDE[:, ::2].copy()), id="strided-view"),
    pytest.param(BASE.astype(">f8"), documented("dtype=<f8;shape=2,3;", BASE), id="big-endian"),
    pytest.param(np.array([[True, False, True]]),
                 documented("dtype=|b1;shape=1,3;", np.array([[1, 0, 1]], dtype=np.uint8)), id="bool"),
    pytest.param(np.array(2.5), documented("dtype=<f8;shape=;", np.array(2.5)), id="0-d"),
    pytest.param(BASE.reshape(3, 2), documented("dtype=<f8;shape=3,2;", BASE), id="shape-change"),
    pytest.param(BASE.astype("<f4"), documented("dtype=<f4;shape=2,3;", BASE.astype("<f4")), id="dtype-change"),
    pytest.param(np.array([-0.0]), documented("dtype=<f8;shape=1;", np.array([-0.0])), id="negative-zero"),
    pytest.param([1, 2, 3], documented("dtype=<i8;shape=3;", np.array([1, 2, 3], dtype="<i8")), id="list-input"),
    pytest.param(np.array([object()]), TypeError, id="object-dtype"),
    pytest.param(np.zeros(2, dtype=[("a", "<f8")]), TypeError, id="structured-dtype"),
])
def test_array_digest_format(values, expected):
    if isinstance(expected, type):
        with pytest.raises(expected):
            array_digest(values)
    else:
        assert array_digest(values) == expected


@pytest.mark.parametrize("value, text", [
    pytest.param({"b": 1, "a": [1.5, None, True]}, '{"a":[1.5,null,true],"b":1}', id="pinned-literal"),
    pytest.param({"b": 2, "a": 1}, '{"a":1,"b":2}', id="key-order"),
    pytest.param({"x": (1, 2), "y": [1, 2]}, '{"x":[1,2],"y":[1,2]}', id="tuple-is-list"),
    pytest.param({"i": np.int64(3), "f": np.float64(0.5), "t": np.bool_(True)}, '{"f":0.5,"i":3,"t":true}',
                 id="numpy-scalars"),
    pytest.param({"z": -0.0}, '{"z":0.0}', id="negative-zero"),
    pytest.param({4: "a", 10: "b"}, '{"10":"b","4":"a"}', id="integer-keys-as-text"),
    pytest.param({"p": PurePosixPath("/data/kernel.npy")}, '{"p":"/data/kernel.npy"}', id="path"),
    pytest.param({"name": "Moliné"}, '{"name":"Molin\\u00e9"}', id="ascii-escape"),
    pytest.param({"f": 0.1, "big": 1e300}, '{"big":1e+300,"f":0.1}', id="shortest-repr"),
])
def test_mapping_digest_canonical_form(value, text):
    assert canonical_json(value) == text
    assert mapping_digest(value) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    native = native_ready(value)
    assert canonical_json(native) == text and mapping_digest(native) == mapping_digest(value)
    if any(isinstance(key, int) for key in value):
        assert native == {4: "a", 10: "b"}
    else:
        assert native == json.loads(text)


@pytest.mark.parametrize("value, error", [
    pytest.param({4: 1, "4": 2}, ValueError, id="integer-text-key-collision"),
    pytest.param({"a": {"b": float("nan")}}, ValueError, id="nan"),
    pytest.param({"a": [float("inf")]}, ValueError, id="inf"),
    pytest.param({"a": np.zeros(2)}, TypeError, id="ndarray"),
    pytest.param({1.5: "x"}, TypeError, id="float-key"),
    pytest.param({"a": {1, 2}}, TypeError, id="set"),
    pytest.param([("a", 1)], TypeError, id="not-a-mapping"),
])
def test_mapping_digest_refuses_values_without_one_canonical_form(value, error):
    with pytest.raises(error):
        mapping_digest(value)
    if isinstance(value, dict):
        with pytest.raises(error):
            native_ready(value)
    else:
        assert native_ready(value) == [["a", 1]]  # only mapping_digest requires a top-level mapping


def test_file_digest_streams_large_files(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(np.random.default_rng(0).bytes(5 * (1 << 20) // 2))
    assert file_digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_text_digest_is_sha256_of_utf8():
    assert text_digest("Moliné") == hashlib.sha256("Moliné".encode("utf-8")).hexdigest()


def test_kernel_identity_mapping_is_strict():
    identity = KernelIdentity("ab" * 32, (17, 17), 0.03)
    record = identity.to_mapping()
    assert record == {"sha256": "ab" * 32, "shape": [17, 17], "pixel_scale_arcsec": 0.03}
    assert KernelIdentity.from_mapping(json.loads(json.dumps(record))) == identity
    for broken in ({**record, "normalize": True}, {"sha256": "ab" * 32, "shape": [17, 17]},
                   {**record, "sha256": "AB" * 32}, {**record, "shape": [17, 0]},
                   {**record, "pixel_scale_arcsec": 0.0}, {**record, "pixel_scale_arcsec": -0.03},
                   {**record, "pixel_scale_arcsec": 1}):
        with pytest.raises(ValueError):
            KernelIdentity.from_mapping(broken)
