"""Content digests in one format for arrays, mappings, text and files.

Every digest is the full 64-character SHA-256 hex string. The array and
mapping encodings below are a storage format: identities recorded in results,
batch markers and provenance compare across runs only while they stay fixed.

Arrays hash a header ``dtype=<dtype.str>;shape=<n,m,...>;`` followed by the
values as little-endian C-order bytes, so equal values give one digest
whatever the memory layout or byte order, and a change of dtype, shape or any
value bit gives another (``-0.0`` and ``0.0`` differ). Mappings hash their
canonical JSON text: sorted keys, no whitespace, ASCII, integer keys written
as strings, ``-0.0`` written as ``0.0``, and non-finite floats refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "KernelIdentity", "array_digest", "canonical_json", "file_digest", "json_ready",
    "mapping_digest", "text_digest",
]

_BLOCK_BYTES = 1 << 20


def array_digest(values: ArrayLike) -> str:
    """SHA-256 of the dtype and shape header plus the little-endian C-order bytes."""
    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.names is not None:
        raise TypeError(f"cannot digest an array of dtype {array.dtype}: object and structured "
                        "dtypes have no portable byte encoding")
    if array.dtype.str[0] == ">":
        array = array.astype(array.dtype.newbyteorder("<"))
    header = f"dtype={array.dtype.str};shape={','.join(str(n) for n in array.shape)};"
    return hashlib.sha256(header.encode("ascii") + array.tobytes(order="C")).hexdigest()


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _ready(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        rendered: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                name = key
            elif _is_integer(key):
                name = str(int(key))
            else:
                raise TypeError(f"{path or 'mapping'}: key {key!r} is neither a string nor an integer")
            if name in rendered:
                raise ValueError(f"{path or 'mapping'}: keys {key!r} and {name!r} render to the same text")
            rendered[name] = _ready(item, f"{path}.{name}" if path else name)
        return rendered
    if isinstance(value, (list, tuple)):
        return [_ready(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if _is_integer(value):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path or 'value'}: non-finite number {number!r}")
        return 0.0 if number == 0.0 else number
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        raise TypeError(f"{path or 'value'}: hash arrays with array_digest")
    raise TypeError(f"{path or 'value'}: cannot normalize {type(value).__name__}")


def json_ready(value: Any) -> Any:
    """Normalize a value of mappings, sequences and scalars to plain JSON types.

    The one normalizer for digests and JSON artifacts: tuples become lists,
    numpy scalars become Python scalars, integer keys become their decimal
    text, paths become strings and ``-0.0`` becomes ``0.0``. Non-finite floats,
    colliding keys, arrays and other types raise, naming the key path.
    """
    return _ready(value, "")


def canonical_json(value: Any) -> str:
    """Sorted-key, whitespace-free ASCII JSON of ``json_ready(value)``."""
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def mapping_digest(mapping: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON text of a mapping."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"mapping_digest needs a mapping, got {type(mapping).__name__}")
    return hashlib.sha256(canonical_json(mapping).encode("utf-8")).hexdigest()


def text_digest(text: str) -> str:
    """SHA-256 of the UTF-8 bytes of ``text``."""
    if not isinstance(text, str):
        raise TypeError(f"text_digest needs a string, got {type(text).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file's bytes, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


_KERNEL_IDENTITY_KEYS = ("sha256", "shape", "pixel_scale_arcsec")


@dataclass(frozen=True)
class KernelIdentity:
    """Identity of a detector kernel: values digest, shape and angular sampling."""

    sha256: str
    shape: tuple[int, int]
    pixel_scale_arcsec: float

    def __post_init__(self) -> None:
        if not (isinstance(self.sha256, str) and len(self.sha256) == 64
                and all(character in "0123456789abcdef" for character in self.sha256)):
            raise ValueError(f"sha256 must be 64 lowercase hex characters, got {self.sha256!r}")
        if not (isinstance(self.shape, tuple) and len(self.shape) == 2
                and all(_is_integer(n) and n > 0 for n in self.shape)):
            raise ValueError(f"shape must be a tuple of two positive integers, got {self.shape!r}")
        scale = self.pixel_scale_arcsec
        if not (isinstance(scale, float) and math.isfinite(scale) and scale > 0):
            raise ValueError(f"pixel_scale_arcsec must be a positive finite float, got {scale!r}")

    def to_mapping(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "shape": [int(n) for n in self.shape],
                "pixel_scale_arcsec": self.pixel_scale_arcsec}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> KernelIdentity:
        keys = set(mapping)
        if keys != set(_KERNEL_IDENTITY_KEYS):
            raise ValueError(f"kernel identity keys must be {list(_KERNEL_IDENTITY_KEYS)}, "
                             f"got {sorted(map(str, keys))}")
        return cls(mapping["sha256"], tuple(mapping["shape"]), mapping["pixel_scale_arcsec"])
