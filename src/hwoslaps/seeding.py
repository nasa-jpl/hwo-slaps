"""Named random streams: one integer seed names one realization.

A user-supplied integer seed drives its consumer through ``root_rng(seed)``,
which is ``numpy.random.default_rng(seed)`` (empty spawn key), so detector
noise and knowledge-error draws keep their historical bytes. Every other
random element draws from a named child stream,
``SeedSequence(entropy, spawn_key=(*indices, *words(name)))``, where
``words(name)`` are the eight little-endian 32-bit words of SHA-256 of the
UTF-8 name. A named stream never equals a root stream (its spawn key is never
empty), two named streams coincide only for equal entropy, indices and name,
and no consumer derives a stream by arithmetic on a seed.

Streams of the package (stream: entropy; indices; consumer):

- ``root_rng(seed)``: a noise seed; none; the detector noise draw.
- ``root_rng(seed)``: the ``seed`` of a ``draw`` block; none; wavefront and
  knowledge-error direction draws.
- ``"scene.injection_position"``: the configuration ``seed``; none; random
  placement of ``scene.injection``.
- ``"scene.perturbers.<quantity>"``: the configuration ``seed``; the population
  index; perturber count, mass, radius and angle.
- ``"population/<variable>"``: the population seed; (index, attempt); one
  population variable.
- ``"population.copula/<name>"``: the population seed; (index, attempt); one
  Gaussian copula.
- ``"batch/simulate/noise"``: the batch seed; (member, replicate); noise seeds of
  simulate jobs.
- ``"batch/nonlinear/<family>/noise/<trial_id>"``: the batch seed; (member,
  replicate); noise seeds of nonlinear jobs, shared across arms.
- ``"batch/nonlinear/<family>/sampler/<arm>/<trial_id>"``: the batch seed;
  (member, replicate, attempt); Nautilus seeds.
- ``"batch/psf.model/direction"``: the batch seed; (member, direction);
  knowledge-error direction seeds.

Population member seeds are the Cantor pairing of (population seed, member
index), which is injective, so members of different populations never share
scene randomness.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from numbers import Integral

import numpy as np

__all__ = ["derived_seed", "member_seed", "root_rng", "stream_rng"]


def _non_negative(value: object, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{what} must be a non-negative integer, got {value!r}")
    return int(value)


def _entropy(entropy: int | Sequence[int]) -> int | tuple[int, ...]:
    if isinstance(entropy, Sequence) and not isinstance(entropy, (str, bytes)):
        if not entropy:
            raise ValueError("entropy must be a non-negative integer or a non-empty sequence of them")
        return tuple(_non_negative(item, "entropy") for item in entropy)
    return _non_negative(entropy, "entropy")


def _spawn_key(name: str, indices: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(name, str) or not name:
        raise ValueError(f"stream name must be a non-empty string, got {name!r}")
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    words = tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4))
    return (*(_non_negative(index, "stream index") for index in indices), *words)


def _sequence(entropy: int | Sequence[int], name: str, indices: tuple[int, ...]) -> np.random.SeedSequence:
    return np.random.SeedSequence(_entropy(entropy), spawn_key=_spawn_key(name, indices))


def root_rng(seed: int) -> np.random.Generator:
    """The generator of a user-named integer seed: ``numpy.random.default_rng(seed)``."""
    return np.random.default_rng(_non_negative(seed, "seed"))


def stream_rng(entropy: int | Sequence[int], name: str, *indices: int) -> np.random.Generator:
    """PCG64 generator of the named child stream ``(entropy, indices, name)``."""
    return np.random.Generator(np.random.PCG64(_sequence(entropy, name, indices)))


def derived_seed(entropy: int | Sequence[int], name: str, *indices: int) -> int:
    """First 64-bit word of the named child stream's state, as an integer seed."""
    return int(_sequence(entropy, name, indices).generate_state(1, dtype=np.uint64)[0])


def member_seed(population_seed: int, index: int) -> int:
    """Cantor pairing ``(s + i)(s + i + 1) / 2 + i``, injective over non-negative pairs."""
    s = _non_negative(population_seed, "population seed")
    i = _non_negative(index, "member index")
    return (s + i) * (s + i + 1) // 2 + i
