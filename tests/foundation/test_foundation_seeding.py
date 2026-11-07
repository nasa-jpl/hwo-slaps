"""Named random streams, derived seeds and member seeds.

Oracles: numpy's own ``default_rng`` and ``SeedSequence`` evaluated here on the
documented spawn-key layout, uint64 literals computed with numpy alone, draws of
the 8fa6209 population streams (fixture written by the run-directory stream
probe on the base tree), and the closed-form Cantor pairing.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.seeding import derived_seed, member_seed, root_rng, stream_rng

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "foundation" / "population_streams_8fa6209.json"


def name_words(name):
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4))


@pytest.mark.parametrize("seed", [0, 11, 20261005, 2**64 + 7])
def test_root_rng_is_numpy_default_rng(seed):
    generator, expected = root_rng(seed), np.random.default_rng(seed)
    assert type(generator.bit_generator) is type(expected.bit_generator)
    assert generator.bit_generator.state == expected.bit_generator.state
    assert generator.normal(size=8).tobytes() == expected.normal(size=8).tobytes()


def test_stream_rng_reproduces_8fa6209_population_streams():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 3
    for case in cases:
        draws = stream_rng(case["seed"], case["name"], case["index"]).random(len(case["random"]))
        assert draws.tolist() == case["random"], case["name"]


@pytest.mark.parametrize("entropy, name, indices, literal", [
    (7, "batch/simulate/noise", (0, 0), 1132586323907944775),
    (7, "batch/simulate/noise", (0, 1), 4848764046799505188),
    (7, "batch/simulate/noise", (1, 0), 15387281872901488106),
    (7, "batch/psf.model/direction", (0, 0), 14886344369075695213),
    (7, "scene.injection_position", (), 10849725077906556550),
    (2**70, "population/x", (3, 0), 10423954780549914825),
])
def test_named_streams_and_derived_seeds_follow_the_documented_spawn_key(entropy, name, indices, literal):
    sequence = np.random.SeedSequence(entropy, spawn_key=(*indices, *name_words(name)))
    assert derived_seed(entropy, name, *indices) == literal
    assert literal == int(sequence.generate_state(1, dtype=np.uint64)[0])
    expected = np.random.Generator(np.random.PCG64(sequence))
    assert stream_rng(entropy, name, *indices).bit_generator.state == expected.bit_generator.state


def test_member_seed_is_injective_cantor():
    assert [member_seed(5, 3), member_seed(4, 4), member_seed(2, 110170), member_seed(2, 111187)] == [
        39, 40, 6069100048, 6181663642]
    seeds = {member_seed(s, i) for s in range(64) for i in range(2000)}
    assert len(seeds) == 64 * 2000


def test_named_scene_streams_never_alias_member_noise_streams():
    """8fa6209 placed a subhalo with default_rng(seed + 1): member (5, 3) used the noise stream of (4, 4)."""
    seeds = [member_seed(s, i) for s in range(64) for i in range(64)]
    noise = {root_rng(seed).random() for seed in seeds}
    placement = {stream_rng(seed, "scene.injection_position").random() for seed in seeds}
    assert len(noise) == len(placement) == len(seeds)
    assert noise.isdisjoint(placement)


@pytest.mark.parametrize("call", [
    pytest.param(lambda: root_rng(True), id="bool-seed"),
    pytest.param(lambda: root_rng(-1), id="negative-seed"),
    pytest.param(lambda: root_rng(1.0), id="float-seed"),
    pytest.param(lambda: stream_rng(1, ""), id="empty-name"),
    pytest.param(lambda: stream_rng(1, "x", -1), id="negative-index"),
    pytest.param(lambda: derived_seed([], "x"), id="empty-entropy-sequence"),
    pytest.param(lambda: derived_seed([1, True], "x"), id="bool-in-entropy"),
    pytest.param(lambda: member_seed(1, -2), id="negative-member"),
])
def test_seeding_rejects_entropy_indices_and_names_outside_the_domain(call):
    with pytest.raises(ValueError):
        call()
