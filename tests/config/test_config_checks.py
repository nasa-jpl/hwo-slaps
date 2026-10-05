"""Key tables: value checks, strict reads, variants, exclusive groups, composition, paths,
settings dataclasses and the reference renderer.

Every schema here is a small table built in this file; expected values are written out
by hand from the documented rules.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

from hwoslaps.config.checks import (
    AnyValue, Boolean, ComponentName, ConfigError, Ellipticity, FilePath, Identifier, Integer, Key,
    ListOf, MapOf, Named, Nullable, Pair, REQUIRED, Real, Rule, Sha256, Shape, Table, Text, Union,
    Variants, dataclass_table, render_reference,
)

AT = "scene.grid"


def ok(value):
    return ("ok", value)


def bad(fragment, path=AT):
    return ("error", path, fragment)


EINSTEIN_OR_NUMBER = Union(Text(choices=("einstein_radius",)), Real(min=0, min_open=True))
UNIQUE_COUNTS = ListOf(Integer(min=1), min_length=1, unique=True)
COEFFICIENTS = MapOf(Integer(min=1), Real())


@pytest.mark.parametrize("check, value, outcome", [
    pytest.param(Boolean(), True, ok(True), id="boolean"),
    pytest.param(Boolean(), 1, bad("true or false"), id="boolean-rejects-int"),
    pytest.param(Integer(min=1), np.int64(4), ok(4), id="integer-from-numpy"),
    pytest.param(Integer(), True, bad("got True"), id="integer-rejects-bool"),
    pytest.param(Integer(), 2.0, bad("integer"), id="integer-rejects-float"),
    pytest.param(Integer(min=1), 0, bad("integer >= 1"), id="integer-below-min"),
    pytest.param(Real(min=0, min_open=True), 2, ok(2.0), id="real-int-to-float"),
    pytest.param(Real(), np.float32(0.5), ok(0.5), id="real-from-numpy"),
    pytest.param(Real(), True, bad("got True"), id="real-rejects-bool"),
    pytest.param(Real(), float("nan"), bad("number"), id="real-rejects-nan"),
    pytest.param(Real(), float("-inf"), bad("number"), id="real-rejects-inf"),
    pytest.param(Real(), 10**400, bad("number"), id="real-rejects-overflow"),
    pytest.param(Real(min=0, min_open=True), 0.0, bad("number > 0"), id="real-open-min"),
    pytest.param(Real(min=0, max=1), 1, ok(1.0), id="real-closed-max"),
    pytest.param(Real(), "1.0", bad("got '1.0'"), id="real-rejects-text"),
    pytest.param(Text(choices=("lens", "grid")), "grid", ok("grid"), id="text-choice"),
    pytest.param(Text(choices=("lens", "grid")), "ring", bad("one of: lens, grid"), id="text-not-a-choice"),
    pytest.param(Text(), "", bad("non-empty text"), id="text-empty"),
    pytest.param(Identifier(), "run_1.b-2", ok("run_1.b-2"), id="identifier"),
    pytest.param(Identifier(), "-run", bad("letters, digits"), id="identifier-leading-dash"),
    pytest.param(ComponentName(), "disk_2", ok("disk_2"), id="component-name"),
    pytest.param(ComponentName(), "Disk", bad("lower-case identifier"), id="component-name-upper"),
    pytest.param(Sha256(), "0f" * 32, ok("0f" * 32), id="sha256"),
    pytest.param(Sha256(), "0F" * 32, bad("lowercase SHA-256"), id="sha256-upper"),
    pytest.param(Pair(Real()), (1, 2), ok([1.0, 2.0]), id="pair-tuple-to-list"),
    pytest.param(Pair(Real()), [1.0], bad("pair"), id="pair-length"),
    pytest.param(Pair(Real()), [1.0, "a"], bad("number", f"{AT}[1]"), id="pair-item-path"),
    pytest.param(Shape(odd=True), [17, 17], ok([17, 17]), id="shape-odd"),
    pytest.param(Shape(odd=True), [16, 17], bad("odd positive integers", f"{AT}[0]"), id="shape-even"),
    pytest.param(Shape(), [3, 0], bad("positive integers", f"{AT}[1]"), id="shape-zero"),
    pytest.param(Ellipticity(), (0.1, 0), ok([0.1, 0.0]), id="ellipticity"),
    pytest.param(Ellipticity(), [0.8, 0.8], bad("< 1"), id="ellipticity-outside-unit-disc"),
    pytest.param(UNIQUE_COUNTS, (1, 2), ok([1, 2]), id="list-tuple-to-list"),
    pytest.param(UNIQUE_COUNTS, [], bad("at least 1 items"), id="list-too-short"),
    pytest.param(UNIQUE_COUNTS, [1, 1], bad("repeats 1", f"{AT}[1]"), id="list-repeat"),
    pytest.param(ListOf(Real(), length=3), [1.0, 2.0], bad("list of 3 items"), id="list-length"),
    pytest.param(ListOf(Integer()), [1, "x"], bad("integer", f"{AT}[1]"), id="list-item-path"),
    pytest.param(COEFFICIENTS, {4: 5}, ok({4: 5.0}), id="map-integer-keys"),
    pytest.param(COEFFICIENTS, {0: 1.0}, bad("integer >= 1", f"{AT}.0"), id="map-key-domain"),
    pytest.param(COEFFICIENTS, {"4": 1.0}, bad("integer >= 1", f"{AT}.4"), id="map-text-key"),
    pytest.param(Nullable(Integer()), None, ok(None), id="nullable-null"),
    pytest.param(Nullable(Integer()), 3, ok(3), id="nullable-value"),
    pytest.param(Nullable(Integer()), "3", bad("integer"), id="nullable-wrong-type"),
    pytest.param(EINSTEIN_OR_NUMBER, "einstein_radius", ok("einstein_radius"), id="union-text-branch"),
    pytest.param(EINSTEIN_OR_NUMBER, 0.5, ok(0.5), id="union-number-branch"),
    pytest.param(EINSTEIN_OR_NUMBER, True, bad("one of: einstein_radius or number > 0"), id="union-no-branch"),
    pytest.param(AnyValue(), [1, {"a": 2}], ok([1, {"a": 2}]), id="any-value"),
])
def test_scalar_checks_normalize_or_reject_with_the_key_path(check, value, outcome):
    if outcome[0] == "ok":
        result = check(value, AT)
        assert type(result) is type(outcome[1]) and repr(result) == repr(outcome[1])
    else:
        with pytest.raises(ConfigError) as caught:
            check(value, AT)
        assert caught.value.path == outcome[1]
        assert outcome[2] in caught.value.message
        assert str(caught.value) == f"{outcome[1]}: {caught.value.message}"


def test_file_path_check_reads_only_existing_resolved_files(tmp_path):
    kernel = tmp_path / "kernel.npy"
    kernel.write_bytes(b"")
    (tmp_path / "kernel.txt").write_bytes(b"")
    check = FilePath((".npy", ".npz"))
    assert check(str(kernel), "psf.truth.path") == str(kernel.resolve())
    for value, fragment in ((kernel.name, "not resolved"), (str(tmp_path / "missing.npy"), "no such file"),
                            (str(tmp_path / "kernel.txt"), ".npy or .npz"), (str(tmp_path), "no such file")):
        with pytest.raises(ConfigError) as caught:
            check(value, "psf.truth.path")
        assert caught.value.path == "psf.truth.path" and fragment in caught.value.message
    with pytest.raises(TypeError):
        FilePath(("npy",))


GRID = Table((
    Key("shape", Shape(), "Grid shape.", [10, 10]),
    Key("pixel_scale_arcsec", Real(min=0, min_open=True), "Pixel scale.", 0.05, "arcsec"),
))


def refuse_bad_name(values, path):
    if values["name"] == "bad":
        raise ConfigError(f"{path}.name" if path else "name", f"must not be bad (grid {values['grid']['shape']})")


RUN = Table((
    Key("name", Text(), "Run name."),
    Key("grid", GRID, "Pixel grid.", {}),
    Key("tags", ListOf(Text()), "Free tags.", []),
), rules=(Rule("name is not bad", refuse_bad_name),))


def test_table_read_fills_defaults_and_rejects_unknown_and_missing_keys():
    values = RUN.read({"tags": ["a"], "name": "x"}, "")
    assert values == {"name": "x", "grid": {"shape": [10, 10], "pixel_scale_arcsec": 0.05}, "tags": ["a"]}
    assert list(values) == ["name", "grid", "tags"]
    assert RUN.read(values, "") == values
    first = RUN.read({"name": "x"}, "")
    first["grid"]["shape"].append(3)
    first["tags"].append("mutated")
    assert RUN.read({"name": "x"}, "")["grid"]["shape"] == [10, 10]
    assert RUN.read({"name": "x"}, "")["tags"] == []
    cases = [
        ({"name": "x", "grid": {"shap": [1, 1]}}, "run.grid.shap", "unknown key; allowed: pixel_scale_arcsec, shape"),
        ({"name": 3, "nmae": "x"}, "run.nmae", "unknown key"),
        ({}, "run.name", "required"),
        ({"name": "x", "grid": {"pixel_scale_arcsec": 0}}, "run.grid.pixel_scale_arcsec", "number > 0"),
        ({"name": "bad"}, "run.name", "must not be bad (grid [10, 10])"),
        ([1], "run", "must be a mapping"),
    ]
    for value, path, fragment in cases:
        with pytest.raises(ConfigError) as caught:
            RUN.read(value, "run")
        assert caught.value.path == path and fragment in caught.value.message
    with pytest.raises(TypeError):
        Table((Key("a", Integer(), ""), Key("a", Real(), "")))


def _refuse(condition, path):
    if condition:
        raise ConfigError(path, "intensity is 7")


def test_table_extend_appends_keys_groups_and_rules():
    amplitude = Table((Key("intensity", Nullable(Real()), "Amplitude.", None),), rules=(
        Rule("intensity is not 7", lambda values, path: _refuse(values["intensity"] == 7.0, path)),))
    light = amplitude.extend([Key("flux", Nullable(Real()), "Flux.", None)], exactly_one=[("intensity", "flux")])
    assert [key.name for key in light.keys] == ["intensity", "flux"]
    assert light.read({"flux": 2}, "") == {"intensity": None, "flux": 2.0}
    for value, fragment in (({}, "exactly one of intensity, flux"), ({"intensity": 7}, "is 7")):
        with pytest.raises(ConfigError, match=fragment):
            light.read(value, "")
    assert [key.name for key in amplitude.keys] == ["intensity"]
    with pytest.raises(TypeError):
        amplitude.extend([Key("intensity", Real(), "")])


KERNEL = Table((Key("normalize", Boolean(), "Normalize to unit sum.", True),))
OPTICAL = Table((Key("wavelength_nm", Real(min=0, min_open=True), "Wavelength.", unit="nm"),))
TRUTH = Variants("kind", {"kernel": KERNEL, "optical": OPTICAL})
MODEL = Variants("kind", {"matched": Table(()), "kernel": KERNEL}, default="matched")


def test_variants_select_the_table_by_discriminator():
    selected = TRUTH.read({"wavelength_nm": 500, "kind": "optical"}, "psf.truth")
    assert selected == {"kind": "optical", "wavelength_nm": 500.0} and list(selected) == ["kind", "wavelength_nm"]
    assert MODEL.read({}, "psf.model") == {"kind": "matched"}
    assert MODEL.read({"kind": "kernel"}, "psf.model") == {"kind": "kernel", "normalize": True}
    cases = [
        ({"wavelength_nm": 500}, "psf.truth.kind", "required; one of: kernel, optical"),
        ({"kind": "laser"}, "psf.truth.kind", "one of: kernel, optical, got 'laser'"),
        ({"kind": None}, "psf.truth.kind", "got None"),
        ({"kind": "optical", "wavelength_nm": 5, "normalize": True}, "psf.truth.normalize", "unknown key"),
    ]
    for value, path, fragment in cases:
        with pytest.raises(ConfigError) as caught:
            TRUTH.read(value, "psf.truth")
        assert caught.value.path == path and fragment in caught.value.message
    for broken in (lambda: Variants("mode", {"a": KERNEL}),
                   lambda: Variants("kind", {"a": KERNEL}, default="b"),
                   lambda: Variants("kind", {"a": Table((Key("kind", Text(), ""),))})):
        with pytest.raises(TypeError):
            broken()


WAVELENGTH = Table((
    Key("wavelength_nm", Nullable(Real(min=0, min_open=True)), "Monochromatic wavelength.", None, "nm"),
    Key("wavelength_samples", Nullable(Integer(min=1)), "Bandpass nodes.", None),
), exactly_one=(("wavelength_nm", "wavelength_samples"),))


def test_exactly_one_counts_set_values():
    assert WAVELENGTH.read({"wavelength_nm": 500}, "")["wavelength_samples"] is None
    assert WAVELENGTH.read({"wavelength_nm": None, "wavelength_samples": 11}, "") == {
        "wavelength_nm": None, "wavelength_samples": 11}
    for value in ({}, {"wavelength_nm": None}, {"wavelength_nm": 500, "wavelength_samples": 11}):
        with pytest.raises(ConfigError) as caught:
            WAVELENGTH.read(value, "psf.truth")
        assert caught.value.path == "psf.truth"
        assert caught.value.message == "exactly one of wavelength_nm, wavelength_samples must be set"
    for member in (Key("a", Real(), "", None), Key("a", Nullable(Real()), "", 1.0)):
        with pytest.raises(TypeError):
            Table((member, Key("b", Nullable(Real()), "", None)), exactly_one=(("a", "b"),))


LIGHT = Named(Table((
    Key("intensity", Real(), "Amplitude."),
    Key("centre", Pair(Real()), "Centre.", [0.0, 0.0]),
)))
PSF = Variants("kind", {
    "kernel": Table((Key("path", Nullable(FilePath((".npy",))), "Kernel file.", None),
                     Key("normalize", Boolean(), "Normalize.", True))),
    "optical": Table((Key("wavelength_nm", Real(), "Wavelength."),
                      Key("zernikes", MapOf(Integer(min=1), Real()), "Coefficients.", {}))),
})
COMPOSED = Table((
    Key("psf", PSF, "Truth PSF.", {}),
    Key("light", LIGHT, "Light components.", {}),
    Key("masses", ListOf(Real()), "Masses.", []),
    Key("seed", Integer(), "Seed.", 0),
))


@pytest.mark.parametrize("base, overlay, expected", [
    pytest.param({"psf": {"kind": "kernel", "path": "/k.npy", "normalize": False}},
                 {"psf": {"kind": "optical", "wavelength_nm": 500}},
                 {"psf": {"kind": "optical", "wavelength_nm": 500}}, id="discriminator-switch-replaces"),
    pytest.param({"psf": {"kind": "kernel", "path": "/k.npy", "normalize": False}},
                 {"psf": {"normalize": True}},
                 {"psf": {"kind": "kernel", "path": "/k.npy", "normalize": True}}, id="same-variant-merges"),
    pytest.param({"psf": {"kind": "kernel", "path": "/k.npy"}}, {"psf": {"path": None}},
                 {"psf": {"kind": "kernel", "path": None}}, id="null-clears"),
    pytest.param({"psf": {"kind": "optical", "wavelength_nm": 500, "zernikes": {4: 1.0, 5: 2.0}}},
                 {"psf": {"zernikes": {6: 3.0}}},
                 {"psf": {"kind": "optical", "wavelength_nm": 500, "zernikes": {6: 3.0}}}, id="free-map-replaces"),
    pytest.param({"masses": [1.0, 2.0], "seed": 1}, {"masses": [3.0], "seed": 2}, {"masses": [3.0], "seed": 2},
                 id="lists-and-scalars-replace"),
    pytest.param({"light": {"disk": {"intensity": 1.0, "centre": [0.1, 0.2]}, "bulge": {"intensity": 2.0}}},
                 {"light": {"bulge": {"intensity": 9.0}, "clump": {"intensity": 1.0}}},
                 {"light": {"disk": {"intensity": 1.0, "centre": [0.1, 0.2]}, "bulge": {"intensity": 9.0},
                            "clump": {"intensity": 1.0}}}, id="named-components-merge-in-place"),
])
def test_merge_switches_variants_and_replaces_free_maps_and_lists(base, overlay, expected):
    base_copy, overlay_copy = repr(base), repr(overlay)
    merged = COMPOSED.merge(base, overlay)
    assert merged == expected
    if "light" in expected:
        assert list(merged["light"]) == list(expected["light"])
    assert repr(base) == base_copy and repr(overlay) == overlay_copy


ASSET = Table((Key("asset_path", Nullable(FilePath((".npz",))), "Image asset.", None),
               Key("label", Text(), "Label.", "x")))
FACTOR = Table((Key("path", FilePath((".dat",)), "Throughput table."),))
PATHS = Table((
    Key("name", Text(), "Cosmology name."),
    Key("psf", PSF, "Truth PSF.", {}),
    Key("light", Named(ASSET), "Light components.", {}),
    Key("factors", ListOf(FACTOR), "Bandpass factors.", []),
))


def test_path_transform_touches_only_path_keys():
    seen = []

    def resolve(value, check):
        seen.append(check)
        return f"/root/{value}"

    document = {
        "name": "Planck15",
        "psf": {"kind": "kernel", "path": "k.npy", "normalize": True},
        "light": {"disk": {"asset_path": "a.npz", "label": "disk.npz"}, "core": {"asset_path": None}},
        "factors": [{"path": "f1.dat"}, {"path": "f2.dat"}],
        "unknown": "x.npy",
    }
    before = repr(document)
    assert PATHS.transform_paths(document, resolve) == {
        "name": "Planck15",
        "psf": {"kind": "kernel", "path": "/root/k.npy", "normalize": True},
        "light": {"disk": {"asset_path": "/root/a.npz", "label": "disk.npz"}, "core": {"asset_path": None}},
        "factors": [{"path": "/root/f1.dat"}, {"path": "/root/f2.dat"}],
        "unknown": "x.npy",
    }
    assert repr(document) == before
    assert all(isinstance(check, FilePath) for check in seen) and len(seen) == 4
    assert PATHS.transform_paths({"psf": {"path": "k.npy"}}, resolve) == {"psf": {"path": "/root/k.npy"}}
    with pytest.raises(TypeError):
        Variants("kind", {"a": Table((Key("path", FilePath((".npy",)), ""),)),
                          "b": Table((Key("path", Text(), ""),))})
    with pytest.raises(TypeError):
        Variants("kind", {"a": Table((Key("pupil", Table(()), "", {}),)),
                          "b": Table((Key("pupil", Table(()), "", {}),))})


def test_named_components_keep_order_and_require_component_names():
    light = Named(Table((Key("intensity", Real(), "Amplitude."),)), min_length=1)
    components = light.read({"disk": {"intensity": 1}, "bulge": {"intensity": 2}}, "scene.source.light")
    assert list(components) == ["disk", "bulge"] and components["bulge"] == {"intensity": 2.0}
    cases = [
        ({"Disk": {"intensity": 1}}, "scene.source.light.Disk", "lower-case identifier"),
        ({"a.b": {"intensity": 1}}, "scene.source.light.a.b", "lower-case identifier"),
        ({}, "scene.source.light", "needs at least 1 components"),
        ({"disk": {"intensity": "1"}}, "scene.source.light.disk.intensity", "number"),
    ]
    for value, path, fragment in cases:
        with pytest.raises(ConfigError) as caught:
            light.read(value, "scene.source.light")
        assert caught.value.path == path and fragment in caught.value.message


@dataclass(frozen=True)
class Settings:
    seed: int
    n_live: int = 100
    f_live: float | None = None
    use_jax: bool = False
    mode: Literal["search", "anchor"] = "search"
    label: str = "case"
    clip: tuple[float, float] | None = None
    names: tuple[str, ...] = ()


SETTINGS_DOCS = {name: f"{name} doc" for name in
                 ("seed", "n_live", "f_live", "use_jax", "mode", "label", "clip", "names")}


def test_dataclass_table_derives_keys_from_fields():
    table = dataclass_table(Settings, docs=SETTINGS_DOCS, units={"f_live": "fraction"})
    assert [key.name for key in table.keys] == list(SETTINGS_DOCS)
    assert table.keys[0].default is REQUIRED and table.keys[2].unit == "fraction"
    defaults = table.read({"seed": 3}, "sampler")
    assert defaults == {"seed": 3, "n_live": 100, "f_live": None, "use_jax": False, "mode": "search",
                        "label": "case", "clip": None, "names": []}
    assert table.read(defaults, "sampler") == defaults
    assert table.read({"seed": 1, "f_live": 1, "mode": "anchor", "clip": (1, 2), "names": ("a",)}, "sampler") == {
        "seed": 1, "n_live": 100, "f_live": 1.0, "use_jax": False, "mode": "anchor", "label": "case",
        "clip": [1.0, 2.0], "names": ["a"]}
    for value, path in (({"seed": 1, "n_live": 1.5}, "sampler.n_live"), ({"seed": True}, "sampler.seed"),
                        ({"seed": 1, "mode": "grid"}, "sampler.mode"), ({"seed": 1, "nlive": 5}, "sampler.nlive"),
                        ({}, "sampler.seed")):
        with pytest.raises(ConfigError) as caught:
            table.read(value, "sampler")
        assert caught.value.path == path

    @dataclass(frozen=True)
    class Nested:
        inner: Settings

    for broken in (lambda: dataclass_table(Settings, docs={"seed": "x"}),
                   lambda: dataclass_table(Settings, docs=SETTINGS_DOCS, units={"other": "s"}),
                   lambda: dataclass_table(Nested, docs={"inner": "x"}),
                   lambda: dataclass_table(dict, docs={})):
        with pytest.raises(TypeError):
            broken()


SKY = Table((
    Key("rate_e_per_s", Nullable(Real(min=0)), "Sky rate.", None, "e-/s per pixel"),
    Key("ab_mag_per_arcsec2", Nullable(Real()), "Sky surface brightness.", None, "AB mag/arcsec^2"),
), exactly_one=(("rate_e_per_s", "ab_mag_per_arcsec2"),), doc="Uniform sky background.")
MODEL_PSF = Variants("kind", {
    "matched": Table(()),
    "kernel": Table((Key("path", FilePath((".npy", ".npz")), "Kernel file."),)),
}, default="matched")
OBSERVATION = Table((
    Key("exposure_time_s", Real(min=0, min_open=True), "Exposure time.", unit="s"),
    Key("sky", SKY, "Sky background.", {}),
    Key("model", MODEL_PSF, "Model PSF.", {}),
    Key("label", Text(), "Free label.", "run"),
), rules=(Rule("exposure_time_s is at most one day", lambda values, path: None),))

OBSERVATION_BLOCK = """\
## observation

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `exposure_time_s` | number > 0 | required | s | Exposure time. |
| `sky` | mapping, see `observation.sky` | `{}` |  | Sky background. |
| `model` | mapping by `kind` (matched, kernel), see `observation.model` | `{}` |  | Model PSF. |
| `label` | non-empty text | `run` |  | Free label. |

- exposure_time_s is at most one day
"""
SKY_BLOCK = """\
## observation.sky

Uniform sky background.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `rate_e_per_s` | null or number >= 0 | `null` | e-/s per pixel | Sky rate. |
| `ab_mag_per_arcsec2` | null or number | `null` | AB mag/arcsec^2 | Sky surface brightness. |

- exactly one of `rate_e_per_s`, `ab_mag_per_arcsec2` is set; write null to clear one
"""
MATCHED_BLOCK = """\
## observation.model (kind: matched)

Selected by `kind: matched` (the default).

No keys.
"""
KERNEL_BLOCK = """\
## observation.model (kind: kernel)

Selected by `kind: kernel`.

| key | value | default | unit | meaning |
|---|---|---|---|---|
| `path` | path to an existing .npy or .npz file | required |  | Kernel file. |
"""


def test_reference_rendering_of_a_small_schema():
    documents = [("observation", OBSERVATION)]
    assert render_reference(documents) == "\n".join((OBSERVATION_BLOCK, SKY_BLOCK, MATCHED_BLOCK, KERNEL_BLOCK))
    assert render_reference(documents, section="observation.sky") == SKY_BLOCK
    assert render_reference(documents, section="observation.model") == "\n".join((MATCHED_BLOCK, KERNEL_BLOCK))
    with pytest.raises(ConfigError, match="unknown section 'observation.skies'"):
        render_reference(documents, section="observation.skies")
