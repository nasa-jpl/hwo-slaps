"""YAML 1.2 reading and writing, and dotted-path editing (``hwoslaps.config.loading``).

Oracle for scalars: the YAML 1.2 core schema. Under PyYAML's default YAML 1.1
resolvers (``yaml.safe_load`` on the pinned PyYAML 6.0.3) the rows marked 1.1
read differently: ``1.0e8`` and ``1e8`` as strings, ``010`` as 8, ``0o17`` as a
string, ``1_000`` as 1000, ``yes`` and ``on`` as True, ``1:20`` as 80 and
``2026-10-05`` as a date.
"""

import math

import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.config.loading import dump_yaml, parse_assignment, parse_yaml_value, read_yaml, set_path

SCALARS = [
    ("1.0e8", 1e8),        # 1.1: '1.0e8'
    ("1e8", 1e8),          # 1.1: '1e8'
    ("1.0e-3", 0.001),
    ("5.0e-07", 5e-07),
    ("1.0E+9", 1e9),
    ("010", 10),           # 1.1: 8
    ("-010", -10),
    ("+7", 7),
    ("0o17", 15),          # 1.1: '0o17'
    ("0x1F", 31),
    ("1_000", "1_000"),    # 1.1: 1000
    ("yes", "yes"),        # 1.1: True
    ("on", "on"),          # 1.1: True
    ("true", True),
    ("FALSE", False),
    (".5", 0.5),
    ("1.", 1.0),
    ("1:20", "1:20"),      # 1.1: 80
    ("2026-10-05", "2026-10-05"),  # 1.1: datetime.date
    ("[1.0e7, 1.0e8]", [1e7, 1e8]),  # 1.1: ['1.0e7', '1.0e8']
    ("{4: 5.0, 5: 1e-3}", {4: 5.0, 5: 0.001}),
    (".inf", math.inf),
    (".nan", math.nan),
    (".NaN", math.nan),
    ("-.Inf", -math.inf),
    ("~", None),
    ("null", None),
    ("", None),
    ("Planck15", "Planck15"),
]


@pytest.mark.parametrize("text, expected", SCALARS, ids=[text or "empty" for text, _ in SCALARS])
def test_yaml_scalars_follow_the_yaml_1_2_core_schema(text, expected, tmp_path):
    value = parse_yaml_value(text)
    assert type(value) is type(expected) and repr(value) == repr(expected)
    document = tmp_path / "config.yaml"
    document.write_text(f"value: {text}\n", encoding="utf-8")
    from_file = read_yaml(document)["value"]
    assert type(from_file) is type(expected) and repr(from_file) == repr(expected)


def test_written_yaml_reads_back_identically(tmp_path):
    mapping = {
        "run_name": "1e8",
        "labels": ["010", "true", "null", ".inf", "1.0", "", "yes", "~", "0o17", "Moliné"],
        "coefficients": {4: 5.0, 11: -2.5e-07},
        "floats": [1e-08, 1e20, -0.0, 100000000.0, 0.1],
        "flags": [True, False, None],
        "nested": {"grid": {"shape": [500, 500], "pixel_scale_arcsec": 0.00716}},
    }
    text = dump_yaml(mapping)
    path = tmp_path / "effective.yaml"
    path.write_text(text, encoding="utf-8")
    read = read_yaml(path)
    assert repr(read) == repr(mapping)
    assert list(read) == list(mapping)
    assert "run_name: '1e8'" in text and "\n- yes\n" in text


@pytest.mark.parametrize("content, fragment", [
    pytest.param(None, "no such file", id="missing-file"),
    pytest.param("a: [1, 2\nb: 3\n", ".yaml:", id="syntax-error"),
    pytest.param("- 1\n- 2\n", "the document must be a mapping, got list", id="list-document"),
    pytest.param("", "the document must be a mapping, got NoneType", id="empty-document"),
    pytest.param("4\n", "the document must be a mapping, got int", id="scalar-document"),
    pytest.param("a: 1\nb: 2\na: 3\n", ".yaml:3: duplicate key 'a'", id="duplicate-key"),
    pytest.param("scene:\n  light:\n    intensity: 1.0\n    intensity: 2.0\n", ".yaml:4: duplicate key 'intensity'",
                 id="nested-duplicate-key"),
    pytest.param("a: 1\n---\nb: 2\n", "single document", id="two-documents"),
])
def test_read_yaml_errors_name_the_file(content, fragment, tmp_path):
    path = tmp_path / "broken.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        read_yaml(path)
    assert caught.value.path == ""
    assert str(path) in caught.value.message and fragment in caught.value.message


def test_set_path_and_assignments():
    mapping = {"scene": {"injection": {"mass_msun": 1e8}, "grid": {"shape": [10, 10]}}}
    before = repr(mapping)
    assert set_path(mapping, "scene.injection.mass_msun", 1e9, create=False) == {
        "scene": {"injection": {"mass_msun": 1e9}, "grid": {"shape": [10, 10]}}}
    assert set_path({}, "psf.truth.kind", "kernel", create=True) == {"psf": {"truth": {"kind": "kernel"}}}
    masses = [1e7]
    edited = set_path(mapping, "scene.masses", masses, create=True)
    masses.append(1e8)
    assert edited["scene"]["masses"] == [1e7]
    assert repr(mapping) == before
    for dotted, create, path, fragment in (
        ("scene.injection.mas_msun", False, "scene.injection.mas_msun", "no such key"),
        ("scene.lens.mass", False, "scene.lens", "no such key"),
        ("scene.grid.shape.x", True, "scene.grid.shape", "is not a mapping"),
        ("scene..grid", True, "scene..grid", "empty segment"),
    ):
        with pytest.raises(ConfigError) as caught:
            set_path(mapping, dotted, 1, create=create)
        assert caught.value.path == path and fragment in caught.value.message
    assert parse_assignment("scene.injection.mass_msun=1e8") == ("scene.injection.mass_msun", 1e8)
    assert parse_assignment("forecast.masses_msun=[1.0e7, 1.0e8]") == ("forecast.masses_msun", [1e7, 1e8])
    assert parse_assignment("psf.model.draw=") == ("psf.model.draw", None)
    assert parse_assignment("run_name=a=b") == ("run_name", "a=b")
    for text in ("novalue", "=1", " a=1"):
        with pytest.raises(ConfigError):
            parse_assignment(text)
