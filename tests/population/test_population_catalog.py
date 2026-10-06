"""Real catalog file rows, finite typed columns and same-byte identities."""

import hashlib

import numpy as np
import pytest

from hwoslaps.population import PopulationError, PopulationSpec, sample_population
from hwoslaps.population.catalog import load_catalog


def catalog_spec(path):
    return {"catalog":{"path":str(path),"columns":{"zl":"redshift"},"text_columns":{"label":"name"}},
            "variables":{"copy":{"kind":"normal","mean":{"var":"zl"},"std":.01}},
            "bind":{"scene.lens.redshift":"zl"}}


@pytest.mark.parametrize("format",["csv","npz"])
def test_catalog_row_is_member_index_and_identity_hashes_decoded_bytes(format,tmp_path):
    path=tmp_path/("lenses."+format)
    if format=="csv":path.write_text("redshift,name\n0.2,first\n0.3,second\n0.4,third\n")
    else:np.savez(path,redshift=np.array([.2,.3,.4]),name=np.array(["first","second","third"]))
    spec=PopulationSpec.from_mapping(catalog_spec(path))
    catalog=load_catalog(spec.catalog)
    assert catalog.digest==hashlib.sha256(path.read_bytes()).hexdigest()
    assert catalog.rows==3 and catalog.row(1)=={"zl":.3,"label":"second"}
    rows=sample_population(spec,2,start=1,seed=3)
    assert [row["zl"] for row in rows]==[.3,.4] and [row["label"] for row in rows]==["second","third"]
    with pytest.raises(PopulationError,match="row3|row 3"):sample_population(spec,1,start=3,seed=3)
    with pytest.raises(ValueError):catalog._columns["zl"][0]=.9


@pytest.mark.parametrize("failure",["missing","nan","length","object","text_bytes","dimension","float_overflow"])
def test_npz_catalog_refuses_malformed_typed_columns(failure,tmp_path):
    path=tmp_path/"bad.npz"
    columns={"redshift":np.array([.2,.3]),"name":np.array(["a","b"])}
    if failure=="missing":del columns["redshift"]
    if failure=="nan":columns["redshift"][1]=np.nan
    if failure=="length":columns["name"]=np.array(["a"])
    if failure=="object":columns["redshift"]=np.array([{"x":1}],dtype=object)
    if failure=="text_bytes":columns["name"]=np.array([b"a",b"b"])
    if failure=="dimension":columns["redshift"]=np.array([[.2,.3]])
    if failure == "float_overflow":
        columns["redshift"] = np.array([.2, np.longdouble("1e400")], dtype=np.longdouble)
        assert np.all(np.isfinite(columns["redshift"]))
    np.savez(path,**columns)
    expected = r"bad\.npz, column redshift, row 1.*finite Python float" if failure == "float_overflow" else "bad.npz"
    with pytest.raises(PopulationError, match=expected):
        sample_population(catalog_spec(path),1,seed=2)


@pytest.mark.parametrize("contents",["name\na\n","redshift,name\nnan,a\n","redshift,name\n0.2\n"])
def test_csv_catalog_refuses_bad_column_and_row(contents,tmp_path):
    path=tmp_path/"bad.csv";path.write_text(contents)
    with pytest.raises(PopulationError,match="bad.csv"):
        sample_population(catalog_spec(path),1,seed=2)


def test_catalog_paths_resolve_from_population_base_directory(tmp_path,monkeypatch):
    path=tmp_path/"lenses.csv";path.write_text("redshift,name\n0.2,a\n")
    mapping=catalog_spec("lenses.csv")
    monkeypatch.chdir(tmp_path.parent)
    spec=PopulationSpec.from_mapping(mapping,base_dir=tmp_path)
    assert spec.catalog.path==path and sample_population(spec,1,seed=1)[0]["zl"]==.2
