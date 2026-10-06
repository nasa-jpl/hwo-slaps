"""Catalog members loaded from the same bytes whose SHA256 identifies the catalog."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..config.checks import ConfigError, Key, MapOf, Table, Text
from ..identity import read_file_snapshot
from .distributions import PopulationError, Reference

__all__ = ["CatalogSpec", "Catalog", "load_catalog"]


@dataclass(frozen=True)
class CatalogSpec:
    path: Path
    columns: Mapping[str,str]
    text_columns: Mapping[str,str]

    def __post_init__(self):
        path=Path(self.path).expanduser().resolve()
        if path.suffix.lower() not in {".csv",".npz"} or not path.is_file():
            raise PopulationError(f"catalog.path: expected existing CSV/NPZ file, got {path}")
        object.__setattr__(self,"path",path)
        for label in ("columns","text_columns"):
            value=getattr(self,label)
            if not isinstance(value,Mapping):raise PopulationError(f"catalog.{label}: must be a mapping")
            for variable,column in value.items():
                Reference(variable)
                if not isinstance(column,str) or not column:raise PopulationError(f"catalog.{label}.{variable}: nonempty column name required")
            object.__setattr__(self,label,MappingProxyType(dict(value)))
        if not self.columns and not self.text_columns:raise PopulationError("catalog: at least one column required")
        if set(self.columns)&set(self.text_columns):raise PopulationError("catalog: numeric and text variables overlap")

    @classmethod
    def from_mapping(cls,mapping:Mapping[str,Any],*,base_dir:Path|None=None,path:str="population.catalog"):
        try:
            values=Table((Key("path",Text(),"CSV/NPZ path"),Key("columns",MapOf(Text()),"numeric columns",{}),
                          Key("text_columns",MapOf(Text()),"text columns",{}))).read(mapping,path)
            filename=Path(values["path"]).expanduser()
            if not filename.is_absolute():filename=(Path.cwd() if base_dir is None else Path(base_dir))/filename
            return cls(filename,values["columns"],values["text_columns"])
        except (ConfigError,PopulationError) as error:raise PopulationError(f"{path}: {error}") from None

    def to_mapping(self):
        return {"path":str(self.path),"columns":dict(self.columns),"text_columns":dict(self.text_columns)}


@dataclass(frozen=True)
class Catalog:
    spec: CatalogSpec
    digest: str
    rows: int
    _columns: Mapping[str,np.ndarray]

    def __post_init__(self):
        columns={}
        for name,value in self._columns.items():
            array=np.array(value,copy=True);array.flags.writeable=False;columns[name]=array
        object.__setattr__(self,"_columns",MappingProxyType(columns))

    def row(self,index:int)->dict[str,float|str]:
        if isinstance(index,bool) or not isinstance(index,(int,np.integer)) or not 0<=index<self.rows:
            raise PopulationError(f"catalog {self.spec.path}: row {index} outside0..{self.rows-1}")
        return {name:str(array[index]) if name in self.spec.text_columns else float(array[index])
                for name,array in self._columns.items()}


def load_catalog(spec:CatalogSpec)->Catalog:
    try:content,digest=read_file_snapshot(spec.path)
    except OSError as error:raise PopulationError(f"catalog {spec.path}: {error}") from error
    columns={}
    selected={**spec.columns,**spec.text_columns}
    try:
        if spec.path.suffix.lower()==".csv":
            reader=csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
            if reader.fieldnames is None or len(reader.fieldnames)!=len(set(reader.fieldnames)):
                raise PopulationError(f"catalog {spec.path}: missing or duplicate header")
            for variable,column in selected.items():
                if column not in reader.fieldnames:raise PopulationError(f"catalog {spec.path}: missing column {column}")
            values={name:[] for name in selected}
            for index,row in enumerate(reader):
                if None in row:raise PopulationError(f"catalog {spec.path}: extra fields at row {index}")
                for variable,column in selected.items():
                    text=row[column]
                    if text is None:raise PopulationError(f"catalog {spec.path}, column {column}, row {index}: missing value")
                    try:value=text if variable in spec.text_columns else float(text)
                    except ValueError:raise PopulationError(f"catalog {spec.path}, column {column}, row {index}: invalid number {text!r}") from None
                    if variable not in spec.text_columns and not np.isfinite(value):
                        raise PopulationError(f"catalog {spec.path}, column {column}, row {index}: non-finite number")
                    values[variable].append(value)
            columns={name:np.asarray(value,dtype=str if name in spec.text_columns else float) for name,value in values.items()}
        else:
            with np.load(io.BytesIO(content),allow_pickle=False) as archive:
                for variable,column in selected.items():
                    if column not in archive.files:raise PopulationError(f"catalog {spec.path}: missing column {column}")
                    array=archive[column]
                    expected="U" if variable in spec.text_columns else "fiu"
                    if array.ndim!=1 or array.dtype.kind not in expected:
                        raise PopulationError(f"catalog {spec.path}, column {column}: expected1-D {'unicode' if variable in spec.text_columns else 'numeric'} array")
                    if variable not in spec.text_columns and not np.all(np.isfinite(array)):
                        index=int(np.flatnonzero(~np.isfinite(array))[0])
                        raise PopulationError(f"catalog {spec.path}, column {column}, row {index}: non-finite number")
                    columns[variable]=np.array(array,copy=True)
        lengths={len(array) for array in columns.values()}
        if len(lengths)!=1:raise PopulationError(f"catalog {spec.path}: columns have unequal lengths")
        return Catalog(spec,digest,next(iter(lengths)),columns)
    except PopulationError:raise
    except (ValueError,TypeError,UnicodeError,csv.Error,OSError) as error:
        raise PopulationError(f"catalog {spec.path}: {error}") from error
