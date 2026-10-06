"""Explicit geometry conversions and caller-provided population function hooks."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..config.checks import ConfigError, Integer, Key, ListOf, MapOf, Real, Table, Text
from ..scene.convert import ell_comps_from, multipole_components_from, polar_offset, shear_components_from
from .distributions import PopulationError, Reference, Value, read_value

__all__ = ["Derivation", "parse_derivation"]


def _input(value, path):
    if isinstance(value, (list,tuple)):
        return tuple(read_value(item,f"{path}[{i}]") for i,item in enumerate(value))
    return read_value(value,path)


def _vector_input(value, path):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise PopulationError(f"{path}: at least2 numeric/reference components required")
    return tuple(read_value(item, f"{path}[{index}]") for index, item in enumerate(value))


def _function_inputs(value, path):
    if not isinstance(value, Mapping):
        raise PopulationError(f"{path}: keyword input mapping required")
    if any(not isinstance(name, str) or not name for name in value):
        raise PopulationError(f"{path}: nonempty keyword names required")
    return {name: _input(item, f"{path}.{name}") for name, item in value.items()}


def _record(value):
    if isinstance(value,Reference): return {"var":value.name()}
    if isinstance(value,tuple): return [_record(item) for item in value]
    return value


def _numeric_output(value):
    if isinstance(value,tuple):
        if not value: raise PopulationError("function output: tuple must not be empty")
        if any(isinstance(item, (tuple, list, Mapping)) for item in value):
            raise PopulationError("function output: expected a flat tuple of finite numbers")
        return tuple(_numeric_output(item) for item in value)
    result=read_value(value,"derivation output")
    if isinstance(result,Reference): raise PopulationError("derivation output must be numeric")
    return result


@dataclass(frozen=True)
class Derivation:
    kind: str
    inputs: Mapping[str, Any]
    function: str | None = None

    def __post_init__(self):
        if self.kind not in _TABLES:
            raise PopulationError(f"unknown derivation kind {self.kind!r}")
        object.__setattr__(self,"inputs",MappingProxyType(dict(self.inputs)))

    def references(self):
        return tuple(reference for value in self.inputs.values()
                     for reference in (value if isinstance(value,tuple) else (value,))
                     if isinstance(reference,Reference))

    @property
    def size(self):
        return len(self.inputs["of"]) if self.kind=="vector" else (None if self.kind=="function" else 2)

    def evaluate(self,resolve:Callable[[Value],float]):
        values={name:tuple(resolve(item) for item in value) if isinstance(value,tuple) else resolve(value)
                for name,value in self.inputs.items()}
        if self.kind=="vector":result=values["of"]
        elif self.kind=="polar_offset":
            if values["radius"]<0:raise PopulationError("radius must be non-negative")
            result=polar_offset(values["radius"],values["angle_deg"],(values["centre_y"],values["centre_x"]))
        elif self.kind=="ell_comps":
            if not 0<values["axis_ratio"]<=1:raise PopulationError("axis_ratio must be in (0,1]")
            result=ell_comps_from(values["axis_ratio"],values["angle_deg"])
        elif self.kind=="shear_components":
            if values["magnitude"]<0:raise PopulationError("magnitude must be non-negative")
            result=shear_components_from(values["magnitude"],values["angle_deg"])
        elif self.kind=="multipole_components":
            if values["strength"]<0:raise PopulationError("strength must be non-negative")
            result=multipole_components_from(values["strength"],values["angle_deg"],values["order"])
        else:
            module,_,name=self.function.partition(":")
            function=importlib.import_module(module)
            for part in name.split("."):function=getattr(function,part)
            if not callable(function):raise PopulationError(f"function {self.function}: not callable")
            result=function(**values)
        return _numeric_output(result)

    def to_mapping(self):
        return ({"kind":self.kind,"function":self.function,"inputs":{name:_record(value) for name,value in self.inputs.items()}}
                if self.kind=="function" else {"kind":self.kind,**{name:_record(value) for name,value in self.inputs.items()}})


_TABLES={
    "vector":Table((Key("of",_vector_input,"vector components"),)),
    "polar_offset":Table((Key("radius",read_value,"radius"),Key("angle_deg",read_value,"angle in degrees"),
                          Key("centre_y",read_value,"centre y",0.),Key("centre_x",read_value,"centre x",0.))),
    "ell_comps":Table((Key("axis_ratio",read_value,"minor/major axis ratio"),Key("angle_deg",read_value,"major axis angle"))),
    "shear_components":Table((Key("magnitude",read_value,"shear magnitude"),Key("angle_deg",read_value,"shear angle"))),
    "multipole_components":Table((Key("strength",read_value,"multipole strength"),Key("angle_deg",read_value,"multipole angle"),
                                  Key("order",Integer(min=1),"multipole order"))),
    "function":Table((Key("function",Text(pattern=r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"),"module:function"),
                      Key("inputs",_function_inputs,"keyword inputs",{}))),
}


def parse_derivation(mapping:Mapping[str,Any],path:str)->Derivation:
    if not isinstance(mapping,Mapping) or not isinstance(mapping.get("kind"),str) or mapping["kind"] not in _TABLES:
        raise PopulationError(f"{path}.kind: unsupported derivation kind")
    kind=mapping["kind"]
    try:
        table=_TABLES[kind]
        values=Table((Key("kind",Text(choices=(kind,)),"derivation kind"),*table.keys)).read(mapping,path)
        values.pop("kind")
        if kind=="function":return Derivation(kind,values["inputs"],values["function"])
        for name,value in values.items():
            if name!="of" and isinstance(value,Reference):continue
            if name in {"radius","magnitude","strength"} and value<0:raise PopulationError(f"{name}: must be non-negative")
            if name=="axis_ratio" and not 0<value<=1:raise PopulationError("axis_ratio: must be in (0,1]")
        if kind=="vector":values["of"]=tuple(values["of"])
        return Derivation(kind,values)
    except (ConfigError,PopulationError) as error:
        raise PopulationError(f"{path}: {error}") from None
