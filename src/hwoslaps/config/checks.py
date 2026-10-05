"""Key tables: strict reading, defaults, composition and the configuration reference.

Every configuration key is defined once, as a ``Key`` in the table of the module
that owns its section. A table reads a mapping strictly (unknown keys are errors
with their dotted path, required keys must be present, defaults are filled in
full), merges overlays for file composition, finds the values that are file
paths, builds the identity record of a configuration, and renders the reference
documentation. Checks never coerce across types: a boolean is never a number and
a string is never a number.

Composition semantics (``merge``): mappings merge recursively; an overlay that
changes a mapping's discriminator replaces that mapping; named components merge
by name (existing names keep their position, new names append); free-keyed maps,
lists and scalars replace; writing null clears a value, which is how an overlay
switches between the members of an exclusive group.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from numbers import Real as _RealNumber
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "AnyValue", "Boolean", "ComponentName", "ConfigError", "Ellipticity", "FilePath", "Identifier",
    "Integer", "Key", "ListOf", "MapOf", "Named", "Nullable", "Pair", "REQUIRED", "Real", "Rule",
    "Sha256", "Shape", "Table", "Text", "Union", "Variants", "dataclass_table", "render_reference",
]

DISCRIMINATORS = ("type", "kind")


class ConfigError(ValueError):
    """A user-input error, located by the dotted key path (``""`` for a whole document)."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Final = _Required()


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def _invalid(check: Any, value: Any, path: str) -> ConfigError:
    return ConfigError(path, f"must be {check.describe()}, got {value!r}")


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, _RealNumber) and not isinstance(value, (bool, np.bool_))


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


# ------------------------------------------------------------------ value checks


@dataclass(frozen=True)
class Boolean:
    def accepts(self, value: Any) -> bool:
        return isinstance(value, bool)

    def describe(self) -> str:
        return "true or false"

    def __call__(self, value: Any, path: str) -> bool:
        if not self.accepts(value):
            raise _invalid(self, value, path)
        return value


@dataclass(frozen=True)
class Integer:
    min: int | None = None
    max: int | None = None

    def accepts(self, value: Any) -> bool:
        return _is_integer(value)

    def describe(self) -> str:
        if self.min is not None and self.max is not None:
            return f"integer in [{self.min}, {self.max}]"
        if self.min is not None:
            return f"integer >= {self.min}"
        if self.max is not None:
            return f"integer <= {self.max}"
        return "integer"

    def __call__(self, value: Any, path: str) -> int:
        if not (self.accepts(value) and (self.min is None or value >= self.min)
                and (self.max is None or value <= self.max)):
            raise _invalid(self, value, path)
        return int(value)


@dataclass(frozen=True)
class Real:
    min: float | None = None
    max: float | None = None
    min_open: bool = False
    max_open: bool = False

    def accepts(self, value: Any) -> bool:
        return _is_number(value)

    def describe(self) -> str:
        if self.min is not None and self.max is not None:
            left, right = "(" if self.min_open else "[", ")" if self.max_open else "]"
            return f"number in {left}{_number_text(self.min)}, {_number_text(self.max)}{right}"
        if self.min is not None:
            return f"number {'>' if self.min_open else '>='} {_number_text(self.min)}"
        if self.max is not None:
            return f"number {'<' if self.max_open else '<='} {_number_text(self.max)}"
        return "number"

    def __call__(self, value: Any, path: str) -> float:
        if not self.accepts(value):
            raise _invalid(self, value, path)
        try:
            number = float(value)
        except OverflowError:
            raise _invalid(self, value, path) from None
        if not math.isfinite(number):
            raise _invalid(self, value, path)
        if self.min is not None and (number < self.min or (self.min_open and number == self.min)):
            raise _invalid(self, value, path)
        if self.max is not None and (number > self.max or (self.max_open and number == self.max)):
            raise _invalid(self, value, path)
        return number


@dataclass(frozen=True)
class Text:
    choices: tuple[str, ...] | None = None
    pattern: str | None = None

    def accepts(self, value: Any) -> bool:
        return isinstance(value, str)

    def describe(self) -> str:
        if self.choices is not None:
            return "one of: " + ", ".join(self.choices)
        if self.pattern is not None:
            return f"text matching {self.pattern}"
        return "non-empty text"

    def __call__(self, value: Any, path: str) -> str:
        if not self.accepts(value) or not value:
            raise _invalid(self, value, path)
        if self.choices is not None and value not in self.choices:
            raise _invalid(self, value, path)
        if self.pattern is not None and re.fullmatch(self.pattern, value) is None:
            raise _invalid(self, value, path)
        return value


@dataclass(frozen=True)
class _Pattern:
    regex: typing.ClassVar[str]
    text: typing.ClassVar[str]

    def accepts(self, value: Any) -> bool:
        return isinstance(value, str)

    def describe(self) -> str:
        return self.text

    def __call__(self, value: Any, path: str) -> str:
        if not self.accepts(value) or re.fullmatch(self.regex, value) is None:
            raise _invalid(self, value, path)
        return value


@dataclass(frozen=True)
class Identifier(_Pattern):
    regex = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    text = "name of letters, digits, '.', '_', '-'"


@dataclass(frozen=True)
class ComponentName(_Pattern):
    regex = r"[a-z][a-z0-9_]*"
    text = "lower-case identifier ([a-z][a-z0-9_]*)"


@dataclass(frozen=True)
class Sha256(_Pattern):
    regex = r"[0-9a-f]{64}"
    text = "lowercase SHA-256 hex digest"


def _sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


@dataclass(frozen=True)
class Pair:
    item: Any

    def accepts(self, value: Any) -> bool:
        return _sequence(value)

    def describe(self) -> str:
        return f"pair, each {self.item.describe()}"

    def __call__(self, value: Any, path: str) -> list[Any]:
        if not self.accepts(value) or len(value) != 2:
            raise _invalid(self, value, path)
        return [self.item(item, f"{path}[{index}]") for index, item in enumerate(value)]


@dataclass(frozen=True)
class Shape:
    odd: bool = False

    def accepts(self, value: Any) -> bool:
        return _sequence(value)

    def describe(self) -> str:
        return "pair of odd positive integers" if self.odd else "pair of positive integers"

    def __call__(self, value: Any, path: str) -> list[int]:
        if not self.accepts(value) or len(value) != 2:
            raise _invalid(self, value, path)
        for index, item in enumerate(value):
            if not (_is_integer(item) and item >= 1 and (not self.odd or item % 2 == 1)):
                raise _invalid(self, value, f"{path}[{index}]")
        return [int(item) for item in value]


@dataclass(frozen=True)
class Ellipticity:
    def accepts(self, value: Any) -> bool:
        return _sequence(value)

    def describe(self) -> str:
        return "pair (e1, e2) with sqrt(e1^2 + e2^2) < 1"

    def __call__(self, value: Any, path: str) -> list[float]:
        if not self.accepts(value) or len(value) != 2:
            raise _invalid(self, value, path)
        components = [Real()(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if not math.hypot(*components) < 1.0:
            raise _invalid(self, value, path)
        return components


@dataclass(frozen=True)
class ListOf:
    item: Any
    min_length: int = 0
    length: int | None = None
    unique: bool = False

    def accepts(self, value: Any) -> bool:
        return _sequence(value)

    def describe(self) -> str:
        if self.length is not None:
            size = f" of {self.length} items"
        elif self.min_length:
            size = f" of at least {self.min_length} items"
        else:
            size = ""
        return f"list{size}, each {self.item.describe()}" + (", no repeats" if self.unique else "")

    def __call__(self, value: Any, path: str) -> list[Any]:
        if not self.accepts(value) or len(value) < self.min_length or (
                self.length is not None and len(value) != self.length):
            raise _invalid(self, value, path)
        items = [self.item(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if self.unique:
            for index, item in enumerate(items):
                if item in items[:index]:
                    raise ConfigError(f"{path}[{index}]", f"repeats {item!r}; items must be distinct")
        return items


@dataclass(frozen=True)
class MapOf:
    key: Any
    value: Any
    min_length: int = 0

    def accepts(self, value: Any) -> bool:
        return isinstance(value, Mapping)

    def describe(self) -> str:
        size = f" with at least {self.min_length} entries" if self.min_length else ""
        return f"mapping{size} of {self.key.describe()} to {self.value.describe()}"

    def __call__(self, value: Any, path: str) -> dict[Any, Any]:
        if not self.accepts(value) or len(value) < self.min_length:
            raise _invalid(self, value, path)
        return {self.key(key, _join(path, key)): self.value(item, _join(path, key))
                for key, item in value.items()}


@dataclass(frozen=True)
class FilePath:
    """An existing regular file. Paths are resolved against the file that declares them
    before the table reads them, so a relative path here is an error."""

    suffixes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.suffixes or not all(isinstance(s, str) and s.startswith(".") for s in self.suffixes):
            raise TypeError(f"FilePath suffixes must be non-empty '.ext' strings, got {self.suffixes!r}")

    def accepts(self, value: Any) -> bool:
        return isinstance(value, str)

    def describe(self) -> str:
        names = list(self.suffixes)
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        return f"path to an existing {listed} file"

    def __call__(self, value: Any, path: str) -> str:
        if not self.accepts(value) or not value:
            raise _invalid(self, value, path)
        location = Path(value).expanduser()
        if not location.is_absolute():
            raise ConfigError(path, f"relative path {value!r} was not resolved against its file")
        if not location.is_file():
            raise ConfigError(path, f"no such file: {value}")
        if location.suffix not in self.suffixes:
            raise _invalid(self, value, path)
        return str(location.resolve())


@dataclass(frozen=True)
class Nullable:
    check: Any

    def accepts(self, value: Any) -> bool:
        return value is None or self.check.accepts(value)

    def describe(self) -> str:
        return f"null or {self.check.describe()}"

    def __call__(self, value: Any, path: str) -> Any:
        return None if value is None else self.check(value, path)


@dataclass(frozen=True, init=False)
class Union:
    checks: tuple[Any, ...]

    def __init__(self, *checks: Any) -> None:
        if len(checks) < 2:
            raise TypeError("Union needs at least two checks")
        object.__setattr__(self, "checks", checks)

    def accepts(self, value: Any) -> bool:
        return any(check.accepts(value) for check in self.checks)

    def describe(self) -> str:
        texts = [check.describe() for check in self.checks]
        return " or ".join(texts) if len(texts) == 2 else ", ".join(texts[:-1]) + ", or " + texts[-1]

    def member(self, value: Any) -> Any:
        return next((check for check in self.checks if check.accepts(value)), None)

    def __call__(self, value: Any, path: str) -> Any:
        check = self.member(value)
        if check is None:
            raise _invalid(self, value, path)
        return check(value, path)


@dataclass(frozen=True)
class AnyValue:
    def accepts(self, value: Any) -> bool:
        return True

    def describe(self) -> str:
        return "any value"

    def __call__(self, value: Any, path: str) -> Any:
        return deepcopy(value)


# ------------------------------------------------------------------ tables


@dataclass(frozen=True)
class Key:
    name: str
    check: Any
    doc: str
    default: Any = REQUIRED
    unit: str | None = None


@dataclass(frozen=True)
class Rule:
    doc: str
    check: Callable[[Mapping[str, Any], str], None]


def _unwrap(check: Any) -> Any:
    return check.check if isinstance(check, Nullable) else check


def _is_container(check: Any) -> bool:
    return isinstance(_unwrap(check), _Container)


def _path_check(check: Any) -> bool:
    check = _unwrap(check)
    if isinstance(check, Union):
        return any(_path_check(member) for member in check.checks)
    return isinstance(check, FilePath)


def _normal_default(key: Key) -> Key:
    """``key`` with its default in read form (so ``read`` is a fixed point), checked against its domain.

    A non-null container default (``{}`` for a table) is read through at read time instead,
    so nested defaults fill in and nested required keys are reported at the user's path.
    """
    if key.default is REQUIRED or (key.default is not None and _is_container(key.check)):
        return key
    try:
        default = key.check(deepcopy(key.default), key.name)
    except ConfigError as error:
        raise TypeError(f"default of key {key.name!r} fails its own check: {error}") from error
    return dataclasses.replace(key, default=default)


def _merge_by_name(base: Any, overlay: Any, check_of: Callable[[Any], Any]) -> Any:
    """Overlay ``overlay`` onto ``base``: containers merge recursively, everything else replaces."""
    if not (isinstance(base, Mapping) and isinstance(overlay, Mapping)):
        return deepcopy(overlay)
    merged = deepcopy(dict(base))
    for name, value in overlay.items():
        check = check_of(name)
        if name in merged and _is_container(check):
            merged[name] = _unwrap(check).merge(merged[name], value)
        else:
            merged[name] = deepcopy(value)
    return merged


class _Container:
    """The surface Table, Variants and Named share: a mapping that reads, merges and walks paths."""

    def accepts(self, value: Any) -> bool:
        return isinstance(value, Mapping)

    def __call__(self, value: Any, path: str) -> dict[str, Any]:
        return self.read(value, path)

    def transform_paths(self, value: Any, fn: Callable[[str, FilePath], Any]) -> Any:
        """A copy of ``value`` with every file path replaced by ``fn(path, check)``."""
        return _walk(self, value, fn, records=False)

    def digest_record(self, value: Any, file_record: Callable[[str, FilePath], Any]) -> Any:
        """The identity record: paths through ``file_record``, named collections as ordered pairs."""
        return _walk(self, value, file_record, records=True)


@dataclass(frozen=True, eq=False)
class Table(_Container):
    """A mapping with a fixed key set, read and merged key by key.

    Defaults are stored in read form, and a default outside its key's domain is a TypeError
    when the table is built.
    """

    keys: tuple[Key, ...]
    exactly_one: tuple[tuple[str, ...], ...] = ()
    rules: tuple[Rule, ...] = ()
    doc: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(_normal_default(key) for key in self.keys))
        object.__setattr__(self, "exactly_one", tuple(tuple(group) for group in self.exactly_one))
        object.__setattr__(self, "rules", tuple(self.rules))
        by_name = {key.name: key for key in self.keys}
        if len(by_name) != len(self.keys):
            raise TypeError(f"duplicate key names in table: {[key.name for key in self.keys]}")
        for group in self.exactly_one:
            for name in group:
                key = by_name.get(name)
                if key is None or key.default is not None or not isinstance(key.check, Nullable):
                    raise TypeError(f"exclusive group member {name!r} must be a Nullable key with default None")
        object.__setattr__(self, "_checks", types.MappingProxyType({n: k.check for n, k in by_name.items()}))

    def describe(self) -> str:
        return "mapping"

    def read(self, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ConfigError(path, f"must be a mapping, got {value!r}")
        for name in value:
            if name not in self._checks:
                raise ConfigError(_join(path, name), "unknown key; allowed: " + ", ".join(sorted(self._checks)))
        values: dict[str, Any] = {}
        for key in self.keys:
            where = _join(path, key.name)
            if key.name in value:
                values[key.name] = key.check(value[key.name], where)
            elif key.default is REQUIRED:
                raise ConfigError(where, "required")
            elif key.default is not None and _is_container(key.check):
                values[key.name] = key.check(deepcopy(key.default), where)
            else:
                values[key.name] = deepcopy(key.default)
        for group in self.exactly_one:
            if sum(values[name] is not None for name in group) != 1:
                raise ConfigError(path, f"exactly one of {', '.join(group)} must be set")
        for rule in self.rules:
            rule.check(values, path)
        return values

    def merge(self, base: Any, overlay: Any) -> Any:
        return _merge_by_name(base, overlay, self._checks.get)

    def extend(self, keys: Sequence[Key] = (), *, exactly_one: Sequence[Sequence[str]] = (),
               rules: Sequence[Rule] = ()) -> Table:
        return Table(self.keys + tuple(keys), self.exactly_one + tuple(tuple(g) for g in exactly_one),
                     self.rules + tuple(rules), self.doc)


@dataclass(frozen=True, eq=False)
class Variants(_Container):
    """A mapping whose discriminator (``type`` or ``kind``) selects the table of its other keys.

    A key name declared by several variants has the same path-ness in all of them, and a
    container under such a name is the same table object, so a fragment without its
    discriminator merges and walks unambiguously through the union of the variant tables.
    """

    discriminator: str
    tables: Mapping[str, Table]
    default: str | None = None
    doc: str = ""

    def __post_init__(self) -> None:
        if self.discriminator not in DISCRIMINATORS:
            raise TypeError(f"discriminator must be one of {DISCRIMINATORS}, got {self.discriminator!r}")
        tables = types.MappingProxyType(dict(self.tables))
        if not tables or (self.default is not None and self.default not in tables):
            raise TypeError(f"variants need tables and a default among them, got {list(tables)}, {self.default!r}")
        shared: dict[str, Any] = {}
        for kind, table in tables.items():
            if self.discriminator in table._checks:
                raise TypeError(f"variant {kind!r} declares the discriminator {self.discriminator!r} as a key")
            for name, check in table._checks.items():
                first, other = _unwrap(shared.setdefault(name, check)), _unwrap(check)
                if _path_check(first) != _path_check(other) or (
                        (_is_container(first) or _is_container(other)) and first is not other):
                    raise TypeError(f"key {name!r} must have the same path checks in every variant "
                                    "(containers must be the same table object)")
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "_checks", types.MappingProxyType(shared))

    def describe(self) -> str:
        return f"mapping selected by {self.discriminator}: {', '.join(self.tables)}"

    def read(self, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ConfigError(path, f"must be a mapping, got {value!r}")
        choices = "one of: " + ", ".join(self.tables)
        kind = value.get(self.discriminator, self.default)
        if kind is None and self.discriminator not in value:
            raise ConfigError(_join(path, self.discriminator), f"required; {choices}")
        if not isinstance(kind, str) or kind not in self.tables:
            raise ConfigError(_join(path, self.discriminator), f"{choices}, got {kind!r}")
        rest = {name: item for name, item in value.items() if name != self.discriminator}
        return {self.discriminator: kind, **self.tables[kind].read(rest, path)}

    def _selected(self, value: Mapping[str, Any]) -> Table | Variants:
        """The table a fragment names, or these variants when its discriminator is absent or unknown."""
        kind = value.get(self.discriminator)
        return self.tables[kind] if isinstance(kind, str) and kind in self.tables else self

    def merge(self, base: Any, overlay: Any) -> Any:
        if not (isinstance(base, Mapping) and isinstance(overlay, Mapping)):
            return deepcopy(overlay)
        base_kind = base.get(self.discriminator, self.default)
        if overlay.get(self.discriminator, base_kind) != base_kind:
            return deepcopy(dict(overlay))
        selected = self.tables.get(base_kind) if isinstance(base_kind, str) else None
        return _merge_by_name(base, overlay, (selected or self)._checks.get)


@dataclass(frozen=True, eq=False)
class Named(_Container):
    """Components keyed by lower-case names, in input order (order is nuisance and summation order)."""

    item: Any
    min_length: int = 0

    def describe(self) -> str:
        size = f" (at least {self.min_length})" if self.min_length else ""
        return f"named components{size}"

    def read(self, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ConfigError(path, f"must be a mapping of named components, got {value!r}")
        components = {}
        for name, item in value.items():
            where = _join(path, name)
            ComponentName()(name, where)
            components[name] = self.item(item, where)
        if len(components) < self.min_length:
            raise ConfigError(path, f"needs at least {self.min_length} components, got {len(components)}")
        return components

    def merge(self, base: Any, overlay: Any) -> Any:
        return _merge_by_name(base, overlay, lambda name: self.item)


def _walk(check: Any, value: Any, fn: Callable[[str, FilePath], Any], *, records: bool) -> Any:
    """Copy ``value``, replacing every path by ``fn(path, check)``.

    Serves both ``transform_paths`` and ``digest_record`` (which also writes every
    named collection as ordered ``[name, record]`` pairs), so the two never disagree
    on which values are paths. Partial fragments and unknown keys are copied as they are.
    """
    check = _unwrap(check)
    if value is None:
        return None
    if isinstance(check, Union):
        check = _unwrap(check.member(value))
    if isinstance(check, FilePath):
        return fn(value, check) if isinstance(value, str) else deepcopy(value)
    if isinstance(check, (Table, Variants)) and isinstance(value, Mapping):
        checks = (check if isinstance(check, Table) else check._selected(value))._checks
        return {name: (_walk(checks[name], item, fn, records=records) if name in checks else deepcopy(item))
                for name, item in value.items()}
    if isinstance(check, Named) and isinstance(value, Mapping):
        pairs = [(name, _walk(check.item, item, fn, records=records)) for name, item in value.items()]
        return [[name, record] for name, record in pairs] if records else dict(pairs)
    if isinstance(check, (ListOf, Pair)) and _sequence(value):
        return [_walk(check.item, item, fn, records=records) for item in value]
    if isinstance(check, MapOf) and isinstance(value, Mapping):
        return {name: _walk(check.value, item, fn, records=records) for name, item in value.items()}
    return deepcopy(value)


# ------------------------------------------------------------------ settings dataclasses


def _annotation_check(annotation: Any, where: str) -> Any:
    origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
    if annotation is bool:
        return Boolean()
    if annotation is int:
        return Integer()
    if annotation is float:
        return Real()
    if annotation is str:
        return Text()
    if origin is typing.Literal and arguments and all(isinstance(a, str) for a in arguments):
        return Text(choices=tuple(arguments))
    if origin in (typing.Union, types.UnionType) and type(None) in arguments and len(arguments) == 2:
        (inner,) = [a for a in arguments if a is not type(None)]
        return Nullable(_annotation_check(inner, where))
    if origin is tuple and len(arguments) == 2 and arguments[1] is Ellipsis:
        return ListOf(_annotation_check(arguments[0], where))
    if origin is tuple and len(arguments) == 2 and arguments[0] == arguments[1]:
        return Pair(_annotation_check(arguments[0], where))
    raise TypeError(f"{where}: no key check for annotation {annotation!r}; build this table by hand")


def dataclass_table(cls: type, *, docs: Mapping[str, str],
                    units: Mapping[str, str] = types.MappingProxyType({})) -> Table:
    """The key table of a Python-constructed settings dataclass, from its fields and defaults.

    Supported field types: bool, int, float, str, Literal of strings, X | None, tuple[X, ...]
    (a list) and tuple[X, X] (a pair). The table stores defaults in their read form (tuples as
    lists, integers of float fields as floats). ``docs`` names every init field; ``units`` some.
    """
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a dataclass")
    fields = [field for field in dataclasses.fields(cls) if field.init]
    names = [field.name for field in fields]
    if sorted(docs) != sorted(names) or not set(units) <= set(names):
        raise TypeError(f"{cls.__name__}: docs must name exactly the fields {names} and units only fields")
    hints = typing.get_type_hints(cls)
    keys = []
    for field in fields:
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        else:
            default = REQUIRED
        check = _annotation_check(hints[field.name], f"{cls.__name__}.{field.name}")
        keys.append(Key(field.name, check, docs[field.name], default, units.get(field.name)))
    return Table(tuple(keys))


# ------------------------------------------------------------------ reference


_PLAIN = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")
_RESERVED = {"null", "true", "false", "yes", "no", "on", "off"}


def _flow(value: Any) -> str:
    """A default value as YAML flow text."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if _is_integer(value):
        return str(int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return ".nan"
        if math.isinf(value):
            return ".inf" if value > 0 else "-.inf"
        return repr(value)
    if isinstance(value, str):
        return value if _PLAIN.fullmatch(value) and value.lower() not in _RESERVED else json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flow(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_flow(key)}: {_flow(item)}" for key, item in value.items()) + "}"
    raise TypeError(f"cannot render default {value!r}")


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _value_text(check: Any, child: str) -> str:
    inner = _unwrap(check)
    prefix = "null or " if isinstance(check, Nullable) else ""
    if isinstance(inner, Table):
        return f"{prefix}mapping, see `{child}`"
    if isinstance(inner, Variants):
        return f"{prefix}mapping by `{inner.discriminator}` ({', '.join(inner.tables)}), see `{child}`"
    if isinstance(inner, Named):
        return f"{prefix}{inner.describe()}, see `{child}.<name>`"
    if isinstance(inner, ListOf) and _is_container(inner.item):
        return f"{prefix}{inner.describe()}, see `{child}[i]`"
    if isinstance(inner, MapOf) and _is_container(inner.value):
        return f"{prefix}{inner.describe()}, see `{child}.<key>`"
    return check.describe()


def _table_block(table: Table, path: str, heading: str, lead: Sequence[str]) -> str:
    lines = [f"## {heading}", ""]
    for paragraph in (*lead, table.doc):
        if paragraph:
            lines += [paragraph, ""]
    if table.keys:
        lines += ["| key | value | default | unit | meaning |", "|---|---|---|---|---|"]
        for key in table.keys:
            default = "required" if key.default is REQUIRED else f"`{_flow(key.default)}`"
            value = _value_text(key.check, _join(path, key.name))
            lines.append(f"| `{key.name}` | {_cell(value)} | {_cell(default)} | {key.unit or ''} "
                         f"| {_cell(key.doc)} |")
        lines.append("")
    else:
        lines += ["No keys.", ""]
    bullets = [f"- exactly one of {', '.join(f'`{name}`' for name in group)} is set; write null to clear one"
               for group in table.exactly_one] + [f"- {rule.doc}" for rule in table.rules]
    if bullets:
        lines += [*bullets, ""]
    return "\n".join(lines)


def _render(check: Any, path: str, blocks: list[tuple[str, str]]) -> None:
    check = _unwrap(check)
    if isinstance(check, Table):
        blocks.append((path, _table_block(check, path, path or "top level", ())))
        _render_children(check, path, blocks)
    elif isinstance(check, Variants):
        for kind, table in check.tables.items():
            selected = f"Selected by `{check.discriminator}: {kind}`" + (
                " (the default)." if kind == check.default else ".")
            heading = f"{path} ({check.discriminator}: {kind})"
            blocks.append((heading, _table_block(table, path, heading, (selected, check.doc))))
            _render_children(table, path, blocks)
    elif isinstance(check, Named):
        _render(check.item, f"{path}.<name>", blocks)
    elif isinstance(check, ListOf):
        _render(check.item, f"{path}[i]", blocks)
    elif isinstance(check, MapOf):
        _render(check.value, f"{path}.<key>", blocks)


def _render_children(table: Table, path: str, blocks: list[tuple[str, str]]) -> None:
    for key in table.keys:
        _render(key.check, _join(path, key.name), blocks)


def render_reference(documents: Sequence[tuple[str, Any]], *, section: str | None = None) -> str:
    """Markdown reference of every table: one heading per table at its dotted path.

    ``section`` keeps the headings at or below that dotted path.
    """
    blocks: list[tuple[str, str]] = []
    for name, check in documents:
        _render(check, name, blocks)
    if section is not None:
        blocks = [(path, text) for path, text in blocks
                  if path == section or path.startswith((f"{section}.", f"{section} (", f"{section}["))]
        if not blocks:
            raise ConfigError("", f"unknown section {section!r}")
    return "\n".join(text for _, text in blocks)
