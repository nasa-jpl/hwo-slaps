"""The ``psf`` configuration section, PSF providers, and the model PSF of every relation.

``psf.truth`` is the PSF that makes the data: an optical system, a detector kernel file,
or tabulated node kernels. ``psf.model`` is the PSF the analysis assumes: the truth
itself (``matched``, the default), another optical system or kernel file, the truth
optics with other wavefront coefficients (``wavefront``: replaced, or an ``offset`` added
to the truth's), or the truth optics plus a knowledge-error draw (``knowledge_error``).

A provider evaluates detector kernels; construction evaluates none. ``build_model_psf``
is the one place the model PSF of a relation is built.
"""

from __future__ import annotations

import io
import math
import numbers
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import numpy as np

from ..config.checks import (
    Boolean, ConfigError, FilePath, Integer, Key, Nullable, Real, Rule, Sha256, Shape, Table, Text, Variants,
)
from ..identity import array_digest, read_file_snapshot
from .kernels import PIXEL_SCALE_ATOL_ARCSEC, DetectorPSF
from .knowledge_error import (
    DRAW_TABLE, KnowledgeErrorDraw, WavefrontDrawSpec, draw_knowledge_error, draw_wavefront, parse_wavefront_draw,
)
from .optical_psf import OpticalPSF, OpticalSpec, check_sampling, checked_wavelengths, resolve_wavelengths
from .pupils import PUPIL_TABLE, CircularPupilSpec, build_pupil, parse_pupil
from .wavefront import WAVEFRONT_TABLE, WavefrontBasis, WavefrontCoefficients, validate_coefficients

__all__ = [
    "CROSS_RULES", "PSF_TABLE", "KernelFileSpec", "KernelPSF", "KnowledgeErrorModel", "MatchedModel", "ModelPSF",
    "PSFProvider", "PsfModelSpec", "PsfSpec", "PsfTruthSpec", "WavefrontModel", "build_model_psf",
    "build_psf_provider", "parse_psf", "KernelCubePSF", "KernelCubeSpec",
]


class PSFProvider(Protocol):
    """What the forecast, observation and inference layers need from a PSF."""

    @property
    def pixel_scale_arcsec(self) -> float: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def wavelengths_m(self) -> tuple[float, ...] | None: ...

    @property
    def basis(self) -> WavefrontBasis | None: ...

    @property
    def coefficients(self) -> WavefrontCoefficients | None: ...

    @property
    def collecting_area_m2(self) -> float | None: ...

    def kernel(self, wavelength_m: float | None = None, *,
               coefficients: WavefrontCoefficients | None = None) -> DetectorPSF: ...

    def kernels(self, wavelengths_m: Sequence[float] | None = None, *,
                coefficients: WavefrontCoefficients | None = None) -> tuple[DetectorPSF, ...]: ...

    @property
    def file_digests(self) -> Mapping[str, str]: ...

    def to_mapping(self) -> dict[str, Any]: ...


class KernelPSF:
    """One fixed detector kernel: no spectral information and no wavefront modes."""

    def __init__(self, psf: DetectorPSF) -> None:
        if not isinstance(psf, DetectorPSF):
            raise ValueError(f"a kernel provider wraps a DetectorPSF, got {type(psf).__name__}")
        self._psf = psf

    @property
    def psf(self) -> DetectorPSF:
        return self._psf

    def kernels(self, wavelengths_m: Sequence[float] | None = None, *,
                coefficients: WavefrontCoefficients | None = None) -> tuple[DetectorPSF, ...]:
        raise ValueError("a kernel PSF has no spectral information; it has one fixed kernel")

    @property
    def file_digests(self) -> Mapping[str, str]:
        source = self._psf.source
        files = {source["path"]: source["file_sha256"]} if "file_sha256" in source else {}
        return types.MappingProxyType(files)

    @property
    def pixel_scale_arcsec(self) -> float:
        return self._psf.pixel_scale_arcsec

    @property
    def shape(self) -> tuple[int, int]:
        return self._psf.shape

    @property
    def wavelengths_m(self) -> None:
        return None

    @property
    def basis(self) -> None:
        return None

    @property
    def coefficients(self) -> None:
        return None

    @property
    def collecting_area_m2(self) -> None:
        return None

    def kernel(self, wavelength_m: float | None = None, *,
               coefficients: WavefrontCoefficients | None = None) -> DetectorPSF:
        """The kernel itself; a wavelength or coefficients cannot be honoured and raise."""
        if wavelength_m is not None:
            raise ValueError("a kernel PSF has no spectral information; it has one kernel at every wavelength")
        if coefficients is not None:
            raise ValueError("a kernel PSF has no wavefront basis; wavefront coefficients need an optical PSF")
        return self._psf

    def to_mapping(self) -> dict[str, Any]:
        return {"provider": "kernel", "kernel": self._psf.kernel_identity().to_mapping(),
                "source": dict(self._psf.source)}


class KernelCubePSF:
    """Tabulated detector kernels evaluated only at their exact declared wavelengths."""

    def __init__(self, kernels: Sequence[DetectorPSF], wavelengths_m: Sequence[float]) -> None:
        wavelengths = checked_wavelengths(wavelengths_m)
        supplied = tuple(kernels)
        if len(supplied) != len(wavelengths) or not all(isinstance(kernel, DetectorPSF) for kernel in supplied):
            raise ValueError("a kernel cube needs one DetectorPSF per declared wavelength")
        first = supplied[0]
        if any(kernel.shape != first.shape or
               abs(kernel.pixel_scale_arcsec - first.pixel_scale_arcsec) > PIXEL_SCALE_ATOL_ARCSEC
               for kernel in supplied):
            raise ValueError("every cube node must have the same support and pixel sampling")
        nodes, files = [], {}
        for kernel, wavelength in zip(supplied, wavelengths):
            source = dict(kernel.source)
            if "file_sha256" in source:
                path, digest = source["path"], source["file_sha256"]
                if path in files and files[path] != digest:
                    raise ValueError(f"{path}: conflicting actual cube input digests")
                files[path] = digest
            if source.get("kind") != "cube" or source.get("wavelength_m") != wavelength \
                    or source["captured_power_fraction"] is not None:
                source.update(kind="cube", wavelength_m=wavelength, captured_power_fraction=None)
                kernel = DetectorPSF.from_array(kernel.kernel, kernel.pixel_scale_arcsec, normalize=False, source=source)
            nodes.append(kernel)
        self._kernels = tuple(nodes)
        self._wavelengths_m = wavelengths
        self._file_digests = types.MappingProxyType(files)
        if not all(math.isfinite(edge) for edge in self.support_m):
            raise ValueError("cube spectral coverage must be finite")

    @property
    def pixel_scale_arcsec(self) -> float:
        return self._kernels[0].pixel_scale_arcsec

    @property
    def shape(self) -> tuple[int, int]:
        return self._kernels[0].shape

    @property
    def wavelengths_m(self) -> tuple[float, ...]:
        return self._wavelengths_m

    @property
    def support_m(self) -> tuple[float, float]:
        nodes = self._wavelengths_m
        if len(nodes) == 1:
            return nodes[0], nodes[0]
        return nodes[0] - (nodes[1] - nodes[0]) / 2, nodes[-1] + (nodes[-1] - nodes[-2]) / 2

    @property
    def basis(self) -> None:
        return None

    @property
    def coefficients(self) -> None:
        return None

    @property
    def collecting_area_m2(self) -> None:
        return None

    @property
    def file_digests(self) -> Mapping[str, str]:
        return self._file_digests

    def validate_bandpass_support(self, support_m: tuple[float, float]) -> None:
        low, high = checked_wavelengths(support_m)
        if len(self._wavelengths_m) == 1:
            raise ValueError("a single-node kernel cube has no finite-width spectral coverage")
        first, last = self.support_m
        # Reconstructed midpoint-cell edges can differ from the original band edge
        # by rounding. This is equality slack, not wavelength extrapolation.
        equality_slack = 4 * np.finfo(float).eps
        below = low < first and not math.isclose(low, first, rel_tol=equality_slack, abs_tol=0.0)
        above = high > last and not math.isclose(high, last, rel_tol=equality_slack, abs_tol=0.0)
        if below or above:
            raise ValueError(f"bandpass support {(low, high)} m is outside cube coverage {(first, last)} m; "
                             "cube kernels are never extrapolated")

    def kernel(self, wavelength_m: float | None = None, *,
               coefficients: WavefrontCoefficients | None = None) -> DetectorPSF:
        if coefficients is not None:
            raise ValueError("a kernel cube has no wavefront basis")
        if wavelength_m is None:
            if len(self._kernels) != 1:
                raise ValueError("a kernel cube needs an exact tabulated wavelength")
            return self._kernels[0]
        wavelength = checked_wavelengths((wavelength_m,))[0]
        if wavelength not in self._wavelengths_m:
            raise ValueError(f"{wavelength!r} m is not a tabulated cube wavelength; no interpolation or extrapolation")
        return self._kernels[self._wavelengths_m.index(wavelength)]

    def kernels(self, wavelengths_m: Sequence[float] | None = None, *,
                coefficients: WavefrontCoefficients | None = None) -> tuple[DetectorPSF, ...]:
        if coefficients is not None:
            raise ValueError("a kernel cube has no wavefront basis")
        if wavelengths_m is None:
            return self._kernels
        return tuple(self.kernel(wavelength) for wavelength in checked_wavelengths(wavelengths_m))

    def to_mapping(self) -> dict[str, Any]:
        return {"provider": "kernel_cube", "wavelengths_m": list(self._wavelengths_m),
                "wavelengths_digest": array_digest(np.asarray(self._wavelengths_m)),
                "support_m": list(self.support_m), "pixel_scale_arcsec": self.pixel_scale_arcsec,
                "kernel_shape": list(self.shape), "file_digests": dict(self._file_digests),
                "kernels": [kernel.kernel_identity().to_mapping() for kernel in self._kernels]}


@dataclass(frozen=True)
class KernelCubeSpec:
    path: Path
    pixel_scale_arcsec: float
    normalize: bool
    file_sha256: str | None


@dataclass(frozen=True)
class KernelFileSpec:
    """A detector kernel file at ``pixel_scale_arcsec``; ``array_key`` names the ``.npz`` member."""

    path: Path
    array_key: str | None
    pixel_scale_arcsec: float
    normalize: bool
    file_sha256: str | None


@dataclass(frozen=True)
class MatchedModel:
    """The model PSF is the truth PSF."""


@dataclass(frozen=True)
class WavefrontModel:
    """The truth optics with ``wavefront`` in place of the truth coefficients, or ``offset`` added to them."""

    wavefront: WavefrontCoefficients | None
    offset: WavefrontCoefficients | None


@dataclass(frozen=True)
class KnowledgeErrorModel:
    """The truth optics plus a knowledge-error draw."""

    draw: WavefrontDrawSpec


PsfTruthSpec = OpticalSpec | KernelFileSpec | KernelCubeSpec
PsfModelSpec = MatchedModel | KernelFileSpec | OpticalSpec | WavefrontModel | KnowledgeErrorModel


@dataclass(frozen=True)
class PsfSpec:
    truth: PsfTruthSpec
    model: PsfModelSpec


@dataclass(frozen=True)
class ModelPSF:
    """The model PSF: its provider, its relation to the truth, and the knowledge error drawn for it."""

    provider: PSFProvider
    relation: Literal["matched", "kernel", "optical", "wavefront", "knowledge_error"]
    knowledge_error: KnowledgeErrorDraw | None


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _check_kernel_file(values: Mapping[str, Any], path: str) -> None:
    if values["path"].endswith(".npy") and values["array_key"] is not None:
        raise ConfigError(_join(path, "array_key"), "a .npy kernel file has no members; array_key must be null")


KERNEL_FILE_TABLE = Table((
    Key("path", FilePath((".npy", ".npz")), "detector kernel file"),
    Key("array_key", Nullable(Text()), "member of a .npz file (null reads kernel); null for .npy", default=None),
    Key("pixel_scale_arcsec", Real(min=0.0, min_open=True), "angular sampling of the kernel", unit="arcsec"),
    Key("normalize", Boolean(), "divide the kernel by its sum; false requires a sum within 1e-10 of one",
        default=True),
    Key("file_sha256", Nullable(Sha256()), "SHA-256 of the file bytes, checked before the file is read",
        default=None),
), rules=(Rule("a .npy file takes no array_key", _check_kernel_file),))


KERNEL_CUBE_TABLE = Table((
    Key("path", FilePath((".npz",)), "kernels and wavelengths_m members in one .npz snapshot"),
    Key("pixel_scale_arcsec", Real(min=0.0, min_open=True), "angular sampling of every cube slice", unit="arcsec"),
    Key("normalize", Boolean(), "divide each slice by its sum; false preserves unit-kernel bytes", default=True),
    Key("file_sha256", Nullable(Sha256()), "SHA-256 of the bytes decoded for both cube members", default=None),
))


def _check_optical(values: Mapping[str, Any], path: str) -> None:
    wavefront = values["wavefront"]
    if values["draw"] is not None and (wavefront["segment_hexikes"] or wavefront["zernikes"]):
        raise ConfigError(_join(path, "draw"), "a drawn optical wavefront excludes listed wavefront coefficients")


OPTICAL_TABLE = Table((
    Key("pupil", PUPIL_TABLE, "pupil geometry and sampling"),
    Key("focal_length_m", Real(min=0.0, min_open=True), "effective focal length", unit="m"),
    Key("wavelength_nm", Nullable(Real(min=0.0, min_open=True)), "wavelength of the monochromatic kernel",
        unit="nm", default=None),
    Key("wavelength_samples", Nullable(Integer(min=1)), "number of caller-supplied bandpass nodes", default=None),
    Key("detector_oversampling", Integer(min=1),
        "sub-samples per detector pixel side for the pixel integral (paper 3)"),
    Key("kernel_shape", Shape(odd=True), "kernel support (ny, nx), both odd", unit="pixels"),
    Key("wavefront", WAVEFRONT_TABLE, "truth wavefront coefficients", default={}),
    Key("draw", Nullable(DRAW_TABLE), "truth wavefront drawn from a prior at an exact RMS", default=None),
), exactly_one=(("wavelength_nm", "wavelength_samples"),),
    rules=(Rule("draw excludes listed wavefront coefficients", _check_optical),))

WAVEFRONT_MODEL_TABLE = Table((
    Key("wavefront", Nullable(WAVEFRONT_TABLE), "coefficients replacing the truth coefficients", default=None),
    Key("offset", Nullable(WAVEFRONT_TABLE), "coefficients added to the truth coefficients", default=None),
), exactly_one=(("wavefront", "offset"),))

_OPTICAL_MODELS = ("optical", "wavefront", "knowledge_error")


def _check_relations(values: Mapping[str, Any], path: str) -> None:
    """Model kinds that need an optical truth; segment modes need a hex pupil with those segments."""
    truth, model = values["truth"], values["model"]
    if model["kind"] in _OPTICAL_MODELS and truth["kind"] != "optical":
        raise ConfigError(_join(path, "model.kind"),
                          f"model kind {model['kind']} needs an optical truth")
    if truth["kind"] != "optical":
        return
    if model["kind"] == "optical":
        if any(model[key] != truth[key] for key in ("wavelength_nm", "wavelength_samples")):
            raise ConfigError(_join(path, "model"), "an optical model must have the truth's wavelength keys")
        model_pupil = parse_pupil(model["pupil"], _join(path, "model.pupil"))
        validate_coefficients(WavefrontCoefficients.from_mapping(model["wavefront"], _join(path, "model.wavefront")),
                              model_pupil, _join(path, "model.wavefront"))
        if model["draw"] is not None and model["draw"]["family"] != "global" and isinstance(model_pupil, CircularPupilSpec):
            raise ConfigError(_join(path, "model.draw.family"), "a segment draw needs a hex-segmented model pupil")
    pupil = parse_pupil(truth["pupil"], _join(path, "truth.pupil"))
    blocks = [("truth.wavefront", truth["wavefront"])]
    if model["kind"] == "wavefront":
        blocks += [(f"model.{name}", model[name]) for name in ("wavefront", "offset") if model[name] is not None]
    for where, block in blocks:
        validate_coefficients(WavefrontCoefficients.from_mapping(block, _join(path, where)), pupil,
                              _join(path, where))
    draws = [("truth.draw", truth["draw"])]
    if model["kind"] == "knowledge_error":
        draws.append(("model.draw", model["draw"]))
    for where, draw in draws:
        if draw is not None and draw["family"] != "global" and isinstance(pupil, CircularPupilSpec):
            raise ConfigError(_join(path, f"{where}.family"),
                              f"a {draw['family']} draw needs segment hexikes; a circular pupil has none (use global)")


PSF_TABLE = Table((
    Key("truth", Variants("kind", {"kernel": KERNEL_FILE_TABLE, "kernel_cube": KERNEL_CUBE_TABLE,
                                  "optical": OPTICAL_TABLE}),
        "the PSF that makes the data"),
    Key("model", Variants("kind", {
        "matched": Table(()),
        "kernel": KERNEL_FILE_TABLE,
        "optical": OPTICAL_TABLE,
        "wavefront": WAVEFRONT_MODEL_TABLE,
        "knowledge_error": Table((Key("draw", DRAW_TABLE, "knowledge-error draw added to the truth coefficients"),)),
    }, default="matched"), "the PSF the analysis assumes", default={}),
), rules=(Rule("optical, wavefront and knowledge_error models need an optical truth; segment hexikes and segment draws "
               "need a hex-segmented pupil with those segments", _check_relations),))


def _kernel_file_spec(values: Mapping[str, Any]) -> KernelFileSpec:
    path = Path(values["path"])
    array_key = values["array_key"]
    if array_key is None and path.suffix == ".npz":
        array_key = "kernel"
    return KernelFileSpec(path, array_key, values["pixel_scale_arcsec"], values["normalize"], values["file_sha256"])


def _optical_spec(values: Mapping[str, Any], path: str) -> OpticalSpec:
    draw = None if values["draw"] is None else parse_wavefront_draw(values["draw"], _join(path, "draw"))
    return OpticalSpec(parse_pupil(values["pupil"], _join(path, "pupil")), values["focal_length_m"],
                       None if values["wavelength_nm"] is None else values["wavelength_nm"] / 1e9,
                       values["detector_oversampling"], tuple(values["kernel_shape"]),
                       WavefrontCoefficients.from_mapping(values["wavefront"], _join(path, "wavefront")), draw,
                       values["wavelength_samples"])


def parse_psf(mapping: Mapping[str, Any], path: str = "psf") -> PsfSpec:
    """The ``psf`` section, read strictly through ``PSF_TABLE``."""
    values = PSF_TABLE.read(mapping, path)
    truth = values["truth"]
    if truth["kind"] == "kernel":
        truth_spec: PsfTruthSpec = _kernel_file_spec(truth)
    elif truth["kind"] == "kernel_cube":
        truth_spec = KernelCubeSpec(Path(truth["path"]), truth["pixel_scale_arcsec"], truth["normalize"], truth["file_sha256"])
    else:
        truth_spec = _optical_spec(truth, _join(path, "truth"))
    model = values["model"]
    where = _join(path, "model")
    if model["kind"] == "matched":
        model_spec: PsfModelSpec = MatchedModel()
    elif model["kind"] == "kernel":
        model_spec = _kernel_file_spec(model)
    elif model["kind"] == "optical":
        model_spec = _optical_spec(model, where)
    elif model["kind"] == "wavefront":
        replaced, offset = (None if model[name] is None else
                            WavefrontCoefficients.from_mapping(model[name], _join(where, name))
                            for name in ("wavefront", "offset"))
        model_spec = WavefrontModel(replaced, offset)
    else:
        model_spec = KnowledgeErrorModel(parse_wavefront_draw(model["draw"], _join(where, "draw")))
    return PsfSpec(truth_spec, model_spec)


def _kernel_files(psf: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    return [(section, psf[section]) for section in ("truth", "model")
            if psf[section]["kind"] in ("kernel", "kernel_cube")]


def _same_pixel_scale(first: float, second: float) -> bool:
    """Whether two pixel scales are one sampling within ``PIXEL_SCALE_ATOL_ARCSEC`` (never for NaN)."""
    return abs(first - second) <= PIXEL_SCALE_ATOL_ARCSEC


def _check_kernel_pixel_scales(root: Mapping[str, Any], path: str) -> None:
    scale = root["scene"]["grid"]["pixel_scale_arcsec"]
    for section, kernel_file in _kernel_files(root["psf"]):
        if not _same_pixel_scale(kernel_file["pixel_scale_arcsec"], scale):
            raise ConfigError(f"psf.{section}.pixel_scale_arcsec",
                              f"the kernel is sampled at {kernel_file['pixel_scale_arcsec']!r} arcsec but "
                              f"scene.grid.pixel_scale_arcsec is {scale!r}; kernels are never resampled")


def _check_truth_sampling(root: Mapping[str, Any], path: str) -> None:
    for section in ("truth", "model"):
        optical = root["psf"][section]
        if optical["kind"] != "optical":
            continue
        where = f"psf.{section}"
        spec = _optical_spec(optical, where)
        if spec.wavelength_m is not None:
            nodes = (spec.wavelength_m,)
        else:
            band = root.get("instrument", {}).get("bandpass")
            if band is None or band["kind"] != "top_hat":
                continue  # loaded table/product nodes are checked by the runtime constructor
            from ..spectra.bandpass import bandpass_nodes
            nodes = bandpass_nodes((band["min_nm"] / 1e9, band["max_nm"] / 1e9), spec.wavelength_samples)
        try:
            for wavelength in nodes:
                check_sampling(spec, pixel_scale_arcsec=root["scene"]["grid"]["pixel_scale_arcsec"],
                               wavelength_m=wavelength)
        except ValueError as error:
            raise ConfigError(where, str(error)) from error


def _check_spectral_inputs(root: Mapping[str, Any], path: str) -> None:
    truth = root["psf"]["truth"]
    if truth["kind"] != "optical" or truth["wavelength_samples"] is None:
        return
    if root.get("instrument", {}).get("bandpass") is None:
        raise ConfigError("instrument.bandpass", "wavelength_samples requires a bandpass")
    for plane in ("lens", "source"):
        for name, light in root["scene"][plane]["light"].items():
            if light.get("sed") is None:
                raise ConfigError(f"scene.{plane}.light.{name}.sed",
                                  "wavelength_samples requires an SED for every light component")


CROSS_RULES: tuple[Rule, ...] = (
    Rule("every kernel file is sampled at scene.grid.pixel_scale_arcsec (X1)", _check_kernel_pixel_scales),
    Rule("optical truth and model nodes are neither aliased nor under-resolved at the scene pixel scale (X4)",
         _check_truth_sampling),
    Rule("wavelength_samples requires a bandpass and component SEDs (X5)", _check_spectral_inputs),
)
"""Rules spanning ``psf`` and ``scene``; ``config.schema`` runs them on the read root values."""


def _scene_pixel_scale(value: Any) -> float:
    """The scene pixel scale a builder receives: a finite positive real number (bool refused)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"pixel_scale_arcsec must be finite and positive, got {value!r}")
    return float(value)


def _require_pixel_scale(psf: DetectorPSF, pixel_scale_arcsec: float, what: str) -> None:
    if not _same_pixel_scale(psf.pixel_scale_arcsec, pixel_scale_arcsec):
        raise ValueError(f"{what} is sampled at {psf.pixel_scale_arcsec!r} arcsec, the scene at "
                         f"{pixel_scale_arcsec!r} arcsec; kernels are never resampled")


def _kernel_provider(spec: KernelFileSpec, pixel_scale_arcsec: float, what: str) -> KernelPSF:
    psf = DetectorPSF.from_file(spec.path, pixel_scale_arcsec=spec.pixel_scale_arcsec, array_key=spec.array_key,
                                normalize=spec.normalize, file_sha256=spec.file_sha256)
    _require_pixel_scale(psf, pixel_scale_arcsec, what)
    return KernelPSF(psf)


def _cube_provider(spec: KernelCubeSpec, pixel_scale_arcsec: float) -> KernelCubePSF:
    content, digest = read_file_snapshot(spec.path)
    if spec.file_sha256 is not None and digest != spec.file_sha256:
        raise ValueError(f"{spec.path}: file SHA-256 is {digest}, the configuration states {spec.file_sha256}")
    with np.load(io.BytesIO(content), allow_pickle=False) as members:
        if not {"kernels", "wavelengths_m"} <= set(members.files):
            raise ValueError(f"{spec.path}: a cube requires kernels and wavelengths_m members")
        values, wavelengths = members["kernels"], members["wavelengths_m"]
    if values.ndim != 3 or wavelengths.ndim != 1 or values.shape[0] != len(wavelengths):
        raise ValueError(f"{spec.path}: kernels must be (n, ny, nx) and wavelengths_m must be (n,)")
    nodes = checked_wavelengths(wavelengths)
    kernels = tuple(DetectorPSF.from_array(values[index], spec.pixel_scale_arcsec, normalize=spec.normalize,
                   source={"kind": "cube", "path": str(spec.path), "file_sha256": digest, "array_index": index,
                           "wavelength_m": wavelength, "normalized": spec.normalize, "captured_power_fraction": None})
                    for index, wavelength in enumerate(nodes))
    provider = KernelCubePSF(kernels, nodes)
    _require_pixel_scale(kernels[0], pixel_scale_arcsec, "psf.truth")
    return provider


def build_psf_provider(spec: PsfTruthSpec, *, pixel_scale_arcsec: float,
                       wavelengths_m: Sequence[float] | None = None,
                       bandpass_support_m: tuple[float, float] | None = None) -> PSFProvider:
    """The truth provider at the scene pixel scale; evaluates no kernel (a kernel file is read)."""
    scale = _scene_pixel_scale(pixel_scale_arcsec)
    if bandpass_support_m is not None and not isinstance(spec, KernelCubeSpec):
        raise ValueError("bandpass_support_m checks only a tabulated kernel cube")
    if isinstance(spec, KernelFileSpec):
        if wavelengths_m is not None:
            raise ValueError("a fixed kernel has no spectral information; supplied wavelength nodes cannot be honoured")
        return _kernel_provider(spec, scale, "psf.truth")
    if isinstance(spec, KernelCubeSpec):
        if wavelengths_m is not None:
            raise ValueError("a kernel cube declares its own tabulated wavelengths; supplied nodes cannot replace them")
        provider = _cube_provider(spec, scale)
        if bandpass_support_m is not None:
            provider.validate_bandpass_support(bandpass_support_m)
        return provider
    if not isinstance(spec, OpticalSpec):
        raise ValueError(f"a truth PSF is optical, a kernel file or a kernel cube, got {type(spec).__name__}")
    nodes = resolve_wavelengths(spec, wavelengths_m=wavelengths_m)
    for wavelength in nodes:
        check_sampling(spec, pixel_scale_arcsec=scale, wavelength_m=wavelength)
    pupil = build_pupil(spec.pupil)
    basis = WavefrontBasis(pupil, reference_wavelength_m=min(nodes))
    draw = None
    coefficients = spec.wavefront
    if spec.draw is not None:
        if not spec.wavefront.is_empty:
            raise ValueError("a drawn truth wavefront excludes listed wavefront coefficients")
        draw = draw_wavefront(basis, spec.draw)
        coefficients = draw.coefficients
    return OpticalPSF(spec, pupil=pupil, basis=basis, pixel_scale_arcsec=scale, coefficients=coefficients,
                      draw=draw, wavelengths_m=nodes)


def _optical_truth(truth: PSFProvider, relation: str) -> OpticalPSF:
    if not isinstance(truth, OpticalPSF):
        raise ValueError(f"a {relation} model needs an optical truth PSF")
    return truth


def build_model_psf(spec: PsfModelSpec, truth: PSFProvider, *, pixel_scale_arcsec: float,
                    wavelengths_m: Sequence[float] | None = None) -> ModelPSF:
    """The model PSF of ``spec`` relative to ``truth``, both at the scene pixel scale."""
    scale = _scene_pixel_scale(pixel_scale_arcsec)
    if not _same_pixel_scale(truth.pixel_scale_arcsec, scale):
        raise ValueError(f"the truth PSF is sampled at {truth.pixel_scale_arcsec!r} arcsec, the scene at "
                         f"{scale!r} arcsec")
    if isinstance(spec, MatchedModel):
        return ModelPSF(truth, "matched", None)
    if isinstance(spec, KernelFileSpec):
        return ModelPSF(_kernel_provider(spec, scale, "psf.model"), "kernel", None)
    if isinstance(spec, OpticalSpec):
        optical = _optical_truth(truth, "optical")
        if (spec.wavelength_m, spec.wavelength_samples) != (optical.spec.wavelength_m, optical.spec.wavelength_samples):
            raise ValueError("an optical model must have the truth's wavelength keys")
        nodes = optical.wavelengths_m if wavelengths_m is None else checked_wavelengths(wavelengths_m)
        if nodes != optical.wavelengths_m:
            raise ValueError("an optical model must use the truth's wavelength nodes")
        return ModelPSF(build_psf_provider(spec, pixel_scale_arcsec=scale, wavelengths_m=nodes), "optical", None)
    if isinstance(spec, WavefrontModel):
        optical = _optical_truth(truth, "wavefront")
        if (spec.wavefront is None) == (spec.offset is None):
            raise ValueError("a wavefront model sets exactly one of wavefront and offset")
        coefficients = spec.wavefront if spec.offset is None else optical.coefficients.plus(spec.offset)
        return ModelPSF(optical.with_coefficients(coefficients), "wavefront", None)
    if isinstance(spec, KnowledgeErrorModel):
        optical = _optical_truth(truth, "knowledge_error")
        knowledge_error = draw_knowledge_error(optical, spec.draw)
        return ModelPSF(optical.with_coefficients(knowledge_error.model), "knowledge_error", knowledge_error)
    raise ValueError(f"unknown model PSF specification {type(spec).__name__}")
