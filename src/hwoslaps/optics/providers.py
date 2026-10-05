"""The ``psf`` configuration section, PSF providers, and the model PSF of every relation.

``psf.truth`` is the PSF that makes the data: an optical system (``kind: optical``) or a
detector kernel file (``kind: kernel``). ``psf.model`` is the PSF the analysis assumes:
the truth itself (``matched``, the default), another kernel file (``kernel``), the truth
optics with other wavefront coefficients (``wavefront``: replaced, or an ``offset`` added
to the truth's), or the truth optics plus a knowledge-error draw (``knowledge_error``).

A provider evaluates detector kernels; construction evaluates none. ``build_model_psf``
is the one place the model PSF of a relation is built.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from ..config.checks import (
    Boolean, ConfigError, FilePath, Integer, Key, Nullable, Real, Rule, Sha256, Shape, Table, Text, Variants,
)
from .kernels import PIXEL_SCALE_ATOL_ARCSEC, DetectorPSF
from .knowledge_error import (
    DRAW_TABLE, KnowledgeErrorDraw, WavefrontDrawSpec, draw_knowledge_error, draw_wavefront, parse_wavefront_draw,
)
from .optical_psf import OpticalPSF, OpticalSpec, check_sampling
from .pupils import PUPIL_TABLE, CircularPupilSpec, build_pupil, parse_pupil
from .wavefront import WAVEFRONT_TABLE, WavefrontBasis, WavefrontCoefficients, validate_coefficients

__all__ = [
    "CROSS_RULES", "PSF_TABLE", "KernelFileSpec", "KernelPSF", "KnowledgeErrorModel", "MatchedModel", "ModelPSF",
    "PSFProvider", "PsfModelSpec", "PsfSpec", "PsfTruthSpec", "WavefrontModel", "build_model_psf",
    "build_psf_provider", "parse_psf",
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


PsfTruthSpec = OpticalSpec | KernelFileSpec
PsfModelSpec = MatchedModel | KernelFileSpec | WavefrontModel | KnowledgeErrorModel


@dataclass(frozen=True)
class PsfSpec:
    truth: PsfTruthSpec
    model: PsfModelSpec


@dataclass(frozen=True)
class ModelPSF:
    """The model PSF: its provider, its relation to the truth, and the knowledge error drawn for it."""

    provider: PSFProvider
    relation: Literal["matched", "kernel", "wavefront", "knowledge_error"]
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


def _check_optical(values: Mapping[str, Any], path: str) -> None:
    wavefront = values["wavefront"]
    if values["draw"] is not None and (wavefront["segment_hexikes"] or wavefront["zernikes"]):
        raise ConfigError(_join(path, "draw"), "a drawn truth wavefront excludes listed wavefront coefficients")


OPTICAL_TABLE = Table((
    Key("pupil", PUPIL_TABLE, "pupil geometry and sampling"),
    Key("focal_length_m", Real(min=0.0, min_open=True), "effective focal length", unit="m"),
    Key("wavelength_nm", Real(min=0.0, min_open=True), "wavelength of the monochromatic kernel", unit="nm"),
    Key("detector_oversampling", Integer(min=1),
        "sub-samples per detector pixel side for the pixel integral (paper 3)"),
    Key("kernel_shape", Shape(odd=True), "kernel support (ny, nx), both odd", unit="pixels"),
    Key("wavefront", WAVEFRONT_TABLE, "truth wavefront coefficients", default={}),
    Key("draw", Nullable(DRAW_TABLE), "truth wavefront drawn from a prior at an exact RMS", default=None),
), rules=(Rule("draw excludes listed wavefront coefficients", _check_optical),))

WAVEFRONT_MODEL_TABLE = Table((
    Key("wavefront", Nullable(WAVEFRONT_TABLE), "coefficients replacing the truth coefficients", default=None),
    Key("offset", Nullable(WAVEFRONT_TABLE), "coefficients added to the truth coefficients", default=None),
), exactly_one=(("wavefront", "offset"),))

_OPTICAL_MODELS = ("wavefront", "knowledge_error")


def _check_relations(values: Mapping[str, Any], path: str) -> None:
    """Model kinds that need an optical truth; segment modes need a hex pupil with those segments."""
    truth, model = values["truth"], values["model"]
    if model["kind"] in _OPTICAL_MODELS and truth["kind"] != "optical":
        raise ConfigError(_join(path, "model.kind"),
                          f"model kind {model['kind']} changes the truth wavefront and needs an optical truth")
    if truth["kind"] != "optical":
        return
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
    Key("truth", Variants("kind", {"kernel": KERNEL_FILE_TABLE, "optical": OPTICAL_TABLE}),
        "the PSF that makes the data"),
    Key("model", Variants("kind", {
        "matched": Table(()),
        "kernel": KERNEL_FILE_TABLE,
        "wavefront": WAVEFRONT_MODEL_TABLE,
        "knowledge_error": Table((Key("draw", DRAW_TABLE, "knowledge-error draw added to the truth coefficients"),)),
    }, default="matched"), "the PSF the analysis assumes", default={}),
), rules=(Rule("wavefront and knowledge_error models need an optical truth; segment hexikes and segment draws "
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
                       values["wavelength_nm"] / 1e9, values["detector_oversampling"], tuple(values["kernel_shape"]),
                       WavefrontCoefficients.from_mapping(values["wavefront"], _join(path, "wavefront")), draw)


def parse_psf(mapping: Mapping[str, Any], path: str = "psf") -> PsfSpec:
    """The ``psf`` section, read strictly through ``PSF_TABLE``."""
    values = PSF_TABLE.read(mapping, path)
    truth = values["truth"]
    truth_spec = (_kernel_file_spec(truth) if truth["kind"] == "kernel"
                  else _optical_spec(truth, _join(path, "truth")))
    model = values["model"]
    where = _join(path, "model")
    if model["kind"] == "matched":
        model_spec: PsfModelSpec = MatchedModel()
    elif model["kind"] == "kernel":
        model_spec = _kernel_file_spec(model)
    elif model["kind"] == "wavefront":
        replaced, offset = (None if model[name] is None else
                            WavefrontCoefficients.from_mapping(model[name], _join(where, name))
                            for name in ("wavefront", "offset"))
        model_spec = WavefrontModel(replaced, offset)
    else:
        model_spec = KnowledgeErrorModel(parse_wavefront_draw(model["draw"], _join(where, "draw")))
    return PsfSpec(truth_spec, model_spec)


def _kernel_files(psf: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    return [(section, psf[section]) for section in ("truth", "model") if psf[section]["kind"] == "kernel"]


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
    truth = root["psf"]["truth"]
    if truth["kind"] != "optical":
        return
    spec = _optical_spec(truth, "psf.truth")
    try:
        check_sampling(spec, pixel_scale_arcsec=root["scene"]["grid"]["pixel_scale_arcsec"],
                       wavelength_m=spec.wavelength_m)
    except ValueError as error:
        raise ConfigError("psf.truth", str(error)) from error


CROSS_RULES: tuple[Rule, ...] = (
    Rule("every kernel file is sampled at scene.grid.pixel_scale_arcsec (X1)", _check_kernel_pixel_scales),
    Rule("an optical truth at the scene pixel scale is neither aliased nor under-resolved (X4)",
         _check_truth_sampling),
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


def build_psf_provider(spec: PsfTruthSpec, *, pixel_scale_arcsec: float) -> PSFProvider:
    """The truth provider at the scene pixel scale; evaluates no kernel (a kernel file is read)."""
    scale = _scene_pixel_scale(pixel_scale_arcsec)
    if isinstance(spec, KernelFileSpec):
        return _kernel_provider(spec, scale, "psf.truth")
    if not isinstance(spec, OpticalSpec):
        raise ValueError(f"a truth PSF is an OpticalSpec or a KernelFileSpec, got {type(spec).__name__}")
    pupil = build_pupil(spec.pupil)
    basis = WavefrontBasis(pupil, reference_wavelength_m=spec.wavelength_m)
    draw = None
    coefficients = spec.wavefront
    if spec.draw is not None:
        if not spec.wavefront.is_empty:
            raise ValueError("a drawn truth wavefront excludes listed wavefront coefficients")
        draw = draw_wavefront(basis, spec.draw)
        coefficients = draw.coefficients
    return OpticalPSF(spec, pupil=pupil, basis=basis, pixel_scale_arcsec=scale, coefficients=coefficients,
                      draw=draw)


def _optical_truth(truth: PSFProvider, relation: str) -> OpticalPSF:
    if not isinstance(truth, OpticalPSF):
        raise ValueError(f"a {relation} model changes the truth wavefront and needs an optical truth PSF")
    return truth


def build_model_psf(spec: PsfModelSpec, truth: PSFProvider, *, pixel_scale_arcsec: float) -> ModelPSF:
    """The model PSF of ``spec`` relative to ``truth``, both at the scene pixel scale."""
    scale = _scene_pixel_scale(pixel_scale_arcsec)
    if not _same_pixel_scale(truth.pixel_scale_arcsec, scale):
        raise ValueError(f"the truth PSF is sampled at {truth.pixel_scale_arcsec!r} arcsec, the scene at "
                         f"{scale!r} arcsec")
    if isinstance(spec, MatchedModel):
        return ModelPSF(truth, "matched", None)
    if isinstance(spec, KernelFileSpec):
        return ModelPSF(_kernel_provider(spec, scale, "psf.model"), "kernel", None)
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
