"""JAX node templates with dynamic radial tables and host profiled-bank finishing."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..statistics import BankReductions, SignalBankResult
from .base import BankAccumulator, EngineContext, checked_positions
from .fft import convolution_fft_shape
from .jax_profiles import build_light_evaluator, verification_points
from .radial import (MATCHED_SAMPLES, MISMATCHED_SAMPLES, check_coverage, interpolate_log_grid,
                     radial_deflection, radial_grid)


class JaxTemplateEngine:
    kind = "jax"

    def __init__(self, context: EngineContext, *, batch_size: int, progress: bool) -> None:
        jax.config.update("jax_enable_x64", True)
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.context, self.batch_size, self.progress = context, int(batch_size), progress
        self._closed = False
        scene = context.smooth_scene
        if scene.plane_count > 2 or context.hypothesis_redshift != scene.spec.lens.redshift:
            raise ValueError("JAX templates require a two-plane scene and a lens-plane subhalo")
        self._shape_native = scene.spec.grid.shape
        n_pix = self._shape_native[0] * self._shape_native[1]
        coordinates = np.asarray(scene.grid.over_sampled, dtype=float)
        self._sub_per_pix = coordinates.shape[0] // n_pix
        if (coordinates.shape[0] != n_pix * self._sub_per_pix or
                not bool(np.all(np.asarray(scene.grid.over_sampler.sub_is_uniform)))):
            raise ValueError("JAX templates require uniform over-sampling with consecutive-block binning")
        macro = np.asarray(scene.tracer.deflections_yx_2d_from(grid=scene.grid.over_sampled), dtype=float)
        traced = coordinates - macro
        from ...scene.image_profile import ImageLightProfile
        source_profiles = [profile for key, group in scene.light_groups.items() if group.plane == "source"
                           for profile in scene.light_profiles[key]]
        points = verification_points(traced, [profile for profile in source_profiles
                                             if isinstance(profile, ImageLightProfile)])
        self._light_evaluators = {
            key: tuple(build_light_evaluator(profile, points) for profile in scene.light_profiles[key])
            for key, group in scene.light_groups.items() if group.plane == "source"}
        self._model_slots = self._kernel_slots(context.model_kernels)
        self._truth_slots = None if context.truth_kernels is None else self._kernel_slots(context.truth_kernels)
        self._grid = radial_grid(coordinates, context.lens_centre_yx, context.domain_radius_arcsec,
                                 samples=MISMATCHED_SAMPLES if context.mismatched else MATCHED_SAMPLES)
        self._radial_tables: OrderedDict[float, Any] = OrderedDict()
        self._coords, self._alpha_macro_fit = jnp.asarray(coordinates), jnp.asarray(macro)
        self._log_radii = jnp.asarray(self._grid.log_radii)
        self._mu0_flat = jnp.asarray(np.asarray(context.mean_model_adu, dtype=float).reshape(-1))
        self._mask_flat_idx = jnp.asarray(np.flatnonzero(context.data_space.mask.reshape(-1)))
        self._scale_source = float(context.exposure.exposure_time_s)
        self._gain = float(context.exposure.detector.gain_e_per_adu)
        self._model_constant = self._constant(context.model_constant_adu)
        self._truth_constant = self._constant(context.truth_constant_adu) if context.mismatched else None
        self._signals = jax.jit(jax.vmap(self._signal_for_position, in_axes=(0, None)))
        self._reductions = None
        self._bias_whitened = None
        if context.data_space.whitener.mode == "diagonal":
            self._sigma_masked = jnp.asarray(context.data_space.whitener.sigma)
            self._nuisance_whitened = jnp.asarray(context.workspace.nuisance_whitened)
            self._bias_whitened = None if context.bias_whitened is None else jnp.asarray(context.bias_whitened)
            self._reductions = jax.jit(jax.vmap(self._reduction_for_position, in_axes=(0, None)))

    @staticmethod
    def _constant(value):
        return float(value) if np.ndim(value) == 0 else jnp.asarray(np.asarray(value, dtype=float).reshape(-1))

    def _kernel_slots(self, binding):
        slots = []
        for index, kernel in enumerate(binding.kernels):
            groups = tuple(key for key in self._light_evaluators if binding.group_index[key] == index)
            if not groups:
                continue
            shape = convolution_fft_shape(self._shape_native, kernel.shape)
            slots.append((jnp.asarray(np.fft.rfft2(kernel.kernel, s=shape)), kernel.shape[0] // 2,
                          kernel.shape[1] // 2, shape, groups))
        return tuple(slots)

    def _images(self, alpha_sub):
        traced = self._coords - self._alpha_macro_fit - alpha_sub
        images = {}
        n_pix = self._shape_native[0] * self._shape_native[1]
        for group, evaluators in self._light_evaluators.items():
            brightness = jnp.zeros(traced.shape[0])
            for evaluator in evaluators:
                brightness = brightness + evaluator(traced)
            image = brightness.reshape(n_pix, self._sub_per_pix).mean(axis=1)
            images[group] = image.reshape(self._shape_native)
        return images

    def _convolution(self, images, slots):
        terms = []
        for kernel_fft, crop_y, crop_x, shape, groups in slots:
            image = images[groups[0]]
            for group in groups[1:]:
                image = image + images[group]
            image_fft = jnp.fft.rfft2(image, s=shape)
            convolved = jnp.fft.irfft2(image_fft * kernel_fft, s=shape)
            terms.append(convolved[crop_y:crop_y + self._shape_native[0], crop_x:crop_x + self._shape_native[1]])
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total

    def _signal_for_position(self, position_yx, alpha_radial):
        delta = self._coords - position_yx[None, :]
        radius = jnp.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        radius_safe = jnp.clip(radius, jnp.exp(self._log_radii[0]), None)
        log_radius = jnp.log(radius_safe)
        alpha_r = interpolate_log_grid(log_radius, self._log_radii, alpha_radial, self._grid.affine)
        alpha_sub = alpha_r[:, None] * delta / radius_safe[:, None]
        images = self._images(alpha_sub)
        model = self._convolution(images, self._model_slots)
        mu_model = model.reshape(-1) * self._scale_source / self._gain + self._model_constant
        signal = (mu_model - self._mu0_flat)[self._mask_flat_idx]
        if self._truth_slots is None:
            return signal
        truth = self._convolution(images, self._truth_slots)
        mu_truth = truth.reshape(-1) * self._scale_source / self._gain + self._truth_constant
        residual = (mu_truth - self._mu0_flat)[self._mask_flat_idx]
        return jnp.stack((signal, residual), axis=0)

    def _reduction_for_position(self, position_yx, alpha_radial):
        signal = self._signal_for_position(position_yx, alpha_radial)
        if self._truth_slots is None:
            model_whitened = signal / self._sigma_masked
            data_whitened = None
        else:
            model_whitened = signal[0] / self._sigma_masked
            data_whitened = signal[1] / self._sigma_masked
        reductions = {"raw": jnp.sum(model_whitened * model_whitened),
                      "cross": model_whitened @ self._nuisance_whitened,
                      "finite": jnp.all(jnp.isfinite(model_whitened))}
        if data_whitened is not None:
            reductions["signal_data_inner"] = jnp.sum(model_whitened * data_whitened)
            reductions["data_cross"] = data_whitened @ self._nuisance_whitened
            reductions["data_finite"] = jnp.all(jnp.isfinite(data_whitened))
        if self._bias_whitened is not None:
            reductions["signal_bias_inner"] = model_whitened @ self._bias_whitened
        return reductions

    def _table(self, mass):
        table = self._radial_tables.pop(float(mass), None)
        if table is None:
            profile = self.context.hypothesis(mass, (0.0, 0.0)).autolens_profile()
            table = jnp.asarray(radial_deflection(profile, self._grid.radii))
        self._radial_tables[float(mass)] = table
        while len(self._radial_tables) > 8:
            self._radial_tables.popitem(last=False)
        return table

    def evaluate(self, positions_yx: np.ndarray, masses_msun: Sequence[float]) -> list[SignalBankResult]:
        if self._closed:
            raise RuntimeError("JAX engine is closed")
        positions = checked_positions(positions_yx)
        check_coverage(self._grid, positions, self.context.lens_centre_yx)
        masses = tuple(masses_msun)
        if not masses:
            raise ValueError("masses_msun must be non-empty")
        for mass in masses:
            self.context.hypothesis(mass, tuple(positions[0]))
        batches = tuple(jnp.asarray(positions[start:start + self.batch_size])
                        for start in range(0, len(positions), self.batch_size))
        progress = None
        if self.progress:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(masses) * len(positions), desc="Fisher templates")
        results = []
        try:
            for mass in masses:
                table, bank = self._table(mass), BankAccumulator(self.context)
                if self._reductions is not None:
                    pending = None
                    for batch in batches:
                        current = self._reductions(batch, table)
                        if pending is not None:
                            bank.add_reductions(self._materialize(pending))
                            if progress is not None:
                                progress.update(pending["raw"].shape[0])
                        pending = current
                    if pending is not None:
                        bank.add_reductions(self._materialize(pending))
                        if progress is not None:
                            progress.update(pending["raw"].shape[0])
                else:
                    for batch in batches:
                        for signal in np.asarray(self._signals(batch, table)):
                            bank.add_signal(signal)
                        if progress is not None:
                            progress.update(len(batch))
                results.append(bank.finish())
        finally:
            if progress is not None:
                progress.close()
        return results

    @staticmethod
    def _materialize(reductions):
        return BankReductions(**{name: np.asarray(values, dtype=bool if name in ("finite", "data_finite") else float)
                                 for name, values in reductions.items()})

    def describe(self) -> Mapping[str, Any]:
        return {"kind": self.kind, "batch_size": self.batch_size, "radial_samples": len(self._grid.radii),
                "device": str(jax.devices()[0]), "projection": self.context.data_space.whitener.mode}

    def close(self) -> None:
        if self._closed:
            return
        for function in (self._signals, self._reductions):
            if function is not None:
                function.clear_cache()
        self._signals = self._reductions = None
        self._radial_tables.clear()
        self._light_evaluators.clear()
        self._model_slots = self._truth_slots = ()
        for name in ("_coords", "_alpha_macro_fit", "_log_radii", "_mu0_flat", "_mask_flat_idx",
                     "_model_constant", "_truth_constant", "_sigma_masked", "_nuisance_whitened",
                     "_bias_whitened"):
            setattr(self, name, None)
        self._closed = True
