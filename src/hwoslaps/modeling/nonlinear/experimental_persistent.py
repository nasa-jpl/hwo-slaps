"""Bounded, detached preparation reuse for an isolated per-lens worker.

Science functions and inputs are unchanged. Cache hits return deep copies;
the provenance configuration belongs to the current call, not the first hit.
This context is process-local and intended for a serial worker, not threads.
"""

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import time
import inspect

import numpy as np


def _key(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@contextmanager
def persistent_preparation(max_lensing=2, max_psf=2, max_grid_bytes=1024**3):
    """Reuse immutable preparation while retaining caller-owned outputs."""
    from hwoslaps.lensing import generator as lens_module
    from hwoslaps.psf import generator as psf_module
    from hwoslaps.modeling.nonlinear import autolens_runner
    from autoarray.operators.over_sampling import over_sample_util

    original_lens = lens_module.generate_lensing_system
    original_psf = psf_module.generate_psf_system
    original_close = autolens_runner._close_training_pools
    original_grid = over_sample_util.grid_2d_slim_over_sampled_via_mask_from
    grid_signature = inspect.signature(original_grid)
    lens_cache = OrderedDict()
    psf_cache = OrderedDict()
    grid_cache = OrderedDict()
    stats = {
        "lensing_hits": 0, "lensing_misses": 0,
        "psf_hits": 0, "psf_misses": 0,
        "copy_seconds": 0.0, "pool_close_deferred": 0,
        "max_lensing_entries": max_lensing, "max_psf_entries": max_psf,
        "grid_hits": 0, "grid_misses": 0, "grid_cache_bytes": 0,
        "grid_peak_cache_bytes": 0, "grid_oversized_bypasses": 0,
        "grid_key_seconds": 0.0, "grid_build_seconds": 0.0,
        "grid_copy_seconds": 0.0, "max_grid_bytes": max_grid_bytes,
    }

    def use(cache, identity, kind, original, config, full_config, limit):
        if identity in cache:
            stats[kind + "_hits"] += 1
            item = cache.pop(identity)
            cache[identity] = item
            start = time.perf_counter()
            result = deepcopy(item)
            result.config = deepcopy(full_config)
            stats["copy_seconds"] += time.perf_counter() - start
            return result
        stats[kind + "_misses"] += 1
        result = original(config, full_config=full_config)
        start = time.perf_counter()
        cache[identity] = deepcopy(result)
        stats["copy_seconds"] += time.perf_counter() - start
        while len(cache) > limit:
            cache.popitem(last=False)
        return result

    def lens(config, full_config):
        identity = _key({"lensing": config, "global_seed": full_config["global_seed"]})
        return use(lens_cache, identity, "lensing", original_lens,
                   config, full_config, max_lensing)

    def psf(config, full_config=None):
        # The generator reads the PSF block and detector scale for its science.
        # Its only output side effect is an optional high-resolution PSF file.
        if full_config is None or config.get("hres_psf", {}).get("save_highres_psf_npy", False):
            return original_psf(config, full_config=full_config)
        identity = _key({"psf": config, "pixel_scale": full_config["lensing"]["grid"]["pixel_scale"]})
        return use(psf_cache, identity, "psf", original_psf,
                   config, full_config, max_psf)

    def defer_close():
        stats["pool_close_deferred"] += 1

    def grid(*args, **kwargs):
        before = time.perf_counter()
        bound = grid_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        digest = hashlib.sha256()
        # Bind every argument, including dtype/shape and signed float bytes.
        for name, value in bound.arguments.items():
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(name.encode())
            digest.update(array.dtype.str.encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        identity = digest.hexdigest()
        stats['grid_key_seconds'] += time.perf_counter() - before
        if identity in grid_cache:
            stats['grid_hits'] += 1
            result = grid_cache.pop(identity)
            grid_cache[identity] = result
            before = time.perf_counter()
            detached = result.copy(order='K')
            stats['grid_copy_seconds'] += time.perf_counter() - before
            return detached
        stats['grid_misses'] += 1
        before = time.perf_counter()
        result = original_grid(*args, **kwargs)
        stats['grid_build_seconds'] += time.perf_counter() - before
        if result.nbytes > max_grid_bytes:
            stats['grid_oversized_bypasses'] += 1
            return result
        while grid_cache and (len(grid_cache) >= 4 or
                              stats['grid_cache_bytes'] + result.nbytes > max_grid_bytes):
            _, removed = grid_cache.popitem(last=False)
            stats['grid_cache_bytes'] -= removed.nbytes
        saved = result.copy(order='K')
        saved.flags.writeable = False
        grid_cache[identity] = saved
        stats['grid_cache_bytes'] += saved.nbytes
        stats['grid_peak_cache_bytes'] = max(stats['grid_peak_cache_bytes'],
                                            stats['grid_cache_bytes'])
        return result

    lens_module.generate_lensing_system = lens
    psf_module.generate_psf_system = psf
    autolens_runner._close_training_pools = defer_close
    over_sample_util.grid_2d_slim_over_sampled_via_mask_from = grid
    try:
        yield stats
    finally:
        lens_module.generate_lensing_system = original_lens
        psf_module.generate_psf_system = original_psf
        autolens_runner._close_training_pools = original_close
        over_sample_util.grid_2d_slim_over_sampled_via_mask_from = original_grid
        original_close()
        lens_cache.clear()
        psf_cache.clear()
        grid_cache.clear()
        stats['grid_cache_bytes'] = 0
