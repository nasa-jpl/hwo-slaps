"""Experimental CUDA admission guard; does not change the likelihood."""

import os


def require_cuda_execution():
    """Require one default GPU and a successful float64 device probe."""
    import jax
    import numpy as np

    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("Nonlinear CUDA execution requires float64")
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu" or len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"CUDA admission rejected: backend={backend}, devices={devices}")
    probe = jax.device_put(np.zeros(1, dtype=np.float64), devices[0])
    probe.block_until_ready()
    actual = tuple(probe.devices())
    if actual != (devices[0],):
        raise RuntimeError(f"CUDA probe is on unexpected devices: {actual}")
    return {
        "backend": backend,
        "device": str(devices[0]),
        "device_kind": devices[0].device_kind,
        "platform": devices[0].platform,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "x64": True,
        "probe_placed_on_requested_device": True,
    }
