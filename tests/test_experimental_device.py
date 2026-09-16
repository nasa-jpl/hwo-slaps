"""Admission guard contracts without a GPU allocation."""

import sys
from types import SimpleNamespace
import pytest
from hwoslaps.modeling.nonlinear.experimental_device import require_cuda_execution


def test_cpu_backend_rejected_before_probe(monkeypatch):
    fake = SimpleNamespace(
        config=SimpleNamespace(jax_enable_x64=True),
        default_backend=lambda: "cpu",
        devices=lambda: [SimpleNamespace(platform="cpu")],
    )
    monkeypatch.setitem(sys.modules, "jax", fake)
    with pytest.raises(RuntimeError, match="CUDA admission rejected"):
        require_cuda_execution()


def test_low_precision_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(config=SimpleNamespace(jax_enable_x64=False)))
    with pytest.raises(RuntimeError, match="float64"):
        require_cuda_execution()


def test_probe_confirms_actual_single_device(monkeypatch):
    class Device:
        platform = "gpu"
        device_kind = "test CUDA"

    device = Device()

    class Probe:
        def block_until_ready(self):
            return self

        def devices(self):
            return {device}

    fake = SimpleNamespace(
        config=SimpleNamespace(jax_enable_x64=True),
        default_backend=lambda: "gpu",
        devices=lambda: [device],
        device_put=lambda values, target: Probe(),
    )
    monkeypatch.setitem(sys.modules, "jax", fake)
    assert require_cuda_execution()["probe_placed_on_requested_device"]
