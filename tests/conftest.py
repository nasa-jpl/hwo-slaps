"""Collection policy and the fixtures shared across test lanes.

Tests that need GPUs run only where cards are assigned: ``xtx_gpu`` items are
skipped without a JAX GPU and ``xtx_multi_gpu`` items without two. The gpu and
multi launchers pass ``--require-gpu``, so a missing card fails there instead.
No global backend configuration or JIT setting is changed here.
"""

import json

import numpy as np
import pytest

def _visible_gpus():
    try:
        import jax
        return len(jax.devices("gpu"))
    except (ImportError, RuntimeError):
        return 0


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests the visible cards cannot run.

    Runs after marker deselection, so a lane that deselects GPU tests never imports JAX.
    """
    needs = []
    for item in items:
        if item.get_closest_marker("xtx_multi_gpu"):
            needs.append((item, 2))
        elif item.get_closest_marker("xtx_gpu"):
            needs.append((item, 1))
    if not needs:
        return
    available = _visible_gpus()
    for item, count in needs:
        if available < count:
            item.add_marker(pytest.mark.skip(reason=f"requires {count} JAX GPU(s), found {available}"))


@pytest.fixture
def tiny_gaussian_kernel():
    """A 7 x 7 Gaussian detector kernel, sigma 1 pixel, unit sum, float64."""
    y, x = np.mgrid[-3:4, -3:4].astype(np.float64)
    kernel = np.exp(-(x**2 + y**2) / 2.0)
    return kernel / kernel.sum()


@pytest.fixture
def image_asset(tmp_path):
    """Path of an image asset prepared from three elliptical Gaussians on 48 x 48 pixels at 0.01 arcsec.

    The image carries seeded noise at 1e-4 of the peak, so the preparation finds a source
    footprint inside the frame; the file has the version-1 asset layout.
    """
    from hwoslaps.identity import json_ready
    from hwoslaps.scene.image_source import prepare_image_asset

    rows, cols = np.indices((48, 48), dtype=float)
    image = np.random.default_rng(20261005).normal(0.0, 1.0e-4, (48, 48))
    for amplitude, centre_y, centre_x, sigma_y, sigma_x in ((1.0, 23.0, 24.5, 3.0, 4.5), (0.6, 27.0, 20.0, 2.0, 1.5),
                                                           (0.4, 19.5, 28.0, 1.5, 2.5)):
        image += amplitude * np.exp(-0.5 * (((rows - centre_y) / sigma_y) ** 2 + ((cols - centre_x) / sigma_x) ** 2))
    asset = prepare_image_asset(image, pixel_scale_arcsec=0.01, provenance={"fixture": "image_asset"})
    path = tmp_path / "image_asset.npz"
    np.savez(path, sb=asset.sb, pixel_scale_arcsec=np.asarray(asset.pixel_scale_arcsec, dtype=np.float64),
             metadata_json=np.asarray(json.dumps(json_ready(asset.metadata), sort_keys=True)))
    return path


@pytest.fixture
def minimal_mapping(tmp_path, tiny_gaussian_kernel):
    """Fresh final-schema inputs for a small kernel-PSF forecast."""
    path = tmp_path / "kernel.npy"
    np.save(path, tiny_gaussian_kernel)
    return {
        "seed": 11,
        "cosmology": {"name": "Planck15"},
        "scene": {
            "grid": {"shape": [41, 41], "pixel_scale_arcsec": 0.05, "over_sample_size": 2},
            "lens": {"redshift": 0.2, "mass": {
                "mass": {"type": "Isothermal", "centre": [0.0, 0.0], "einstein_radius": 0.8,
                         "ell_comps": [0.05, 0.0]}}},
            "source": {"redshift": 0.6, "light": {
                "light": {"type": "Exponential", "centre": [-0.03, 0.08], "ell_comps": [0.0, 0.0],
                          "intensity": 1.0, "effective_radius": 0.12}}},
            "subhalo": {"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.0}},
        },
        "psf": {"truth": {"kind": "kernel", "path": str(path), "pixel_scale_arcsec": 0.05}},
        "instrument": {"detector": {"gain_e_per_adu": 1.0, "read_noise_e": 0.2,
                                     "dark_current_e_per_s": 0.002}},
        "observation": {"exposure_time_s": 900.0, "sky": {"rate_e_per_s": 1.0}},
        "forecast": {"positions": {"kind": "explicit", "positions_yx": [[0.0, 0.8], [0.4, -0.6]]},
                     "mask": {"kind": "all_pixels"}},
    }


@pytest.fixture
def write_config(tmp_path):
    """Write a YAML mapping beside its assets and return its path."""
    from hwoslaps.config.loading import dump_yaml

    def write(mapping, name="config.yaml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_yaml(mapping), encoding="utf-8")
        return path
    return write
