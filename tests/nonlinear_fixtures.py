"""Small generic scene and synthetic-image inputs for nonlinear contracts."""
from pathlib import Path
import pytest
from test_nonlinear_freed_mode import _config
from test_image_source import _write_asset


@pytest.fixture
def synthetic_source_asset(tmp_path):
    return _write_asset(tmp_path / "source.npz", pixel_scale=0.02)


def scene_config(source_kind="analytic", asset_path=None):
    config = _config()
    config["global_seed"] = 11
    config["lensing"]["grid"] = {"shape": [31, 31], "pixel_scale": 0.04, "over_sample_size": 4}
    config["lensing"]["subhalo"]["position"] = {"type": "direct", "centre": [0.8, 0.05]}
    if source_kind == "image":
        if asset_path is None:
            raise ValueError("synthetic image source requires its prepared asset")
        config["lensing"]["source_galaxy"]["light"] = {
            "type": "Image", "asset_path": str(Path(asset_path)), "centre": [-0.03, 0.08],
            "rotation_deg": 0.0, "total_flux": 0.5, "flux_scale": 1.0, "size_scale": 1.0,
        }
    return config
