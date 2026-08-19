"""Static checks for the reproducibility pins in the installer."""

from pathlib import Path


INSTALL = (Path(__file__).resolve().parents[1] / "install.sh").read_text(
    encoding="utf-8"
)


def test_installer_checks_out_validated_dependency_commits():
    assert "10bfea51ea95" in INSTALL
    assert "cc853b392463" in INSTALL


def test_installer_enforces_validated_runtime_versions():
    for requirement in (
        "autoarray==2026.5.14.2",
        "autofit==2026.5.14.2",
        "autogalaxy==2026.5.14.2",
        "autoconf==2026.5.14.2",
        "nautilus-sampler==1.0.5",
        "scikit-learn==1.8.0",
        "scipy==1.17.1",
        "threadpoolctl==3.6.0",
        "jax==0.4.38",
        "jaxlib==0.4.38",
        "jax-cuda12-plugin==0.4.38",
        "jax-cuda12-pjrt==0.4.38",
        "numpy==1.26.4",
    ):
        assert requirement in INSTALL
