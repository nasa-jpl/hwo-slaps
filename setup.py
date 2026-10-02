"""Package the reusable engine; study archives remain source-only."""
from setuptools import find_packages, setup

setup(
    name="hwoslaps",
    version="0.0.1",
    description="Configurable strong-lensing simulations and sensitivity forecasts",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=["numpy", "scipy", "PyYAML"],
    entry_points={"console_scripts": ["hwoslaps=hwoslaps.cli:main"]},
)
