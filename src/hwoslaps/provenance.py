"""Run provenance capture for reproducible study outputs.

This module records the information needed to rerun any produced result:
the command line, git hash, Python version, key package versions, and a
stable hash of the exact configuration used.
"""

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import yaml

_PROVENANCE_PACKAGES = (
    'numpy',
    'scipy',
    'matplotlib',
    'pyyaml',
    'autolens',
    'autofit',
    'hcipy',
    'hwoslaps',
)
"""Packages recorded by `capture_provenance` (`tuple` of `str`)."""

_PROVENANCE_MODULE_NAMES = {
    'pyyaml': 'yaml',
}
"""Import names for packages whose module name differs from the
distribution name (`dict` of `str` to `str`)."""


def _package_version(name):
    """Return the version of an installed package.

    Parameters
    ----------
    name : `str`
        Distribution name of the package.

    Returns
    -------
    version : `str` or `None`
        The imported module's ``__version__`` when available, since
        distribution metadata can be stale for source installs; otherwise
        the distribution-metadata version, or `None` when the package is
        not installed.
    """
    try:
        module = importlib.import_module(_PROVENANCE_MODULE_NAMES.get(name, name))
        version = getattr(module, '__version__', None)
        if version is not None:
            return str(version)
    except Exception:
        pass
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def config_hash(config):
    """Return a stable short hash of a configuration dictionary.

    Parameters
    ----------
    config : `dict`
        Configuration to hash.

    Returns
    -------
    digest : `str`
        First 16 hex characters of the SHA-256 of the key-sorted YAML
        rendering. The rendering convention is part of the provenance
        contract; changing it invalidates recorded hashes.
    """
    rendered = yaml.safe_dump(config, sort_keys=True).encode('utf-8')
    return hashlib.sha256(rendered).hexdigest()[:16]


def _git_hash(repo_dir=None):
    """Return the short git hash of ``repo_dir``, or `None` outside a repo.

    Parameters
    ----------
    repo_dir : `str` or `pathlib.Path`, optional
        Repository directory. Defaults to the current working directory.

    Returns
    -------
    git_hash : `str` or `None`
        Short commit hash, or `None` when git or the repository is
        unavailable.
    """
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=None if repo_dir is None else str(repo_dir),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def capture_provenance(config=None, command=None, repo_dir=None):
    """Build a provenance record for one pipeline invocation.

    Parameters
    ----------
    config : `dict`, optional
        Full pipeline configuration; recorded as its `config_hash`.
    command : `list` of `str`, optional
        Command-line argument vector to record, e.g. ``sys.argv``.
    repo_dir : `str` or `pathlib.Path`, optional
        Repository directory whose git hash is recorded. Defaults to the
        directory containing this package, so pipeline runs started from
        other working directories still record the code version.

    Returns
    -------
    provenance : `dict`
        Record with the command line, config hash, git hash, Python
        version, and the versions of the packages in
        ``_PROVENANCE_PACKAGES``.
    """
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parent
    versions = {name: _package_version(name) for name in _PROVENANCE_PACKAGES}
    return {
        'command': None if command is None else list(command),
        'config_hash': None if config is None else config_hash(config),
        'git_hash': _git_hash(repo_dir),
        'python': platform.python_version(),
        'package_versions': versions,
    }


def write_provenance(path, config=None, command=None, repo_dir=None):
    """Capture provenance and write it to a YAML file.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Output file path, e.g. ``<run_dir>/provenance.yaml``. Parent
        directories must already exist.
    config : `dict`, optional
        Full pipeline configuration; recorded as its `config_hash`.
    command : `list` of `str`, optional
        Command-line argument vector to record, e.g. ``sys.argv``.
    repo_dir : `str` or `pathlib.Path`, optional
        Repository directory whose git hash is recorded.

    Returns
    -------
    provenance : `dict`
        The written provenance record.
    """
    provenance = capture_provenance(config=config, command=command, repo_dir=repo_dir)
    with Path(path).open('w', encoding='utf-8') as handle:
        yaml.safe_dump(provenance, handle, sort_keys=False)
    return provenance
