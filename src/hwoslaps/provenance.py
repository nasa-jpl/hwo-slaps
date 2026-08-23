"""Run provenance capture for reproducible study outputs.

This module records the information needed to rerun any produced result:
the command line, git hash, Python version, key package versions, and a
stable hash of the exact configuration used.
"""

import hashlib
import importlib.metadata
import json
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
    'nautilus-sampler',
    'scikit-learn',
    'threadpoolctl',
    'jax',
    'jaxlib',
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


def revision_digest(revision):
    """Return one scalar identifying a source-revision record.

    Parameters
    ----------
    revision : `dict`
        Record from `revision_provenance`.

    Returns
    -------
    digest : `str`
        Full SHA-256 of the canonical JSON rendering, so a single string
        identifies the commit, the dirty flag and the working-tree diff
        a result was produced at.
    """
    rendered = json.dumps(revision, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(rendered.encode('utf-8')).hexdigest()


def _git_hash(repo_dir=None):
    """Return the full git hash of ``repo_dir``, or `None` outside a repo.

    Parameters
    ----------
    repo_dir : `str` or `pathlib.Path`, optional
        Repository directory. Defaults to the current working directory.

    Returns
    -------
    git_hash : `str` or `None`
        Full commit hash, or `None` when git or the repository is
        unavailable.
    """
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=None if repo_dir is None else str(repo_dir),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_state(repo_dir=None):
    """Return commit and tracked-tree state for ``repo_dir``.

    Parameters
    ----------
    repo_dir : `str` or `pathlib.Path`, optional
        Repository directory. Defaults to the current working directory.

    Returns
    -------
    state : `tuple`
        Full commit hash, dirty flag, sorted dirty paths, and SHA-256
        digests of the tracked diff and complete source worktree state.
    """
    git_hash = _git_hash(repo_dir)
    if git_hash is None:
        return None, None, None, None, None
    try:
        repo_root = Path(
            subprocess.check_output(
                ['git', 'rev-parse', '--show-toplevel'],
                cwd=None if repo_dir is None else str(repo_dir),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        tracked_status = subprocess.check_output(
            [
                'git',
                'status',
                '--porcelain',
                '--no-renames',
                '--untracked-files=no',
            ],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        all_status = subprocess.check_output(
            [
                'git',
                'status',
                '--porcelain',
                '--no-renames',
                '--untracked-files=all',
            ],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        untracked_source = subprocess.check_output(
            [
                'git',
                'ls-files',
                '--others',
                '--exclude-standard',
                '--',
                'src',
            ],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        dirty = bool(all_status)
        dirty_paths = sorted(
            line[3:]
            for line in tracked_status.splitlines()
            if line and len(line) >= 4
        )
        dirty_paths = sorted(set(dirty_paths).union(untracked_source))
        diff_sha256 = None
        worktree_diff_sha256 = None
        if dirty:
            diff_output = subprocess.check_output(
                ['git', 'diff', 'HEAD'],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
            )
            diff_sha256 = hashlib.sha256(diff_output).hexdigest()
            worktree_digest = hashlib.sha256(diff_output)
            for relative in sorted(untracked_source):
                path = repo_root / relative
                if path.is_file():
                    worktree_digest.update(b'\0')
                    worktree_digest.update(relative.encode('utf-8'))
                    worktree_digest.update(b'\0')
                    worktree_digest.update(path.read_bytes())
            worktree_diff_sha256 = worktree_digest.hexdigest()
        return (
            git_hash,
            dirty,
            dirty_paths,
            diff_sha256,
            worktree_diff_sha256,
        )
    except Exception:
        return None, None, None, None, None


def _source_image_asset_provenance(config):
    """Return persistent identity metadata for a configured Image source.

    Parameters
    ----------
    config : `dict` or `None`
        Full pipeline configuration.

    Returns
    -------
    asset_provenance : `dict` or `None`
        Resolved asset path, content hash, pixel scale, and image shape for
        an Image source; `None` for other source light types.

    Raises
    ------
    ValueError
        Raised by the source-image loader when the configured asset is
        missing or invalid.
    """
    if not isinstance(config, dict):
        return None
    lensing = config.get('lensing')
    source_galaxy = lensing.get('source_galaxy') if isinstance(lensing, dict) else None
    light = source_galaxy.get('light') if isinstance(source_galaxy, dict) else None
    if not isinstance(light, dict) or light.get('type') != 'Image':
        return None

    asset_path = Path(light['asset_path']).expanduser().resolve()
    from .lensing.image_source import load_source_image_asset

    asset = load_source_image_asset(asset_path)
    return {
        'asset_path': str(asset_path),
        'sha256_16': asset.sha256_16,
        'pixel_scale_arcsec': float(asset.pixel_scale_arcsec),
        'shape': [int(dimension) for dimension in asset.sb.shape],
    }


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
        Record with the command line, config hash, git state, optional
        source-image identity, Python version, and the versions of the
        packages in ``_PROVENANCE_PACKAGES``.
    """
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parent
    versions = {name: _package_version(name) for name in _PROVENANCE_PACKAGES}
    (
        git_hash,
        git_dirty,
        git_dirty_paths,
        git_diff_sha256,
        worktree_diff_sha256,
    ) = _git_state(repo_dir)
    provenance = {
        'command': None if command is None else list(command),
        'config_hash': None if config is None else config_hash(config),
        'git_hash': git_hash,
        'git_dirty': git_dirty,
        'git_dirty_paths': git_dirty_paths,
        'git_diff_sha256': git_diff_sha256,
        'worktree_diff_sha256': worktree_diff_sha256,
        'python': platform.python_version(),
        'package_versions': versions,
    }
    source_image_asset = _source_image_asset_provenance(config)
    if source_image_asset is not None:
        provenance['source_image_asset'] = source_image_asset
    return provenance


def _git_toplevel(repo_dir):
    """Return the git working-tree root for ``repo_dir``, or `None`."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=str(repo_dir),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def revision_provenance(repo_dir=None):
    """Build a source-revision record for one result object.

    Parameters
    ----------
    repo_dir : `str` or `pathlib.Path`, optional
        Repository directory whose git state is recorded. When omitted,
        the directory containing this package is used, and the record is
        returned only if this module actually lives inside the discovered
        working tree; an installed package running under an unrelated
        checkout records all-`None` fields instead of the wrong
        repository.

    Returns
    -------
    revision : `dict`
        Git commit hash, dirty flag, sorted dirty paths, and SHA-256
        digests of the tracked diff and complete source worktree state.
        All values are `None` outside a usable git repository.
    """
    null_record = {
        'git_hash': None,
        'git_dirty': None,
        'git_dirty_paths': None,
        'git_diff_sha256': None,
        'worktree_diff_sha256': None,
    }
    if repo_dir is None:
        module_path = Path(__file__).resolve()
        repo_dir = module_path.parent
        toplevel = _git_toplevel(repo_dir)
        if toplevel is None:
            return null_record
        try:
            module_relpath = module_path.relative_to(
                Path(toplevel).resolve()
            )
        except ValueError:
            return null_record
        try:
            tracked = subprocess.call(
                ['git', 'ls-files', '--error-unmatch', str(module_relpath)],
                cwd=toplevel,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) == 0
        except Exception:
            return null_record
        if not tracked:
            return null_record
    (
        git_hash,
        git_dirty,
        git_dirty_paths,
        git_diff_sha256,
        worktree_diff_sha256,
    ) = _git_state(repo_dir)
    return {
        'git_hash': git_hash,
        'git_dirty': git_dirty,
        'git_dirty_paths': git_dirty_paths,
        'git_diff_sha256': git_diff_sha256,
        'worktree_diff_sha256': worktree_diff_sha256,
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
