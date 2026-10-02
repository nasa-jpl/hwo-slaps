"""Persistence of forecast outputs and their execution metadata."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import yaml

from .config import run_directory
from .provenance import config_hash, revision_provenance


def write_fisher_grid_map(detection_data: Any, config: Mapping[str, Any]) -> Path | None:
    """Save a grid forecast and bind it to an exact adjacent run snapshot.

    No file is written for a result without a grid map. A snapshot, when
    present, must contain the exact resolved configuration used for the run;
    historical relative snapshots must be migrated explicitly. Standalone
    outputs remain unbound to a campaign unless HWOSLAPS_CAMPAIGN_UUID is set.
    Numerical arrays and the existing NPZ format are unchanged.
    """
    if not detection_data.has_grid_map:
        return None
    from .modeling.utils_fisher import save_fisher_grid_map_npz

    grid_map = detection_data.grid_map
    output_dir = run_directory(config) / "modeling"
    snapshot_path = output_dir.parent / "config_used.yaml"
    grid_map.config_hash = None
    if snapshot_path.is_file():
        with snapshot_path.open("r", encoding="utf-8") as stream:
            snapshot_config = yaml.safe_load(stream)
        if not isinstance(snapshot_config, dict):
            raise ValueError(f"Adjacent config snapshot {snapshot_path} must contain a mapping")
        snapshot_hash = config_hash(snapshot_config)
        if snapshot_hash != config_hash(config):
            raise ValueError(
                f"Adjacent config snapshot {snapshot_path} does not describe this run; "
                "refusing to bind the grid map to it."
            )
        grid_map.config_hash = snapshot_hash

    revision = revision_provenance(Path(__file__).resolve().parent)
    grid_map.git_hash = revision["git_hash"]
    grid_map.git_dirty = revision["git_dirty"]
    grid_map.worktree_diff_sha256 = revision["worktree_diff_sha256"]
    grid_map.runtime_provenance = {
        **(grid_map.runtime_provenance or {}),
        "source_git_dirty": revision["git_dirty"],
        "source_worktree_diff_sha256": revision["worktree_diff_sha256"],
    }
    grid_map.campaign_uuid = os.environ.get("HWOSLAPS_CAMPAIGN_UUID")
    output_dir.mkdir(parents=True, exist_ok=True)
    return save_fisher_grid_map_npz(grid_map, output_dir / "fisher_grid_map.npz")


__all__ = ["write_fisher_grid_map"]
