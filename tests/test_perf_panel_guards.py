"""Pure contract tests for the performance-panel orchestration guards."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


panel_driver = _load_module(
    "perf_panel_driver",
    PROJECT_ROOT / "scratch/panel/panel_driver.py",
)
panel_b6 = _load_module(
    "perf_panel_b6",
    PROJECT_ROOT / "scratch/panel/panel_b6.py",
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"logm": 8.5001},
        {"exposure_s": 2000.1},
        {"x_sub": 0.10000001},
        {"offset_dex": 0.0},
    ],
)
def test_extension_job_ids_preserve_float_and_none_identity(kwargs):
    baseline = panel_driver.job("X1", offset_dex=None)
    changed = panel_driver.job("X1", offset_dex=None)
    changed.update(kwargs)
    changed["id"] = panel_driver._signature(changed)

    if kwargs == {"offset_dex": 0.0}:
        assert baseline["id"] != changed["id"]
    else:
        other = panel_driver.job("X1", **kwargs)
        assert baseline["id"] != other["id"]


def test_identical_extension_job_configs_have_identical_ids():
    left = panel_driver.job("X1", logm=8.5001, x_sub=0.10000001)
    right = panel_driver.job("X1", logm=8.5001, x_sub=0.10000001)
    assert left["id"] == right["id"]


def test_exact_wave_keeps_v1_job_names():
    assert all("-" not in record["id"] for record in panel_driver.exact_jobs())
    assert all("-" in record["id"] for record in panel_driver.extension_jobs())


def test_default_half_width_leaves_job_identity_untouched():
    baseline = panel_driver.job("X1")
    explicit_none = panel_driver.job("X1", half_width=None)
    assert "half_width" not in baseline
    assert baseline["id"] == explicit_none["id"]
    assert "hw" not in baseline["id"]

    expanded = panel_driver.job("X1", half_width=2.5)
    assert expanded["half_width"] == 2.5
    assert "hw2.5" in expanded["id"]
    assert expanded["id"] != baseline["id"]


def test_expanded_job_preserves_physics_and_overrides_extent():
    original = panel_driver.job(
        "C", scene="cosmos", ref="RA", logm=9.0, _id_version=1
    )
    expanded = panel_driver._expanded_job(
        "GBX", original, panel_driver.GATE_HALF_WIDTH
    )
    assert expanded["family"] == "GBX"
    assert expanded["half_width"] == panel_driver.GATE_HALF_WIDTH
    for field in panel_driver._JOB_FIELDS:
        assert expanded[field] == original[field]


def test_common10_alias_expands_only_in_the_config_layer():
    record = panel_driver.job("GC", nuisance_subset="common10")
    assert record["nuisance_subset"] == "common10"
    assert "common10" in record["id"]
    assert len(panel_driver.COMMON_NUISANCE_10) == 10
    assert "observation.background_offset_adu" in (
        panel_driver.COMMON_NUISANCE_10
    )
    assert not any(
        "ell_comp" in name
        for name in panel_driver.COMMON_NUISANCE_10
        if name.startswith("source.")
    )


def test_memo_serializer_is_type_aware_and_rejects_unknown_values():
    assert panel_driver._canonical_json([1, 2]) != panel_driver._canonical_json(
        (1, 2)
    )
    with pytest.raises(TypeError):
        panel_driver._canonical_json(object())


def test_debug_memo_digest_detects_mutation(monkeypatch):
    class Product:
        def __init__(self):
            import numpy as np

            self.values = np.asarray([1.0, 2.0])

    product = Product()
    panel_driver._LENSING_MEMO.clear()
    panel_driver._LENSING_MEMO_DIGESTS.clear()
    panel_driver._LENSING_MEMO["key"] = product
    panel_driver._LENSING_MEMO_DIGESTS["key"] = panel_driver._memo_digest(product)
    monkeypatch.setenv("HWOSLAPS_DEBUG_LENSING_MEMO", "1")
    panel_driver._assert_lensing_memo_unchanged()
    product.values[0] = 3.0
    with pytest.raises(RuntimeError, match="memo value was mutated"):
        panel_driver._assert_lensing_memo_unchanged()
    panel_driver._LENSING_MEMO.clear()
    panel_driver._LENSING_MEMO_DIGESTS.clear()


def test_science_comparison_ignores_source_revision_metadata(tmp_path):
    import numpy as np

    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    payload = {
        "q_asimov_2d": np.asarray([[1.0, 2.0]]),
        "config_hash": np.asarray("abc"),
        "git_hash": np.asarray("a" * 40),
    }
    np.savez(left, **payload)
    payload["git_hash"] = np.asarray("b" * 40)
    np.savez(right, **payload)
    assert panel_driver.compare_science_npz(left, right, "job", "job")
    with pytest.raises(AssertionError, match="job IDs"):
        panel_driver.compare_science_npz(left, right, "job-a", "job-b")


def test_wave_manifest_is_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_driver, "BASE", tmp_path)
    jobs = [panel_driver.job("X1", logm=8.5)]

    path = panel_driver._write_wave_manifest("extension", jobs)
    original = json.loads(path.read_text(encoding="utf-8"))
    assert original["job_ids"] == [jobs[0]["id"]]

    with pytest.raises(RuntimeError, match="immutable"):
        panel_driver._write_wave_manifest(
            "extension", [panel_driver.job("X1", logm=8.6)]
        )


def test_panel_rejects_empty_gpu_list_and_zero_workers(monkeypatch):
    with pytest.raises(ValueError, match="gpus"):
        panel_driver.master("extension", (), 1)
    with pytest.raises(ValueError, match="n_workers"):
        panel_driver.master("extension", (0,), 0)


def test_row_without_npz_is_not_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_driver, "ROWS", tmp_path / "rows")
    monkeypatch.setattr(panel_driver, "OUTPUTS", tmp_path / "outputs")
    record = panel_driver.job("X1")
    row_path = panel_driver.ROWS / f"{record['id']}.json"
    row_path.parent.mkdir()
    row_path.write_text(
        json.dumps({"id": record["id"], "config_hash": "abc"}),
        encoding="utf-8",
    )
    assert not panel_driver._job_artifacts_complete(record)


def test_harvest_ignores_rows_from_other_waves(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_driver, "ROWS", tmp_path / "rows")
    monkeypatch.setattr(panel_driver, "OUTPUTS", tmp_path / "outputs")
    record = panel_driver.job("X1")
    alien = panel_driver.job("X1", logm=8.6)
    panel_driver._write_wave_manifest("extension", [record])
    for item in (record, alien):
        row_path = panel_driver.ROWS / f"{item['id']}.json"
        npz_path = panel_driver.OUTPUTS / item["id"] / "modeling" / "fisher_grid_map.npz"
        row_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text(
            json.dumps({"id": item["id"], "config_hash": "abc"}),
            encoding="utf-8",
        )
        import numpy as np

        np.savez(npz_path, config_hash=np.asarray("abc"))

    assert panel_driver._harvest_wave_rows("extension", [record]) == [record["id"]]


def test_b6_failed_result_is_not_resumable():
    cell = panel_b6.CELLS[0]
    failed = {"cell": cell, "status": "failed", "cases": []}
    assert not panel_b6._b6_result_is_success(failed, cell)


def test_b6_requires_both_roles_for_every_seed():
    cell = panel_b6.CELLS[0]
    record = {
        "cell": cell,
        "status": "success",
        "cases": [
            {
                "sampler_seed": seed,
                "smooth": {"status": "success"},
                "subhalo": {"status": "success"},
            }
            for seed in panel_b6.SAMPLER_SEEDS
        ],
    }
    assert panel_b6._b6_result_is_success(record, cell)
    record["cases"][0]["subhalo"]["status"] = "failed"
    assert not panel_b6._b6_result_is_success(record, cell)
