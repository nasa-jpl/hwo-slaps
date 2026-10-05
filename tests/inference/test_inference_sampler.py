"""Role searches: identity-named paths, settings read back, retained state, failures and a real fit."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.inference.sampler import effective_settings, inspect_retention, make_search, run_search
from hwoslaps.inference.settings import RefineSettings, SamplerSettings
from hwoslaps.inference.starts import select_starts

DATA = {"data_digest": "aa" * 32, "noise_digest": "bb" * 32}


def _search(light_model, settings, case_dir, *, seed=7, n_live=20):
    return make_search(model=light_model, role="smooth", n_live=n_live, settings=settings, seed=seed,
                       case_dir=case_dir, case_id="case", data_identity=DATA)


@pytest.mark.backend
def test_search_paths_differ_for_every_sampler_setting(light_model, tmp_path):
    """C6: runs differing in one setting AutoFit's Nautilus identifier omits never share an output path."""
    base = SamplerSettings()
    variants = {"base": (base, 7), "f_live": (SamplerSettings(f_live=0.02), 7),
                "n_like_max": (SamplerSettings(n_like_max=5000), 7),
                "number_of_cores": (SamplerSettings(number_of_cores=2), 7),
                "discard_exploration": (SamplerSettings(discard_exploration=True), 7),
                "n_eff": (SamplerSettings(n_eff=400.0), 7), "n_shell": (SamplerSettings(n_shell=2), 7),
                "jax_n_batch": (SamplerSettings(use_jax=True, jax_n_batch=50), 7),
                "use_jax": (SamplerSettings(use_jax=True), 7), "seed": (base, 8)}
    locations = {name: (search.paths.path_prefix, search.paths.name)
                 for name, (settings, seed) in variants.items()
                 for search in [_search(light_model, settings, tmp_path, seed=seed)]}
    assert len(set(locations.values())) == len(variants)
    assert all(Path(prefix) == tmp_path for prefix, _ in locations.values())


@pytest.mark.backend
def test_effective_settings_are_read_back_and_recorded(light_model, raw_imaging, tmp_path, monkeypatch):
    import autofit as af
    import autogalaxy as ag
    from hwoslaps.inference.backend import BackendSession, make_analysis

    requested = SamplerSettings(n_eff=200, n_shell=2, f_live=0.02, discard_exploration=True, n_like_max=900)
    assert effective_settings(_search(light_model, requested, tmp_path, n_live=30)) == {
        "n_live": 30, "n_eff": 200.0, "n_shell": 2, "f_live": 0.02, "discard_exploration": True, "n_like_max": 900,
        "number_of_cores": 1, "seed": 7, "n_batch": 100, "use_jax_vmap": True}
    defaults = effective_settings(_search(light_model, SamplerSettings(), tmp_path))
    assert {name: defaults[name] for name in ("n_eff", "n_shell", "f_live", "discard_exploration", "n_like_max")} \
        == {"n_eff": 500, "n_shell": 1, "f_live": 0.01, "discard_exploration": False, "n_like_max": float("inf")}

    class DroppingNautilus(af.Nautilus):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.f_live = 0.01

    monkeypatch.setattr(af, "Nautilus", DroppingNautilus)
    analysis = make_analysis(raw_imaging, cosmology=ag.cosmo.Planck15(), use_jax=False)
    with BackendSession() as session, pytest.raises(RuntimeError, match="f_live: requested 0.02, constructed 0.01"):
        run_search(fit_model=light_model, model=None, analysis=analysis, role="smooth", n_live=20,
                   settings=SamplerSettings(f_live=0.02), seed=7, case_dir=tmp_path, case_id="case",
                   data_identity=DATA, session=session)
    assert list(tmp_path.iterdir()) == []


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.mark.parametrize("layout", ["directory", "timer-only", "zip", "zip-empty-state", "foreign-result", "nothing"])
def test_retention_inventory_reads_directory_and_zip(layout, tmp_path):
    output = tmp_path / "smooth_0123456789abcdef" / "identifier"
    state, timer = b"sampler-state", b"0.1"
    if layout in ("directory", "foreign-result"):
        _write(output / "files" / "search_internal" / "search_internal.dill", state)
        _write(output / "files" / "search_internal" / ".time", timer)
    elif layout == "timer-only":
        _write(output / "files" / "search_internal" / ".time", timer)
    elif layout in ("zip", "zip-empty-state"):
        output.parent.mkdir(parents=True)
        with zipfile.ZipFile(f"{output}.zip", "w") as archive:
            archive.writestr("files/model.json", "{}")
            archive.writestr("files/search_internal/.time", timer)
            archive.writestr("files/search_internal/search_internal.dill", b"" if layout == "zip-empty-state" else state)
    result_path = tmp_path / "other_run" if layout == "foreign-result" else output
    inventory = inspect_retention(output, result_path=result_path)
    expected_route = {"directory": "directory", "timer-only": "directory", "foreign-result": "directory",
                      "zip": "zip", "zip-empty-state": "zip", "nothing": None}[layout]
    assert inventory.route == expected_route
    assert inventory.retained is (layout in ("directory", "zip"))
    assert inventory.bound_to_result_path is (layout != "foreign-result")
    assert inventory.missing_required == (() if layout in ("directory", "zip", "foreign-result")
                                          else ("search_internal.dill",))
    if layout in ("directory", "zip"):
        entry = inventory.files["search_internal.dill"]
        assert (entry["bytes"], entry["sha256"]) == (len(state), hashlib.sha256(state).hexdigest())
        assert entry["location"] == "files/search_internal/search_internal.dill"
        assert inventory.files[".time"]["bytes"] == len(timer)


@pytest.mark.backend
def test_failure_inside_search_fit_is_recorded_and_construction_errors_raise(light_model, raw_imaging, tmp_path):
    """D35: a sampler failure is a failed outcome with its settings; an error before the fit raises."""
    import autogalaxy as ag
    from hwoslaps.inference.backend import ANALYSIS_CLASS, BackendSession
    from hwoslaps.inference.fit_model import autofit_model

    class FailingAnalysis(ANALYSIS_CLASS):
        def log_likelihood_function(self, instance):
            raise RuntimeError("boom")

    analysis = FailingAnalysis(dataset=raw_imaging, cosmology=ag.cosmo.Planck15(), use_jax=False)
    settings = SamplerSettings(n_like_max=200)
    with BackendSession() as session:
        outcome = run_search(fit_model=light_model, model=autofit_model(light_model), analysis=analysis,
                             role="smooth", n_live=20, settings=settings, seed=7, case_dir=tmp_path, case_id="case",
                             data_identity=DATA, session=session)
        assert outcome.error == "RuntimeError: boom" and outcome.result is None
        record = outcome.record
        assert (record.log_likelihood_max, record.log_evidence, record.likelihood_calls) == (None, None, None)
        assert record.effective["n_like_max"] == 200 and record.requested == settings.to_mapping()
        assert record.output_path.startswith(record.name + "/") and record.seed == 7
        for case_dir, n_live, message in ((Path("relative"), 20, "case_dir must be absolute"),
                                          (tmp_path, 0, "n_live must be at least 1")):
            with pytest.raises(ValueError, match=message):
                run_search(fit_model=light_model, model=autofit_model(light_model), analysis=analysis,
                           role="smooth", n_live=n_live, settings=settings, seed=7, case_dir=case_dir,
                           case_id="case", data_identity=DATA, session=session)


@pytest.mark.backend
def test_successful_search_records_its_outputs_and_retained_state(light_model, raw_imaging, tmp_path):
    """The record holds the sampler maximum, evidence, calls and retained state; the saved files feed
    select_starts from the case directory."""
    import autogalaxy as ag
    from hwoslaps.inference.backend import BackendSession, make_analysis
    from hwoslaps.inference.fit_model import autofit_model

    analysis = make_analysis(raw_imaging, cosmology=ag.cosmo.Planck15(), use_jax=False)
    settings = SamplerSettings(n_like_max=400, retain_search_internal=True)
    with BackendSession() as session:
        outcome = run_search(fit_model=light_model, model=autofit_model(light_model), analysis=analysis,
                             role="smooth", n_live=20, settings=settings, seed=11, case_dir=tmp_path,
                             case_id="case", data_identity=DATA, session=session)
    record = outcome.record
    assert outcome.error is None and record.training_workers == 1
    files = tmp_path / record.output_path / "files"
    with (files / "samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    assert record.log_likelihood_max == pytest.approx(max(float(row["log_likelihood"]) for row in rows), rel=1e-12)
    assert record.likelihood_calls >= 400 and np.isfinite(record.log_evidence)
    assert record.retention.retained and record.retention.route == "directory"
    starts = select_starts(files, light_model.parameter_names, light_model.lower, light_model.upper,
                           RefineSettings(original_start_count=2))
    assert starts[0].origin["saved_log_likelihood"] == pytest.approx(record.log_likelihood_max, rel=1e-12)
