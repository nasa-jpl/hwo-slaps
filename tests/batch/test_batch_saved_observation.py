"""The controller and workers reuse a pinned observation across independent fits."""

import sys
from types import SimpleNamespace

import pytest

from hwoslaps import load_case, save_observation, simulate
from hwoslaps.batch import BatchConflict, open_batch, plan_batch, run_batch
from hwoslaps.batch.frozen import persist_plan, restored_plan
from hwoslaps.batch.jobs import follow_ups
from hwoslaps.batch.worker import PreparationCache, run_job
from hwoslaps.identity import array_digest, file_digest
from hwoslaps.scene.cosmology import Cosmology
from hwoslaps.scene.subhalo import configured_injection

pytestmark = pytest.mark.backend


@pytest.fixture
def saved_batch(tiny_batch_spec, tmp_path):
    truth = tiny_batch_spec().base
    candidate = configured_injection(truth.scene, Cosmology(truth.cosmology), seed=truth.seed)
    observation = simulate(truth, subhalo=candidate, noise_seed=19)
    path = save_observation(observation, tmp_path / "observation.npz")
    fit_scene = {"injection": None}
    spec = tiny_batch_spec(population=None, forecast=None, execution={"workers_per_device": 1},
                           arms=[{"name": "standard", "overrides": {"scene": fit_scene, "forecast": {
                               "nuisances": {"fixed": ["*"]}}}},
                                 {"name": "elliptical", "overrides": {
                                     "scene": {**fit_scene, "lens": {"mass": {"mass": {"ell_comps": [0.08, 0.02]}}}},
                                     "forecast": {"nuisances": {"fixed": ["*"]}}}}],
                           nonlinear={"saved": {"observation": path.name, "fit": {"mode": "fixed_template"},
                               "retry": {"require_retained_state": False, "stationarity_tolerance": None,
                                         "acceptance": {"smooth": ["sampler_only"], "subhalo": ["sampler_only"]}},
                               "trials": {"kind": "explicit", "explicit": [
                                   {"members": "all", "mass_msun": 9e7, "position_yx": [0.1, 0.25]}]}}})
    return spec, path, observation


def test_saved_observation_plan_worker_and_retry_bind_the_same_archive(saved_batch, tmp_path):
    spec, path, observation = saved_batch
    plan = plan_batch(spec)
    root = tmp_path / "workers"
    persist_plan(root, plan)
    restored = restored_plan(root, spec)
    assert [job.digest() for job in plan.jobs] == [job.digest() for job in restored.jobs]
    cache = PreparationCache(1, spec.execution.forecast)
    events = []
    cases = []
    for first_job in restored.jobs:
        job = first_job
        first = None
        for attempt in (0, 1):
            payload = job.payload(root, 0, spec.execution)
            result = run_job(payload, cache, None, connection=SimpleNamespace(send=events.append),
                             worker={}, revision="test")
            assert result["type"] == "complete", result
            assert result["prepared"] == 0
            artifact = result["marker"]["artifacts"]["case"]["path"]
            case = load_case(root / job.job_id / artifact)
            assert case.observation.data_digest == array_digest(observation.data_adu)
            if attempt == 0:
                first = case
                cases.append(case)
                job, = follow_ups(restored, job, root)
            else:
                assert case.observation == first.observation and case.hypothesis == first.hypothesis
                assert case.models == first.models and case.sampler_seed != first.sampler_seed
    assert cases[0].models["smooth"] != cases[1].models["smooth"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    failure = run_job(restored.jobs[0].payload(root, 1, spec.execution), cache, None,
                      connection=SimpleNamespace(send=events.append), worker={}, revision="test")
    assert failure["type"] == "failed"
    assert "saved observation bytes differ" in failure["failure"]["message"]
    with pytest.raises(BatchConflict, match="consumed input bytes"):
        restored_plan(root, spec)


@pytest.mark.skipif(sys.platform != "linux", reason="the batch controller requires Linux process support")
def test_saved_observation_batch_reuses_pixels_and_refuses_changed_archive(saved_batch, tmp_path):
    spec, path, observation = saved_batch
    plan = plan_batch(spec)
    assert plan.file_digests[str(path)] == file_digest(path)
    assert all(job.preparation_key is None for job in plan.jobs)
    root = tmp_path / "batch"
    report = run_batch(spec, root)
    assert report.counts["completed"] == 4 and report.counts["prepared"] == 0
    first, second = (case for _, case in open_batch(root).cases(family="saved"))
    assert first.observation.data_digest == second.observation.data_digest == array_digest(observation.data_adu)
    assert first.models["smooth"] != second.models["smooth"]
    for _, original, retry in open_batch(root).case_pairs(family="saved"):
        assert retry.observation == original.observation
        assert retry.hypothesis == original.hypothesis
        assert retry.models == original.models
        assert retry.sampler_seed != original.sampler_seed
    assert run_batch(spec, root, verify=True).counts["skipped"] == 4
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(BatchConflict, match="consumed input bytes"):
        run_batch(spec, root)
