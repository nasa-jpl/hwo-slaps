"""A hung owned job becomes a reported failure without losing other members."""
import json
import subprocess
import sys

import pytest

from hwoslaps.batch import BatchIncomplete, run_batch
from hwoslaps.batch.processes import group_members


@pytest.mark.backend
def test_job_timeout_drains_owned_worker_and_other_members_complete(tiny_batch_spec, tmp_path, monkeypatch):
    entry = tmp_path / 'timeout-worker.py'
    entry.write_text('''
if __name__ == '__main__':
    import time
    import hwoslaps.batch.worker as worker
    actual = worker.run_job
    def delayed(payload, *args, **kwargs):
        if 'system_000000/' in payload.job_id:
            time.sleep(60.)
        return actual(payload, *args, **kwargs)
    worker.run_job = delayed
    raise SystemExit(worker.main())
''')
    actual_popen = subprocess.Popen
    def wrapped(command, *args, **kwargs):
        if command == [sys.executable, '-m', 'hwoslaps.batch.worker']:
            command = [sys.executable, str(entry)]
        return actual_popen(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', wrapped)
    spec = tiny_batch_spec(execution={'workers_per_device': 2, 'job_timeout_seconds': 5.})
    output = tmp_path / 'timeout'
    with pytest.raises(BatchIncomplete) as error:
        run_batch(spec, output)
    report = error.value.report
    assert report.counts['completed'] == 1 and report.counts['failed'] == 1
    assert report.failures[0]['error_type'] == 'JobTimeout'
    workers = [json.loads(line) for line in (output / 'sessions/1/workers.jsonl').read_text().splitlines()]
    assert workers and all(not group_members(worker) for worker in workers)
    assert json.loads((output / 'sessions/1/report.json').read_text())['counts']['failed'] == 1
    monkeypatch.setattr(subprocess, 'Popen', actual_popen)
    resumed = run_batch(spec, output)
    assert resumed.counts['skipped'] == 1 and resumed.counts['completed'] == 1 and resumed.counts['failed'] == 0
