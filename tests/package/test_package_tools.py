"""The forecast timing script runs on the minimal configuration."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.backend
def test_forecast_timing_script_reports_the_minimal_forecast(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "tools/benchmark_forecast.py"),
                                str(ROOT / "configs/minimal.yaml"), "--masses", "1e8", "1e9"],
                               cwd=tmp_path, capture_output=True, text=True, timeout=300, check=True)
    record = json.loads(completed.stdout)
    assert record["engine"] == "reference" and record["nodes"] > 0
    assert set(record["seconds"]) == {"prepare", "forecast_first", "forecast_repeat"}
    assert not any(tmp_path.iterdir())
