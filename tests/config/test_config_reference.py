"""The saved configuration document is the actual complete CLI reference."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_saved_reference_equals_the_actual_command_bytes(tmp_path):
    """Protect the published file from stale or manually edited keys/defaults.

    A stale CONFIG.md fails here even when the reference command is correct. The
    CLI owner covers document composition, selectors and backend-free imports;
    the checker/schema owners cover accepted keys and strict parsing. This keeper
    owns only the persisted documentation boundary and uses no production seam.
    """
    completed = subprocess.run([sys.executable, "-m", "hwoslaps", "reference"], cwd=tmp_path,
                               check=True, capture_output=True, timeout=30)
    assert completed.stderr == b""
    assert completed.stdout == (ROOT / "docs/CONFIG.md").read_bytes()
