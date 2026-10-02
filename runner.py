#!/usr/bin/env python
"""Source-checkout wrapper for the installed ``hwoslaps`` command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hwoslaps.cli import main  # noqa: E402


if __name__ == '__main__':
    raise SystemExit(main())
