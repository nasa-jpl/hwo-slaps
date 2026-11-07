"""Write the illustrative Moffat truth and 2-percent-wider model kernels."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np


def kernel(fwhm_pixels):
    beta = 3.
    alpha = fwhm_pixels / (2 * np.sqrt(2 ** (1 / beta) - 1))
    centres = (np.arange(-7, 8)[:, None] + (np.arange(11) + .5) / 11 - .5).reshape(-1)
    y, x = np.meshgrid(centres, centres, indexing="ij")
    samples = (1 + (y * y + x * x) / alpha ** 2) ** (-beta)
    values = samples.reshape(15, 11, 15, 11).mean(axis=(1, 3))
    return np.asarray(values / values.sum(), dtype=np.float64)


def publish_array(target, values):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [(output / "truth_kernel.npy", 2.), (output / "model_kernel.npy", 2.04)]
    for target, _ in targets:
        if target.exists():
            raise FileExistsError(target)
    for target, width in targets:
        print(publish_array(target, kernel(width)), target.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
