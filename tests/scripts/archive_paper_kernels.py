#!/usr/bin/env python3
"""Write ``tests/fixtures/paper_parity/archived_kernels.npz``: the paper's optical kernels as arrays.

The parity fixtures record these kernels only as SHA-256 digests, which a different NumPy
cannot reproduce bit for bit. This script rebuilds each kernel with the RASTI scientific
stack (Python 3.11, NumPy 1.26.4) and the last engine commit that reproduced the digests
(212fe01), refuses to write unless every digest matches its record, and stores the kernels
the parity tests compare against: the 17x17 and 51x51 kernels whole, and the central
101x101 block of each 999x999 kernel.

Run on xtx from the root of a checkout, with ``hwoslaps`` resolving in a ``git archive 212fe01``
tree::

    git archive 212fe01 src | tar -x -C $WORK/engine_212fe01
    env PYTHONPATH=$WORK/engine_212fe01/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
        /data/home/gvassilakis/Software/miniconda3/envs/hwo-slaps/bin/python \\
        tests/scripts/archive_paper_kernels.py --out tests/fixtures/paper_parity/archived_kernels.npz
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "tests/fixtures/paper_parity"
CORE = 101


def _kernel_sha256(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(f"{array.shape[0]}x{array.shape[1]}:".encode() + array.tobytes()).hexdigest()


def _k1_truth(shape):
    return {"truth": {
        "kind": "optical",
        "pupil": {"kind": "hex_segmented", "diameter_m": 7.225765, "pixels": 512, "supersampling": 4, "rings": 2,
                  "segment_point_to_point_m": 1.65, "gap_m": 0.006},
        "focal_length_m": 144.0, "wavelength_nm": 500.0, "detector_oversampling": 3, "kernel_shape": [shape, shape],
        "draw": {"prior": {"packaged": "jwst_wss_static_v1"}, "amplitude_rms_nm": 35.0, "seed": 20260835,
                 "family": "combined"}}}


def _core(kernel):
    start = (kernel.shape[0] - CORE) // 2
    return kernel[start:start + CORE, start:start + CORE]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from hwoslaps.optics.providers import build_model_psf, build_psf_provider, parse_psf

    manifest = json.loads((PARITY / "manifest.json").read_text())["scenes"]
    anchor = json.loads((PARITY / "k1_paper_kernel.json").read_text())
    oracle = json.loads((PARITY / "production/oracle.json").read_text())

    def k1(shape):
        return build_psf_provider(parse_psf(_k1_truth(shape)).truth,
                                  pixel_scale_arcsec=anchor["pixel_scale_arcsec"]).kernel().kernel

    def scene(path):
        config = yaml.safe_load((PARITY / path).read_text())
        spec, scale = parse_psf(config["psf"]), config["scene"]["grid"]["pixel_scale_arcsec"]
        truth = build_psf_provider(spec.truth, pixel_scale_arcsec=scale)
        return truth.kernel().kernel, build_model_psf(spec.model, truth, pixel_scale_arcsec=scale).provider.kernel().kernel

    fisher_truth, _ = scene("engine/p1_optical_matched.yaml")
    _, knowledge_error = scene("engine/p2_delta_knowledge_error.yaml")
    production, _ = scene("production/engine/production_sys0043.yaml")
    k1_51, k1_999 = k1(51), k1(999)
    checks = ((fisher_truth, manifest["p1_optical_matched"]["truth_kernel_sha256"]),
              (knowledge_error, manifest["p2_delta_knowledge_error"]["fit_kernel_sha256"]),
              (k1_51, anchor["kernels"]["51x51"]["sha256"]), (k1_999, anchor["kernels"]["999x999"]["sha256"]),
              (production, oracle["truth_kernel_sha256"]))
    for kernel, expected in checks:
        if _kernel_sha256(kernel) != expected:
            raise SystemExit(f"a {kernel.shape} kernel does not reproduce its archived digest {expected}; "
                             "run with the RASTI environment and the 212fe01 engine")
    np.savez(args.out, fisher_optical_17x17=fisher_truth, fisher_knowledge_error_17x17=knowledge_error,
             k1_51x51=k1_51, k1_999x999_core=_core(k1_999), production_999x999_core=_core(production))


if __name__ == "__main__":
    main()
