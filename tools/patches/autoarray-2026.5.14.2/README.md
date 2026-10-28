# Validated autoarray patches

These are the two patches present in the supported XTX environment: content-keyed
Convolver state reuse and separable blurring-mask dilation. Their original and
patched source hashes were retrieved read-only and matched the declared catalog on
2026-10-05. The source receipt is in the run-directory handoff.

The installer first validates every target against SHA256SUMS, skips already patched
files, and otherwise applies the exact diff and requires the patched hash. Unexpected
source stops installation. Acceptance uses a fresh scratch environment; the existing
scientific XTX environment is never modified.

Optics and paper-parity validation protect the affected convolution paths. The patch
diffs and hashes identify the source; they do not replace numerical gate receipts.
