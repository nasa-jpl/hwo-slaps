"""Write tests/fixtures/analysis/table5_knowledge_error.json from the archived masks.

The fixture holds integer aperture-mask counts for the 24 Fisher tolerances behind RASTI
Table 5 (12 systems, M10 and M50 rungs), the expected per-system tolerances and the
Table 5 Retained and Pass columns. Every count is recomputed from the archived masks and
every quantile, gate, tolerance and median is checked against the reducer output
(``tests/scripts/psf_retention_table.py`` run on the same campaign)::

    python tests/scripts/psf_retention_table.py --campaign CAMPAIGN --output TABLE
    python tests/scripts/extract_table5_knowledge.py --campaign CAMPAIGN --table TABLE --output FIXTURE

``FIXTURE`` gets a sibling ``.inputs.sha256`` listing every mask file read; the fixture
records that list's digest.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLE_SHA256 = "6134857968389595e378398b8feba52882676cd6156d2077f16e207622b09e6b"
"""Digest of the reducer output behind the submitted table."""
MANUSCRIPT_SHA256 = "8a3338798687255822aebf2583815018331fc89c081ae02c5ed0f04027330768"
AMPLITUDES = (1, 2, 5, 10, 20)
TABLE_AMPLITUDES = (2, 5, 10, 20)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(archive, name):
    shape = tuple(archive[name + "_shape"])
    return np.unpackbits(archive[name + "_packed"], bitorder="big")[:np.prod(shape)].astype(bool).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("campaign", "table", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.table) != TABLE_SHA256:
        raise ValueError("the reducer output differs from the one behind the submitted table")
    table = json.loads(args.table.read_text())
    inputs, cases, table5 = {}, [], {}
    for rung_class in ("m10", "m50"):
        q10_k = {amplitude: [] for amplitude in AMPLITUDES}
        passes = {amplitude: 0 for amplitude in AMPLITUDES}
        for system in table[rung_class]["systems"]:
            sid, logm = system["system_id"], system["logm"]

            def path(amplitude, direction):
                return args.campaign / "outputs" / sid / f"psf_knowledge_map_m{logm:.2f}_delta{amplitude}_dir{direction}.npz"

            def remember(filename):
                inputs[str(filename.relative_to(args.campaign))] = sha256(filename)

            reference_path = path(0, 0)
            with np.load(reference_path, allow_pickle=False) as archive:
                aperture = unpack(archive, "aperture_mask")
                reference = unpack(archive, "detectable_mask") & aperture
            remember(reference_path)
            reference_count = int(np.count_nonzero(reference))
            if reference_count != system["matched_cells"] or reference_count < 33:
                raise ValueError("reference count differs from the reducer or is below its floor")
            counts_by_amplitude = {}
            passing_K, passing_R = [], []
            for amplitude in AMPLITUDES:
                directions = []
                for direction in range(1, 9):
                    filename = path(amplitude, direction)
                    with np.load(filename, allow_pickle=False) as archive:
                        if not np.array_equal(unpack(archive, "aperture_mask"), aperture):
                            raise ValueError("direction aperture differs from the matched map")
                        mismatch = unpack(archive, "mismatch_detectable_mask") & aperture
                        spurious = unpack(archive, "false_positive_mask") & aperture
                    remember(filename)
                    directions.append([int(np.count_nonzero(reference & mismatch)),
                                       int(np.count_nonzero(mismatch)), int(np.count_nonzero(spurious))])
                ratios = np.asarray(directions) / reference_count
                quantiles = dict(q10_K=float(np.quantile(ratios[:, 0], .1, method="linear")),
                                 q10_R=float(np.quantile(ratios[:, 1], .1, method="linear")),
                                 q90_F=float(np.quantile(ratios[:, 2], .9, method="linear")))
                original = system["per_amp"][str(amplitude)]
                for name, actual in quantiles.items():
                    if not np.isclose(actual, original[name], rtol=0, atol=1e-12):
                        raise ValueError(f"raw masks do not reproduce the reducer {sid}/{rung_class}/{amplitude}/{name}")
                for label, passing in (("K", passing_K), ("R", passing_R)):
                    passed = quantiles["q10_" + label] >= .9 and quantiles["q90_F"] <= .1
                    if passed != original["pass_" + label]:
                        raise ValueError("the reducer's per-amplitude gate differs")
                    if passed:
                        passing.append(amplitude)
                q10_k[amplitude].append(quantiles["q10_K"])
                passes[amplitude] += quantiles["q10_K"] >= .9 and quantiles["q90_F"] <= .1
                counts_by_amplitude[str(amplitude)] = directions
            for label, passing in (("K", passing_K), ("R", passing_R)):
                if passing != list(AMPLITUDES[:len(passing)]):
                    raise ValueError("the reducer's contiguous tolerance rule differs from largest passing on nonmonotone data")
                if max(passing, default=0) != system["tolerance_" + label]:
                    raise ValueError("raw masks do not reproduce the reducer tolerance")
            cases.append(dict(system_id=sid, rung_class=rung_class, log10_mass_msun=logm,
                              reference_count=reference_count, directions_by_amplitude_nm=counts_by_amplitude,
                              expected_tolerance_K_nm=system["tolerance_K"],
                              expected_tolerance_total_area_nm=system["tolerance_R"]))
        expected = table[rung_class]["table"]
        for amplitude in AMPLITUDES:
            retained = float(np.quantile(q10_k[amplitude], .5, method="inverted_cdf"))
            if retained != expected[str(amplitude)]["median_q10_K"] or passes[amplitude] != expected[str(amplitude)]["pass_K"]:
                raise ValueError(f"raw masks do not reproduce the reducer Table 5 row {rung_class}/{amplitude}")
        table5[rung_class] = {str(amplitude): dict(retained_median_q10_K=expected[str(amplitude)]["median_q10_K"],
                                                   pass_K=expected[str(amplitude)]["pass_K"])
                              for amplitude in TABLE_AMPLITUDES}
    if len(cases) != 24:
        raise ValueError("expected exactly 24 system/rung-class cases")
    manifest = "".join(f"{digest}  {name}\n" for name, digest in sorted(inputs.items()))
    fixture = dict(schema=1, provenance={
        "description": "Real integer aperture-mask counts for the 24 Fisher tolerances underlying submitted RASTI Table 5, with expected K values and Table 5 columns from the original figure reducer output.",
        "source_archive": "psf_knowledge_fisher_v1",
        "source_reducer": {"path": "tests/scripts/psf_retention_table.py", "sha256": sha256(HERE / "psf_retention_table.py")},
        "source_table": {"path": "psf_retention_table.json, written by the reducer from the archive", "sha256": TABLE_SHA256},
        "source_extractor": {"path": "tests/scripts/extract_table5_knowledge.py", "sha256": sha256(Path(__file__).resolve())},
        "source_manuscript": {"path": "main.tex of the submitted RASTI manuscript", "sha256": MANUSCRIPT_SHA256,
                              "definition": "K = retained correct-PSF sensitive positions / correct-PSF sensitive positions; Q10(K)>=0.9 and Q90(F)<=0.1, inside radius 2 theta_E_eff."},
        "mask_archive_file_count": len(inputs),
        "mask_archive_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "mask_archive_manifest_format": "SHA256, two spaces, campaign-relative filename, newline; sorted by filename.",
        "extraction": "NumPy unpackbits (MSB first) of actual aperture-restricted masks; all per-amplitude K, R and F quantiles, gates and Table 5 medians checked against the original figure reducer output.",
        "expected_tolerance_rule": "The original reducer uses the largest contiguous passing amplitude. All 24 archived passing sets are monotone, so the goldens also test the generic largest-passing reducer.",
        "table5_rule": "Retained is the median over the 12 systems (inverse empirical CDF) of Q10(K) over the eight directions (linear interpolation); Pass counts systems with Q10(K)>=0.9 and Q90(F)<=0.1.",
        "excluded_amplitude_nm": 35,
    }, criterion=dict(retention_quantile=.1, retention_min=.9, spurious_quantile=.9, spurious_max=.1),
        direction_order=list(range(1, 9)),
        count_columns=["retained_in_aperture", "mismatched_in_aperture", "spurious_in_aperture"], cases=cases,
        table5=table5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep each eight-direction block on one line so the fixture is small and inspectable.
    text = json.dumps(fixture, indent=2) + "\n"
    for case in cases:
        block = json.dumps(case["directions_by_amplitude_nm"], indent=2)
        compact = "{\n" + ",\n".join(f'  "{amplitude}": {json.dumps(rows)}' for amplitude, rows in case["directions_by_amplitude_nm"].items()) + "\n}"
        # Apply the same indent as the nested fixture block.
        rendered = "\n".join("      " + line if index else line for index, line in enumerate(block.splitlines()))
        replacement = "\n".join("      " + line if index else line for index, line in enumerate(compact.splitlines()))
        text = text.replace(rendered, replacement, 1)
    if json.loads(text) != fixture:
        raise ValueError("compact serialization changed fixture content")
    args.output.write_text(text)
    args.output.with_suffix(".inputs.sha256").write_text(manifest)
    print(json.dumps(dict(fixture=str(args.output), fixture_sha256=sha256(args.output),
                          cases=len(cases), unique_mask_inputs=len(inputs),
                          manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
                          source_quantiles_and_tolerances_verified=True), indent=2))


if __name__ == "__main__":
    main()
