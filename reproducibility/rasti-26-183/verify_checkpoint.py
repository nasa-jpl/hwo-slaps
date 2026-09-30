"""Verify the paper checkpoint without importing the fitting runtime."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checkpoint = Path(__file__).resolve().parent
    repository = checkpoint.parents[1]
    for line in (checkpoint / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(digest(checkpoint / name) == expected, f"Hash mismatch: {name}")

    provenance = json.loads((checkpoint / "provenance.json").read_text())
    settings = json.loads((checkpoint / "nonlinear_settings.json").read_text())
    require(
        digest(repository / provenance["fresh_profile_path"])
        == provenance["fresh_profile_sha256"],
        "The fitting engine differs from the paper checkpoint.",
    )
    with (checkpoint / "nonlinear_cases.csv").open(newline="") as stream:
        cases = list(csv.DictReader(stream))
    require(len(cases) == provenance["case_count"], "Case count differs.")
    require(len({c["case_id"] for c in cases}) == len(cases), "Duplicate case IDs.")
    for field, expected in (
        ("final_code_revision", provenance["nonlinear_adopted_revision_counts"]),
        ("prior_variant", provenance["adopted_variant_counts"]),
        ("final_numerical_status", provenance["final_status_counts"]),
    ):
        require(dict(Counter(c[field] for c in cases)) == expected, f"Different {field} counts.")

    for case in cases:
        for variant, column in (
            ("slam1", "first_pass_sampler_seed"),
            ("slam1_retry", "second_pass_sampler_seed"),
        ):
            material = f"wide_priors_20260926_v1|{variant}|{case['case_id']}".encode()
            seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
            require(seed == int(case[column]), f"Seed mismatch: {case['case_id']}")
            if variant == case["prior_variant"]:
                require(seed == int(case["sampler_seed"]), f"Adopted seed mismatch: {case['case_id']}")
        require(
            settings["original_protocol_hash_to_variant"][case["release_protocol_content_sha256"]]
            == case["prior_variant"],
            f"Protocol mismatch: {case['case_id']}",
        )

    for name, expected in provenance["archived_helper_sha256"].items():
        path = checkpoint / "archived_helpers" / name
        require(digest(path) == expected, f"Archived helper differs: {name}")
        compile(path.read_bytes(), str(path), "exec")
    print(f"Checkpoint verified: {len(cases)} cases, recorded seeds, file hashes, and fitting engine.")
    print("Archived helper syntax checked. No fits launched; no GPU or fitting runtime imported.")


if __name__ == "__main__":
    main()
