"""Micro-bench for one compiled nonlinear likelihood batch.

Builds the same dataset, analysis, and freed subhalo model the fixed-seed
bench fits, compiles the AutoFit vectorized likelihood at the production
batch size, and reports the compiled cost analysis, the optimized-HLO
time share by operation category, and the measured evaluation rate.

The Nautilus search evaluates this executable tens of thousands of times,
so its per-evaluation cost is the science floor of a nonlinear fit.

Engineering bench, disposable. Not paper data.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import yaml

DELTA_BLOCK = {
    "prior_table": "configs/psf_priors/jwst_wss_drift_v1.yaml",
    "seed": 20260814,
    "family": "combined",
    "amplitude_rms_nm": 0.0,
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the micro-bench command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Truth-state configuration YAML")
    parser.add_argument("output_dir", help="Directory for the JSON report")
    parser.add_argument("--jax-n-batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=200)
    return parser


def _hlo_category(name: str) -> str:
    """Map an HLO opcode to a coarse cost category."""
    if "fft" in name:
        return "fft_convolution"
    if "convolution" in name:
        return "direct_convolution"
    if "dot" in name:
        return "linear_algebra"
    if "reduce" in name:
        return "reduction"
    if any(k in name for k in ("gather", "scatter", "dynamic-slice", "slice")):
        return "gather_scatter"
    if any(k in name for k in ("transpose", "reshape", "broadcast", "copy",
                               "concatenate", "pad", "bitcast")):
        return "data_movement"
    if "fusion" in name or "custom-call" in name:
        return "fusion_or_custom"
    return "elementwise_other"


def main(argv=None) -> None:
    """Compile and time one vectorized likelihood batch."""
    args = _build_parser().parse_args(argv)

    import jax
    import numpy as np

    from hwoslaps.lensing.generator import generate_lensing_system
    from hwoslaps.modeling.nonlinear.autolens_model_builder import (
        autofit_model_from_spec,
        subhalo_model_spec_from_trial,
    )
    from hwoslaps.modeling.nonlinear.autolens_runner import (
        AutoLensFitRunner,
        NonlinearSearchSettings,
    )
    from hwoslaps.modeling.nonlinear.dataset_builder import (
        imaging_from_observation,
    )
    from hwoslaps.modeling.nonlinear.mass_mapping import (
        build_mass_mapping_context,
    )
    from hwoslaps.modeling.nonlinear.trial import trial_from_lensing_truth
    from hwoslaps.observation.generator import generate_observation
    from hwoslaps.psf.generator import generate_psf_system

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    base_config.pop("provenance_note", None)
    base_config["plotting"] = {"enabled": False}

    lensing_data = generate_lensing_system(
        base_config["lensing"], full_config=base_config
    )
    psf_data = generate_psf_system(base_config["psf"], full_config=base_config)
    observation = generate_observation(
        lensing_data=lensing_data,
        psf_data=psf_data,
        observation_config=base_config["observation"],
        full_config=base_config,
    )
    trial = trial_from_lensing_truth(lensing_data, case_id="micro")
    mass_context = build_mass_mapping_context(base_config)

    config = copy.deepcopy(base_config)
    config["modeling"]["fit_psf"] = {"mode": "delta", "delta": dict(DELTA_BLOCK)}

    dataset, metadata = imaging_from_observation(observation)
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(
            number_of_cores=1,
            use_jax=True,
            jax_n_batch=args.jax_n_batch,
        ),
        output_dir=str(output_dir),
    )
    spec = subhalo_model_spec_from_trial(
        config,
        trial=trial,
        fit_mode="freed",
        mass_context=mass_context,
    )
    model = autofit_model_from_spec(spec)
    analysis = runner.make_analysis(dataset, model_metadata=spec.metadata)

    def call(parameters):
        instance = model.instance_from_vector(vector=parameters, xp=jax.numpy)
        return analysis.log_likelihood_function(instance=instance)

    n_param = model.prior_count
    rng = np.random.default_rng(0)
    unit = rng.uniform(0.4, 0.6, size=(args.jax_n_batch, n_param))
    vectors = np.array(
        [model.vector_from_unit_vector(unit_vector=list(row)) for row in unit]
    )

    # AutoFit builds the vectorized likelihood as vmap(jit(call)); the
    # outer-jit form is compiled alongside it so the dispatch cost of the
    # two orderings can be compared on the same executable body.
    batched = jax.vmap(jax.jit(call))

    start = time.time()
    lowered = jax.jit(batched).lower(vectors)
    compiled = lowered.compile()
    compile_s = time.time() - start

    cost = compiled.cost_analysis()
    if isinstance(cost, list):
        cost = cost[0]

    hlo_text = compiled.as_text()
    op_counts = defaultdict(int)
    for line in hlo_text.splitlines():
        match = re.search(r"=\s+\S+\s+([a-z0-9-]+)\(", line)
        if match:
            op_counts[_hlo_category(match.group(1))] += 1

    def time_form(func):
        """Return seconds for ``repeats`` batches through one callable."""
        func(vectors).block_until_ready()
        start = time.time()
        for _ in range(args.repeats):
            values = func(vectors)
        values.block_until_ready()
        return time.time() - start

    elapsed = time_form(batched)
    elapsed_outer_jit = time_form(compiled)

    n_eval = args.repeats * args.jax_n_batch
    report = {
        "n_free_parameters": int(n_param),
        "jax_n_batch": args.jax_n_batch,
        "n_unmasked_pixels": metadata.n_unmasked_pixels,
        "compile_s": compile_s,
        "repeats": args.repeats,
        "elapsed_s_autofit_vmap_of_jit": elapsed,
        "elapsed_s_jit_of_vmap": elapsed_outer_jit,
        "batches_per_s": args.repeats / elapsed,
        "likelihood_evals_per_s": n_eval / elapsed,
        "ms_per_evaluation": elapsed * 1000.0 / n_eval,
        "ms_per_batch": elapsed * 1000.0 / args.repeats,
        "ms_per_batch_jit_of_vmap": elapsed_outer_jit * 1000.0 / args.repeats,
        "flops": cost.get("flops"),
        "bytes_accessed": cost.get("bytes_accessed"),
        "hlo_op_counts_by_category": dict(op_counts),
        "hlo_instruction_count": sum(op_counts.values()),
    }
    (output_dir / "likelihood_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
