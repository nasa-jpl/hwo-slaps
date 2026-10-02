# Migration to explicit subhalo forecasting

The supported public operations are `simulate`, `prepare_forecast`, `forecast`,
`summarize_forecast`, `mass_reach`, and optional `validate_nonlinear`. The old
Pipeline enabled-mode union and run_pipeline/run_enhanced_pipeline wrappers are
removed. CLI operations are explicit: validate, simulate, forecast. Output paths
belong to CLI/I/O, not mandatory scientific configuration.

Configurations retain lensing/psf/observation/modeling/global_seed sections for
supported scientific conventions. run_name and plotting are optional metadata;
modeling.enabled, detection and fisher.mode no longer select a public operation.
Paths belong to the file declaring them. External PSFs use provider kernel with
path/sampling; fitted kernel identity is recorded from actual inputs.

Study and reproduction namespaces, frozen cohorts/assets, campaign release and
harvest tools, historical objective replay and test-only compatibility wrappers
are deleted. Git history retains the submitted code. Generic numerical math,
optical providers, source-image support, explicit population/statistics utilities,
current inference and validated acceleration remain supported.

Selection calls now require policy: cuts, weights and selected size. There is no
golden tier. Mass-reach reductions take numeric thresholds/targets and expose
censoring. Signed mismatch detection uses the actual data/model statistic by
default; raw matched-template power remains an explicit diagnostic.

No automatic artifacts or plots are produced by Python forecasting. NPZ result
I/O is explicit, versioned, without pickle and collision safe. Current nonlinear
fitting requires an output directory and explicit support for freed mass priors.
