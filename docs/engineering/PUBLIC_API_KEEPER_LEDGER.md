# Public forecasting API keeper ledger

The old Pipeline class and mode-dependent run_pipeline/run_enhanced_pipeline
entrypoints are intentionally retired. The three original routing declarations
only exercised fake collaborators or configuration transport. Complete source
and R/F/C/D evidence is in /private/tmp/hwoslaps-test-audit-20261002/public-pipeline-ledger.json.

Canonical owners now prove real file/provided PSF preparation, zero-noise smooth
expectations, explicit injected trial simulation, actual mass-position results,
calibration identity, real NPZ persistence, and CLI config/log/provenance binding.
New declarations in test_forecasting_api.py have distinct credible regressions:

- File/model PSF loss: a recorded fit kernel was dropped during configuration
  normalization. The keeper uses both actual kernels and requires mismatch fields.
- Injection/control confusion: a validation trial could be fitted to the smooth
  expectation. Real injection must change the image and report the actual mass.
- Sampling mismatch: angularly incompatible kernels must fail before model setup.
- Stale calibration: replay of an effective configuration must reject changed
  truth or model normalized detector-response bytes.
- Cached scientific identity: a caller editing a returned settings dictionary
  cannot relabel prepared weights or change the cached forecast.
- Hidden model-PSF exports: preparation rejects an explicit fitted optical PSF
  export request before generating artifacts; output is a caller operation.

Numerical mass-retarget/reference-JAX parity stays at test_forecast_evaluation;
this file does not duplicate those numerical oracles. Old snapshot serializer
contracts move to explicit forecast artifact and CLI owners, preserving typed
config hashing at test_provenance. No local tests were executed.

## Explicit whole-file retirement

`tests/test_pipeline_fisher_routing.py` is deliberately retired with its
production owners (`Pipeline`, `run_pipeline`, `run_enhanced_pipeline` and
`generator_fisher`). All three baseline declarations are accounted for:

| Baseline declaration | Mark | Current keeper or reason |
| --- | --- | --- |
| `test_programmatic_entry_resolves_without_mutating_or_exporting_maps` | C | Real public preparation/config nonmutation and explicit NPZ ownership in `test_forecasting_api.py`; composed path ownership in `test_config_loading.py`; CLI output/provenance in `test_cli.py` |
| `test_pipeline_routes_detection_to_fisher` | D | Fake-collaborator routing for the retired mode-dispatch API; actual prepared Fisher evaluation is proved by `test_forecast_evaluation.py` and `test_forecasting_api.py` |
| `test_generator_uses_fisher_detector` | D | Fake-collaborator delegation through the retired generator wrapper; real mass/position, reference/JAX and result-transport keepers own the scientific calculation |

No physical or numerical assertion in this file is retired without a current
owner. The no-automatic-model-PSF-export keeper also guards the explicit optical
fit branch, independently of the removed Pipeline routing implementation.
