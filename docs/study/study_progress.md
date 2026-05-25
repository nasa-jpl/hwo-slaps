# Study progress: HWO-SLAPS PSF stability and subhalo detectability

This document tracks what has actually been run, decided, and produced for the
SPIE-to-RASTI HWO-SLAPS study. The roadmap defines the intended plan; this file
is the live progress log and result index.

## Status summary

Current phase: Stage 1 and Stage 2 complete; ready for SPIE manuscript/poster
assembly.

Overall state:

- [x] Canonical SCDD/SPIE baseline config locked.
- [x] Perfect-PSF mass sweep completed.
- [x] First PSF perturbation-family sweep completed.
- [x] Optional second PSF perturbation-family sweep completed.
- [x] Ring-map / detectable-ring-fraction demonstration completed.
- [x] Aggregate results table produced.
- [x] Initial Stage 0 figures produced.
- [x] PSF-mode coupling ranking produced for the selected SPIE scan basis.
- [x] Sparse nonlinear validation subset selected.
- [x] Sparse nonlinear validation pilot completed.
- [x] Full Stage 2 local nonlinear/profile validation completed on CPU; full
  Bayesian evidence / broad sampler calibration remains deferred to RASTI.

## Metric convention

Use the convention from `docs/study/study_roadmap.md` in every result table,
plot, and note:

- `q_F = Delta chi^2_F`
- `Z_F = sqrt(q_F)`
- `Delta log L_F,equiv = q_F / 2`
- SCDD local threshold: `Delta log L > 5`, equivalently `q_F > 10` and
  `Z_F > sqrt(10)`
- Nonlinear validation statistic: `q_fit = 2 * (log L_subhalo - log L_smooth)`

Fisher values should be described as Fisher-equivalent or Asimov forecasts until
validated against nonlinear fits.

## Decisions log

| Date | Decision | Rationale | Status |
|---|---|---|---|
| 2026-05-24 | Canonical baseline config | SCDD-like redshifts, NFW subhalo, high-resolution grid, perfect PSF by default. | Done |
| 2026-05-24 | First PSF perturbation family | Segment hexikes are HWO-segment-specific and already implemented. | Done |
| 2026-05-24 | PSF amplitude ladder | Wide `0, 1, 2, 5, 10, 20, 50, 100 nm RMS` ladder at `1e7 Msun`. | Done |
| 2026-05-24 | Degradation metric for first plot | Use local `q_F`, `Z_F`, profiling degradation, Strehl, and endpoint detectable-ring fraction. | Done |
| 2026-05-24 | SPIE nonlinear validation scope | Four-case fixed-template GPU pilot: `1e7`, `10^7.25`, `10^7.75`, and `1e7` with 100 nm segment hexike. | Done |
| 2026-05-24 | SPIE-plus verification scope | Four injected-subhalo local nonlinear profile fits plus one no-subhalo PSF-mismatch false-positive case. | Done |
| 2026-05-25 | Optional second SPIE PSF family | Add a global Zernike Noll 4 amplitude ladder and include global Zernikes Noll 4-6 in the selected mode-coupling scan basis. | Done |
| 2026-05-25 | Full CPU nonlinear validation | Validate all `21` Stage 2 injected cases with local nonlinear/profile checks, plus segment-hexike and global-Zernike false-positive controls. | Done |

## Stage 0: internal-review priority

Goal: produce a credible first poster package quickly.

Required artifacts:

- [x] Canonical SCDD-like baseline config.
- [x] Perfect-PSF mass sweep.
- [x] One PSF perturbation family with an amplitude ladder.
- [x] One ring-map / detectable-ring-fraction demonstration.
- [x] Metric-definition table.
- [x] Limitations box.

Progress notes:

- Stage 0 ran on 2026-05-24 with `13` independent Fisher cases.
- Stage 1/2 completion reran on 2026-05-25 with `21` independent Fisher cases:
  five perfect-PSF mass points, eight segment-hexike amplitude points, and
  eight global-Zernike amplitude points.
- The 2026-05-25 run was parallelized with `14` workers and an `8` thread cap
  per worker. All `21` cases completed successfully.
- The original `13`-case run completed after fixing a perfect-PSF
  segment-hexike derivative config edge case.
- The pivot mass `1e7 Msun` is above the SCDD threshold in the perfect-PSF
  local metric: `q_F = 17.6703`, `Z_F = 4.2036`.
- The selected segment-hexike mode produces modest local degradation across
  `0-100 nm RMS`: `q_F` changes from `17.6703` to `16.8690`.
- The optional global-Zernike Noll 4 sweep produces stronger but still
  above-threshold local degradation across `0-100 nm RMS`: `q_F` changes from
  `17.6703` to `14.7227`.

Results:

| Artifact | Path | Notes |
|---|---|---|
| Baseline config | `configs/study/scdd_spie_baseline.yaml` | Canonical Stage 0 SCDD/SPIE baseline. |
| Manifest | `scratch/study/stage0_manifest.yaml` | Expands the mass, segment-hexike, and optional global-Zernike sweeps. |
| Study runner | `scripts/run_stage0_study.py` | Supports process-level parallel execution via `--workers`. |
| Aggregate results table | `outputs/stage0_internal_review/results.csv` | Contains local Fisher metrics, map summaries, PSF diagnostics, provenance fields. |
| Reproducibility summary | `outputs/stage0_internal_review/study_provenance.json` | Includes git hash, command line, Python, and package versions. |
| Aggregate figures | `outputs/stage0_internal_review/figures/` | Mass sweep, segment-hexike, global-Zernike, and detectable-ring fraction summaries. |

## Stage 1: SPIE-level codebase hardening

Goal: make the current pipeline produce repeatable proceedings-level results.

Progress:

- [x] Create canonical SCDD/SPIE config separate from `configs/master_config.yaml`.
- [x] Add lightweight study manifest.
- [x] Add study runner for manifest expansion.
- [x] Add cross-run aggregation.
- [x] Record provenance: config hash, git hash, package versions, Python version,
  and command line.
- [x] Store PSF diagnostics.
- [x] Add initial Stage 0 plotting scripts.
- [x] Make canonical output paths portable.
- [x] Define optional global-Zernike PSF family for SPIE.
- [x] Add raw peak-ratio and perfect-kernel difference diagnostics.

Open implementation notes:

- `configs/master_config.yaml` remains a runnable example; the locked Stage 0
  config is `configs/study/scdd_spie_baseline.yaml`.
- The maintained modeling route is Fisher/Asimov; nonlinear validation should
  be treated as a sparse calibration layer.
- The selected Stage 2 PSF scan basis is segment hexike `segment 0, Noll 2`
  plus global Zernikes `Noll 4-6`. The required amplitude ladder remains
  segment hexike `segment 0, Noll 2`; the optional second amplitude ladder is
  global Zernike `Noll 4`.

## Stage 2: SPIE-level study

Goal: run the bounded SPIE grid without overclaiming final HWO requirements.

Planned baseline mass ladder:

| Mass label | Mass Msun | Status | Notes |
|---|---:|---|---|
| `1e7` | `1.0e7` | Run | SCDD anchor; ring map also run. |
| `10^7.25` | `1.778279e7` | Run | SCDD anchor. |
| `10^7.5` | `3.162278e7` | Run | SCDD anchor. |
| `10^7.75` | `5.623413e7` | Run | SCDD anchor. |
| `1e8` | `1.0e8` | Run | SCDD anchor. |

Perfect-PSF results:

| Run | Mass Msun | Position | q_F | Z_F | Delta log L_F,equiv | Pass q_F > 10 | Notes |
|---|---:|---|---:|---:|---:|---|---|
| `stage0_internal_review_mass_m1e7_perfect` | `1.000e7` | Einstein ring, 90 deg | `17.6703` | `4.2036` | `8.8352` | Yes | Ring-map fraction `0.875`. |
| `stage0_internal_review_mass_m10p7p25_perfect` | `1.778e7` | Einstein ring, 90 deg | `38.2564` | `6.1852` | `19.1282` | Yes | Local only. |
| `stage0_internal_review_mass_m10p7p5_perfect` | `3.162e7` | Einstein ring, 90 deg | `80.7494` | `8.9861` | `40.3747` | Yes | Local only. |
| `stage0_internal_review_mass_m10p7p75_perfect` | `5.623e7` | Einstein ring, 90 deg | `165.9551` | `12.8824` | `82.9775` | Yes | Local only. |
| `stage0_internal_review_mass_m1e8_perfect` | `1.000e8` | Einstein ring, 90 deg | `331.6119` | `18.2102` | `165.8059` | Yes | Local only. |

PSF perturbation sweep results:

| Run | PSF family | Mode | Amplitude | Units | Mass Msun | q_F | Z_F | Degradation | Detectable-ring fraction | Notes |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `stage0_internal_review_hexike_s0_n2_a0p0nm_m1e7` | segment hexike | segment 0, Noll 2 | `0` | nm RMS | `1.0e7` | `17.6703` | `4.2036` | `0.3202` | `0.875` | Perfect endpoint, mode scan enabled. |
| `stage0_internal_review_hexike_s0_n2_a1p0nm_m1e7` | segment hexike | segment 0, Noll 2 | `1` | nm RMS | `1.0e7` | `17.6702` | `4.2036` | `0.3202` |  | Local only. |
| `stage0_internal_review_hexike_s0_n2_a2p0nm_m1e7` | segment hexike | segment 0, Noll 2 | `2` | nm RMS | `1.0e7` | `17.6699` | `4.2036` | `0.3202` |  | Local only. |
| `stage0_internal_review_hexike_s0_n2_a5p0nm_m1e7` | segment hexike | segment 0, Noll 2 | `5` | nm RMS | `1.0e7` | `17.6678` | `4.2033` | `0.3202` |  | Local only. |
| `stage0_internal_review_hexike_s0_n2_a10p0nm_m1e7` | segment hexike | segment 0, Noll 2 | `10` | nm RMS | `1.0e7` | `17.6602` | `4.2024` | `0.3201` |  | Local only. |
| `stage0_internal_review_hexike_s0_n2_a20p0nm_m1e7` | segment hexike | segment 0, Noll 2 | `20` | nm RMS | `1.0e7` | `17.6298` | `4.1988` | `0.3200` |  | Local only. |
| `stage0_internal_review_hexike_s0_n2_a50p0nm_m1e7` | segment hexike | segment 0, Noll 2 | `50` | nm RMS | `1.0e7` | `17.4293` | `4.1748` | `0.3190` |  | Local only. |
| `stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7` | segment hexike | segment 0, Noll 2 | `100` | nm RMS | `1.0e7` | `16.8690` | `4.1072` | `0.3161` | `0.875` | High-amplitude endpoint, mode scan enabled. |

Optional global-Zernike sweep results:

| Run | PSF family | Mode | Amplitude | Units | Mass Msun | q_F | Z_F | Degradation | Detectable-ring fraction | Notes |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `stage0_internal_review_global_zernike_n4_a0p0nm_m1e7` | global Zernike | Noll 4 | `0` | nm RMS | `1.0e7` | `17.6703` | `4.2036` | `0.3202` | `0.875` | Perfect endpoint, mode scan enabled. |
| `stage0_internal_review_global_zernike_n4_a1p0nm_m1e7` | global Zernike | Noll 4 | `1` | nm RMS | `1.0e7` | `17.6699` | `4.2036` | `0.3202` |  | Local only. |
| `stage0_internal_review_global_zernike_n4_a2p0nm_m1e7` | global Zernike | Noll 4 | `2` | nm RMS | `1.0e7` | `17.6687` | `4.2034` | `0.3202` |  | Local only. |
| `stage0_internal_review_global_zernike_n4_a5p0nm_m1e7` | global Zernike | Noll 4 | `5` | nm RMS | `1.0e7` | `17.6601` | `4.2024` | `0.3201` |  | Local only. |
| `stage0_internal_review_global_zernike_n4_a10p0nm_m1e7` | global Zernike | Noll 4 | `10` | nm RMS | `1.0e7` | `17.6296` | `4.1988` | `0.3199` |  | Local only. |
| `stage0_internal_review_global_zernike_n4_a20p0nm_m1e7` | global Zernike | Noll 4 | `20` | nm RMS | `1.0e7` | `17.5091` | `4.1844` | `0.3191` |  | Local only. |
| `stage0_internal_review_global_zernike_n4_a50p0nm_m1e7` | global Zernike | Noll 4 | `50` | nm RMS | `1.0e7` | `16.7372` | `4.0911` | `0.3139` |  | Local only. |
| `stage0_internal_review_global_zernike_n4_a100p0nm_m1e7` | global Zernike | Noll 4 | `100` | nm RMS | `1.0e7` | `14.7227` | `3.8370` | `0.2995` | `0.875` | High-amplitude endpoint, mode scan enabled. |

Ring-map summary:

| Run | Mass Msun | PSF case | Positions | Median Z_F | Max Z_F | Detectable-ring fraction | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| `stage0_internal_review_mass_m1e7_perfect` | `1.0e7` | perfect | `24` | `4.2638` | `4.9372` | `0.875` | Perfect-PSF ring-map demonstration. |
| `stage0_internal_review_hexike_s0_n2_a0p0nm_m1e7` | `1.0e7` | perfect | `24` | `4.2638` | `4.9372` | `0.875` | Duplicate perfect endpoint for PSF sweep. |
| `stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7` | `1.0e7` | segment hexike | `24` | `4.1782` | `4.8622` | `0.875` | High-amplitude endpoint. |
| `stage0_internal_review_global_zernike_n4_a0p0nm_m1e7` | `1.0e7` | perfect | `24` | `4.2638` | `4.9372` | `0.875` | Duplicate perfect endpoint for optional global-Zernike sweep. |
| `stage0_internal_review_global_zernike_n4_a100p0nm_m1e7` | `1.0e7` | global Zernike | `24` | `3.9926` | `4.7604` | `0.875` | High-amplitude optional global-Zernike endpoint. |

PSF-mode coupling scan summary:

Mode scan source: `outputs/stage0_internal_review/results.csv`, columns
`mode_scan_*`. The selected scan basis is `psf.segment_hexikes[0][2]` and
`psf.global_zernikes[4-6]`, with `5 nm RMS` one-sigma amplitudes for the
selected coefficient families.

| Run | Leading mode | z per unit | z at 1 sigma | Tolerance for z=1 | Notes |
|---|---|---:|---:|---:|---|
| `stage0_internal_review_hexike_s0_n2_a0p0nm_m1e7` | `psf.segment_hexikes[0][2]` | `0.000848` | `0.00424` | `1179.42` | Perfect endpoint; global modes have negligible coupling. |
| `stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7` | `psf.segment_hexikes[0][2]` | `0.07989` | `0.39945` | `12.52` | Segment-hexike endpoint; injected family remains dominant. |
| `stage0_internal_review_global_zernike_n4_a100p0nm_m1e7` | `psf.global_zernikes[5]` | `0.65456` | `3.27282` | `1.53` | Global-Zernike endpoint; Noll 5 is the strongest selected coupling, followed by Noll 4. |

## Nonlinear validation

SPIE scope: sparse validation only, if it does not block the Fisher study.

Candidate validation cases:

- [x] Perfect PSF at `1e7 Msun`, one near-threshold anchor position.
- [x] Perfect PSF at `10^7.25 Msun`, one position.
- [x] Perfect PSF at `10^7.75 Msun`, one position.
- [x] One perturbed-PSF case at the same position.
- [x] Optional no-subhalo PSF-mismatch false-positive case.

Pilot run notes:

- Ran on 2026-05-24 using PyAutoLens JAX mode on GPUs `0,1,2,3`.
- Settings: fixed-template fits, Asimov datasets, correct PSF supplied to the
  fit, `n_live_smooth=100`, `n_live_subhalo=100`, `maxcall=1000`.
- All four workers completed successfully, but the resulting likelihood-ratio
  values are not internally consistent enough to use as a Fisher calibration.
- Follow-up debugging found two setup issues and one remaining search issue:
  nonlinear validation data must be in PyAutoLens rate units, not
  exposure-integrated ADU; the subhalo model should be attached as a lens-galaxy
  component to match the forward generator; and the bounded Nautilus search is
  still not reliable enough as a calibration optimizer.
- Deterministic PyAutoLens truth-tracer fits now agree with direct chi-squared
  diagnostics and track the Fisher raw statistic. The lower reported Stage 0
  `q_F` is expected because it is profiled over lens/source nuisance directions.

Rejected sampler-pilot results:

| Case | Fit mode | PSF case | q_F | q_fit | q_fit / q_F | Fisher pass | Fit pass | Status | Notes |
|---|---|---|---:|---:|---:|---|---|---|---|
| `perfect_m1e7_near_threshold` | fixed_template | perfect | `17.6703` | `13176.5261` | `745.6862` | Yes | Yes | Pilot only | Positive but implausibly high relative to Fisher. |
| `perfect_m10p7p25_moderate` | fixed_template | perfect | `38.2564` | `16180.1275` | `422.9396` | Yes | Yes | Pilot only | Positive but implausibly high relative to Fisher. |
| `perfect_m10p7p75_high` | fixed_template | perfect | `165.9551` | `0.0000` | `0.0000` | Yes | No | Pilot only | Subhalo search maximum landed below smooth maximum. |
| `hexike100_m1e7_endpoint` | fixed_template | segment hexike | `16.8690` | `0.0000` | `0.0000` | Yes | No | Pilot only | Subhalo search maximum landed below smooth maximum. |

Accepted local-profile calibration:

Run source: `outputs/stage0_profile_calibration/results.csv`.
Full Stage 2 run source: `outputs/stage0_profile_calibration_full/results.csv`.

Definition: evaluate the full nonlinear HWO-SLAPS forward model at the smooth
model nuisance point selected by the Fisher profile solution, then compare the
resulting profile chi-squared to `q_F`. This tests the metric calibration while
separating it from global-sampler search failures.

| Case | PSF case | q_F | q_profile,nonlinear | q_profile / q_F | Relative diff | Status | Notes |
|---|---|---:|---:|---:|---:|---|---|
| `perfect_m1e7_near_threshold` | perfect | `17.6703` | `17.6645` | `0.9997` | `0.0328%` | Accepted | Near-threshold anchor. |
| `perfect_m10p7p25_moderate` | perfect | `38.2564` | `38.2375` | `0.9995` | `0.0494%` | Accepted | Moderate-mass anchor. |
| `perfect_m10p7p75_high` | perfect | `165.9551` | `165.7874` | `0.9990` | `0.1011%` | Accepted | High-mass anchor. |
| `hexike100_m1e7_endpoint` | segment hexike | `16.8690` | `16.8635` | `0.9997` | `0.0325%` | Accepted | 100 nm RMS segment-hexike endpoint. |

Full CPU profile-calibration sweep:

- Completed on 2026-05-25 for all `21` Stage 2 cases in
  `outputs/stage0_internal_review/results.csv`.
- Result: `0 / 21` status failures and `0 / 21` alignment failures.
- Nonlinear-profile-to-Fisher ratio range:
  `0.9987277594-0.9996937663`.
- Maximum relative `q` difference: `0.127224%`.
- Maximum absolute `q` difference: `0.421890`.

SPIE-plus local nonlinear optimization verification:

Run source: `outputs/stage0_spie_plus_validation/results.csv`.
Full Stage 2 run source:
`outputs/stage0_spie_plus_validation_full/results.csv`.

Definition: locally optimize the full nonlinear forward model nuisance
parameters for the smooth and fixed-subhalo hypotheses, then compute
`q_fit = chi2_smooth,min - chi2_subhalo,min`. Injected-subhalo cases use the
same PSF, mask, noise convention, and scalar nuisance set as Fisher. The
false-positive case generates no-subhalo data with the 100 nm segment-hexike PSF
and fits it with a perfect-PSF model.

| Case | Truth PSF | Fit PSF | Injected subhalo | q_F | q_fit | q_fit / q_F | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| `perfect_m1e7_near_threshold` | perfect | perfect | Yes | `17.6703` | `17.6645` | `0.9997` | Accepted | Two smooth starts converge to the same profile value. |
| `perfect_m10p7p25_moderate` | perfect | perfect | Yes | `38.2564` | `38.2374` | `0.9995` | Accepted | Threshold agreement: both pass `q > 10`. |
| `perfect_m10p7p75_high` | perfect | perfect | Yes | `165.9551` | `165.7833` | `0.9990` | Accepted | Threshold agreement: both pass `q > 10`. |
| `hexike100_m1e7_endpoint` | segment hexike | segment hexike | Yes | `16.8690` | `16.8635` | `0.9997` | Accepted | Endpoint PSF case. |
| `false_positive_hexike100_fit_perfect` | segment hexike | perfect | No |  | `0.0000` |  | Accepted | No unexplained threshold-level false positive. |

Full CPU SPIE-plus sweep:

- Completed on 2026-05-25 for all `21` injected Stage 2 cases plus `2`
  no-subhalo PSF-mismatch false-positive controls.
- Result: `0 / 23` SPIE-plus failures.
- Injected-subhalo `q_fit / q_F` range:
  `0.9986564700-0.9996927562`.
- Injected-subhalo `q_fit` range: `14.7181269588-331.1663649054`;
  all injected cases agree with Fisher on the `q > 10` threshold.
- False-positive controls:
  `stage0_spie_plus_false_positive_hexike100_truth_fit_perfect` and
  `stage0_spie_plus_false_positive_global_zernike100_truth_fit_perfect` both
  give `q_fit = 0.0 < 10`.
- Full-run validation plot:
  `outputs/stage0_spie_plus_validation_full/q_f_vs_q_fit.png`.

Forecast robustness checks:

Run source: `outputs/stage0_forecast_robustness/`.

Scope: lightweight forecast checks added to reduce dependence on one clean
Asimov demonstration. The noisy-ensemble check uses the profiled linear Fisher
likelihood on noisy data, not a full nonlinear noisy search. The position
variation check evaluates the full nonlinear forward model at the
Fisher-profiled smooth solution for multiple ring positions.

Noisy ensemble summary:

| Case | Seeds | q_F Asimov | Noisy median q | 16-84% range | Median / q_F | Detected fraction q > 10 |
|---|---:|---:|---:|---:|---:|---:|
| `perfect_m1e7_near_threshold` | `5` | `17.6703` | `14.9248` | `7.2384-23.8377` | `0.8446` | `0.6000` |
| `perfect_m10p7p25_moderate` | `5` | `38.2564` | `34.0907` | `21.5383-44.2779` | `0.8911` | `1.0000` |

False-positive controls:

| Case | Truth PSF | Fit PSF | q_fit | Pass q < 10 | Notes |
|---|---|---|---:|---|---|
| `false_positive_perfect_truth_fit_perfect` | perfect | perfect | `0.0000` | Yes | Exact same-PSF no-subhalo control. |
| `false_positive_hexike100_truth_fit_hexike100` | segment hexike | segment hexike | `0.0000` | Yes | Exact same-PSF hexike no-subhalo control. |
| `false_positive_hexike100_truth_fit_perfect` | segment hexike | perfect | `0.0000` | Yes | 100 nm segment-hexike truth data fit with perfect-PSF model. |

Position variation at `1e7 Msun`, perfect PSF:

| Angle deg | q_F | q_forward_profile | Ratio | Pass q > 10 |
|---:|---:|---:|---:|---|
| `0` | `15.3831` | `15.3921` | `1.0006` | Yes |
| `45` | `20.3889` | `20.3842` | `0.9998` | Yes |
| `90` | `17.6703` | `17.6645` | `0.9997` | Yes |
| `135` | `19.6042` | `19.6063` | `1.0001` | Yes |
| `180` | `22.7974` | `22.7843` | `0.9994` | Yes |
| `225` | `17.1165` | `17.1064` | `0.9994` | Yes |
| `270` | `7.7433` | `7.7464` | `1.0004` | No |
| `315` | `16.8871` | `16.8923` | `1.0003` | Yes |

Deterministic truth-tracer checks:

| Case | PSF case | q_F profiled | q_truth PyAutoLens | q_truth / q_F | Status | Notes |
|---|---|---:|---:|---:|---|---|
| `perfect_m1e7_near_threshold` | perfect | `17.6703` | `55.1764` | `3.1225` | Diagnostic pass | Matches Fisher raw `55.1902`. |
| `perfect_m10p7p25_moderate` | perfect | `38.2564` | `136.5977` | `3.5706` | Diagnostic pass | Truth model beats smooth model at fixed parameters. |
| `perfect_m10p7p75_high` | perfect | `165.9551` | `819.1794` | `4.9362` | Diagnostic pass | Truth model beats smooth model at fixed parameters. |
| `hexike100_m1e7_endpoint` | segment hexike | `16.8690` | `53.3598` | `3.1632` | Diagnostic pass | Truth model beats smooth model at fixed parameters. |

Calibration summary:

- Calibration relation: accepted for the local profiled likelihood metric over
  all `21` Stage 2 cases. The nonlinear forward-model profile statistic agrees
  with `q_F` to better than `0.13%` in every accepted case.
- Bounded Nautilus fixed-template pilot remains rejected as a calibration
  source because the search did not reliably find the intended likelihood
  basins.
- Median `q_truth / q_F`: `3.3666` for deterministic truth-tracer checks.
- Threshold-confusion summary: all `21` full-sweep local-profile calibration
  cases pass both Fisher and nonlinear-profile `q > 10`.
- SPIE-plus optimization summary: all `21` injected-subhalo local optimizer
  cases satisfy `0.8 <= q_fit/q_F <= 1.2`, and both no-subhalo PSF-mismatch
  cases have `q_fit = 0 < 10`.
- Forecast robustness summary: the near-threshold `1e7 Msun` case has noisy
  median `q = 14.9248` with `60%` of five noisy seeds above threshold; the
  `10^7.25 Msun` case has noisy median `q = 34.0907` with `100%` above
  threshold; all three deterministic no-subhalo controls stay below threshold;
  and all eight ring-position forward checks match `q_F` to better than `0.1%`.
- Current claim boundary: calibrated local Fisher/Asimov forecast with
  deterministic PyAutoLens truth-likelihood sanity checks and full Stage 2
  local nonlinear optimization verification, plus lightweight noisy,
  false-positive, and position-variation forecast checks. Full Bayesian
  evidence / broad sampler calibration remains a separate sampler-engineering
  problem.

## Stage 3: SPIE manuscript and poster

Required figure/status tracker:

- [ ] Pipeline schematic.
- [x] Example lensing scene and subhalo residual.
- [x] Example perfect and perturbed PSF diagnostic.
- [x] Detection statistic versus subhalo mass.
- [x] Detection degradation or detectable-ring fraction versus PSF amplitude.
- [x] PSF-mode coupling or tolerance-style plot.
- [x] Fisher ring-map or detectable-ring-fraction figure.
- [x] Optional Fisher-versus-nonlinear calibration plot.
- [x] SPIE-plus Fisher-versus-local-optimizer validation plot.

Manuscript/poster notes:

- Use preliminary framework language.
- Avoid final engineering requirement claims.
- State Fisher/Asimov limitations clearly.
- State that local nonlinear optimization verifies the full Stage 2 grid, while
  full Bayesian evidence / broad sampler calibration is deferred to RASTI.
- State that the current noisy ensemble is a small profiled-Fisher ensemble, not
  a full noisy nonlinear recovery campaign.
- State PSF amplitude units in every relevant figure caption.

## Stage 4-6: RASTI expansion

RASTI work starts after the SPIE baseline is reproducible.

Deferred items:

- [ ] Supported analysis module.
- [ ] Full study manifest and aggregator.
- [ ] Full 2D detectable-area maps.
- [ ] Requirement-curve generation.
- [ ] Mandatory nonlinear calibration grid.
- [ ] False-positive PSF-mismatch study.
- [ ] Source-realism stress tests.
- [ ] Lens-light and subtraction-residual stress tests.

## Artifact index

| Artifact type | Path | Created | Notes |
|---|---|---|---|
| Canonical config | `configs/study/scdd_spie_baseline.yaml` | Yes | Stage 0 SCDD/SPIE baseline. |
| Manifest | `scratch/study/stage0_manifest.yaml` | Yes | Stage 0 mass and PSF sweeps. |
| Generated run configs | `outputs/stage0_internal_review/generated_configs/` | Yes | One config per run. |
| Aggregate results CSV | `outputs/stage0_internal_review/results.csv` | Yes | `21` successful rows. |
| Aggregate nonlinear CSV | `outputs/stage0_nonlinear_validation/results.csv` | Yes | Four-case bounded GPU pilot; not accepted as calibration. |
| Nonlinear truth diagnostics | `outputs/stage0_nonlinear_validation/truth_diagnostics_after_unitfix.csv` | Yes | Deterministic PyAutoLens truth-tracer checks. |
| SPIE-plus validation CSV | `outputs/stage0_spie_plus_validation/results.csv` | Yes | Local nonlinear optimizer verification plus false-positive case. |
| SPIE-plus validation plot | `outputs/stage0_spie_plus_validation/q_f_vs_q_fit.png` | Yes | Fisher versus local optimizer statistic for injected-subhalo cases. |
| Full profile calibration CSV | `outputs/stage0_profile_calibration_full/results.csv` | Yes | Full `21`-case local-profile calibration; `0 / 21` failures. |
| Full SPIE-plus validation CSV | `outputs/stage0_spie_plus_validation_full/results.csv` | Yes | Full `21` injected cases plus `2` false-positive controls; `0 / 23` failures. |
| Full SPIE-plus validation plot | `outputs/stage0_spie_plus_validation_full/q_f_vs_q_fit.png` | Yes | Fisher versus local optimizer statistic for all injected Stage 2 cases. |
| Forecast robustness outputs | `outputs/stage0_forecast_robustness/` | Yes | Noisy ensembles, false-positive controls, and ring-position variation checks. |
| Figures | `outputs/stage0_internal_review/figures/` | Yes | Aggregate Stage 0 figures, including segment-hexike and global-Zernike sweeps. |
| Reproducibility summary | `outputs/stage0_internal_review/study_provenance.json` | Yes | Includes command, git hash, Python, package versions. |

## Open questions

1. Which Stage 0 figures should be promoted into a poster/manuscript figure
   script with final styling?
2. Should the next sweep vary segment IDs/modes, or promote global-Zernike
   variants beyond Noll 4 into full amplitude ladders?
