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
- [x] Full PyAutoLens local-search nonlinear evidence validation completed for
  the controlled SPIE grid.
- [x] Matched-PSF and wrong-PSF no-subhalo control grids completed.
- [x] `n_live=800` convergence subset completed.
- [x] Compact noisy PyAutoLens local-search pilot completed.
- [x] Full noisy PyAutoLens local-search validation completed.
- [x] High-`n_live` noisy disagreement check completed.
- [x] Discrete PSF-bank marginalized nonlinear validation completed.
- [x] Amplitude-matched `1e7 Msun` PSF-bank no-subhalo controls completed.
- [x] Five-point PSF-bank marginalized mass-completeness curve completed.

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
| 2026-05-24 | SPIE nonlinear validation scope | Initial four-case fixed-template GPU pilot used to debug units/model setup; superseded by full PyAutoLens local-search evidence validation. | Superseded |
| 2026-05-25 | Optional second SPIE PSF family | Add a global Zernike Noll 4 amplitude ladder and include global Zernikes Noll 4-6 in the selected mode-coupling scan basis. | Done |
| 2026-05-29 | Full PyAutoLens local-search evidence validation | Completed full matched-PSF grid, full wrong-PSF controls, `n_live=800` convergence subset, and compact noisy pilot. | Done |
| 2026-05-29 | Literature-grade noisy PyAutoLens validation | Completed noisy injected-subhalo grid, noisy matched controls, noisy wrong-PSF controls, and high-`n_live` disagreement reruns. | Done |
| 2026-05-30 | Discrete PSF-bank marginalized validation | Added Bayesian evidence marginalization over a four-member PSF bank for both smooth and subhalo hypotheses. | Done |
| 2026-05-31 | Amplitude-matched `1e7 Msun` PSF-bank controls | Extended the PSF-bank marginalized `1e7 Msun` injected curve with `289` matched no-subhalo controls. | Done |
| 2026-06-02 | PSF-bank marginalized mass-completeness curve | Extended PSF-bank marginalized nonlinear validation across `10^6.5`, `10^6.75`, `10^7`, `10^7.25`, and `10^7.5 Msun`. | Done |

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
- The maintained SPIE modeling route is Fisher/Asimov screening calibrated by
  PyAutoLens local-search nonlinear evidence for the controlled validation
  grid.
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

PyAutoLens nonlinear evidence validation:

Run package: `outputs/spie_draft_results/`.
Alias: `outputs/spie_draft_study/`.

Definition: compare smooth-lens and subhalo-lens PyAutoLens models with
Nautilus evidence and maximum-likelihood summaries. The current accepted SPIE
validation uses `fit_mode = local_search`, where the subhalo center has a local
prior window around the forecast position. This supersedes the earlier
fixed-template and local-optimizer validation notes for poster/manuscript
purposes.

Core run settings:

- `dataset_kind = asimov` for the full validation grid.
- `n_live_smooth = 400`, `n_live_subhalo = 400` for the full grid.
- `n_live_smooth = 800`, `n_live_subhalo = 800` for the convergence subset.
- `use_jax = true`.
- `fast_output = true`.
- GPU 0.

Full matched-PSF local-search validation:

| Set | N | Success | Injected N | Injected detections | Matched-control N | Matched false positives |
|---|---:|---:|---:|---:|---:|---:|
| Full matched-PSF PyAutoLens local search | `433` | `433` | `370` | `211` | `63` | `0` |

Injected-subhalo family calibration:

| Family | N | Nonlinear detections | Fisher detections | Median `q_fit` | Median `q_F` |
|---|---:|---:|---:|---:|---:|
| Combined | `120` | `69` | `72` | `14.36` | `14.91` |
| Global-only | `118` | `70` | `70` | `16.14` | `16.64` |
| Perfect | `8` | `5` | `5` | `27.55` | `27.96` |
| Segment-only | `124` | `67` | `68` | `11.70` | `12.16` |

Interpretation:

- Fisher and PyAutoLens local-search detection counts agree closely by PSF
  family in the controlled matched-PSF validation grid.
- Matched no-subhalo controls produce `0/63` false positives.
- This supports using `q_F` as a calibrated screening forecast in the controlled
  matched-PSF SPIE setup.

Wrong-PSF no-subhalo controls:

| Set | N | Success | `q_fit >= 10` false positives | `Delta logZ > 5` false positives |
|---|---:|---:|---:|---:|
| Wrong/perfect-PSF local-search controls | `63` | `63` | `49` | `47` |

Wrong-PSF family breakdown by `q_fit >= 10`:

| Truth PSF family | N | False positives | Rate |
|---|---:|---:|---:|
| Combined | `21` | `16` | `76.2%` |
| Global-only | `21` | `14` | `66.7%` |
| Segment-only | `21` | `19` | `90.5%` |

Interpretation:

- PSF mismatch is the dominant nonlinear false-positive pathway in the current
  validation.
- The matched/wrong contrast isolates the failure mode: the same no-subhalo
  cases stay clean when the PSF is modeled correctly and become false positives
  when the fit assumes a perfect PSF.

Convergence subset:

| Set | N | Result |
|---|---:|---|
| Matched `n_live=800` controls | `2` | `0/2` false positives |
| Wrong-PSF `n_live=800` controls | `2` | `2/2` false positives |
| Matched `n_live=800` injected cases | `4` | `2/4` detected |

The convergence subset preserves the same matched-clean / wrong-PSF-false-positive
contrast as the `n_live=400` grid.

Noisy PyAutoLens pilot:

| Pilot set | Fit PSF | Case type | N | Main result |
|---|---|---|---:|---|
| Original representative | Matched/truth PSF | Injected subhalos | `4` | `3/4` recovered, matching Fisher's `3/4`. |
| Original representative | Matched/truth PSF | No-subhalo controls | `2` | `0/2` false positives. |
| Original representative | Wrong/perfect PSF | No-subhalo controls | `2` | `2/2` false positives. |
| Refined 75 nm controls | Matched/truth PSF | No-subhalo controls | `6` | `0/6` false positives. |
| Refined 75 nm controls | Wrong/perfect PSF | No-subhalo controls | `6` | `3/6` false positives. |

The noisy pilot is not a full noisy nonlinear ensemble, but it shows that the
central matched-clean / wrong-PSF-false-positive result is not purely an Asimov
artifact.

Full noisy PyAutoLens validation:

| Set | N | Success | Main result |
|---|---:|---:|---|
| Noisy injected-subhalo local search | `370` | `370` | `217/370` detections by `q_fit >= 10`; Fisher predicted `215/370`; Spearman `rho(q_F, q_fit) = 0.918`. |
| Noisy matched-PSF no-subhalo controls | `63` | `63` | `0/63` false positives by both `q_fit >= 10` and `Delta logZ > 5`. |
| Noisy wrong/perfect-PSF no-subhalo controls | `63` | `63` | `48/63` false positives by `q_fit >= 10`; `47/63` by `Delta logZ > 5`. |
| High-`n_live` noisy disagreement reruns | `24` | `24` | Largest Fisher/nonlinear threshold disagreements persisted; median absolute changes were `0.292` in `q_fit` and `0.030` in `Delta logZ`. |

Interpretation:

- The full noisy local-search ensemble preserves the central matched-clean /
  wrong-PSF-false-positive result from the Asimov grid.
- Fisher and noisy nonlinear detection counts agree at the ensemble level, but
  individual near-threshold cases can scatter across the threshold.
- The high-`n_live` reruns show that the largest threshold disagreements are not
  explained by low-live-point search noise.

PSF-bank marginalized validation:

Definition: compare the smooth-lens and subhalo-lens PyAutoLens hypotheses
after marginalizing each model evidence over a discrete PSF bank. The current
bank uses PSF scale factors `0.0`, `0.5`, `1.0`, and `1.5` with equal weights.
This is not a continuous PSF posterior, but it is the current most realistic
poster-facing nonlinear nuisance treatment.

| Set | N | Success | `q_fit_psf_profile >= 10` | `Delta logZ_psf_marg > 5` | Notes |
|---|---:|---:|---:|---:|---|
| Initial PSF-bank injected subset at `1e7 Msun` | `24` | `24` | `12/24` | `11/24` | Direct paired injected subset. |
| Amplitude-matched `1e7 Msun` injected ensemble | `289` | `289` | `222/289` | `209/289` | One perfect reference plus `3` PSF families, `8` nonzero amplitudes, `12` draws. |
| Amplitude-matched `1e7 Msun` no-subhalo controls | `289` | `289` | `0/289` | `0/289` | Largest control response remains sub-threshold: max `q_fit_psf_profile = 6.353`, max `Delta logZ_psf_marg = 2.322`. |

PSF-bank marginalized mass-completeness validation:

The 2026-06-02 run extended the PSF-bank marginalized injected-subhalo ensemble
from the single `1e7 Msun` mass to a five-point mass-completeness curve. The new
run added `1156/1156` successful fits across four additional masses, using the
same `289`-state PSF ensemble structure per mass. Combined with the existing
`1e7 Msun` ensemble:

| Injected subhalo mass | Cases | `q_fit_psf_profile >= 10` | `Delta logZ_psf_marg > 5` |
|---:|---:|---:|---:|
| `3.16e6 Msun` | `289` | `15/289 = 5.2%` | `8/289 = 2.8%` |
| `5.62e6 Msun` | `289` | `110/289 = 38.1%` | `87/289 = 30.1%` |
| `1.00e7 Msun` | `289` | `222/289 = 76.8%` | `209/289 = 72.3%` |
| `1.78e7 Msun` | `289` | `284/289 = 98.3%` | `281/289 = 97.2%` |
| `3.16e7 Msun` | `289` | `289/289 = 100.0%` | `289/289 = 100.0%` |

Interpretation:

- In the canonical scene with discrete PSF-bank marginalization, `3e6 Msun`
  subhalos are generally not recovered.
- `5-10e6 Msun` spans the nonlinear transition region.
- `1e7 Msun` is usually, but not universally, recovered under PSF uncertainty.
- `>= 1.8e7 Msun` is near-saturated in this setup.
- This is the current cleanest nonlinear detection-limit result, but remains
  conditional on the current lens/source/noise setup and the discrete PSF-bank
  nuisance model.

Primary validation artifacts:

| Artifact | Path | Notes |
|---|---|---|
| Study package | `outputs/spie_draft_results/README.md` | Full written synthesis and conclusion set. |
| Study package alias | `outputs/spie_draft_study/` | Symlink to `outputs/spie_draft_results/`. |
| Full local-search cases | `outputs/spie_draft_results/csv/overnight_local_search_all_cases.csv` | `504` rows including A/B/C. |
| Full local-search phase summary | `outputs/spie_draft_results/csv/overnight_local_search_phase_summary.csv` | Matched, wrong-PSF, and convergence summaries. |
| Full local-search family summary | `outputs/spie_draft_results/csv/overnight_local_search_family_summary.csv` | Family-level Fisher/PyAutoLens comparison. |
| Full false-positive summary | `outputs/spie_draft_results/csv/overnight_local_search_false_positive_summary.csv` | Matched versus wrong-PSF controls. |
| Noisy pilot cases | `outputs/spie_draft_results/csv/noisy_pyautolens_local_search_pilot_all_cases.csv` | `20/20` successful noisy fits. |
| Noisy pilot summary | `outputs/spie_draft_results/csv/noisy_pyautolens_local_search_pilot_summary.csv` | Matched/wrong noisy control summary. |
| Full noisy validation overview | `outputs/spie_draft_results/csv/literature_grade_noisy_validation_overview.csv` | Noisy injected, matched-control, wrong-PSF-control, and high-`n_live` summaries. |
| Full noisy injected cases | `outputs/spie_draft_results/csv/litgrade_noisy_injected_all_cases.csv` | `370/370` successful noisy injected-subhalo local-search fits. |
| Noisy control summary | `outputs/spie_draft_results/csv/litgrade_noisy_controls_phase_summary.csv` | Matched noisy controls clean; wrong-PSF noisy controls false-positive prone. |
| High-`n_live` disagreement summary | `outputs/spie_draft_results/csv/litgrade_noisy_nlive800_disagreement_summary.csv` | Largest threshold disagreements rerun at `n_live=800`. |
| PSF-bank mass-completeness cases | `outputs/spie_draft_results/csv/psf_marginalized_mass_completeness_all_cases.csv` | Five-mass PSF-bank marginalized injected-subhalo validation table. |
| PSF-bank mass-completeness summary | `outputs/spie_draft_results/csv/psf_marginalized_mass_completeness_by_mass.csv` | Detection completeness by subhalo mass. |
| PSF-bank mass-completeness family summary | `outputs/spie_draft_results/csv/psf_marginalized_mass_completeness_by_mass_family.csv` | Detection completeness split by mass and PSF family. |
| PSF-bank mass-completeness metadata | `outputs/spie_draft_results/metadata/psf_marginalized_mass_completeness_summary.json` | Run summary and provenance for the mass-completeness package. |
| Fisher/PyAutoLens plot | `outputs/spie_draft_results/plots/overnight_local_search_fisher_vs_qfit.png` | Full matched-PSF injected calibration. |
| False-positive rates plot | `outputs/spie_draft_results/plots/overnight_local_search_false_positive_rates.png` | Matched versus wrong-PSF controls. |
| Noisy control rates plot | `outputs/spie_draft_results/plots/noisy_pyautolens_local_search_control_rates.png` | Noisy pilot false-positive rates. |
| Full noisy Fisher/PyAutoLens plot | `outputs/spie_draft_results/plots/litgrade_noisy_injected_fisher_vs_qfit.png` | Full noisy injected-subhalo Fisher versus nonlinear comparison. |
| PSF-bank mass-completeness curve | `outputs/spie_draft_results/plots/psf_marginalized_mass_completeness_detection_curve.png` | Current strongest nonlinear mass-reach plot. |
| PSF-bank mass-completeness statistics | `outputs/spie_draft_results/plots/psf_marginalized_mass_completeness_statistics.png` | Distribution summary by mass. |
| PSF-bank mass-completeness by family | `outputs/spie_draft_results/plots/psf_marginalized_mass_completeness_by_family.png` | Completeness split by PSF family. |

Legacy forecast robustness checks:

Run source: `outputs/stage0_forecast_robustness/`.

Scope: lightweight pre-evidence forecast checks added before the full
PyAutoLens validation grid was available. These are retained as historical
supporting diagnostics, but they are no longer the main validation basis. The
current validation basis is the PyAutoLens local-search evidence grid described
above, including the compact noisy PyAutoLens pilot.

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

Legacy deterministic truth-tracer checks:

| Case | PSF case | q_F profiled | q_truth PyAutoLens | q_truth / q_F | Status | Notes |
|---|---|---:|---:|---:|---|---|
| `perfect_m1e7_near_threshold` | perfect | `17.6703` | `55.1764` | `3.1225` | Diagnostic pass | Matches Fisher raw `55.1902`. |
| `perfect_m10p7p25_moderate` | perfect | `38.2564` | `136.5977` | `3.5706` | Diagnostic pass | Truth model beats smooth model at fixed parameters. |
| `perfect_m10p7p75_high` | perfect | `165.9551` | `819.1794` | `4.9362` | Diagnostic pass | Truth model beats smooth model at fixed parameters. |
| `hexike100_m1e7_endpoint` | segment hexike | `16.8690` | `53.3598` | `3.1632` | Diagnostic pass | Truth model beats smooth model at fixed parameters. |

Current calibration summary:

- Current accepted calibration basis: PyAutoLens local-search nonlinear
  evidence for the controlled SPIE grid, with the PSF-bank marginalized
  evidence curve as the strongest current mass-reach validation.
- Full matched-PSF injected validation: `211/370` nonlinear detections, with
  family-level detection counts closely matching Fisher counts.
- Full matched-PSF no-subhalo controls: `0/63` false positives.
- Full wrong-PSF no-subhalo controls: `49/63` false positives by `q_fit >= 10`,
  with `47/63` also exceeding `Delta logZ > 5`.
- `n_live=800` convergence subset: matched controls remain clean (`0/2`) and
  wrong-PSF controls remain false positives (`2/2`).
- Noisy PyAutoLens pilot: matched noisy controls remain clean (`0/8`) and
  wrong-PSF noisy controls produce false positives (`5/8`).
- Full noisy PyAutoLens local-search validation: noisy injected detections
  `217/370` versus Fisher predictions `215/370`, matched noisy controls
  `0/63`, and wrong-PSF noisy controls `48/63` false positives by `q_fit >= 10`.
- PSF-bank marginalized `1e7 Msun` controls: matched no-subhalo controls
  `0/289`, paired injected detections `222/289` by `q_fit_psf_profile >= 10`
  and `209/289` by `Delta logZ_psf_marg > 5`.
- PSF-bank marginalized mass completeness: `2.8%`, `30.1%`, `72.3%`, `97.2%`,
  and `100.0%` strong-evidence completeness at `3.16e6`, `5.62e6`, `1.0e7`,
  `1.78e7`, and `3.16e7 Msun`, respectively.
- Current claim boundary: Fisher/Asimov is calibrated as a screening forecast
  for the controlled matched-PSF SPIE setup. PSF mismatch can mimic subhalo
  evidence if not modeled. The current discrete PSF-bank marginalization is a
  stronger nuisance treatment, but final requirement language still requires
  broader scenes, source realism, full 2D maps, larger noisy ensembles, and
  continuous or otherwise expanded PSF-nuisance treatment.

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
- [x] PyAutoLens Fisher-versus-nonlinear evidence validation plot.
- [x] Matched versus wrong-PSF false-positive plot.
- [x] Noisy PyAutoLens pilot plot.
- [x] Full noisy PyAutoLens validation plot.
- [x] PSF-bank marginalized mass-completeness plot.

Manuscript/poster notes:

- Use preliminary framework language.
- Avoid final engineering requirement claims.
- State Fisher/Asimov limitations clearly.
- State that full PyAutoLens local-search evidence validates the controlled
  matched-PSF SPIE grid and isolates PSF mismatch as the dominant false-positive
  pathway.
- State that the noisy PyAutoLens pilot is a compact robustness check, not a
  full noisy nonlinear recovery campaign.
- State that the later full noisy PyAutoLens grid supersedes the pilot for the
  main validation claim, while the pilot remains a historical/debugging check.
- State that the PSF-bank marginalized mass-completeness curve is the strongest
  current nonlinear mass-reach result for the canonical scene.
- State PSF amplitude units in every relevant figure caption.

## Stage 4-6: RASTI expansion

RASTI work starts after the SPIE baseline is reproducible.

Deferred items:

- [ ] Supported analysis module.
- [ ] Full study manifest and aggregator.
- [ ] Full 2D detectable-area maps.
- [ ] Requirement-curve generation.
- [ ] Broader nonlinear calibration grid across scenes, sources, positions, and
  PSF states.
- [ ] Continuous or broader PSF-nuisance marginalized false-positive study
  beyond the current discrete PSF-bank canonical-scene validation.
- [ ] Source-realism stress tests.
- [ ] Lens-light and subtraction-residual stress tests.

## Artifact index

| Artifact type | Path | Created | Notes |
|---|---|---|---|
| Canonical config | `configs/study/scdd_spie_baseline.yaml` | Yes | Stage 0 SCDD/SPIE baseline. |
| Manifest | `scratch/study/stage0_manifest.yaml` | Yes | Stage 0 mass and PSF sweeps. |
| Generated run configs | `outputs/stage0_internal_review/generated_configs/` | Yes | One config per run. |
| Aggregate results CSV | `outputs/stage0_internal_review/results.csv` | Yes | `21` successful rows. |
| SPIE draft study package | `outputs/spie_draft_results/` | Yes | Current aggregate study package with Fisher, PyAutoLens evidence, false-positive, convergence, and noisy-pilot summaries. |
| SPIE draft study alias | `outputs/spie_draft_study/` | Yes | Symlink to `outputs/spie_draft_results/`. |
| Full PyAutoLens local-search CSV | `outputs/spie_draft_results/csv/overnight_local_search_all_cases.csv` | Yes | `504` successful A/B/C validation rows. |
| PyAutoLens phase summary | `outputs/spie_draft_results/csv/overnight_local_search_phase_summary.csv` | Yes | Matched full grid, wrong-PSF controls, and `n_live=800` convergence subset. |
| PyAutoLens family summary | `outputs/spie_draft_results/csv/overnight_local_search_family_summary.csv` | Yes | Family-level Fisher/PyAutoLens detection agreement. |
| PyAutoLens false-positive summary | `outputs/spie_draft_results/csv/overnight_local_search_false_positive_summary.csv` | Yes | Matched controls `0/63`; wrong-PSF controls `49/63`. |
| Noisy PyAutoLens pilot CSV | `outputs/spie_draft_results/csv/noisy_pyautolens_local_search_pilot_all_cases.csv` | Yes | `20/20` successful noisy local-search fits. |
| Literature-grade noisy validation CSVs | `outputs/spie_draft_results/csv/literature_grade_noisy_validation_overview.csv` | Yes | Full noisy injected, control, and high-`n_live` summaries. |
| PSF-bank mass-completeness CSVs | `outputs/spie_draft_results/csv/psf_marginalized_mass_completeness_by_mass.csv` | Yes | Five-point nonlinear mass-completeness curve with discrete PSF-bank marginalization. |
| Full PyAutoLens calibration plot | `outputs/spie_draft_results/plots/overnight_local_search_fisher_vs_qfit.png` | Yes | Fisher versus PyAutoLens `q_fit` for full matched injected grid. |
| Full false-positive plot | `outputs/spie_draft_results/plots/overnight_local_search_false_positive_rates.png` | Yes | Matched versus wrong-PSF false-positive rates. |
| Noisy PyAutoLens plot | `outputs/spie_draft_results/plots/noisy_pyautolens_local_search_control_rates.png` | Yes | Noisy matched/wrong control rates. |
| PSF-bank mass-completeness plots | `outputs/spie_draft_results/plots/psf_marginalized_mass_completeness_detection_curve.png` | Yes | Current strongest nonlinear mass-reach plot. |
| Forecast robustness outputs | `outputs/stage0_forecast_robustness/` | Yes | Noisy ensembles, false-positive controls, and ring-position variation checks. |
| Figures | `outputs/stage0_internal_review/figures/` | Yes | Aggregate Stage 0 figures, including segment-hexike and global-Zernike sweeps. |
| Reproducibility summary | `outputs/stage0_internal_review/study_provenance.json` | Yes | Includes command, git hash, Python, package versions. |

## Open questions

1. Which PSF-bank mass-completeness plot should be central in the manuscript
   versus backup material?
2. Should the poster/manuscript lead with the mass-completeness curve, the
   matched/wrong-PSF false-positive result, or the Fisher PSF-degradation curve?
3. Should the main PSF-degradation x-axis be nominal amplitude, measured WFE,
   or a paired presentation?
4. Which RASTI expansion comes first: PSF nuisance fitting, full 2D sensitivity
   maps, or source-realism stress tests?
