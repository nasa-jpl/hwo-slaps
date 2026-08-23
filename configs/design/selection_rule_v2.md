# D-F4 golden-selection rule, v2: RULED and PRE-REGISTERED

Signed: George, 2026-08-23.

SUPERSEDES the v1 document of the same path (signed 2026-08-23,
`score = z(log S) + z(log G)`). v1 is void: the Sol Pro go-time
sign-off finding P0-6 showed that `G` scales approximately as `S^2`
under both background-dominated and source-dominated flux scaling, so
the v1 score counted brightness twice and did not isolate the
morphology information this study exploits. v2 adopts the recommended
methodology exactly: a brightness-normalized complexity statistic, a
pre-registered three-curve comparison, and a noise-seed rank-stability
test.

This document is the G-B artifact and the signed definition. The
implementation is `src/hwoslaps/analysis/selection_score.py` with
hand-calculable unit tests in `tests/test_selection_score.py`, and the
driver that produces the curves and the stability test is
`scratch/panel/selection_stability.py`. Implementation and document
must agree exactly; where they ever disagree, this document is the
authority and the code is the bug.

## Pre-registration statement

The score defined in section 2 is FROZEN BY THIS SIGNATURE, BEFORE any
injected-subhalo `M_lim` value exists for any Stage 0 pool member. No
`M_lim` measurement, no sensitivity ladder and no Fisher map informed
the choice of statistics, their functional form, the standardization,
the floor cuts, the tie rule or the tier sizes. The oracle curve of
section 3 can only be formed after the ladders are measured, and it is
reported as a labelled upper bound, never as an operational selector.
This restates the Sol Pro W2/W3 ruling, verbatim: "Do not select the
primary 'golden' sample after inspecting its true injected-subhalo
M_lim."

## 1. The observables

All signal quantities are electrons in the exposure and all variances
are electrons squared. Sums run over the pixels of the D-F7 aperture,
`R <= 2 theta_E`, centred on the lens, radius closed.

Blank-pixel variance, the source-free limit of the engine noise map:

    B = (sky_background + dark_current) * exposure_time + read_noise^2

with `read_noise` the effective combined-image value, because the noise
model applies exactly one squared read-noise term.

Expected per-pixel variance of the noiseless source map `s`:

    sigma_i^2 = s_i + B                          (no noise realization)

Integrated arc signal-to-noise, the engine's validated convention (C5),
on the PSF-convolved noiseless lensed-source electrons:

    S = sqrt( sum_i s_i^2 / sigma_i^2 )          (dimensionless)

Noise-weighted gradient power, central-difference gradients on the same
PSF-convolved image:

    G = sum_i |grad s|_i^2 / sigma_i^2           (arcsec^-2)

The gradient is an ANGULAR derivative: differences are divided by the
pixel scale, so `|grad s|` carries `e- arcsec^-1` and `G` carries
`arcsec^-2`. The stencil is `numpy.gradient` with the pixel scale as
the spacing on both axes: second-order central differences in the array
interior and first-order one-sided differences on the array border.
The estimator is OURS by construction (declared); the physics it
measures is literature-verified (section 5).

Diffraction scale of the delivered instrument:

    theta_res = lambda / D                       (arcsec)

Brightness-normalized complexity:

    C = theta_res^2 * G / S^2                    (dimensionless)

`C` is dimensionless by construction and, in the background-dominated
limit where the variance does not follow the source, invariant under a
uniform rescaling of the arc flux: `S` scales linearly and `G`
quadratically, so the brightness cancels. That invariance is the whole
purpose of the statistic and it is a unit test.

## 2. The rule

Stage 1, floor cuts, verbatim Collett 2015 / Euclid practice:

    theta_E > 0.5 arcsec   AND   S > 20

Both cuts are STRICT: a member exactly on a floor fails it.

Stage 2, rank the survivors by

    score = z(log S) + z(log C)

where `z` standardizes with the POPULATION standard deviation
(`ddof = 0`) over the post-floor-cut pool, which is exactly the set of
members being ranked. A pool with zero spread in a statistic
standardizes to zeros on that statistic, so identical members cannot
rank above one another.

Ranking is by descending score. Ties break on the ASCENDING sha256 hex
digest of the system id. The digest depends on neither pool membership
nor input order, so a re-run or a re-ordered pool reproduces the
identical ranking.

Top 12 of the ranking is the selected tier; top 5 of those are the
goldens.

Zero and non-finite handling: `log S` and `log C` require strictly
positive finite inputs. A survivor whose `S` or `C` is zero, negative
or non-finite REJECTS THE POOL LOUDLY rather than being reordered or
silently dropped. `C = 0` means a perfectly flat arc inside the
aperture, which is not a physical member of a lensed-source pool, so it
is treated as an upstream failure.

Freeze procedure, unchanged from v1 except for the score definition:

1. This document is the signed definition (George, 2026-08-23).
2. Implemented with hand-calculable unit tests, including a toy image
   where `S`, `G`, `C`, the cuts, the standardization, the tie rule and
   the stability metrics are verified by hand.
3. Stage 0 runs; observables, scores and the ranked list are computed
   and hashed into provenance BEFORE the manifest generator emits any
   injected-subhalo job. The rank-list hash joins the launch seed list
   George signs.

DROPPED from the score, carried over from v1 and still ruled out:
lens/source contrast, incoherent to validate under the D-F6
ceiling-only physics, since our `M_lim` has zero lens-light dependence,
so the term could only add noise to the rank validation; and
magnification, redundant with `S` and `C` (it acts through delivered
photons and stretched gradients, both measured directly; W2 measured
the mu-lever compressing; mu is not strictly a pre-modeling
observable). Both are RECORDED as descriptive columns per pool member,
with `theta_E` and source magnitude, so the selected tier's properties
are fully displayed; contrast additionally gets a limitations sentence
quoting the W4 bracket.

Also dropped: the v1 score term `z(log G)`. It is superseded by
`z(log C)` and must not appear in any reported ranking.

## 3. Pre-registered three-curve comparison

Exactly three rankings are computed and reported side by side. No
fourth curve is introduced after the fact, and no curve is dropped
because it performed badly.

1. `s_only`: rank by `z(log S)`. The brightness-only control. This is
   the ranking a survey would produce from signal-to-noise alone, and
   it is the null hypothesis the complexity term must beat.
2. `s_plus_c`: rank by `z(log S) + z(log C)`. The frozen operational
   score of section 2. This is the rule the campaign selects on.
3. `oracle`: rank by measured sensitivity, ascending `log10(M_lim)`,
   lowest detectable mass first. Formed ONLY after the injected-subhalo
   ladders are measured. Reported as a labelled upper bound and
   explicitly distinguished from operational selection (Sol Pro
   sanction).

Reported for the comparison, over the mapped systems:

- Spearman rank correlation of each score against measured `M_lim`,
  pooled and PER MORPHOLOGY. The five-anchor template bank makes the
  per-morphology split meaningful; W4 established that no single scalar
  transfers between morphologies, which is why the joint-pool number is
  reported alongside and not instead. A working score correlates
  NEGATIVELY with `log10 M_lim`: higher score, lower detectable mass.
- Oracle-recovered fraction: the share of the oracle top-12 that each
  operational ranking's top-12 recovers.
- Top-12 Jaccard index between the operational rankings, and between
  each operational ranking and the oracle.

The success claim `s_plus_c` must earn is a stronger rank correlation
with measured `M_lim`, and a higher oracle-recovered fraction, than
`s_only`. If it does not, that is reported as the result: the selection
adds nothing beyond arc signal-to-noise, and the paper says so.

## 4. Noise-seed rank-stability test

P0-6, verbatim: "The current S/G statistics are computed from noiseless
source-only truth. Unless rank stability is demonstrated using
noisy/reconstructed observables, call this an idealized no-subhalo
proxy selection, not an operational Roman/Euclid/HWO target selector."

The test, pre-registered here:

- For each declared noise seed, the production detector model
  (`hwoslaps.observation.noise_models.apply_detector_noise`) generates
  one realization of every Stage 0 member.
- The observables are recomputed by the IDENTICAL estimators on the
  background-subtracted realization: signal is realization minus the
  known mean sky and dark, and the variance is
  `max(signal, 0) + B`. No truth enters the noisy path, so the test
  measures what an observer could actually rank on.
- Floor cuts are re-applied per seed on that seed's own `S`, so a
  member may leave the pool under noise. The seed and reference
  rankings are compared on the intersection of their survivor sets, and
  the intersection size is reported.
- Reported per seed against the noiseless reference ranking, and
  pairwise between seeds: Spearman rank correlation of ranking
  positions, top-12 Jaccard, and, once the ladders exist,
  oracle-recovered fraction.

Pre-registered interpretation, fixed before the numbers are seen:

- If the noisy rankings are stable (high Spearman, high top-12
  Jaccard), the selection is reported as an operational target
  selector.
- If they are not, the selection is reported as an IDEALIZED
  NO-SUBHALO PROXY SELECTION, in exactly those words, and every claim
  about transferring the rule to Roman, Euclid or HWO target lists is
  withdrawn. The campaign's own use of the rule is unaffected, because
  the Stage 0 pool is simulated and its noiseless truth is available by
  construction.

## 5. Literature grounding (verified quotes, carried from v1)

- Preselection concept, O'Riordan et al. 2023 (arXiv:2211.15679),
  verbatim: "Pre-selecting the most sensitive systems from a large
  sample which are mostly poor in sensitivity should drastically
  improve the constraints in a gravitational imaging study where
  analysis time is limited." Yield contrast 1-per-70 vs 1-per-3 at
  f_sub = 0.01; ~2500 detections forecast from 170,000 Euclid lenses.
- Floor cuts, O'Riordan verbatim: "We impose the same cuts as
  [Collett 2015], namely, lenses must have an Einstein radius
  theta_E > 0.5 arcsec and a total signal to noise ratio S/N > 20."
- S: Despali et al. 2022 (arXiv:2111.08718) eq. 6, M_low linear in
  S/N: Delta log M_low = 1.5(+/-0.1) - 0.725(+/-0.12) (SNR/SNR0);
  O'Riordan: sensitivity "primarily a function of the instrument
  angular resolution and signal to noise ratio"; Vegetti et al.
  2014 (arXiv:1405.3666) selected the real gravitational-imaging
  sample by arc-pixel S/N thresholds (>=200 px at S/N>=2, >=50 at
  S/N>=3).
- G: Despali verbatim: sensitivity tracks "the gradient of the
  source galaxy light — the larger the complexity of the source
  light, the lower the detectable mass"; O'Riordan verbatim:
  "vital to use complex source brightness distributions" (with the
  source-absorption degeneracy mechanism); our Panel C / Gate C
  measurements (12.6-20x at matched arc S/N).
- theta_res in C: O'Riordan verbatim, sensitivity is "primarily a
  function of the instrument angular resolution and signal to noise
  ratio", which is why the complexity statistic is stated per
  resolution element rather than per arcsecond.
- Pre-registration: Sol Pro W2/W3 ruling verbatim: "Do not select
  the primary 'golden' sample after inspecting its true
  injected-subhalo M_lim."

## 6. Provenance of this revision

- Finding: P0-6, `scratch/SolPro_gotime_signoff/`,
  `HWO_SLAPS_go_time_final_signoff_2026-08-23.md`.
- Ruling: George, 2026-08-23, adopting the recommended methodology
  exactly, logged in `scratch/FableRASTISummary.md` and in
  `scratch/OvernightExecutionPlan_2026-08-23.md` (ruling 2, task T4).
- Every number produced under this rule is provisional until the
  morning GPT Pro review.
