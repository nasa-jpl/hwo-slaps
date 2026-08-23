# Source-magnitude anchors: typical and golden (G-A artifact, 2026-08-23)

STATUS: RULED. George confirmed both anchors on 2026-08-23 ("standard
should be medians"; golden anchor 23.795 V accepted). The committed
byte-identical copy of this document is
configs/design/golden_magnitude_anchor.md and is bound by
DesignFreeze v1 (amended).

Purpose: replace the retired ad-hoc golden magnitude 23.345 V (a
panel-driver constant, never ruled, 0.40 mag brighter than the parent's
own bright bound) with literature-backed anchors for (a) the typical
source and (b) the golden brightness canary. Everything below is
computed from Newton et al. 2011 (arXiv:1104.2608, SLACS XI) Table 1,
extracted verbatim from the paper PDF on 2026-08-23, and from Vegetti
et al. 2014 (arXiv:1405.3666) Table 1.

## The data

Newton et al. 2011 Table 1 gives magnification-corrected (unlensed,
source-plane Sersic-fit) AB magnitudes for 46 SLACS sources, with
per-source magnifications. Quoted photometry uncertainty 0.3 mag.
Full-sample I814 statistics: min 21.05, p10 22.83, p25 23.50, median
24.13, max 27.00; magnification median 8.6.

Restricting to the z_s window 0.4-0.8 that brackets our design point
z_s = 0.6 (n = 34): min 22.33, p05 22.84, p10 23.25, p25 23.56,
median 24.20.

The D1 color chain (declared assumption, unchanged): F814W -> V at
500 nm adds +0.545 mag for a blue star-forming source.

## Anchor 1: the TYPICAL source (standard/central)

RULED (unchanged). D1 = 24.845 V stands as the SLACS median
through the declared color (24.3 F814W + 0.545). The exact table
medians are 24.13 (full 46) and 24.20 (z_s window); the ruled 24.3 sits
within the 0.3 mag per-source photometric uncertainty of either, and
moving D1 would re-derive the reference, every asset contract, and
every canary for a 0.04 dex M_lim effect. Add one provenance sentence
to the reference YAML note quoting the exact table statistics.

Corroboration: Roman detected-lens forecast mean source F129
23.1 +/- 1.1 at z_s ~ 1.5 (arXiv:2506.03390; redder band, higher
redshift); O'Riordan et al. 2023 Euclid evaluation uses a total
S/N > 20 cut on sources drawn 20-26 M_VIS (a S/N selection, not a
magnitude anchor); Despali et al. 2022 varies exposure/SNR, not source
magnitude, and offers no anchor.

## Anchor 2: the GOLDEN brightness canary

RULED: golden = 23.795 V, defined as the brightest-decile
(p10) magnitude of the measured z_s-matched SLACS source sample
(23.25 F814W, Newton Table 1) through the same D1 color. One sentence
for the paper: "the golden-source canary adopts the brightest-decile
source magnitude of the SLACS z_s ~ 0.4-0.8 sample". Properties:

- It is a measured quantile of real data, not a fitted-distribution
  quantile (the parent TruncNormal p10 is 24.101 V; the difference is
  the parent's spread calibration, and quoting the measured value
  avoids defending the fit).
- It sits INSIDE the declared parent (0.05 mag fainter than the
  parent bright bound 23.745 V = SLACS single brightest through D1),
  so the canary is in-population; nothing is quoted outside the
  declared distribution.
- Alternatives considered: parent bright bound 23.745 V (single
  brightest of 46; an extremum of a 46-object sample with 0.3 mag
  photometry errors is weaker than a decile); parent-fit p10 24.101 V
  (defensible but one step removed from data); Roman 23.1 F129
  (cross-band, cross-redshift; corroboration only).

Headline consequence (0.39 dex/mag lever on the current frontier,
R2 = 6.962 at the retired 23.345): golden canary at 23.795 moves the
R2 crossing to approximately 7.14. Exact values come from re-running
the R-ladder crossings at the new anchor (the refs_mu/RA variant
references regenerate; cheap, part of the ladder campaign).
"Sub-10^7 at R2" is retired; the honest claim is "~10^7.1 with golden
target selection at baseline throughput".

## The empirical finding that reframes "golden"

The literature's actual golden substructure lenses are the 11 SLACS
systems Vegetti et al. 2014 selected on lensed-image S/N (J0252+0039,
J0737+3216, J0946+1006, J0956+5100, J0959+4416, J1023+4230,
J1205+4910, J1430+4105, J1627-0053, J2238-0754, J2300+0022; the
J0946+1006 detection of Vegetti et al. 2010 is in this family). Four
of the 11 have Newton source photometry:

| system | I814 (unlensed) | mu | z_s |
| --- | --- | --- | --- |
| J0737+3216 | 24.04 | 15.5 | 0.581 |
| J1205+4910 | 24.14 | 13.9 | 0.481 |
| J2238-0754 | 24.17 | 15.1 | 0.713 |
| J2300+0022 | 25.83 | 14.0 | 0.463 |

Their median source magnitude is 24.16, i.e. the SURVEY MEDIAN
(24.13/24.20), and one is well below it; but their median
magnification is 14.6 against the survey median 8.6. Golden-ness in
observed substructure work is magnification / arc S/N, not intrinsic
source brightness. Two design consequences, both already in place:

1. The operational golden tier is selected on observables (arc S/N
   enters the frozen score), exactly mirroring the Vegetti selection;
   its members keep their native pool magnitudes (realized golden 5
   at 24.416-24.689 V = parent p25-p40), per ratified B8.
2. The golden-magnitude canary is therefore labelled a controlled
   brightness FACTORIZATION AXIS (what does a brighter source buy at
   fixed everything else), never a claim about what real golden
   lenses look like. The Vegetti table above is the citation for why.

## Citations

- Newton et al. 2011, ApJ 734, 104 (arXiv:1104.2608), Table 1.
- Vegetti et al. 2014, MNRAS 442, 2017 (arXiv:1405.3666), Table 1;
  Vegetti et al. 2010 (arXiv:0910.0760) for the J0946+1006 detection.
- Roman forecast: arXiv:2506.03390. Euclid: O'Riordan et al. 2023
  (arXiv:2211.15679); Collett 2015 (population precedent).
- Despali et al. 2022 (arXiv:2111.08718): no source-magnitude anchor
  (exposure/SNR is their brightness knob); see
  despali2022_comparison.md.
- D1 color chain and parent distribution: lensing_anchors.md,
  B8ParentDesign_v1.md section 3.5.
