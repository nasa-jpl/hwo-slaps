# HWO Science Engineering Interface (SEI) v0.1.9 — vendored provenance

Vendored 2026-08-23 after George ruled the SEI the authoritative
instrument source for HWO-SLAPS observing parameters ("I trust that
supremely"). Files copied verbatim from the PyPI wheel
`hwo_sci_eng-0.1.9-py3-none-any.whl` (the wheel itself stays in the
untracked `scratch/q1_observing_conditions/sei_v0.1.9/`; its digest is
the unmatched SHA256SUMS entry); hashes in
SHA256SUMS.

Package: "HWO Science Engineering Interface", authors Breann Sitarski
(NASA/GSFC) and Jason Tumlinson (STScI),
https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface.
Consumed by the official HWO ETC (spacetelescope/hwo-tools via
syotools; syotools/models/camera.py `set_from_sei` loads HRI.yaml).

## Files and what we use them for

- HRI.yaml — HRI UVIS channel: 9 reflective optics M4-to-focal-plane,
  Al+XeLiF coating, Teledyne COSMOS CMOS detector, plate scale
  7.16 mas, RN 0.2 e-, dark 0.002 e-/pix/s, instrument RMS WFE 35 nm,
  FoV 3'x2'. NIR channel: protected silver, H4RG.
- EAC1.yaml — telescope prescription: PM (19 hex segments, 1.65 m
  point-to-point, 6 mm optical gap, 5.977/7.226 m
  inscribed/circumscribed) + SM + M3 + M4, all Al+XeLiF.
- XeLiF_refl.yaml — measured Al+XeLiF reflectivity curve
  (M. Quijada, GSFC). R(500 nm) = 0.9073.
- ProtectedAg_refl.yaml — protected-silver reflectivity curve.
  R(500 nm) = 0.9680.
- Teledyne_COSMOS_CMOS_QE.yaml — UVIS detector QE. QE(500 nm) = 0.894.

## Derived end-to-end chains at 500 nm (computed 2026-08-23)

13 reflective surfaces total (4 telescope + 9 HRI UVIS).

Baseline (SEI as-specified, Al+XeLiF):
  0.9073^13 = 0.2823 optics; x QE 0.894 = 0.2524 before filters;
  0.20-0.24 for filter transmission 0.80-0.95. The committed
  reference chain value 0.21 (LUVOIR HDI Table 8-5 heritage) equals
  the SEI chain at filter transmission 0.832 — inside the plausible
  range. RULING CONSEQUENCE: 0.21 stands as the realized baseline,
  now cited to SEI-consistency + LUVOIR heritage jointly.

Coating-trade chains (2026-08-23 refinement; three variants, filters
0.80-0.95):
  baseline all-XeLiF (as specified):        0.202-0.240 (0.252 no-filter)
  instrument-only Ag (telescope XeLiF):     0.362-0.429
  full-train Ag (telescope recoated too):   0.469-0.556
Per-surface at 500 nm: XeLiF 0.9073 (9.3% loss/bounce), protected
silver 0.9680 (3.2%); compounded over 13 surfaces the coating swap is
worth 2.32x. The NIR channel's silver coating is cited ONLY as an
existence proof that the project already uses Ag where UV response is
not required; the NIR channel itself (0.8-2.5 um, H4RG) cannot
observe 500 nm and is never part of any chain here.
HONESTY NOTE: the committed bracket 0.504 sits inside the FULL-TRAIN
band, i.e. it implicitly recoats the 4 telescope mirrors, which serve
HWO's UV instruments and are XeLiF by mission-level choice; the
defensible instrument-only trade gives ~0.40. Provenance of 0.504 is
REPLACED regardless (the Stark et al. 2025 coronagraph ETC fiducial,
arXiv:2502.18556, disclaims representing HWO and models a different
train).

## Status of the bracket choice (RULED IN PART, 2026-08-23)

- George RULED: the as-specified Al+XeLiF chain is the BASELINE AND
  STANDARD for all headline results (committed 0.21; R0/R2 arms).
- The upper-bracket variant choice — (a) full-train silver envelope
  0.47-0.56 (keeps the existing 0.504 maps; hypothetical, implies
  recoating the UV-serving telescope) vs (b) instrument-only silver
  0.36-0.43 (physically defensible channel-level trade; R1/R3
  ladders rerun ~30 maps; M_lim(R1) ~10^7.32, M_lim(R3) ~10^6.79 by
  the measured 0.39 dex/mag lever) — is ESCALATED: George is
  consulting his advisor on partial- vs full-train silver, and this
  note is the standing input for GPT Sol Pro to weigh in at the
  final review. Until ruled, R1/R3 numbers are quoted with the
  envelope caveat and the instrument-only alternative alongside.
- RULED 2026-08-23 evening (A6-1): XeLiF only for now. No silver
  bracket is adopted into the design; the 0.504 arms survive only as
  labelled hypothetical-diagnostic axes outside the headline scope,
  and no silver-throughput number is quoted in central results. The
  partial- vs full-train question remains open with the advisor as
  paper wording only.

## Citations for the paper and reviews

- SEI package: hwo_sci_eng v0.1.9, "HWO Science Engineering
  Interface", B. Sitarski (NASA/GSFC) and J. Tumlinson (STScI),
  https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface
  (PyPI wheel in the untracked scratch vendor dir, SHA256SUMS). Consumed by the official
  HWO ETC: spacetelescope/hwo-tools via syotools
  (syotools/models/camera.py set_from_sei).
- XeLiF_refl.yaml header: "XeLiF coating reflectivity curve provided
  by M. Quijada (GSFC)". ProtectedAg_refl.yaml and
  Teledyne_COSMOS_CMOS_QE.yaml: SEI package data (no further
  attribution in file headers).
- EAC1 telescope prescription: SEI EAC1.yaml; cross-consistent with
  Liu et al. 2026, JATIS 12(4) 041017, arXiv:2602.11046 (S1).
- Baseline heritage anchor: LUVOIR Final Report (2019),
  arXiv:1912.06219, HDI Table 8-5 end-to-end system QE 0.21 at V —
  equals the SEI chain at filter T 0.832.
- Retired provenance (documented for the record): Stark et al. 2025,
  arXiv:2502.18556, Table 2 coronagraph ETC fiducial (T_optical
  0.56, raw QE 0.9, effective 0.75); Sec. 3 states the assumptions
  are not intended to represent HWO.
- NIR-channel silver existence proof: SEI HRI.yaml NIR block
  (protected silver, 0.8-2.5 um, H4RG per RST acceptance test
  NTRS TM-20210011344). The NIR channel cannot observe 500 nm; it
  evidences only that Ag coatings are used in HRI where UV response
  is not required.

## Engine-parameter corroborations (SEI vs committed engine values)

- plate scale 7.16 mas = ours exactly;
- read noise 0.2 e- = ours exactly (previously flagged as a
  discrepancy vs LUVOIR HDI 2.5 e-; SEI vindicates our value);
- dark current 0.002 e-/pix/s = ours exactly;
- instrument RMS WFE 35 nm — consistent in scale with our science35
  truth amplitude (note: SEI quotes instrument focal-plane WFE, ours
  is piston-removed telescope aperture OPD; kindred scale, not the
  same quantity — never cite as identity).

## Declared caveats

- v0.1.9 pre-formulation placeholder values; EACs are exploratory.
- Filter transmission curves (UVIS_filters.yaml) are referenced by
  HRI.yaml but NOT shipped in the wheel; our 0.80-0.95 filter range
  and the 0.85/0.832 working points are declared assumptions.
- n_refl_optics = 9 counts the channel-select mechanism as one
  reflective surface (SEI's own note).
- Telescope surface count 4 (PM/SM/M3/M4) read from EAC1.yaml.
