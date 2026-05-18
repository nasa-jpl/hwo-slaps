# Publication venue plan

## Strategy

Use one coherent study, not two separate studies.

The detection and PSF-stability questions are scientifically coupled: a PSF-requirement paper needs a credible detection metric, and a mass-reach paper needs to account for PSF uncertainty. The clean publication strategy is therefore to publish the same scientific arc at two different maturity levels:

- **SPIE 2026 proceedings and poster:** preliminary but real conference version.
- **RASTI HWO Special Issue:** expanded, calibrated archival journal version.

This keeps the work coherent while making the RASTI manuscript meaningfully distinct from the SPIE proceedings.

## External constraints and venue context

This plan is designed to satisfy the SPIE proceedings/poster requirements while preserving a stronger archival submission for RASTI.

### SPIE context

SPIE Astronomical Telescopes + Instrumentation 2026 is scheduled for 5-10 July 2026 in Copenhagen. The relevant preparation dates for this project are:

- poster PDFs due: 10 June 2026,
- manuscripts due: 17 June 2026.

The SPIE manuscript should be a real technical proceedings paper, not a placeholder. It can be a status report or first forecast if it is technically sound, contains new scientific or technical content, and includes enough data to support the conclusions.

### RASTI context

The RASTI Habitable World Observatory Special Issue explicitly welcomes HWO mission concept development software, tools, methodologies, data simulators, instrument performance evaluators, uncertainty quantification, integrated modeling, and preparatory science tools. The submission deadline is 31 August 2026.

HWO-SLAPS fits this call well because it is a science-engineering interface tool: it connects optical/PSF stability assumptions to a quantitative dark-matter science-return metric.

## Publication distinction

| Element | SPIE version | RASTI version |
|---|---|---|
| Purpose | First public technical demonstration | Archival calibrated study |
| Claim level | Framework and first forecasts | Calibrated PSF-stability implications |
| Scene count | One canonical SCDD-like scene | Multiple scenes or source/lens stress tests |
| Subhalo positions | Local plus Einstein-ring map | Full 2D sensitivity maps |
| PSF modes | One required family, optional second family | Multiple segment/global mode families and combinations |
| Validation | Sparse PyAutoLens-JAX subset if time permits | Mandatory nonlinear validation grid |
| False positives | Optional demo | Required no-subhalo PSF-mismatch study |
| Source realism | Smooth source acceptable if caveated | Clumpy/cuspy source stress tests required |
| Requirement language | Preliminary requirement-style curves | Calibrated requirement translation |

## SPIE 2026 proceedings and poster

Presentation title:

> Determining PSF stability requirements for low-mass dark matter subhalo detection with the Habitable Worlds Observatory

Purpose:

Present HWO-SLAPS as an end-to-end pipeline and show first quantitative PSF-stability forecasts for strong-lensing subhalo detection.

Core scope:

- Introduce the science motivation from the SCDD PSF-stability future-work case.
- Present HWO-SLAPS as the pipeline connecting lens simulation, segmented-mirror PSFs, detector noise, and Fisher detectability.
- Demonstrate subhalo detectability forecasts using a bounded canonical setup.
- Show first controlled PSF-aberration or drift results.
- Produce early plots of detection significance, detectable-ring fraction, or minimum detectable mass versus PSF error.

Recommended bounds:

- One canonical SCDD-like strong-lens scene.
- SCDD anchor masses: `1e7`, `10^7.25`, `10^7.5`, `10^7.75`, and `1e8 Msun`.
- Optional exploratory low masses below `1e7 Msun`, clearly labeled.
- Perfect-PSF baseline.
- One or two PSF mode families, such as segment piston/hexike and low-order global Zernikes.
- One stability or amplitude axis.
- One Fisher ring-map demonstration around the Einstein ring, reported as a detectable-ring fraction rather than full 2D detectable area.
- Sparse PyAutoLens-JAX nonlinear-fit validation only if it can be completed without risking the main SPIE figures.

Recommended SPIE figures:

- Pipeline schematic.
- Canonical lensing scene and subhalo residual.
- Perfect and perturbed PSF diagnostic.
- Detection metric versus subhalo mass.
- Detection degradation or detectable-ring fraction versus PSF amplitude.
- PSF-mode coupling or tolerance-style figure.
- Optional Fisher-versus-nonlinear validation figure.

Tone:

- “We present a framework and first forecasts.”
- “We translate PSF perturbations into preliminary subhalo-detection sensitivity curves.”
- “We provide sparse nonlinear validation where available.”

Avoid:

- “We derive final HWO engineering requirements.”
- “HWO must meet this exact wavefront stability number.”
- “The Fisher metric is fully calibrated,” unless the PyAutoLens validation grid is actually complete.

## RASTI HWO Special Issue

Purpose:

Submit the full archival science study to the RASTI Habitable World Observatory Special Issue.

Core scope:

- Expand the SPIE framework into a calibrated, publishable study.
- Expand sparse SPIE nonlinear checks into a Fisher-versus-fit validation grid.
- Run a full PSF-mode sweep.
- Run the full subhalo mass-reach sweep.
- Produce full 2D sensitivity maps, detectable-area curves, and mass-threshold summaries.
- Test false positives from PSF mismatch.
- Translate PSF errors into HWO-relevant requirement language.

Recommended expanded content:

- Multiple PSF families: segment piston, segment tip/tilt, segment hexikes, global Zernikes, and selected combinations.
- A broader PSF amplitude ladder.
- Time-varying PSF or drift cases, but only after defining the assumed observing cadence and stability timescale.
- Full 2D grids of subhalo positions around lensed arcs, not only Einstein-ring samples.
- More subhalo masses.
- Better source realism, especially clumpy or cuspy sources motivated by the SCDD.
- Perfect-PSF and perturbed-PSF comparison grids.
- Fisher-versus-fit calibration across masses, positions, PSF states, and false-positive cases.
- Lens-light and subtraction-residual stress tests.
- Reproducible configs, run manifests, analysis tables, and figure-generation scripts.

Tone:

- “We derive calibrated PSF-stability implications and mass-reach forecasts.”
- “We quantify which PSF modes most degrade low-mass subhalo detectability.”
- “We estimate the HWO-accessible subhalo mass floor under realistic PSF uncertainty.”

RASTI deliverables:

- Full journal manuscript.
- Calibrated detection-statistic section.
- Full method validation.
- Full PSF requirement curves.
- Mass-reach and detectable-area figures.
- False-positive analysis.
- Reproducible analysis outputs.

## PyAutoLens validation placement

For SPIE, PyAutoLens nonlinear modeling is best treated as a small validation subset or planned validation. The SPIE story should not depend on completing a large nonlinear grid in one week.

For RASTI, nonlinear validation should be a central requirement. The RASTI claim that PSF stability requirements are calibrated depends on showing how the Fisher/Asimov statistic maps to direct smooth-versus-subhalo nonlinear fits.

## Bottom line

The SPIE proceedings should be the first public, proceedings-quality version of the integrated study. The RASTI submission should be the substantially expanded and validated journal version. This avoids splitting the project into two weaker papers while preserving a clear distinction between preliminary conference results and archival science conclusions.
