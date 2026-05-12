# Publication venue plan

## Strategy

Use one coherent study, not two separate studies.

The detection and PSF-stability questions are scientifically coupled: a PSF-requirement paper needs a credible detection metric, and a mass-reach paper needs to account for PSF uncertainty. The clean publication strategy is therefore to publish the same scientific arc at two different maturity levels:

- SPIE 2026 proceedings and poster: a preliminary but real conference version.
- RASTI HWO Special Issue: the expanded, calibrated archival journal version.

This keeps the work coherent while making the RASTI manuscript meaningfully distinct from the SPIE proceedings.

## External constraints and venue context

This plan is designed to satisfy the SPIE funding and proceedings requirements while preserving a stronger archival submission for RASTI.

SPIE context:

- The SPIE Astronomical Telescopes + Instrumentation 2026 [manuscript submission guidelines](https://www.spiecareercenter.org/conferences-and-exhibitions/astronomical-telescopes-and-instrumentation/presenters/manuscript-submission-guidelines) list the manuscript deadline as 17 June 2026 and the poster PDF deadline as 10 June 2026.
- SPIE specifies a 2-page minimum and says papers should include the normal elements of a full-length technical manuscript: title, authors, abstract, keywords, sections such as introduction/methods/results, acknowledgments if applicable, and references.
- SPIE proceedings are framed as rapid reporting of current research. The guidelines state that papers may be status reports of work in progress or descriptions of completed research, provided they are technically sound, contain new research or scientific concepts, include enough technical data to support conclusions, and have adequate references.
- SPIE also states that conference chairs/editors may require revisions or reject proceedings papers that do not meet technical, suitability, clarity, or quality expectations.
- The student support email adds a stricter practical constraint for this project: reimbursement is contingent on submitting a full-length manuscript for Paper 14145-260, having it approved for publication, and presenting the poster in person.

RASTI context:

- The RASTI [Habitable World Observatory Special Issue call](https://academic.oup.com/rasti/pages/habitable-world-observatory) explicitly welcomes HWO mission concept development software, tools, methodologies, data simulators, instrument performance evaluators, uncertainty quantification, and preparatory science tools.
- The call lists a submission deadline of 31 August 2026.
- Submissions go through the normal RASTI peer-review process and must meet the journal's criteria for acceptance.

Implication:

The SPIE manuscript must be a real technical proceedings paper, not a placeholder. However, it does not need to be the final archival version of the study. The RASTI submission should be visibly expanded and revised, with fuller validation, broader sweeps, stronger conclusions, and clearer requirement translation.

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
- Produce early plots of detection significance or minimum detectable mass versus PSF error.

Recommended bounds:

- One canonical SCDD-like strong-lens scene.
- A small subhalo mass ladder, for example `10^7`, `10^7.5`, and `10^8 Msun`.
- A perfect-PSF baseline.
- One or two PSF mode families, such as segment piston and low-order global Zernikes.
- One stability or drift axis.
- One sensitivity-map or detectable-area demonstration if feasible.

Tone:

- "We present a framework and first forecasts."
- "We translate PSF perturbations into preliminary subhalo-detection sensitivity curves."
- Avoid claiming final HWO engineering requirements unless the validation is strong.

SPIE deliverables:

- Proceedings manuscript.
- Poster.
- Pipeline schematic.
- One baseline sensitivity or detectability figure.
- One PSF-mode illustration.
- One preliminary requirement-style plot.

## RASTI HWO Special Issue

Purpose:

Submit the full archival science study to the RASTI Habitable World Observatory Special Issue.

Core scope:

- Expand the SPIE framework into a calibrated, publishable study.
- Validate the Fisher detector against a sparse set of full nonlinear likelihood fits.
- Run a full PSF-mode sweep.
- Run the full subhalo mass-reach sweep.
- Produce sensitivity maps, detectable-area curves, and mass-threshold summaries.
- Test false positives from PSF mismatch.
- Translate PSF errors into HWO-relevant requirement language.

Recommended expanded content:

- Multiple PSF families: segment piston, segment tip/tilt, segment hexikes, global Zernikes, and selected combinations.
- A broader PSF amplitude ladder.
- Time varying PSF
- EAC 1 and maybe EAC 2
- More subhalo locations and masses.
- Better source realism, especially clumpy or cuspy sources motivated by the SCDD.
- Perfect-PSF and perturbed-PSF comparison grids.
- Fisher-versus-fit calibration cases.
- Reproducible configs, run manifests, analysis tables, and figure-generation scripts.

Tone:

- "We derive calibrated PSF-stability requirements and mass-reach implications."
- "We quantify which PSF modes most degrade low-mass subhalo detectability."
- "We estimate the HWO-accessible subhalo mass floor under realistic PSF uncertainty."

RASTI deliverables:

- Full journal manuscript.
- Calibrated detection-statistic section.
- Full method validation.
- Full PSF requirement curves.
- Mass-reach and detectable-area figures.
- Reproducible analysis outputs.

## Bottom line

The SPIE proceedings should be the first public, proceedings-quality version of the integrated study. The RASTI submission should be the substantially expanded and validated journal version.

This avoids splitting the project into two weaker papers while preserving a clear publication distinction between the preliminary conference result and the archival journal result.
