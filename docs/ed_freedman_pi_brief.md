# A Paired Resting-and-Photic EEG Study for AD/FTD/CN Discrimination

## Presentation and decision brief for Ed G. Freedman

**Prepared by Reuben Noronha | 14 July 2026 | Exploratory analysis completed**

## The decision I am bringing

I am asking for approval to move this into a university-affiliated,
independently validated study. The analysis plan, preliminary pipeline, and
candidate model are already built. The institutional needs are a faculty
sponsor, the appropriate IRB/data-governance determination, and access to or
collection of an independent cohort containing Alzheimer disease (AD),
frontotemporal dementia (FTD), and cognitively normal controls (CN) with both
resting and photic-stimulation EEG.

## My 90-second opening

> I have a concrete EEG study that targets a weakness in current dementia
> machine learning. Many papers report very high accuracy from large models on
> one repeatedly used dataset, but the results are difficult to compare and may
> not transfer. My approach uses a small, physiologically organized detection
> mechanism in two stages. First, resting posterior slowing and complexity
> separate dementia from cognitively normal controls. Second, frequency-specific
> responses to 5, 10, 15, and 20 Hz photic stimulation help distinguish AD from
> FTD. I have already implemented the leakage-safe pipeline. In 87 paired
> participants, the two-stage model achieved a repeated nested-validation macro
> AUC of 0.777, compared with 0.663 for resting EEG alone. The paired improvement
> was 0.113, with a 95% confidence interval from 0.035 to 0.191. This is not yet
> a clinical model and it does not beat the highest published internal score.
> The next scientifically decisive step is to freeze a reduced panel and test it
> once in an independent AD/FTD/CN cohort. I am asking for the go-ahead to run
> that confirmatory study with university affiliation and the required
> institutional oversight.

## The study in one sentence

Test whether a two-stage, low-parameter EEG model combining a resting neural
trait with a frequency-resolved response to photic stimulation improves
participant-level discrimination among AD, FTD, and CN and transfers to a new
clinical cohort.

## Main research question

Does a leakage-safe, two-stage model combining resting EEG topography with a
frequency-resolved photic-response fingerprint improve participant-level
AD/FTD/CN discrimination compared with resting EEG alone?

## Main hypothesis

AD and FTD differ not only in the amount and location of resting EEG slowing,
but also in how cortical oscillations, aperiodic activity, and complexity respond
to controlled visual drive. A model that first detects dementia from resting
posterior theta/complexity and then separates AD from FTD using the spatial 5
and 20 Hz response will outperform a resting-only model and retain useful
performance in an independent cohort.

## How the model works

### Stage 1: dementia screen

The first head separates AD plus FTD from CN. It emphasizes stable resting
features such as posterior theta activity, caudal complexity, and
surrogate-normalized complexity.

### Stage 2: subtype test

Among participants classified as having dementia, the second head separates AD
from FTD. It emphasizes central-temporal frequency-response features, including
20 Hz median-frequency response, 5 Hz temporal complexity and beta/gamma power,
and 20 Hz parietal aperiodic exponent.

### Final probabilities

The two heads produce one probability for each diagnosis. No age, sex, MMSE,
epoch identity, or participant identifier is used as a predictor.

## What I have already completed

- Checksum-validated public OpenNeuro photic EEG data for 88 participants.
- A standardized 2-second response window after the eye-opening transient.
- Periodic and aperiodic spectral parameterization, spectral entropy,
  stimulus/harmonic entrainment, HFD complexity, and spatial gradients.
- Participant-level leakage controls.
- Ten repeats of five-fold outer validation, with four-fold tuning restricted
  to each training set.
- Five thousand participant-stratified bootstrap resamples.
- Reproducible code, held-out probabilities, tuning records, feature-selection
  frequencies, tables, and figures in a GitHub repository.

## Preliminary result

The paired analysis included 87 participants: 35 AD, 23 FTD, and 29 CN.

| Model | Macro AUC | AD AUC | CN AUC | FTD AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resting EEG only | 0.663 | 0.741 | 0.798 | 0.451 | 0.514 |
| Paired direct model | 0.724 | 0.753 | 0.907 | 0.512 | 0.546 |
| Paired two-stage model | **0.777** | **0.804** | **0.914** | **0.611** | 0.578 |
| Paired hybrid model | 0.753 | 0.781 | 0.914 | 0.563 | **0.610** |

For the selected two-stage model, macro AUC was 0.777 (95% CI 0.723-0.829).
The improvement over resting was +0.113 (95% paired CI +0.035 to +0.191).
FTD AUC improved by +0.160, but its paired CI was -0.012 to +0.330. The overall
gain is supported; the FTD-specific gain remains uncertain.

The default decision threshold identified only 13% of FTD participants
correctly. That is the most important current weakness. AUC shows improved
ranking, but the operating threshold is not clinically acceptable.

## Is it novel?

The individual ingredients are not new. Spectral parameterization, HFD,
photic driving, SVMs, and two-stage classification all exist. The candidate
novelty is their specific organization:

- a resting trait plus a frequency-resolved perturbation fingerprint;
- explicit separation of periodic and aperiodic activity;
- rostrocaudal and channel-level response topography;
- separate dementia-screening and AD/FTD-subtyping heads;
- repeated subject-level nested validation with no demographic predictors;
- a planned external transportability test rather than another internal-only
  accuracy report.

A focused literature search did not identify a published AD/FTD/CN model with
this exact design. The defensible wording is "to our knowledge" until a formal
systematic review is completed.

## Does it beat the current best paper?

No. A June 2026 Scientific Reports paper reported macro AUC 0.99 and 87.5%
accuracy using AutoSSM-ICA-EEG on the photic benchmark. A Coherence-CNN paper
reported 94.32% three-class accuracy. Our current internal result is lower.

That does not make the study unpublishable. A strict 2026 functional-connectivity
study found AD-versus-FTD discrimination remained difficult and warned that
epoch-level or fixed-split validation can overestimate performance. A 2026
scoping review of 46 AHEPA studies found that reported accuracy falls as
validation rigor increases. Our paper should compete on a novel perturbational
mechanism, transparent uncertainty, and independent transportability, not on an
unsupported state-of-the-art claim.

## The confirmatory study I propose

### Design

Prospective or retrospective multisite validation with three diagnostic groups.
Every participant contributes one eyes-closed resting EEG and one standardized
open-eye photic recording containing 5, 10, 15, and 20 Hz stimulation. Clinical
diagnosis must be assigned independently of the study EEG model.

### Target cohort

Target 240 participants, approximately 80 per group, with a minimum viable
cohort of 180, approximately 60 per group. Recruit from at least two sites and
reserve one complete site as the external test set where feasible. Final sample
size will be confirmed statistically before registration, without inspecting
test outcomes.

### Locked model before test labels are opened

1. Reduce the development model to the most stable resting and photic features.
2. Freeze preprocessing, missing-data rules, feature count, model family,
   probability calibration, and class thresholds.
3. Publish a timestamped protocol and code version.
4. Fit only on the development cohort.
5. Run the external cohort once with no model amendment.

### Primary endpoint

Participant-level macro one-versus-rest ROC-AUC for AD/FTD/CN in the external
cohort. The primary success criterion is a paired improvement over the locked
resting-only comparator with a 95% confidence interval excluding zero.

### Key secondary endpoints

- FTD one-versus-rest AUC and its paired difference from resting.
- Per-class sensitivity and specificity at frozen thresholds.
- Calibration intercept, calibration slope, and Brier score.
- Site, age, sex, recording-system, and disease-severity subgroup performance.
- Performance when only the common 19 routine EEG channels are available.

### Failure is still informative

If the paired model does not transport, the study becomes a rigorous negative
transportability result showing which photic-response features are cohort- or
site-specific. The analysis and paper remain useful because the protocol is
locked before external testing.

## Exactly what I need from Ed

1. Approval to proceed with this as a university-affiliated confirmatory study.
2. Faculty sponsorship for the protocol and data-access process.
3. Institutional determination of IRB, exempt-status, or data-use requirements.
4. Introduction to a clinical collaborator or data source with independent AD,
   FTD, and CN EEG, ideally including the standardized photic protocol.
5. Permission to prepare the preregistration and grant/data-access materials
   under the university affiliation.

I am not asking him to invent the research question or redesign the analysis.
I am bringing a completed exploratory result and a specific confirmatory plan
for approval and institutional execution.

## Likely questions and concise answers

### Why photic stimulation?

Resting EEG measures a trait. Photic stimulation adds a controlled stress test
of the system: how strongly, where, and at what frequencies the cortex follows
the drive. AD and FTD may have overlapping resting abnormalities but different
network response profiles.

### What is spectral parameterization?

An EEG power spectrum contains oscillatory peaks sitting on a broadband
background. Spectral parameterization separates the periodic peaks from the
aperiodic background, so apparent band-power changes are not automatically
treated as true oscillatory changes.

### Why a small model instead of a large neural network?

The cohort is small. Restricting each fitted component to at most 40 selected
features and using interpretable physiological groups reduces model capacity,
makes leakage controls auditable, and produces a panel that can be implemented
across ordinary 19-channel EEG systems.

### Is the preliminary model clinically ready?

No. It is internally validated and improves overall AUC, but FTD sensitivity at
the default threshold is poor and there is no independent paired FTD cohort yet.

### Can we say it will be universally applicable?

No. We can say the study is designed to test transportability across sites,
systems, and patient groups. Universal applicability is an empirical question,
not a starting claim.

### What is the publication angle?

A perturbational EEG biomarker paper with a locked external test: resting trait
plus frequency-response fingerprint, explicit periodic/aperiodic decomposition,
two-stage clinical architecture, and transparent comparison with a resting
baseline. A successful external result supports a compact transferable marker;
a failed result is a publishable transportability audit.

## Statements to avoid

- "We beat the current best model."
- "This diagnoses AD or FTD."
- "The model is universally applicable."
- "The FTD result is confirmed."
- "The model was independently validated."

Use instead: "The paired two-stage candidate improved overall repeated
participant-level discrimination and now requires a locked independent test."

## Glossary

**AD:** Alzheimer disease.

**FTD:** Frontotemporal dementia.

**CN:** Cognitively normal control.

**AUC:** How well a model ranks a class above the alternatives across all
possible thresholds; 0.5 is chance and 1.0 is perfect ranking.

**Nested validation:** The test fold is kept separate while model choices are
made only inside the training data.

**Photic driving:** The EEG response to rhythmic flashes of light.

**Aperiodic exponent:** The slope-like background component of the EEG power
spectrum after separating oscillatory peaks.

**HFD:** Higuchi fractal dimension, a measure of signal complexity.

**Transportability:** Whether performance survives a change in site, cohort,
equipment, or recording conditions.

## Key references

1. Saeed F, Aldera S. Scientific Reports. 2026.
   doi:10.1038/s41598-026-57069-1.
2. Mlinaric T, et al. Scientific Reports. 2026.
   doi:10.1038/s41598-026-35316-9.
3. Jiang R, et al. Cognitive Neurodynamics. 2025.
   doi:10.1007/s11571-025-10232-2.
4. Lee DG, Lee SB. IEEE Transactions on Neural Systems and Rehabilitation
   Engineering. 2025. doi:10.1109/TNSRE.2025.3575840.
5. The AHEPA EEG benchmark: setting the standard for machine learning in
   dementia diagnosis, a scoping review. 2026. PMID:42165009.
