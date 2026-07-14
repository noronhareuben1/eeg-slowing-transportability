# A locked three-feature EEG slowing index for Alzheimer disease screening: external validation across independent routine-EEG cohorts

## Abstract

**Background:** EEG slowing is a robust physiological correlate of Alzheimer
disease (AD), but many EEG machine-learning studies remain limited by small
single-cohort evaluations and leakage-prone validation. We tested whether a
compact, prespecified EEG slowing panel trained in one routine-EEG cohort
transports to an independent clinical EEG cohort.

**Methods:** A three-feature logistic model was trained in the OpenNeuro
ds004504 derivation cohort using AD and cognitively normal participants only.
The locked features were posterior delta/alpha ratio, posterior alpha relative
power, and global aperiodic exponent. External validation used the P-ADIC Dryad
AD/control files after checksum verification. The external cohort was not used
for feature selection, threshold selection, or model modification.

**Results:** The derivation cohort included 65 AD/control participants. The
external P-ADIC analysis included 145 AD/control recordings. The locked EEG
panel achieved external ROC-AUC 0.702 (95% bootstrap CI, 0.611 to 0.788) and
average precision 0.480. At the derivation-selected threshold, external
sensitivity was 0.796 and specificity was 0.490. Calibration was weak
(intercept -1.182, slope 0.383; Brier score 0.345). A complete-case age/sex
reference trained in the derivation cohort did not transport in the same
direction (external ROC-AUC 0.245; n=142).

**Conclusions:** A compact EEG slowing panel showed modest cross-cohort AD
discrimination and narrowly met the prespecified feasibility AUC threshold, but
threshold transport and calibration were poor. These findings support further
transportability work, not a diagnostic-use claim.

## Introduction

Routine EEG is inexpensive, widely available, and physiologically sensitive to
the cortical slowing observed in neurodegenerative disease. However, the field
has many high-accuracy reports that are difficult to interpret because models
are often trained and tested within one heavily reused cohort. A clinically
useful EEG biomarker should transport between cohorts, sites, and acquisition
conditions.

This study therefore asked a deliberately narrow question: can a locked,
interpretable EEG slowing index trained in one routine-EEG cohort retain
AD-versus-control discrimination in an independent cohort?

## Methods

### Design

The analysis followed `transportability/protocol.md`. The derivation cohort was
OpenNeuro ds004504. The external validation cohort was P-ADIC Dryad
`10.5061/dryad.8gtht76pw`. Only AD and cognitively normal control recordings
were used for the primary validation.

### Features

The locked EEG-only feature panel contained:

1. posterior delta/alpha ratio;
2. posterior alpha relative power;
3. global aperiodic exponent.

For ds004504, the derivation table was built from the completed regional
spectral output. For P-ADIC, MATLAB v7.3 files were read through HDF5, valid
19-channel recordings were identified from the `G` field, and the first 120
seconds were used with the recording-specific sampling frequency.

### Model

The primary model was standardized logistic regression with balanced class
weights, trained only on derivation participants. The decision threshold was
selected in derivation data at specificity at least 90% where possible.

### Metrics

The primary external metrics were ROC-AUC, bootstrap 95% confidence interval,
average precision, Brier score, calibration intercept and slope, and
sensitivity/specificity at the derivation-selected threshold.

## Results

The locked model used 65 AD/control derivation participants and 145 external
AD/control recordings.

| Metric | Value |
| --- | ---: |
| External ROC-AUC | 0.702 |
| 95% bootstrap CI | 0.611 to 0.788 |
| Average precision | 0.480 |
| Brier score | 0.345 |
| Calibration intercept | -1.182 |
| Calibration slope | 0.383 |
| Sensitivity at derivation threshold | 0.796 |
| Specificity at derivation threshold | 0.490 |

The age/sex reference was evaluable in 65 derivation participants and 142
external recordings. It achieved external ROC-AUC 0.245 and Brier score 0.390,
indicating that demographic transport failed in the learned direction.

## Discussion

The locked three-feature EEG slowing panel showed modest external
discrimination, with an AUC just above the prespecified feasibility threshold.
This is a stronger claim than within-cohort accuracy because the external
cohort differs in source, recording format, and participant mix.

The result is not yet clinically persuasive. The confidence interval includes
values that would be too weak for screening, calibration was poor, and the
derivation threshold did not retain high specificity externally. The best
interpretation is that a small slowing panel may carry transportable signal,
but it requires larger multisite validation and explicit recalibration.

## Exploratory compact-model amendment

After the locked external result was known, a dated exploratory amendment
tested compact spectral-parameterization and rostrocaudal-complexity additions.
The analysis used repeated nested participant-level validation for AD, FTD, and
CN in the 88-person AHEPA cohort. The locked three-feature model achieved macro
ROC-AUC 0.578 (95% CI 0.500 to 0.648). The compact spectral model achieved the
highest internal estimate, 0.657 (0.580 to 0.732), for a paired difference of
+0.079 (-0.001 to +0.161). FTD remained the weakest class (one-versus-rest AUC
0.447). Complexity and combined extensions did not exceed the spectral-only
model.

The five-feature panel shared by AHEPA and P-ADIC achieved external AD/CN AUC
0.705 (95% CI 0.619 to 0.786), essentially unchanged from the locked model. The
paired difference was +0.003 (-0.028 to +0.036). Although its Brier score was
lower (0.272 versus 0.345), specificity at the derivation threshold remained
poor (0.406). These exploratory results support a future independently locked
spectral panel, but they do not show improved external discrimination and do not
establish an externally validated AD/FTD/CN classifier.

## Reproducibility

Key outputs:

- `outputs/transportability/derivation_features.csv`
- `outputs/transportability/external_padic_features.csv`
- `outputs/transportability/external_validation_results.json`
- `docs/transportability_results.md`
- `transportability/amendment_v1_1.md`
- `outputs/amendment_v1_1/table_internal_models.csv`
- `outputs/amendment_v1_1/table_external_models.csv`
- `outputs/amendment_v1_1/figure_amendment_results.png`

Raw P-ADIC files are intentionally not committed.

## Exploratory paired perturbational amendment

A subsequent exploratory analysis used paired resting and photic recordings
from 87 AHEPA participants to test a frequency-resolved perturbational
fingerprint. The selected two-stage model first separated dementia from CN and
then separated AD from FTD. Under 10 x 5 repeated nested participant-level
validation, macro ROC-AUC was 0.777 (95% CI 0.723 to 0.829), compared with 0.663
(0.582 to 0.744) for a resting-feature comparator. The paired macro-AUC
difference was +0.113 (0.035 to 0.191). FTD AUC improved from 0.451 to 0.611,
but the FTD difference interval included zero (-0.012 to 0.330), and
default-threshold FTD sensitivity remained low.

This amendment improves internal discrimination but does not externally
validate the paired model: P-ADIC lacks both FTD and the matched photic protocol.
It identifies a panel to lock before a new independent multisite study.
