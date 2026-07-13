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

## Reproducibility

Key outputs:

- `outputs/transportability/derivation_features.csv`
- `outputs/transportability/external_padic_features.csv`
- `outputs/transportability/external_validation_results.json`
- `docs/transportability_results.md`

Raw P-ADIC files are intentionally not committed.
