# Frozen protocol v1.0 — transportable EEG slowing index

**Freeze date:** 2026-07-13  
**Status:** prespecified feasibility/validation protocol; no external outcome
results have been used to choose features.

## Scientific question

Does a compact, physiologically specified EEG feature panel derived in the
OpenNeuro AHEPA cohort provide subject-level Alzheimer disease screening that
transports to an independent routine-EEG cohort?

## Primary hypothesis

A three-feature logistic model, trained only on the derivation cohort, will
discriminate AD from controls in the external cohort with ROC-AUC above 0.70
and retain calibration better than an age/sex-only reference model.

This is a directional feasibility hypothesis, not a promise of significance.
The study will report the complete estimate and confidence interval even if the
hypothesis is not supported.

## Cohort roles

1. `ds004504` v1.0.9 is the derivation cohort. Its labels may be used to fit
   preprocessing parameters, feature scaling, model coefficients, and a
   decision threshold.
2. P-ADIC (`10.5061/dryad.8gtht76pw`) is the locked external cohort. The
   external test set is not used for feature selection, threshold tuning, or
   model modification.
3. AD-versus-FTD is secondary. MCI, depression, and schizophrenia are reserved
   for exploratory specificity analyses if their labels and signal formats can
   be mapped without ambiguity.

## Features

The panel is chosen before external results are examined and is intended to
represent EEG slowing rather than a novel black-box architecture:

1. posterior relative delta-to-alpha power ratio;
2. posterior alpha peak or alpha-band relative power, using the prespecified
   fallback when a resolvable alpha peak is absent;
3. global aperiodic exponent estimated from the 1–30 Hz spectrum.

All features are computed after the same channel harmonization, reference,
bandpass, artifact, and eyes-closed selection rules. No MMSE, diagnosis-derived
feature, or participant identifier is an input.

## Models and comparisons

- Primary: standardized three-feature logistic regression, fit on derivation
  data only.
- Reference: age and sex logistic regression.
- Secondary: EEG-only ridge logistic regression with the same fixed feature
  vocabulary, and EEGNet only as a prespecified descriptive comparator.

## Evaluation

The primary external metrics are ROC-AUC, sensitivity at 90% specificity,
specificity at the derivation-selected threshold, Brier score, calibration
intercept/slope, and bootstrap 95% confidence intervals. Confidence intervals
are subject-level and stratified by diagnosis. No epoch-level split is allowed.

Internal derivation performance uses repeated subject-level nested cross-
validation. External validation is one-way and untouched.

## Stopping and amendment rules

If signal formats, eyes-open/closed metadata, or labels cannot be aligned
without unverifiable assumptions, the external cohort will not be forced into
the analysis. The project will report the data-compatibility failure and may
switch to a different independent cohort only through a dated protocol
amendment. No post hoc feature search is permitted to create a positive result.
