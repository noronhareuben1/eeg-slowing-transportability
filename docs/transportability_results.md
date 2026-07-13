# Transportability Results Checkpoint

Date: 2026-07-13

## Data Acquisition

P-ADIC Dryad files were downloaded through Dryad's browser-mediated selected
ZIP flow and extracted into `data/p_adic/raw/`. The checksum verifier completed
successfully for:

- `alz_c1_new.mat`
- `controls_c1_new.mat`
- `AUTHOR_DATASET_SHOR_BENNINGER.txt`

The generated manifest is `data/p_adic/raw/manifest.json`.

## Locked Analysis

Derivation features were built from the completed ds004504 regional spectral
table. The locked EEG-only panel is:

- posterior delta/alpha ratio;
- posterior alpha relative power;
- global aperiodic exponent.

P-ADIC MATLAB v7.3 files were read with an HDF5 path that extracts the first
120 seconds of each valid 19-channel recording and preserves per-recording
sampling frequency.

## Primary External Result

The locked three-feature logistic model trained on ds004504 AD/control
participants was evaluated once on P-ADIC AD/control recordings.

- Derivation n: 65
- External n: 145
- External ROC-AUC: 0.702
- 95% bootstrap CI: 0.611 to 0.788
- Average precision: 0.480
- Brier score: 0.345
- Calibration intercept: -1.182
- Calibration slope: 0.383
- Sensitivity at the derivation-selected threshold: 0.796
- Specificity at the derivation-selected threshold: 0.490

## Interpretation

The external AUC narrowly meets the prespecified feasibility threshold of
0.70, but the wide confidence interval and poor transported specificity argue
against a polished diagnostic claim. The most defensible manuscript framing is
that a compact EEG slowing panel shows modest cross-cohort discrimination but
requires recalibration before any screening use.

The age/sex reference was run as a complete-case comparator:

- Reference derivation n: 65
- Reference external n: 142
- Reference external ROC-AUC: 0.245
- Reference Brier score: 0.390

Because the reference learned from ds004504 does not transport in the same
direction, it should be reported as a failed demographic transport comparator,
not as evidence that demographics are generally uninformative.

## Files

- `outputs/transportability/derivation_features.csv`
- `outputs/transportability/external_padic_features.csv`
- `outputs/transportability/external_validation_results.json`
