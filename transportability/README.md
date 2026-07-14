# Transportable EEG slowing index

This directory is the second, separate analysis track for the dementia EEG
project. The completed rostrocaudal-complexity validity audit remains in the
repository; this study asks a different question:

> Can a small, physiologically specified EEG feature panel trained in one
> routine-EEG cohort retain discrimination and calibration in an independent
> cohort recorded at another institution?

The primary target is Alzheimer disease versus cognitively normal controls.
AD-versus-FTD is a secondary stress test, not the primary endpoint.

## Cohorts

- **Derivation:** OpenNeuro `ds004504` v1.0.9, eyes-closed EEG, 36 AD, 23 FTD,
  29 controls.
- **External validation:** P-ADIC Dryad `10.5061/dryad.8gtht76pw`, using the
  AD and control files first. The data are not copied into this repository.

## Frozen analysis principle

Feature definitions, preprocessing, model class, and thresholds are specified
in `protocol.md` before the external cohort is inspected for outcomes. Any
feature or preprocessing change made after external labels are read is an
exploratory amendment and must be reported as such.

## Running

The first step is data acquisition and checksum verification:

```bash
python -m transportability.download_padic --output data/p_adic/raw
```

Raw EEG is intentionally excluded from git and from the reproducibility
archive. The script records source DOI, file IDs, sizes, and SHA-256 checksums.

## Exploratory compact-model amendment

The dated `amendment_v1_1.md` retains the locked baseline and tests compact
spectral-parameterization and rostrocaudal-complexity additions. It reports
internal participant-level AD/FTD/CN validation and one-way external AD/CN
validation in P-ADIC. Run it with:

```bash
python -m transportability.run_amendment_analysis --project-root .
```

The external cohort does not include FTD in the files used here, so the
three-class analysis is not described as externally validated.

## Paired resting-and-photic amendment

The dated `amendment_v1_2.md` tests the proposed AD/FTD/CN improvement: a
resting dementia-screening head followed by an AD-versus-FTD head informed by
frequency-resolved 5, 10, 15, and 20 Hz photic responses. Generate the paired
features and run the repeated nested participant-level analysis with:

```bash
python -m transportability.photic_response_features --project-root . --n-jobs 4
python -m transportability.run_amendment_v1_2 --project-root . \
  --outer-repeats 10 --bootstrap-iterations 5000 --n-jobs 4
```

The two-stage candidate achieved macro ROC-AUC 0.777 (95% CI 0.723-0.829),
compared with 0.663 (0.582-0.744) for resting EEG alone. This is an exploratory
internal result. It identifies a panel to lock before independent multisite
validation; it is not a clinical or state-of-the-art claim.
