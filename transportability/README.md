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
