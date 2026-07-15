# Reproducibility guide

## Source snapshots

- OpenNeuro `ds004504` v1.0.9, DOI
  `10.18112/openneuro.ds004504.v1.0.9`;
- OpenNeuro `ds006036` v1.0.6, DOI
  `10.18112/openneuro.ds006036.v1.0.6`;
- P-ADIC Dryad DOI `10.5061/dryad.8gtht76pw`, published 2022-10-27.

The OpenNeuro pipeline records source hashes in `outputs/manifests/`. The
P-ADIC downloader fixes file IDs, byte sizes, and SHA-256 hashes in
`transportability/download_padic.py` and writes a local manifest after
verification.

## Analysis boundaries

### Frozen analyses

- `docs/preregistration.md` defines the original complexity audit before its
  confirmatory tests.
- `transportability/protocol.md` defines the three-feature external AD/CN model
  before P-ADIC outcomes were examined.

### Exploratory analyses

- `transportability/amendment_v1_1.md` was dated after the external AD/CN result
  and tests compact spectral and complexity additions.
- `transportability/amendment_v1_2.md` was dated after earlier results and tests
  the paired photic candidate.

Exploratory improvements generate a future locked protocol. They are not
retroactive confirmation.

## Expected results

A full reproduction should recover these canonical values within ordinary
floating-point tolerance:

- external three-feature AD/CN AUC `0.7015306122`, n = 145;
- external AUC 95% interval `0.6111786383` to `0.7882706207`;
- external calibration slope `0.3828818059`;
- v1.1 internal spectral macro AUC `0.6574775646`, n = 88;
- v1.1 external spectral AUC `0.7047193878`;
- v1.2 paired two-stage macro AUC `0.7765489437`, n = 87;
- v1.2 paired macro-AUC improvement `0.1131637975`;
- v1.2 paired improvement interval `0.0351535887` to `0.1911697404`;
- v1.2 FTD AUC `0.6114130435` and default-rule sensitivity `0.1304347826`.

Canonical machine-readable values are in `outputs/transportability/`,
`outputs/amendment_v1_1/`, and `outputs/amendment_v1_2/`.

## Data-free verification

```bash
ruff check src tests transportability
pytest
```

The suite tests dataset parsing, event handling, EEG features, spectra,
complexity and surrogates, participant-level prediction, P-ADIC extraction,
external validation, and both amendments. The EEGNet test is optional when
PyTorch is absent.

## Re-running the models

The exact command sequence is in the root `README.md`. All splits are grouped
by participant. Scaling, imputation, feature selection, hyperparameter tuning,
and calibration are fitted on training participants only. The two OpenNeuro
datasets contain the same cohort, so cross-state analyses use shared subject
splits and are never described as external validation.

Deep-learning checkpoints are retained for the original audit. Raw EEG,
downloaded P-ADIC matrices, package caches, and local environments are ignored.
