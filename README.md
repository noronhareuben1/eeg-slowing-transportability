# Rostrocaudal EEG complexity in AD and FTD

> **Current project direction:** a separate transportability study is now being
> developed in `transportability/`. It will test a locked, compact EEG slowing
> index on an independent routine-EEG cohort. The completed complexity audit
> remains unchanged and is retained as a separate reproducibility manuscript.

Reproducible analysis for the study:

> **Is rostrocaudal EEG complexity a state-invariant biomarker of dementia subtype? Spectral-surrogate decomposition and leakage-safe deep learning in paired Alzheimer’s disease, frontotemporal dementia, and control recordings**

The completed manuscript is titled **Testing rostrocaudal EEG complexity as a state-invariant marker of dementia subtype: spectral-surrogate decomposition and leakage-safe prediction in paired Alzheimer’s disease, frontotemporal dementia, and control recordings**.

## Scientific question

A 2025 *npj Aging* study reported opposite rostrocaudal fractal-dimension patterns in Alzheimer’s disease (AD) and frontotemporal dementia (FTD) using OpenNeuro dataset `ds004504`. This project asks whether that effect:

1. reproduces under a fully open Python pipeline;
2. survives adjustment for oscillatory power and the aperiodic spectrum;
3. exceeds values expected from spectrum-preserving IAAFT surrogate signals;
4. generalizes to a paired photic-stimulation recording (`ds006036`) without participant leakage; and
5. adds subject-level predictive value beyond a compact EEGNet baseline.

The primary goal is mechanistic validity, not maximization of epoch-level accuracy.

## Data

- OpenNeuro `ds004504`: eyes-closed EEG, 36 AD / 23 FTD / 29 cognitively normal participants.
- OpenNeuro `ds006036`: paired photic-stimulation EEG from the same 88 participants.

Both datasets are CC0. Large EEG files are downloaded locally and are never committed.

## Validation principles

- The participant is the statistical and predictive unit.
- Every epoch from a participant stays in the same fold.
- Preprocessing parameters learned from data, feature selection, normalization, calibration, and model tuning are fit using training participants only.
- Cross-state tests use participants absent from training, despite the two datasets sharing the same cohort.
- Confirmatory hypotheses and endpoints are fixed in `docs/preregistration.md` before outcome inspection.
- Negative results are retained and reported.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ml]"

rcd download --dataset ds004504 --derivatives-only
rcd download --dataset ds006036 --derivatives-only
rcd validate-data
rcd run-reproduction
rcd run-mechanistic --n-surrogates 20
rcd run-state
rcd run-classical
rcd run-deep --repeats 10
rcd make-report
```

The deep stage uses CUDA automatically when it is exposed to PyTorch and otherwise runs on CPU. Run `rcd --help` for stage-specific options.

## Completed result

All five frozen hypotheses were unsupported after Holm correction. EEGNet achieved participant-level three-class macro AUC 0.638 (95% bootstrap CI 0.560–0.711); late complexity fusion achieved 0.669 (0.586–0.749), with a paired improvement of 0.032 (−0.025 to 0.091; permutation p=0.413). The defensible contribution is a reproducibility and validity audit, not a high-accuracy diagnostic claim.

Validated outputs include:

- `outputs/manuscript_values.json`: machine-readable manuscript values;
- `outputs/reporting/figure_main_results.png`: four-panel primary figure;
- `manuscript/rostrocaudal_eeg_validity_audit.docx`: editable journal-style manuscript;
- `docs/submission_readiness.md`: required institutional and submission steps;
- `outputs/manifests/`: stage inputs, outputs, versions, and hashes.

The test suite contains 20 tests, including one optional PyTorch test. Raw EEG
and local environments are intentionally excluded from the reproducibility
archive.

## Authorship and clinical scope

This is research software, not a clinical diagnostic device. Final authorship requires direct intellectual contribution, verification of all results, responsibility for the submitted text, and disclosure of AI assistance according to the target journal’s policy.

## Completed transportability track

The new candidate paper is **A locked three-feature EEG slowing index for
dementia screening: external validation across independent routine-EEG
cohorts**. Its primary endpoint is AD-versus-control external validation. The
analysis is deliberately small and interpretable: posterior delta/alpha
slowing, posterior alpha activity, and the global aperiodic exponent, compared
with an age/sex reference and EEGNet. The frozen protocol and checksum-aware
P-ADIC downloader are in `transportability/protocol.md` and
`transportability/download_padic.py`.

The P-ADIC files are public through Dryad but are several gigabytes and are
never committed to git. If the current execution environment cannot pass
Dryad's signed redirect, the same downloader can be run on the user's local
machine and the resulting manifest can be copied into `data/p_adic/raw/`.

The locked model achieved external AD/CN ROC-AUC 0.702 (95% CI 0.611-0.788).
A dated exploratory amendment then tested compact spectral and rostrocaudal
complexity additions. The spectral model improved the internal AD/FTD/CN point
estimate from 0.578 to 0.657 macro AUC, but the corresponding external AD/CN
extension was essentially unchanged at 0.705. Complexity additions did not
improve on the spectral-only model, and FTD discrimination remained weak. See
`transportability/amendment_v1_1.md` and `docs/amendment_v1_1_results.md`.

## Paired perturbational amendment

A second exploratory amendment combines resting topography with
frequency-resolved 5/10/15/20 Hz photic responses and a two-stage
dementia-versus-control then AD-versus-FTD classifier. In 87 paired participants,
10 x 5 repeated nested validation produced macro ROC-AUC 0.777 (95% CI
0.723-0.829), compared with 0.663 (0.582-0.744) for the resting comparator. The
paired macro-AUC difference was +0.113 (0.035 to 0.191). FTD AUC improved from
0.451 to 0.611, but the paired difference interval still included zero.

This is a promising internally validated candidate, not a state-of-the-art or
clinical claim. The highest recent report on the same benchmark is substantially
higher, and independent paired AD/FTD/CN validation is still required. See
`transportability/amendment_v1_2.md` and `docs/amendment_v1_2_results.md`.
