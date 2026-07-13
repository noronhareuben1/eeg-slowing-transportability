# Preregistered study protocol

Protocol freeze date: **2026-07-10, before inferential outcome inspection**

Protocol status: **frozen for the direct-reproduction stage**. Any amendment will be appended with a timestamp, rationale, and an explicit confirmatory/exploratory designation; the original text will remain in version history.

## Title

Is rostrocaudal EEG complexity a state-invariant biomarker of dementia subtype? Spectral-surrogate decomposition and leakage-safe deep learning in paired Alzheimer’s disease, frontotemporal dementia, and control recordings

## Rationale

Ghassemkhani, Saroka, and Dotta (2025) reported a significant AD–FTD difference in a rostrocaudal box-counting fractal-dimension contrast in OpenNeuro `ds004504`. However, nonlinear EEG metrics can be strongly determined by the power spectrum, and Alzheimer-related slowing is itself associated with apparent loss of complexity. The published result was derived from 15 seconds per participant, used a data-dependent Higuchi parameter analysis, and did not test whether the effect survives spectrum-preserving surrogate controls. A paired photic-stimulation dataset from the same participants (`ds006036`) now permits a direct state-dependence test.

## Design

Secondary analysis of two public, deidentified, CC0 EEG datasets. The participant is the unit of inference. There are 88 participants: 36 AD, 23 FTD, and 29 cognitively normal controls. One FTD participant may be excluded from a particular analysis only if the preregistered signal-quality criterion cannot be satisfied; exclusions and reasons will be reported before group outcomes are computed.

## Regions and sign convention

- Rostral: Fp1, Fp2, F7, F3, Fz, F4, F8.
- Caudal: T5, P3, Pz, P4, T6, O1, O2.
- Rostrocaudal asymmetry = mean(rostral) minus mean(caudal).
- Positive values therefore mean rostral dominance; negative values mean caudal dominance.

This explicit convention resolves contradictory wording in the prior paper’s Methods; its figure caption and reported direction indicate that the formula above is the intended one.

## Confirmatory hypotheses

### H1: direct reproduction

The AD–FTD difference in broadband rostrocaudal fractal-dimension asymmetry will reproduce in `ds004504` with AD more rostrally dominant than FTD. The primary reproduction feature is the open-source Python implementation that most closely matches the published box-counting procedure. HFD is a prespecified supporting analysis rather than a substitute for the reported measure.

### H2: spectral independence

The diagnosis-by-region effect will remain nonzero after adjustment for age, sex, MMSE, aperiodic exponent, aperiodic offset, and relative delta, theta, and alpha power. The primary test is the adjusted AD–FTD coefficient for the rostrocaudal contrast. Because MMSE may be both a severity marker and a collider, results will be reported with and without MMSE adjustment.

### H3: nonlinear excess complexity

For each participant and channel, IAAFT surrogates will preserve the empirical amplitude distribution and approximate Fourier magnitude spectrum while disrupting nonlinear temporal organization. Nonlinear excess complexity is the standardized difference between the observed metric and its surrogate distribution. The primary H3 test is the AD–FTD difference in rostrocaudal nonlinear-excess HFD. A null result will be interpreted as evidence that the raw HFD effect is sufficiently explained by linear spectral and distributional structure, not as evidence that the diseases have identical neurophysiology.

### H4: state dependence

The diagnosis-by-region effect will be tested in paired eyes-closed and photic-stimulation recordings using a participant random intercept. The confirmatory term is diagnosis by region by recording state. A significant term indicates state dependence; a nonsignificant term with a sufficiently narrow confidence interval supports state robustness. Stimulation frequency is modeled as a repeated factor for event-defined photic intervals.

### H5: incremental prediction

A complexity-informed fusion model will be compared with (a) demographic-only multinomial regression, (b) spectral-feature elastic net, (c) complexity-feature elastic net, and (d) raw-signal EEGNet. The primary predictive endpoint is repeated nested subject-level cross-validated macro one-vs-rest ROC-AUC. The fusion model is successful only if its paired participant-bootstrap confidence interval for improvement over EEGNet excludes zero. Accuracy is secondary.

## Signal processing

The released derivatives are the primary input because they already contain the data custodians’ 0.5–45 Hz filtering, ASR, ICA/ICLabel artifact rejection, channel interpolation, and mastoid rereferencing. The analysis adds a 1–45 Hz zero-phase filter, common-average reference, and resampling to 250 Hz. Sensitivity analyses vary reference, epoch length, and HFD `kmax`.

No preprocessing choice will be selected by maximizing group separation. Quality thresholds are applied blind to diagnosis.

## Outcome hierarchy

1. Primary mechanistic endpoint: AD–FTD difference in surrogate-normalized rostrocaudal HFD in eyes-closed data.
2. Key secondary endpoint: diagnosis-by-region-by-state interaction across paired recordings.
3. Primary predictive endpoint: subject-level macro ROC-AUC improvement of fusion over EEGNet.
4. Secondary endpoints: macro F1, balanced accuracy, log loss, Brier score, calibration slope/intercept, pairwise AD–FTD AUC, and decision-curve net benefit.

## Inference and multiplicity

Effect estimates and 95% confidence intervals are primary. Confirmatory p-values for H1–H5 will be Holm corrected as one family. Channel-wise maps are exploratory and FDR corrected. Bootstrap and permutation resampling operate at the participant level. Random seeds and split assignments are saved before model fitting.

## Missingness and exclusions

No outcome-driven imputation is permitted. Missing MMSE covariates are handled by complete-case adjusted analysis alongside the unadjusted full-sample analysis. A recording fails quality control if fewer than 60 seconds of clean eyes-closed EEG remain for short-window metrics or fewer than 300 seconds remain for DFA. DFA exclusions do not exclude the participant from other metrics.

## Leakage controls

All epochs from one participant remain in one outer fold. Normalization, feature selection, augmentation, early stopping, hyperparameter selection, and probability calibration use outer-training participants only. Because `ds004504` and `ds006036` contain the same people, cross-state evaluation uses shared subject splits: a person absent from training in one state is also absent from training in the other state. A deliberately leaky analysis may be shown only as a labeled cautionary demonstration and never as evidence of model performance.

## Interpretation limits

The diagnoses were clinically assigned in a single cohort; external FTD validation is unavailable. The study can establish reproducibility, state sensitivity, and within-cohort predictive validity, but not clinical diagnostic utility. Any model is investigational and will not be described as detecting preclinical or early disease unless the data support that claim.

## Timestamped operational amendment: 2026-07-12

This amendment was fixed after H1-H3 output inspection but before any group-level H4 state result was computed. It does not alter the H1-H3 endpoints.

- OpenNeuro `ds006036` stores EEGLAB derivatives under `derivatives/eeglab/`; the path resolver now records this dataset-specific pipeline directory.
- The H4 matched-state analysis uses the first 2.0 seconds of each event-defined open-eyes photic interval lasting at least 2.0 seconds. The eyes-closed comparator uses a 2.0-second segment so duration is matched exactly. Multiple intervals at the same frequency are averaged within participant before inference.
- The primary H4 model averages valid photic intervals within participant and tests the AD-versus-FTD diagnosis-by-region-by-state interaction with a participant random intercept. Age, sex, and MMSE adjustment is reported as a sensitivity analysis.
- The repeated-frequency sensitivity analysis is restricted to the four well-represented protocol frequencies (5, 10, 15, and 20 Hz). Rare 3, 7, 25, and 30 Hz labels are retained in the released derived table but not used for confirmatory frequency contrasts.
- A participant with no valid open-eyes interval is absent only from paired H4 analyses and remains in every analysis for which data exist. Exclusion is determined from event availability without reference to the participant's EEG outcome.
- If the random-intercept variance reaches the zero boundary and the mixed-model Hessian is singular, the numerical fallback is a Gaussian generalized estimating equation with exchangeable within-participant correlation and the identical fixed-effects formula. This fallback was specified after the initial optimizer failed but before any H4 interaction estimate was available; the fitted method and optimizer errors will be reported.
- Before any EEGNet result was computed, the CPU-executable H5 schedule was fixed at six evenly spaced 4-second epochs per participant, 20 maximum raw-network epochs, and five-epoch early-stopping patience. Every split remains participant-level, and all 10 outer repeats remain required. The complexity fusion is a frozen-EEGNet late-fusion head trained only on outer-training participants, with validation participants used for early stopping. This is a computational amendment to the originally stated 200-epoch cap; it is not selected from predictive outcomes and must be disclosed in the manuscript.
