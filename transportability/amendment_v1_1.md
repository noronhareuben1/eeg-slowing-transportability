# Analysis amendment v1.1: compact spectral and complexity extensions

**Amendment date:** 2026-07-14  
**Status:** exploratory extension written after completion of the locked v1.0
external AD-versus-control analysis

## Rationale

The v1.0 study tested a locked three-feature EEG slowing model trained in the
AHEPA OpenNeuro cohort and evaluated once in P-ADIC. The external result is
already known (ROC-AUC 0.702, 95% CI 0.611-0.788), so no analysis added here is
described as a new confirmatory test.

This amendment implements the next study direction: retain the small,
physiologically interpretable baseline and test whether compact additions from
periodic/aperiodic spectral parameterization and rostrocaudal complexity improve
classification without sacrificing transportability.

## Questions

1. Does the locked three-feature EEG slowing model provide useful internal
   separation of AD, FTD, and cognitively normal (CN) participants?
2. Do compact spectral-parameterization features improve three-class
   participant-level macro ROC-AUC over the three-feature baseline?
3. Do rostrocaudal complexity features improve three-class discrimination alone
   or when combined with the compact spectral panel?
4. In the external P-ADIC AD/CN cohort, does a five-feature spectral extension
   improve discrimination or calibration over the locked baseline?

## Cohort roles and claim boundaries

- AHEPA OpenNeuro `ds004504` supplies the internal AD/FTD/CN comparison.
- P-ADIC supplies one-way external validation for AD versus CN only.
- P-ADIC does not provide an FTD group in the files used here. The three-class
  model is therefore internally validated and must not be described as an
  externally validated AD/FTD/CN classifier.
- Participant-level splits are mandatory. No recording segment or epoch from a
  participant may appear in both training and test data.

## Fixed feature families

### Locked baseline (3 features)

1. Posterior delta-to-alpha power ratio
2. Posterior relative alpha power
3. Global aperiodic exponent

### Compact spectral extension (8 features total)

The baseline plus global relative theta power, global median frequency, global
aperiodic offset, rostrocaudal relative alpha difference, and rostrocaudal
aperiodic-exponent difference.

### Compact complexity extension (8 features total)

The baseline plus global Higuchi fractal dimension, rostrocaudal Higuchi fractal
dimension, global spectrum-preserving surrogate HFD z-score, rostrocaudal
surrogate HFD z-score, and rostrocaudal box-counting fractal dimension.

### Combined extension (13 features total)

The union of the baseline, compact spectral, and compact complexity features.

### Externally shared spectral extension (5 features total)

The locked baseline plus global relative delta and global relative alpha power.
Only this smaller extension is tested in P-ADIC because those features were
computed with the same vocabulary in both cohorts.

## Models and evaluation

- Multinomial ridge logistic regression is used for the internal three-class
  analysis. The regularization strength is selected inside each training fold.
- Internal evaluation uses 10 repeats of stratified five-fold outer validation
  with four-fold inner tuning. Predictions are averaged per participant across
  repeats before final metrics are calculated.
- The external baseline retains the original fixed logistic-regression setting.
  The shared spectral extension selects regularization using derivation data
  only, then is fitted once and evaluated once in P-ADIC.
- Primary comparison metrics are participant-level macro one-versus-rest ROC-AUC
  for three-class analysis and ROC-AUC for external AD/CN analysis. Balanced
  accuracy, macro F1, log loss, Brier score, calibration slope/intercept, and
  sensitivity/specificity at a derivation-selected threshold are secondary.
- Stratified subject-level bootstrap intervals and paired bootstrap differences
  are reported. Complete estimates are retained regardless of direction.

## Interpretation rule

An extension is considered promising only if its point estimate improves over
the baseline and the uncertainty interval does not indicate a large likely
loss. These exploratory results generate a future independent validation
protocol; they do not establish universal clinical applicability.
