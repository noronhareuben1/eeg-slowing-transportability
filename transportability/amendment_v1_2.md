# Exploratory amendment v1.2: paired perturbational EEG fingerprint

- **Dated:** 14 July 2026
- **Status:** completed exploratory model-development amendment
- **Scope:** internal AD/FTD/CN discrimination in the paired AHEPA cohort

## Rationale

Amendment v1.1 showed that compact resting spectral features improved the
internal point estimate but did not solve FTD discrimination or improve
external P-ADIC AD/CN AUC. Amendment v1.2 tests a different biological idea:
combine a resting EEG trait with the frequency-resolved response to a controlled
photic perturbation. The intended signal is not only how slow or complex the
brain is at rest, but how its oscillatory and aperiodic activity changes when it
is driven at 5, 10, 15, and 20 Hz.

This amendment was written after earlier outcomes were known and after model
families were screened. It is therefore exploratory and cannot serve as a
preregistration or independent confirmation.

## Main research question

Does a leakage-safe, two-stage model combining resting EEG topography with a
frequency-resolved photic-response fingerprint improve participant-level
AD/FTD/CN discrimination compared with the same model family using resting EEG
alone?

## Hypotheses

1. The paired model will improve participant-level macro one-versus-rest
   ROC-AUC relative to the resting-only comparator.
2. A dedicated AD-versus-FTD subtype head will improve FTD one-versus-rest AUC
   relative to the resting-only comparator.

The overall hypothesis is supported only if the paired bootstrap confidence
interval for the macro-AUC difference excludes zero. The FTD hypothesis is
supported only if its paired difference interval excludes zero.

## Cohort and unit of analysis

- OpenNeuro `ds004504` supplies eyes-closed resting EEG features.
- OpenNeuro `ds006036` snapshot `1.0.6` supplies paired open-eye photic EEG.
- The datasets contain recordings from the same AHEPA participants.
- The analysis includes 87 participants with a usable photic interval: 35 AD,
  23 FTD, and 29 cognitively normal participants.
- The participant is the only validation and reporting unit. No epoch-level
  result is reported as an independent observation.
- Age, sex, and MMSE are excluded from prediction.

## Photic-response fingerprint

For each common stimulation frequency (5, 10, 15, and 20 Hz), a standardized
2.0-second segment begins 0.25 seconds after the open-eye marker. Repeated
segments at the same frequency are averaged within participant and channel.
The feature layer contains:

- relative delta, theta, alpha, beta, and gamma power;
- spectral edge and median frequency;
- log-log spectral slope, aperiodic offset, and aperiodic exponent;
- spectral entropy;
- stimulus-frequency and second-harmonic signal-to-noise ratios;
- Higuchi fractal dimension;
- channel-resolved values, global/rostral/caudal summaries,
  rostrocaudal and left-right gradients, and frequency-response trends.

Missing frequency blocks are imputed with the training-fold median. Candidate
features present in fewer than 60% of participants or constant across the
cohort are removed before validation. Within each training fold, univariate
selection is restricted to 5, 10, 20, or 40 features.

## Models

Four outputs share identical outer splits:

1. `resting_direct`: direct three-class prediction from resting features;
2. `paired_direct`: direct three-class prediction from resting and photic
   features;
3. `paired_hierarchical`: a dementia-versus-CN head followed by a separately
   tuned AD-versus-FTD head;
4. `paired_hybrid`: an equal-probability average of the paired direct and
   hierarchical outputs.

Training-fold model selection compares balanced logistic regression,
shrinkage LDA, and balanced SVM. The hierarchical probabilities are combined
as `P(CN)=1-P(dementia)`, `P(AD)=P(dementia)*P(AD|dementia)`, and
`P(FTD)=P(dementia)*P(FTD|dementia)`.

## Validation and uncertainty

- 10 repeats of stratified five-fold outer validation;
- four-fold inner tuning inside each outer training set;
- no participant overlap between training and testing;
- one held-out probability per participant per repeat;
- participant probabilities averaged over the 10 repeats;
- 5,000 class-stratified participant bootstrap resamples;
- paired bootstrap differences relative to `resting_direct`.

## Interpretation boundary

This amendment can identify a candidate model for an independently locked
study. It cannot establish clinical utility, universal applicability, or
state/site transportability because the paired AD/FTD/CN development cohort is
not independent. The P-ADIC cohort can externally test AD/CN resting features,
but it contains neither FTD nor the matched photic protocol needed to validate
this model.
