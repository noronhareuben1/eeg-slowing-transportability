# Testing rostrocaudal EEG complexity as a state-invariant marker of dementia subtype

## Spectral-surrogate decomposition and leakage-safe prediction in paired Alzheimer’s disease, frontotemporal dementia, and control recordings

**Reuben Noronha**

Cognitive Neurophysiology Laboratory, University of Rochester Medical Center, Rochester, New York, USA

Correspondence: Reuben Noronha, noronhareuben1@gmail.com

**Draft status:** Complete scientific manuscript for institutional, coauthor, and journal-format review. Affiliation, funding, competing-interest, ethics-determination, and authorship statements must be confirmed before submission.

## Abstract

### Background

Rostrocaudal electroencephalographic (EEG) complexity has been proposed as a marker distinguishing Alzheimer’s disease (AD) from frontotemporal dementia (FTD). We tested whether the reported effect is reproducible, spectrum-independent, stable across recording states, and useful beyond EEGNet.

### Methods

We analyzed public, deidentified paired eyes-closed and photic-stimulation EEG from 88 participants (36 AD, 23 FTD, and 29 cognitively normal controls) under a prospectively frozen five-hypothesis protocol. We reproduced the box-counting method, adjusted for periodic and aperiodic spectra, normalized Higuchi fractal dimension (HFD) against 20 spectrum-preserving surrogates per channel, and tested paired state effects with mixed models. Elastic-net models, EEGNet, and complexity fusion were evaluated with 5-fold participant-level cross-validation repeated 10 times. The predictive endpoint was three-class macro one-vs-rest area under the receiver-operating-characteristic curve (AUC).

### Results

The AD–FTD box-count difference was small and uncertain (0.0101, 95% confidence interval [CI] −0.0107 to 0.0305; p=0.348). Spectral adjustment did not establish independence (coefficient 0.0050, 95% CI −0.0092 to 0.0192; p=0.491), and surrogate-normalized HFD was null (difference −0.122, 95% CI −1.161 to 0.967; p=0.824). The diagnosis-by-region-by-state interaction was unsupported (−0.0207, 95% CI −0.1072 to 0.0659; p=0.640). EEGNet achieved AUC 0.638 (95% CI 0.560–0.711); complexity fusion achieved 0.669 (95% CI 0.586–0.749), an improvement of 0.032 (95% CI −0.025 to 0.091; p=0.413). Every Holm-adjusted p value was 1.000.

### Conclusions

In this cohort, rostrocaudal complexity was not robustly reproduced, shown to be spectrum-independent or state-invariant, or found to add reliable value to EEGNet. External participant-level validation is needed before advancing the measure as a dementia-subtype biomarker.

**Keywords:** Alzheimer’s disease; frontotemporal dementia; electroencephalography; fractal dimension; surrogate data; EEGNet; reproducibility; data leakage

## Introduction

Alzheimer’s disease (AD) and frontotemporal dementia (FTD) are clinically and biologically heterogeneous neurodegenerative syndromes whose symptoms can overlap, particularly outside specialist settings. Electroencephalography (EEG) is attractive as a candidate adjunctive biomarker because it is non-invasive, temporally precise, and more accessible than many imaging or fluid-based measures. Quantitative EEG studies have repeatedly described slowing in AD, including increased low-frequency and reduced alpha activity, alongside changes in connectivity and apparent signal complexity [1–5]. However, a candidate marker intended to distinguish dementia subtypes must add information beyond global disease severity and common spectral changes.

The release of a 19-channel eyes-closed EEG dataset containing 36 participants with AD, 23 with FTD, and 29 cognitively normal (CN) controls enabled rapid method development [1]. A subsequent analysis of this cohort reported a significant difference in rostrocaudal box-counting fractal-dimension asymmetry between AD and FTD, with an effect size of approximately 0.78 [3]. The finding is intriguing because a spatial contrast could, in principle, capture disease-specific network topography rather than nonspecific global deterioration. Yet the published method used a short segment, contained contradictory prose about the sign of the rostrocaudal contrast, and did not determine whether the effect reflected nonlinear temporal structure rather than the power spectrum.

This distinction matters because EEG slowing and loss of estimated complexity are tightly coupled [4,5]. Many complexity estimators respond to autocorrelation, spectral slope, filtering, amplitude distribution, epoch length, and noise. Periodic band power and the aperiodic spectrum can therefore confound biological interpretations of a nominally nonlinear metric [6]. Iterative amplitude-adjusted Fourier transform (IAAFT) surrogates offer a direct control: they approximately preserve the empirical Fourier magnitude spectrum and exactly preserve the ranked amplitude distribution while disrupting phase-dependent temporal organization [7]. If a raw complexity contrast disappears after surrogate normalization, the parsimonious interpretation is that linear spectral and distributional structure sufficiently explains that estimate—not that AD and FTD have identical physiology.

A complementary public dataset contains eyes-open photic-stimulation EEG from the same 88 participants [2]. The paired design provides a stringent within-person test of recording-state sensitivity, although it is not an independent validation cohort. It also creates a leakage hazard: training on one recording and testing on the other does not establish generalization if the same participant appears in both sets. EEG signals contain stable person-specific information, and epoch-randomized validation can markedly inflate deep-learning performance [10,11]. Accordingly, predictive evaluation should split participants, not epochs, and all model selection and normalization should occur within training data.

We conducted a prospectively frozen validity audit of rostrocaudal EEG complexity. Five confirmatory questions were specified: whether the published box-counting AD–FTD contrast reproduces (H1); remains after spectral and clinical adjustment (H2); survives spectrum-preserving surrogate normalization (H3); changes across eyes-closed and photic-stimulation states (H4); and adds participant-level predictive value to EEGNet (H5). We treated estimation and uncertainty as primary and regarded a higher accuracy point estimate without paired evidence of improvement as insufficient.

## Methods

### Study design and protocol freeze

This was a secondary analysis of two public, deidentified datasets. The participant was the unit of inference. The analysis protocol, hypotheses, regional definitions, sign convention, preprocessing, exclusion rules, multiplicity family, and leakage controls were frozen on July 10, 2026, before group-level outcome inspection. Timestamped operational amendments were appended without replacing the original protocol. They documented the released derivative path for the photic dataset, event-defined 2-second paired state windows, a prespecified mixed-model numerical fallback, and a CPU-executable deep-learning schedule fixed before EEGNet outcomes were computed. The complete protocol accompanies the code archive.

### Participants and datasets

Eyes-closed resting EEG was obtained from OpenNeuro dataset ds004504, version 1.0.9 [1]. The paired photic-stimulation data were obtained from ds006036, version 1.0.6 [2]. Both releases represent the same 88 participants: 36 with clinically assigned AD, 23 with FTD, and 29 CN controls. Demographic and Mini-Mental State Examination (MMSE) data were taken from the released participant tables. No participant was excluded from the eyes-closed analyses. One participant with AD (sub-011) lacked a valid event-defined photic interval and was excluded only from paired state analyses.

The original recordings used 19 scalp electrodes in the international 10–20 system: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2. The present analysis used the custodians’ released EEGLAB derivatives. The dataset descriptors report 0.5–45 Hz filtering, artifact subspace reconstruction, independent component analysis with ICLabel-assisted artifact removal, interpolation of unsuitable channels, and rereferencing [1,2]. Files were matched to the released manifests and verified with SHA-256 checksums before analysis.

### Preprocessing and regional definition

Released derivatives were loaded in a fixed canonical channel order, zero-phase filtered from 1 to 45 Hz, rereferenced to the common average, and resampled to 250 Hz. No preprocessing parameter was selected by maximizing diagnosis separation. Rostral channels were Fp1, Fp2, F7, F3, Fz, F4, and F8. Caudal channels were T5, P3, Pz, P4, T6, O1, and O2. For every feature, rostrocaudal asymmetry was defined as the mean across rostral channels minus the mean across caudal channels; positive values therefore indicate rostral dominance.

### H1: direct reproduction of box-counting asymmetry

The primary reproduction used the first fixed 15 seconds of preprocessed eyes-closed EEG. The published MATLAB `myFractal` procedure was independently translated into Python and covered by unit tests. Briefly, each channel time series was embedded in the unit square, dyadic boxes were counted over available scales, local-slope outliers were removed using the original interquartile rule, and fractal dimension was estimated from the negative slope of log box count against log box width. The primary H1 statistic was the AD-minus-FTD difference in participant-level rostrocaudal box-count dimension. HFD (kmax=32), normalized permutation entropy, sample entropy, and normalized Lempel–Ziv complexity were prespecified supporting measures. The H1 group comparison used Welch’s t test, Hedges g, and a 5,000-iteration participant bootstrap confidence interval.

### H2: spectral independence

Spectral features were calculated from the same 15-second segment using Welch power spectral density estimates with 4-second windows and 50% overlap. Relative delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz) power were computed within 1–45 Hz. Median frequency, 95% spectral edge, and log–log slope were also extracted. Periodic peaks and a fixed-mode aperiodic offset and exponent were parameterized with the specparam approach [6].

The primary H2 model was an AD–FTD ordinary least-squares regression of box-count rostrocaudal asymmetry on diagnosis, age, sex, MMSE, and rostrocaudal contrasts in aperiodic offset, aperiodic exponent, and relative delta, theta, and alpha power. Continuous covariates were standardized. HC3 heteroskedasticity-consistent standard errors were used. Because MMSE can index severity while also being closely tied to diagnosis, a model without MMSE was reported as a sensitivity analysis.

### H3: nonlinear-excess complexity

For every participant and channel, 20 deterministic IAAFT surrogates were generated from the 15-second eyes-closed segment [7]. Surrogates preserved the observed amplitude ranks and iteratively approximated the Fourier magnitude spectrum while disrupting phase organization. HFD was calculated for the observed signal and every surrogate using kmax=32 [8]. Nonlinear-excess HFD was defined as the observed HFD minus the surrogate mean, divided by the surrogate standard deviation. The primary H3 endpoint was the AD-minus-FTD difference in the rostrocaudal contrast of this z score. Inference used Welch’s test and a stratified 5,000-iteration participant bootstrap. An age-, sex-, and MMSE-adjusted HC3 regression was prespecified as a sensitivity analysis.

### H4: paired recording-state analysis

The photic dataset contains event labels for eyes-open intervals associated with incremental stimulation frequencies. For each event-defined open-eyes interval lasting at least 2 seconds, HFD was estimated from the first 2 seconds. Multiple intervals at the same frequency were averaged within participant; the primary photic estimate was then averaged across valid frequencies. The eyes-closed comparator used a duration-matched 2-second segment. The primary analysis included the 58 participants with AD or FTD who had paired state data and fit a participant random-intercept model with diagnosis, region, state, and all interactions. The confirmatory term was diagnosis-by-region-by-state. Powell optimization was used after the initial limited-memory BFGS fit reached a singular Hessian. A sensitivity model added age, sex, and MMSE. Frequency specificity at 5, 10, 15, and 20 Hz was evaluated in an adjusted repeated-measures model.

### H5: leakage-safe prediction

All models predicted three classes (AD, FTD, and CN). A single set of stratified participant-level outer folds was frozen and reused across models: 5 folds repeated 10 times. No participant appeared in both training and testing within an outer split. Classical models used a four-fold inner loop for hyperparameter selection. Scaling and model fitting were restricted to outer-training participants.

Four elastic-net multinomial logistic models provided clinical and feature baselines: demographic (age, sex, and MMSE), spectral (20 global and rostrocaudal spectral features), complexity (10 raw global and rostrocaudal complexity features plus two surrogate-HFD features), and feature fusion (spectral plus complexity). The regularization grid used C values of 0.01, 0.1, and 1.0 and L1 ratios of 0, 0.5, and 1. The demographic model was treated as a clinical-information reference, not evidence that EEG alone encoded diagnosis, because MMSE differs structurally between patient and control groups.

The raw-signal model was a compact EEGNet-style convolutional network [9]. For each participant, six evenly spaced 4-second epochs were selected from the released derivative. Channel normalization parameters were estimated from outer-training data. Each outer-training set was divided into training and validation participants; a maximum of 20 epochs and five-epoch early-stopping patience were used. Class weighting was based on training participants. Epoch logits were aggregated to one participant probability by averaging log probabilities.

The complexity-fusion model concatenated the frozen EEGNet participant embedding with a learned representation of fold-standardized complexity features. The late-fusion head was trained only on outer-training participants, with early stopping on inner validation participants. Its primary comparison with EEGNet was the difference in participant-aggregated macro one-vs-rest ROC AUC. Success required the paired participant-bootstrap 95% CI for the improvement to exclude zero. Accuracy, balanced accuracy, macro F1, and log loss were secondary.

### Statistical inference, multiplicity, and reproducibility

Effect estimates and 95% CIs were primary. Confirmatory p values from H1 through H5 formed one Holm-corrected family. Resampling operated at the participant level and preserved diagnosis strata for predictive AUCs. The H5 paired permutation test randomly swapped model probabilities within participant. Random seeds, outer split assignments, package versions, model checkpoints, and file hashes were recorded. Python 3.11 was used with MNE-Python for EEG input and signal processing [14], SciPy, pandas, statsmodels, scikit-learn, specparam, AntroPy, and PyTorch. Automated tests covered event parsing, channel layout, regional signs, complexity functions, spectral features, surrogate preservation, participant splits, and EEGNet dimensions.

## Results

### Cohort characteristics and data completeness

All 88 eyes-closed recordings passed the prespecified minimum duration criterion and contributed to H1–H3 and predictive analyses. The sample contained 36 participants with AD, 23 with FTD, and 29 CN controls (Table 1). Age was broadly similar across groups. MMSE was lower in AD than FTD and was fixed at 30 in the CN data, making it a strong but non-independent diagnostic reference. Paired state features were available for 87 participants; sub-011 had no valid photic interval. The confirmatory AD–FTD state model contained 58 participants and 232 region-by-state observations.

**Table 1. Participant characteristics.** Values are mean ± standard deviation unless otherwise indicated.

| Group | n | Age, years | Female/male | MMSE |
|---|---:|---:|---:|---:|
| AD | 36 | 66.39 ± 7.89 | 24/12 | 17.75 ± 4.50 |
| FTD | 23 | 63.65 ± 8.22 | 9/14 | 22.17 ± 2.64 |
| CN | 29 | 67.90 ± 5.40 | 11/18 | 30.00 ± 0.00 |

AD, Alzheimer’s disease; CN, cognitively normal; FTD, frontotemporal dementia; MMSE, Mini-Mental State Examination.

### H1: the reported box-counting contrast was not robustly reproduced

Mean rostrocaudal box-count asymmetry was −0.00855 in AD, −0.01866 in FTD, and −0.01856 in CN (Figure 1A). The direction of the AD–FTD difference matched the hypothesized greater rostral dominance in AD, but its magnitude was small and the interval included zero (mean difference 0.01011, 95% bootstrap CI −0.01074 to 0.03048; Welch t=0.950; Hedges g=0.267; unadjusted p=0.348). H1 was not supported. Supporting raw complexity contrasts were also uncertain: AD–FTD p values were 0.088 for HFD, 0.281 for Lempel–Ziv complexity, 0.525 for permutation entropy, and 0.864 for sample entropy.

### H2: spectral independence was not established

After adjustment for age, sex, MMSE, and rostrocaudal contrasts in relative delta, theta, alpha, aperiodic offset, and aperiodic exponent, the AD–FTD coefficient was 0.00498 (HC3 95% CI −0.00919 to 0.01916; p=0.491; n=59). Omitting MMSE produced a coefficient of 0.00197 (95% CI −0.01257 to 0.01650; p=0.791). H2 was not supported. The reduced point estimate and overlapping intervals indicate that the present data do not establish box-count asymmetry as information independent of spectral topography.

### H3: surrogate-normalized HFD did not distinguish AD and FTD

The mean rostrocaudal nonlinear-excess HFD z score was −0.077 in AD and 0.045 in FTD (Figure 1B). The AD–FTD difference was −0.122 (95% bootstrap CI −1.161 to 0.967; Hedges g=−0.064; p=0.824). H3 was not supported. An age-, sex-, and MMSE-adjusted sensitivity model yielded a negative AD coefficient (−1.306, 95% CI −2.491 to −0.121; p=0.031). This isolated adjusted result was opposite the original raw box-count direction, did not agree with the prespecified unadjusted H3 test, and was not used to claim confirmatory support.

### H4: no material diagnosis-by-region-by-state interaction was detected

HFD differed descriptively across diagnosis and recording state (Figure 1C), but the confirmatory AD–FTD diagnosis-by-region-by-state coefficient was −0.02067 (95% CI −0.10720 to 0.06586; p=0.640). Covariate adjustment produced the same estimate to the reported precision. H4 was not supported. Frequency-specific three-way contrasts relative to 5 Hz were also uncertain at 10 Hz (0.00665; p=0.787), 15 Hz (−0.00459; p=0.850), and 20 Hz (−0.01925; p=0.433). Because the state-interaction CI remained moderately wide, the null result does not demonstrate state invariance; it shows that a material state effect was not resolved in this sample.

### H5: complexity fusion did not reliably improve EEGNet

Participant-level predictive results are summarized in Table 2 and Figure 1D. EEGNet achieved a macro AUC of 0.638 (95% bootstrap CI 0.560–0.711), accuracy of 0.534, and balanced accuracy of 0.495. Frozen-EEGNet late fusion with complexity features increased the point estimate to 0.669 (95% CI 0.586–0.749), with accuracy 0.602 and balanced accuracy 0.566. The paired AUC improvement was 0.0316 (95% CI −0.0253 to 0.0911; paired permutation p=0.413), so H5 was not supported.

Among EEG-only classical baselines, the complexity model had the highest point AUC (0.685, 95% CI 0.601–0.768), followed by feature fusion (0.650, 95% CI 0.573–0.729) and spectral features (0.616, 95% CI 0.545–0.687). Feature fusion did not reliably outperform complexity alone (difference CI −0.092 to 0.019) or spectral features alone (−0.019 to 0.084). The demographic reference achieved AUC 0.924 (95% CI 0.882–0.960), largely reflecting the diagnostic information in MMSE and therefore not representing an EEG biomarker result.

**Table 2. Repeated participant-level cross-validated three-class performance.** Probabilities were averaged across 10 outer repeats before calculating participant-level metrics.

| Model | Macro AUC (95% CI) | Accuracy | Balanced accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|---:|---:|
| Demographic | 0.924 (0.882–0.960) | 0.795 | 0.797 | 0.792 | 0.402 |
| Complexity | 0.685 (0.601–0.768) | 0.568 | 0.549 | 0.542 | 0.984 |
| Spectral | 0.616 (0.545–0.687) | 0.443 | 0.422 | 0.409 | 1.013 |
| Feature fusion | 0.650 (0.573–0.729) | 0.545 | 0.521 | 0.514 | 0.989 |
| EEGNet | 0.638 (0.560–0.711) | 0.534 | 0.495 | 0.460 | 1.044 |
| EEGNet + complexity | 0.669 (0.586–0.749) | 0.602 | 0.566 | 0.549 | 0.967 |

AUC, area under the receiver-operating-characteristic curve; CI, confidence interval.

### Multiplicity-adjusted confirmatory summary

None of the five confirmatory hypotheses was supported (Table 3). Holm adjustment yielded p=1.000 for every member of the family.

**Table 3. Confirmatory hypothesis results.** Estimates are AD minus FTD unless otherwise indicated.

| Hypothesis | Estimate (95% CI) | Unadjusted p | Holm p | Supported |
|---|---:|---:|---:|:---:|
| H1 direct reproduction | 0.0101 (−0.0107 to 0.0305) | 0.348 | 1.000 | No |
| H2 spectral independence | 0.0050 (−0.0092 to 0.0192) | 0.491 | 1.000 | No |
| H3 nonlinear-excess HFD | −0.122 (−1.161 to 0.967) | 0.824 | 1.000 | No |
| H4 diagnosis × region × state | −0.0207 (−0.1072 to 0.0659) | 0.640 | 1.000 | No |
| H5 fusion minus EEGNet AUC | 0.0316 (−0.0253 to 0.0911) | 0.413 | 1.000 | No |

## Discussion

This prospectively specified validity audit did not find robust support for rostrocaudal EEG complexity as a dementia-subtype marker in the analyzed cohort. The direction of the directly translated box-count contrast was consistent with greater rostral dominance in AD than FTD, but the effect was approximately one third of the previously reported standardized magnitude and its confidence interval crossed zero. Spectral adjustment did not rescue the contrast, surrogate-normalized HFD did not distinguish AD and FTD, photic stimulation did not produce a resolved diagnosis-specific topographic change, and complexity fusion did not reliably improve EEGNet. The coherent conclusion across these tests is not that EEG contains no information about dementia. Rather, the specific claim that rostrocaudal complexity provides a robust, nonlinear, state-invariant, and incrementally predictive AD–FTD signal is not established by these data.

### Reproduction and algorithmic specificity

The prior report found a significant AD–FTD box-count asymmetry with Cohen d near 0.78 [3]. Our tested Python translation produced a smaller Hedges g of 0.27. Several factors could contribute. The published Methods contain an internally contradictory description of the sign despite a figure caption that supports rostral minus caudal; we fixed the convention before analysis. The prior analysis excluded one FTD recording, whereas all 88 eyes-closed derivatives met our frozen criterion. Box-count estimates are also sensitive to implementation details, scale selection, filtering, and short-window choice. These differences do not identify a single error in either analysis, but they show that the proposed marker is algorithmically fragile enough to require independent code, exact parameter reporting, and an external cohort before biological interpretation.

### Complexity, slowing, and surrogate controls

Complexity and spectral slowing should not be treated as automatically independent biomarker families. A smoother, more autocorrelated signal can yield lower entropy or fractal estimates without requiring a distinct nonlinear mechanism [4–6,12]. In our H2 model, adjustment for periodic and aperiodic topography reduced the AD–FTD box-count estimate and left a CI spanning effects in both directions. In H3, IAAFT normalization directly asked whether observed HFD departed from signals with nearly the same spectrum and amplitude distribution. The primary group contrast was near zero relative to its uncertainty. The adjusted H3 sensitivity coefficient was nominally nonzero, but its direction was opposite the original box-count hypothesis and it lacked agreement with the primary unadjusted test. Treating that isolated result as discovery would invert the prespecified evidential hierarchy.

IAAFT surrogates test a particular null model, not “nonlinearity” in every biological sense. They can show that a statistic is or is not unusual under a linear stochastic process with the observed marginal distribution and approximate spectrum [7]. A null surrogate result does not rule out nonlinear neuronal dynamics; it limits the interpretation of this finite-window scalp HFD contrast as evidence for such dynamics.

### Recording state and photic stimulation

The paired photic dataset is a distinctive strength because between-person confounding is reduced. Mean HFD increased during photic-open intervals in the dementia groups, but the diagnosis-specific rostrocaudal interaction was small relative to its CI. A nonsignificant interaction alone cannot establish state invariance, particularly with 58 AD/FTD participants and short event-matched windows. The result instead suggests that state dependence is not sufficiently large or stable to rescue the biomarker claim under this protocol. Future work could model full stimulation-response curves, phase locking, and steady-state visual evoked responses rather than using photic stimulation only as a perturbation of broadband complexity.

### Prediction under participant-level validation

The predictive results illustrate why participant-level evaluation is essential in small clinical EEG datasets. Under 50 held-out outer folds, EEGNet and the complexity-fusion model achieved modest three-class macro AUCs with overlapping intervals. The fusion point estimate was higher, but the paired CI included both no improvement and a practically meaningful gain. H5 therefore failed its prespecified incremental-value criterion. This performance is consistent with recent work showing that AD–FTD discrimination becomes substantially more difficult when evaluation is organized at the participant level [10,11].

The demographic reference should not be read as evidence for a deployable clinical classifier. MMSE was 30 for every CN participant and differed between AD and FTD, so the model partly recapitulates variables used in clinical characterization. Its role was to reveal how much apparent prediction could be obtained from non-EEG information. More generally, this cohort is suitable for method auditing but not for claiming clinical utility: diagnoses were clinically assigned, sample size was small, and no independent site was available.

### Strengths and limitations

Strengths include a frozen hypothesis hierarchy, exact participant-level inference, explicit sign convention, independent translation of the reported algorithm, spectral and surrogate controls, paired recordings from the same individuals, common outer splits across models, participant-level probability aggregation, and complete release of non-identifying derived outputs and code. All 88 eyes-closed files were checksum verified, and every reported model prediction was generated for a participant held out from training and model selection.

The study also has important limitations. First, both recording states came from one 88-person clinical cohort; the photic data provide within-person perturbation, not external validation. Second, diagnostic labels were not defined by a contemporary biomarker framework, and clinical subtypes and comorbidities may be heterogeneous. Third, a 19-channel montage limits spatial resolution, while common-average rereferencing can alter topographic contrasts. Fourth, the primary mechanistic features used a fixed 15-second segment; this matched the reproduction aim but may not capture temporal variability. Fifth, 20 IAAFT surrogates per channel provide a computationally feasible standardized reference but relatively coarse surrogate uncertainty. Sixth, the deep-learning schedule was deliberately compact—six 4-second epochs per person and a 20-epoch cap—to complete all 50 leakage-safe outer folds; larger schedules were not compared on an external validation set. Seventh, the H4 analysis used 2-second event-defined windows and one participant lacked a valid interval. Finally, planned secondary calibration, Brier-score, and decision-curve summaries were not advanced after the primary incremental-value criterion was null; the reported predictive conclusions rest on discrimination and proper-scoring-rule results only.

### Implications

Open datasets make rapid EEG biomarker development possible, but reuse also creates a risk that many models optimize against the same cohort while appearing independent. A credible next step is not another architecture tuned to ds004504. It is external validation with harmonized participant-level splits, complete preprocessing provenance, code-level reproduction of complexity estimators, and surrogate or generative controls that separate spectral from higher-order temporal information. Multisite cohorts should also evaluate whether spatial contrasts are stable across reference schemes, hardware, medication status, disease severity, and repeat sessions.

## Conclusions

Rostrocaudal EEG complexity did not meet any of five prespecified criteria for a robust dementia-subtype marker in this cohort. The direct box-count effect was uncertain, spectral and surrogate controls did not establish independent nonlinear information, paired photic stimulation did not resolve a diagnosis-specific state interaction, and complexity fusion did not reliably improve EEGNet under participant-level validation. The main scientific value of these results is methodological: EEG complexity claims should be accompanied by exact algorithmic reproduction, spectrum-preserving controls, perturbational testing, and externally validated subject-level prediction before clinical interpretation.

## Figure legend

**Figure 1. Reproduction, mechanistic controls, state perturbation, and leakage-safe prediction.** (A) Participant-level rostrocaudal box-count fractal-dimension asymmetry in AD, FTD, and CN. Positive values indicate rostral dominance. (B) Rostrocaudal nonlinear-excess HFD, expressed as the observed value relative to 20 IAAFT surrogates per channel. (C) Mean regional HFD in matched eyes-closed and photic-open intervals; points and lines summarize diagnosis-by-region state means rather than independent observations. (D) Participant-aggregated three-class macro one-vs-rest AUC with stratified bootstrap 95% confidence intervals. The demographic model includes MMSE and is shown as a clinical-information reference. AD, Alzheimer’s disease; AUC, area under the receiver-operating-characteristic curve; CN, cognitively normal; FTD, frontotemporal dementia; HFD, Higuchi fractal dimension; IAAFT, iterative amplitude-adjusted Fourier transform.

## Declarations

### Ethics approval and consent to participate

This study reused public, deidentified data. The original data custodians report approval by the Scientific and Ethics Committee of AHEPA University Hospital, Aristotle University of Thessaloniki [1,2]. No attempt was made to identify or contact participants. A written determination from the author’s institution regarding whether this secondary analysis is non-human-subjects research or exempt must be obtained before submission.

### Consent for publication

Not applicable to this secondary analysis of deidentified public data.

### Data availability

The source data are available under CC0 from OpenNeuro: ds004504 version 1.0.9 (https://doi.org/10.18112/openneuro.ds004504.v1.0.9) and ds006036 version 1.0.6 (https://doi.org/10.18112/openneuro.ds006036.v1.0.6). The analysis package includes the frozen protocol, configuration, subject split assignments, source code, tests, manifests, model checkpoints, and non-identifying derived results. Raw EEG is excluded from the analysis archive and should be obtained from OpenNeuro.

### Code availability

All analysis code required to reproduce the reported results is included in the accompanying versioned project archive. A public repository and archival DOI should be created before journal submission.

### Competing interests

The author must confirm the competing-interest statement before submission.

### Funding

The author must confirm funding and institutional support before submission.

### Author contributions

R.N.: conceptualization, methodology, software, formal analysis, investigation, data curation, visualization, writing—original draft, and writing—review and editing. The final contribution statement must be revised if additional authors join the project.

### Acknowledgments

The author thanks the investigators and participants responsible for the public OpenNeuro datasets. Institutional and scientific contributors should be added after authorship review.

### Generative-AI assistance

OpenAI Codex assisted with code scaffolding, test generation, literature organization, numerical report generation, and language editing. Every statistical result in the manuscript was populated from saved pipeline outputs, and the human author is responsible for verifying the code, scientific interpretation, citations, and final text. Generative AI is not listed as an author. This disclosure should be adapted to the selected journal’s policy at submission.

## References

1. Miltiadous A, Tzimourta KD, Afrantou T, Ioannidis P, Grigoriadis N, Tsalikakis DG, et al. A dataset of scalp EEG recordings of Alzheimer’s disease, frontotemporal dementia and healthy subjects from routine EEG. Data. 2023;8:95. doi:10.3390/data8060095.
2. Ntetska A, Miltiadous A, Tsipouras MG, Tzimourta KD, Afrantou T, Ioannidis P, et al. A complementary dataset of scalp EEG recordings featuring participants with Alzheimer’s disease, frontotemporal dementia, and healthy controls, obtained from photostimulation EEG. Data. 2025;10:64. doi:10.3390/data10050064.
3. Ghassemkhani K, Saroka KS, Dotta BT. Evaluating EEG complexity and spectral signatures in Alzheimer’s disease and frontotemporal dementia: evidence for rostrocaudal asymmetry. npj Aging. 2025;11:50. doi:10.1038/s41514-025-00243-y.
4. Dauwels J, Vialatte F, Cichocki A. Slowing and loss of complexity in Alzheimer’s EEG: two sides of the same coin? Int J Alzheimers Dis. 2011;2011:539621. doi:10.4061/2011/539621.
5. Jeong J. EEG dynamics in patients with Alzheimer’s disease. Clin Neurophysiol. 2004;115:1490–1505. doi:10.1016/j.clinph.2004.01.001.
6. Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, et al. Parameterizing neural power spectra into periodic and aperiodic components. Nat Neurosci. 2020;23:1655–1665. doi:10.1038/s41593-020-00744-x.
7. Schreiber T, Schmitz A. Improved surrogate data for nonlinearity tests. Phys Rev Lett. 1996;77:635–638. doi:10.1103/PhysRevLett.77.635.
8. Higuchi T. Approach to an irregular time series on the basis of the fractal theory. Physica D. 1988;31:277–283. doi:10.1016/0167-2789(88)90081-4.
9. Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J Neural Eng. 2018;15:056013. doi:10.1088/1741-2552/aace8c.
10. Del Pup F, Zanola A, Tshimanga LF, Bertoldo A, Finos L, Atzori M. The role of data partitioning on the performance of EEG-based deep learning models in supervised cross-subject analysis: a preliminary study. Comput Biol Med. 2025;196:110608. doi:10.1016/j.compbiomed.2025.110608.
11. Mlinarič T, Van Den Kerchove A, Barinaga ZI, Van Hulle MM. EEG-based classification of Alzheimer’s disease and frontotemporal dementia using functional connectivity. Sci Rep. 2026;16:4903. doi:10.1038/s41598-026-35316-9.
12. Stam CJ. Nonlinear dynamical analysis of EEG and MEG: review of an emerging field. Clin Neurophysiol. 2005;116:2266–2301. doi:10.1016/j.clinph.2005.06.011.
13. Rossini PM, Di Iorio R, Vecchio F, Anfossi M, Babiloni C, Bozzali M, et al. Early diagnosis of Alzheimer’s disease: the role of biomarkers including advanced EEG signal analysis. Report from the IFCN-sponsored panel of experts. Clin Neurophysiol. 2020;131:1287–1310. doi:10.1016/j.clinph.2020.03.003.
14. Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci. 2013;7:267. doi:10.3389/fnins.2013.00267.
