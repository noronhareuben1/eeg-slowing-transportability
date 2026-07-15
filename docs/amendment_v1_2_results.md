# Paired perturbational EEG amendment results

## Decision summary

The paired two-stage model is the exploratory winner. It improved overall
participant-level AD/FTD/CN discrimination relative to resting EEG alone under
10 x 5 repeated nested validation. The FTD point estimate also improved, but
its paired uncertainty interval still includes zero and default-threshold FTD
sensitivity remains poor. This is a candidate for a newly locked external
study, not a diagnostic model.

## Main research question

Does a leakage-safe, two-stage model combining resting EEG topography with a
frequency-resolved photic-response fingerprint improve participant-level
AD/FTD/CN discrimination compared with resting EEG alone?

## Main hypothesis

The paired two-stage model will improve macro one-versus-rest ROC-AUC, with a
secondary improvement in FTD one-versus-rest AUC, because AD and FTD differ not
only in resting slowing but in their spatially distributed response to
frequency-specific visual drive.

## Results

The analysis included 87 participants (35 AD, 23 FTD, 29 CN). Every model was
evaluated on the same 50 outer participant folds, with all feature selection,
imputation, scaling, and tuning restricted to training participants.

| Model | Macro AUC (95% CI) | AD AUC | CN AUC | FTD AUC | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Resting direct | 0.663 (0.582-0.744) | 0.741 | 0.798 | 0.451 | 0.514 | 0.472 |
| Paired direct | 0.724 (0.657-0.789) | 0.753 | 0.907 | 0.512 | 0.546 | 0.521 |
| **Paired two-stage** | **0.777 (0.723-0.829)** | **0.804** | **0.914** | **0.611** | **0.578** | 0.541 |
| Paired hybrid | 0.753 (0.695-0.808) | 0.781 | 0.914 | 0.563 | **0.610** | **0.580** |

For the selected two-stage model, the macro-AUC improvement over resting was
**+0.113** (95% paired bootstrap CI, **+0.035 to +0.191**), supporting the
overall hypothesis. FTD AUC improved by **+0.160**, but its CI was **-0.012 to
+0.330**; the FTD-specific hypothesis is therefore not confirmed.

At the default maximum-probability decision rule, the selected model correctly
classified 74.3% of AD, 86.2% of CN, and only 13.0% of FTD participants. The
model ranks FTD better than the resting comparator, but its operating point is
not clinically acceptable and must not be presented as a diagnostic test.

## What the model repeatedly selected

The dementia-versus-control head was dominated by resting posterior theta and
complexity features, including caudal Higuchi FD and theta at O1/O2/P3/Fz. The
AD-versus-FTD head repeatedly selected frequency-specific photic features,
especially 20 Hz median frequency at C3 (selected in 50/50 folds), temporal and
central median-frequency responses, 5 Hz temporal HFD and beta/gamma power, and
20 Hz parietal aperiodic exponent. This separation is biologically coherent
with a first-stage slowing screen followed by a perturbational subtype test,
but selection stability is mechanistic evidence only, not causal proof.

## Position against current papers

This model does **not** outperform the highest recently reported score on this
benchmark. Saeed and Aldera reported macro AUC 0.99 and 87.5% accuracy for a
subject-level AutoSSM-ICA-EEG framework using the photic dataset. Jiang and
colleagues reported 94.32% three-class accuracy with Coherence-CNN. Other
high-scoring papers report 90% or greater accuracy using Hjorth or connectivity
features, although validation units and feature-selection procedures differ.

The more comparable strict subject-level functional-connectivity study by
Mlinaric and colleagues found AD-FTD discrimination most difficult (best
pairwise AUC about 0.71) and explicitly warned that epoch-level and fixed-split
studies can overestimate performance. The 2026 AHEPA scoping review likewise
reported that accuracy decreases as validation rigor increases.

Accordingly, the contribution here is not a new state-of-the-art accuracy
claim. The candidate novelty is the explicit combination of a resting trait
with a frequency-resolved photic perturbation fingerprint, spectral
parameterization, spatial gradients, and separate dementia/subtype heads under
repeated participant-level nested validation. A focused literature review found
older AD photic-driving studies and recent models using the photic dataset, but
did not identify a published AD/FTD/CN classifier with this exact
frequency-response and two-stage design. That novelty statement must remain
"to our knowledge" until a formal systematic search is completed.

## Concrete next study

Lock a reduced panel before accessing a new cohort:

1. retain stable resting posterior theta/complexity features for dementia
   screening;
2. retain stable central-temporal 5 and 20 Hz response features for AD/FTD
   subtyping;
3. freeze preprocessing, missing-data handling, feature count, model family,
   calibration, and decision thresholds;
4. test once in an independent multisite cohort containing AD, FTD, and CN with
   the same photic protocol;
5. report macro AUC, FTD AUC, calibration, class sensitivity/specificity, and
   decision-curve utility, with no model changes after test labels are opened.

The next self-published protocol will lock this reduced panel before a new
paired AD/FTD/CN cohort is opened. Any future collection or restricted-data
access must obtain the determinations required by the responsible institution.

## Reproducible outputs

- `transportability/photic_response_features.py`
- `transportability/run_amendment_v1_2.py`
- `outputs/amendment_v1_2/photic_response_features.csv`
- `outputs/amendment_v1_2/outer_predictions.csv`
- `outputs/amendment_v1_2/inner_tuning.csv`
- `outputs/amendment_v1_2/feature_selection_frequency.csv`
- `outputs/amendment_v1_2/results.json`
- `outputs/amendment_v1_2/table_models.csv`
- `outputs/amendment_v1_2/figure_paired_response_results.png`

## Key literature checked 14 July 2026

1. Saeed F, Aldera S. Explainable EEG-based machine learning for early
   diagnosis of Alzheimer's disease and frontotemporal dementia. *Scientific
   Reports*. 2026. doi:10.1038/s41598-026-57069-1.
2. Mlinaric T, et al. EEG-based classification of Alzheimer's disease and
   frontotemporal dementia using functional connectivity. *Scientific
   Reports*. 2026. doi:10.1038/s41598-026-35316-9.
3. Jiang R, et al. Classification for Alzheimer's disease and frontotemporal
   dementia via resting-state EEG-based coherence and convolutional neural
   network. *Cognitive Neurodynamics*. 2025. doi:10.1007/s11571-025-10232-2.
4. Lee DG, Lee SB. Diagnosis of Alzheimer's Disease and Frontotemporal Dementia
   From Electroencephalography Signals. *IEEE TNSRE*. 2025.
   doi:10.1109/TNSRE.2025.3575840.
5. The AHEPA EEG benchmark: setting the standard for machine learning in
   dementia diagnosis, a scoping review. 2026. PMID:42165009.
