# Literature audit and publication positioning

Audit date: 2026-07-10

## Decision

The original idea, “train EEGNet with rostrocaudal complexity asymmetry to classify AD, FTD, and CN,” is feasible but no longer sufficiently novel as a standalone paper.

Three developments changed the positioning:

1. Ghassemkhani et al. published the rostrocaudal complexity result on `ds004504` in *npj Aging* in 2025.
2. Multiple 2025–2026 papers applied deep or hybrid classifiers, explainability, and cross-condition evaluation to `ds004504` and its paired photostimulation dataset `ds006036`.
3. A 2026 *Scientific Reports* paper demonstrated that strict participant-level validation produces materially more modest and credible performance than many epoch-randomized reports, especially for AD versus FTD.

The project is therefore positioned as a mechanistic replication and validity audit, with classification as a secondary translational test.

## Primary novelty

### 1. Spectral-surrogate decomposition

The key question is whether the reported “complexity” asymmetry contains nonlinear temporal information beyond the amplitude distribution and power spectrum. IAAFT surrogates preserve both as closely as possible while disrupting higher-order temporal organization. The primary metric is observed-minus-surrogate standardized complexity, not raw HFD alone.

### 2. Explicit spectral mediation/confounding analysis

AD-related EEG slowing and reduced apparent complexity are known to be strongly coupled. The analysis jointly models relative band power, the aperiodic spectrum, and complexity rather than treating them as independent biomarker families.

### 3. Paired state-dependence test

The same 88 participants were recorded with eyes closed and during incremental 5–30 Hz photic stimulation. This permits a within-participant test of whether diagnostic topography is stable or state-dependent.

### 4. Leakage-safe cross-state prediction

Because the two recordings share participants, simply training on one and testing on the other can retain person-specific EEG signatures. The primary cross-state test holds out the same participants from all training data. A deliberately identity-leaky comparison may be included only as a cautionary methodological demonstration.

### 5. Incremental-value criterion

Fusion is considered useful only if it improves participant-level nested cross-validated macro ROC-AUC over EEGNet with a paired confidence interval excluding zero. A higher point estimate without uncertainty is not sufficient.

## Core sources

- Miltiadous et al. (2023), dataset descriptor: <https://doi.org/10.3390/data8060095>
- OpenNeuro eyes-closed dataset: <https://openneuro.org/datasets/ds004504/versions/1.0.9>
- Ntetska et al. (2025), paired photostimulation dataset descriptor: <https://doi.org/10.3390/data10050064>
- OpenNeuro photostimulation dataset: <https://openneuro.org/datasets/ds006036/versions/1.0.6>
- Ghassemkhani et al. (2025), rostrocaudal complexity asymmetry: <https://doi.org/10.1038/s41514-025-00243-y>
- Dauwels et al. (2011), coupling of EEG slowing and complexity loss in AD: <https://doi.org/10.4061/2011/539621>
- Lawhern et al. (2018), EEGNet: <https://doi.org/10.1088/1741-2552/aace8c>
- Mlinarič et al. (2026), subject-level connectivity classification and leakage discussion: <https://doi.org/10.1038/s41598-026-35316-9>
- Del Pup et al. (2025), effect of EEG data partitioning: <https://doi.org/10.1016/j.compbiomed.2025.110608>
- Higuchi (1988), original fractal-dimension method: <https://doi.org/10.1016/0167-2789(88)90081-4>

## Journal strategy

Journal placement must follow the actual results.

- **Stretch target:** *Alzheimer’s & Dementia: Diagnosis, Assessment & Disease Monitoring* if the surrogate-normalized effect replicates, state-dependence is informative, and predictive improvement is credible.
- **Strong field target:** *NeuroImage: Clinical* or *Clinical Neurophysiology* if the main contribution is a rigorous mechanistic and methodological analysis.
- **Realistic open-access target:** *Alzheimer’s Research & Therapy*, *Journal of Neural Engineering*, or *Scientific Reports*, depending on result strength and emphasis.
- **Negative-but-important result:** a reproducibility or methods-oriented venue if the apparent complexity effect disappears under spectral-surrogate control.

No journal acceptance can be guaranteed. PubMed is an indexing database, not a journal; the goal is a PubMed-indexed journal with a defensible contribution.

## Claims that are prohibited without additional evidence

- “early diagnosis” or “preclinical detection” (the datasets do not contain prodromal/longitudinal labels);
- “clinical-grade,” “clinically validated,” or “ready for deployment”;
- “external FTD validation” (both FTD recordings come from the same cohort);
- treating epochs as independent patients;
- reporting cross-state performance when train and test contain the same participant as evidence of generalization;
- selecting preprocessing, `kmax`, windows, or model hyperparameters on the held-out test results.

