# Compact EEG amendment results

## Main research question

Can a low-parameter, physiologically interpretable EEG model distinguish AD,
FTD, and cognitively normal participants, and do compact spectral or
rostrocaudal-complexity additions improve accuracy without weakening external
AD-versus-control transportability?

## Hypotheses

The working hypothesis was that the locked three-feature slowing model would
provide a transferable baseline, and that compact spectral parameterization and
rostrocaudal complexity features might improve three-class discrimination while
preserving external performance.

## Results

The internally validated three-feature baseline reached macro ROC-AUC
0.578 (95% CI 0.500 to
0.648) across 88 AHEPA
participants. The highest internal point estimate was **spectral**
with macro ROC-AUC 0.657 (95% CI
0.580 to 0.732), a difference
of +0.079 (95% CI
-0.001 to +0.161) from the baseline.
FTD remained the weakest class even in the best model (one-versus-rest AUC
0.447).

In independent P-ADIC AD/CN validation, the locked baseline reached ROC-AUC
0.702 (95% CI 0.615 to
0.785). The shared five-feature spectral
extension reached 0.705 (95% CI
0.619 to
0.786). The paired AUC difference was
+0.003 (95% CI
-0.028 to
+0.036). The extension reduced the
Brier score from 0.345 to 0.272,
but transported specificity remained poor (0.406).

## Interpretation

These are exploratory amendment results. The spectral panel is a reasonable
candidate for a future independently locked test because its internal estimate
improved, but this dataset provides no evidence that it improves external AD/CN
discrimination. The complexity additions did not improve on the spectral-only
model. The external cohort validates AD versus controls only; it does not
externally validate FTD classification. The next confirmatory study requires a
new independent cohort with FTD and must lock the selected panel before testing.
