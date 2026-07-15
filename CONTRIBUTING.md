# Contributing

Corrections and outcome-independent validation are welcome.

1. Keep raw EEG, participant files, credentials, and signed download URLs out
   of git.
2. Label every change as a protocol correction, implementation correction, or
   explicitly exploratory extension.
3. Keep every participant and all of their epochs in one validation fold.
4. Fit preprocessing, imputation, selection, scaling, tuning, and calibration
   using training participants only.
5. Add or update a synthetic-data test, then run `ruff check src tests
   transportability` and `pytest`.
6. Preserve negative findings and avoid diagnostic or state-of-the-art claims.

Use coded or synthetic examples in issues. Do not upload human-subject files.
