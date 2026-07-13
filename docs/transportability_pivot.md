# Project pivot memo — 2026-07-13

The first study was a prespecified validity audit of rostrocaudal complexity.
Its null results are scientifically interpretable and remain preserved. They
are not being rewritten to appear positive.

The next study asks a different question with a stronger route to clinical
utility: whether a very small, interpretable EEG slowing panel transports from
one routine-EEG cohort to an independent institution. The novelty is the
locked external validation, calibration, threshold utility, and explicit
dataset-shift analysis—not a new deep-learning architecture.

The proposed external cohort is P-ADIC (`10.5061/dryad.8gtht76pw`), a Dryad
release with separate Alzheimer and control MATLAB files. The raw files are
large and excluded from version control. The acquisition script records their
published sizes and SHA-256 digests before analysis.

## Decision rules

- If the independent cohort can be aligned without unverifiable assumptions,
  it becomes the external test set.
- If alignment fails, we report that failure and amend the protocol before
  selecting another cohort.
- No feature search, threshold tuning, or architecture search is allowed on
  the external labels.
- The final paper will report the result whether positive or null.
