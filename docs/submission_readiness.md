# Self-publication readiness

## Scientific status

The repository contains four clearly separated analyses:

1. a completed negative rostrocaudal-complexity validity audit;
2. a locked one-way external AD/CN transportability test;
3. a dated exploratory compact spectral and complexity amendment;
4. a dated exploratory paired resting-and-photic AD/FTD/CN amendment.

The external slowing index achieved modest discrimination but poor calibration.
The paired photic candidate improved internal macro AUC, while its FTD operating
point remained inadequate. No result is a clinical diagnostic claim.

## Public release checklist

- Source datasets are identified by repository, version, DOI, and license.
- Raw EEG and local environments are excluded from git.
- Frozen and exploratory analyses are labeled separately.
- Participant-level validation and leakage controls are documented.
- Machine-readable results, predictions, figures, and manuscript drafts are
  included.
- Tests and lint run without participant data.
- `CITATION.cff`, code license, data-license boundaries, and release notes are
  included.
- The release is tagged only after the public branch and CI checks agree.

## Claim rules

- Do not describe internal paired analysis as external validation.
- Do not claim state-of-the-art performance.
- Do not use epoch counts as the inferential sample size.
- Report calibration, class-specific behavior, and uncertainty alongside AUC.
- Preserve negative and null results.
- State that P-ADIC validates AD/CN only and cannot validate FTD or the paired
  photic model.

## Independent review

Self-publication does not replace scientific review. Readers should be able to
audit code, source manifests, validation units, model-selection boundaries,
and numerical claims. Corrections should be versioned transparently and should
not overwrite the historical protocol or dated amendments.

No university affiliation, ethics determination, funding source, or clinical
endorsement is claimed by this repository. Anyone extending the work must meet
their own institutional, legal, and data-governance obligations.
