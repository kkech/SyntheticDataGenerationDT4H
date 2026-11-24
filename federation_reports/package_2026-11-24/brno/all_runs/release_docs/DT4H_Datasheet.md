# Datasheet: DT4H UC1 Synthetic Heart-Failure Cohort

Structure follows *Datasheets for Datasets* (Gebru et al., 2021).

## Motivation
- **Purpose**: a privacy-preserving synthetic version of the DataTools4Heart
  (DT4H) Use Case 1 heart-failure cohort, enabling method development,
  benchmarking, education and analysis piloting without access to patient-level
  data.
- **Context**: created within the DataTools4Heart project, a European
  multi-partner initiative building a federated cardiology data toolbox. The
  synthetic release lets researchers outside the secure environment work with
  realistic UC1-shaped data while the real records never leave the provider.

## Composition
- Synthetic patient-level records: 1549
  rows x 256 columns per released file, matching the training-split schema.
- Source (never released): 1549 training records (25% of the
  cohort held out for evaluation and never shown to any generator).
- Column semantics: see `DT4H_Codebook.md`. Missingness is preserved by design and
  carries meaning (structural "no event" vs "not measured").
- No real patient records or identifiers are included. Verbatim-row check --
  verified: zero exact training-row reproductions across all 45 assessed released file(s) (DT4H_Privacy_Assessment.json).

## Collection & preprocessing
- Source data were extracted from the electronic health records of the providing
  DT4H clinical partner site under the project's federated data protocol
  (standardized onFHIR/Feast feature extraction; see the feature-set metadata),
  and delivered 2026-08-12 into the project's secure research environment.
- All processing -- including generator training -- took place inside that secure
  environment (an isolated analysis workspace with no patient-level data export);
  only aggregate statistics, synthetic records and reports leave it. Processing
  is governed by the project's data-sharing and governance agreements between
  the consortium partners.
- Preprocessing is fully scripted (see `DT4H_Preprocessing_Summary.md`). NOT VERIFIED IN THIS BUILD: no per-column KS/TVD-vs-raw distribution-preservation check is recorded, so no distribution-preserving guarantee is asserted here.
- Generators: see the run plan in `DT4H_Generation_Summary.md` (seeds, epsilon
  values, library versions, git commit `c349cdf576dec77b926a946db0ffa846a5f3ca9b`,
  training-file SHA-256).

## Uses
- Intended: methods development, education, benchmarking, pre-analysis piloting.
- Cautions: effect estimates and model performance are attenuated relative to real
  data (quantified in `DT4H_Utility_TSTR.md` and `DT4H_Survival_Fidelity.md`);
  fidelity is weakest for sparsely observed laboratory values. Synthetic data must
  not be represented as real patients in any publication.

## Privacy & release gating
- Record-level distances, membership-inference and attribute-inference attacks are
  reported in `DT4H_Privacy_Assessment.md` and `DT4H_Privacy_Attacks.md`, all
  evaluated against a genuine unseen-patient baseline.
- DP-labelled files record a per-run (epsilon, delta) (epsilon in {0.5, 1, 5, 8, 10, 15, 20}, delta in {1.6403e-05}). Public column domains are taken from a reviewed public metadata source (public_domains.json (not yet loaded), public_domains.json rev 7b486344dcd7a66aae7464c179c9885eaed2e83e9420fd836bcbf7abc87288bf), so no epsilon is spent discovering them. Categorical vocabularies and the AIM column selection are data-dependent and are disclosed as such.
- Every file must pass `release_gate.py` before distribution.

## Distribution & maintenance
- **Hosting**: the project repository
  (github.com/kkech/SyntheticDataGenerationDT4H) carries the pipeline, all
  evaluation reports and the release documentation; vetted synthetic files are
  added there deliberately after passing `release_gate.py`. An archival deposit
  with a DOI (Zenodo) will be minted for the exact release accompanying the
  publication.
- **License**: documentation, reports and code are released openly with the
  repository; the synthetic data files are intended for release under CC BY 4.0,
  subject to final consortium approval.
- **Point of contact**: the repository maintainers, via GitHub issues on the
  repository above.
- **Versioning**: every release is reproducible from its recorded git commit,
  seeds and training-file SHA-256 (see the generation summary and
  `DT4H_Environment_Freeze.txt`); releases are tagged in git and superseded
  versions remain available in history.
- **Retraction**: should any privacy or integrity concern be identified, the
  affected files will be removed from the repository and archival deposit, the
  release tag withdrawn, and the concern documented in the repository.
