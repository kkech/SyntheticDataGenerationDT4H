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
- Synthetic patient-level records: 3520
  rows x 249 columns per released file, matching the training-split schema.
- Source (never released): 3520 training records (25% of the
  cohort held out for evaluation and never shown to any generator).
- Column semantics: see `DT4H_Codebook.md`. Missingness is preserved by design and
  carries meaning (structural "no event" vs "not measured").
- No real patient records, identifiers, or verbatim rows are included (verified:
  zero exact training-row reproductions in every released file).

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
- Preprocessing is provably distribution-preserving (KS = 0, TVD = 0 vs raw on all
  retained columns) and fully scripted; see `DT4H_Preprocessing_Summary.md`.
- Generators: see the run plan in `DT4H_Generation_Summary.md` (seeds, epsilon
  values, library versions, git commit `8e6809374b5caec2dc52369eb4c1bb5781e123a9`,
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
- DP-labelled files carry a formal (epsilon, delta) guarantee by construction;
  column domains are treated as public metadata (released in the encoding map).
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
