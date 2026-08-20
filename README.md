# SyntheticDataGenerationDT4H

A reproducible pipeline for generating and evaluating synthetic versions
of the DataTools4Heart (DT4H) UC1 heart-failure cohort, with and without
differential privacy, intended to support a public dataset release.

## Pipeline

```
python main.py
```

Seven steps, run in order, each skipped once completed (tracked in
`pipeline_status.json`) unless forced:

| # | step | what it does | output |
|---|------|--------------|--------|
| 1 | `load_data` | concatenates the transfer's Spark `part-*.parquet` files into the full dataset | `output/load_data/` (local only) |
| 2 | `profile_data` | privacy-safe per-column statistics of the raw data (rare values suppressed), metadata copy, seeded row sample | `output/profile_data/` |
| 3 | `preprocess` | metadata-driven feature engineering; **no imputation anywhere** (see Principles); hard-fails if any null/NaN survives | `output/preprocess/` |
| 4 | `profile_preprocessed_data` | same profiler over the training frame | `output/profile_preprocessed_data/` |
| 5 | `generate` | trains every configured synthesizer, writes synthetic CSVs, saves fitted generators, checks for verbatim training records | `output/generate/` |
| 6 | `evaluate` | marginal fidelity (KS, Wasserstein, TVD, missingness) **and** pairwise association structure (Spearman, Cramer's V, correlation ratio) across original / preprocessed / synthetic | `output/evaluate/` |
| 7 | `privacy` | distance-to-closest-record and nearest-neighbor-distance-ratio per synthesizer against a real-to-real baseline | `output/privacy/` |

### CLI

```
python main.py --preflight          # verify libs, GPU, inputs, disk BEFORE a long run
python main.py --status             # step completion status
python main.py --force              # rerun everything
python main.py --force-step generate --force-step evaluate --force-step privacy
python main.py --only evaluate --force-step evaluate
python regenerate.py --model output/generate/models/<name>.pkl --rows N --out file.csv
```

All console output (stdout, stderr, warnings) is teed to `logs.txt` with
per-line timestamps.

## Synthesizers

Configured in `pipeline/config.py` (`synthesizers`), executed
cheapest/most-reliable first so a late failure or timeout costs only the
tail of a run:

| name | library | DP | notes |
|------|---------|----|-------|
| `gaussian_copula` | SDV | no | statistical baseline, seconds |
| `tvae` | SDV | no | usually the strongest non-DP model here |
| `ctgan` | SDV | no | the long-standing GAN baseline |
| `mst` | smartnoise-synth | ε | marginal-based; excellent categorical fidelity |
| `dpctgan` | smartnoise-synth | ε | DP-GAN comparison point |
| `aim` | smartnoise-synth | ε | strongest DP method in the literature, but Private-PGM scales poorly with column count — runs last, bounded by a per-model timeout |

Every model run records full provenance: seed, library versions, git
commit, hardware, and the SHA-256 of the exact training file.

## Principles

- **Missingness is information, never noise.** Numeric nulls are not
  imputed: time-to-event nulls mean *the event never happened*, lab
  nulls mean *not measured* (and clinicians measure selectively). Both
  are encoded as per-column sentinels below the observed range, modelled
  jointly by the synthesizer, and decoded back to null in the output —
  so synthetic patients have realistic missingness patterns. Categorical
  nulls become an explicit `Missing` category.
- **Nothing is fabricated.** No placeholder values, no bootstrap fills.
  The evaluation proves preprocessing is distribution-preserving
  (original vs preprocessed: KS = 0, TVD = 0 on every common column).
- **The pipeline is the only path that produces files.** Reports verify
  files as they are on disk and flag stale ones loudly; nothing is ever
  silently repaired.
- **Reports cannot look better than the data.** Leakage is checked in
  sentinel space before decoding; every claim in a summary is computed,
  not assumed.

## Data & privacy boundaries

Committed to git: statistics, summaries, metadata, small seeded samples,
evaluation and privacy reports (aggregates only). **Never committed**
(`.gitignore`): the full patient-level parquets, synthetic CSVs pending
review, and fitted generator pickles (a trained generator memorizes
aspects of the real data and is treated as sensitively as the data
itself).

The privacy step's DCR/NNDR analysis bounds record-copying; it is not a
membership-inference evaluation (that requires a training holdout — see
the limitations section of `output/privacy/DT4H_Privacy_Assessment.md`).
DP synthesizers carry their ε guarantee by construction.

## Setup

Python 3.10+, a CUDA GPU recommended for the GAN/VAE models.

```
python -m venv .synthenv && source .synthenv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python main.py --preflight
```
