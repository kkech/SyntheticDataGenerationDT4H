# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `708090ff254eb667e65b55f03aa08cfbebfc01e4` (branch `deploy`, **uncommitted changes present**)
- Training data: `/home/yugurses/cardiosynth/mst_results/preprocess/UC1_Train.parquet`
- Training data SHA-256: `2baeed8fe109f124ce4395006ab7406ad8817373c9ef7e075d90def70cb63dc6`
- Python 3.10.12 on Linux-6.8.0-1070-azure-x86_64-with-glibc2.35
- GPU: Tesla T4 (CUDA 12.4)

| package | version |
|---|---|
| sdv | 1.38.0 |
| ctgan | 0.12.1 |
| smartnoise-synth | 1.0.8 |
| opendp | 0.14.2 |
| torch | 2.5.1 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| polars | 1.44.2 |
| scikit-learn | 1.7.2 |

## Data
- Training split: 754 rows x 257 columns (the holdout split is never shown to any generator)
- Trained on: 220 columns (65 continuous, 155 categorical)
- Constant columns held out and re-attached verbatim: 37
- Synthetic rows generated per run: 754

## Runs

| run | model | ε | δ | seed | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|---|---|---|
| mst_eps15_seed0 | mst | 15 | 4.83e-05 | 0 | ok | 754 x 257 | 12659.9s | 0 ✅ | 21 duplicate rows within output; bounds `111791dfa11e` |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- DP runs are bounded by the reviewed a-priori public domains in `public_domains.json` (sha-256 recorded per run above), so no epsilon is spent on -- and no bound is derived from -- the training data. Two residual, disclosed leaks remain: snsynth learns categorical vocabularies from the training data, and AIM's column selection is computed on the train split without noise.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
