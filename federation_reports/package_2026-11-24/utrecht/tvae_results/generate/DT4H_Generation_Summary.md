# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `a297d8d5f011263c41398a91e923c86734e01050` (branch `deploy`, **uncommitted changes present**)
- Training data: `/home/lorenzo.pratesi@mydre.org/cardiosynth/tvae_results/preprocess/UC1_Train.parquet`
- Training data SHA-256: `81e9ffcea9bd0fa255349d821fd368e925541a63c99a2ce566e64ef28cf5e62f`
- Python 3.10.12 on Linux-6.8.0-1070-azure-x86_64-with-glibc2.35
- GPU: Tesla T4 (CUDA 12.4)

| package | version |
|---|---|
| sdv | 1.38.0 |
| ctgan | 0.12.1 |
| smartnoise-synth | 1.0.8 |
| opendp | 0.14.2 |
| torch | 2.5.1 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| polars | 1.44.2 |
| scikit-learn | 1.7.2 |

## Data
- Training split: 1862 rows x 272 columns (the holdout split is never shown to any generator)
- Trained on: 198 columns (82 continuous, 116 categorical)
- Constant columns held out and re-attached verbatim: 74
- Synthetic rows generated per run: 1862

## Runs

| run | model | ε | δ | seed | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|---|---|---|
| tvae_seed0 | tvae | - | - | 0 | ok | 1862 x 272 | 243.8s | 0 ✅ |  |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- DP runs are bounded by the reviewed a-priori public domains in `public_domains.json` (sha-256 recorded per run above), so no epsilon is spent on -- and no bound is derived from -- the training data. Two residual, disclosed leaks remain: snsynth learns categorical vocabularies from the training data, and AIM's column selection is computed on the train split without noise.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
