# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `a297d8d5f011263c41398a91e923c86734e01050` (branch `deploy`, **uncommitted changes present**)
- Training data: `/home/dt4h/cardiosynth/mst_results/preprocess/UC1_Train.parquet`
- Training data SHA-256: `1c4eedaddaa85dff071806a1b133f71b228958cb7f80166882ac9cb26b9e734c`
- Python 3.10.12 on Linux-6.8.0-138-generic-x86_64-with-glibc2.35
- GPU: NVIDIA RTX A4000 (CUDA 12.4)

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
- Training split: 602 rows x 253 columns (the holdout split is never shown to any generator)
- Trained on: 182 columns (62 continuous, 120 categorical)
- Constant columns held out and re-attached verbatim: 71
- Synthetic rows generated per run: 602

## Runs

| run | model | ε | δ | seed | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|---|---|---|
| mst_eps15_seed0 | mst | 15 | 6.77e-05 | 0 | ok | 602 x 253 | 3629.3s | 0 ✅ | 94 duplicate rows within output; bounds `111791dfa11e` |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- DP runs are bounded by the reviewed a-priori public domains in `public_domains.json` (sha-256 recorded per run above), so no epsilon is spent on -- and no bound is derived from -- the training data. Two residual, disclosed leaks remain: snsynth learns categorical vocabularies from the training data, and AIM's column selection is computed on the train split without noise.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
