# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `c8b6d0241fa03d791e5d76237aa7266bef0c68f0` (branch `claude/readme-access-qnoz43`, **uncommitted changes present**)
- Training data: `/home/konstantinos.kechagi@mydre.org/generationV2/SyntheticDataGenerationDT4H/output/preprocess/UC1_Preprocessed.parquet`
- Training data SHA-256: `a9ffcc10fec5f6e5ecc54a4aaafdf1e0be3f1d812ba12eeb15792c375560e557`
- Python 3.10.12 on Linux-6.8.0-1064-azure-x86_64-with-glibc2.35
- GPU: Tesla T4 (CUDA 13.0)

| package | version |
|---|---|
| sdv | 1.38.0 |
| ctgan | 0.12.1 |
| smartnoise-synth | 1.0.8 |
| opendp | 0.14.2 |
| torch | 2.13.0 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| polars | 1.43.2 |
| scikit-learn | 1.7.2 |

## Data
- Input: 4694 rows x 329 columns
- Trained on: 291 columns (73 continuous, 218 categorical)
- Constant columns held out and re-attached verbatim: 38
- Synthetic rows generated: 4694

## Runs

| synthesizer | DP | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|
| gaussian_copula | no | ok | 4694 x 329 | 57.2s | 0 ✅ |  |
| tvae | no | ok | 4694 x 329 | 559.0s | 0 ✅ |  |
| ctgan | no | ok | 4694 x 329 | 1134.7s | 0 ✅ |  |
| mst | ε=15.0 | failed | - | 4.5s | - | ValueError: BinTransformer could not find bounds. |
| aim | ε=15.0 | failed | - | 1.7s | - | ValueError: BinTransformer could not find bounds. |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates and does not bound membership-inference risk; a full privacy assessment (distance-to-closest-record, membership inference) is still required before release.
- Non-DP models carry no formal privacy guarantee regardless of this check.
