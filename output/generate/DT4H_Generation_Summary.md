# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `e5597cc39964251113ce54afaff5203220461c0b` (branch `claude/readme-access-qnoz43`, **uncommitted changes present**)
- Training data: `/home/konstantinos.kechagi@mydre.org/generationV2/SyntheticDataGenerationDT4H/output/preprocess/UC1_Preprocessed.parquet`
- Training data SHA-256: `99432d169c1ce489664fc5ff3d02d372a65e7265e924525b766609d86e79e4fd`
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
- Input: 4694 rows x 249 columns
- Trained on: 211 columns (61 continuous, 150 categorical)
- Constant columns held out and re-attached verbatim: 38
- Synthetic rows generated: 4694

## Runs

| synthesizer | DP | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|
| gaussian_copula | no | ok | 4694 x 249 | 38.8s | 0 ✅ |  |
| tvae | no | ok | 4694 x 249 | 419.7s | 0 ✅ |  |
| ctgan | no | ok | 4694 x 249 | 820.6s | 0 ✅ |  |
| mst | ε=15.0 | ok | 4694 x 249 | 10017.4s | 0 ✅ | 2601 duplicate rows within output |
| dpctgan | ε=15.0 | ok | 4694 x 249 | 2538.0s | 0 ✅ |  |
| aim | ε=15.0 | failed | - | 21600.0s | - | TimeoutError: 'aim' fit exceeded the 21600s time limit |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates and does not bound membership-inference risk; a full privacy assessment (distance-to-closest-record, membership inference) is still required before release.
- Non-DP models carry no formal privacy guarantee regardless of this check.
