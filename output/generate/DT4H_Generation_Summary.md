# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `442d534d53f38fab51473120ea7a1c3fec6aca28` (branch `claude/readme-access-qnoz43`, **uncommitted changes present**)
- Training data: `/home/konstantinos.kechagi@mydre.org/generationV2/SyntheticDataGenerationDT4H/output/preprocess/UC1_Train.parquet`
- Training data SHA-256: `71680f98e17f78a3e09fa14aec88cf9f14fd8610e5c767578e283174e47036e6`
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
- Training split: 3520 rows x 249 columns (the holdout split is never shown to any generator)
- Trained on: 211 columns (61 continuous, 150 categorical)
- Constant columns held out and re-attached verbatim: 38
- Synthetic rows generated per run: 3520
- Width-limited (AIM) runs train on 50 outcome-relevant columns (selection: `DT4H_AIM_Column_Selection.json`)

## Runs

| run | model | ε | seed | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|---|---|
| gaussian_copula_seed0 | gaussian_copula | - | 0 | ok | 3520 x 249 | 32.2s | 0 ✅ |  |
| gaussian_copula_seed1 | gaussian_copula | - | 1 | ok | 3520 x 249 | 31.5s | 0 ✅ |  |
| gaussian_copula_seed2 | gaussian_copula | - | 2 | ok | 3520 x 249 | 31.8s | 0 ✅ |  |
| tvae_seed0 | tvae | - | 0 | ok | 3520 x 249 | 343.3s | 0 ✅ |  |
| tvae_seed1 | tvae | - | 1 | ok | 3520 x 249 | 345.0s | 0 ✅ |  |
| tvae_seed2 | tvae | - | 2 | ok | 3520 x 249 | 344.0s | 0 ✅ |  |
| ctgan_seed0 | ctgan | - | 0 | ok | 3520 x 249 | 652.8s | 0 ✅ |  |
| ctgan_seed1 | ctgan | - | 1 | ok | 3520 x 249 | 654.6s | 0 ✅ |  |
| ctgan_seed2 | ctgan | - | 2 | ok | 3520 x 249 | 654.6s | 0 ✅ |  |
| dpctgan_eps1_seed0 | dpctgan | 1 | 0 | ok | 3520 x 249 | 325.2s | 0 ✅ |  |
| dpctgan_eps5_seed0 | dpctgan | 5 | 0 | ok | 3520 x 249 | 1959.4s | 0 ✅ |  |
| dpctgan_eps8_seed0 | dpctgan | 8 | 0 | ok | 3520 x 249 | 1957.5s | 0 ✅ |  |
| dpctgan_eps10_seed0 | dpctgan | 10 | 0 | ok | 3520 x 249 | 1954.6s | 0 ✅ |  |
| dpctgan_eps15_seed0 | dpctgan | 15 | 0 | ok | 3520 x 249 | 1953.6s | 0 ✅ |  |
| dpctgan_eps20_seed0 | dpctgan | 20 | 0 | ok | 3520 x 249 | 1967.1s | 0 ✅ |  |
| dpctgan_eps15_seed1 | dpctgan | 15 | 1 | ok | 3520 x 249 | 1979.6s | 0 ✅ |  |
| dpctgan_eps15_seed2 | dpctgan | 15 | 2 | ok | 3520 x 249 | 1962.9s | 0 ✅ |  |
| aim50_eps1_seed0 | aim | 1 | 0 | ok | 3520 x 88 | 1855.3s | 0 ✅ | width-limited (50 cols) |
| aim50_eps5_seed0 | aim | 5 | 0 | ok | 3520 x 88 | 5300.4s | 0 ✅ | 101 duplicate rows within output; width-limited (50 cols) |
| aim50_eps8_seed0 | aim | 8 | 0 | failed | - | 7200.1s | - | TimeoutError: 'aim50_eps8_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| aim50_eps10_seed0 | aim | 10 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim50_eps10_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| aim50_eps15_seed0 | aim | 15 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim50_eps15_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| aim50_eps20_seed0 | aim | 20 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim50_eps20_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| mst_eps1_seed0 | mst | 1 | 0 | ok | 3520 x 249 | 10976.2s | 0 ✅ | 531 duplicate rows within output |
| mst_eps5_seed0 | mst | 5 | 0 | ok | 3520 x 249 | 10520.2s | 0 ✅ | 1553 duplicate rows within output |
| mst_eps8_seed0 | mst | 8 | 0 | ok | 3520 x 249 | 10418.4s | 0 ✅ | 1614 duplicate rows within output |
| mst_eps10_seed0 | mst | 10 | 0 | ok | 3520 x 249 | 10525.5s | 0 ✅ | 1734 duplicate rows within output |
| mst_eps15_seed0 | mst | 15 | 0 | ok | 3520 x 249 | 10397.9s | 0 ✅ | 1639 duplicate rows within output |
| mst_eps20_seed0 | mst | 20 | 0 | ok | 3520 x 249 | 10361.2s | 0 ✅ | 1623 duplicate rows within output |
| mst_eps15_seed1 | mst | 15 | 1 | ok | 3520 x 249 | 10447.2s | 0 ✅ | 1667 duplicate rows within output |
| mst_eps15_seed2 | mst | 15 | 2 | ok | 3520 x 249 | 10341.6s | 0 ✅ | 1665 duplicate rows within output |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
