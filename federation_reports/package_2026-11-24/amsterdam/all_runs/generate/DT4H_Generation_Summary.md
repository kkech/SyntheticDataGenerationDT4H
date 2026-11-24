# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `8e6809374b5caec2dc52369eb4c1bb5781e123a9` (branch `claude/bounded-improvements`, **uncommitted changes present**)
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
| gaussian_copula_seed0 | gaussian_copula | - | 0 | ok | 3520 x 249 | 33.9s | 0 ✅ |  |
| gaussian_copula_seed1 | gaussian_copula | - | 1 | ok | 3520 x 249 | 32.8s | 0 ✅ |  |
| gaussian_copula_seed2 | gaussian_copula | - | 2 | ok | 3520 x 249 | 32.8s | 0 ✅ |  |
| tvae_seed0 | tvae | - | 0 | ok | 3520 x 249 | 354.3s | 0 ✅ |  |
| tvae_seed1 | tvae | - | 1 | ok | 3520 x 249 | 355.4s | 0 ✅ |  |
| tvae_seed2 | tvae | - | 2 | ok | 3520 x 249 | 355.4s | 0 ✅ |  |
| ctgan_seed0 | ctgan | - | 0 | ok | 3520 x 249 | 673.1s | 0 ✅ |  |
| ctgan_seed1 | ctgan | - | 1 | ok | 3520 x 249 | 674.6s | 0 ✅ |  |
| ctgan_seed2 | ctgan | - | 2 | ok | 3520 x 249 | 668.9s | 0 ✅ |  |
| dpctgan_eps1_seed0 | dpctgan | 1 | 0 | ok | 3520 x 249 | 331.3s | 0 ✅ |  |
| dpctgan_eps5_seed0 | dpctgan | 5 | 0 | ok | 3520 x 249 | 1983.9s | 0 ✅ |  |
| dpctgan_eps8_seed0 | dpctgan | 8 | 0 | ok | 3520 x 249 | 1985.8s | 0 ✅ |  |
| dpctgan_eps10_seed0 | dpctgan | 10 | 0 | ok | 3520 x 249 | 1984.9s | 0 ✅ |  |
| dpctgan_eps15_seed0 | dpctgan | 15 | 0 | ok | 3520 x 249 | 1986.5s | 0 ✅ |  |
| dpctgan_eps20_seed0 | dpctgan | 20 | 0 | ok | 3520 x 249 | 1995.8s | 0 ✅ |  |
| dpctgan_eps15_seed1 | dpctgan | 15 | 1 | ok | 3520 x 249 | 1997.3s | 0 ✅ |  |
| dpctgan_eps15_seed2 | dpctgan | 15 | 2 | ok | 3520 x 249 | 2261.0s | 0 ✅ |  |
| aim50_eps1_seed0 | aim | 1 | 0 | ok | 3520 x 88 | 2119.4s | 0 ✅ | width-limited (50 cols) |
| aim50_eps5_seed0 | aim | 5 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim50_eps5_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| aim50_eps8_seed0 | aim | 8 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim50_eps8_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| aim50_eps10_seed0 | aim | 10 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim50_eps10_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| aim50_eps15_seed0 | aim | 15 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim50_eps15_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| aim50_eps20_seed0 | aim | 20 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim50_eps20_seed0' fit exceeded the 7200s time limit; width-limited (50 cols) |
| mst_eps1_seed0 | mst | 1 | 0 | ok | 3520 x 249 | 11105.2s | 0 ✅ | 698 duplicate rows within output |
| mst_eps5_seed0 | mst | 5 | 0 | ok | 3520 x 249 | 10546.5s | 0 ✅ | 1532 duplicate rows within output |
| mst_eps8_seed0 | mst | 8 | 0 | ok | 3520 x 249 | 10460.0s | 0 ✅ | 1587 duplicate rows within output |
| mst_eps10_seed0 | mst | 10 | 0 | ok | 3520 x 249 | 10398.8s | 0 ✅ | 1588 duplicate rows within output |
| mst_eps15_seed0 | mst | 15 | 0 | ok | 3520 x 249 | 10368.5s | 0 ✅ | 1687 duplicate rows within output |
| mst_eps20_seed0 | mst | 20 | 0 | ok | 3520 x 249 | 10412.7s | 0 ✅ | 1638 duplicate rows within output |
| mst_eps15_seed1 | mst | 15 | 1 | ok | 3520 x 249 | 10436.1s | 0 ✅ | 1638 duplicate rows within output |
| mst_eps15_seed2 | mst | 15 | 2 | ok | 3520 x 249 | 10390.2s | 0 ✅ | 1663 duplicate rows within output |
| tvae_qt_seed0 | tvae_qt | - | 0 | ok | 3520 x 249 | 456.8s | 0 ✅ |  |
| tvae_qt_seed1 | tvae_qt | - | 1 | ok | 3520 x 249 | 445.2s | 0 ✅ |  |
| tvae_qt_seed2 | tvae_qt | - | 2 | ok | 3520 x 249 | 450.1s | 0 ✅ |  |
| tvae_cap256_seed0 | tvae_cap256 | - | 0 | ok | 3520 x 249 | 447.0s | 0 ✅ |  |
| tvae_ep1000_seed0 | tvae_ep1000 | - | 0 | ok | 3520 x 249 | 808.2s | 0 ✅ |  |
| tvae_ind_seed0 | tvae_ind | - | 0 | ok | 3520 x 249 | 522.1s | 0 ✅ |  |
| ctgan_qt_seed0 | ctgan_qt | - | 0 | ok | 3520 x 249 | 2303.6s | 0 ✅ |  |
| aim40_eps1_seed0 | aim40 | 1 | 0 | ok | 3520 x 78 | 1124.9s | 0 ✅ | width-limited (40 cols) |
| aim40_eps5_seed0 | aim40 | 5 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim40_eps5_seed0' fit exceeded the 7200s time limit; width-limited (40 cols) |
| aim40_eps8_seed0 | aim40 | 8 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim40_eps8_seed0' fit exceeded the 7200s time limit; width-limited (40 cols) |
| aim40_eps10_seed0 | aim40 | 10 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim40_eps10_seed0' fit exceeded the 7200s time limit; width-limited (40 cols) |
| aim40_eps15_seed0 | aim40 | 15 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim40_eps15_seed0' fit exceeded the 7200s time limit; width-limited (40 cols) |
| aim40_eps20_seed0 | aim40 | 20 | 0 | failed | - | 7200.0s | - | TimeoutError: 'aim40_eps20_seed0' fit exceeded the 7200s time limit; width-limited (40 cols) |
| ddpm_seed0 | ddpm | - | 0 | ok | 3520 x 249 | 37.4s | 0 ✅ |  |
| ddpm_seed1 | ddpm | - | 1 | ok | 3520 x 249 | 37.3s | 0 ✅ |  |
| ddpm_seed2 | ddpm | - | 2 | ok | 3520 x 249 | 37.9s | 0 ✅ |  |
| ddpm_g_seed0 | ddpm_g | - | 0 | ok | 3520 x 249 | 109.9s | 0 ✅ |  |
| patectgan_eps1_seed0 | patectgan | 1 | 0 | ok | 3520 x 249 | 92.9s | 0 ✅ |  |
| patectgan_eps5_seed0 | patectgan | 5 | 0 | ok | 3520 x 249 | 1731.3s | 0 ✅ |  |
| patectgan_eps15_seed0 | patectgan | 15 | 0 | ok | 3520 x 249 | 11988.5s | 0 ✅ |  |
| mst_eps0p5_seed0 | mst | 0.5 | 0 | ok | 3520 x 249 | 10901.3s | 0 ✅ | 892 duplicate rows within output |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
