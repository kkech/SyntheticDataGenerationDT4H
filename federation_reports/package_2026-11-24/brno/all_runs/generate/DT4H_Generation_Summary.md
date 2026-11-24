# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `c349cdf576dec77b926a946db0ffa846a5f3ca9b` (branch `feature/synthesizer-filter`, **uncommitted changes present**)
- Training data: `/home/translated/kon/output/preprocess/UC1_Train.parquet`
- Training data SHA-256: `d6c0f327f2e2b79363385d92556eeb8cac1da5eee9a09b2ebfbc68f0aec2690d`
- Python 3.10.12 on Linux-5.19.0-1010-nvidia-lowlatency-x86_64-with-glibc2.35
- GPU: NVIDIA A30 (CUDA 12.1)

| package | version |
|---|---|
| sdv | 1.38.0 |
| ctgan | 0.12.1 |
| smartnoise-synth | 1.0.8 |
| opendp | 0.14.2 |
| torch | 2.5.1+cu121 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| polars | 1.44.1 |
| scikit-learn | 1.7.2 |

## Data
- Training split: 1549 rows x 256 columns (the holdout split is never shown to any generator)
- Trained on: 234 columns (66 continuous, 168 categorical)
- Constant columns held out and re-attached verbatim: 22
- Synthetic rows generated per run: 1549
- Width-limited (AIM) runs train on 50 outcome-relevant columns (selection: `DT4H_AIM_Column_Selection.json`)

## Runs

| run | model | ε | δ | seed | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|---|---|---|
| gaussian_copula_seed1 | gaussian_copula | - | - | 1 | ok | 1549 x 256 | 19.9s | 0 ✅ |  |
| gaussian_copula_seed2 | gaussian_copula | - | - | 2 | ok | 1549 x 256 | 20.7s | 0 ✅ |  |
| gaussian_copula_seed0 | gaussian_copula | - | - | 0 | ok | 1549 x 256 | 22.1s | 0 ✅ |  |
| tvae_seed2 | tvae | - | - | 2 | ok | 1549 x 256 | 299.1s | 0 ✅ |  |
| tvae_seed1 | tvae | - | - | 1 | ok | 1549 x 256 | 301.7s | 0 ✅ |  |
| tvae_seed0 | tvae | - | - | 0 | ok | 1549 x 256 | 302.9s | 0 ✅ |  |
| ctgan_seed0 | ctgan | - | - | 0 | ok | 1549 x 256 | 702.6s | 0 ✅ |  |
| ctgan_seed1 | ctgan | - | - | 1 | ok | 1549 x 256 | 707.2s | 0 ✅ |  |
| ctgan_seed2 | ctgan | - | - | 2 | ok | 1549 x 256 | 706.9s | 0 ✅ |  |
| dpctgan_eps1_seed0 | dpctgan | 1 | 1.64e-05 | 0 | ok | 1549 x 256 | 139.3s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps5_seed0 | dpctgan | 5 | 1.64e-05 | 0 | ok | 1549 x 256 | 1539.3s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps8_seed0 | dpctgan | 8 | 1.64e-05 | 0 | ok | 1549 x 256 | 1581.3s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps10_seed0 | dpctgan | 10 | 1.64e-05 | 0 | ok | 1549 x 256 | 1609.6s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps15_seed0 | dpctgan | 15 | 1.64e-05 | 0 | ok | 1549 x 256 | 1543.9s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps20_seed0 | dpctgan | 20 | 1.64e-05 | 0 | ok | 1549 x 256 | 1540.9s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps15_seed1 | dpctgan | 15 | 1.64e-05 | 1 | ok | 1549 x 256 | 1602.3s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps15_seed2 | dpctgan | 15 | 1.64e-05 | 2 | ok | 1549 x 256 | 1142.5s | 0 ✅ | bounds `7b486344dcd7` |
| tvae_cap256_seed0 | tvae_cap256 | - | - | 0 | ok | 1549 x 256 | 296.6s | 0 ✅ |  |
| tvae_ind_seed0 | tvae_ind | - | - | 0 | ok | 1549 x 256 | 279.6s | 0 ✅ |  |
| tvae_ep1000_seed0 | tvae_ep1000 | - | - | 0 | ok | 1549 x 256 | 538.9s | 0 ✅ |  |
| aim50_eps1_seed0 | aim | 1 | 1.64e-05 | 0 | ok | 1549 x 72 | 4592.5s | 0 ✅ | width-limited (50 cols); bounds `7b486344dcd7` |
| aim50_eps5_seed0 | aim | 5 | 1.64e-05 | 0 | ok | 1549 x 72 | 13514.3s | 0 ✅ | 28 duplicate rows within output; width-limited (50 cols); bounds `7b486344dcd7` |
| aim50_eps8_seed0 | aim | 8 | 1.64e-05 | 0 | ok | 1549 x 72 | 19086.5s | 0 ✅ | 136 duplicate rows within output; width-limited (50 cols); bounds `7b486344dcd7` |
| aim50_eps10_seed0 | aim | 10 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim50_eps10_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| aim50_eps15_seed0 | aim | 15 | - | 0 | failed | - | 21600.1s | - | TimeoutError: 'aim50_eps15_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| aim50_eps20_seed0 | aim | 20 | - | 0 | failed | - | 21600.1s | - | TimeoutError: 'aim50_eps20_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| mst_eps1_seed0 | mst | 1 | 1.64e-05 | 0 | ok | 1549 x 256 | 19628.4s | 0 ✅ | 40 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps5_seed0 | mst | 5 | 1.64e-05 | 0 | ok | 1549 x 256 | 19148.8s | 0 ✅ | 404 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps8_seed0 | mst | 8 | 1.64e-05 | 0 | ok | 1549 x 256 | 19066.1s | 0 ✅ | 498 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps10_seed0 | mst | 10 | 1.64e-05 | 0 | ok | 1549 x 256 | 18865.5s | 0 ✅ | 526 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps15_seed0 | mst | 15 | 1.64e-05 | 0 | ok | 1549 x 256 | 19008.7s | 0 ✅ | 579 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps20_seed0 | mst | 20 | 1.64e-05 | 0 | ok | 1549 x 256 | 18866.2s | 0 ✅ | 619 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps15_seed1 | mst | 15 | 1.64e-05 | 1 | ok | 1549 x 256 | 18831.1s | 0 ✅ | 595 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps15_seed2 | mst | 15 | 1.64e-05 | 2 | ok | 1549 x 256 | 18844.6s | 0 ✅ | 615 duplicate rows within output; bounds `7b486344dcd7` |
| tvae_qt_seed0 | tvae_qt | - | - | 0 | ok | 1549 x 256 | 324.1s | 0 ✅ |  |
| tvae_qt_seed1 | tvae_qt | - | - | 1 | ok | 1549 x 256 | 337.8s | 0 ✅ |  |
| tvae_qt_seed2 | tvae_qt | - | - | 2 | ok | 1549 x 256 | 346.9s | 0 ✅ |  |
| ctgan_qt_seed0 | ctgan_qt | - | - | 0 | ok | 1549 x 256 | 530.2s | 0 ✅ |  |
| aim40_eps1_seed0 | aim40 | 1 | 1.64e-05 | 0 | ok | 1549 x 62 | 1824.7s | 0 ✅ | width-limited (40 cols); bounds `7b486344dcd7` |
| aim40_eps5_seed0 | aim40 | 5 | 1.64e-05 | 0 | ok | 1549 x 62 | 6315.6s | 0 ✅ | 208 duplicate rows within output; width-limited (40 cols); bounds `7b486344dcd7` |
| aim40_eps8_seed0 | aim40 | 8 | 1.64e-05 | 0 | ok | 1549 x 62 | 9066.2s | 0 ✅ | 232 duplicate rows within output; width-limited (40 cols); bounds `7b486344dcd7` |
| aim40_eps10_seed0 | aim40 | 10 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim40_eps10_seed0' fit exceeded the 21600s time limit; width-limited (40 cols) |
| aim40_eps15_seed0 | aim40 | 15 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim40_eps15_seed0' fit exceeded the 21600s time limit; width-limited (40 cols) |
| aim40_eps20_seed0 | aim40 | 20 | - | 0 | failed | - | 21600.1s | - | TimeoutError: 'aim40_eps20_seed0' fit exceeded the 21600s time limit; width-limited (40 cols) |
| ddpm_seed0 | ddpm | - | - | 0 | ok | 1549 x 256 | 66.0s | 0 ✅ |  |
| ddpm_seed1 | ddpm | - | - | 1 | ok | 1549 x 256 | 66.2s | 0 ✅ |  |
| ddpm_seed2 | ddpm | - | - | 2 | ok | 1549 x 256 | 66.0s | 0 ✅ |  |
| ddpm_g_seed0 | ddpm_g | - | - | 0 | ok | 1549 x 256 | 117.0s | 0 ✅ |  |
| patectgan_eps1_seed0 | patectgan | 1 | 1.64e-05 | 0 | ok | 1549 x 256 | 68.8s | 0 ✅ | bounds `7b486344dcd7` |
| patectgan_eps5_seed0 | patectgan | 5 | 1.64e-05 | 0 | ok | 1549 x 256 | 1354.4s | 0 ✅ | bounds `7b486344dcd7` |
| patectgan_eps15_seed0 | patectgan | 15 | 1.64e-05 | 0 | ok | 1549 x 256 | 9138.2s | 0 ✅ | bounds `7b486344dcd7` |
| mst_eps0p5_seed0 | mst | 0.5 | - | 0 | failed | - | 14400.0s | - | TimeoutError: 'mst_eps0p5_seed0' fit exceeded the 14400s time limit |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- DP runs are bounded by the reviewed a-priori public domains in `public_domains.json` (sha-256 recorded per run above), so no epsilon is spent on -- and no bound is derived from -- the training data. Two residual, disclosed leaks remain: snsynth learns categorical vocabularies from the training data, and AIM's column selection is computed on the train split without noise.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
