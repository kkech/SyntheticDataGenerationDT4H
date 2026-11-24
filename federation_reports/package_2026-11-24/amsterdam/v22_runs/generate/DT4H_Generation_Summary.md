# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `f77bc78e5bdc81fdeb21329ddb0c2e904be45e32` (branch `claude/bounded-improvements`, **uncommitted changes present**)
- Training data: `/home/konstantinos.kechagi@mydre.org/generationV2/dt4h-v22-probe/output/preprocess/UC1_Train.parquet`
- Training data SHA-256: `98e716177d5cd04cffb50c5f59f1dfad90ae0ec252f78bbc584eb6c5e1c0a26b`
- Python 3.10.12 on Linux-6.8.0-1065-azure-x86_64-with-glibc2.35
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
- Training split: 3334 rows x 254 columns (the holdout split is never shown to any generator)
- Trained on: 217 columns (64 continuous, 153 categorical)
- Constant columns held out and re-attached verbatim: 37
- Synthetic rows generated per run: 3334
- Width-limited (AIM) runs train on 50 outcome-relevant columns (selection: `DT4H_AIM_Column_Selection.json`)

## Runs

| run | model | ε | δ | seed | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|---|---|---|
| gaussian_copula_seed0 | gaussian_copula | - | - | 0 | ok | 3334 x 254 | 32.8s | 0 ✅ |  |
| gaussian_copula_seed1 | gaussian_copula | - | - | 1 | ok | 3334 x 254 | 32.1s | 0 ✅ |  |
| gaussian_copula_seed2 | gaussian_copula | - | - | 2 | ok | 3334 x 254 | 31.9s | 0 ✅ |  |
| tvae_seed0 | tvae | - | - | 0 | ok | 3334 x 254 | 348.4s | 0 ✅ |  |
| tvae_seed1 | tvae | - | - | 1 | ok | 3334 x 254 | 347.4s | 0 ✅ |  |
| tvae_seed2 | tvae | - | - | 2 | ok | 3334 x 254 | 347.8s | 0 ✅ |  |
| ctgan_seed0 | ctgan | - | - | 0 | ok | 3334 x 254 | 665.8s | 0 ✅ |  |
| ctgan_seed1 | ctgan | - | - | 1 | ok | 3334 x 254 | 666.5s | 0 ✅ |  |
| ctgan_seed2 | ctgan | - | - | 2 | ok | 3334 x 254 | 666.9s | 0 ✅ |  |
| dpctgan_eps1_seed0 | dpctgan | 1 | 5.19e-06 | 0 | ok | 3334 x 254 | 330.6s | 0 ✅ | bounds `756745e7591d` |
| dpctgan_eps5_seed0 | dpctgan | 5 | 5.19e-06 | 0 | ok | 3334 x 254 | 2064.3s | 0 ✅ | bounds `756745e7591d` |
| dpctgan_eps8_seed0 | dpctgan | 8 | 5.19e-06 | 0 | ok | 3334 x 254 | 2064.5s | 0 ✅ | bounds `756745e7591d` |
| dpctgan_eps10_seed0 | dpctgan | 10 | 5.19e-06 | 0 | ok | 3334 x 254 | 2064.9s | 0 ✅ | bounds `756745e7591d` |
| dpctgan_eps15_seed0 | dpctgan | 15 | 5.19e-06 | 0 | ok | 3334 x 254 | 2064.5s | 0 ✅ | bounds `756745e7591d` |
| dpctgan_eps20_seed0 | dpctgan | 20 | 5.19e-06 | 0 | ok | 3334 x 254 | 2065.7s | 0 ✅ | bounds `756745e7591d` |
| dpctgan_eps15_seed1 | dpctgan | 15 | 5.19e-06 | 1 | ok | 3334 x 254 | 2068.8s | 0 ✅ | bounds `756745e7591d` |
| dpctgan_eps15_seed2 | dpctgan | 15 | 5.19e-06 | 2 | ok | 3334 x 254 | 2070.5s | 0 ✅ | bounds `756745e7591d` |
| aim50_eps1_seed0 | aim | 1 | 5.19e-06 | 0 | ok | 3334 x 87 | 2818.1s | 0 ✅ | width-limited (50 cols); bounds `756745e7591d` |
| aim50_eps5_seed0 | aim | 5 | 5.19e-06 | 0 | ok | 3334 x 87 | 13526.6s | 0 ✅ | 584 duplicate rows within output; width-limited (50 cols); bounds `756745e7591d` |
| aim50_eps8_seed0 | aim | 8 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim50_eps8_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| aim50_eps10_seed0 | aim | 10 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim50_eps10_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| aim50_eps15_seed0 | aim | 15 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim50_eps15_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| aim50_eps20_seed0 | aim | 20 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim50_eps20_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| mst_eps1_seed0 | mst | 1 | 5.19e-06 | 0 | ok | 3334 x 254 | 12050.7s | 0 ✅ | 670 duplicate rows within output; bounds `756745e7591d` |
| mst_eps5_seed0 | mst | 5 | 5.19e-06 | 0 | ok | 3334 x 254 | 11586.1s | 0 ✅ | 1763 duplicate rows within output; bounds `756745e7591d` |
| mst_eps8_seed0 | mst | 8 | 5.19e-06 | 0 | ok | 3334 x 254 | 11489.1s | 0 ✅ | 1820 duplicate rows within output; bounds `756745e7591d` |
| mst_eps10_seed0 | mst | 10 | 5.19e-06 | 0 | ok | 3334 x 254 | 11635.2s | 0 ✅ | 1805 duplicate rows within output; bounds `756745e7591d` |
| mst_eps15_seed0 | mst | 15 | 5.19e-06 | 0 | ok | 3334 x 254 | 11482.2s | 0 ✅ | 1842 duplicate rows within output; bounds `756745e7591d` |
| mst_eps20_seed0 | mst | 20 | 5.19e-06 | 0 | ok | 3334 x 254 | 11492.9s | 0 ✅ | 1851 duplicate rows within output; bounds `756745e7591d` |
| mst_eps15_seed1 | mst | 15 | 5.19e-06 | 1 | ok | 3334 x 254 | 11471.5s | 0 ✅ | 1803 duplicate rows within output; bounds `756745e7591d` |
| mst_eps15_seed2 | mst | 15 | 5.19e-06 | 2 | ok | 3334 x 254 | 11505.9s | 0 ✅ | 1834 duplicate rows within output; bounds `756745e7591d` |
| tvae_qt_seed0 | tvae_qt | - | - | 0 | ok | 3334 x 254 | 424.3s | 0 ✅ |  |
| tvae_qt_seed1 | tvae_qt | - | - | 1 | ok | 3334 x 254 | 424.7s | 0 ✅ |  |
| tvae_qt_seed2 | tvae_qt | - | - | 2 | ok | 3334 x 254 | 437.7s | 0 ✅ |  |
| tvae_cap256_seed0 | tvae_cap256 | - | - | 0 | ok | 3334 x 254 | 428.1s | 0 ✅ |  |
| tvae_ep1000_seed0 | tvae_ep1000 | - | - | 0 | ok | 3334 x 254 | 770.2s | 0 ✅ |  |
| tvae_ind_seed0 | tvae_ind | - | - | 0 | ok | 3334 x 254 | 492.6s | 0 ✅ |  |
| ctgan_qt_seed0 | ctgan_qt | - | - | 0 | ok | 3334 x 254 | 776.7s | 0 ✅ |  |
| aim40_eps1_seed0 | aim40 | 1 | 5.19e-06 | 0 | ok | 3334 x 77 | 1315.7s | 0 ✅ | 3 duplicate rows within output; width-limited (40 cols); bounds `756745e7591d` |
| aim40_eps5_seed0 | aim40 | 5 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim40_eps5_seed0' fit exceeded the 21600s time limit; width-limited (40 cols) |
| aim40_eps8_seed0 | aim40 | 8 | 5.19e-06 | 0 | ok | 3334 x 77 | 10876.4s | 0 ✅ | 487 duplicate rows within output; width-limited (40 cols); bounds `756745e7591d` |
| aim40_eps10_seed0 | aim40 | 10 | 5.19e-06 | 0 | ok | 3334 x 77 | 16550.7s | 0 ✅ | 279 duplicate rows within output; width-limited (40 cols); bounds `756745e7591d` |
| aim40_eps15_seed0 | aim40 | 15 | 5.19e-06 | 0 | ok | 3334 x 77 | 20649.0s | 0 ✅ | 229 duplicate rows within output; width-limited (40 cols); bounds `756745e7591d` |
| aim40_eps20_seed0 | aim40 | 20 | 5.19e-06 | 0 | ok | 3334 x 77 | 19832.7s | 0 ✅ | 315 duplicate rows within output; width-limited (40 cols); bounds `756745e7591d` |
| ddpm_seed0 | ddpm | - | - | 0 | ok | 3334 x 254 | 52.2s | 0 ✅ |  |
| ddpm_seed1 | ddpm | - | - | 1 | ok | 3334 x 254 | 51.8s | 0 ✅ |  |
| ddpm_seed2 | ddpm | - | - | 2 | ok | 3334 x 254 | 51.9s | 0 ✅ |  |
| ddpm_g_seed0 | ddpm_g | - | - | 0 | ok | 3334 x 254 | 114.4s | 0 ✅ |  |
| patectgan_eps1_seed0 | patectgan | 1 | 5.19e-06 | 0 | ok | 3334 x 254 | 79.8s | 0 ✅ | bounds `756745e7591d` |
| patectgan_eps5_seed0 | patectgan | 5 | 5.19e-06 | 0 | ok | 3334 x 254 | 1521.2s | 0 ✅ | bounds `756745e7591d` |
| patectgan_eps15_seed0 | patectgan | 15 | 5.19e-06 | 0 | ok | 3334 x 254 | 10505.0s | 0 ✅ | bounds `756745e7591d` |
| mst_eps0p5_seed0 | mst | 0.5 | 5.19e-06 | 0 | ok | 3334 x 254 | 12000.5s | 0 ✅ | 265 duplicate rows within output; bounds `756745e7591d` |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- DP runs are bounded by the reviewed a-priori public domains in `public_domains.json` (sha-256 recorded per run above), so no epsilon is spent on -- and no bound is derived from -- the training data. Two residual, disclosed leaks remain: snsynth learns categorical vocabularies from the training data, and AIM's column selection is computed on the train split without noise.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
