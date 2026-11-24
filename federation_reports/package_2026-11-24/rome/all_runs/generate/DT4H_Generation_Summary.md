# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `e07e8a127d1e1b3ab2bd2ce1eb94d146d050472c` (branch `main`, **uncommitted changes present**)
- Training data: `/home/rmucsc.rm.unicatt.it/sb005127/SyntheticDataGenerationTabularDT4H/output/preprocess/UC1_Train.parquet`
- Training data SHA-256: `fac7fd592ffb0771e0ea5a4023e3d03d10723d1660bfb05117c8965b89a1588f`
- Python 3.10.21 on Linux-6.1.0-41-amd64-x86_64-with-glibc2.36
- GPU: NVIDIA L40S-16C (CUDA 12.1)

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
- Training split: 3773 rows x 255 columns (the holdout split is never shown to any generator)
- Trained on: 151 columns (65 continuous, 86 categorical)
- Constant columns held out and re-attached verbatim: 104
- Synthetic rows generated per run: 3773
- Width-limited (AIM) runs train on 50 outcome-relevant columns (selection: `DT4H_AIM_Column_Selection.json`)

## Runs

| run | model | ε | δ | seed | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|---|---|---|
| gaussian_copula_seed0 | gaussian_copula | - | - | 0 | ok | 3773 x 255 | 22.0s | 0 ✅ |  |
| gaussian_copula_seed1 | gaussian_copula | - | - | 1 | ok | 3773 x 255 | 21.0s | 0 ✅ |  |
| gaussian_copula_seed2 | gaussian_copula | - | - | 2 | ok | 3773 x 255 | 21.0s | 0 ✅ |  |
| tvae_seed0 | tvae | - | - | 0 | ok | 3773 x 255 | 259.7s | 0 ✅ |  |
| tvae_seed1 | tvae | - | - | 1 | ok | 3773 x 255 | 246.2s | 0 ✅ |  |
| tvae_seed2 | tvae | - | - | 2 | ok | 3773 x 255 | 246.7s | 0 ✅ |  |
| ctgan_seed0 | ctgan | - | - | 0 | ok | 3773 x 255 | 428.7s | 0 ✅ |  |
| ctgan_seed1 | ctgan | - | - | 1 | ok | 3773 x 255 | 430.4s | 0 ✅ |  |
| ctgan_seed2 | ctgan | - | - | 2 | ok | 3773 x 255 | 430.3s | 0 ✅ |  |
| dpctgan_eps1_seed0 | dpctgan | 1 | 4.31e-06 | 0 | ok | 3773 x 255 | 221.3s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps5_seed0 | dpctgan | 5 | 4.31e-06 | 0 | ok | 3773 x 255 | 1212.5s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps8_seed0 | dpctgan | 8 | 4.31e-06 | 0 | ok | 3773 x 255 | 1215.8s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps10_seed0 | dpctgan | 10 | 4.31e-06 | 0 | ok | 3773 x 255 | 1196.0s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps15_seed0 | dpctgan | 15 | 4.31e-06 | 0 | ok | 3773 x 255 | 1163.4s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps20_seed0 | dpctgan | 20 | 4.31e-06 | 0 | ok | 3773 x 255 | 1192.8s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps15_seed1 | dpctgan | 15 | 4.31e-06 | 1 | ok | 3773 x 255 | 1193.2s | 0 ✅ | bounds `7b486344dcd7` |
| dpctgan_eps15_seed2 | dpctgan | 15 | 4.31e-06 | 2 | ok | 3773 x 255 | 1163.6s | 0 ✅ | bounds `7b486344dcd7` |
| aim50_eps1_seed0 | aim | 1 | 4.31e-06 | 0 | ok | 3773 x 154 | 5455.8s | 0 ✅ | 13 duplicate rows within output; width-limited (50 cols); bounds `7b486344dcd7` |
| aim50_eps5_seed0 | aim | 5 | - | 0 | failed | - | 21600.1s | - | TimeoutError: 'aim50_eps5_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| aim50_eps8_seed0 | aim | 8 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim50_eps8_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| aim50_eps10_seed0 | aim | 10 | - | 0 | failed | - | 19283.3s | - | XlaRuntimeError: RESOURCE_EXHAUSTED: Shared memory size limit exceeded: requested 131072, availab; width-limited (50 cols) |
| aim50_eps15_seed0 | aim | 15 | - | 0 | failed | - | 21600.1s | - | TimeoutError: 'aim50_eps15_seed0' fit exceeded the 21600s time limit; width-limited (50 cols) |
| aim50_eps20_seed0 | aim | 20 | - | 0 | failed | - | 14664.2s | - | XlaRuntimeError: RESOURCE_EXHAUSTED: Shared memory size limit exceeded: requested 139264, availab; width-limited (50 cols) |
| mst_eps1_seed0 | mst | 1 | 4.31e-06 | 0 | ok | 3773 x 255 | 4956.8s | 0 ✅ | 1743 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps5_seed0 | mst | 5 | 4.31e-06 | 0 | ok | 3773 x 255 | 4885.9s | 0 ✅ | 2433 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps8_seed0 | mst | 8 | 4.31e-06 | 0 | ok | 3773 x 255 | 4781.3s | 0 ✅ | 2483 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps10_seed0 | mst | 10 | 4.31e-06 | 0 | ok | 3773 x 255 | 4840.0s | 0 ✅ | 2389 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps15_seed0 | mst | 15 | 4.31e-06 | 0 | ok | 3773 x 255 | 4874.5s | 0 ✅ | 2364 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps20_seed0 | mst | 20 | 4.31e-06 | 0 | ok | 3773 x 255 | 4743.1s | 0 ✅ | 2432 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps15_seed1 | mst | 15 | 4.31e-06 | 1 | ok | 3773 x 255 | 4733.5s | 0 ✅ | 2342 duplicate rows within output; bounds `7b486344dcd7` |
| mst_eps15_seed2 | mst | 15 | 4.31e-06 | 2 | ok | 3773 x 255 | 4753.3s | 0 ✅ | 2458 duplicate rows within output; bounds `7b486344dcd7` |
| tvae_qt_seed0 | tvae_qt | - | - | 0 | ok | 3773 x 255 | 282.2s | 0 ✅ |  |
| tvae_qt_seed1 | tvae_qt | - | - | 1 | ok | 3773 x 255 | 275.3s | 0 ✅ |  |
| tvae_qt_seed2 | tvae_qt | - | - | 2 | ok | 3773 x 255 | 266.0s | 0 ✅ |  |
| tvae_cap256_seed0 | tvae_cap256 | - | - | 0 | ok | 3773 x 255 | 256.2s | 0 ✅ |  |
| tvae_ep1000_seed0 | tvae_ep1000 | - | - | 0 | ok | 3773 x 255 | 476.2s | 0 ✅ |  |
| tvae_ind_seed0 | tvae_ind | - | - | 0 | ok | 3773 x 255 | 299.7s | 0 ✅ |  |
| ctgan_qt_seed0 | ctgan_qt | - | - | 0 | ok | 3773 x 255 | 442.3s | 0 ✅ |  |
| aim40_eps1_seed0 | aim40 | 1 | 4.31e-06 | 0 | ok | 3773 x 144 | 2555.4s | 0 ✅ | 73 duplicate rows within output; width-limited (40 cols); bounds `7b486344dcd7` |
| aim40_eps5_seed0 | aim40 | 5 | 4.31e-06 | 0 | ok | 3773 x 144 | 8044.7s | 0 ✅ | 596 duplicate rows within output; width-limited (40 cols); bounds `7b486344dcd7` |
| aim40_eps8_seed0 | aim40 | 8 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim40_eps8_seed0' fit exceeded the 21600s time limit; width-limited (40 cols) |
| aim40_eps10_seed0 | aim40 | 10 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim40_eps10_seed0' fit exceeded the 21600s time limit; width-limited (40 cols) |
| aim40_eps15_seed0 | aim40 | 15 | - | 0 | failed | - | 21600.0s | - | TimeoutError: 'aim40_eps15_seed0' fit exceeded the 21600s time limit; width-limited (40 cols) |
| aim40_eps20_seed0 | aim40 | 20 | - | 0 | failed | - | 13409.4s | - | XlaRuntimeError: RESOURCE_EXHAUSTED: Shared memory size limit exceeded: requested 131072, availab; width-limited (40 cols) |
| ddpm_seed0 | ddpm | - | - | 0 | ok | 3773 x 255 | 50.2s | 0 ✅ |  |
| ddpm_seed1 | ddpm | - | - | 1 | ok | 3773 x 255 | 52.0s | 0 ✅ |  |
| ddpm_seed2 | ddpm | - | - | 2 | ok | 3773 x 255 | 49.9s | 0 ✅ |  |
| ddpm_g_seed0 | ddpm_g | - | - | 0 | ok | 3773 x 255 | 66.8s | 0 ✅ |  |
| patectgan_eps1_seed0 | patectgan | 1 | 4.31e-06 | 0 | ok | 3773 x 255 | 42.2s | 0 ✅ | bounds `7b486344dcd7` |
| patectgan_eps5_seed0 | patectgan | 5 | 4.31e-06 | 0 | ok | 3773 x 255 | 793.3s | 0 ✅ | bounds `7b486344dcd7` |
| patectgan_eps15_seed0 | patectgan | 15 | 4.31e-06 | 0 | ok | 3773 x 255 | 5454.0s | 0 ✅ | bounds `7b486344dcd7` |
| mst_eps0p5_seed0 | mst | 0.5 | 4.31e-06 | 0 | ok | 3773 x 255 | 4934.6s | 0 ✅ | 816 duplicate rows within output; bounds `7b486344dcd7` |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- DP runs are bounded by the reviewed a-priori public domains in `public_domains.json` (sha-256 recorded per run above), so no epsilon is spent on -- and no bound is derived from -- the training data. Two residual, disclosed leaks remain: snsynth learns categorical vocabularies from the training data, and AIM's column selection is computed on the train split without noise.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
