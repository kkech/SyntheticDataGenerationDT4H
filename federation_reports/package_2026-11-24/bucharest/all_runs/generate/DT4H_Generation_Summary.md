# Generation Summary

## Reproducibility
- Seed: `0`
- Git commit: `0874c4da71f613db8320b27c0f27f6b2da26fe31` (branch `logs/buch`, **uncommitted changes present**)
- Training data: `/home/translated/cardioData/SyntheticDataGenerationTabularDT4H/output/preprocess/UC1_Train.parquet`
- Training data SHA-256: `1e97972e78059fed5f623198c645bf77f7e7e68d455cddc564ce5264ca911509`
- Python 3.10.12 on Linux-6.8.0-138-generic-x86_64-with-glibc2.35
- GPU: NVIDIA L4 (CUDA 12.4)

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
- Training split: 4706 rows x 229 columns (the holdout split is never shown to any generator)
- Trained on: 101 columns (38 continuous, 63 categorical)
- Constant columns held out and re-attached verbatim: 128
- Synthetic rows generated per run: 4706

## Runs

| run | model | ε | δ | seed | status | rows x cols | duration | verbatim training rows | notes |
|---|---|---|---|---|---|---|---|---|---|
| gaussian_copula_seed0 | gaussian_copula | - | - | 0 | ok | 4706 x 229 | 17.2s | 0 ✅ |  |
| gaussian_copula_seed1 | gaussian_copula | - | - | 1 | ok | 4706 x 229 | 16.1s | 0 ✅ |  |
| gaussian_copula_seed2 | gaussian_copula | - | - | 2 | ok | 4706 x 229 | 16.1s | 0 ✅ |  |
| tvae_seed0 | tvae | - | - | 0 | ok | 4706 x 229 | 221.4s | 0 ✅ |  |
| tvae_seed1 | tvae | - | - | 1 | ok | 4706 x 229 | 210.8s | 0 ✅ |  |
| tvae_seed2 | tvae | - | - | 2 | ok | 4706 x 229 | 208.0s | 0 ✅ |  |
| ctgan_seed0 | ctgan | - | - | 0 | ok | 4706 x 229 | 413.4s | 0 ✅ |  |
| ctgan_seed1 | ctgan | - | - | 1 | ok | 4706 x 229 | 414.2s | 0 ✅ |  |
| ctgan_seed2 | ctgan | - | - | 2 | ok | 4706 x 229 | 413.1s | 0 ✅ |  |
| dpctgan_eps1_seed0 | dpctgan | 1 | 3.1e-06 | 0 | ok | 4706 x 229 | 299.0s | 0 ✅ | bounds `283379d243f3` |
| dpctgan_eps5_seed0 | dpctgan | 5 | 3.1e-06 | 0 | ok | 4706 x 229 | 1433.6s | 0 ✅ | bounds `283379d243f3` |
| dpctgan_eps8_seed0 | dpctgan | 8 | 3.1e-06 | 0 | ok | 4706 x 229 | 1426.8s | 0 ✅ | bounds `283379d243f3` |
| dpctgan_eps10_seed0 | dpctgan | 10 | 3.1e-06 | 0 | ok | 4706 x 229 | 1445.7s | 0 ✅ | bounds `283379d243f3` |
| dpctgan_eps15_seed0 | dpctgan | 15 | 3.1e-06 | 0 | ok | 4706 x 229 | 1411.7s | 0 ✅ | bounds `283379d243f3` |
| dpctgan_eps20_seed0 | dpctgan | 20 | 3.1e-06 | 0 | ok | 4706 x 229 | 1408.7s | 0 ✅ | bounds `283379d243f3` |
| dpctgan_eps15_seed1 | dpctgan | 15 | 3.1e-06 | 1 | ok | 4706 x 229 | 1423.6s | 0 ✅ | bounds `283379d243f3` |
| dpctgan_eps15_seed2 | dpctgan | 15 | 3.1e-06 | 2 | ok | 4706 x 229 | 1418.0s | 0 ✅ | bounds `283379d243f3` |
| tvae_qt_seed0 | tvae_qt | - | - | 0 | ok | 4706 x 229 | 240.9s | 0 ✅ |  |
| tvae_qt_seed1 | tvae_qt | - | - | 1 | ok | 4706 x 229 | 237.5s | 0 ✅ |  |
| tvae_qt_seed2 | tvae_qt | - | - | 2 | ok | 4706 x 229 | 241.0s | 0 ✅ |  |
| tvae_cap256_seed0 | tvae_cap256 | - | - | 0 | ok | 4706 x 229 | 231.5s | 0 ✅ |  |
| tvae_ep1000_seed0 | tvae_ep1000 | - | - | 0 | ok | 4706 x 229 | 455.8s | 0 ✅ |  |
| tvae_ind_seed0 | tvae_ind | - | - | 0 | ok | 4706 x 229 | 250.5s | 0 ✅ |  |
| ctgan_qt_seed0 | ctgan_qt | - | - | 0 | ok | 4706 x 229 | 465.7s | 0 ✅ |  |
| ddpm_seed0 | ddpm | - | - | 0 | ok | 4706 x 229 | 67.2s | 0 ✅ |  |
| ddpm_seed1 | ddpm | - | - | 1 | ok | 4706 x 229 | 66.5s | 0 ✅ |  |
| ddpm_seed2 | ddpm | - | - | 2 | ok | 4706 x 229 | 67.1s | 0 ✅ |  |
| ddpm_g_seed0 | ddpm_g | - | - | 0 | ok | 4706 x 229 | 68.2s | 0 ✅ |  |
| patectgan_eps1_seed0 | patectgan | 1 | 3.1e-06 | 0 | ok | 4706 x 229 | 39.7s | 0 ✅ | bounds `283379d243f3` |
| patectgan_eps5_seed0 | patectgan | 5 | 3.1e-06 | 0 | ok | 4706 x 229 | 742.7s | 0 ✅ | bounds `283379d243f3` |
| patectgan_eps15_seed0 | patectgan | 15 | 3.1e-06 | 0 | ok | 4706 x 229 | 5183.7s | 0 ✅ | bounds `283379d243f3` |
| mst_eps15_seed0 | mst | 15 | 3.1e-06 | 0 | ok | 4706 x 229 | 664.6s | 0 ✅ | 3921 duplicate rows within output; bounds `283379d243f3` |

## Caveats
- The leakage column counts EXACT reproductions of training rows only. It does not detect near-duplicates; the privacy step's distance-to-closest-record analysis against the holdout baseline covers the rest.
- Non-DP models carry no formal privacy guarantee regardless of this check.
- DP runs are bounded by the reviewed a-priori public domains in `public_domains.json` (sha-256 recorded per run above), so no epsilon is spent on -- and no bound is derived from -- the training data. Two residual, disclosed leaks remain: snsynth learns categorical vocabularies from the training data, and AIM's column selection is computed on the train split without noise.
- Width-limited runs synthesize a column subset by design; their files have fewer columns and are evaluated over those columns only.
