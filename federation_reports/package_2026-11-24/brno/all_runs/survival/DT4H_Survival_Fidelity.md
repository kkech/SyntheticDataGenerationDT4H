# Survival Fidelity

Endpoints use the five-year follow-up columns: a recorded days-to-event is an event, a null is administrative censoring at 1825 days -- the same rule for real and synthetic data. A recorded time beyond 1825 days is treated as censoring at 1825 (out of horizon), counted per frame as `times_beyond_horizon`. The train-vs-holdout log-rank p-value calibrates what pure sampling noise looks like.

Disclosure: synthetic days-to-event values below the real observed minimum were nulled by the sentinel decode upstream and are read here as censoring; those erased early events cannot be recovered from the released CSVs (see `decode_note` in the JSON).

Effect-replication covariates are standardized in EVERY frame (real train, real holdout, synthetic) by the REAL TRAIN split's mean/SD, so scale infidelity in a synthetic frame shows up as a coefficient discrepancy instead of being re-normalized away.

## all_cause_death

train events 746/1549 | holdout 231/516 | log-rank train-vs-holdout p = 0.129

| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |
|---|---|---|---|---|
| **train** | 0.71917 | 0.52485 | - | no |
| **holdout** | 0.75969 | 0.55233 |  |  |
| aim40_eps1_seed0 | - | - | None | - |
| aim40_eps5_seed0 | - | - | None | - |
| aim40_eps8_seed0 | - | - | None | - |
| aim50_eps1_seed0 | - | - | None | - |
| aim50_eps5_seed0 | - | - | None | - |
| aim50_eps8_seed0 | - | - | None | - |
| ctgan_qt_seed0 | 0.88057 | 0.65139 | 0.0 | no |
| ctgan_seed0 | 0.78502 | 0.62427 | 0.025 | no |
| ctgan_seed1 | 0.8909 | 0.80116 | 0.0 | no |
| ctgan_seed2 | 0.70884 | 0.48031 | 0.0002 | no |
| ddpm_g_seed0 | 0.92899 | 0.76759 | 0.081 | no |
| ddpm_seed0 | 0.94125 | 0.79664 | 0.0 | no |
| ddpm_seed1 | 0.92059 | 0.77082 | 0.0 | no |
| ddpm_seed2 | 0.93157 | 0.75662 | 0.0004 | no |
| dpctgan_eps10_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed0 | 1.0 | 0.0 | 0.0 | no |
| dpctgan_eps15_seed1 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed2 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps1_seed0 | 0.95675 | 0.95675 | 0.0 | no |
| dpctgan_eps20_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps5_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps8_seed0 | 0.8328 | 0.8328 | 0.0 | no |
| gaussian_copula_seed0 | 0.85926 | 0.63525 | 0.0001 | no |
| gaussian_copula_seed1 | 0.86766 | 0.64041 | 0.0 | no |
| gaussian_copula_seed2 | 0.85668 | 0.61911 | 0.0009 | no |
| mst_eps10_seed0 | 0.69981 | 0.54099 | 0.6087 | no |
| mst_eps15_seed0 | 0.69658 | 0.53066 | 0.9174 | no |
| mst_eps15_seed1 | 0.69077 | 0.52356 | 0.8889 | no |
| mst_eps15_seed2 | 0.70045 | 0.51582 | 0.7234 | no |
| mst_eps1_seed0 | 0.71724 | 0.53648 | 0.6537 | no |
| mst_eps20_seed0 | 0.7011 | 0.51646 | 0.7427 | no |
| mst_eps5_seed0 | 0.69722 | 0.5481 | 0.3951 | no |
| mst_eps8_seed0 | 0.69981 | 0.54551 | 0.4831 | no |
| patectgan_eps15_seed0 | 0.91672 | 0.76114 | 0.0 | no |
| patectgan_eps1_seed0 | 0.95868 | 0.95739 | 0.0 | no |
| patectgan_eps5_seed0 | 0.99484 | 0.98709 | 0.0 | no |
| tvae_cap256_seed0 | 0.72692 | 0.48741 | 0.0423 | no |
| tvae_ep1000_seed0 | 0.75339 | 0.48483 | 0.0415 | no |
| tvae_ind_seed0 | 0.67915 | 0.45901 | 0.0003 | no |
| tvae_qt_seed0 | 0.65332 | 0.44868 | 0.0 | no |
| tvae_qt_seed1 | 0.72369 | 0.48418 | 0.0082 | no |
| tvae_qt_seed2 | 0.70239 | 0.45255 | 0.0 | no |
| tvae_seed0 | 0.73402 | 0.47902 | 0.0225 | no |
| tvae_seed1 | 0.75016 | 0.47127 | 0.0148 | no |
| tvae_seed2 | 0.75403 | 0.47063 | 0.0165 | no |

Equivalence is a POSITIVE claim (90% CI of the survival difference within ±5pp at every horizon) -- unlike a non-significant log-rank, which is only absence of evidence.

## Effect-estimate replication

Model: logistic_1y_mortality (native fallback), per-SD coefficients standardized by the real TRAIN split's mean/SD in every frame. The coefficient error is computed only over covariates present in BOTH the real and the synthetic fit (`matched`); runs sharing fewer than 4 covariates are not comparable.

| frame | n | events | sign agreement | coef matched | mean |coef error| |
|---|---|---|---|---|---|
| real train | 712 | 200 | - | - | - |
| real holdout | 235 | 59 | - | - | - |
| aim40_eps1_seed0 | - | - | - | - | not estimable |
| aim40_eps5_seed0 | - | - | - | - | not estimable |
| aim40_eps8_seed0 | - | - | - | - | not estimable |
| aim50_eps1_seed0 | - | - | - | - | not estimable |
| aim50_eps5_seed0 | - | - | - | - | not estimable |
| aim50_eps8_seed0 | - | - | - | - | not estimable |
| ctgan_qt_seed0 | 658 | 76 | 3/5 | 6/6 | 0.3592 |
| ctgan_seed0 | 599 | 128 | 3/5 | 6/6 | 0.3022 |
| ctgan_seed1 | 579 | 64 | 0/5 | 6/6 | 0.4442 |
| ctgan_seed2 | 593 | 175 | 1/5 | 6/6 | 0.3419 |
| ddpm_g_seed0 | 512 | 40 | 2/5 | 6/6 | 0.4191 |
| ddpm_seed0 | 416 | 24 | 3/5 | 6/6 | 0.3774 |
| ddpm_seed1 | 396 | 30 | 2/5 | 6/6 | 0.3213 |
| ddpm_seed2 | 409 | 37 | 4/5 | 6/6 | 0.2497 |
| dpctgan_eps10_seed0 | - | - | - | - | not estimable |
| dpctgan_eps15_seed0 | - | - | - | - | not estimable |
| dpctgan_eps15_seed1 | - | - | - | - | not estimable |
| dpctgan_eps15_seed2 | - | - | - | - | not estimable |
| dpctgan_eps1_seed0 | - | - | - | - | not estimable |
| dpctgan_eps20_seed0 | - | - | - | - | not estimable |
| dpctgan_eps5_seed0 | - | - | - | - | not estimable |
| dpctgan_eps8_seed0 | - | - | - | - | not estimable |
| gaussian_copula_seed0 | 439 | 64 | 4/5 | 6/6 | 0.1748 |
| gaussian_copula_seed1 | 469 | 66 | 4/5 | 6/6 | 0.2161 |
| gaussian_copula_seed2 | 467 | 64 | 5/5 | 6/6 | 0.1853 |
| mst_eps10_seed0 | 742 | 312 | 2/5 | 6/6 | 0.445 |
| mst_eps15_seed0 | 712 | 315 | 4/5 | 6/6 | 0.3006 |
| mst_eps15_seed1 | 739 | 333 | 5/5 | 6/6 | 0.5101 |
| mst_eps15_seed2 | 736 | 298 | 3/5 | 6/6 | 0.3946 |
| mst_eps1_seed0 | 579 | 163 | 1/5 | 6/6 | 0.4283 |
| mst_eps20_seed0 | 718 | 314 | 3/5 | 6/6 | 0.2938 |
| mst_eps5_seed0 | 708 | 263 | 2/5 | 6/6 | 0.3804 |
| mst_eps8_seed0 | 752 | 365 | 4/5 | 6/6 | 0.3104 |
| patectgan_eps15_seed0 | 645 | 62 | 3/5 | 6/6 | 0.2365 |
| patectgan_eps1_seed0 | 989 | 34 | 3/5 | 6/6 | 0.2753 |
| patectgan_eps5_seed0 | - | - | - | - | not estimable |
| tvae_cap256_seed0 | 710 | 204 | 5/5 | 6/6 | 0.1677 |
| tvae_ep1000_seed0 | 747 | 184 | 4/5 | 6/6 | 0.1942 |
| tvae_ind_seed0 | 771 | 238 | 5/5 | 6/6 | 0.1818 |
| tvae_qt_seed0 | 688 | 237 | 5/5 | 6/6 | 0.1335 |
| tvae_qt_seed1 | 664 | 159 | 4/5 | 6/6 | 0.224 |
| tvae_qt_seed2 | 662 | 213 | 4/5 | 6/6 | 0.2877 |
| tvae_seed0 | 666 | 183 | 3/5 | 6/6 | 0.2931 |
| tvae_seed1 | 691 | 175 | 4/5 | 6/6 | 0.2653 |
| tvae_seed2 | 689 | 173 | 4/5 | 6/6 | 0.217 |
