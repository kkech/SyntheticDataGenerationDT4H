# Survival Fidelity

Endpoints use the five-year follow-up columns: a recorded days-to-event is an event, a null is administrative censoring at 1825 days -- the same rule for real and synthetic data. A recorded time beyond 1825 days is treated as censoring at 1825 (out of horizon), counted per frame as `times_beyond_horizon`. The train-vs-holdout log-rank p-value calibrates what pure sampling noise looks like.

Disclosure: synthetic days-to-event values below the real observed minimum were nulled by the sentinel decode upstream and are read here as censoring; those erased early events cannot be recovered from the released CSVs (see `decode_note` in the JSON).

Effect-replication covariates are standardized in EVERY frame (real train, real holdout, synthetic) by the REAL TRAIN split's mean/SD, so scale infidelity in a synthetic frame shows up as a coefficient discrepancy instead of being re-normalized away.

## all_cause_death

train events 1069/3334 | holdout 388/1111 | log-rank train-vs-holdout p = 0.1201

| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |
|---|---|---|---|---|
| **train** | 0.78944 | 0.67996 | - | no |
| **holdout** | 0.78668 | 0.65077 |  |  |
| aim40_eps10_seed0 | - | - | None | - |
| aim40_eps15_seed0 | - | - | None | - |
| aim40_eps1_seed0 | - | - | None | - |
| aim40_eps20_seed0 | - | - | None | - |
| aim40_eps8_seed0 | - | - | None | - |
| aim50_eps1_seed0 | - | - | None | - |
| aim50_eps5_seed0 | - | - | None | - |
| ctgan_qt_seed0 | 0.92142 | 0.81554 | 0.0 | no |
| ctgan_seed0 | 0.86743 | 0.77564 | 0.0 | no |
| ctgan_seed1 | 0.89112 | 0.79784 | 0.0 | no |
| ctgan_seed2 | 0.75945 | 0.63077 | 0.0799 | no |
| ddpm_g_seed0 | 0.9997 | 0.9994 | 0.0 | no |
| ddpm_seed0 | 0.92771 | 0.79484 | 0.0 | no |
| ddpm_seed1 | 0.94031 | 0.83113 | 0.0 | no |
| ddpm_seed2 | 0.92681 | 0.76485 | 0.0 | no |
| dpctgan_eps10_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed1 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed2 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps1_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps20_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps5_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps8_seed0 | 1.0 | 1.0 | 0.0 | no |
| gaussian_copula_seed0 | 0.89982 | 0.78584 | 0.0 | no |
| gaussian_copula_seed1 | 0.88902 | 0.76695 | 0.0 | no |
| gaussian_copula_seed2 | 0.89142 | 0.77325 | 0.0 | no |
| mst_eps0p5_seed0 | 0.80444 | 0.67457 | 0.004 | no |
| mst_eps10_seed0 | 0.77235 | 0.68236 | 0.0023 | no |
| mst_eps15_seed0 | 0.76905 | 0.67786 | 0.0056 | no |
| mst_eps15_seed1 | 0.77115 | 0.68116 | 0.003 | no |
| mst_eps15_seed2 | 0.77145 | 0.68086 | 0.0031 | no |
| mst_eps1_seed0 | 0.80864 | 0.72136 | 0.0 | no |
| mst_eps20_seed0 | 0.77205 | 0.67876 | 0.0046 | no |
| mst_eps5_seed0 | 0.77085 | 0.68776 | 0.0008 | no |
| mst_eps8_seed0 | 0.77385 | 0.68566 | 0.0011 | no |
| patectgan_eps15_seed0 | 0.9901 | 0.96941 | 0.0 | no |
| patectgan_eps1_seed0 | 0.9796 | 0.979 | 0.0 | no |
| patectgan_eps5_seed0 | 1.0 | 1.0 | 0.0 | no |
| tvae_cap256_seed0 | 0.79994 | 0.67516 | 0.0934 | no |
| tvae_ep1000_seed0 | 0.81104 | 0.67996 | 0.0321 | no |
| tvae_ind_seed0 | 0.70936 | 0.65447 | 0.734 | no |
| tvae_qt_seed0 | 0.75105 | 0.66047 | 0.7643 | no |
| tvae_qt_seed1 | 0.76785 | 0.66257 | 0.4997 | yes ✅ |
| tvae_qt_seed2 | 0.78524 | 0.71056 | 0.0005 | no |
| tvae_seed0 | 0.78464 | 0.67397 | 0.1257 | no |
| tvae_seed1 | 0.81794 | 0.68986 | 0.0049 | no |
| tvae_seed2 | 0.80084 | 0.67007 | 0.1418 | no |

Equivalence is a POSITIVE claim (90% CI of the survival difference within ±5pp at every horizon) -- unlike a non-significant log-rank, which is only absence of evidence.

## hf_rehospitalization

train events 234/3334 | holdout 60/1111 | log-rank train-vs-holdout p = 0.0587

| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |
|---|---|---|---|---|
| **train** | 0.93551 | 0.93011 | - | yes ✅ |
| **holdout** | 0.94779 | 0.94599 |  |  |
| aim40_eps10_seed0 | - | - | None | - |
| aim40_eps15_seed0 | - | - | None | - |
| aim40_eps1_seed0 | - | - | None | - |
| aim40_eps20_seed0 | - | - | None | - |
| aim40_eps8_seed0 | - | - | None | - |
| aim50_eps1_seed0 | - | - | None | - |
| aim50_eps5_seed0 | - | - | None | - |
| ctgan_qt_seed0 | 0.9871 | 0.9841 | 0.0 | no |
| ctgan_seed0 | 0.95861 | 0.95501 | 0.2036 | yes ✅ |
| ctgan_seed1 | 0.93521 | 0.92951 | 0.0665 | yes ✅ |
| ctgan_seed2 | 0.95711 | 0.95081 | 0.474 | yes ✅ |
| ddpm_g_seed0 | 1.0 | 1.0 | 0.0 | no |
| ddpm_seed0 | 0.991 | 0.9787 | 0.0 | no |
| ddpm_seed1 | 0.9925 | 0.9763 | 0.0001 | no |
| ddpm_seed2 | 0.94751 | 0.89712 | 0.0 | no |
| dpctgan_eps10_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed1 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed2 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps1_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps20_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps5_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps8_seed0 | 1.0 | 1.0 | 0.0 | no |
| gaussian_copula_seed0 | 1.0 | 1.0 | 0.0 | no |
| gaussian_copula_seed1 | 1.0 | 1.0 | 0.0 | no |
| gaussian_copula_seed2 | 1.0 | 1.0 | 0.0 | no |
| mst_eps0p5_seed0 | 0.93761 | 0.86233 | 0.0 | no |
| mst_eps10_seed0 | 0.93401 | 0.93131 | 0.14 | yes ✅ |
| mst_eps15_seed0 | 0.93911 | 0.93371 | 0.2193 | yes ✅ |
| mst_eps15_seed1 | 0.93311 | 0.93011 | 0.1086 | yes ✅ |
| mst_eps15_seed2 | 0.93761 | 0.93281 | 0.1861 | yes ✅ |
| mst_eps1_seed0 | 0.94211 | 0.88512 | 0.0 | no |
| mst_eps20_seed0 | 0.93611 | 0.93161 | 0.1468 | yes ✅ |
| mst_eps5_seed0 | 0.93491 | 0.92951 | 0.0944 | yes ✅ |
| mst_eps8_seed0 | 0.93881 | 0.93551 | 0.3042 | yes ✅ |
| patectgan_eps15_seed0 | 0.9982 | 0.9982 | 0.0 | no |
| patectgan_eps1_seed0 | 0.9952 | 0.9952 | 0.0 | no |
| patectgan_eps5_seed0 | 1.0 | 1.0 | 0.0 | no |
| tvae_cap256_seed0 | 0.95201 | 0.94091 | 0.5555 | yes ✅ |
| tvae_ep1000_seed0 | 0.94571 | 0.93011 | 0.071 | yes ✅ |
| tvae_ind_seed0 | 0.95381 | 0.94811 | 0.7893 | yes ✅ |
| tvae_qt_seed0 | 0.96071 | 0.96071 | 0.0371 | yes ✅ |
| tvae_qt_seed1 | 0.96551 | 0.96491 | 0.005 | yes ✅ |
| tvae_qt_seed2 | 0.94511 | 0.94331 | 0.7607 | yes ✅ |
| tvae_seed0 | 0.95201 | 0.93911 | 0.4243 | yes ✅ |
| tvae_seed1 | 0.95231 | 0.93731 | 0.3153 | yes ✅ |
| tvae_seed2 | 0.96701 | 0.95861 | 0.0736 | yes ✅ |

Equivalence is a POSITIVE claim (90% CI of the survival difference within ±5pp at every horizon) -- unlike a non-significant log-rank, which is only absence of evidence.

## Effect-estimate replication

Model: cox_ph (lifelines), per-SD coefficients standardized by the real TRAIN split's mean/SD in every frame. The coefficient error is computed only over covariates present in BOTH the real and the synthetic fit (`matched`); runs sharing fewer than 4 covariates are not comparable.

| frame | n | events | sign agreement | coef matched | mean |coef error| |
|---|---|---|---|---|---|
| real train | 2666 | 884 | - | - | - |
| real holdout | 873 | 305 | - | - | - |
| aim40_eps10_seed0 | - | - | - | - | not estimable |
| aim40_eps15_seed0 | - | - | - | - | not estimable |
| aim40_eps1_seed0 | - | - | - | - | not estimable |
| aim40_eps20_seed0 | - | - | - | - | not estimable |
| aim40_eps8_seed0 | - | - | - | - | not estimable |
| aim50_eps1_seed0 | - | - | - | - | not estimable |
| aim50_eps5_seed0 | - | - | - | - | not estimable |
| ctgan_qt_seed0 | 1048 | 207 | 4/6 | 6/6 | 0.0983 |
| ctgan_seed0 | 2459 | 601 | 3/6 | 6/6 | 0.1299 |
| ctgan_seed1 | 2744 | 615 | 3/6 | 6/6 | 0.1215 |
| ctgan_seed2 | 2591 | 1008 | 3/6 | 6/6 | 0.155 |
| ddpm_g_seed0 | - | - | - | - | not estimable |
| ddpm_seed0 | 1008 | 263 | 4/6 | 6/6 | 0.0898 |
| ddpm_seed1 | 1139 | 248 | 4/6 | 6/6 | 0.0942 |
| ddpm_seed2 | 1439 | 373 | 4/6 | 6/6 | 0.0764 |
| dpctgan_eps10_seed0 | - | - | - | - | not estimable |
| dpctgan_eps15_seed0 | - | - | - | - | not estimable |
| dpctgan_eps15_seed1 | - | - | - | - | not estimable |
| dpctgan_eps15_seed2 | - | - | - | - | not estimable |
| dpctgan_eps1_seed0 | - | - | - | - | not estimable |
| dpctgan_eps20_seed0 | - | - | - | - | not estimable |
| dpctgan_eps5_seed0 | - | - | - | - | not estimable |
| dpctgan_eps8_seed0 | - | - | - | - | not estimable |
| gaussian_copula_seed0 | 2788 | 588 | 4/6 | 6/6 | 0.0777 |
| gaussian_copula_seed1 | 2776 | 627 | 4/6 | 6/6 | 0.0792 |
| gaussian_copula_seed2 | 2832 | 632 | 2/6 | 6/6 | 0.0992 |
| mst_eps0p5_seed0 | 2839 | 988 | 3/6 | 6/6 | 0.3385 |
| mst_eps10_seed0 | 2781 | 1030 | 2/6 | 6/6 | 0.2019 |
| mst_eps15_seed0 | 2822 | 1063 | 3/6 | 6/6 | 0.3135 |
| mst_eps15_seed1 | 2838 | 1052 | 3/6 | 6/6 | 0.283 |
| mst_eps15_seed2 | 2853 | 1049 | 3/6 | 6/6 | 0.4275 |
| mst_eps1_seed0 | 2535 | 896 | 4/6 | 6/6 | 0.242 |
| mst_eps20_seed0 | 2843 | 1069 | 4/6 | 6/6 | 0.3445 |
| mst_eps5_seed0 | 2756 | 1004 | 4/6 | 6/6 | 0.3604 |
| mst_eps8_seed0 | 2871 | 1012 | 5/6 | 6/6 | 0.2412 |
| patectgan_eps15_seed0 | 2613 | 74 | 4/6 | 6/6 | 0.1663 |
| patectgan_eps1_seed0 | 2285 | 57 | 3/6 | 6/6 | 0.1446 |
| patectgan_eps5_seed0 | - | - | - | - | not estimable |
| tvae_cap256_seed0 | 2666 | 908 | 6/6 | 6/6 | 0.1622 |
| tvae_ep1000_seed0 | 2645 | 885 | 5/6 | 6/6 | 0.12 |
| tvae_ind_seed0 | 3007 | 1079 | 5/6 | 6/6 | 0.2152 |
| tvae_qt_seed0 | 2399 | 801 | 6/6 | 6/6 | 0.0959 |
| tvae_qt_seed1 | 2400 | 829 | 6/6 | 6/6 | 0.201 |
| tvae_qt_seed2 | 2354 | 743 | 6/6 | 6/6 | 0.154 |
| tvae_seed0 | 2597 | 892 | 5/6 | 6/6 | 0.3 |
| tvae_seed1 | 2602 | 831 | 5/6 | 6/6 | 0.238 |
| tvae_seed2 | 2720 | 922 | 5/6 | 6/6 | 0.1699 |
