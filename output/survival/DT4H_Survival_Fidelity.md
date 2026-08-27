# Survival Fidelity

Endpoints use the five-year follow-up columns: a recorded days-to-event is an event, a null is administrative censoring at 1825 days -- the same rule for real and synthetic data. A recorded time beyond 1825 days is treated as censoring at 1825 (out of horizon), counted per frame as `times_beyond_horizon`. The train-vs-holdout log-rank p-value calibrates what pure sampling noise looks like.

Disclosure: synthetic days-to-event values below the real observed minimum were nulled by the sentinel decode upstream and are read here as censoring; those erased early events cannot be recovered from the released CSVs (see `decode_note` in the JSON).

Effect-replication covariates are standardized in EVERY frame (real train, real holdout, synthetic) by the REAL TRAIN split's mean/SD, so scale infidelity in a synthetic frame shows up as a coefficient discrepancy instead of being re-normalized away.

## all_cause_death

train events 1421/3520 | holdout 483/1174 | log-rank train-vs-holdout p = 0.7078 | times beyond 1825d censored at horizon: {'holdout': 1}

| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |
|---|---|---|---|---|
| **train** | 0.76335 | 0.5983 | - | yes ✅ |
| **holdout** | 0.77172 | 0.59029 |  |  |
| aim40_eps1_seed0 | - | - | None | - |
| aim50_eps1_seed0 | - | - | None | - |
| ctgan_qt_seed0 | 0.87102 | 0.70909 | 0.0 | no |
| ctgan_seed0 | 0.80199 | 0.65455 | 0.0002 | no |
| ctgan_seed1 | 0.88295 | 0.81847 | 0.0 | no |
| ctgan_seed2 | 0.8679 | 0.79148 | 0.0 | no |
| ddpm_g_seed0 | 1.0 | 1.0 | 0.0 | no |
| ddpm_seed0 | 1.0 | 1.0 | 0.0 | no |
| ddpm_seed1 | 1.0 | 1.0 | 0.0 | no |
| ddpm_seed2 | 1.0 | 0.99972 | 0.0 | no |
| dpctgan_eps10_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed1 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps15_seed2 | 0.50142 | 0.07273 | 0.0 | no |
| dpctgan_eps1_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps20_seed0 | 1.0 | 1.0 | 0.0 | no |
| dpctgan_eps5_seed0 | 0.60114 | 0.02699 | 0.0 | no |
| dpctgan_eps8_seed0 | 1.0 | 1.0 | 0.0 | no |
| gaussian_copula_seed0 | 0.86364 | 0.67159 | 0.0 | no |
| gaussian_copula_seed1 | 0.87216 | 0.69972 | 0.0 | no |
| gaussian_copula_seed2 | 0.87074 | 0.6983 | 0.0 | no |
| mst_eps0p5_seed0 | 0.77614 | 0.5429 | 0.65 | no |
| mst_eps10_seed0 | 0.74119 | 0.6071 | 0.0184 | no |
| mst_eps15_seed0 | 0.74574 | 0.59659 | 0.0562 | yes ✅ |
| mst_eps15_seed1 | 0.74261 | 0.59261 | 0.0938 | no |
| mst_eps15_seed2 | 0.74403 | 0.60142 | 0.0349 | no |
| mst_eps1_seed0 | 0.77926 | 0.64545 | 0.0 | no |
| mst_eps20_seed0 | 0.74318 | 0.59347 | 0.088 | no |
| mst_eps5_seed0 | 0.74261 | 0.61307 | 0.0059 | no |
| mst_eps8_seed0 | 0.74119 | 0.59886 | 0.0516 | no |
| patectgan_eps15_seed0 | 0.99602 | 0.9892 | 0.0 | no |
| patectgan_eps1_seed0 | 0.96818 | 0.96761 | 0.0 | no |
| patectgan_eps5_seed0 | 0.99631 | 0.99574 | 0.0 | no |
| tvae_cap256_seed0 | 0.7875 | 0.59403 | 0.626 | yes ✅ |
| tvae_ep1000_seed0 | 0.82955 | 0.65227 | 0.0 | no |
| tvae_ind_seed0 | 0.74716 | 0.56989 | 0.3155 | yes ✅ |
| tvae_qt_seed0 | 0.72727 | 0.62699 | 0.062 | no |
| tvae_qt_seed1 | 0.75256 | 0.6358 | 0.0038 | no |
| tvae_qt_seed2 | 0.77102 | 0.62472 | 0.0111 | no |
| tvae_seed0 | 0.79659 | 0.64517 | 0.0001 | no |
| tvae_seed1 | 0.80284 | 0.66875 | 0.0 | no |
| tvae_seed2 | 0.77273 | 0.61932 | 0.0333 | no |

Equivalence is a POSITIVE claim (90% CI of the survival difference within ±5pp at every horizon) -- unlike a non-significant log-rank, which is only absence of evidence.

## hf_rehospitalization

train events 190/3520 | holdout 52/1174 | log-rank train-vs-holdout p = 0.185

| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |
|---|---|---|---|---|
| **train** | 0.9483 | 0.94631 | - | yes ✅ |
| **holdout** | 0.95997 | 0.95571 |  |  |
| aim40_eps1_seed0 | - | - | None | - |
| aim50_eps1_seed0 | - | - | None | - |
| ctgan_qt_seed0 | 0.98722 | 0.98466 | 0.0 | yes ✅ |
| ctgan_seed0 | 0.99205 | 0.99205 | 0.0 | yes ✅ |
| ctgan_seed1 | 0.97386 | 0.97386 | 0.0015 | yes ✅ |
| ctgan_seed2 | 0.99602 | 0.99602 | 0.0 | no |
| ddpm_g_seed0 | 1.0 | 1.0 | 0.0 | no |
| ddpm_seed0 | 1.0 | 0.99972 | 0.0 | no |
| ddpm_seed1 | 0.99972 | 0.99972 | 0.0 | no |
| ddpm_seed2 | 1.0 | 1.0 | 0.0 | no |
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
| mst_eps0p5_seed0 | 0.93267 | 0.8233 | 0.0 | no |
| mst_eps10_seed0 | 0.94574 | 0.94318 | 0.1453 | yes ✅ |
| mst_eps15_seed0 | 0.94631 | 0.94176 | 0.1035 | yes ✅ |
| mst_eps15_seed1 | 0.95256 | 0.95142 | 0.6776 | yes ✅ |
| mst_eps15_seed2 | 0.95199 | 0.94744 | 0.3477 | yes ✅ |
| mst_eps1_seed0 | 0.98097 | 0.92784 | 0.0016 | yes ✅ |
| mst_eps20_seed0 | 0.94858 | 0.94631 | 0.2818 | yes ✅ |
| mst_eps5_seed0 | 0.95341 | 0.94716 | 0.3262 | yes ✅ |
| mst_eps8_seed0 | 0.9429 | 0.94063 | 0.0795 | yes ✅ |
| patectgan_eps15_seed0 | 0.98807 | 0.98466 | 0.0 | yes ✅ |
| patectgan_eps1_seed0 | 0.98636 | 0.98636 | 0.0 | yes ✅ |
| patectgan_eps5_seed0 | 1.0 | 1.0 | 0.0 | no |
| tvae_cap256_seed0 | 0.95455 | 0.95455 | 0.8614 | yes ✅ |
| tvae_ep1000_seed0 | 0.96534 | 0.96534 | 0.1371 | yes ✅ |
| tvae_ind_seed0 | 0.96477 | 0.96477 | 0.1632 | yes ✅ |
| tvae_qt_seed0 | 0.9642 | 0.96364 | 0.2149 | yes ✅ |
| tvae_qt_seed1 | 0.95909 | 0.95881 | 0.654 | yes ✅ |
| tvae_qt_seed2 | 0.97301 | 0.97301 | 0.0034 | yes ✅ |
| tvae_seed0 | 0.96335 | 0.96335 | 0.2427 | yes ✅ |
| tvae_seed1 | 0.94602 | 0.94602 | 0.1958 | yes ✅ |
| tvae_seed2 | 0.9733 | 0.9733 | 0.0026 | yes ✅ |

Equivalence is a POSITIVE claim (90% CI of the survival difference within ±5pp at every horizon) -- unlike a non-significant log-rank, which is only absence of evidence.

## Effect-estimate replication

Model: cox_ph (lifelines), per-SD coefficients standardized by the real TRAIN split's mean/SD in every frame. The coefficient error is computed only over covariates present in BOTH the real and the synthetic fit (`matched`); runs sharing fewer than 4 covariates are not comparable.

| frame | n | events | sign agreement | coef matched | mean |coef error| |
|---|---|---|---|---|---|
| real train | 727 | 294 | - | - | - |
| real holdout | 281 | 108 | - | - | - |
| aim40_eps1_seed0 | - | - | - | - | not estimable |
| aim50_eps1_seed0 | - | - | - | - | not estimable |
| ctgan_qt_seed0 | 271 | 99 | 6/6 | 6/6 | 0.0442 |
| ctgan_seed0 | 442 | 187 | 2/6 | 6/6 | 0.1732 |
| ctgan_seed1 | 774 | 164 | 2/6 | 6/6 | 0.208 |
| ctgan_seed2 | 397 | 94 | 5/6 | 6/6 | 0.083 |
| ddpm_g_seed0 | - | - | - | - | not estimable |
| ddpm_seed0 | - | - | - | - | not estimable |
| ddpm_seed1 | - | - | - | - | not estimable |
| ddpm_seed2 | - | - | - | - | not estimable |
| dpctgan_eps10_seed0 | - | - | - | - | not estimable |
| dpctgan_eps15_seed0 | - | - | - | - | not estimable |
| dpctgan_eps15_seed1 | - | - | - | - | not estimable |
| dpctgan_eps15_seed2 | - | - | - | - | not estimable |
| dpctgan_eps1_seed0 | - | - | - | - | not estimable |
| dpctgan_eps20_seed0 | - | - | - | - | not estimable |
| dpctgan_eps5_seed0 | - | - | - | - | not estimable |
| dpctgan_eps8_seed0 | - | - | - | - | not estimable |
| gaussian_copula_seed0 | 712 | 220 | 5/6 | 6/6 | 0.0776 |
| gaussian_copula_seed1 | 721 | 215 | 4/6 | 6/6 | 0.0919 |
| gaussian_copula_seed2 | 736 | 217 | 3/6 | 6/6 | 0.1097 |
| mst_eps0p5_seed0 | 1122 | 664 | 3/6 | 6/6 | 0.398 |
| mst_eps10_seed0 | 810 | 329 | 5/6 | 6/6 | 0.1767 |
| mst_eps15_seed0 | 815 | 506 | 5/6 | 6/6 | 0.2392 |
| mst_eps15_seed1 | 814 | 402 | 5/6 | 6/6 | 0.2898 |
| mst_eps15_seed2 | 821 | 444 | 5/6 | 6/6 | 0.1834 |
| mst_eps1_seed0 | 759 | 546 | 1/6 | 6/6 | 0.3704 |
| mst_eps20_seed0 | 835 | 439 | 5/6 | 6/6 | 0.2184 |
| mst_eps5_seed0 | 783 | 304 | 3/6 | 6/6 | 0.286 |
| mst_eps8_seed0 | 824 | 622 | 5/6 | 6/6 | 0.1229 |
| patectgan_eps15_seed0 | - | - | - | - | not estimable |
| patectgan_eps1_seed0 | 2402 | 81 | 1/6 | 6/6 | 0.3767 |
| patectgan_eps5_seed0 | - | - | - | - | not estimable |
| tvae_cap256_seed0 | 109 | 54 | 3/6 | 6/6 | 0.5218 |
| tvae_ep1000_seed0 | 127 | 27 | 5/6 | 6/6 | 0.7413 |
| tvae_ind_seed0 | - | - | - | - | not estimable |
| tvae_qt_seed0 | 245 | 71 | 6/6 | 6/6 | 0.4001 |
| tvae_qt_seed1 | 193 | 36 | 4/6 | 6/6 | 0.6953 |
| tvae_qt_seed2 | 162 | 49 | 4/6 | 6/6 | 0.197 |
| tvae_seed0 | 117 | 26 | 5/6 | 6/6 | 1.1049 |
| tvae_seed1 | 128 | 41 | 2/6 | 6/6 | 0.6827 |
| tvae_seed2 | - | - | - | - | not estimable |
