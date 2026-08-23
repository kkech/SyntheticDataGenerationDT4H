# Survival Fidelity

Endpoints use the five-year follow-up columns: a recorded days-to-event is an event, a null is administrative censoring at 1825 days -- the same rule for real and synthetic data. The train-vs-holdout log-rank p-value calibrates what pure sampling noise looks like.

## all_cause_death

train events 1421/3520 | holdout 484/1174 | log-rank train-vs-holdout p = 0.6785

| run | 1y survival | 5y survival | log-rank vs holdout (p) |
|---|---|---|---|
| **train** | 0.76335 | 0.5983 | - |
| **holdout** | 0.77172 | 0.59029 |  |
| aim50_eps1_seed0 | - | - | None |
| aim50_eps5_seed0 | - | - | None |
| ctgan_seed0 | 0.80199 | 0.65455 | 0.0001 |
| ctgan_seed1 | 0.88295 | 0.81847 | 0.0 |
| ctgan_seed2 | 0.8679 | 0.79148 | 0.0 |
| dpctgan_eps10_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps15_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps15_seed1 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps15_seed2 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps1_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps20_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps5_seed0 | 0.99886 | 0.0 | 0.0 |
| dpctgan_eps8_seed0 | 1.0 | 1.0 | 0.0 |
| gaussian_copula_seed0 | 0.86364 | 0.67159 | 0.0 |
| gaussian_copula_seed1 | 0.87216 | 0.69972 | 0.0 |
| gaussian_copula_seed2 | 0.87074 | 0.6983 | 0.0 |
| mst_eps10_seed0 | 0.74688 | 0.59801 | 0.0483 |
| mst_eps15_seed0 | 0.74347 | 0.59489 | 0.0686 |
| mst_eps15_seed1 | 0.74432 | 0.59489 | 0.0676 |
| mst_eps15_seed2 | 0.74034 | 0.59773 | 0.0511 |
| mst_eps1_seed0 | 0.76676 | 0.65966 | 0.0 |
| mst_eps20_seed0 | 0.74574 | 0.59688 | 0.053 |
| mst_eps5_seed0 | 0.74318 | 0.61222 | 0.0055 |
| mst_eps8_seed0 | 0.74432 | 0.60653 | 0.0151 |
| tvae_seed0 | 0.79659 | 0.64517 | 0.0001 |
| tvae_seed1 | 0.80284 | 0.66875 | 0.0 |
| tvae_seed2 | 0.77273 | 0.61932 | 0.0301 |

## hf_rehospitalization

train events 190/3520 | holdout 52/1174 | log-rank train-vs-holdout p = 0.185

| run | 1y survival | 5y survival | log-rank vs holdout (p) |
|---|---|---|---|
| **train** | 0.9483 | 0.94631 | - |
| **holdout** | 0.95997 | 0.95571 |  |
| aim50_eps1_seed0 | - | - | None |
| aim50_eps5_seed0 | - | - | None |
| ctgan_seed0 | 0.99205 | 0.99205 | 0.0 |
| ctgan_seed1 | 0.97386 | 0.97386 | 0.0015 |
| ctgan_seed2 | 0.99602 | 0.99602 | 0.0 |
| dpctgan_eps10_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps15_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps15_seed1 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps15_seed2 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps1_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps20_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps5_seed0 | 1.0 | 1.0 | 0.0 |
| dpctgan_eps8_seed0 | 1.0 | 1.0 | 0.0 |
| gaussian_copula_seed0 | 1.0 | 1.0 | 0.0 |
| gaussian_copula_seed1 | 1.0 | 1.0 | 0.0 |
| gaussian_copula_seed2 | 1.0 | 1.0 | 0.0 |
| mst_eps10_seed0 | 0.95114 | 0.94858 | 0.4291 |
| mst_eps15_seed0 | 0.94602 | 0.94403 | 0.1763 |
| mst_eps15_seed1 | 0.94773 | 0.94517 | 0.2238 |
| mst_eps15_seed2 | 0.94943 | 0.94602 | 0.2642 |
| mst_eps1_seed0 | 0.92415 | 0.87017 | 0.0 |
| mst_eps20_seed0 | 0.95 | 0.94801 | 0.3892 |
| mst_eps5_seed0 | 0.94347 | 0.94063 | 0.0796 |
| mst_eps8_seed0 | 0.95057 | 0.94801 | 0.3887 |
| tvae_seed0 | 0.96335 | 0.96335 | 0.2427 |
| tvae_seed1 | 0.94602 | 0.94602 | 0.1958 |
| tvae_seed2 | 0.9733 | 0.9733 | 0.0026 |

## Effect-estimate replication

Model: cox_ph (lifelines), standardized (per-SD) coefficients.

| frame | n | events | sign agreement | mean |coef error| |
|---|---|---|---|---|
| real train | 727 | 294 | - | - |
| real holdout | 281 | 108 | - | - |
| aim50_eps1_seed0 | - | - | not estimable | - |
| aim50_eps5_seed0 | - | - | not estimable | - |
| ctgan_seed0 | 442 | 187 | 2/6 | 0.1648 |
| ctgan_seed1 | 774 | 164 | 2/6 | 0.2128 |
| ctgan_seed2 | 397 | 94 | 5/6 | 0.0981 |
| dpctgan_eps10_seed0 | - | - | not estimable | - |
| dpctgan_eps15_seed0 | - | - | not estimable | - |
| dpctgan_eps15_seed1 | - | - | not estimable | - |
| dpctgan_eps15_seed2 | - | - | not estimable | - |
| dpctgan_eps1_seed0 | - | - | not estimable | - |
| dpctgan_eps20_seed0 | - | - | not estimable | - |
| dpctgan_eps5_seed0 | - | - | not estimable | - |
| dpctgan_eps8_seed0 | - | - | not estimable | - |
| gaussian_copula_seed0 | 712 | 220 | 5/6 | 0.086 |
| gaussian_copula_seed1 | 721 | 215 | 4/6 | 0.0981 |
| gaussian_copula_seed2 | 736 | 217 | 3/6 | 0.1125 |
| mst_eps10_seed0 | 791 | 514 | 3/6 | 0.2158 |
| mst_eps15_seed0 | 811 | 665 | 5/6 | 0.1436 |
| mst_eps15_seed1 | 839 | 549 | 4/6 | 0.1534 |
| mst_eps15_seed2 | 803 | 415 | 4/6 | 0.2758 |
| mst_eps1_seed0 | 801 | 159 | 1/6 | 0.6175 |
| mst_eps20_seed0 | 825 | 419 | 4/6 | 0.2804 |
| mst_eps5_seed0 | 802 | 491 | 2/6 | 0.5493 |
| mst_eps8_seed0 | 816 | 345 | 5/6 | 0.2107 |
| tvae_seed0 | 117 | 26 | 5/6 | 0.9705 |
| tvae_seed1 | 128 | 41 | 2/6 | 0.4664 |
| tvae_seed2 | - | - | not estimable | - |
