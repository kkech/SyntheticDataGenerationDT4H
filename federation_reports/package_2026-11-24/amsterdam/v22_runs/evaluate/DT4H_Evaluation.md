# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 37 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 165 | 0.0002 | 0.0 | 1.0 | 0.0011 | 0.0 | 1.0 | 0.0002 |
| train vs holdout | 217 | 0.046 | 0.0348 | 0.9062 | 0.077 | 0.0131 | 1.0 | 0.0087 |
| train vs synthetic[aim40_eps10_seed0] | 40 | 0.459 | 0.4561 | 0.0714 | 0.643 | 0.0153 | 1.0 | 0.002 |
| train vs synthetic[aim40_eps15_seed0] | 40 | 0.459 | 0.4602 | 0.0714 | 0.6317 | 0.0139 | 0.9615 | 0.0023 |
| train vs synthetic[aim40_eps1_seed0] | 40 | 0.4625 | 0.4619 | 0.0714 | 0.8267 | 0.0195 | 0.9231 | 0.0038 |
| train vs synthetic[aim40_eps20_seed0] | 40 | 0.462 | 0.4669 | 0.0714 | 0.6366 | 0.0163 | 0.9231 | 0.0034 |
| train vs synthetic[aim40_eps8_seed0] | 40 | 0.4598 | 0.4578 | 0.0714 | 0.6429 | 0.0126 | 1.0 | 0.0016 |
| train vs synthetic[aim50_eps1_seed0] | 50 | 0.4455 | 0.4396 | 0.0556 | 0.7318 | 0.0153 | 0.9688 | 0.0066 |
| train vs synthetic[aim50_eps5_seed0] | 50 | 0.4403 | 0.4326 | 0.0556 | 0.5948 | 0.0145 | 0.9688 | 0.0041 |
| train vs synthetic[ctgan_qt_seed0] | 217 | 0.2457 | 0.2182 | 0.2031 | 0.6592 | 0.0726 | 0.5359 | 0.1133 |
| train vs synthetic[ctgan_seed0] | 217 | 0.2911 | 0.267 | 0.1094 | 0.5546 | 0.0729 | 0.5229 | 0.0799 |
| train vs synthetic[ctgan_seed1] | 217 | 0.3203 | 0.3009 | 0.125 | 0.6534 | 0.0717 | 0.451 | 0.0767 |
| train vs synthetic[ctgan_seed2] | 217 | 0.2944 | 0.27 | 0.1406 | 0.6205 | 0.0787 | 0.3987 | 0.0747 |
| train vs synthetic[ddpm_g_seed0] | 217 | 0.4935 | 0.57 | 0.1094 | 2.1091 | 0.2391 | 0.2157 | 0.1039 |
| train vs synthetic[ddpm_seed0] | 217 | 0.4447 | 0.5358 | 0.1406 | 2.197 | 0.0642 | 0.5098 | 0.0997 |
| train vs synthetic[ddpm_seed1] | 217 | 0.4445 | 0.538 | 0.1562 | 2.1844 | 0.0641 | 0.549 | 0.0992 |
| train vs synthetic[ddpm_seed2] | 217 | 0.4338 | 0.4436 | 0.1719 | 2.1515 | 0.08 | 0.4118 | 0.0917 |
| train vs synthetic[dpctgan_eps10_seed0] | 217 | 0.9326 | 1.0 | 0.0 | 4.5717 | 0.3032 | 0.1765 | 0.2962 |
| train vs synthetic[dpctgan_eps15_seed0] | 217 | 0.9586 | 1.0 | 0.0 | 4.6374 | 0.315 | 0.1699 | 0.3667 |
| train vs synthetic[dpctgan_eps15_seed1] | 217 | 0.9595 | 1.0 | 0.0 | 4.9039 | 0.2972 | 0.183 | 0.3995 |
| train vs synthetic[dpctgan_eps15_seed2] | 217 | 0.9191 | 1.0 | 0.0 | 3.7222 | 0.3325 | 0.1961 | 0.3157 |
| train vs synthetic[dpctgan_eps1_seed0] | 217 | 0.885 | 1.0 | 0.0156 | 4.1881 | 0.302 | 0.183 | 0.258 |
| train vs synthetic[dpctgan_eps20_seed0] | 217 | 0.8941 | 1.0 | 0.0 | 3.5995 | 0.3011 | 0.183 | 0.2965 |
| train vs synthetic[dpctgan_eps5_seed0] | 217 | 0.9507 | 1.0 | 0.0 | 4.3674 | 0.3175 | 0.183 | 0.3324 |
| train vs synthetic[dpctgan_eps8_seed0] | 217 | 0.8888 | 1.0 | 0.0 | 3.944 | 0.3045 | 0.1765 | 0.3245 |
| train vs synthetic[gaussian_copula_seed0] | 217 | 0.4173 | 0.2908 | 0.2344 | 1.3059 | 0.0116 | 0.9935 | 0.0974 |
| train vs synthetic[gaussian_copula_seed1] | 217 | 0.4154 | 0.281 | 0.2344 | 1.3028 | 0.0116 | 0.9935 | 0.0976 |
| train vs synthetic[gaussian_copula_seed2] | 217 | 0.4134 | 0.3 | 0.2344 | 1.2901 | 0.0112 | 0.9935 | 0.0951 |
| train vs synthetic[mst_eps0p5_seed0] | 217 | 0.5511 | 0.4914 | 0.0312 | 1.7649 | 0.0377 | 0.7124 | 0.0417 |
| train vs synthetic[mst_eps10_seed0] | 217 | 0.4926 | 0.4383 | 0.0312 | 0.7408 | 0.01 | 1.0 | 0.0045 |
| train vs synthetic[mst_eps15_seed0] | 217 | 0.4854 | 0.4073 | 0.0312 | 0.7095 | 0.0107 | 1.0 | 0.0037 |
| train vs synthetic[mst_eps15_seed1] | 217 | 0.4879 | 0.4212 | 0.0312 | 0.7065 | 0.0108 | 1.0 | 0.0044 |
| train vs synthetic[mst_eps15_seed2] | 217 | 0.486 | 0.4112 | 0.0312 | 0.7002 | 0.0103 | 1.0 | 0.0033 |
| train vs synthetic[mst_eps1_seed0] | 217 | 0.5181 | 0.4463 | 0.0312 | 1.5804 | 0.0234 | 0.9542 | 0.0241 |
| train vs synthetic[mst_eps20_seed0] | 217 | 0.4879 | 0.4143 | 0.0312 | 0.7187 | 0.0103 | 1.0 | 0.0035 |
| train vs synthetic[mst_eps5_seed0] | 217 | 0.4924 | 0.4273 | 0.0312 | 0.7864 | 0.0105 | 1.0 | 0.0054 |
| train vs synthetic[mst_eps8_seed0] | 217 | 0.4872 | 0.4156 | 0.0312 | 0.7445 | 0.0119 | 1.0 | 0.0045 |
| train vs synthetic[patectgan_eps15_seed0] | 217 | 0.3933 | 0.3529 | 0.0312 | 1.1618 | 0.088 | 0.5229 | 0.0845 |
| train vs synthetic[patectgan_eps1_seed0] | 217 | 0.5099 | 0.4686 | 0.0156 | 1.9198 | 0.0762 | 0.3922 | 0.1154 |
| train vs synthetic[patectgan_eps5_seed0] | 217 | 0.449 | 0.3781 | 0.0156 | 1.5014 | 0.1266 | 0.5425 | 0.1262 |
| train vs synthetic[tvae_cap256_seed0] | 217 | 0.1542 | 0.128 | 0.375 | 0.2265 | 0.0468 | 0.6144 | 0.0285 |
| train vs synthetic[tvae_ep1000_seed0] | 217 | 0.1608 | 0.1468 | 0.3906 | 0.2482 | 0.0482 | 0.634 | 0.0267 |
| train vs synthetic[tvae_ind_seed0] | 217 | 0.1885 | 0.1768 | 0.2031 | 0.2618 | 0.0585 | 0.5359 | 0.0256 |
| train vs synthetic[tvae_qt_seed0] | 217 | 0.168 | 0.1677 | 0.25 | 0.2712 | 0.0472 | 0.6078 | 0.0316 |
| train vs synthetic[tvae_qt_seed1] | 217 | 0.1776 | 0.1747 | 0.1875 | 0.2868 | 0.0578 | 0.5425 | 0.0348 |
| train vs synthetic[tvae_qt_seed2] | 217 | 0.1725 | 0.1407 | 0.25 | 0.2706 | 0.0486 | 0.6275 | 0.0323 |
| train vs synthetic[tvae_seed0] | 217 | 0.161 | 0.128 | 0.3594 | 0.2383 | 0.0504 | 0.6144 | 0.0302 |
| train vs synthetic[tvae_seed1] | 217 | 0.161 | 0.1377 | 0.3594 | 0.2451 | 0.0486 | 0.5882 | 0.0294 |
| train vs synthetic[tvae_seed2] | 217 | 0.16 | 0.1328 | 0.3125 | 0.2356 | 0.0499 | 0.6078 | 0.0269 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| aim | 1 | 1 | 0.4455 | 0.0153 | 0.0066 |
| aim | 5 | 1 | 0.4403 | 0.0145 | 0.0041 |
| aim40 | 1 | 1 | 0.4625 | 0.0195 | 0.0038 |
| aim40 | 8 | 1 | 0.4598 | 0.0126 | 0.0016 |
| aim40 | 10 | 1 | 0.459 | 0.0153 | 0.002 |
| aim40 | 15 | 1 | 0.459 | 0.0139 | 0.0023 |
| aim40 | 20 | 1 | 0.462 | 0.0163 | 0.0034 |
| ctgan | - | 3 | 0.3019 ± 0.016 | 0.0744 ± 0.0037 | 0.0771 ± 0.0026 |
| ctgan_qt | - | 1 | 0.2457 | 0.0726 | 0.1133 |
| ddpm | - | 3 | 0.441 ± 0.0062 | 0.0694 ± 0.0092 | 0.0969 ± 0.0045 |
| ddpm_g | - | 1 | 0.4935 | 0.2391 | 0.1039 |
| dpctgan | 1 | 1 | 0.885 | 0.302 | 0.258 |
| dpctgan | 5 | 1 | 0.9507 | 0.3175 | 0.3324 |
| dpctgan | 8 | 1 | 0.8888 | 0.3045 | 0.3245 |
| dpctgan | 10 | 1 | 0.9326 | 0.3032 | 0.2962 |
| dpctgan | 15 | 3 | 0.9457 ± 0.0231 | 0.3149 ± 0.0177 | 0.3606 ± 0.0422 |
| dpctgan | 20 | 1 | 0.8941 | 0.3011 | 0.2965 |
| gaussian_copula | - | 3 | 0.4154 ± 0.002 | 0.0115 ± 0.0002 | 0.0967 ± 0.0014 |
| mst | 0.5 | 1 | 0.5511 | 0.0377 | 0.0417 |
| mst | 1 | 1 | 0.5181 | 0.0234 | 0.0241 |
| mst | 5 | 1 | 0.4924 | 0.0105 | 0.0054 |
| mst | 8 | 1 | 0.4872 | 0.0119 | 0.0045 |
| mst | 10 | 1 | 0.4926 | 0.01 | 0.0045 |
| mst | 15 | 3 | 0.4864 ± 0.0013 | 0.0106 ± 0.0003 | 0.0038 ± 0.0006 |
| mst | 20 | 1 | 0.4879 | 0.0103 | 0.0035 |
| patectgan | 1 | 1 | 0.5099 | 0.0762 | 0.1154 |
| patectgan | 5 | 1 | 0.449 | 0.1266 | 0.1262 |
| patectgan | 15 | 1 | 0.3933 | 0.088 | 0.0845 |
| tvae | - | 3 | 0.1607 ± 0.0006 | 0.0496 ± 0.0009 | 0.0288 ± 0.0017 |
| tvae_cap256 | - | 1 | 0.1542 | 0.0468 | 0.0285 |
| tvae_ep1000 | - | 1 | 0.1608 | 0.0482 | 0.0267 |
| tvae_ind | - | 1 | 0.1885 | 0.0585 | 0.0256 |
| tvae_qt | - | 3 | 0.1727 ± 0.0048 | 0.0512 ± 0.0058 | 0.0329 ± 0.0017 |

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.4999**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| aim40_eps10_seed0 | 1.0 | 0.0 | 0.1843 |
| aim40_eps15_seed0 | 1.0 | 0.0 | 0.1843 |
| aim40_eps1_seed0 | 1.0 | 0.0 | 0.1843 |
| aim40_eps20_seed0 | 1.0 | 0.0 | 0.1843 |
| aim40_eps8_seed0 | 1.0 | 0.0 | 0.1843 |
| aim50_eps1_seed0 | 1.0 | 0.0 | 0.2304 |
| aim50_eps5_seed0 | 1.0 | 0.0 | 0.2304 |
| ctgan_qt_seed0 | 1.0 | 0.0 | 1.0 |
| ctgan_seed0 | 1.0 | 0.0001 | 1.0 |
| ctgan_seed1 | 1.0 | 0.0 | 1.0 |
| ctgan_seed2 | 1.0 | 0.0 | 1.0 |
| ddpm_g_seed0 | 1.0 | 0.0 | 1.0 |
| ddpm_seed0 | 0.9995 | 0.0007 | 1.0 |
| ddpm_seed1 | 0.9996 | 0.0004 | 1.0 |
| ddpm_seed2 | 0.9997 | 0.0003 | 1.0 |
| dpctgan_eps10_seed0 | 0.9996 | 0.0007 | 1.0 |
| dpctgan_eps15_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps15_seed1 | 0.9996 | 0.0007 | 1.0 |
| dpctgan_eps15_seed2 | 0.9998 | 0.0003 | 1.0 |
| dpctgan_eps1_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps20_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps5_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps8_seed0 | 1.0 | 0.0 | 1.0 |
| gaussian_copula_seed0 | 1.0 | 0.0 | 1.0 |
| gaussian_copula_seed1 | 1.0 | 0.0 | 1.0 |
| gaussian_copula_seed2 | 1.0 | 0.0 | 1.0 |
| mst_eps0p5_seed0 | 1.0 | 0.0 | 1.0 |
| mst_eps10_seed0 | 1.0 | 0.0 | 1.0 |
| mst_eps15_seed0 | 1.0 | 0.0 | 1.0 |
| mst_eps15_seed1 | 1.0 | 0.0 | 1.0 |
| mst_eps15_seed2 | 1.0 | 0.0 | 1.0 |
| mst_eps1_seed0 | 1.0 | 0.0 | 1.0 |
| mst_eps20_seed0 | 1.0 | 0.0 | 1.0 |
| mst_eps5_seed0 | 1.0 | 0.0 | 1.0 |
| mst_eps8_seed0 | 1.0 | 0.0 | 1.0 |
| patectgan_eps15_seed0 | 1.0 | 0.0 | 1.0 |
| patectgan_eps1_seed0 | 1.0 | 0.0 | 1.0 |
| patectgan_eps5_seed0 | 1.0 | 0.0 | 1.0 |
| tvae_cap256_seed0 | 0.9982 | 0.001 | 1.0 |
| tvae_ep1000_seed0 | 0.9973 | 0.0014 | 1.0 |
| tvae_ind_seed0 | 0.9993 | 0.0002 | 1.0 |
| tvae_qt_seed0 | 0.9978 | 0.0005 | 1.0 |
| tvae_qt_seed1 | 0.9986 | 0.0008 | 1.0 |
| tvae_qt_seed2 | 0.9976 | 0.001 | 1.0 |
| tvae_seed0 | 0.9983 | 0.001 | 1.0 |
| tvae_seed1 | 0.9984 | 0.0009 | 1.0 |
| tvae_seed2 | 0.9986 | 0.0005 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.0748 | 0.0609 | 0.0787 | 0.0673 | 0.0874 |
| aim40_eps10_seed0 | 0.4837 | 0.4703 | 0.5033 | 0.4785 | 0.5286 |
| aim40_eps15_seed0 | 0.4892 | 0.4708 | 0.5436 | 0.4811 | 0.5482 |
| aim40_eps1_seed0 | 0.4864 | 0.4625 | 0.5192 | 0.4756 | 0.5315 |
| aim40_eps20_seed0 | 0.4952 | 0.4748 | 0.5107 | 0.4779 | 0.5325 |
| aim40_eps8_seed0 | 0.4958 | 0.4719 | 0.5455 | 0.4824 | 0.5666 |
| aim50_eps1_seed0 | 0.4696 | 0.4441 | 0.4882 | 0.459 | 0.4989 |
| aim50_eps5_seed0 | 0.5113 | 0.4762 | 0.4933 | 0.4458 | 0.4896 |
| ctgan_qt_seed0 | 0.2592 | 0.2747 | 0.2897 | 0.2669 | 0.2852 |
| ctgan_seed0 | 0.3088 | 0.2998 | 0.3225 | 0.2947 | 0.3272 |
| ctgan_seed1 | 0.3339 | 0.3324 | 0.3374 | 0.3158 | 0.362 |
| ctgan_seed2 | 0.3056 | 0.3094 | 0.316 | 0.308 | 0.3319 |
| ddpm_g_seed0 | 0.501 | 0.495 | 0.5005 | 0.4956 | 0.5111 |
| ddpm_seed0 | 0.464 | 0.4447 | 0.4449 | 0.4533 | 0.46 |
| ddpm_seed1 | 0.4523 | 0.4439 | 0.4555 | 0.4376 | 0.4557 |
| ddpm_seed2 | 0.4434 | 0.4389 | 0.4536 | 0.4383 | 0.4422 |
| dpctgan_eps10_seed0 | - | 0.9338 | - | - | 0.9431 |
| dpctgan_eps15_seed0 | - | 0.9603 | - | - | 0.9637 |
| dpctgan_eps15_seed1 | 0.962 | - | 0.9533 | - | - |
| dpctgan_eps15_seed2 | 0.92 | - | 0.9179 | - | - |
| dpctgan_eps1_seed0 | 0.8852 | 0.8886 | - | - | 0.893 |
| dpctgan_eps20_seed0 | 0.8914 | - | - | 0.8905 | 0.8952 |
| dpctgan_eps5_seed0 | - | 0.9522 | - | - | 0.9526 |
| dpctgan_eps8_seed0 | 0.8914 | 0.8867 | - | - | 0.8989 |
| gaussian_copula_seed0 | 0.4304 | 0.4281 | 0.4411 | 0.4227 | 0.4211 |
| gaussian_copula_seed1 | 0.4278 | 0.4196 | 0.4378 | 0.4187 | 0.433 |
| gaussian_copula_seed2 | 0.4315 | 0.4269 | 0.4458 | 0.4223 | 0.441 |
| mst_eps0p5_seed0 | 0.5782 | 0.5582 | 0.5594 | 0.5715 | 0.629 |
| mst_eps10_seed0 | 0.512 | 0.4999 | 0.5338 | 0.5109 | 0.5524 |
| mst_eps15_seed0 | 0.5096 | 0.4954 | 0.5804 | 0.5092 | 0.6051 |
| mst_eps15_seed1 | 0.5062 | 0.494 | 0.5414 | 0.5083 | 0.5696 |
| mst_eps15_seed2 | 0.5122 | 0.4959 | 0.5968 | 0.5224 | 0.5856 |
| mst_eps1_seed0 | 0.5393 | 0.5201 | 0.5833 | 0.5543 | 0.6223 |
| mst_eps20_seed0 | 0.5041 | 0.4966 | 0.6085 | 0.5054 | 0.5987 |
| mst_eps5_seed0 | 0.515 | 0.498 | 0.5529 | 0.5221 | 0.5676 |
| mst_eps8_seed0 | 0.5055 | 0.4932 | 0.5246 | 0.5031 | 0.5591 |
| patectgan_eps15_seed0 | 0.4158 | 0.4083 | 0.4055 | 0.407 | 0.4468 |
| patectgan_eps1_seed0 | 0.5318 | 0.5235 | 0.5351 | 0.532 | 0.5635 |
| patectgan_eps5_seed0 | 0.4606 | 0.4517 | 0.4574 | 0.4546 | 0.4826 |
| tvae_cap256_seed0 | 0.1832 | 0.1651 | 0.1797 | 0.1664 | 0.2101 |
| tvae_ep1000_seed0 | 0.1989 | 0.1758 | 0.1932 | 0.1671 | 0.2176 |
| tvae_ind_seed0 | 0.2203 | 0.2006 | 0.2164 | 0.1942 | 0.2283 |
| tvae_qt_seed0 | 0.2145 | 0.1716 | 0.1869 | 0.1678 | 0.2175 |
| tvae_qt_seed1 | 0.2271 | 0.1889 | 0.2058 | 0.1845 | 0.2534 |
| tvae_qt_seed2 | 0.2122 | 0.1779 | 0.1941 | 0.1734 | 0.2191 |
| tvae_seed0 | 0.1944 | 0.175 | 0.1986 | 0.1696 | 0.2185 |
| tvae_seed1 | 0.1922 | 0.1735 | 0.1992 | 0.1758 | 0.2127 |
| tvae_seed2 | 0.1956 | 0.1733 | 0.1869 | 0.1676 | 0.2263 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| aim40_eps10_seed0 | 0.459 | 0.4631 | 0.0153 | 0.0174 |
| aim40_eps15_seed0 | 0.459 | 0.4627 | 0.0139 | 0.0192 |
| aim40_eps1_seed0 | 0.4625 | 0.4624 | 0.0195 | 0.0232 |
| aim40_eps20_seed0 | 0.462 | 0.4645 | 0.0163 | 0.0184 |
| aim40_eps8_seed0 | 0.4598 | 0.4645 | 0.0126 | 0.016 |
| aim50_eps1_seed0 | 0.4455 | 0.4454 | 0.0153 | 0.0203 |
| aim50_eps5_seed0 | 0.4403 | 0.4454 | 0.0145 | 0.0175 |
| ctgan_qt_seed0 | 0.2457 | 0.2457 | 0.0726 | 0.0737 |
| ctgan_seed0 | 0.2911 | 0.2857 | 0.0729 | 0.0743 |
| ctgan_seed1 | 0.3203 | 0.3089 | 0.0717 | 0.0724 |
| ctgan_seed2 | 0.2944 | 0.2833 | 0.0787 | 0.0793 |
| ddpm_g_seed0 | 0.4935 | 0.483 | 0.2391 | 0.2443 |
| ddpm_seed0 | 0.4447 | 0.4441 | 0.0642 | 0.0606 |
| ddpm_seed1 | 0.4445 | 0.4409 | 0.0641 | 0.0581 |
| ddpm_seed2 | 0.4338 | 0.4328 | 0.08 | 0.0781 |
| dpctgan_eps10_seed0 | 0.9326 | 0.932 | 0.3032 | 0.3053 |
| dpctgan_eps15_seed0 | 0.9586 | 0.9576 | 0.315 | 0.3163 |
| dpctgan_eps15_seed1 | 0.9595 | 0.9598 | 0.2972 | 0.3005 |
| dpctgan_eps15_seed2 | 0.9191 | 0.918 | 0.3325 | 0.3347 |
| dpctgan_eps1_seed0 | 0.885 | 0.8846 | 0.302 | 0.3055 |
| dpctgan_eps20_seed0 | 0.8941 | 0.8862 | 0.3011 | 0.3017 |
| dpctgan_eps5_seed0 | 0.9507 | 0.9484 | 0.3175 | 0.3202 |
| dpctgan_eps8_seed0 | 0.8888 | 0.8868 | 0.3045 | 0.307 |
| gaussian_copula_seed0 | 0.4173 | 0.4068 | 0.0116 | 0.0178 |
| gaussian_copula_seed1 | 0.4154 | 0.4063 | 0.0116 | 0.0156 |
| gaussian_copula_seed2 | 0.4134 | 0.4076 | 0.0112 | 0.0158 |
| mst_eps0p5_seed0 | 0.5511 | 0.5447 | 0.0377 | 0.0378 |
| mst_eps10_seed0 | 0.4926 | 0.4927 | 0.01 | 0.0134 |
| mst_eps15_seed0 | 0.4854 | 0.4863 | 0.0107 | 0.0133 |
| mst_eps15_seed1 | 0.4879 | 0.4876 | 0.0108 | 0.0131 |
| mst_eps15_seed2 | 0.486 | 0.4873 | 0.0103 | 0.0134 |
| mst_eps1_seed0 | 0.5181 | 0.518 | 0.0234 | 0.0249 |
| mst_eps20_seed0 | 0.4879 | 0.4875 | 0.0103 | 0.0133 |
| mst_eps5_seed0 | 0.4924 | 0.4935 | 0.0105 | 0.0139 |
| mst_eps8_seed0 | 0.4872 | 0.4859 | 0.0119 | 0.0139 |
| patectgan_eps15_seed0 | 0.3933 | 0.3938 | 0.088 | 0.0931 |
| patectgan_eps1_seed0 | 0.5099 | 0.4982 | 0.0762 | 0.0715 |
| patectgan_eps5_seed0 | 0.449 | 0.4462 | 0.1266 | 0.1263 |
| tvae_cap256_seed0 | 0.1542 | 0.1538 | 0.0468 | 0.047 |
| tvae_ep1000_seed0 | 0.1608 | 0.1616 | 0.0482 | 0.0495 |
| tvae_ind_seed0 | 0.1885 | 0.1915 | 0.0585 | 0.0617 |
| tvae_qt_seed0 | 0.168 | 0.1693 | 0.0472 | 0.0482 |
| tvae_qt_seed1 | 0.1776 | 0.1749 | 0.0578 | 0.0593 |
| tvae_qt_seed2 | 0.1725 | 0.1687 | 0.0486 | 0.0506 |
| tvae_seed0 | 0.161 | 0.1618 | 0.0504 | 0.0537 |
| tvae_seed1 | 0.161 | 0.167 | 0.0486 | 0.0479 |
| tvae_seed2 | 0.16 | 0.1647 | 0.0499 | 0.0494 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 1799 | 0.0523 | 0.0375 | 0.8755 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 11628 | 0.0221 | 0.0174 | 0.997 | 0 | - |
| *noise floor* | corr-ratio (num-cat) | 12160 | 0.0276 | 0.0169 | 0.9649 | 5 | - |
| aim40_eps10_seed0 | Spearman (num-num) | 91 | 0.2118 | 0.1167 | 0.4505 | 1 | `lab_results_valideGFR_value_first|lab_results_creatBS_value_p3a_avg` (-0.8588 -> 0.0426) |
| aim40_eps10_seed0 | Cramer's V (cat-cat) | 325 | 0.1574 | 0.0761 | 0.5723 | 19 | `ckd_severity_categorizedValue|conditions_mc` (0.0407 -> 0.8235) |
| aim40_eps10_seed0 | corr-ratio (num-cat) | 882 | 0.0574 | 0.0 | 0.8152 | 14 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.8632 -> 0.1499) |
| aim40_eps15_seed0 | Spearman (num-num) | 91 | 0.2178 | 0.1323 | 0.4286 | 2 | `lab_results_valideGFR_value_first|lab_results_creatBS_value_p3a_avg` (-0.8588 -> -0.0036) |
| aim40_eps15_seed0 | Cramer's V (cat-cat) | 325 | 0.1388 | 0.076 | 0.5692 | 8 | `ckd_severity_categorizedValue|conditions_mc` (0.0407 -> 0.8198) |
| aim40_eps15_seed0 | corr-ratio (num-cat) | 882 | 0.0608 | 0.0 | 0.805 | 14 | `lab_results_creatBS_value_p3a_avg|ckd_severity_categorizedValue` (0.7433 -> 0.0654) |
| aim40_eps1_seed0 | Spearman (num-num) | 91 | 0.2189 | 0.1251 | 0.4615 | 3 | `lab_results_valideGFR_value_first|lab_results_validSerumCreatinine_value_pET` (-0.7309 -> 0.5137) |
| aim40_eps1_seed0 | Cramer's V (cat-cat) | 325 | 0.1333 | 0.0569 | 0.6369 | 17 | `encounters_admissionYear|med_activeDuringEncounter_ace_inhibitors_arb_use` (0.0504 -> 0.833) |
| aim40_eps1_seed0 | corr-ratio (num-cat) | 882 | 0.0568 | 0.0 | 0.8311 | 14 | `nyha_nyha_pET|encounters_admissionYear` (0.0583 -> 0.8796) |
| aim40_eps20_seed0 | Spearman (num-num) | 90 | 0.1708 | 0.0889 | 0.5556 | 3 | `lab_results_creatBS_value_p3a_avg|lab_results_validSerumCreatinine_value_pET` (0.8279 -> -0.0551) |
| aim40_eps20_seed0 | Cramer's V (cat-cat) | 325 | 0.1416 | 0.0662 | 0.5815 | 12 | `ckd_severity_calculated_or_measured|conditions_mc` (0.0507 -> 0.8184) |
| aim40_eps20_seed0 | corr-ratio (num-cat) | 882 | 0.0507 | 0.0 | 0.8231 | 4 | `lab_results_crpNonHs_value_last|ckd_severity_calculated_or_measured` (0.0896 -> 0.7528) |
| aim40_eps8_seed0 | Spearman (num-num) | 91 | 0.2448 | 0.181 | 0.3407 | 3 | `lab_results_valideGFR_value_first|lab_results_creatBS_value_p3a_avg` (-0.8588 -> 0.0078) |
| aim40_eps8_seed0 | Cramer's V (cat-cat) | 325 | 0.1601 | 0.087 | 0.5692 | 24 | `conditions_heart_failure_occurred_prior_to_18_months_any|encounter_primary_reason_HF_Disease_f5a_w1mo_first` (0.0385 -> 0.7831) |
| aim40_eps8_seed0 | corr-ratio (num-cat) | 882 | 0.0597 | 0.0 | 0.7971 | 9 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_categorizedValue` (0.8768 -> 0.1559) |
| aim50_eps1_seed0 | Spearman (num-num) | 153 | 0.1581 | 0.0869 | 0.5817 | 0 | `lab_results_valideGFR_value_last|lab_results_creatBS_value_p3a_avg` (-0.8382 -> -0.0002) |
| aim50_eps1_seed0 | Cramer's V (cat-cat) | 496 | 0.1277 | 0.0532 | 0.6492 | 19 | `encounters_admissionYear|med_activeDuringEncounter_ace_inhibitors_arb_use` (0.0504 -> 0.8558) |
| aim50_eps1_seed0 | corr-ratio (num-cat) | 1242 | 0.0456 | 0.0 | 0.8567 | 2 | `conditions_heartFailure_timeFromEarliest_first|conditions_heart_failure_occurred_prior_to_18_months_any` (0.8515 -> 0.0118) |
| aim50_eps5_seed0 | Spearman (num-num) | 153 | 0.2195 | 0.1316 | 0.4183 | 7 | `lab_results_valideGFR_value_first|lab_results_creatBS_value_p3a_avg` (-0.8588 -> 0.0092) |
| aim50_eps5_seed0 | Cramer's V (cat-cat) | 496 | 0.1797 | 0.1206 | 0.4778 | 31 | `ckd_severity_from_calculated_egfr|med_sglt2i` (0.0591 -> 0.9345) |
| aim50_eps5_seed0 | corr-ratio (num-cat) | 1242 | 0.0793 | 0.0 | 0.7625 | 40 | `vital_signs_heartRate_value_last|encounters_admissionYear` (0.0547 -> 0.8424) |
| ctgan_qt_seed0 | Spearman (num-num) | 1372 | 0.1194 | 0.0789 | 0.5816 | 0 | `lab_results_validSerumCreatinine_value_pET|eGFR_2021_ckd_epi_creatinine` (-0.9257 -> 0.1128) |
| ctgan_qt_seed0 | Cramer's V (cat-cat) | 11628 | 0.0729 | 0.0348 | 0.7795 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0051) |
| ctgan_qt_seed0 | corr-ratio (num-cat) | 10640 | 0.0432 | 0.0237 | 0.8945 | 1 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9561 -> 0.0209) |
| ctgan_seed0 | Spearman (num-num) | 1575 | 0.1179 | 0.0778 | 0.593 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.9914 -> -0.074) |
| ctgan_seed0 | Cramer's V (cat-cat) | 11628 | 0.0598 | 0.027 | 0.8744 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0044) |
| ctgan_seed0 | corr-ratio (num-cat) | 11590 | 0.046 | 0.0244 | 0.8808 | 1 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9733 -> 0.094) |
| ctgan_seed1 | Spearman (num-num) | 1558 | 0.12 | 0.0842 | 0.5732 | 0 | `vital_signs_weight_value_p6mo_last|vital_signs_weight_value_last` (1.0 -> 0.0019) |
| ctgan_seed1 | Cramer's V (cat-cat) | 11628 | 0.0685 | 0.0286 | 0.8535 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0041) |
| ctgan_seed1 | corr-ratio (num-cat) | 11400 | 0.0455 | 0.0246 | 0.8835 | 4 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9561 -> 0.0531) |
| ctgan_seed2 | Spearman (num-num) | 1589 | 0.1167 | 0.079 | 0.589 | 0 | `lab_results_ntProBnp_value_first|lab_results_ntProBnp_value_last` (0.9138 -> -0.2177) |
| ctgan_seed2 | Cramer's V (cat-cat) | 11628 | 0.0679 | 0.03 | 0.8485 | 0 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0036) |
| ctgan_seed2 | corr-ratio (num-cat) | 11210 | 0.0454 | 0.024 | 0.8821 | 3 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9733 -> 0.0621) |
| ddpm_g_seed0 | Spearman (num-num) | 1346 | 0.1065 | 0.0734 | 0.6152 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2512 -> 0.9434) |
| ddpm_g_seed0 | Cramer's V (cat-cat) | 3828 | 0.0484 | 0.0225 | 0.9177 | 0 | `cause_of_death_isRenal_f5a_w6mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first` (1.0 -> 0.0033) |
| ddpm_g_seed0 | corr-ratio (num-cat) | 10640 | 0.0455 | 0.0277 | 0.8696 | 1 | `smoking_status_smoker_startTime_count|conditions_dep` (0.0114 -> 0.7068) |
| ddpm_seed0 | Spearman (num-num) | 1596 | 0.103 | 0.0719 | 0.6316 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2512 -> 0.9462) |
| ddpm_seed0 | Cramer's V (cat-cat) | 11628 | 0.0476 | 0.0246 | 0.8867 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0016) |
| ddpm_seed0 | corr-ratio (num-cat) | 11590 | 0.0383 | 0.0211 | 0.9147 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_renal_complications_f5a_w1a_first` (0.8438 -> 0.0228) |
| ddpm_seed1 | Spearman (num-num) | 1482 | 0.1021 | 0.0723 | 0.6235 | 9 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2512 -> 0.9444) |
| ddpm_seed1 | Cramer's V (cat-cat) | 11628 | 0.0482 | 0.0254 | 0.8912 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0016) |
| ddpm_seed1 | corr-ratio (num-cat) | 10830 | 0.0371 | 0.0213 | 0.9207 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (0.8411 -> 0.0713) |
| ddpm_seed2 | Spearman (num-num) | 1558 | 0.1006 | 0.0734 | 0.6406 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2512 -> 0.9534) |
| ddpm_seed2 | Cramer's V (cat-cat) | 11476 | 0.047 | 0.0256 | 0.8884 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.1418) |
| ddpm_seed2 | corr-ratio (num-cat) | 11020 | 0.0381 | 0.0214 | 0.9184 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (0.8411 -> 0.036) |
| dpctgan_eps10_seed0 | Spearman (num-num) | 660 | 0.3118 | 0.222 | 0.2758 | 83 | `lab_results_creatBS_value_p3a_avg|lab_results_validSerumCreatinine_value_pET` (0.8279 -> -0.7663) |
| dpctgan_eps10_seed0 | Cramer's V (cat-cat) | 4005 | 0.1131 | 0.0475 | 0.772 | 1 | `cause_of_death_isCV_f5a_w1mo_first|cause_of_death_isAllCause_f5a_w1mo_first` (1.0 -> 0.0009) |
| dpctgan_eps10_seed0 | corr-ratio (num-cat) | 7030 | 0.0477 | 0.0248 | 0.8636 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9561 -> 0.0041) |
| dpctgan_eps15_seed0 | Spearman (num-num) | 296 | 0.4305 | 0.3506 | 0.2027 | 84 | `lab_results_creatBS_value_p3a_avg|lab_results_validSerumCreatinine_value_pET` (0.8279 -> -0.9153) |
| dpctgan_eps15_seed0 | Cramer's V (cat-cat) | 3570 | 0.0927 | 0.0421 | 0.7852 | 5 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.0009) |
| dpctgan_eps15_seed0 | corr-ratio (num-cat) | 4750 | 0.0447 | 0.0243 | 0.8899 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9561 -> 0.0715) |
| dpctgan_eps15_seed1 | Spearman (num-num) | 323 | 0.4629 | 0.4373 | 0.1858 | 96 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7469 -> 0.9763) |
| dpctgan_eps15_seed1 | Cramer's V (cat-cat) | 5151 | 0.0933 | 0.0438 | 0.7886 | 1 | `encounter_primary_reason_CV_Disease_f5a_w7d_first|encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (1.0 -> 0.0009) |
| dpctgan_eps15_seed1 | corr-ratio (num-cat) | 4940 | 0.0458 | 0.0254 | 0.8767 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9561 -> 0.0612) |
| dpctgan_eps15_seed2 | Spearman (num-num) | 528 | 0.3599 | 0.1974 | 0.2917 | 88 | `lab_results_validSerumCreatinine_value_pET|eGFR_2021_ckd_epi_creatinine` (-0.9257 -> 0.8932) |
| dpctgan_eps15_seed2 | Cramer's V (cat-cat) | 4950 | 0.0987 | 0.0453 | 0.7851 | 2 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0009) |
| dpctgan_eps15_seed2 | corr-ratio (num-cat) | 6270 | 0.0471 | 0.0252 | 0.8786 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9733 -> 0.0118) |
| dpctgan_eps1_seed0 | Spearman (num-num) | 495 | 0.3206 | 0.2109 | 0.2949 | 104 | `vital_signs_weight_value_last|vital_signs_bmi_value_pET` (0.7748 -> -0.4432) |
| dpctgan_eps1_seed0 | Cramer's V (cat-cat) | 7140 | 0.0982 | 0.0415 | 0.7854 | 2 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.0009) |
| dpctgan_eps1_seed0 | corr-ratio (num-cat) | 6080 | 0.0442 | 0.0227 | 0.8844 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9561 -> 0.0227) |
| dpctgan_eps20_seed0 | Spearman (num-num) | 619 | 0.2725 | 0.1874 | 0.3183 | 54 | `lab_results_validSerumCreatinine_value_first|lab_results_creatBS_value_p3a_avg` (0.8851 -> -0.7313) |
| dpctgan_eps20_seed0 | Cramer's V (cat-cat) | 4851 | 0.0922 | 0.0438 | 0.785 | 1 | `encounter_primary_reason_CV_Disease_f5a_w6mo_first|encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (1.0 -> 0.0016) |
| dpctgan_eps20_seed0 | corr-ratio (num-cat) | 6840 | 0.0489 | 0.0264 | 0.8665 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9733 -> 0.0272) |
| dpctgan_eps5_seed0 | Spearman (num-num) | 460 | 0.361 | 0.2983 | 0.2087 | 74 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7469 -> 0.982) |
| dpctgan_eps5_seed0 | Cramer's V (cat-cat) | 4278 | 0.089 | 0.0414 | 0.813 | 5 | `encounter_primary_reason_CV_Disease_f5a_w6mo_first|encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (1.0 -> 0.0013) |
| dpctgan_eps5_seed0 | corr-ratio (num-cat) | 5890 | 0.0466 | 0.0252 | 0.8716 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9561 -> 0.0) |
| dpctgan_eps8_seed0 | Spearman (num-num) | 628 | 0.302 | 0.2101 | 0.3089 | 86 | `lab_results_validSerumCreatinine_value_first|lab_results_creatBS_value_p3a_avg` (0.8851 -> -0.8282) |
| dpctgan_eps8_seed0 | Cramer's V (cat-cat) | 4851 | 0.0985 | 0.044 | 0.7963 | 0 | `cause_of_death_isRenal_f5a_w3mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first` (1.0 -> 0.0009) |
| dpctgan_eps8_seed0 | corr-ratio (num-cat) | 6840 | 0.0462 | 0.0257 | 0.873 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9561 -> 0.0134) |
| gaussian_copula_seed0 | Spearman (num-num) | 1236 | 0.1024 | 0.0625 | 0.6497 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2512 -> 0.9411) |
| gaussian_copula_seed0 | Cramer's V (cat-cat) | 11628 | 0.0613 | 0.0252 | 0.8764 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0026) |
| gaussian_copula_seed0 | corr-ratio (num-cat) | 9690 | 0.0349 | 0.0192 | 0.926 | 2 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isAllCause_f5a_w1a_first` (0.8381 -> 0.1303) |
| gaussian_copula_seed1 | Spearman (num-num) | 1224 | 0.1011 | 0.065 | 0.6503 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2512 -> 0.9424) |
| gaussian_copula_seed1 | Cramer's V (cat-cat) | 11628 | 0.0616 | 0.0252 | 0.8764 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0024) |
| gaussian_copula_seed1 | corr-ratio (num-cat) | 9690 | 0.0346 | 0.0193 | 0.9288 | 3 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isCV_f5a_w1a_first` (0.8381 -> 0.1048) |
| gaussian_copula_seed2 | Spearman (num-num) | 1240 | 0.0988 | 0.0609 | 0.6734 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2512 -> 0.9396) |
| gaussian_copula_seed2 | Cramer's V (cat-cat) | 11628 | 0.062 | 0.0253 | 0.875 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0029) |
| gaussian_copula_seed2 | corr-ratio (num-cat) | 9690 | 0.0355 | 0.0201 | 0.9269 | 1 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isCV_f5a_w1a_first` (0.8381 -> 0.2092) |
| mst_eps0p5_seed0 | Spearman (num-num) | 1802 | 0.2666 | 0.2146 | 0.2458 | 139 | `lab_results_validSerumCreatinine_value_pET|eGFR_2021_ckd_epi_creatinine` (-0.9257 -> 0.2877) |
| mst_eps0p5_seed0 | Cramer's V (cat-cat) | 10878 | 0.2854 | 0.2599 | 0.2232 | 1890 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first` (1.0 -> 0.021) |
| mst_eps0p5_seed0 | corr-ratio (num-cat) | 11970 | 0.172 | 0.103 | 0.4952 | 936 | `lab_results_triGly_value_first|encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (0.0031 -> 1.0) |
| mst_eps10_seed0 | Spearman (num-num) | 1880 | 0.2972 | 0.2647 | 0.1899 | 188 | `eGFR_2021_ckd_epi_creatinine|maggic_total_score` (-0.5152 -> 0.6519) |
| mst_eps10_seed0 | Cramer's V (cat-cat) | 11628 | 0.2645 | 0.2329 | 0.2762 | 2003 | `med_arni_history|conditions_tia` (0.0073 -> 0.9799) |
| mst_eps10_seed0 | corr-ratio (num-cat) | 12160 | 0.1882 | 0.1309 | 0.4539 | 1120 | `vital_signs_oxygenSaturation_value_first|cause_of_death_isAllCause_f5a_w6mo_first` (0.0227 -> 1.0) |
| mst_eps15_seed0 | Spearman (num-num) | 1773 | 0.3398 | 0.3135 | 0.1517 | 251 | `eGFR_2021_ckd_epi_creatinine|maggic_total_score` (-0.5152 -> 0.6434) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 11628 | 0.2751 | 0.2519 | 0.2448 | 1895 | `med_arni_history|conditions_tia` (0.0073 -> 0.9932) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 11780 | 0.2049 | 0.1632 | 0.4072 | 1245 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_renal_complications_f5a_w5a_first` (0.0 -> 0.9165) |
| mst_eps15_seed1 | Spearman (num-num) | 1808 | 0.3006 | 0.2744 | 0.172 | 170 | `eGFR_2021_ckd_epi_creatinine|maggic_total_score` (-0.5152 -> 0.5372) |
| mst_eps15_seed1 | Cramer's V (cat-cat) | 11628 | 0.2711 | 0.2437 | 0.2308 | 1863 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0031) |
| mst_eps15_seed1 | corr-ratio (num-cat) | 11780 | 0.1945 | 0.1535 | 0.4159 | 1073 | `lab_results_ferritin_value_first|encounter_primary_reason_renal_complications_f5a_w3a_first` (0.0705 -> 0.9282) |
| mst_eps15_seed2 | Spearman (num-num) | 1784 | 0.3161 | 0.2829 | 0.1833 | 238 | `vital_signs_bmi_value_pET|maggic_total_score` (-0.2519 -> 0.894) |
| mst_eps15_seed2 | Cramer's V (cat-cat) | 11628 | 0.2636 | 0.229 | 0.2812 | 1945 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0018) |
| mst_eps15_seed2 | corr-ratio (num-cat) | 11780 | 0.1919 | 0.1407 | 0.4345 | 1129 | `lab_results_triGly_value_first|cause_of_death_isCV_f5a_w6mo_first` (0.0813 -> 0.9713) |
| mst_eps1_seed0 | Spearman (num-num) | 1784 | 0.2452 | 0.2058 | 0.2618 | 98 | `lab_results_creatUS_value_first|lab_results_creatUS_value_last` (0.6713 -> -0.739) |
| mst_eps1_seed0 | Cramer's V (cat-cat) | 11628 | 0.2545 | 0.2314 | 0.2469 | 1478 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.0304) |
| mst_eps1_seed0 | corr-ratio (num-cat) | 11780 | 0.1562 | 0.102 | 0.4963 | 632 | `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count|med_rdoad_syst_history` (0.0185 -> 0.9047) |
| mst_eps20_seed0 | Spearman (num-num) | 1793 | 0.3288 | 0.3071 | 0.1517 | 255 | `eGFR_2021_ckd_epi_creatinine|maggic_total_score` (-0.5152 -> 0.5823) |
| mst_eps20_seed0 | Cramer's V (cat-cat) | 11628 | 0.2702 | 0.2457 | 0.2626 | 1905 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (1.0 -> 0.0031) |
| mst_eps20_seed0 | corr-ratio (num-cat) | 11780 | 0.1969 | 0.1514 | 0.4258 | 1135 | `lab_results_ferritin_value_last|cause_of_death_isCV_f5a_w7d_first` (0.0594 -> 0.9277) |
| mst_eps5_seed0 | Spearman (num-num) | 1820 | 0.2812 | 0.2504 | 0.2011 | 149 | `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count|encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first` (-0.1164 -> 0.9375) |
| mst_eps5_seed0 | Cramer's V (cat-cat) | 11476 | 0.2912 | 0.2728 | 0.2254 | 2244 | `med_bb|med_anti_coag_history` (0.0047 -> 0.9618) |
| mst_eps5_seed0 | corr-ratio (num-cat) | 11780 | 0.1745 | 0.1249 | 0.4553 | 843 | `lab_results_triGly_value_last|encounter_primary_reason_renal_complications_f5a_w7d_first` (0.0243 -> 0.892) |
| mst_eps8_seed0 | Spearman (num-num) | 1843 | 0.3007 | 0.2684 | 0.1899 | 209 | `lab_results_cholTot_value_first|conditions_heartFailure_timeFromEarliest_first` (-0.1934 -> 0.8231) |
| mst_eps8_seed0 | Cramer's V (cat-cat) | 11628 | 0.295 | 0.2661 | 0.2354 | 2429 | `med_bb|med_anti_coag_history` (0.0047 -> 0.9414) |
| mst_eps8_seed0 | corr-ratio (num-cat) | 11970 | 0.187 | 0.1274 | 0.4514 | 1209 | `lab_results_triGly_value_last|encounter_primary_reason_renal_complications_f5a_w6mo_first` (0.0264 -> 1.0) |
| patectgan_eps15_seed0 | Spearman (num-num) | 1678 | 0.1157 | 0.084 | 0.5763 | 1 | `echocardiographs_lvef|echocardiographs_lvef_pET_last` (1.0 -> 0.0787) |
| patectgan_eps15_seed0 | Cramer's V (cat-cat) | 10440 | 0.0546 | 0.0275 | 0.9099 | 1 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.0009) |
| patectgan_eps15_seed0 | corr-ratio (num-cat) | 11400 | 0.041 | 0.0247 | 0.8985 | 2 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (0.8437 -> 0.024) |
| patectgan_eps1_seed0 | Spearman (num-num) | 1251 | 0.1233 | 0.0821 | 0.5851 | 0 | `vital_signs_systolicBp_value_last|vital_signs_systolicBpDuringEncounter_value_pET` (0.9965 -> -0.0138) |
| patectgan_eps1_seed0 | Cramer's V (cat-cat) | 11628 | 0.0764 | 0.0274 | 0.8572 | 0 | `cause_of_death_isRenal_f5a_w1a_first|cause_of_death_isNonRenalAndNonCV_f5a_w1a_first` (1.0 -> 0.0025) |
| patectgan_eps1_seed0 | corr-ratio (num-cat) | 9690 | 0.0389 | 0.0203 | 0.9083 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9733 -> 0.0395) |
| patectgan_eps5_seed0 | Spearman (num-num) | 1613 | 0.1296 | 0.0942 | 0.5263 | 4 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7469 -> 0.3777) |
| patectgan_eps5_seed0 | Cramer's V (cat-cat) | 9870 | 0.0683 | 0.0281 | 0.8867 | 1 | `cause_of_death_isCV_f5a_w6mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first` (1.0 -> 0.0009) |
| patectgan_eps5_seed0 | corr-ratio (num-cat) | 11020 | 0.0436 | 0.0257 | 0.8933 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_CV_Disease_f5a_w1a_first` (0.8437 -> 0.0251) |
| tvae_cap256_seed0 | Spearman (num-num) | 1634 | 0.075 | 0.0558 | 0.7472 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.9914 -> 0.4099) |
| tvae_cap256_seed0 | Cramer's V (cat-cat) | 8911 | 0.0454 | 0.0291 | 0.8915 | 0 | `med_acei_history|med_arb_history` (0.0172 -> 0.466) |
| tvae_cap256_seed0 | corr-ratio (num-cat) | 11400 | 0.0371 | 0.0216 | 0.9088 | 1 | `lab_results_triGly_value_first|conditions_ap` (0.0711 -> 0.5158) |
| tvae_ep1000_seed0 | Spearman (num-num) | 1635 | 0.0815 | 0.0631 | 0.7028 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.9914 -> 0.3755) |
| tvae_ep1000_seed0 | Cramer's V (cat-cat) | 9316 | 0.0493 | 0.0299 | 0.8594 | 2 | `conditions_aidshiv|conditions_rd` (0.0052 -> 1.0) |
| tvae_ep1000_seed0 | corr-ratio (num-cat) | 11400 | 0.0413 | 0.0248 | 0.8854 | 6 | `lab_results_ferritin_value_last|med_ivabradine` (0.0229 -> 0.7774) |
| tvae_ind_seed0 | Spearman (num-num) | 1606 | 0.0895 | 0.071 | 0.6644 | 1 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.9914 -> 0.2692) |
| tvae_ind_seed0 | Cramer's V (cat-cat) | 8001 | 0.058 | 0.0338 | 0.8294 | 1 | `med_acei_history|med_arb_history` (0.0172 -> 0.5796) |
| tvae_ind_seed0 | corr-ratio (num-cat) | 11210 | 0.0459 | 0.0257 | 0.8649 | 7 | `lab_results_ferritin_value_last|med_potassium_binders_history` (0.041 -> 0.6632) |
| tvae_qt_seed0 | Spearman (num-num) | 1736 | 0.0883 | 0.0686 | 0.6613 | 1 | `lab_results_albuminBS_value_first|lab_results_albuminBS_value_last` (0.8455 -> 0.2033) |
| tvae_qt_seed0 | Cramer's V (cat-cat) | 8646 | 0.0581 | 0.0356 | 0.8284 | 1 | `med_acei_history|med_arb_history` (0.0172 -> 0.6253) |
| tvae_qt_seed0 | corr-ratio (num-cat) | 11590 | 0.0472 | 0.026 | 0.8516 | 4 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (0.3245 -> 1.0) |
| tvae_qt_seed1 | Spearman (num-num) | 1497 | 0.095 | 0.0733 | 0.6433 | 1 | `lab_results_albuminBS_value_first|lab_results_albuminBS_value_last` (0.8455 -> 0.1113) |
| tvae_qt_seed1 | Cramer's V (cat-cat) | 8256 | 0.065 | 0.0387 | 0.8058 | 1 | `med_acei_history|med_arb_history` (0.0172 -> 0.6051) |
| tvae_qt_seed1 | corr-ratio (num-cat) | 10640 | 0.0504 | 0.0293 | 0.8381 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5933 -> 0.0617) |
| tvae_qt_seed2 | Spearman (num-num) | 1688 | 0.0958 | 0.0733 | 0.6345 | 1 | `lab_results_albuminBS_value_first|lab_results_albuminBS_value_last` (0.8455 -> 0.1952) |
| tvae_qt_seed2 | Cramer's V (cat-cat) | 8515 | 0.0626 | 0.038 | 0.7958 | 1 | `med_acei_history|med_arb_history` (0.0172 -> 0.6312) |
| tvae_qt_seed2 | corr-ratio (num-cat) | 11590 | 0.0486 | 0.0286 | 0.8476 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5933 -> 0.1392) |
| tvae_seed0 | Spearman (num-num) | 1678 | 0.0808 | 0.0586 | 0.7122 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.9914 -> 0.3827) |
| tvae_seed0 | Cramer's V (cat-cat) | 8778 | 0.0558 | 0.0374 | 0.8438 | 1 | `med_acei_history|med_arb_history` (0.0172 -> 0.6135) |
| tvae_seed0 | corr-ratio (num-cat) | 11780 | 0.0436 | 0.0251 | 0.8731 | 5 | `lab_results_creatUS_value_last|encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (0.0667 -> 0.8792) |
| tvae_seed1 | Spearman (num-num) | 1742 | 0.0816 | 0.0618 | 0.7078 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.9914 -> 0.2763) |
| tvae_seed1 | Cramer's V (cat-cat) | 8515 | 0.0515 | 0.0332 | 0.8594 | 1 | `med_acei_history|med_arb_history` (0.0172 -> 0.5892) |
| tvae_seed1 | corr-ratio (num-cat) | 11780 | 0.0419 | 0.023 | 0.8855 | 10 | `lab_results_ferritin_value_last|med_antiinfl` (0.012 -> 0.6934) |
| tvae_seed2 | Spearman (num-num) | 1773 | 0.0766 | 0.0554 | 0.7428 | 0 | `vital_signs_oxygenSaturation_value_first|vital_signs_oxygenSaturation_value_last` (0.8168 -> -0.0213) |
| tvae_seed2 | Cramer's V (cat-cat) | 8515 | 0.0515 | 0.0312 | 0.8526 | 1 | `med_acei_history|med_arb_history` (0.0172 -> 0.5362) |
| tvae_seed2 | corr-ratio (num-cat) | 11970 | 0.0403 | 0.0234 | 0.8911 | 5 | `lab_results_ferritin_value_last|encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (0.0295 -> 0.5652) |

## original vs preprocessed

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.0103, W/std=0.066, mean 5.1539 -> 4.4212, missing 0% -> 1%
- `lab_results_sodium_value_last`: KS=0.0002, W/std=0.0064, mean 137.9002 -> 137.9325, missing 5% -> 5%
- `patient_demographics_age`: KS=0.0, W/std=0.0, mean 71.1062 -> 71.1062, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.0, W/std=0.0, mean 11.7102 -> 11.7102, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.0, W/std=0.0, mean 50.416 -> 50.416, missing 0% -> 0%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 11 -> 11 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `lab_results_cholTot_value_last`: KS=0.1466, W/std=0.3112, mean 3.7755 -> 4.1075, missing 95% -> 95%
- `lab_results_cholTot_value_first`: KS=0.1409, W/std=0.3035, mean 3.7887 -> 4.1075, missing 95% -> 95%
- `lab_results_ferritin_value_first`: KS=0.1192, W/std=0.0744, mean 587.3171 -> 627.9267, missing 85% -> 86%
- `lab_results_creatUS_value_first`: KS=0.1114, W/std=0.1329, mean 669.1336 -> 699.0473, missing 91% -> 91%
- `maggic_total_score`: KS=0.1058, W/std=0.1149, mean 23.2382 -> 23.8397, missing 80% -> 79%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.0491, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.0449, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.0397, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.0382, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.0373, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps10_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9829, W/std=3.3906, mean 15.4495 -> 52.5203, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8638, W/std=1.3264, mean 12.7232 -> 41.6892, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.8374, W/std=0.9351, mean 11.7451 -> 23.5884, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.7033, W/std=0.5766, mean 57.0956 -> 85.4914, missing 41% -> 42%
- `encounters_numOfPreviousHFStays_count`: KS=0.6452, W/std=0.5209, mean 49.716 -> 72.5622, missing 0% -> 0%
Worst categorical columns (by TVD):
- `med_diuretics_history`: TVD=0.0395, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0345, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0313, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0306, 5 -> 5 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.0222, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps15_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9829, W/std=3.3102, mean 15.4495 -> 51.4702, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8638, W/std=1.3116, mean 12.7232 -> 40.7709, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.8374, W/std=0.8805, mean 11.7451 -> 22.7243, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.7033, W/std=0.5619, mean 57.0956 -> 84.9358, missing 41% -> 42%
- `encounters_numOfPreviousHFStays_count`: KS=0.6452, W/std=0.5056, mean 49.716 -> 70.8821, missing 0% -> 0%
Worst categorical columns (by TVD):
- `med_diuretics_history`: TVD=0.0521, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0304, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0303, 11 -> 11 categories, missing 0% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.0284, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0271, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps1_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9829, W/std=4.7474, mean 15.4495 -> 67.5517, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8638, W/std=1.5543, mean 12.7232 -> 45.8368, missing 0% -> 1%
- `encounters_lengthOfStay`: KS=0.8374, W/std=1.2378, mean 11.7451 -> 27.7408, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.7033, W/std=0.689, mean 57.0956 -> 96.9511, missing 41% -> 41%
- `encounters_numOfPreviousHFStays_count`: KS=0.6452, W/std=0.5982, mean 49.716 -> 80.303, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0953, 5 -> 5 categories, missing 0% -> 0%
- `med_diuretics_history`: TVD=0.0636, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0462, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.03, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0282, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps20_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9829, W/std=3.3455, mean 15.4495 -> 51.9202, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8638, W/std=1.3159, mean 12.7232 -> 41.3084, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.8374, W/std=0.9187, mean 11.7451 -> 23.1323, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.7033, W/std=0.5509, mean 57.0956 -> 82.2003, missing 41% -> 41%
- `encounters_numOfPreviousHFStays_count`: KS=0.6452, W/std=0.5248, mean 49.716 -> 72.0822, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0504, 7 -> 7 categories, missing 0% -> 0%
- `med_diuretics_history`: TVD=0.05, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0442, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0284, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0278, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps8_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9829, W/std=3.3279, mean 15.4495 -> 51.6502, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8638, W/std=1.3408, mean 12.7232 -> 42.0996, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.8374, W/std=0.9488, mean 11.7451 -> 23.9004, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.7033, W/std=0.5586, mean 57.0956 -> 82.6895, missing 41% -> 41%
- `encounters_numOfPreviousHFStays_count`: KS=0.6452, W/std=0.5264, mean 49.716 -> 72.0222, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0371, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.027, 7 -> 7 categories, missing 0% -> 0%
- `med_mra_history`: TVD=0.027, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0264, 5 -> 5 categories, missing 0% -> 0%
- `med_ccb_history`: TVD=0.0208, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps1_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9829, W/std=4.239, mean 15.4495 -> 61.9712, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8638, W/std=1.3741, mean 12.7232 -> 42.9664, missing 0% -> 2%
- `encounters_lengthOfStay`: KS=0.8374, W/std=0.9616, mean 11.7451 -> 23.6364, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.7033, W/std=0.9746, mean 57.0956 -> 115.774, missing 41% -> 41%
- `encounters_numOfPreviousHFStays_count`: KS=0.6452, W/std=0.8388, mean 49.716 -> 94.6445, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0596, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0345, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0304, 5 -> 5 categories, missing 0% -> 0%
- `med_arb`: TVD=0.0271, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0266, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps5_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9829, W/std=3.5022, mean 15.4495 -> 53.7804, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8638, W/std=1.3157, mean 12.7232 -> 41.1928, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.8374, W/std=0.9264, mean 11.7451 -> 23.4083, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.7033, W/std=0.5889, mean 57.0956 -> 83.9209, missing 41% -> 42%
- `encounters_numOfPreviousHFStays_count`: KS=0.6452, W/std=0.5428, mean 49.716 -> 76.0726, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0577, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0381, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0353, 5 -> 5 categories, missing 0% -> 0%
- `med_mra_history`: TVD=0.0323, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0278, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[ctgan_qt_seed0]

Worst numeric columns (by KS):
- `lab_results_cholTot_value_first`: KS=0.7313, W/std=2.0099, mean 3.7887 -> 5.9344, missing 95% -> 96%
- `lab_results_triGly_value_first`: KS=0.7262, W/std=4.2881, mean 1.6003 -> 6.5473, missing 95% -> 94%
- `echocardiographs_lvef_pET_first`: KS=0.7067, W/std=1.9172, mean 41.6027 -> 75.601, missing 78% -> 81%
- `lab_results_creatUS_value_first`: KS=0.6006, W/std=2.2534, mean 669.1336 -> 1645.8295, missing 91% -> 97%
- `lab_results_albuminBS_value_first`: KS=0.4846, W/std=1.0635, mean 29.2957 -> 36.3683, missing 66% -> 87%
Worst categorical columns (by TVD):
- `med_vasodil`: TVD=0.3211, 2 -> 2 categories, missing 0% -> 0%
- `med_mra`: TVD=0.275, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.2543, 2 -> 2 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2389, 2 -> 2 categories, missing 0% -> 0%
- `med_rdoad`: TVD=0.232, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_first`: KS=0.8162, W/std=1.0271, mean 99.7885 -> 95.4588, missing 58% -> 80%
- `vital_signs_oxygenSaturation_value_last`: KS=0.6431, W/std=0.8233, mean 99.8586 -> 97.0208, missing 58% -> 76%
- `lab_results_ferritin_value_last`: KS=0.6099, W/std=0.5347, mean 551.2342 -> 1574.4257, missing 85% -> 94%
- `lab_results_triGly_value_first`: KS=0.5792, W/std=1.0929, mean 1.6003 -> 2.7512, missing 95% -> 96%
- `electrocardiographs_ecg_qrs_duration_pET_first`: KS=0.56, W/std=1.0996, mean 119.3898 -> 81.3473, missing 45% -> 59%
Worst categorical columns (by TVD):
- `med_activeDuringEncounter_bb`: TVD=0.3109, 2 -> 2 categories, missing 0% -> 0%
- `med_vasodil_history`: TVD=0.2776, 2 -> 2 categories, missing 0% -> 0%
- `med_sglt2i`: TVD=0.2767, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.2686, 2 -> 2 categories, missing 0% -> 0%
- `med_activeDuringEncounter_ace_inhibitors_arb_use`: TVD=0.2666, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed1]

Worst numeric columns (by KS):
- `lab_results_sodium_value_first`: KS=0.7284, W/std=1.94, mean 137.9966 -> 147.1237, missing 5% -> 2%
- `vital_signs_height_value_p1a_avg`: KS=0.723, W/std=2.1809, mean 171.1801 -> 193.6219, missing 24% -> 42%
- `lab_results_cholTot_value_first`: KS=0.7118, W/std=1.974, mean 3.7887 -> 5.8962, missing 95% -> 86%
- `vital_signs_oxygenSaturation_value_last`: KS=0.6569, W/std=0.8587, mean 99.8586 -> 96.8871, missing 58% -> 68%
- `vital_signs_oxygenSaturation_value_first`: KS=0.6184, W/std=0.5718, mean 99.7885 -> 97.5479, missing 58% -> 82%
Worst categorical columns (by TVD):
- `med_rasi_history`: TVD=0.2582, 2 -> 2 categories, missing 0% -> 0%
- `med_vasodil`: TVD=0.2527, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.2447, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag_history`: TVD=0.2155, 2 -> 2 categories, missing 0% -> 0%
- `med_platelet`: TVD=0.2116, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed2]

Worst numeric columns (by KS):
- `vital_signs_height_value_p1a_avg`: KS=0.7017, W/std=2.1994, mean 171.1801 -> 148.5567, missing 24% -> 24%
- `lab_results_sodium_value_last`: KS=0.6996, W/std=1.9745, mean 137.9785 -> 146.913, missing 5% -> 7%
- `echocardiographs_lvef_pET_first`: KS=0.6844, W/std=2.0918, mean 41.6027 -> 4.5087, missing 78% -> 78%
- `maggic_total_score`: KS=0.5741, W/std=1.5111, mean 23.2382 -> 34.506, missing 80% -> 78%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5293, W/std=0.3675, mean 12.7232 -> 21.583, missing 0% -> 21%
Worst categorical columns (by TVD):
- `med_anti_plat_history`: TVD=0.298, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.2533, 3 -> 3 categories, missing 0% -> 0%
- `conditions_revasc`: TVD=0.2359, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.2147, 11 -> 11 categories, missing 0% -> 0%
- `med_acei_history`: TVD=0.209, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ddpm_g_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=1.0, W/std=None, mean 104.9701 -> None, missing 93% -> 100%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=1.0, W/std=None, mean 125.4931 -> None, missing 78% -> 100%
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=1.0, W/std=None, mean 132.7354 -> None, missing 64% -> 100%
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first`: KS=1.0, W/std=None, mean 369.3873 -> None, missing 68% -> 100%
- `lab_results_triGly_value_first`: KS=0.8274, W/std=6.9041, mean 1.6003 -> 9.5663, missing 95% -> 92%
Worst categorical columns (by TVD):
- `med_diuretics_loop_history`: TVD=0.799, 2 -> 1 categories, missing 0% -> 0%
- `med_oral_antidiabetic_history`: TVD=0.7475, 2 -> 1 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.665, 2 -> 2 categories, missing 0% -> 0%
- `med_arb`: TVD=0.6514, 2 -> 2 categories, missing 0% -> 0%
- `med_oral_antidiabetic`: TVD=0.6407, 2 -> 1 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed0]

Worst numeric columns (by KS):
- `lab_results_triGly_value_last`: KS=0.8453, W/std=6.8952, mean 1.6019 -> 9.5009, missing 95% -> 94%
- `lab_results_triGly_value_first`: KS=0.8371, W/std=6.808, mean 1.6003 -> 9.4554, missing 95% -> 94%
- `lab_results_cholTot_value_last`: KS=0.815, W/std=2.2576, mean 3.7755 -> 6.2056, missing 95% -> 94%
- `lab_results_cholTot_value_first`: KS=0.7976, W/std=2.2653, mean 3.7887 -> 6.207, missing 95% -> 94%
- `lab_results_albuminBS_value_first`: KS=0.7702, W/std=2.1659, mean 29.2957 -> 43.7109, missing 66% -> 50%
Worst categorical columns (by TVD):
- `med_cortico_syst`: TVD=0.1957, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w5a_first`: TVD=0.1951, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.1864, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w5a_first`: TVD=0.1837, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.1831, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed1]

Worst numeric columns (by KS):
- `lab_results_cholTot_value_last`: KS=0.7991, W/std=2.224, mean 3.7755 -> 6.1694, missing 95% -> 95%
- `lab_results_cholTot_value_first`: KS=0.782, W/std=2.2165, mean 3.7887 -> 6.155, missing 95% -> 95%
- `lab_results_triGly_value_first`: KS=0.7685, W/std=5.4239, mean 1.6003 -> 7.8584, missing 95% -> 96%
- `lab_results_albuminBS_value_last`: KS=0.7668, W/std=2.1669, mean 28.7035 -> 43.4317, missing 66% -> 51%
- `lab_results_albuminBS_value_first`: KS=0.7647, W/std=2.1538, mean 29.2957 -> 43.6301, missing 66% -> 51%
Worst categorical columns (by TVD):
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.2185, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.2176, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.211, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`: TVD=0.2089, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.2041, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed2]

Worst numeric columns (by KS):
- `lab_results_cholTot_value_first`: KS=0.8245, W/std=2.3402, mean 3.7887 -> 6.2871, missing 95% -> 97%
- `lab_results_cholTot_value_last`: KS=0.8126, W/std=2.2583, mean 3.7755 -> 6.2063, missing 95% -> 97%
- `lab_results_triGly_value_first`: KS=0.8026, W/std=5.9264, mean 1.6003 -> 8.4383, missing 95% -> 97%
- `lab_results_triGly_value_last`: KS=0.7845, W/std=6.1367, mean 1.6019 -> 8.6319, missing 95% -> 97%
- `lab_results_ntProBnp_value_last`: KS=0.775, W/std=3.4023, mean 10128.1155 -> 59580.736, missing 62% -> 54%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.3124, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.2932, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1a_first`: TVD=0.2871, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.2656, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.2626, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps10_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=1.0, W/std=2.6632, mean 71.1071 -> 108.3052, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=1.0, W/std=0.824, mean 11.7451 -> 0.001, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 77.0434 -> None, missing 13% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=7.3132, mean 171.1801 -> 246.4336, missing 24% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 68.6276 -> None, missing 10% -> 100%
Worst categorical columns (by TVD):
- `conditions_hyperthyroid`: TVD=0.9712, 2 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9436, 11 -> 4 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first`: TVD=0.8689, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.8491, 3 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.8368, 5 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=1.0, W/std=2.7811, mean 71.1071 -> 109.9534, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=1.0, W/std=0.8239, mean 11.7451 -> 0.0022, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 77.0434 -> None, missing 13% -> 100%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.5134 -> None, missing 13% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=13.967, mean 77.0522 -> 349.7891, missing 18% -> 0%
Worst categorical columns (by TVD):
- `med_ivabradine`: TVD=0.9724, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`: TVD=0.9052, 3 -> 1 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w1mo_first`: TVD=0.9038, 3 -> 3 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8938, 11 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first`: TVD=0.8863, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed1]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=1.0, W/std=0.824, mean 11.7451 -> 0.0008, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 77.0434 -> None, missing 13% -> 100%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.5134 -> None, missing 13% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=7.6593, mean 171.1801 -> 249.9953, missing 24% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 77.0522 -> None, missing 18% -> 100%
Worst categorical columns (by TVD):
- `conditions_stroke`: TVD=0.8992, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.8371, 5 -> 3 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8119, 7 -> 5 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first`: TVD=0.8113, 3 -> 3 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.7966, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed2]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=1.0, W/std=0.8237, mean 11.7451 -> 0.0054, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.5134 -> None, missing 13% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 77.0522 -> None, missing 18% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 75.4776 -> None, missing 10% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 81.2546 -> None, missing 0% -> 100%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.9259, 5 -> 2 categories, missing 0% -> 0%
- `med_arni`: TVD=0.9118, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.8848, 3 -> 3 categories, missing 0% -> 0%
- `conditions_tia`: TVD=0.8725, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8527, 7 -> 1 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps1_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=1.0, W/std=2.777, mean 71.1071 -> 109.8955, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=1.0, W/std=0.824, mean 11.7451 -> 0.0019, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 77.0434 -> None, missing 13% -> 100%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.5134 -> None, missing 13% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=13.2788, mean 77.0522 -> 336.3505, missing 18% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.9649, 3 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.9295, 7 -> 3 categories, missing 0% -> 0%
- `med_antiinfl`: TVD=0.8638, 2 -> 2 categories, missing 0% -> 0%
- `med_antiinfl_history`: TVD=0.8593, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.7858, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps20_seed0]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=1.0, W/std=0.824, mean 11.7451 -> 0.0018, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 77.0434 -> None, missing 13% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 77.0522 -> None, missing 18% -> 100%
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 171.387 -> None, missing 46% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=10.18, mean 68.6276 -> 200.2382, missing 10% -> 0%
Worst categorical columns (by TVD):
- `conditions_ibd`: TVD=0.9526, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8563, 11 -> 6 categories, missing 0% -> 0%
- `conditions_substance_abuse`: TVD=0.8533, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8407, 7 -> 4 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w6mo_first`: TVD=0.8233, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps5_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=1.0, W/std=2.6386, mean 71.1071 -> 107.9617, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=1.0, W/std=0.824, mean 11.7451 -> 0.0008, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=13.4688, mean 79.5134 -> 347.7271, missing 13% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=13.8365, mean 77.0522 -> 347.2403, missing 18% -> 0%
- `vital_signs_height_value_last`: KS=1.0, W/std=6.7999, mean 171.387 -> 242.216, missing 46% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.9157, 3 -> 3 categories, missing 0% -> 0%
- `conditions_ld`: TVD=0.9079, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8986, 11 -> 2 categories, missing 0% -> 0%
- `med_sglt2i_history`: TVD=0.898, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.8314, 5 -> 4 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps8_seed0]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=1.0, W/std=0.824, mean 11.7451 -> 0.0007, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=11.8459, mean 77.0434 -> 307.7041, missing 13% -> 0%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.5134 -> None, missing 13% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=7.6595, mean 171.1801 -> 249.9979, missing 24% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 77.0522 -> None, missing 18% -> 100%
Worst categorical columns (by TVD):
- `conditions_dep`: TVD=0.9397, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.9367, 5 -> 2 categories, missing 0% -> 0%
- `med_rdoad_syst_history`: TVD=0.9193, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9121, 11 -> 4 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.8401, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_first`: KS=1.0, W/std=None, mean 587.3171 -> None, missing 85% -> 100%
- `lab_results_ferritin_value_last`: KS=1.0, W/std=None, mean 551.2342 -> None, missing 85% -> 100%
- `lab_results_triGly_value_first`: KS=1.0, W/std=None, mean 1.6003 -> None, missing 95% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=None, mean 1.6019 -> None, missing 95% -> 100%
- `lab_results_cholTot_value_first`: KS=1.0, W/std=None, mean 3.7887 -> None, missing 95% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2245, 11 -> 10 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w3a_first`: TVD=0.0291, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0283, 7 -> 7 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w3a_first`: TVD=0.0278, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.0275, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed1]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_first`: KS=1.0, W/std=None, mean 587.3171 -> None, missing 85% -> 100%
- `lab_results_ferritin_value_last`: KS=1.0, W/std=None, mean 551.2342 -> None, missing 85% -> 100%
- `lab_results_triGly_value_first`: KS=1.0, W/std=None, mean 1.6003 -> None, missing 95% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=None, mean 1.6019 -> None, missing 95% -> 100%
- `lab_results_cholTot_value_first`: KS=1.0, W/std=None, mean 3.7887 -> None, missing 95% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2278, 11 -> 10 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.0337, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0293, 7 -> 7 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.0262, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.0248, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed2]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=None, mean 551.2342 -> None, missing 85% -> 100%
- `lab_results_triGly_value_first`: KS=1.0, W/std=None, mean 1.6003 -> None, missing 95% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=None, mean 1.6019 -> None, missing 95% -> 100%
- `lab_results_cholTot_value_first`: KS=1.0, W/std=None, mean 3.7887 -> None, missing 95% -> 100%
- `lab_results_cholTot_value_last`: KS=1.0, W/std=None, mean 3.7755 -> None, missing 95% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2197, 11 -> 10 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first`: TVD=0.0322, 3 -> 3 categories, missing 0% -> 0%
- `med_ll_history`: TVD=0.0299, 2 -> 2 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.0284, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0263, 7 -> 7 categories, missing 0% -> 0%

## train vs synthetic[mst_eps0p5_seed0]

Worst numeric columns (by KS):
- `lab_results_triGly_value_first`: KS=1.0, W/std=17.2057, mean 1.6003 -> 21.4523, missing 95% -> 96%
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=2.6611, mean 99.8586 -> 89.8754, missing 58% -> 58%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=2.5619, mean 99.7885 -> 88.0648, missing 58% -> 56%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=3.826, mean 551.2342 -> 8750.6003, missing 85% -> 81%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=6.1273, mean 0.0639 -> 4.5327, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.1446, 7 -> 7 categories, missing 0% -> 0%
- `med_thrombolytic_history`: TVD=0.1443, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1377, 11 -> 11 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.1109, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.1004, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps10_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=1.7331, mean 99.8586 -> 93.4216, missing 58% -> 58%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=1.4022, mean 99.7885 -> 93.613, missing 58% -> 57%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=2.9859, mean 551.2342 -> 6883.1609, missing 85% -> 85%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=3.8689, mean 0.0639 -> 2.8855, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.988, W/std=2.9863, mean 587.3171 -> 7482.2695, missing 85% -> 84%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0393, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.03, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0253, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0238, 5 -> 5 categories, missing 0% -> 0%
- `med_ll`: TVD=0.022, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=1.6989, mean 99.8586 -> 93.5974, missing 58% -> 58%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=1.3936, mean 99.7885 -> 93.5999, missing 58% -> 58%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=2.7528, mean 551.2342 -> 6250.0, missing 85% -> 85%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=3.5009, mean 0.0639 -> 2.6155, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.988, W/std=2.5659, mean 587.3171 -> 6250.0, missing 85% -> 85%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0329, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0308, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0274, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0265, 5 -> 5 categories, missing 0% -> 0%
- `med_activeDuringEncounter_ace_inhibitors_arb_use`: TVD=0.0223, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed1]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=1.7348, mean 99.8586 -> 93.4745, missing 58% -> 58%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=1.4025, mean 99.7885 -> 93.5467, missing 58% -> 58%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=2.76, mean 551.2342 -> 6301.1247, missing 85% -> 85%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=3.4711, mean 0.0639 -> 2.593, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.988, W/std=2.6965, mean 587.3171 -> 6726.0785, missing 85% -> 85%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0329, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0293, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.026, 5 -> 5 categories, missing 0% -> 0%
- `med_platelet_history`: TVD=0.0252, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_plat_history`: TVD=0.0251, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed2]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=1.7149, mean 99.8586 -> 93.4548, missing 58% -> 58%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=1.3974, mean 99.7885 -> 93.7234, missing 58% -> 58%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=2.7915, mean 551.2342 -> 6434.2695, missing 85% -> 86%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=3.4507, mean 0.0639 -> 2.5495, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.988, W/std=2.5659, mean 587.3171 -> 6250.0, missing 85% -> 86%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0325, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0291, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.0218, 3 -> 3 categories, missing 0% -> 0%
- `med_activeDuringEncounter_ace_inhibitors_arb_use`: TVD=0.0214, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.0211, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps1_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=2.383, mean 99.8586 -> 90.9205, missing 58% -> 57%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=1.3993, mean 99.7885 -> 93.7323, missing 58% -> 57%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=18.0584, mean 551.2342 -> 39250.8873, missing 85% -> 87%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=5.7324, mean 0.0639 -> 4.2447, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.988, W/std=2.5659, mean 587.3171 -> 6250.0, missing 85% -> 85%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.1109, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.101, 11 -> 11 categories, missing 0% -> 0%
- `med_bb`: TVD=0.0562, 2 -> 2 categories, missing 0% -> 0%
- `med_mra`: TVD=0.0545, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ap`: TVD=0.054, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps20_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=1.6959, mean 99.8586 -> 93.75, missing 58% -> 58%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=1.3962, mean 99.7885 -> 93.6613, missing 58% -> 58%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=2.9001, mean 551.2342 -> 6698.3122, missing 85% -> 86%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=3.5773, mean 0.0639 -> 2.6725, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.988, W/std=2.7733, mean 587.3171 -> 6978.8496, missing 85% -> 86%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0347, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0272, 7 -> 7 categories, missing 0% -> 0%
- `med_activeDuringEncounter_ace_inhibitors_arb_use`: TVD=0.0208, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.0205, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.0204, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps5_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=1.6959, mean 99.8586 -> 93.75, missing 58% -> 58%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=1.4032, mean 99.7885 -> 93.75, missing 58% -> 58%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=3.302, mean 551.2342 -> 7627.5218, missing 85% -> 85%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=4.2905, mean 0.0639 -> 3.1931, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.988, W/std=2.8435, mean 587.3171 -> 7141.5376, missing 85% -> 84%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0301, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0286, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0266, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0253, 5 -> 5 categories, missing 0% -> 0%
- `med_anti_plat`: TVD=0.022, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps8_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=1.7347, mean 99.8586 -> 93.4158, missing 58% -> 57%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=1.4085, mean 99.7885 -> 93.4611, missing 58% -> 57%
- `lab_results_ferritin_value_last`: KS=0.992, W/std=3.0187, mean 551.2342 -> 7020.4201, missing 85% -> 86%
- `smoking_status_smoker_startTime_count`: KS=0.9916, W/std=3.5493, mean 0.0639 -> 2.6485, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.988, W/std=2.9097, mean 587.3171 -> 7305.5679, missing 85% -> 85%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0381, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0334, 11 -> 11 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0268, 5 -> 5 categories, missing 0% -> 0%
- `med_platelet_history`: TVD=0.0258, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_plat_history`: TVD=0.0245, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps15_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9856, W/std=1.353, mean 0.0639 -> 1.0507, missing 0% -> 0%
- `vital_signs_oxygenSaturation_value_last`: KS=0.9751, W/std=4.8191, mean 99.8586 -> 81.7482, missing 58% -> 55%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9741, W/std=4.005, mean 99.7885 -> 81.4218, missing 58% -> 54%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9316, W/std=0.2524, mean 0.2151 -> 0.4829, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8542, W/std=0.2397, mean 0.3557 -> 0.265, missing 0% -> 0%
Worst categorical columns (by TVD):
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.3296, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w5a_first`: TVD=0.329, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w5a_first`: TVD=0.3275, 3 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w5a_first`: TVD=0.3275, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w3a_first`: TVD=0.2999, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps1_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9986, W/std=16.8718, mean 99.8586 -> 36.4508, missing 58% -> 1%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9979, W/std=15.88, mean 99.7885 -> 26.9644, missing 58% -> 6%
- `smoking_status_smoker_startTime_count`: KS=0.9856, W/std=4.1904, mean 0.0639 -> 3.1172, missing 0% -> 0%
- `lab_results_cholTot_value_first`: KS=0.9505, W/std=1.8696, mean 3.7887 -> 1.7939, missing 95% -> 100%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9316, W/std=2.9502, mean 0.2151 -> 3.3663, missing 0% -> 0%
Worst categorical columns (by TVD):
- `cause_of_death_isRenal_f5a_w5a_first`: TVD=0.3268, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w3a_first`: TVD=0.2764, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w1a_first`: TVD=0.2728, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w5a_first`: TVD=0.2197, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w1a_first`: TVD=0.1886, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps5_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=1.0, W/std=None, mean 104.9701 -> None, missing 93% -> 100%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=1.0, W/std=None, mean 125.4931 -> None, missing 78% -> 100%
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first`: KS=1.0, W/std=None, mean 369.3873 -> None, missing 68% -> 100%
- `vital_signs_oxygenSaturation_value_last`: KS=0.9952, W/std=6.6689, mean 99.8586 -> 74.7965, missing 58% -> 47%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9934, W/std=5.2753, mean 99.7885 -> 75.5967, missing 58% -> 46%
Worst categorical columns (by TVD):
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.6443, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.6434, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`: TVD=0.6296, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.6236, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.6227, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[tvae_cap256_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.4636, W/std=0.2301, mean 12.7232 -> 17.4766, missing 0% -> 31%
- `lab_results_creatUS_value_first`: KS=0.4191, W/std=0.5874, mean 669.1336 -> 508.3425, missing 91% -> 94%
- `lab_results_ferritin_value_last`: KS=0.383, W/std=0.1488, mean 551.2342 -> 421.1105, missing 85% -> 93%
- `lab_results_ferritin_value_first`: KS=0.3723, W/std=0.1392, mean 587.3171 -> 509.0041, missing 85% -> 93%
- `lab_results_creatUS_value_last`: KS=0.3491, W/std=0.4992, mean 635.7344 -> 515.0983, missing 91% -> 94%
Worst categorical columns (by TVD):
- `conditions_copd`: TVD=0.1674, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.1572, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1507, 11 -> 11 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1487, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.1443, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ep1000_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.475, W/std=0.3181, mean 12.7232 -> 20.33, missing 0% -> 28%
- `lab_results_ferritin_value_last`: KS=0.3901, W/std=0.1421, mean 551.2342 -> 469.2939, missing 85% -> 90%
- `lab_results_ferritin_value_first`: KS=0.3772, W/std=0.1478, mean 587.3171 -> 471.4104, missing 85% -> 90%
- `lab_results_creatUS_value_first`: KS=0.3295, W/std=0.5439, mean 669.1336 -> 534.5592, missing 91% -> 94%
- `lab_results_creatUS_value_last`: KS=0.3266, W/std=0.5334, mean 635.7344 -> 492.1407, missing 91% -> 95%
Worst categorical columns (by TVD):
- `conditions_mc`: TVD=0.1629, 2 -> 2 categories, missing 0% -> 0%
- `med_activeDuringEncounter_bb`: TVD=0.158, 2 -> 2 categories, missing 0% -> 0%
- `med_bb`: TVD=0.1544, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1515, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.1502, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[tvae_ind_seed0]

Worst numeric columns (by KS):
- `lab_results_crpNonHs_value_first`: KS=0.4487, W/std=0.5842, mean 77.0746 -> 27.8999, missing 41% -> 33%
- `lab_results_crpNonHs_value_last`: KS=0.3836, W/std=0.4792, mean 57.0956 -> 24.9552, missing 41% -> 33%
- `lab_results_cholTot_value_last`: KS=0.3675, W/std=0.4736, mean 3.7755 -> 3.4178, missing 95% -> 95%
- `lab_results_cholTot_value_first`: KS=0.3512, W/std=0.4375, mean 3.7887 -> 3.4136, missing 95% -> 95%
- `lab_results_ntProBnp_value_last`: KS=0.3284, W/std=0.3484, mean 10128.1155 -> 5089.3784, missing 62% -> 55%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.254, 11 -> 11 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1934, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.1911, 2 -> 2 categories, missing 0% -> 0%
- `med_activeDuringEncounter_bb`: TVD=0.179, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1764, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=0.4055, W/std=0.1752, mean 551.2342 -> 422.305, missing 85% -> 83%
- `lab_results_ferritin_value_first`: KS=0.3678, W/std=0.182, mean 587.3171 -> 410.3336, missing 85% -> 83%
- `lab_results_creatUS_value_first`: KS=0.3673, W/std=0.4907, mean 669.1336 -> 659.422, missing 91% -> 90%
- `lab_results_triGly_value_first`: KS=0.3571, W/std=0.3321, mean 1.6003 -> 1.5032, missing 95% -> 92%
- `lab_results_triGly_value_last`: KS=0.3511, W/std=0.336, mean 1.6019 -> 1.5291, missing 95% -> 92%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1909, 11 -> 11 categories, missing 0% -> 0%
- `med_rdoad`: TVD=0.1793, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.1773, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.168, 2 -> 2 categories, missing 0% -> 0%
- `med_ccb`: TVD=0.1541, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed1]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_first`: KS=0.4096, W/std=0.1532, mean 587.3171 -> 444.9525, missing 85% -> 95%
- `lab_results_ferritin_value_last`: KS=0.4006, W/std=0.1487, mean 551.2342 -> 474.2784, missing 85% -> 95%
- `lab_results_hemoglobin_value_last`: KS=0.3803, W/std=0.6268, mean 115.8605 -> 101.5122, missing 28% -> 34%
- `lab_results_creatUS_value_first`: KS=0.3588, W/std=0.5278, mean 669.1336 -> 645.4219, missing 91% -> 93%
- `lab_results_creatUS_value_last`: KS=0.344, W/std=0.5056, mean 635.7344 -> 610.8827, missing 91% -> 93%
Worst categorical columns (by TVD):
- `conditions_mc`: TVD=0.1959, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1877, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1698, 2 -> 2 categories, missing 0% -> 0%
- `med_inotropes_history`: TVD=0.1582, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.1557, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed2]

Worst numeric columns (by KS):
- `lab_results_triGly_value_first`: KS=0.4158, W/std=0.3322, mean 1.6003 -> 1.6572, missing 95% -> 95%
- `lab_results_triGly_value_last`: KS=0.4023, W/std=0.3489, mean 1.6019 -> 1.723, missing 95% -> 95%
- `lab_results_cholTot_value_first`: KS=0.3865, W/std=0.5572, mean 3.7887 -> 3.8432, missing 95% -> 95%
- `lab_results_cholTot_value_last`: KS=0.3838, W/std=0.5477, mean 3.7755 -> 3.8968, missing 95% -> 95%
- `lab_results_creatUS_value_last`: KS=0.3552, W/std=0.5399, mean 635.7344 -> 627.4834, missing 91% -> 89%
Worst categorical columns (by TVD):
- `conditions_mc`: TVD=0.1887, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1754, 11 -> 11 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1712, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1704, 2 -> 1 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.1511, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.4925, W/std=0.3222, mean 12.7232 -> 20.4508, missing 0% -> 26%
- `lab_results_creatUS_value_last`: KS=0.4386, W/std=0.6107, mean 635.7344 -> 453.6096, missing 91% -> 95%
- `lab_results_creatUS_value_first`: KS=0.4218, W/std=0.6157, mean 669.1336 -> 495.0372, missing 91% -> 95%
- `lab_results_ferritin_value_first`: KS=0.3824, W/std=0.1456, mean 587.3171 -> 456.8562, missing 85% -> 91%
- `lab_results_ferritin_value_last`: KS=0.3585, W/std=0.1331, mean 551.2342 -> 482.6467, missing 85% -> 91%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.19, 11 -> 11 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1799, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1656, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.1599, 2 -> 2 categories, missing 0% -> 0%
- `med_activeDuringEncounter_bb`: TVD=0.1427, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed1]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.4678, W/std=0.2357, mean 12.7232 -> 18.2324, missing 0% -> 28%
- `lab_results_ferritin_value_last`: KS=0.4006, W/std=0.149, mean 551.2342 -> 590.3529, missing 85% -> 93%
- `lab_results_creatUS_value_first`: KS=0.3993, W/std=0.5901, mean 669.1336 -> 506.6273, missing 91% -> 93%
- `lab_results_ferritin_value_first`: KS=0.3872, W/std=0.1697, mean 587.3171 -> 695.0006, missing 85% -> 93%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.3435, W/std=0.1058, mean 4.2903 -> 4.5946, missing 1% -> 19%
Worst categorical columns (by TVD):
- `conditions_mc`: TVD=0.171, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1698, 2 -> 1 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.167, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1632, 2 -> 2 categories, missing 0% -> 0%
- `med_vasodil`: TVD=0.1586, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed2]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.4843, W/std=0.1871, mean 12.7232 -> 16.7745, missing 0% -> 25%
- `lab_results_ferritin_value_first`: KS=0.4225, W/std=0.1482, mean 587.3171 -> 733.9595, missing 85% -> 88%
- `lab_results_ferritin_value_last`: KS=0.415, W/std=0.147, mean 551.2342 -> 550.2365, missing 85% -> 88%
- `lab_results_creatUS_value_first`: KS=0.3967, W/std=0.5657, mean 669.1336 -> 505.4661, missing 91% -> 94%
- `lab_results_crpNonHs_value_last`: KS=0.3659, W/std=0.4649, mean 57.0956 -> 25.9808, missing 41% -> 48%
Worst categorical columns (by TVD):
- `conditions_mc`: TVD=0.1827, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1695, 2 -> 2 categories, missing 0% -> 0%
- `med_vasodil`: TVD=0.1547, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1508, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1482, 2 -> 2 categories, missing 0% -> 0%
