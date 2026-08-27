# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 38 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 164 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| train vs holdout | 211 | 0.0486 | 0.0405 | 0.9344 | 0.0782 | 0.01 | 1.0 | 0.0075 |
| train vs synthetic[aim40_eps1_seed0] | 40 | 0.3379 | 0.1992 | 0.0 | 0.314 | 0.0186 | 0.9167 | 0.0065 |
| train vs synthetic[aim50_eps1_seed0] | 50 | 0.3255 | 0.2574 | 0.0 | 0.3324 | 0.0184 | 0.9655 | 0.0121 |
| train vs synthetic[ctgan_qt_seed0] | 211 | 0.286 | 0.2687 | 0.082 | 0.7985 | 0.0749 | 0.4267 | 0.1214 |
| train vs synthetic[ctgan_seed0] | 211 | 0.3036 | 0.2925 | 0.0656 | 0.5594 | 0.1071 | 0.3 | 0.0782 |
| train vs synthetic[ctgan_seed1] | 211 | 0.3371 | 0.3454 | 0.0492 | 0.6657 | 0.0803 | 0.4267 | 0.0661 |
| train vs synthetic[ctgan_seed2] | 211 | 0.3106 | 0.2768 | 0.082 | 0.5942 | 0.0744 | 0.4067 | 0.0916 |
| train vs synthetic[ddpm_g_seed0] | 211 | 0.9648 | 0.9987 | 0.0 | 6.5255 | 0.3092 | 0.04 | 0.322 |
| train vs synthetic[ddpm_seed0] | 211 | 0.9646 | 0.9986 | 0.0 | 6.5244 | 0.3354 | 0.0333 | 0.3223 |
| train vs synthetic[ddpm_seed1] | 211 | 0.9642 | 0.9986 | 0.0 | 6.5358 | 0.3436 | 0.04 | 0.2934 |
| train vs synthetic[ddpm_seed2] | 211 | 0.9649 | 0.9985 | 0.0 | 6.5411 | 0.3458 | 0.04 | 0.2817 |
| train vs synthetic[dpctgan_eps10_seed0] | 211 | 0.9485 | 1.0 | 0.0 | 2.8405 | 0.292 | 0.1667 | 0.3684 |
| train vs synthetic[dpctgan_eps15_seed0] | 211 | 0.9811 | 1.0 | 0.0 | 4.1343 | 0.3208 | 0.1467 | 0.4046 |
| train vs synthetic[dpctgan_eps15_seed1] | 211 | 0.9257 | 1.0 | 0.0 | 3.3381 | 0.3346 | 0.1667 | 0.3404 |
| train vs synthetic[dpctgan_eps15_seed2] | 211 | 0.8968 | 1.0 | 0.0 | 2.3056 | 0.3089 | 0.14 | 0.31 |
| train vs synthetic[dpctgan_eps1_seed0] | 211 | 0.9346 | 1.0 | 0.0 | 2.6987 | 0.3331 | 0.1533 | 0.354 |
| train vs synthetic[dpctgan_eps20_seed0] | 211 | 0.9194 | 1.0 | 0.0 | 2.7532 | 0.3438 | 0.1467 | 0.3426 |
| train vs synthetic[dpctgan_eps5_seed0] | 211 | 0.8865 | 1.0 | 0.0164 | 2.3039 | 0.3028 | 0.16 | 0.3582 |
| train vs synthetic[dpctgan_eps8_seed0] | 211 | 0.9609 | 1.0 | 0.0 | 3.7219 | 0.3271 | 0.1467 | 0.3621 |
| train vs synthetic[gaussian_copula_seed0] | 211 | 0.5221 | 0.4915 | 0.0656 | 1.3377 | 0.0134 | 0.9933 | 0.1234 |
| train vs synthetic[gaussian_copula_seed1] | 211 | 0.5189 | 0.4812 | 0.0656 | 1.3481 | 0.0126 | 0.9933 | 0.1238 |
| train vs synthetic[gaussian_copula_seed2] | 211 | 0.5233 | 0.4795 | 0.0656 | 1.365 | 0.013 | 0.9933 | 0.1226 |
| train vs synthetic[mst_eps0p5_seed0] | 211 | 0.5076 | 0.4463 | 0.0164 | 1.3213 | 0.0493 | 0.6067 | 0.0437 |
| train vs synthetic[mst_eps10_seed0] | 211 | 0.418 | 0.2981 | 0.0164 | 0.4041 | 0.0095 | 1.0 | 0.0057 |
| train vs synthetic[mst_eps15_seed0] | 211 | 0.4137 | 0.2896 | 0.0164 | 0.3933 | 0.0088 | 1.0 | 0.0049 |
| train vs synthetic[mst_eps15_seed1] | 211 | 0.4119 | 0.2895 | 0.0164 | 0.3823 | 0.0091 | 1.0 | 0.0044 |
| train vs synthetic[mst_eps15_seed2] | 211 | 0.4124 | 0.2921 | 0.0164 | 0.3873 | 0.0095 | 1.0 | 0.0042 |
| train vs synthetic[mst_eps1_seed0] | 211 | 0.4667 | 0.3649 | 0.0 | 0.8886 | 0.0249 | 0.92 | 0.0332 |
| train vs synthetic[mst_eps20_seed0] | 211 | 0.4126 | 0.2845 | 0.0164 | 0.3802 | 0.0089 | 1.0 | 0.0031 |
| train vs synthetic[mst_eps5_seed0] | 211 | 0.4323 | 0.3043 | 0.0164 | 0.4631 | 0.0095 | 1.0 | 0.012 |
| train vs synthetic[mst_eps8_seed0] | 211 | 0.4186 | 0.2992 | 0.0164 | 0.4231 | 0.0087 | 1.0 | 0.0053 |
| train vs synthetic[patectgan_eps15_seed0] | 211 | 0.3754 | 0.293 | 0.0 | 1.4076 | 0.0778 | 0.56 | 0.091 |
| train vs synthetic[patectgan_eps1_seed0] | 211 | 0.5717 | 0.4986 | 0.0328 | 1.4448 | 0.0741 | 0.38 | 0.1357 |
| train vs synthetic[patectgan_eps5_seed0] | 211 | 0.4156 | 0.3171 | 0.0 | 1.2696 | 0.15 | 0.4467 | 0.1052 |
| train vs synthetic[tvae_cap256_seed0] | 211 | 0.1942 | 0.1698 | 0.2295 | 0.2532 | 0.0568 | 0.5267 | 0.0359 |
| train vs synthetic[tvae_ep1000_seed0] | 211 | 0.1942 | 0.1679 | 0.1803 | 0.2531 | 0.0557 | 0.5667 | 0.037 |
| train vs synthetic[tvae_ind_seed0] | 211 | 0.2084 | 0.2087 | 0.1639 | 0.254 | 0.0609 | 0.4667 | 0.0301 |
| train vs synthetic[tvae_qt_seed0] | 211 | 0.188 | 0.1492 | 0.1967 | 0.2592 | 0.0536 | 0.6133 | 0.0376 |
| train vs synthetic[tvae_qt_seed1] | 211 | 0.1894 | 0.1953 | 0.2787 | 0.2674 | 0.0676 | 0.5133 | 0.0345 |
| train vs synthetic[tvae_qt_seed2] | 211 | 0.1748 | 0.1526 | 0.2459 | 0.2532 | 0.058 | 0.6067 | 0.0379 |
| train vs synthetic[tvae_seed0] | 211 | 0.2044 | 0.1747 | 0.1311 | 0.2734 | 0.0557 | 0.6067 | 0.0396 |
| train vs synthetic[tvae_seed1] | 211 | 0.2105 | 0.1993 | 0.2295 | 0.2685 | 0.0602 | 0.4867 | 0.037 |
| train vs synthetic[tvae_seed2] | 211 | 0.2111 | 0.1746 | 0.1967 | 0.2727 | 0.0622 | 0.5533 | 0.0351 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| aim | 1 | 1 | 0.3255 | 0.0184 | 0.0121 |
| aim40 | 1 | 1 | 0.3379 | 0.0186 | 0.0065 |
| ctgan | - | 3 | 0.3171 ± 0.0177 | 0.0873 ± 0.0174 | 0.0786 ± 0.0128 |
| ctgan_qt | - | 1 | 0.286 | 0.0749 | 0.1214 |
| ddpm | - | 3 | 0.9646 ± 0.0004 | 0.3416 ± 0.0055 | 0.2991 ± 0.0209 |
| ddpm_g | - | 1 | 0.9648 | 0.3092 | 0.322 |
| dpctgan | 1 | 1 | 0.9346 | 0.3331 | 0.354 |
| dpctgan | 5 | 1 | 0.8865 | 0.3028 | 0.3582 |
| dpctgan | 8 | 1 | 0.9609 | 0.3271 | 0.3621 |
| dpctgan | 10 | 1 | 0.9485 | 0.292 | 0.3684 |
| dpctgan | 15 | 3 | 0.9345 ± 0.0428 | 0.3214 ± 0.0129 | 0.3517 ± 0.0483 |
| dpctgan | 20 | 1 | 0.9194 | 0.3438 | 0.3426 |
| gaussian_copula | - | 3 | 0.5214 ± 0.0023 | 0.013 ± 0.0004 | 0.1233 ± 0.0006 |
| mst | 0.5 | 1 | 0.5076 | 0.0493 | 0.0437 |
| mst | 1 | 1 | 0.4667 | 0.0249 | 0.0332 |
| mst | 5 | 1 | 0.4323 | 0.0095 | 0.012 |
| mst | 8 | 1 | 0.4186 | 0.0087 | 0.0053 |
| mst | 10 | 1 | 0.418 | 0.0095 | 0.0057 |
| mst | 15 | 3 | 0.4127 ± 0.0009 | 0.0091 ± 0.0004 | 0.0045 ± 0.0004 |
| mst | 20 | 1 | 0.4126 | 0.0089 | 0.0031 |
| patectgan | 1 | 1 | 0.5717 | 0.0741 | 0.1357 |
| patectgan | 5 | 1 | 0.4156 | 0.15 | 0.1052 |
| patectgan | 15 | 1 | 0.3754 | 0.0778 | 0.091 |
| tvae | - | 3 | 0.2087 ± 0.0037 | 0.0594 ± 0.0033 | 0.0372 ± 0.0023 |
| tvae_cap256 | - | 1 | 0.1942 | 0.0568 | 0.0359 |
| tvae_ep1000 | - | 1 | 0.1942 | 0.0557 | 0.037 |
| tvae_ind | - | 1 | 0.2084 | 0.0609 | 0.0301 |
| tvae_qt | - | 3 | 0.1841 ± 0.0081 | 0.0597 ± 0.0072 | 0.0367 ± 0.0019 |

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.4924**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| aim40_eps1_seed0 | 1.0 | 0.0 | 0.1896 |
| aim50_eps1_seed0 | 1.0 | 0.0 | 0.237 |
| ctgan_qt_seed0 | 1.0 | 0.0 | 1.0 |
| ctgan_seed0 | 1.0 | 0.0 | 1.0 |
| ctgan_seed1 | 0.9999 | 0.0001 | 1.0 |
| ctgan_seed2 | 0.9999 | 0.0001 | 1.0 |
| ddpm_g_seed0 | 1.0 | 0.0 | 1.0 |
| ddpm_seed0 | 1.0 | 0.0 | 1.0 |
| ddpm_seed1 | 0.9995 | 0.0001 | 1.0 |
| ddpm_seed2 | 0.9999 | 0.0 | 1.0 |
| dpctgan_eps10_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps15_seed0 | 0.9996 | 0.0004 | 1.0 |
| dpctgan_eps15_seed1 | 0.9999 | 0.0003 | 1.0 |
| dpctgan_eps15_seed2 | 0.9998 | 0.0006 | 1.0 |
| dpctgan_eps1_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps20_seed0 | 0.9998 | 0.0006 | 1.0 |
| dpctgan_eps5_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps8_seed0 | 0.9999 | 0.0003 | 1.0 |
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
| patectgan_eps15_seed0 | 0.9992 | 0.0027 | 1.0 |
| patectgan_eps1_seed0 | 1.0 | 0.0 | 1.0 |
| patectgan_eps5_seed0 | 1.0 | 0.0 | 1.0 |
| tvae_cap256_seed0 | 0.9976 | 0.0011 | 1.0 |
| tvae_ep1000_seed0 | 0.9971 | 0.0025 | 1.0 |
| tvae_ind_seed0 | 0.9987 | 0.0009 | 1.0 |
| tvae_qt_seed0 | 0.9987 | 0.0005 | 1.0 |
| tvae_qt_seed1 | 0.9996 | 0.0003 | 1.0 |
| tvae_qt_seed2 | 0.9991 | 0.0003 | 1.0 |
| tvae_seed0 | 0.9982 | 0.0006 | 1.0 |
| tvae_seed1 | 0.9984 | 0.0008 | 1.0 |
| tvae_seed2 | 0.9979 | 0.0022 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.0745 | 0.0684 | 0.0997 | 0.0754 | 0.0961 |
| aim40_eps1_seed0 | 0.3641 | 0.3516 | 0.421 | 0.349 | 0.3834 |
| aim50_eps1_seed0 | 0.345 | 0.34 | 0.3952 | 0.339 | 0.3762 |
| ctgan_qt_seed0 | 0.3129 | 0.2976 | 0.3186 | 0.305 | 0.3198 |
| ctgan_seed0 | 0.3366 | 0.3053 | 0.3321 | 0.3153 | 0.3498 |
| ctgan_seed1 | 0.3571 | 0.3414 | 0.358 | 0.3529 | 0.3645 |
| ctgan_seed2 | 0.3219 | 0.3293 | 0.3283 | 0.3248 | 0.339 |
| ddpm_g_seed0 | 0.9657 | 0.9641 | 0.9657 | - | 0.975 |
| ddpm_seed0 | 0.9645 | 0.9642 | 0.9659 | - | 0.9752 |
| ddpm_seed1 | 0.9644 | 0.9637 | 0.9636 | - | 0.9755 |
| ddpm_seed2 | 0.963 | 0.9658 | 0.9657 | - | 0.9756 |
| dpctgan_eps10_seed0 | - | 0.9501 | 0.9534 | 0.9461 | 0.9428 |
| dpctgan_eps15_seed0 | 0.9792 | 0.9831 | - | - | 0.983 |
| dpctgan_eps15_seed1 | 0.9252 | - | - | 0.9254 | 0.9246 |
| dpctgan_eps15_seed2 | 0.905 | 0.8943 | - | - | 0.9054 |
| dpctgan_eps1_seed0 | 0.9421 | 0.9357 | - | 0.9266 | 0.9404 |
| dpctgan_eps20_seed0 | - | 0.9212 | - | - | 0.9214 |
| dpctgan_eps5_seed0 | - | 0.8866 | 0.8957 | 0.887 | 0.8951 |
| dpctgan_eps8_seed0 | - | 0.96 | - | - | 0.9693 |
| gaussian_copula_seed0 | 0.5372 | 0.5354 | 0.547 | 0.5295 | 0.5451 |
| gaussian_copula_seed1 | 0.5346 | 0.53 | 0.548 | 0.5314 | 0.5355 |
| gaussian_copula_seed2 | 0.542 | 0.521 | 0.5414 | 0.5365 | 0.5462 |
| mst_eps0p5_seed0 | 0.5247 | 0.5175 | 0.6264 | 0.5731 | 0.5559 |
| mst_eps10_seed0 | 0.4433 | 0.429 | 0.4863 | 0.4696 | 0.4745 |
| mst_eps15_seed0 | 0.4404 | 0.4236 | 0.6414 | 0.4434 | 0.4871 |
| mst_eps15_seed1 | 0.4449 | 0.426 | 0.557 | 0.4555 | 0.4697 |
| mst_eps15_seed2 | 0.433 | 0.4291 | 0.5226 | 0.4944 | 0.4926 |
| mst_eps1_seed0 | 0.5146 | 0.4934 | 0.6097 | 0.5912 | 0.6005 |
| mst_eps20_seed0 | 0.4342 | 0.43 | 0.5002 | 0.4708 | 0.4833 |
| mst_eps5_seed0 | 0.4632 | 0.4438 | 0.5555 | 0.6119 | 0.535 |
| mst_eps8_seed0 | 0.4483 | 0.4304 | 0.5291 | 0.4583 | 0.4677 |
| patectgan_eps15_seed0 | 0.389 | 0.3838 | 0.3989 | 0.3785 | 0.4153 |
| patectgan_eps1_seed0 | 0.5757 | 0.5897 | 0.6256 | 0.5979 | 0.6152 |
| patectgan_eps5_seed0 | 0.4486 | 0.4289 | 0.4459 | 0.4325 | 0.4825 |
| tvae_cap256_seed0 | 0.24 | 0.2005 | 0.2179 | 0.2053 | 0.2312 |
| tvae_ep1000_seed0 | 0.2306 | 0.2061 | 0.2199 | 0.2086 | 0.2273 |
| tvae_ind_seed0 | 0.2485 | 0.2249 | 0.2168 | 0.2148 | 0.2328 |
| tvae_qt_seed0 | 0.2446 | 0.1935 | 0.2041 | 0.1945 | 0.2274 |
| tvae_qt_seed1 | 0.2449 | 0.1954 | 0.21 | 0.2056 | 0.2273 |
| tvae_qt_seed2 | 0.2397 | 0.1847 | 0.1961 | 0.1887 | 0.2179 |
| tvae_seed0 | 0.2481 | 0.2156 | 0.2347 | 0.2162 | 0.2398 |
| tvae_seed1 | 0.2381 | 0.2278 | 0.2341 | 0.2299 | 0.2385 |
| tvae_seed2 | 0.2578 | 0.2222 | 0.2375 | 0.2246 | 0.2578 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.3379 | 0.3259 | 0.0186 | 0.0243 |
| aim50_eps1_seed0 | 0.3255 | 0.3203 | 0.0184 | 0.0201 |
| ctgan_qt_seed0 | 0.286 | 0.2873 | 0.0749 | 0.0744 |
| ctgan_seed0 | 0.3036 | 0.2997 | 0.1071 | 0.1066 |
| ctgan_seed1 | 0.3371 | 0.339 | 0.0803 | 0.0801 |
| ctgan_seed2 | 0.3106 | 0.3054 | 0.0744 | 0.0793 |
| ddpm_g_seed0 | 0.9648 | 0.9667 | 0.3092 | 0.3086 |
| ddpm_seed0 | 0.9646 | 0.9666 | 0.3354 | 0.335 |
| ddpm_seed1 | 0.9642 | 0.9661 | 0.3436 | 0.3437 |
| ddpm_seed2 | 0.9649 | 0.9665 | 0.3458 | 0.3474 |
| dpctgan_eps10_seed0 | 0.9485 | 0.947 | 0.292 | 0.2922 |
| dpctgan_eps15_seed0 | 0.9811 | 0.9813 | 0.3208 | 0.3198 |
| dpctgan_eps15_seed1 | 0.9257 | 0.9249 | 0.3346 | 0.3345 |
| dpctgan_eps15_seed2 | 0.8968 | 0.8972 | 0.3089 | 0.3082 |
| dpctgan_eps1_seed0 | 0.9346 | 0.9325 | 0.3331 | 0.3341 |
| dpctgan_eps20_seed0 | 0.9194 | 0.9177 | 0.3438 | 0.3437 |
| dpctgan_eps5_seed0 | 0.8865 | 0.8857 | 0.3028 | 0.3023 |
| dpctgan_eps8_seed0 | 0.9609 | 0.9613 | 0.3271 | 0.327 |
| gaussian_copula_seed0 | 0.5221 | 0.5113 | 0.0134 | 0.0145 |
| gaussian_copula_seed1 | 0.5189 | 0.5063 | 0.0126 | 0.0144 |
| gaussian_copula_seed2 | 0.5233 | 0.5215 | 0.013 | 0.0142 |
| mst_eps0p5_seed0 | 0.5076 | 0.5083 | 0.0493 | 0.0487 |
| mst_eps10_seed0 | 0.418 | 0.4122 | 0.0095 | 0.0104 |
| mst_eps15_seed0 | 0.4137 | 0.408 | 0.0088 | 0.01 |
| mst_eps15_seed1 | 0.4119 | 0.4027 | 0.0091 | 0.0102 |
| mst_eps15_seed2 | 0.4124 | 0.4066 | 0.0095 | 0.0102 |
| mst_eps1_seed0 | 0.4667 | 0.4673 | 0.0249 | 0.0263 |
| mst_eps20_seed0 | 0.4126 | 0.4048 | 0.0089 | 0.0102 |
| mst_eps5_seed0 | 0.4323 | 0.4262 | 0.0095 | 0.0116 |
| mst_eps8_seed0 | 0.4186 | 0.4136 | 0.0087 | 0.0106 |
| patectgan_eps15_seed0 | 0.3754 | 0.376 | 0.0778 | 0.0772 |
| patectgan_eps1_seed0 | 0.5717 | 0.5653 | 0.0741 | 0.0738 |
| patectgan_eps5_seed0 | 0.4156 | 0.4205 | 0.15 | 0.1533 |
| tvae_cap256_seed0 | 0.1942 | 0.1939 | 0.0568 | 0.0577 |
| tvae_ep1000_seed0 | 0.1942 | 0.1952 | 0.0557 | 0.057 |
| tvae_ind_seed0 | 0.2084 | 0.2157 | 0.0609 | 0.0624 |
| tvae_qt_seed0 | 0.188 | 0.1926 | 0.0536 | 0.0557 |
| tvae_qt_seed1 | 0.1894 | 0.1926 | 0.0676 | 0.0668 |
| tvae_qt_seed2 | 0.1748 | 0.1807 | 0.058 | 0.0606 |
| tvae_seed0 | 0.2044 | 0.2054 | 0.0557 | 0.056 |
| tvae_seed1 | 0.2105 | 0.2096 | 0.0602 | 0.0599 |
| tvae_seed2 | 0.2111 | 0.206 | 0.0622 | 0.0634 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 1621 | 0.0649 | 0.0453 | 0.7964 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 11175 | 0.0222 | 0.0177 | 0.9975 | 0 | - |
| *noise floor* | corr-ratio (num-cat) | 11468 | 0.0278 | 0.0169 | 0.9613 | 0 | - |
| aim40_eps1_seed0 | Spearman (num-num) | 120 | 0.1618 | 0.132 | 0.4083 | 0 | `lab_results_crpNonHs_value_last|lab_results_crpNonHs_value_first` (0.7183 -> 0.0059) |
| aim40_eps1_seed0 | Cramer's V (cat-cat) | 276 | 0.1431 | 0.0596 | 0.6196 | 7 | `cause_of_death_isAllCause_f5a_w3a_first|med_cortico_syst_history` (0.0863 -> 0.7433) |
| aim40_eps1_seed0 | corr-ratio (num-cat) | 992 | 0.0324 | 0.0 | 0.8871 | 1 | `encounters_numOfPreviousHFStays_count|encounter_primary_reason_HF_Disease_f5a_w1mo_first` (0.0804 -> 0.6476) |
| aim50_eps1_seed0 | Spearman (num-num) | 210 | 0.187 | 0.1238 | 0.3952 | 3 | `lab_results_ntProBnp_value_first|lab_results_valideGFR_value_first` (-0.4614 -> 0.5042) |
| aim50_eps1_seed0 | Cramer's V (cat-cat) | 406 | 0.1039 | 0.0503 | 0.6847 | 13 | `encounter_primary_reason_CV_Disease_f5a_w1mo_first|conditions_pad` (0.0507 -> 0.8447) |
| aim50_eps1_seed0 | corr-ratio (num-cat) | 1407 | 0.0451 | 0.0 | 0.8671 | 14 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_categorizedValue` (0.9194 -> 0.078) |
| ctgan_qt_seed0 | Spearman (num-num) | 1205 | 0.1226 | 0.088 | 0.5477 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.989 -> -0.0517) |
| ctgan_qt_seed0 | Cramer's V (cat-cat) | 11175 | 0.0767 | 0.0395 | 0.7623 | 0 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.0399) |
| ctgan_qt_seed0 | corr-ratio (num-cat) | 10528 | 0.0456 | 0.0257 | 0.8817 | 3 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0686) |
| ctgan_seed0 | Spearman (num-num) | 1334 | 0.1311 | 0.0965 | 0.5172 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.989 -> -0.1742) |
| ctgan_seed0 | Cramer's V (cat-cat) | 11175 | 0.0767 | 0.0353 | 0.7687 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.0245) |
| ctgan_seed0 | corr-ratio (num-cat) | 10716 | 0.0487 | 0.0268 | 0.8576 | 2 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9497 -> 0.0952) |
| ctgan_seed1 | Spearman (num-num) | 1445 | 0.1322 | 0.0971 | 0.5059 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.989 -> 0.0142) |
| ctgan_seed1 | Cramer's V (cat-cat) | 11175 | 0.0679 | 0.0285 | 0.8311 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.0127) |
| ctgan_seed1 | corr-ratio (num-cat) | 11280 | 0.0452 | 0.0237 | 0.8807 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0581) |
| ctgan_seed2 | Spearman (num-num) | 1318 | 0.1231 | 0.0872 | 0.5539 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.989 -> -0.0528) |
| ctgan_seed2 | Cramer's V (cat-cat) | 11175 | 0.0748 | 0.0389 | 0.7552 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.0051) |
| ctgan_seed2 | corr-ratio (num-cat) | 10340 | 0.0461 | 0.0247 | 0.8677 | 2 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9497 -> 0.1007) |
| ddpm_g_seed0 | Spearman (num-num) | 72 | 0.1116 | 0.0764 | 0.6667 | 0 | `lab_results_cholTot_value_first|lab_results_ldl_value_last` (0.8631 -> -0.0016) |
| ddpm_g_seed0 | Cramer's V (cat-cat) | 11175 | 0.0543 | 0.0281 | 0.8952 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.104) |
| ddpm_g_seed0 | corr-ratio (num-cat) | 4136 | 0.2398 | 0.0292 | 0.7413 | 731 | `lab_results_ldl_value_first|cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first` (0.0079 -> 3.0) |
| ddpm_seed0 | Spearman (num-num) | 105 | 0.0867 | 0.06 | 0.6952 | 0 | `encounters_numOfPreviousHFStays_count|conditions_heartFailure_timeFromEarliest_first` (0.4967 -> 0.0538) |
| ddpm_seed0 | Cramer's V (cat-cat) | 11175 | 0.0548 | 0.0286 | 0.8937 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.078) |
| ddpm_seed0 | corr-ratio (num-cat) | 4888 | 0.19 | 0.0265 | 0.7735 | 682 | `lab_results_ldl_value_first|encounter_primary_reason_HF_Disease_f5a_w7d_first` (0.02 -> 3.0) |
| ddpm_seed1 | Spearman (num-num) | 80 | 0.0948 | 0.0569 | 0.675 | 0 | `lab_results_tropTHs_value_last|eGFR_2021_ckd_epi_creatinine` (-0.207 -> 1.0) |
| ddpm_seed1 | Cramer's V (cat-cat) | 11175 | 0.0539 | 0.0282 | 0.9043 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.0599) |
| ddpm_seed1 | corr-ratio (num-cat) | 5076 | 0.2833 | 0.0313 | 0.6885 | 979 | `lab_results_validSerumCreatinine_value_first|conditions_osa` (0.0564 -> 3.0) |
| ddpm_seed2 | Spearman (num-num) | 109 | 0.1214 | 0.0901 | 0.5413 | 0 | `lab_results_cholTot_value_last|lab_results_ldl_value_last` (0.8766 -> -0.0018) |
| ddpm_seed2 | Cramer's V (cat-cat) | 11175 | 0.0564 | 0.0279 | 0.8954 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.0695) |
| ddpm_seed2 | corr-ratio (num-cat) | 5264 | 0.1892 | 0.0279 | 0.7682 | 627 | `lab_results_validSerumCreatinine_value_first|cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first` (0.1103 -> 2.6905) |
| dpctgan_eps10_seed0 | Spearman (num-num) | 168 | 0.3441 | 0.238 | 0.2798 | 30 | `lab_results_potassium_value_first|lab_results_sodium_value_first` (-0.1694 -> 0.8556) |
| dpctgan_eps10_seed0 | Cramer's V (cat-cat) | 4560 | 0.1078 | 0.0438 | 0.7831 | 1 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.0009) |
| dpctgan_eps10_seed0 | corr-ratio (num-cat) | 3572 | 0.0469 | 0.0258 | 0.867 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0) |
| dpctgan_eps15_seed0 | Spearman (num-num) | 231 | 0.3802 | 0.3267 | 0.1688 | 38 | `lab_results_validSerumCreatinine_value_last|lab_results_valideGFR_value_last` (-0.9109 -> 0.4169) |
| dpctgan_eps15_seed0 | Cramer's V (cat-cat) | 4851 | 0.1066 | 0.0457 | 0.7761 | 2 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isAllCause_f5a_w1mo_first` (1.0 -> 0.0009) |
| dpctgan_eps15_seed0 | corr-ratio (num-cat) | 4136 | 0.0464 | 0.0256 | 0.869 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.0039) |
| dpctgan_eps15_seed1 | Spearman (num-num) | 300 | 0.2947 | 0.197 | 0.3067 | 50 | `vital_signs_weight_value_last|vital_signs_bmi_value_last` (0.8256 -> -0.255) |
| dpctgan_eps15_seed1 | Cramer's V (cat-cat) | 4950 | 0.0958 | 0.0405 | 0.7966 | 0 | `encounter_primary_reason_CV_Disease_f5a_w1a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (1.0 -> 0.0012) |
| dpctgan_eps15_seed1 | corr-ratio (num-cat) | 4700 | 0.0438 | 0.0231 | 0.8866 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.027) |
| dpctgan_eps15_seed2 | Spearman (num-num) | 300 | 0.2598 | 0.183 | 0.3033 | 20 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_last` (-0.7136 -> 0.9457) |
| dpctgan_eps15_seed2 | Cramer's V (cat-cat) | 4278 | 0.0946 | 0.0414 | 0.8034 | 1 | `cause_of_death_isCV_f5a_w1a_first|cause_of_death_isNonRenalAndNonCV_f5a_w1a_first` (1.0 -> 0.0012) |
| dpctgan_eps15_seed2 | corr-ratio (num-cat) | 4700 | 0.0478 | 0.0259 | 0.8711 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0347) |
| dpctgan_eps1_seed0 | Spearman (num-num) | 231 | 0.3175 | 0.259 | 0.2597 | 34 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_first` (-0.9057 -> 0.2749) |
| dpctgan_eps1_seed0 | Cramer's V (cat-cat) | 8001 | 0.0974 | 0.0414 | 0.7919 | 5 | `cause_of_death_isCV_f5a_w5a_first|cause_of_death_isRenal_f5a_w5a_first` (1.0 -> 0.0009) |
| dpctgan_eps1_seed0 | corr-ratio (num-cat) | 4136 | 0.0469 | 0.0245 | 0.875 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9497 -> 0.0249) |
| dpctgan_eps20_seed0 | Spearman (num-num) | 299 | 0.3721 | 0.2695 | 0.204 | 49 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_first` (-0.9057 -> 0.8714) |
| dpctgan_eps20_seed0 | Cramer's V (cat-cat) | 5565 | 0.0972 | 0.0423 | 0.7885 | 3 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.0009) |
| dpctgan_eps20_seed0 | corr-ratio (num-cat) | 4700 | 0.0487 | 0.0259 | 0.863 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0202) |
| dpctgan_eps5_seed0 | Spearman (num-num) | 276 | 0.2869 | 0.2109 | 0.2609 | 31 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_last` (-0.7136 -> 0.7588) |
| dpctgan_eps5_seed0 | Cramer's V (cat-cat) | 4095 | 0.0989 | 0.0414 | 0.7919 | 0 | `cause_of_death_isCV_f5a_w1mo_first|cause_of_death_isAllCause_f5a_w1mo_first` (1.0 -> 0.0009) |
| dpctgan_eps5_seed0 | corr-ratio (num-cat) | 4512 | 0.0514 | 0.0273 | 0.8641 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.0) |
| dpctgan_eps8_seed0 | Spearman (num-num) | 228 | 0.4216 | 0.3282 | 0.25 | 53 | `lab_results_validSerumCreatinine_value_last|lab_results_valideGFR_value_last` (-0.9109 -> 0.5868) |
| dpctgan_eps8_seed0 | Cramer's V (cat-cat) | 4851 | 0.1011 | 0.0432 | 0.7765 | 2 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first` (1.0 -> 0.0009) |
| dpctgan_eps8_seed0 | corr-ratio (num-cat) | 4136 | 0.048 | 0.0281 | 0.8632 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.0596) |
| gaussian_copula_seed0 | Spearman (num-num) | 879 | 0.1091 | 0.0766 | 0.6052 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> -0.0057) |
| gaussian_copula_seed0 | Cramer's V (cat-cat) | 11175 | 0.0654 | 0.0252 | 0.8617 | 0 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0012) |
| gaussian_copula_seed0 | corr-ratio (num-cat) | 8084 | 0.0374 | 0.0204 | 0.9232 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (0.8476 -> 0.0951) |
| gaussian_copula_seed1 | Spearman (num-num) | 880 | 0.1116 | 0.077 | 0.5898 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.016) |
| gaussian_copula_seed1 | Cramer's V (cat-cat) | 11175 | 0.065 | 0.0253 | 0.862 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0334) |
| gaussian_copula_seed1 | corr-ratio (num-cat) | 8084 | 0.0372 | 0.0211 | 0.925 | 0 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isAllCause_f5a_w1a_first` (0.8285 -> 0.088) |
| gaussian_copula_seed2 | Spearman (num-num) | 890 | 0.1162 | 0.0797 | 0.6079 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0212) |
| gaussian_copula_seed2 | Cramer's V (cat-cat) | 11175 | 0.0654 | 0.0256 | 0.859 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.0172) |
| gaussian_copula_seed2 | corr-ratio (num-cat) | 8272 | 0.0382 | 0.0215 | 0.9201 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_HF_Disease_f5a_w1a_first` (0.8476 -> 0.0729) |
| mst_eps0p5_seed0 | Spearman (num-num) | 1682 | 0.2559 | 0.2012 | 0.3002 | 102 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` (1.0 -> -0.5085) |
| mst_eps0p5_seed0 | Cramer's V (cat-cat) | 10440 | 0.179 | 0.1302 | 0.4088 | 589 | `cause_of_death_isRenal_f5a_w1a_first|cause_of_death_isNonRenalAndNonCV_f5a_w1a_first` (1.0 -> 0.0143) |
| mst_eps0p5_seed0 | corr-ratio (num-cat) | 11468 | 0.1388 | 0.0718 | 0.5685 | 617 | `lab_results_ldl_value_first|conditions_aidshiv` (0.0112 -> 0.9238) |
| mst_eps10_seed0 | Spearman (num-num) | 1730 | 0.274 | 0.2299 | 0.2173 | 148 | `lab_results_tropTHs_value_first|lab_results_tropTnHs_value_first` (1.0 -> -0.0513) |
| mst_eps10_seed0 | Cramer's V (cat-cat) | 11175 | 0.2483 | 0.1946 | 0.3102 | 1761 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.0023) |
| mst_eps10_seed0 | corr-ratio (num-cat) | 11468 | 0.1701 | 0.106 | 0.4877 | 1092 | `vital_signs_heartRate_value_last|med_arni_history` (0.0262 -> 0.8925) |
| mst_eps15_seed0 | Spearman (num-num) | 1753 | 0.2911 | 0.2479 | 0.1968 | 176 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> -0.0256) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 11175 | 0.2696 | 0.2251 | 0.2431 | 2013 | `med_arb|med_inotropes_history` (0.0077 -> 0.9788) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 11468 | 0.1829 | 0.1182 | 0.4641 | 1231 | `lab_results_tropTHs_value_first|conditions_aidshiv` (0.0077 -> 0.8321) |
| mst_eps15_seed1 | Spearman (num-num) | 1738 | 0.3081 | 0.2684 | 0.187 | 215 | `lab_results_crpNonHs_value_first|lab_results_albuminBS_value_first` (-0.4598 -> 0.5695) |
| mst_eps15_seed1 | Cramer's V (cat-cat) | 11175 | 0.2772 | 0.2349 | 0.2498 | 2139 | `med_arb|med_inotropes_history` (0.0077 -> 0.9624) |
| mst_eps15_seed1 | corr-ratio (num-cat) | 11468 | 0.1938 | 0.1231 | 0.4635 | 1471 | `lab_results_tropTnHs_value_first|cause_of_death_isRenal_f5a_w1mo_first` (0.0087 -> 0.997) |
| mst_eps15_seed2 | Spearman (num-num) | 1669 | 0.3019 | 0.262 | 0.1821 | 189 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` (1.0 -> -0.1037) |
| mst_eps15_seed2 | Cramer's V (cat-cat) | 11175 | 0.2594 | 0.2121 | 0.278 | 1883 | `med_antiinfl|med_thrombolytic_history` (0.0079 -> 1.0) |
| mst_eps15_seed2 | corr-ratio (num-cat) | 11468 | 0.1978 | 0.1247 | 0.4543 | 1374 | `lab_results_tropTHs_value_first|conditions_rd` (0.0009 -> 1.8929) |
| mst_eps1_seed0 | Spearman (num-num) | 1732 | 0.2819 | 0.2355 | 0.261 | 138 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_first` (0.8503 -> -0.5454) |
| mst_eps1_seed0 | Cramer's V (cat-cat) | 11175 | 0.3416 | 0.3289 | 0.1487 | 3092 | `hyperkalemia_severity_categorizedValue|med_thrombolytic` (0.0318 -> 0.978) |
| mst_eps1_seed0 | corr-ratio (num-cat) | 11468 | 0.1883 | 0.1212 | 0.4637 | 1103 | `lab_results_hdl_value_last|med_thrombolytic` (0.0127 -> 1.0) |
| mst_eps20_seed0 | Spearman (num-num) | 1681 | 0.326 | 0.2913 | 0.1684 | 234 | `vital_signs_weight_value_p6mo_first|lab_results_hdl_value_first` (-0.2677 -> 0.7944) |
| mst_eps20_seed0 | Cramer's V (cat-cat) | 11175 | 0.2626 | 0.2131 | 0.2803 | 1887 | `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first|conditions_ap` (0.0198 -> 0.9809) |
| mst_eps20_seed0 | corr-ratio (num-cat) | 11280 | 0.1904 | 0.1241 | 0.454 | 1343 | `lab_results_tropTnHs_value_last|cause_of_death_isAllCause_f5a_w1mo_first` (0.0138 -> 0.9937) |
| mst_eps5_seed0 | Spearman (num-num) | 1643 | 0.2755 | 0.2406 | 0.2319 | 123 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> -0.0489) |
| mst_eps5_seed0 | Cramer's V (cat-cat) | 11175 | 0.288 | 0.2487 | 0.2334 | 2292 | `med_arb|med_inotropes_history` (0.0077 -> 0.9623) |
| mst_eps5_seed0 | corr-ratio (num-cat) | 11280 | 0.1678 | 0.104 | 0.4917 | 906 | `lab_results_tropTnHs_value_first|med_potassium_binders_history` (0.0007 -> 0.9393) |
| mst_eps8_seed0 | Spearman (num-num) | 1743 | 0.2778 | 0.2494 | 0.1842 | 134 | `lab_results_tropTnHs_value_first|encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first` (-0.1225 -> 0.9991) |
| mst_eps8_seed0 | Cramer's V (cat-cat) | 11175 | 0.2498 | 0.2187 | 0.2648 | 1484 | `med_arb|med_inotropes_history` (0.0077 -> 0.9652) |
| mst_eps8_seed0 | corr-ratio (num-cat) | 11468 | 0.1657 | 0.1121 | 0.4779 | 742 | `lab_results_ferritin_value_first|cause_of_death_isAllCause_f5a_w3mo_first` (0.0414 -> 1.0) |
| patectgan_eps15_seed0 | Spearman (num-num) | 1366 | 0.1238 | 0.096 | 0.522 | 0 | `lab_results_triGly_value_first|lab_results_hdl_value_first` (-0.1999 -> 0.6066) |
| patectgan_eps15_seed0 | Cramer's V (cat-cat) | 11026 | 0.0557 | 0.0264 | 0.9112 | 5 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first` (1.0 -> 0.0009) |
| patectgan_eps15_seed0 | corr-ratio (num-cat) | 10904 | 0.0404 | 0.0268 | 0.9066 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_renal_complications_f5a_w1a_first` (0.8476 -> 0.0426) |
| patectgan_eps1_seed0 | Spearman (num-num) | 810 | 0.1381 | 0.0912 | 0.542 | 0 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.099) |
| patectgan_eps1_seed0 | Cramer's V (cat-cat) | 11175 | 0.0779 | 0.0271 | 0.8489 | 0 | `cause_of_death_isRenal_f5a_w3a_first|cause_of_death_isNonRenalAndNonCV_f5a_w3a_first` (1.0 -> 0.0044) |
| patectgan_eps1_seed0 | corr-ratio (num-cat) | 7708 | 0.0383 | 0.0209 | 0.9127 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0803) |
| patectgan_eps5_seed0 | Spearman (num-num) | 1147 | 0.156 | 0.1174 | 0.4342 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0962) |
| patectgan_eps5_seed0 | Cramer's V (cat-cat) | 9316 | 0.073 | 0.0298 | 0.873 | 0 | `cause_of_death_isCV_f5a_w3mo_first|cause_of_death_isRenal_f5a_w3mo_first` (1.0 -> 0.0012) |
| patectgan_eps5_seed0 | corr-ratio (num-cat) | 9776 | 0.0443 | 0.0281 | 0.883 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_renal_complications_f5a_w1a_first` (0.8476 -> 0.0222) |
| tvae_cap256_seed0 | Spearman (num-num) | 1370 | 0.0855 | 0.0614 | 0.6985 | 2 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.989 -> 0.1708) |
| tvae_cap256_seed0 | Cramer's V (cat-cat) | 8515 | 0.052 | 0.0315 | 0.8532 | 4 | `smoking_status_smoker_last|conditions_substance_abuse` (0.067 -> 0.6535) |
| tvae_cap256_seed0 | corr-ratio (num-cat) | 10528 | 0.0422 | 0.0232 | 0.8862 | 6 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_CV_Disease_f5a_w3a_first` (0.8082 -> 0.0) |
| tvae_ep1000_seed0 | Spearman (num-num) | 1459 | 0.0852 | 0.0637 | 0.6799 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.0527) |
| tvae_ep1000_seed0 | Cramer's V (cat-cat) | 8256 | 0.0539 | 0.0347 | 0.841 | 2 | `med_acei_history|med_arb_history` (0.007 -> 0.5026) |
| tvae_ep1000_seed0 | corr-ratio (num-cat) | 10716 | 0.043 | 0.0251 | 0.8803 | 9 | `nyha_nyha_pET|conditions_af` (0.0723 -> 0.7687) |
| tvae_ind_seed0 | Spearman (num-num) | 1524 | 0.0949 | 0.0776 | 0.6024 | 0 | `lab_results_tropTHs_value_last|lab_results_tropTHs_value_first` (0.8641 -> 0.1646) |
| tvae_ind_seed0 | Cramer's V (cat-cat) | 8001 | 0.0543 | 0.0342 | 0.8424 | 0 | `conditions_ap|conditions_dysl` (0.1435 -> 0.7106) |
| tvae_ind_seed0 | corr-ratio (num-cat) | 10904 | 0.0437 | 0.0251 | 0.8755 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6666 -> 0.007) |
| tvae_qt_seed0 | Spearman (num-num) | 1513 | 0.1036 | 0.0813 | 0.579 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.1114) |
| tvae_qt_seed0 | Cramer's V (cat-cat) | 7875 | 0.0589 | 0.0359 | 0.8441 | 5 | `conditions_ap|conditions_dysl` (0.1435 -> 0.7877) |
| tvae_qt_seed0 | corr-ratio (num-cat) | 11092 | 0.047 | 0.0276 | 0.8533 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6666 -> 0.0871) |
| tvae_qt_seed1 | Spearman (num-num) | 1366 | 0.1038 | 0.0811 | 0.5747 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.0263) |
| tvae_qt_seed1 | Cramer's V (cat-cat) | 7750 | 0.0639 | 0.0364 | 0.8028 | 10 | `conditions_ap|conditions_dysl` (0.1435 -> 0.7639) |
| tvae_qt_seed1 | corr-ratio (num-cat) | 10528 | 0.0498 | 0.0278 | 0.8436 | 6 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6666 -> 0.0) |
| tvae_qt_seed2 | Spearman (num-num) | 1441 | 0.106 | 0.0854 | 0.5642 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.031) |
| tvae_qt_seed2 | Cramer's V (cat-cat) | 7875 | 0.0599 | 0.0373 | 0.829 | 6 | `cause_of_death_isAllCause_f5a_w1mo_first|med_potassium_binders` (0.0382 -> 0.7065) |
| tvae_qt_seed2 | corr-ratio (num-cat) | 10716 | 0.0493 | 0.0276 | 0.8401 | 1 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6666 -> 0.0) |
| tvae_seed0 | Spearman (num-num) | 1443 | 0.0954 | 0.071 | 0.6535 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> -0.0298) |
| tvae_seed0 | Cramer's V (cat-cat) | 8001 | 0.0658 | 0.0419 | 0.7897 | 3 | `med_oral_antidiabetic_history|conditions_devices` (0.1156 -> 0.6579) |
| tvae_seed0 | corr-ratio (num-cat) | 10716 | 0.0461 | 0.0256 | 0.8602 | 10 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_renal_complications_f5a_w5a_first` (0.0 -> 0.7055) |
| tvae_seed1 | Spearman (num-num) | 1444 | 0.0918 | 0.068 | 0.6697 | 0 | `lab_results_tropTHs_value_last|lab_results_tropTHs_value_first` (0.8641 -> -0.0068) |
| tvae_seed1 | Cramer's V (cat-cat) | 8001 | 0.0688 | 0.0417 | 0.7853 | 4 | `med_acei_history|med_arb_history` (0.007 -> 0.6362) |
| tvae_seed1 | corr-ratio (num-cat) | 10904 | 0.0455 | 0.0248 | 0.8728 | 10 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_CV_Disease_f5a_w1a_first` (0.8193 -> 0.0) |
| tvae_seed2 | Spearman (num-num) | 1401 | 0.0888 | 0.0672 | 0.6667 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.0629) |
| tvae_seed2 | Cramer's V (cat-cat) | 8001 | 0.0584 | 0.0334 | 0.8315 | 2 | `conditions_ap|conditions_dysl` (0.1435 -> 0.7831) |
| tvae_seed2 | corr-ratio (num-cat) | 10528 | 0.0422 | 0.0249 | 0.8842 | 4 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6666 -> 0.0862) |

## original vs preprocessed

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=0.0, W/std=0.0, mean 70.9842 -> 70.9842, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.0, W/std=0.0, mean 10.5254 -> 10.5254, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.0, W/std=0.0, mean 52.5614 -> 52.5614, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=0.0, W/std=0.0, mean 77.4701 -> 77.4701, missing 8% -> 8%
- `vital_signs_weight_value_p6mo_first`: KS=0.0, W/std=0.0, mean 79.7398 -> 79.7398, missing 8% -> 8%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 10 -> 10 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.1802, W/std=0.2533, mean 82.6105 -> 120.1538, missing 95% -> 96%
- `lab_results_creatUS_value_first`: KS=0.1225, W/std=0.1885, mean 754.8314 -> 672.9652, missing 90% -> 89%
- `lab_results_triGly_value_last`: KS=0.102, W/std=0.1875, mean 1.5467 -> 1.3626, missing 90% -> 92%
- `lab_results_triGly_value_first`: KS=0.102, W/std=0.1905, mean 1.5982 -> 1.3738, missing 90% -> 92%
- `lab_results_cholTot_value_last`: KS=0.0936, W/std=0.1611, mean 3.9492 -> 3.8205, missing 89% -> 90%
Worst categorical columns (by TVD):
- `med_mra_history`: TVD=0.0489, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.042, 7 -> 7 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.0333, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0328, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0322, 7 -> 7 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps1_seed0]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=0.7531, W/std=0.6532, mean 10.5818 -> 17.6378, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.6775, W/std=0.697, mean 43.6592 -> 83.7671, missing 11% -> 11%
- `lab_results_crpNonHs_value_first`: KS=0.6207, W/std=0.3089, mean 45.8171 -> 47.3329, missing 11% -> 11%
- `encounters_numOfPreviousHFStays_count`: KS=0.5443, W/std=0.2739, mean 51.846 -> 64.4298, missing 0% -> 0%
- `lab_results_ntProBnp_value_last`: KS=0.4938, W/std=0.1806, mean 10078.8465 -> 11192.1835, missing 19% -> 19%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0898, 10 -> 10 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0549, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0407, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0406, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first`: TVD=0.0232, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps1_seed0]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=0.7531, W/std=0.7216, mean 10.5818 -> 19.8693, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.6775, W/std=0.4821, mean 43.6592 -> 63.8786, missing 11% -> 11%
- `lab_results_crpNonHs_value_first`: KS=0.6207, W/std=0.2817, mean 45.8171 -> 60.7276, missing 11% -> 12%
- `encounters_numOfPreviousHFStays_count`: KS=0.5443, W/std=0.3916, mean 51.846 -> 73.5514, missing 0% -> 0%
- `lab_results_ntProBnp_value_last`: KS=0.4938, W/std=0.192, mean 10078.8465 -> 10798.0971, missing 19% -> 21%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0933, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0467, 10 -> 10 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0434, 5 -> 4 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0409, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first`: TVD=0.0239, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ctgan_qt_seed0]

Worst numeric columns (by KS):
- `lab_results_ldl_value_last`: KS=0.731, W/std=2.474, mean 2.1208 -> 4.6224, missing 91% -> 95%
- `lab_results_cholTot_value_first`: KS=0.6852, W/std=2.1835, mean 3.9591 -> 6.5134, missing 89% -> 89%
- `lab_results_cholTot_value_last`: KS=0.6217, W/std=1.8587, mean 3.9492 -> 6.1405, missing 89% -> 84%
- `lab_results_creatUS_value_last`: KS=0.5043, W/std=1.6418, mean 694.4878 -> 1527.4322, missing 90% -> 83%
- `lab_results_hdl_value_first`: KS=0.5029, W/std=1.5874, mean 1.1873 -> 1.8972, missing 91% -> 86%
Worst categorical columns (by TVD):
- `med_mra`: TVD=0.2584, 2 -> 2 categories, missing 0% -> 0%
- `med_ll_history`: TVD=0.25, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2465, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.2446, 3 -> 3 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.2405, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed0]

Worst numeric columns (by KS):
- `lab_results_tropTHs_value_last`: KS=0.6963, W/std=0.348, mean 0.4652 -> 1.0361, missing 63% -> 55%
- `echocardiographs_lvef_pET_last`: KS=0.6466, W/std=1.777, mean 40.6716 -> 75.3478, missing 83% -> 90%
- `lab_results_tropTnHs_value_last`: KS=0.6195, W/std=0.5328, mean 281.7666 -> 686.3831, missing 88% -> 92%
- `lab_results_ferritin_value_first`: KS=0.5896, W/std=0.4129, mean 561.1328 -> 1055.2606, missing 80% -> 74%
- `lab_results_tropTHs_value_first`: KS=0.5794, W/std=0.2971, mean 0.2171 -> 0.3891, missing 63% -> 84%
Worst categorical columns (by TVD):
- `med_acei_history`: TVD=0.3402, 2 -> 2 categories, missing 0% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.2889, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w1a_first`: TVD=0.2466, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.24, 3 -> 3 categories, missing 0% -> 0%
- `med_insulins`: TVD=0.2386, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed1]

Worst numeric columns (by KS):
- `lab_results_tropTHs_value_last`: KS=0.7162, W/std=0.4958, mean 0.4652 -> 1.5468, missing 63% -> 81%
- `lab_results_tropTnHs_value_last`: KS=0.6286, W/std=0.477, mean 281.7666 -> 596.2326, missing 88% -> 86%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.623, W/std=1.7962, mean 471.831 -> 382.3341, missing 49% -> 38%
- `lab_results_sodium_value_first`: KS=0.6219, W/std=1.4185, mean 137.0846 -> 144.2056, missing 4% -> 7%
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.6105, W/std=0.513, mean 82.6105 -> 128.0564, missing 95% -> 97%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.299, 2 -> 2 categories, missing 0% -> 0%
- `med_rdoad`: TVD=0.2956, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.2689, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_plat_history`: TVD=0.2599, 2 -> 2 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.2241, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed2]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_last`: KS=0.7426, W/std=0.6741, mean 281.7666 -> 773.3162, missing 88% -> 97%
- `lab_results_tropTHs_value_first`: KS=0.6393, W/std=0.439, mean 0.2171 -> 0.513, missing 63% -> 41%
- `lab_results_tropTHs_value_last`: KS=0.6176, W/std=0.3568, mean 0.4652 -> 1.0394, missing 63% -> 98%
- `lab_results_tropTnHs_value_first`: KS=0.5933, W/std=0.4682, mean 212.254 -> 399.4239, missing 88% -> 97%
- `lab_results_potassium_value_last`: KS=0.573, W/std=1.4282, mean 4.1751 -> 3.3427, missing 4% -> 8%
Worst categorical columns (by TVD):
- `med_anti_plat`: TVD=0.3357, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.3212, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.2319, 2 -> 2 categories, missing 0% -> 0%
- `med_vasodil_history`: TVD=0.2233, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.2143, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[ddpm_g_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_last`: KS=0.9997, W/std=6.5374, mean 77.3464 -> 207.0, missing 15% -> 44%
- `vital_signs_systolicBp_value_first`: KS=0.9997, W/std=4.3831, mean 130.3643 -> 257.0, missing 5% -> 61%
- `vital_signs_diastolicBp_value_first`: KS=0.9997, W/std=6.795, mean 75.7013 -> 201.0, missing 5% -> 62%
- `lab_results_hemoglobin_value_last`: KS=0.9997, W/std=3.9153, mean 120.0203 -> 209.482, missing 3% -> 69%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.6299, mean 121.9202 -> 209.482, missing 3% -> 67%
Worst categorical columns (by TVD):
- `med_thrombolytic`: TVD=0.7967, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hypothyroid`: TVD=0.6839, 2 -> 2 categories, missing 0% -> 0%
- `conditions_dem`: TVD=0.6743, 2 -> 2 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.6482, 5 -> 5 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w3mo_first`: TVD=0.647, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=0.9997, W/std=6.5386, mean 77.4079 -> 207.0, missing 8% -> 43%
- `vital_signs_weight_value_last`: KS=0.9997, W/std=6.5374, mean 77.3464 -> 207.0, missing 15% -> 44%
- `vital_signs_systolicBp_value_last`: KS=0.9997, W/std=4.4648, mean 121.0177 -> 221.0, missing 5% -> 54%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.6299, mean 121.9202 -> 209.482, missing 3% -> 67%
- `lab_results_crpNonHs_value_last`: KS=0.9997, W/std=9.6461, mean 43.6592 -> 629.7, missing 11% -> 61%
Worst categorical columns (by TVD):
- `med_thrombolytic`: TVD=0.7941, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_smoker_last`: TVD=0.7297, 3 -> 3 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.7098, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1mo_first`: TVD=0.7075, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.7004, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed1]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=0.9997, W/std=6.5386, mean 77.4079 -> 207.0, missing 8% -> 36%
- `vital_signs_weight_value_p6mo_first`: KS=0.9997, W/std=7.9548, mean 79.6287 -> 240.0, missing 8% -> 36%
- `vital_signs_weight_value_last`: KS=0.9997, W/std=6.5374, mean 77.3464 -> 207.0, missing 15% -> 55%
- `vital_signs_systolicBp_value_last`: KS=0.9997, W/std=4.4648, mean 121.0177 -> 221.0, missing 5% -> 41%
- `lab_results_hemoglobin_value_last`: KS=0.9997, W/std=3.9153, mean 120.0203 -> 209.482, missing 3% -> 50%
Worst categorical columns (by TVD):
- `cause_of_death_isCV_f5a_w7d_first`: TVD=0.7601, 3 -> 3 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.7325, 5 -> 5 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.7169, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.6978, 3 -> 3 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.6817, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed2]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=0.9997, W/std=7.9548, mean 79.6287 -> 240.0, missing 8% -> 43%
- `lab_results_potassium_value_last`: KS=0.9997, W/std=5.361, mean 4.1751 -> 7.3, missing 4% -> 45%
- `lab_results_validSerumCreatinine_value_last`: KS=0.9997, W/std=2.3007, mean 12.1376 -> 22.5748, missing 10% -> 48%
- `eGFR_2021_ckd_epi_creatinine`: KS=0.9997, W/std=4.9822, mean 64.4886 -> 193.8265, missing 10% -> 38%
- `vital_signs_systolicBp_value_last`: KS=0.9996, W/std=4.4645, mean 121.0177 -> 220.9941, missing 5% -> 55%
Worst categorical columns (by TVD):
- `cause_of_death_isAllCause_f5a_w1mo_first`: TVD=0.8111, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w7d_first`: TVD=0.7788, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.751, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w1mo_first`: TVD=0.7436, 3 -> 3 categories, missing 0% -> 0%
- `smoking_status_smoker_last`: TVD=0.7342, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps10_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.6287 -> None, missing 8% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=None, mean 171.0332 -> None, missing 21% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 77.3464 -> None, missing 15% -> 100%
- `vital_signs_bmi_value_last`: KS=1.0, W/std=None, mean 26.9378 -> None, missing 46% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 69.0778 -> None, missing 5% -> 100%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.8827, 6 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8744, 10 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.8563, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8284, 7 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.8273, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 77.4079 -> None, missing 8% -> 100%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.6287 -> None, missing 8% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 77.3464 -> None, missing 15% -> 100%
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 171.4561 -> None, missing 44% -> 100%
- `vital_signs_bmi_value_last`: KS=1.0, W/std=None, mean 26.9378 -> None, missing 46% -> 100%
Worst categorical columns (by TVD):
- `conditions_aidshiv`: TVD=0.9866, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_smoker_last`: TVD=0.9676, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.9537, 7 -> 2 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.9193, 5 -> 4 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w1mo_first`: TVD=0.9193, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed1]

Worst numeric columns (by KS):
- `vital_signs_systolicBp_value_last`: KS=1.0, W/std=None, mean 121.0177 -> None, missing 5% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 75.7013 -> None, missing 5% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 69.0778 -> None, missing 5% -> 100%
- `lab_results_ferritin_value_last`: KS=1.0, W/std=None, mean 523.1277 -> None, missing 80% -> 100%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=None, mean 561.1328 -> None, missing 80% -> 100%
Worst categorical columns (by TVD):
- `conditions_dep`: TVD=0.975, 2 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.9548, 3 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.9352, 3 -> 2 categories, missing 0% -> 0%
- `conditions_tia`: TVD=0.9247, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.9028, 7 -> 4 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed2]

Worst numeric columns (by KS):
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 171.4561 -> None, missing 44% -> 100%
- `vital_signs_bmi_value_last`: KS=1.0, W/std=None, mean 26.9378 -> None, missing 46% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 75.7013 -> None, missing 5% -> 100%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 112.7017 -> None, missing 48% -> 100%
- `lab_results_ferritin_value_last`: KS=1.0, W/std=None, mean 523.1277 -> None, missing 80% -> 100%
Worst categorical columns (by TVD):
- `conditions_pericardial`: TVD=0.9569, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first`: TVD=0.8639, 2 -> 1 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.8538, 3 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8526, 10 -> 4 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first`: TVD=0.848, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps1_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 77.3464 -> None, missing 15% -> 100%
- `vital_signs_bmi_value_last`: KS=1.0, W/std=None, mean 26.9378 -> None, missing 46% -> 100%
- `vital_signs_systolicBp_value_last`: KS=1.0, W/std=None, mean 121.0177 -> None, missing 5% -> 100%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 112.7017 -> None, missing 48% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 109.7305 -> None, missing 48% -> 100%
Worst categorical columns (by TVD):
- `cause_of_death_isCV_f5a_w1mo_first`: TVD=0.9926, 3 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3mo_first`: TVD=0.9852, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.9483, 3 -> 3 categories, missing 0% -> 0%
- `med_diuretics`: TVD=0.9233, 2 -> 2 categories, missing 0% -> 0%
- `med_antiarrhytmic_history`: TVD=0.9017, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps20_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.6287 -> None, missing 8% -> 100%
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 171.4561 -> None, missing 44% -> 100%
- `vital_signs_systolicBp_value_last`: KS=1.0, W/std=None, mean 121.0177 -> None, missing 5% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 69.0778 -> None, missing 5% -> 100%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 112.7017 -> None, missing 48% -> 100%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.9562, 3 -> 1 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.9531, 5 -> 3 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.9341, 7 -> 5 categories, missing 0% -> 0%
- `conditions_ld`: TVD=0.9241, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hypothyroid`: TVD=0.9221, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps5_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 77.4079 -> None, missing 8% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 77.3464 -> None, missing 15% -> 100%
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 171.4561 -> None, missing 44% -> 100%
- `vital_signs_bmi_value_last`: KS=1.0, W/std=None, mean 26.9378 -> None, missing 46% -> 100%
- `vital_signs_systolicBp_value_first`: KS=1.0, W/std=None, mean 130.3643 -> None, missing 5% -> 100%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.9455, 5 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w1mo_first`: TVD=0.9244, 3 -> 2 categories, missing 0% -> 0%
- `med_antiarrhytmic_history`: TVD=0.9145, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9068, 10 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`: TVD=0.9068, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps8_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 79.6287 -> None, missing 8% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=None, mean 171.0332 -> None, missing 21% -> 100%
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 171.4561 -> None, missing 44% -> 100%
- `vital_signs_bmi_value_last`: KS=1.0, W/std=None, mean 26.9378 -> None, missing 46% -> 100%
- `vital_signs_systolicBp_value_first`: KS=1.0, W/std=None, mean 130.3643 -> None, missing 5% -> 100%
Worst categorical columns (by TVD):
- `med_thrombolytic`: TVD=0.9855, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.9497, 3 -> 1 categories, missing 0% -> 0%
- `conditions_hypothyroid`: TVD=0.8975, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8753, 10 -> 1 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8588, 7 -> 3 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=None, mean 523.1277 -> None, missing 80% -> 100%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=None, mean 561.1328 -> None, missing 80% -> 100%
- `lab_results_tropTnHs_value_last`: KS=1.0, W/std=None, mean 281.7666 -> None, missing 88% -> 100%
- `lab_results_tropTnHs_value_first`: KS=1.0, W/std=None, mean 212.254 -> None, missing 88% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=None, mean 1.5467 -> None, missing 90% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4507, 10 -> 9 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0368, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0282, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.0265, 3 -> 3 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.025, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed1]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=None, mean 523.1277 -> None, missing 80% -> 100%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=None, mean 561.1328 -> None, missing 80% -> 100%
- `lab_results_tropTnHs_value_first`: KS=1.0, W/std=None, mean 212.254 -> None, missing 88% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=None, mean 1.5467 -> None, missing 90% -> 100%
- `lab_results_ldl_value_last`: KS=1.0, W/std=None, mean 2.1208 -> None, missing 91% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4482, 10 -> 9 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0374, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.0272, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w5a_first`: TVD=0.0261, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0258, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed2]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=None, mean 523.1277 -> None, missing 80% -> 100%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=None, mean 561.1328 -> None, missing 80% -> 100%
- `lab_results_tropTnHs_value_last`: KS=1.0, W/std=None, mean 281.7666 -> None, missing 88% -> 100%
- `lab_results_tropTnHs_value_first`: KS=1.0, W/std=None, mean 212.254 -> None, missing 88% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=None, mean 1.5467 -> None, missing 90% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4269, 10 -> 9 categories, missing 0% -> 0%
- `med_mra`: TVD=0.0338, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.0322, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w3a_first`: TVD=0.0313, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.0306, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps0p5_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.962, mean 0.0804 -> 1.7387, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=2.0018, mean 0.4652 -> 5.451, missing 63% -> 58%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=7.8525, mean 523.1277 -> 15527.9913, missing 80% -> 63%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=5.5127, mean 561.1328 -> 12381.8607, missing 80% -> 78%
- `lab_results_triGly_value_last`: KS=0.9676, W/std=6.4778, mean 1.5467 -> 8.8575, missing 90% -> 83%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.193, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.171, 7 -> 7 categories, missing 0% -> 0%
- `conditions_devices`: TVD=0.1566, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1549, 6 -> 6 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1536, 10 -> 9 categories, missing 0% -> 0%

## train vs synthetic[mst_eps10_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1828, mean 0.0804 -> 1.0811, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.6221, mean 0.4652 -> 4.4233, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.322, mean 523.1277 -> 3034.3424, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1031, mean 561.1328 -> 2847.181, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8494, mean 0.2171 -> 0.8967, missing 63% -> 63%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.035, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0299, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0221, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.021, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.0209, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1927, mean 0.0804 -> 1.0897, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.6136, mean 0.4652 -> 4.4231, missing 63% -> 65%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2663, mean 523.1277 -> 2895.4228, missing 80% -> 81%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1139, mean 561.1328 -> 2889.2197, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.9359, mean 0.2171 -> 0.9907, missing 63% -> 64%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0287, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0284, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0282, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0251, 7 -> 7 categories, missing 0% -> 0%
- `med_ll`: TVD=0.0216, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed1]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.2023, mean 0.0804 -> 1.0978, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.6277, mean 0.4652 -> 4.4633, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.234, mean 523.1277 -> 2748.4728, missing 80% -> 81%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.0872, mean 561.1328 -> 2734.4313, missing 80% -> 81%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8711, mean 0.2171 -> 0.913, missing 63% -> 64%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0297, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0277, 6 -> 6 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0242, 10 -> 10 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.0235, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1a_first`: TVD=0.0233, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed2]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1593, mean 0.0804 -> 1.0584, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.6476, mean 0.4652 -> 4.5224, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.231, mean 523.1277 -> 2710.927, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1378, mean 561.1328 -> 2967.1687, missing 80% -> 81%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8766, mean 0.2171 -> 0.8039, missing 63% -> 63%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0315, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0275, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0257, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0224, 7 -> 7 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.0218, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps1_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=2.5354, mean 0.0804 -> 2.2269, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=2.6434, mean 0.4652 -> 7.0512, missing 63% -> 67%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2367, mean 523.1277 -> 2773.9696, missing 80% -> 78%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=2.299, mean 561.1328 -> 5492.553, missing 80% -> 78%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=1.5423, mean 0.2171 -> 1.524, missing 63% -> 65%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1208, 10 -> 10 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.1011, 5 -> 5 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.0748, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0698, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0669, 7 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps20_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1623, mean 0.0804 -> 1.0584, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.596, mean 0.4652 -> 4.3808, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2448, mean 523.1277 -> 2633.5, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1266, mean 561.1328 -> 2640.8347, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8784, mean 0.2171 -> 0.9214, missing 63% -> 63%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0274, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0265, 7 -> 7 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.0233, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0224, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.0219, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps5_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.3006, mean 0.0804 -> 1.1804, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.6899, mean 0.4652 -> 4.6519, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2448, mean 523.1277 -> 2633.5, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.259, mean 561.1328 -> 3248.2457, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8339, mean 0.2171 -> 0.849, missing 63% -> 62%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.039, 6 -> 6 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0376, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0364, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0272, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.0221, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps8_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1641, mean 0.0804 -> 1.0622, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.6152, mean 0.4652 -> 4.3935, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.6962, mean 523.1277 -> 3753.3077, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1266, mean 561.1328 -> 2640.8655, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8433, mean 0.2171 -> 0.8405, missing 63% -> 64%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0259, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.022, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0217, 6 -> 6 categories, missing 0% -> 0%
- `med_ll`: TVD=0.0207, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.0192, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps15_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9801, W/std=0.3518, mean 0.0804 -> 0.3686, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.8771, W/std=6.1629, mean 0.4652 -> 15.8257, missing 63% -> 81%
- `lab_results_tropTnHs_value_last`: KS=0.8268, W/std=5.1185, mean 281.7666 -> 4792.8914, missing 88% -> 96%
- `lab_results_ferritin_value_last`: KS=0.8256, W/std=4.9157, mean 523.1277 -> 9911.9774, missing 80% -> 87%
- `lab_results_tropTHs_value_first`: KS=0.8205, W/std=4.0541, mean 0.2171 -> 3.6523, missing 63% -> 83%
Worst categorical columns (by TVD):
- `cause_of_death_isCV_f5a_w5a_first`: TVD=0.4202, 3 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.4196, 3 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w5a_first`: TVD=0.4148, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w5a_first`: TVD=0.4142, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w3a_first`: TVD=0.3631, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps1_seed0]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_last`: KS=1.0, W/std=None, mean 281.7666 -> None, missing 88% -> 100%
- `echocardiographs_lvef_pET_first`: KS=0.9984, W/std=4.1962, mean 40.9911 -> -35.5821, missing 82% -> 94%
- `echocardiographs_lvef_pET_last`: KS=0.9983, W/std=10.071, mean 40.6716 -> -157.0726, missing 83% -> 81%
- `lab_results_hdl_value_first`: KS=0.9833, W/std=1.9851, mean 1.1873 -> 0.2993, missing 91% -> 99%
- `lab_results_ldl_value_last`: KS=0.9832, W/std=1.9127, mean 2.1208 -> 0.1862, missing 91% -> 99%
Worst categorical columns (by TVD):
- `cause_of_death_isNonRenalAndNonCV_f5a_w3a_first`: TVD=0.2352, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag_history`: TVD=0.2155, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.2084, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.1898, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w5a_first`: TVD=0.1863, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps5_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=1.0, W/std=None, mean 82.6105 -> None, missing 95% -> 100%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=1.0, W/std=None, mean 118.2794 -> None, missing 80% -> 100%
- `lab_results_tropTnHs_value_last`: KS=0.9931, W/std=2.0115, mean 281.7666 -> 1961.4219, missing 88% -> 100%
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=0.2844, mean 0.0804 -> 0.2299, missing 0% -> 0%
- `lab_results_ferritin_value_last`: KS=0.8672, W/std=2.4636, mean 523.1277 -> 5157.2048, missing 80% -> 98%
Worst categorical columns (by TVD):
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.6992, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first`: TVD=0.6969, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first`: TVD=0.6943, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.6821, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.6625, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[tvae_cap256_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5485, W/std=0.1211, mean 11.0621 -> 10.0586, missing 0% -> 30%
- `lab_results_tropTHs_value_last`: KS=0.4643, W/std=0.1624, mean 0.4652 -> 0.2769, missing 63% -> 57%
- `lab_results_creatUS_value_last`: KS=0.4531, W/std=0.5528, mean 694.4878 -> 484.93, missing 90% -> 98%
- `lab_results_ferritin_value_first`: KS=0.3989, W/std=0.1948, mean 561.1328 -> 451.728, missing 80% -> 83%
- `lab_results_ferritin_value_last`: KS=0.3727, W/std=0.1415, mean 523.1277 -> 490.1252, missing 80% -> 84%
Worst categorical columns (by TVD):
- `med_bb`: TVD=0.1802, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1733, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.1696, 2 -> 2 categories, missing 0% -> 0%
- `med_arb`: TVD=0.1685, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.1647, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ep1000_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5458, W/std=0.1164, mean 11.0621 -> 10.185, missing 0% -> 37%
- `lab_results_tropTHs_value_last`: KS=0.4822, W/std=0.1284, mean 0.4652 -> 0.5691, missing 63% -> 60%
- `lab_results_creatUS_value_last`: KS=0.4634, W/std=0.635, mean 694.4878 -> 457.9084, missing 90% -> 99%
- `lab_results_ferritin_value_first`: KS=0.4546, W/std=0.2139, mean 561.1328 -> 526.2699, missing 80% -> 84%
- `lab_results_ferritin_value_last`: KS=0.3977, W/std=0.1843, mean 523.1277 -> 433.0055, missing 80% -> 84%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.2021, 7 -> 7 categories, missing 0% -> 0%
- `med_platelet`: TVD=0.1909, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_plat`: TVD=0.1843, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1834, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1805, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[tvae_ind_seed0]

Worst numeric columns (by KS):
- `lab_results_creatUS_value_last`: KS=0.4002, W/std=0.4963, mean 694.4878 -> 464.3406, missing 90% -> 91%
- `vital_signs_heartRate_value_first`: KS=0.3976, W/std=0.5146, mean 112.7017 -> 104.8118, missing 48% -> 51%
- `lab_results_creatUS_value_first`: KS=0.3962, W/std=0.5204, mean 754.8314 -> 511.0617, missing 90% -> 92%
- `lab_results_ferritin_value_first`: KS=0.3757, W/std=0.1821, mean 561.1328 -> 251.0415, missing 80% -> 83%
- `echocardiographs_lvef_pET_last`: KS=0.3696, W/std=0.4302, mean 40.6716 -> 33.9577, missing 83% -> 84%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2053, 10 -> 10 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1867, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1736, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.1633, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.1595, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_first`: KS=0.4103, W/std=0.1845, mean 561.1328 -> 424.6081, missing 80% -> 90%
- `lab_results_tropTnHs_value_last`: KS=0.4061, W/std=0.1915, mean 281.7666 -> 246.5453, missing 88% -> 92%
- `lab_results_tropTnHs_value_first`: KS=0.3948, W/std=0.2212, mean 212.254 -> 178.1776, missing 88% -> 92%
- `lab_results_ferritin_value_last`: KS=0.3883, W/std=0.1839, mean 523.1277 -> 403.8987, missing 80% -> 90%
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.3626, W/std=0.2459, mean 82.6105 -> 79.847, missing 95% -> 96%
Worst categorical columns (by TVD):
- `med_bb`: TVD=0.25, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.2262, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.2044, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1886, 10 -> 10 categories, missing 0% -> 0%
- `beta_blocker_use_pre_dc`: TVD=0.187, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed1]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_last`: KS=0.3794, W/std=0.2278, mean 281.7666 -> 129.9669, missing 88% -> 97%
- `vital_signs_heartRate_value_first`: KS=0.3709, W/std=0.538, mean 112.7017 -> 104.4617, missing 48% -> 43%
- `lab_results_tropTnHs_value_first`: KS=0.3703, W/std=0.2718, mean 212.254 -> 104.2795, missing 88% -> 97%
- `lab_results_ferritin_value_first`: KS=0.3654, W/std=0.1739, mean 561.1328 -> 377.5183, missing 80% -> 96%
- `lab_results_ferritin_value_last`: KS=0.3639, W/std=0.1827, mean 523.1277 -> 371.6618, missing 80% -> 95%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.3039, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2608, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.2268, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.2181, 2 -> 2 categories, missing 0% -> 0%
- `med_mra`: TVD=0.212, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed2]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_first`: KS=0.3818, W/std=0.2697, mean 212.254 -> 112.3845, missing 88% -> 92%
- `lab_results_tropTnHs_value_last`: KS=0.3574, W/std=0.2242, mean 281.7666 -> 131.5681, missing 88% -> 92%
- `lab_results_creatUS_value_first`: KS=0.35, W/std=0.5446, mean 754.8314 -> 716.2502, missing 90% -> 96%
- `lab_results_albuminBS_value_first`: KS=0.3342, W/std=0.4833, mean 31.1435 -> 28.2499, missing 57% -> 63%
- `vital_signs_heartRate_value_first`: KS=0.3215, W/std=0.4887, mean 112.7017 -> 105.33, missing 48% -> 59%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2789, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2494, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.2101, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.1962, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.1893, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.549, W/std=0.1181, mean 11.0621 -> 11.4304, missing 0% -> 36%
- `lab_results_creatUS_value_last`: KS=0.4799, W/std=0.651, mean 694.4878 -> 433.5764, missing 90% -> 98%
- `lab_results_tropTHs_value_last`: KS=0.4344, W/std=0.166, mean 0.4652 -> 0.2249, missing 63% -> 68%
- `lab_results_creatUS_value_first`: KS=0.4026, W/std=0.5556, mean 754.8314 -> 831.751, missing 90% -> 99%
- `lab_results_ferritin_value_first`: KS=0.3936, W/std=0.1954, mean 561.1328 -> 444.7827, missing 80% -> 89%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.3016, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.1924, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.1822, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1801, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1733, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed1]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5466, W/std=0.1389, mean 11.0621 -> 12.8307, missing 0% -> 34%
- `lab_results_creatUS_value_first`: KS=0.4729, W/std=0.565, mean 754.8314 -> 549.7791, missing 90% -> 100%
- `lab_results_tropTHs_value_last`: KS=0.4584, W/std=0.1699, mean 0.4652 -> 0.2523, missing 63% -> 63%
- `lab_results_creatUS_value_last`: KS=0.444, W/std=0.6084, mean 694.4878 -> 448.2278, missing 90% -> 97%
- `lab_results_ferritin_value_last`: KS=0.3914, W/std=0.1581, mean 523.1277 -> 662.1953, missing 80% -> 88%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2797, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.202, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1869, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1761, 2 -> 1 categories, missing 0% -> 0%
- `med_arb`: TVD=0.1691, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed2]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5474, W/std=0.0901, mean 11.0621 -> 10.907, missing 0% -> 34%
- `lab_results_creatUS_value_last`: KS=0.5271, W/std=0.6746, mean 694.4878 -> 391.3517, missing 90% -> 99%
- `lab_results_ferritin_value_first`: KS=0.5047, W/std=0.2406, mean 561.1328 -> 589.2562, missing 80% -> 91%
- `lab_results_tropTHs_value_last`: KS=0.4731, W/std=0.1703, mean 0.4652 -> 0.2502, missing 63% -> 66%
- `lab_results_ferritin_value_last`: KS=0.46, W/std=0.1753, mean 523.1277 -> 605.8144, missing 80% -> 91%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2936, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2023, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.2003, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.177, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.1766, 2 -> 2 categories, missing 0% -> 0%
