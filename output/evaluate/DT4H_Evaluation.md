# Evaluation: fidelity against the sampling-noise floor

Metrics are computed per column over observed values (nulls excluded); missingness rates are compared separately. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. 38 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 164 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| train vs holdout | 211 | 0.0486 | 0.0405 | 0.9344 | 0.0782 | 0.01 | 1.0 | 0.0075 |
| train vs synthetic[aim40_eps1_seed0] | 40 | 0.3366 | 0.1938 | 0.0 | 0.309 | 0.015 | 0.9167 | 0.0066 |
| train vs synthetic[aim50_eps1_seed0] | 50 | 0.3251 | 0.256 | 0.0 | 0.33 | 0.0144 | 0.931 | 0.0123 |
| train vs synthetic[ctgan_qt_seed0] | 211 | 0.2767 | 0.2656 | 0.1148 | 0.7896 | 0.0734 | 0.42 | 0.1221 |
| train vs synthetic[ctgan_seed0] | 211 | 0.2957 | 0.2678 | 0.0656 | 0.5578 | 0.1053 | 0.3 | 0.0792 |
| train vs synthetic[ctgan_seed1] | 211 | 0.3318 | 0.3409 | 0.0492 | 0.6602 | 0.0794 | 0.4333 | 0.0673 |
| train vs synthetic[ctgan_seed2] | 211 | 0.3009 | 0.2561 | 0.082 | 0.5863 | 0.0759 | 0.4133 | 0.0902 |
| train vs synthetic[ddpm_g_seed0] | 211 | 0.9649 | 0.9986 | 0.0 | 6.5254 | 0.3088 | 0.04 | 0.3189 |
| train vs synthetic[ddpm_seed0] | 211 | 0.9648 | 0.9986 | 0.0 | 6.5247 | 0.3353 | 0.0333 | 0.3191 |
| train vs synthetic[ddpm_seed1] | 211 | 0.9643 | 0.9986 | 0.0 | 6.5345 | 0.344 | 0.0333 | 0.293 |
| train vs synthetic[ddpm_seed2] | 211 | 0.9647 | 0.9986 | 0.0 | 6.5392 | 0.3472 | 0.04 | 0.2818 |
| train vs synthetic[dpctgan_eps10_seed0] | 171 | 0.8477 | 0.9757 | 0.0 | 2.8375 | 0.2921 | 0.1667 | 0.2513 |
| train vs synthetic[dpctgan_eps15_seed0] | 175 | 0.9531 | 0.9918 | 0.0 | 4.1274 | 0.3205 | 0.1533 | 0.3326 |
| train vs synthetic[dpctgan_eps15_seed1] | 177 | 0.8307 | 0.9688 | 0.0 | 3.3349 | 0.3347 | 0.16 | 0.2328 |
| train vs synthetic[dpctgan_eps15_seed2] | 178 | 0.7748 | 0.9103 | 0.0 | 2.3055 | 0.3089 | 0.14 | 0.2334 |
| train vs synthetic[dpctgan_eps1_seed0] | 176 | 0.8407 | 0.9437 | 0.0 | 2.7005 | 0.3331 | 0.1533 | 0.3158 |
| train vs synthetic[dpctgan_eps20_seed0] | 177 | 0.8135 | 0.896 | 0.0 | 2.6775 | 0.3438 | 0.1467 | 0.247 |
| train vs synthetic[dpctgan_eps5_seed0] | 176 | 0.7316 | 0.9414 | 0.0385 | 2.3057 | 0.3027 | 0.16 | 0.2503 |
| train vs synthetic[dpctgan_eps8_seed0] | 172 | 0.8889 | 0.9812 | 0.0 | 3.7186 | 0.327 | 0.1467 | 0.2146 |
| train vs synthetic[gaussian_copula_seed0] | 202 | 0.424 | 0.3641 | 0.0769 | 1.3393 | 0.0093 | 0.9933 | 0.1232 |
| train vs synthetic[gaussian_copula_seed1] | 204 | 0.4396 | 0.3897 | 0.0741 | 1.3379 | 0.0088 | 0.9933 | 0.1229 |
| train vs synthetic[gaussian_copula_seed2] | 202 | 0.4356 | 0.3711 | 0.0769 | 1.3367 | 0.009 | 0.9933 | 0.1258 |
| train vs synthetic[mst_eps0p5_seed0] | 211 | 0.5062 | 0.4315 | 0.0164 | 1.3234 | 0.0481 | 0.5867 | 0.0446 |
| train vs synthetic[mst_eps10_seed0] | 211 | 0.413 | 0.2914 | 0.0164 | 0.3984 | 0.0026 | 1.0 | 0.0039 |
| train vs synthetic[mst_eps15_seed0] | 211 | 0.408 | 0.2868 | 0.0164 | 0.3793 | 0.0018 | 1.0 | 0.0028 |
| train vs synthetic[mst_eps15_seed1] | 211 | 0.4051 | 0.2844 | 0.0164 | 0.3744 | 0.0018 | 1.0 | 0.0032 |
| train vs synthetic[mst_eps15_seed2] | 211 | 0.4098 | 0.2923 | 0.0164 | 0.3792 | 0.0019 | 1.0 | 0.0025 |
| train vs synthetic[mst_eps1_seed0] | 211 | 0.4663 | 0.3601 | 0.0 | 0.8966 | 0.0233 | 0.9267 | 0.0327 |
| train vs synthetic[mst_eps20_seed0] | 211 | 0.405 | 0.2859 | 0.0164 | 0.3684 | 0.0015 | 1.0 | 0.0018 |
| train vs synthetic[mst_eps5_seed0] | 211 | 0.4287 | 0.2999 | 0.0164 | 0.4592 | 0.0049 | 1.0 | 0.0091 |
| train vs synthetic[mst_eps8_seed0] | 211 | 0.4134 | 0.2936 | 0.0164 | 0.4121 | 0.003 | 1.0 | 0.0046 |
| train vs synthetic[patectgan_eps15_seed0] | 211 | 0.3661 | 0.2872 | 0.0 | 1.3902 | 0.0776 | 0.5733 | 0.09 |
| train vs synthetic[patectgan_eps1_seed0] | 210 | 0.5511 | 0.4829 | 0.0333 | 1.4461 | 0.0732 | 0.3467 | 0.1352 |
| train vs synthetic[patectgan_eps5_seed0] | 209 | 0.3898 | 0.3196 | 0.0 | 1.2615 | 0.1501 | 0.44 | 0.1046 |
| train vs synthetic[tvae_cap256_seed0] | 211 | 0.187 | 0.1585 | 0.2295 | 0.2499 | 0.0558 | 0.5333 | 0.0359 |
| train vs synthetic[tvae_ep1000_seed0] | 211 | 0.187 | 0.1594 | 0.1967 | 0.2478 | 0.055 | 0.5667 | 0.0391 |
| train vs synthetic[tvae_ind_seed0] | 211 | 0.2042 | 0.2041 | 0.1639 | 0.2489 | 0.0605 | 0.4733 | 0.0299 |
| train vs synthetic[tvae_qt_seed0] | 211 | 0.1816 | 0.1459 | 0.2787 | 0.2553 | 0.0545 | 0.5933 | 0.0369 |
| train vs synthetic[tvae_qt_seed1] | 211 | 0.183 | 0.1976 | 0.2787 | 0.2634 | 0.0662 | 0.5 | 0.0334 |
| train vs synthetic[tvae_qt_seed2] | 211 | 0.1704 | 0.1474 | 0.2951 | 0.252 | 0.0592 | 0.5867 | 0.0356 |
| train vs synthetic[tvae_seed0] | 211 | 0.1977 | 0.163 | 0.1639 | 0.2678 | 0.055 | 0.6067 | 0.0402 |
| train vs synthetic[tvae_seed1] | 211 | 0.2032 | 0.1963 | 0.2459 | 0.2626 | 0.0589 | 0.5067 | 0.0356 |
| train vs synthetic[tvae_seed2] | 211 | 0.204 | 0.1699 | 0.2131 | 0.2673 | 0.0625 | 0.52 | 0.0352 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| aim | 1 | 1 | 0.3251 | 0.0144 | 0.0123 |
| aim40 | 1 | 1 | 0.3366 | 0.015 | 0.0066 |
| ctgan | - | 3 | 0.3095 ± 0.0195 | 0.0869 ± 0.0161 | 0.0789 ± 0.0115 |
| ctgan_qt | - | 1 | 0.2767 | 0.0734 | 0.1221 |
| ddpm | - | 3 | 0.9646 ± 0.0003 | 0.3422 ± 0.0062 | 0.298 ± 0.0191 |
| ddpm_g | - | 1 | 0.9649 | 0.3088 | 0.3189 |
| dpctgan | 1 | 1 | 0.8407 | 0.3331 | 0.3158 |
| dpctgan | 5 | 1 | 0.7316 | 0.3027 | 0.2503 |
| dpctgan | 8 | 1 | 0.8889 | 0.327 | 0.2146 |
| dpctgan | 10 | 1 | 0.8477 | 0.2921 | 0.2513 |
| dpctgan | 15 | 3 | 0.8529 ± 0.0912 | 0.3214 ± 0.0129 | 0.2663 ± 0.0574 |
| dpctgan | 20 | 1 | 0.8135 | 0.3438 | 0.247 |
| gaussian_copula | - | 3 | 0.4331 ± 0.0081 | 0.009 ± 0.0003 | 0.124 ± 0.0016 |
| mst | 0.5 | 1 | 0.5062 | 0.0481 | 0.0446 |
| mst | 1 | 1 | 0.4663 | 0.0233 | 0.0327 |
| mst | 5 | 1 | 0.4287 | 0.0049 | 0.0091 |
| mst | 8 | 1 | 0.4134 | 0.003 | 0.0046 |
| mst | 10 | 1 | 0.413 | 0.0026 | 0.0039 |
| mst | 15 | 3 | 0.4076 ± 0.0024 | 0.0018 ± 0.0001 | 0.0028 ± 0.0004 |
| mst | 20 | 1 | 0.405 | 0.0015 | 0.0018 |
| patectgan | 1 | 1 | 0.5511 | 0.0732 | 0.1352 |
| patectgan | 5 | 1 | 0.3898 | 0.1501 | 0.1046 |
| patectgan | 15 | 1 | 0.3661 | 0.0776 | 0.09 |
| tvae | - | 3 | 0.2016 ± 0.0034 | 0.0588 ± 0.0038 | 0.037 ± 0.0028 |
| tvae_cap256 | - | 1 | 0.187 | 0.0558 | 0.0359 |
| tvae_ep1000 | - | 1 | 0.187 | 0.055 | 0.0391 |
| tvae_ind | - | 1 | 0.2042 | 0.0605 | 0.0299 |
| tvae_qt | - | 3 | 0.1783 ± 0.0069 | 0.06 ± 0.0059 | 0.0353 ± 0.0018 |

## Full-joint distinguishability (C2ST)

AUC of a classifier separating real from synthetic rows; 0.5 = joints indistinguishable. Floor (train vs holdout): **0.4572**.

| run | C2ST AUC |
|---|---|
| aim40_eps1_seed0 | 1.0 |
| aim50_eps1_seed0 | 1.0 |
| ctgan_qt_seed0 | 1.0 |
| ctgan_seed0 | 1.0 |
| ctgan_seed1 | 1.0 |
| ctgan_seed2 | 1.0 |
| ddpm_g_seed0 | 1.0 |
| ddpm_seed0 | 1.0 |
| ddpm_seed1 | 1.0 |
| ddpm_seed2 | 1.0 |
| dpctgan_eps10_seed0 | 1.0 |
| dpctgan_eps15_seed0 | 1.0 |
| dpctgan_eps15_seed1 | 1.0 |
| dpctgan_eps15_seed2 | 1.0 |
| dpctgan_eps1_seed0 | 1.0 |
| dpctgan_eps20_seed0 | 1.0 |
| dpctgan_eps5_seed0 | 1.0 |
| dpctgan_eps8_seed0 | 0.9995 |
| gaussian_copula_seed0 | 1.0 |
| gaussian_copula_seed1 | 1.0 |
| gaussian_copula_seed2 | 1.0 |
| mst_eps0p5_seed0 | 1.0 |
| mst_eps10_seed0 | 1.0 |
| mst_eps15_seed0 | 1.0 |
| mst_eps15_seed1 | 1.0 |
| mst_eps15_seed2 | 1.0 |
| mst_eps1_seed0 | 1.0 |
| mst_eps20_seed0 | 1.0 |
| mst_eps5_seed0 | 1.0 |
| mst_eps8_seed0 | 1.0 |
| patectgan_eps15_seed0 | 1.0 |
| patectgan_eps1_seed0 | 1.0 |
| patectgan_eps5_seed0 | 1.0 |
| tvae_cap256_seed0 | 0.998 |
| tvae_ep1000_seed0 | 0.998 |
| tvae_ind_seed0 | 0.9995 |
| tvae_qt_seed0 | 0.9992 |
| tvae_qt_seed1 | 0.9997 |
| tvae_qt_seed2 | 0.9994 |
| tvae_seed0 | 0.9982 |
| tvae_seed1 | 0.9991 |
| tvae_seed2 | 0.9994 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.0745 | 0.0684 | 0.0997 | 0.0754 | 0.0961 |
| aim40_eps1_seed0 | 0.3651 | 0.3489 | 0.4183 | 0.3464 | 0.375 |
| aim50_eps1_seed0 | 0.3427 | 0.3391 | 0.3894 | 0.3376 | 0.3758 |
| ctgan_qt_seed0 | 0.2891 | 0.2874 | 0.307 | 0.3 | 0.2906 |
| ctgan_seed0 | 0.3192 | 0.2976 | 0.3149 | 0.3018 | 0.3294 |
| ctgan_seed1 | 0.3421 | 0.339 | 0.3396 | 0.3439 | 0.3507 |
| ctgan_seed2 | 0.3016 | 0.325 | 0.3131 | 0.3191 | 0.3178 |
| ddpm_g_seed0 | 0.9652 | 0.9643 | 0.9658 | - | 0.9748 |
| ddpm_seed0 | 0.965 | 0.9643 | 0.9658 | - | 0.9747 |
| ddpm_seed1 | 0.9644 | 0.9642 | 0.964 | - | 0.9752 |
| ddpm_seed2 | 0.9634 | 0.9658 | 0.9653 | - | 0.975 |
| dpctgan_eps10_seed0 | - | 0.8518 | 0.8594 | 0.8394 | 0.8256 |
| dpctgan_eps15_seed0 | 0.9501 | 0.9551 | - | - | 0.9578 |
| dpctgan_eps15_seed1 | 0.8285 | - | - | 0.8315 | 0.8271 |
| dpctgan_eps15_seed2 | 0.7891 | 0.7697 | - | - | 0.7911 |
| dpctgan_eps1_seed0 | 0.8528 | 0.837 | - | 0.8133 | 0.844 |
| dpctgan_eps20_seed0 | - | 0.8173 | - | - | 0.8202 |
| dpctgan_eps5_seed0 | - | 0.7297 | 0.7518 | 0.7282 | 0.753 |
| dpctgan_eps8_seed0 | - | 0.8896 | - | - | 0.9088 |
| gaussian_copula_seed0 | 0.4419 | 0.4376 | 0.4462 | 0.4391 | 0.4285 |
| gaussian_copula_seed1 | 0.4396 | 0.4476 | 0.4497 | 0.4458 | 0.438 |
| gaussian_copula_seed2 | 0.4346 | 0.4248 | 0.432 | 0.4212 | 0.4436 |
| mst_eps0p5_seed0 | 0.5198 | 0.5163 | 0.6046 | 0.5661 | 0.5568 |
| mst_eps10_seed0 | 0.4408 | 0.4243 | 0.4858 | 0.4644 | 0.4688 |
| mst_eps15_seed0 | 0.4355 | 0.4215 | 0.5605 | 0.4385 | 0.4826 |
| mst_eps15_seed1 | 0.4371 | 0.4252 | 0.5531 | 0.4551 | 0.4642 |
| mst_eps15_seed2 | 0.4298 | 0.4304 | 0.5224 | 0.4892 | 0.488 |
| mst_eps1_seed0 | 0.5138 | 0.4919 | 0.6075 | 0.5786 | 0.597 |
| mst_eps20_seed0 | 0.435 | 0.4248 | 0.4938 | 0.4678 | 0.4769 |
| mst_eps5_seed0 | 0.4619 | 0.4406 | 0.5364 | 0.531 | 0.5326 |
| mst_eps8_seed0 | 0.443 | 0.4258 | 0.514 | 0.4527 | 0.4606 |
| patectgan_eps15_seed0 | 0.3741 | 0.372 | 0.385 | 0.3684 | 0.3927 |
| patectgan_eps1_seed0 | 0.5576 | 0.5635 | 0.5799 | 0.5683 | 0.5699 |
| patectgan_eps5_seed0 | 0.4067 | 0.3925 | 0.3951 | 0.4057 | 0.4389 |
| tvae_cap256_seed0 | 0.2282 | 0.191 | 0.2039 | 0.1968 | 0.2025 |
| tvae_ep1000_seed0 | 0.2166 | 0.1967 | 0.2082 | 0.2033 | 0.2156 |
| tvae_ind_seed0 | 0.2413 | 0.2179 | 0.2107 | 0.2153 | 0.2307 |
| tvae_qt_seed0 | 0.2385 | 0.1857 | 0.1867 | 0.1874 | 0.2135 |
| tvae_qt_seed1 | 0.235 | 0.1892 | 0.1975 | 0.1976 | 0.2079 |
| tvae_qt_seed2 | 0.2324 | 0.1765 | 0.1824 | 0.1809 | 0.1997 |
| tvae_seed0 | 0.2423 | 0.2058 | 0.2162 | 0.2083 | 0.2239 |
| tvae_seed1 | 0.232 | 0.2186 | 0.2184 | 0.2187 | 0.2235 |
| tvae_seed2 | 0.2481 | 0.2108 | 0.2202 | 0.2113 | 0.2211 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.3366 | 0.3259 | 0.015 | 0.0243 |
| aim50_eps1_seed0 | 0.3251 | 0.3203 | 0.0144 | 0.0201 |
| ctgan_qt_seed0 | 0.2767 | 0.2873 | 0.0734 | 0.0744 |
| ctgan_seed0 | 0.2957 | 0.2997 | 0.1053 | 0.1066 |
| ctgan_seed1 | 0.3318 | 0.339 | 0.0794 | 0.0801 |
| ctgan_seed2 | 0.3009 | 0.3054 | 0.0759 | 0.0793 |
| ddpm_g_seed0 | 0.9649 | 0.9667 | 0.3088 | 0.3086 |
| ddpm_seed0 | 0.9648 | 0.9666 | 0.3353 | 0.335 |
| ddpm_seed1 | 0.9643 | 0.9661 | 0.344 | 0.3437 |
| ddpm_seed2 | 0.9647 | 0.9665 | 0.3472 | 0.3474 |
| dpctgan_eps10_seed0 | 0.8477 | 0.846 | 0.2921 | 0.2922 |
| dpctgan_eps15_seed0 | 0.9531 | 0.9544 | 0.3205 | 0.3198 |
| dpctgan_eps15_seed1 | 0.8307 | 0.8304 | 0.3347 | 0.3345 |
| dpctgan_eps15_seed2 | 0.7748 | 0.776 | 0.3089 | 0.3082 |
| dpctgan_eps1_seed0 | 0.8407 | 0.8417 | 0.3331 | 0.3341 |
| dpctgan_eps20_seed0 | 0.8135 | 0.814 | 0.3438 | 0.3437 |
| dpctgan_eps5_seed0 | 0.7316 | 0.7318 | 0.3027 | 0.3023 |
| dpctgan_eps8_seed0 | 0.8889 | 0.8928 | 0.327 | 0.327 |
| gaussian_copula_seed0 | 0.424 | 0.4267 | 0.0093 | 0.0145 |
| gaussian_copula_seed1 | 0.4396 | 0.4423 | 0.0088 | 0.0144 |
| gaussian_copula_seed2 | 0.4356 | 0.4387 | 0.009 | 0.0142 |
| mst_eps0p5_seed0 | 0.5062 | 0.5083 | 0.0481 | 0.0487 |
| mst_eps10_seed0 | 0.413 | 0.4122 | 0.0026 | 0.0104 |
| mst_eps15_seed0 | 0.408 | 0.408 | 0.0018 | 0.01 |
| mst_eps15_seed1 | 0.4051 | 0.4027 | 0.0018 | 0.0102 |
| mst_eps15_seed2 | 0.4098 | 0.4066 | 0.0019 | 0.0102 |
| mst_eps1_seed0 | 0.4663 | 0.4673 | 0.0233 | 0.0263 |
| mst_eps20_seed0 | 0.405 | 0.4048 | 0.0015 | 0.0102 |
| mst_eps5_seed0 | 0.4287 | 0.4262 | 0.0049 | 0.0116 |
| mst_eps8_seed0 | 0.4134 | 0.4136 | 0.003 | 0.0106 |
| patectgan_eps15_seed0 | 0.3661 | 0.376 | 0.0776 | 0.0772 |
| patectgan_eps1_seed0 | 0.5511 | 0.558 | 0.0732 | 0.0738 |
| patectgan_eps5_seed0 | 0.3898 | 0.4008 | 0.1501 | 0.1533 |
| tvae_cap256_seed0 | 0.187 | 0.1939 | 0.0558 | 0.0577 |
| tvae_ep1000_seed0 | 0.187 | 0.1952 | 0.055 | 0.057 |
| tvae_ind_seed0 | 0.2042 | 0.2157 | 0.0605 | 0.0624 |
| tvae_qt_seed0 | 0.1816 | 0.1926 | 0.0545 | 0.0557 |
| tvae_qt_seed1 | 0.183 | 0.1926 | 0.0662 | 0.0668 |
| tvae_qt_seed2 | 0.1704 | 0.1807 | 0.0592 | 0.0606 |
| tvae_seed0 | 0.1977 | 0.2054 | 0.055 | 0.056 |
| tvae_seed1 | 0.2032 | 0.2096 | 0.0589 | 0.0599 |
| tvae_seed2 | 0.204 | 0.206 | 0.0625 | 0.0634 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 1621 | 0.0649 | 0.0453 | 0.7964 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 11175 | 0.0222 | 0.0177 | 0.9975 | 0 | - |
| *noise floor* | corr-ratio (num-cat) | 11468 | 0.0278 | 0.0169 | 0.9613 | 0 | - |
| aim40_eps1_seed0 | Spearman (num-num) | 120 | 0.1605 | 0.1178 | 0.4083 | 0 | `lab_results_crpNonHs_value_last|lab_results_crpNonHs_value_first` (0.7183 -> -0.0017) |
| aim40_eps1_seed0 | Cramer's V (cat-cat) | 276 | 0.1445 | 0.0695 | 0.6051 | 7 | `cause_of_death_isAllCause_f5a_w3a_first|med_cortico_syst_history` (0.0863 -> 0.7336) |
| aim40_eps1_seed0 | corr-ratio (num-cat) | 992 | 0.0349 | 0.0 | 0.878 | 1 | `encounters_numOfPreviousHFStays_count|ckd_severity_from_calculated_egfr` (0.2846 -> 0.8147) |
| aim50_eps1_seed0 | Spearman (num-num) | 210 | 0.1866 | 0.122 | 0.419 | 2 | `lab_results_ntProBnp_value_first|lab_results_valideGFR_value_first` (-0.4614 -> 0.4713) |
| aim50_eps1_seed0 | Cramer's V (cat-cat) | 406 | 0.1068 | 0.0582 | 0.665 | 11 | `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first|conditions_pad` (0.0507 -> 0.849) |
| aim50_eps1_seed0 | corr-ratio (num-cat) | 1407 | 0.0469 | 0.0 | 0.85 | 14 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_categorizedValue` (0.9194 -> 0.0675) |
| ctgan_qt_seed0 | Spearman (num-num) | 1545 | 0.1211 | 0.0858 | 0.5579 | 0 | `lab_results_cholTot_value_last|lab_results_ldl_value_first` (0.857 -> -0.3381) |
| ctgan_qt_seed0 | Cramer's V (cat-cat) | 11175 | 0.0747 | 0.0353 | 0.7646 | 0 | `med_anti_plat|med_platelet` (0.9886 -> 0.0177) |
| ctgan_qt_seed0 | corr-ratio (num-cat) | 11468 | 0.041 | 0.0219 | 0.9074 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0686) |
| ctgan_seed0 | Spearman (num-num) | 1601 | 0.1261 | 0.0905 | 0.5465 | 0 | `lab_results_ldl_value_last|lab_results_ldl_value_first` (0.9836 -> -0.1046) |
| ctgan_seed0 | Cramer's V (cat-cat) | 11175 | 0.0739 | 0.0309 | 0.7749 | 0 | `med_anti_plat_history|med_platelet_history` (0.9936 -> 0.0032) |
| ctgan_seed0 | corr-ratio (num-cat) | 11280 | 0.0424 | 0.0222 | 0.8952 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.1141) |
| ctgan_seed1 | Spearman (num-num) | 1637 | 0.1266 | 0.0917 | 0.5357 | 0 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> -0.0712) |
| ctgan_seed1 | Cramer's V (cat-cat) | 11175 | 0.0663 | 0.025 | 0.827 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.012) |
| ctgan_seed1 | corr-ratio (num-cat) | 11468 | 0.0403 | 0.0199 | 0.9071 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.0625) |
| ctgan_seed2 | Spearman (num-num) | 1603 | 0.1196 | 0.082 | 0.582 | 0 | `lab_results_tropTHs_value_first|lab_results_tropTnHs_value_first` (1.0 -> -0.1228) |
| ctgan_seed2 | Cramer's V (cat-cat) | 11175 | 0.0731 | 0.0351 | 0.7586 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.0327) |
| ctgan_seed2 | corr-ratio (num-cat) | 11280 | 0.0425 | 0.0227 | 0.8942 | 1 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.1461) |
| ddpm_g_seed0 | Spearman (num-num) | 277 | 0.095 | 0.0733 | 0.6462 | 0 | `lab_results_cholTot_value_first|lab_results_ldl_value_last` (0.8631 -> -0.0005) |
| ddpm_g_seed0 | Cramer's V (cat-cat) | 11175 | 0.0511 | 0.0236 | 0.8971 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.0258) |
| ddpm_g_seed0 | corr-ratio (num-cat) | 7520 | 0.0926 | 0.0191 | 0.8555 | 454 | `lab_results_tropTHs_value_first|conditions_heart_failure_hf_within_18mo_any` (0.0062 -> 1.7284) |
| ddpm_seed0 | Spearman (num-num) | 318 | 0.112 | 0.0755 | 0.6132 | 0 | `lab_results_cholTot_value_last|lab_results_cholTot_value_first` (0.9878 -> -0.0005) |
| ddpm_seed0 | Cramer's V (cat-cat) | 11175 | 0.0517 | 0.0241 | 0.8932 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.0159) |
| ddpm_seed0 | corr-ratio (num-cat) | 7332 | 0.0815 | 0.0221 | 0.8395 | 254 | `lab_results_creatUS_value_last|encounter_primary_reason_HF_Disease_f5a_w7d_first` (0.0598 -> 1.7296) |
| ddpm_seed1 | Spearman (num-num) | 274 | 0.1133 | 0.0659 | 0.6131 | 1 | `vital_signs_heartRate_value_first|lab_results_cholTot_value_first` (-0.079 -> 1.0) |
| ddpm_seed1 | Cramer's V (cat-cat) | 11175 | 0.0513 | 0.0244 | 0.9058 | 0 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.0079) |
| ddpm_seed1 | corr-ratio (num-cat) | 7332 | 0.1086 | 0.0198 | 0.8261 | 450 | `lab_results_ldl_value_last|cause_of_death_isAllCause_f5a_w1mo_first` (0.0887 -> 2.4192) |
| ddpm_seed2 | Spearman (num-num) | 332 | 0.1174 | 0.0814 | 0.5723 | 0 | `lab_results_ntProBnp_value_last|lab_results_ntProBnp_value_first` (0.8984 -> -0.0005) |
| ddpm_seed2 | Cramer's V (cat-cat) | 11175 | 0.0532 | 0.0235 | 0.9025 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.0357) |
| ddpm_seed2 | corr-ratio (num-cat) | 8460 | 0.1224 | 0.0204 | 0.8139 | 570 | `lab_results_ldl_value_first|conditions_hyperthyroid` (0.0585 -> 2.7214) |
| dpctgan_eps10_seed0 | Spearman (num-num) | 188 | 0.3184 | 0.1786 | 0.3138 | 30 | `lab_results_valideGFR_value_last|eGFR_2021_ckd_epi_creatinine` (0.9788 -> -0.052) |
| dpctgan_eps10_seed0 | Cramer's V (cat-cat) | 8128 | 0.0933 | 0.0415 | 0.7965 | 0 | `encounter_primary_reason_CV_Disease_f5a_w7d_first|encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (1.0 -> 0.0005) |
| dpctgan_eps10_seed0 | corr-ratio (num-cat) | 3760 | 0.0456 | 0.0243 | 0.8691 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.024) |
| dpctgan_eps15_seed0 | Spearman (num-num) | 231 | 0.3875 | 0.34 | 0.1861 | 39 | `lab_results_validSerumCreatinine_value_last|lab_results_valideGFR_value_last` (-0.9109 -> 0.4282) |
| dpctgan_eps15_seed0 | Cramer's V (cat-cat) | 8385 | 0.0974 | 0.0425 | 0.7939 | 1 | `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (1.0 -> 0.0003) |
| dpctgan_eps15_seed0 | corr-ratio (num-cat) | 4136 | 0.0454 | 0.0239 | 0.8772 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.0398) |
| dpctgan_eps15_seed1 | Spearman (num-num) | 324 | 0.2866 | 0.1898 | 0.287 | 49 | `vital_signs_weight_value_last|vital_signs_bmi_value_last` (0.8256 -> -0.292) |
| dpctgan_eps15_seed1 | Cramer's V (cat-cat) | 8256 | 0.0948 | 0.0402 | 0.8003 | 1 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0003) |
| dpctgan_eps15_seed1 | corr-ratio (num-cat) | 4888 | 0.0431 | 0.0212 | 0.8858 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0298) |
| dpctgan_eps15_seed2 | Spearman (num-num) | 348 | 0.247 | 0.1744 | 0.3276 | 23 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_last` (-0.7136 -> 0.9491) |
| dpctgan_eps15_seed2 | Cramer's V (cat-cat) | 8256 | 0.0917 | 0.0418 | 0.8 | 1 | `cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first|cause_of_death_isAllCause_f5a_w6mo_first` (1.0 -> 0.0004) |
| dpctgan_eps15_seed2 | corr-ratio (num-cat) | 5076 | 0.0458 | 0.0234 | 0.879 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0219) |
| dpctgan_eps1_seed0 | Spearman (num-num) | 251 | 0.3013 | 0.2223 | 0.2629 | 32 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_first` (-0.9057 -> 0.274) |
| dpctgan_eps1_seed0 | Cramer's V (cat-cat) | 10585 | 0.091 | 0.0386 | 0.8077 | 1 | `cause_of_death_isCV_f5a_w5a_first|cause_of_death_isRenal_f5a_w5a_first` (1.0 -> 0.0006) |
| dpctgan_eps1_seed0 | corr-ratio (num-cat) | 4324 | 0.0457 | 0.023 | 0.877 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9497 -> 0.0211) |
| dpctgan_eps20_seed0 | Spearman (num-num) | 321 | 0.3623 | 0.2695 | 0.2025 | 51 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_first` (-0.9057 -> 0.8761) |
| dpctgan_eps20_seed0 | Cramer's V (cat-cat) | 8515 | 0.0937 | 0.0412 | 0.8009 | 2 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.0004) |
| dpctgan_eps20_seed0 | corr-ratio (num-cat) | 4888 | 0.0479 | 0.0251 | 0.8689 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.0359) |
| dpctgan_eps5_seed0 | Spearman (num-num) | 322 | 0.2761 | 0.202 | 0.2764 | 33 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_last` (-0.7136 -> 0.7627) |
| dpctgan_eps5_seed0 | Cramer's V (cat-cat) | 7750 | 0.0958 | 0.0424 | 0.7948 | 2 | `cause_of_death_isRenal_f5a_w1a_first|cause_of_death_isAllCause_f5a_w1a_first` (1.0 -> 0.0004) |
| dpctgan_eps5_seed0 | corr-ratio (num-cat) | 4888 | 0.0515 | 0.0262 | 0.864 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.0053) |
| dpctgan_eps8_seed0 | Spearman (num-num) | 229 | 0.4166 | 0.3276 | 0.262 | 53 | `lab_results_validSerumCreatinine_value_last|lab_results_valideGFR_value_last` (-0.9109 -> 0.5748) |
| dpctgan_eps8_seed0 | Cramer's V (cat-cat) | 8911 | 0.0979 | 0.0424 | 0.7875 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.0003) |
| dpctgan_eps8_seed0 | corr-ratio (num-cat) | 4136 | 0.0455 | 0.0242 | 0.8772 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9497 -> 0.029) |
| gaussian_copula_seed0 | Spearman (num-num) | 973 | 0.1068 | 0.0745 | 0.6177 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> -0.0004) |
| gaussian_copula_seed0 | Cramer's V (cat-cat) | 11175 | 0.0667 | 0.025 | 0.8514 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.0115) |
| gaussian_copula_seed0 | corr-ratio (num-cat) | 8460 | 0.0376 | 0.0194 | 0.9176 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.2435) |
| gaussian_copula_seed1 | Spearman (num-num) | 996 | 0.1109 | 0.0724 | 0.6235 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0078) |
| gaussian_copula_seed1 | Cramer's V (cat-cat) | 11175 | 0.0666 | 0.025 | 0.854 | 0 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0428) |
| gaussian_copula_seed1 | corr-ratio (num-cat) | 8648 | 0.0385 | 0.0197 | 0.9127 | 1 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.2272) |
| gaussian_copula_seed2 | Spearman (num-num) | 976 | 0.1106 | 0.0733 | 0.6291 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0042) |
| gaussian_copula_seed2 | Cramer's V (cat-cat) | 11175 | 0.0665 | 0.0248 | 0.8545 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0202) |
| gaussian_copula_seed2 | corr-ratio (num-cat) | 8460 | 0.0381 | 0.02 | 0.915 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_HF_Disease_f5a_w1a_first` (0.8476 -> 0.0454) |
| mst_eps0p5_seed0 | Spearman (num-num) | 1726 | 0.2566 | 0.196 | 0.3024 | 109 | `lab_results_hdl_value_last|lab_results_hdl_value_first` (0.9946 -> -0.9017) |
| mst_eps0p5_seed0 | Cramer's V (cat-cat) | 10440 | 0.1761 | 0.1275 | 0.415 | 547 | `cause_of_death_isCV_f5a_w1mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first` (1.0 -> 0.0167) |
| mst_eps0p5_seed0 | corr-ratio (num-cat) | 11468 | 0.1354 | 0.068 | 0.5786 | 577 | `lab_results_ldl_value_first|conditions_aidshiv` (0.0112 -> 0.9006) |
| mst_eps10_seed0 | Spearman (num-num) | 1752 | 0.2705 | 0.2308 | 0.2317 | 147 | `echocardiographs_lvef_pET_last|encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` (-0.1394 -> 0.8839) |
| mst_eps10_seed0 | Cramer's V (cat-cat) | 11175 | 0.2344 | 0.1783 | 0.3413 | 1629 | `encounter_primary_reason_CV_Disease_f5a_w1mo_first|conditions_ap` (0.0198 -> 0.9716) |
| mst_eps10_seed0 | corr-ratio (num-cat) | 11468 | 0.1619 | 0.0981 | 0.504 | 968 | `vital_signs_heartRate_value_last|med_arni_history` (0.0262 -> 0.8988) |
| mst_eps15_seed0 | Spearman (num-num) | 1769 | 0.2834 | 0.242 | 0.2052 | 165 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> -0.0112) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 11175 | 0.2562 | 0.2124 | 0.2626 | 1851 | `med_arni_history|conditions_tia` (0.0117 -> 0.9755) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 11468 | 0.173 | 0.1066 | 0.4871 | 1102 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|cause_of_death_isAllCause_f5a_w7d_first` (0.0205 -> 0.8226) |
| mst_eps15_seed1 | Spearman (num-num) | 1763 | 0.2879 | 0.2524 | 0.2008 | 177 | `lab_results_crpNonHs_value_first|lab_results_albuminBS_value_first` (-0.4598 -> 0.5707) |
| mst_eps15_seed1 | Cramer's V (cat-cat) | 11175 | 0.2586 | 0.2105 | 0.2779 | 1897 | `med_arb|med_inotropes_history` (0.0077 -> 0.9521) |
| mst_eps15_seed1 | corr-ratio (num-cat) | 11468 | 0.1786 | 0.1113 | 0.4808 | 1190 | `electrocardiographs_ecg_qrs_duration_pET_first|med_insulins` (0.0045 -> 0.815) |
| mst_eps15_seed2 | Spearman (num-num) | 1753 | 0.2965 | 0.264 | 0.1934 | 183 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> -0.0139) |
| mst_eps15_seed2 | Cramer's V (cat-cat) | 11175 | 0.2448 | 0.1949 | 0.2969 | 1664 | `med_arb|med_inotropes_history` (0.0077 -> 0.9752) |
| mst_eps15_seed2 | corr-ratio (num-cat) | 11468 | 0.1808 | 0.118 | 0.469 | 1144 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_renal_complications_f5a_w5a_first` (0.0 -> 0.899) |
| mst_eps1_seed0 | Spearman (num-num) | 1755 | 0.2769 | 0.2318 | 0.2587 | 130 | `lab_results_hdl_value_last|lab_results_hdl_value_first` (0.9946 -> -0.7837) |
| mst_eps1_seed0 | Cramer's V (cat-cat) | 11175 | 0.3278 | 0.3114 | 0.1625 | 2809 | `encounter_primary_reason_renal_complications_f5a_w7d_first|med_rdoad_syst_history` (0.0141 -> 0.9558) |
| mst_eps1_seed0 | corr-ratio (num-cat) | 11468 | 0.1826 | 0.1139 | 0.4745 | 997 | `lab_results_hdl_value_last|med_thrombolytic` (0.0127 -> 0.9974) |
| mst_eps20_seed0 | Spearman (num-num) | 1735 | 0.3103 | 0.2718 | 0.1879 | 221 | `vital_signs_weight_value_p6mo_first|lab_results_hdl_value_first` (-0.2677 -> 0.7762) |
| mst_eps20_seed0 | Cramer's V (cat-cat) | 11175 | 0.2482 | 0.199 | 0.2975 | 1716 | `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first|conditions_ap` (0.0198 -> 0.9836) |
| mst_eps20_seed0 | corr-ratio (num-cat) | 11468 | 0.1785 | 0.111 | 0.4769 | 1204 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|conditions_mc` (0.0263 -> 0.883) |
| mst_eps5_seed0 | Spearman (num-num) | 1684 | 0.2704 | 0.2374 | 0.2369 | 115 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> -0.0474) |
| mst_eps5_seed0 | Cramer's V (cat-cat) | 11175 | 0.2769 | 0.2357 | 0.247 | 2152 | `encounter_primary_reason_renal_complications_f5a_w7d_first|med_arni` (0.0221 -> 0.9723) |
| mst_eps5_seed0 | corr-ratio (num-cat) | 11280 | 0.1613 | 0.0976 | 0.5048 | 830 | `conditions_heartFailure_timeFromEarliest_first|encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (0.0238 -> 0.8318) |
| mst_eps8_seed0 | Spearman (num-num) | 1770 | 0.2653 | 0.234 | 0.1972 | 107 | `lab_results_tropTnHs_value_first|encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first` (-0.1225 -> 0.9657) |
| mst_eps8_seed0 | Cramer's V (cat-cat) | 11175 | 0.2362 | 0.205 | 0.289 | 1313 | `med_arb|med_inotropes_history` (0.0077 -> 0.9723) |
| mst_eps8_seed0 | corr-ratio (num-cat) | 11468 | 0.1554 | 0.1043 | 0.4918 | 614 | `lab_results_tropTHs_value_last|med_rdoad_syst_history` (0.0102 -> 0.9424) |
| patectgan_eps15_seed0 | Spearman (num-num) | 1611 | 0.1152 | 0.0862 | 0.5487 | 1 | `lab_results_triGly_value_last|lab_results_hdl_value_first` (-0.2263 -> 0.5686) |
| patectgan_eps15_seed0 | Cramer's V (cat-cat) | 11175 | 0.0553 | 0.0243 | 0.9121 | 0 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first` (1.0 -> 0.0007) |
| patectgan_eps15_seed0 | corr-ratio (num-cat) | 11280 | 0.0387 | 0.0234 | 0.9239 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_renal_complications_f5a_w1a_first` (0.855 -> 0.0235) |
| patectgan_eps1_seed0 | Spearman (num-num) | 987 | 0.1329 | 0.0891 | 0.5471 | 0 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0811) |
| patectgan_eps1_seed0 | Cramer's V (cat-cat) | 11175 | 0.081 | 0.0266 | 0.8304 | 0 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isAllCause_f5a_w1mo_first` (1.0 -> 0.0053) |
| patectgan_eps1_seed0 | corr-ratio (num-cat) | 8648 | 0.0422 | 0.0209 | 0.8905 | 1 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0441) |
| patectgan_eps5_seed0 | Spearman (num-num) | 1415 | 0.1477 | 0.1152 | 0.4516 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0768) |
| patectgan_eps5_seed0 | Cramer's V (cat-cat) | 10440 | 0.0763 | 0.0287 | 0.8676 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.0003) |
| patectgan_eps5_seed0 | corr-ratio (num-cat) | 10528 | 0.0425 | 0.0263 | 0.8935 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9497 -> 0.1235) |
| tvae_cap256_seed0 | Spearman (num-num) | 1657 | 0.0829 | 0.0588 | 0.7109 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.989 -> 0.1616) |
| tvae_cap256_seed0 | Cramer's V (cat-cat) | 8778 | 0.0481 | 0.0278 | 0.8682 | 1 | `conditions_ap|conditions_dysl` (0.1435 -> 0.6669) |
| tvae_cap256_seed0 | corr-ratio (num-cat) | 11468 | 0.0376 | 0.02 | 0.9098 | 8 | `smoking_status_smoker_startTime_count|med_antiinfl_history` (0.0372 -> 1.0) |
| tvae_ep1000_seed0 | Spearman (num-num) | 1600 | 0.0815 | 0.0583 | 0.7181 | 0 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> 0.1462) |
| tvae_ep1000_seed0 | Cramer's V (cat-cat) | 8515 | 0.0482 | 0.0284 | 0.8644 | 0 | `conditions_ap|conditions_dysl` (0.1435 -> 0.6733) |
| tvae_ep1000_seed0 | corr-ratio (num-cat) | 11092 | 0.039 | 0.0225 | 0.9026 | 9 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (0.8193 -> 0.0) |
| tvae_ind_seed0 | Spearman (num-num) | 1671 | 0.0895 | 0.0684 | 0.6613 | 4 | `lab_results_tropTHs_value_first|lab_results_tropTnHs_value_first` (1.0 -> -0.2047) |
| tvae_ind_seed0 | Cramer's V (cat-cat) | 8128 | 0.0499 | 0.0296 | 0.862 | 0 | `conditions_ap|conditions_dysl` (0.1435 -> 0.6703) |
| tvae_ind_seed0 | corr-ratio (num-cat) | 11280 | 0.042 | 0.0234 | 0.8903 | 6 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (0.8082 -> 0.0581) |
| tvae_qt_seed0 | Spearman (num-num) | 1650 | 0.0948 | 0.0765 | 0.6279 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.1215) |
| tvae_qt_seed0 | Cramer's V (cat-cat) | 8001 | 0.0574 | 0.0325 | 0.8476 | 5 | `conditions_ap|conditions_dysl` (0.1435 -> 0.8139) |
| tvae_qt_seed0 | corr-ratio (num-cat) | 11280 | 0.0431 | 0.0251 | 0.8755 | 2 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (0.8082 -> 0.1569) |
| tvae_qt_seed1 | Spearman (num-num) | 1652 | 0.098 | 0.0743 | 0.6138 | 0 | `lab_results_tropTnHs_value_last|lab_results_tropTnHs_value_first` (0.8494 -> -0.0226) |
| tvae_qt_seed1 | Cramer's V (cat-cat) | 8256 | 0.0585 | 0.0311 | 0.8273 | 12 | `conditions_dep|conditions_ibd` (0.007 -> 1.0) |
| tvae_qt_seed1 | corr-ratio (num-cat) | 11280 | 0.0461 | 0.0254 | 0.8598 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_renal_complications_f5a_w3a_first` (0.8082 -> 0.0) |
| tvae_qt_seed2 | Spearman (num-num) | 1663 | 0.1003 | 0.081 | 0.5971 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.0562) |
| tvae_qt_seed2 | Cramer's V (cat-cat) | 8001 | 0.0582 | 0.0338 | 0.8244 | 4 | `conditions_ap|conditions_dysl` (0.1435 -> 0.7361) |
| tvae_qt_seed2 | corr-ratio (num-cat) | 11280 | 0.0441 | 0.0251 | 0.8694 | 1 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6666 -> 0.0) |
| tvae_seed0 | Spearman (num-num) | 1592 | 0.0861 | 0.0607 | 0.7067 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> -0.0174) |
| tvae_seed0 | Cramer's V (cat-cat) | 8128 | 0.0595 | 0.0352 | 0.8204 | 2 | `conditions_ap|conditions_dysl` (0.1435 -> 0.6727) |
| tvae_seed0 | corr-ratio (num-cat) | 11092 | 0.0414 | 0.0227 | 0.8881 | 10 | `lab_results_creatUS_value_last|encounter_primary_reason_HF_Disease_f5a_w5a_first` (0.1061 -> 0.8403) |
| tvae_seed1 | Spearman (num-num) | 1610 | 0.0817 | 0.0569 | 0.7093 | 1 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.989 -> 0.1661) |
| tvae_seed1 | Cramer's V (cat-cat) | 8515 | 0.0624 | 0.0373 | 0.8078 | 4 | `med_acei_history|med_arb_history` (0.007 -> 0.6323) |
| tvae_seed1 | corr-ratio (num-cat) | 11092 | 0.041 | 0.0225 | 0.8945 | 13 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (0.8082 -> 0.0) |
| tvae_seed2 | Spearman (num-num) | 1549 | 0.0827 | 0.0588 | 0.7069 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.0484) |
| tvae_seed2 | Cramer's V (cat-cat) | 8256 | 0.0565 | 0.031 | 0.831 | 1 | `conditions_ap|conditions_dysl` (0.1435 -> 0.7761) |
| tvae_seed2 | corr-ratio (num-cat) | 10904 | 0.0396 | 0.0213 | 0.8993 | 8 | `nyha_nyha_pET|med_arb` (0.0696 -> 0.7599) |

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
- `encounters_lengthOfStay`: KS=0.7531, W/std=0.6596, mean 10.5818 -> 17.7345, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.6775, W/std=0.6621, mean 43.6592 -> 81.3096, missing 11% -> 11%
- `lab_results_crpNonHs_value_first`: KS=0.6207, W/std=0.3029, mean 45.8171 -> 47.9764, missing 11% -> 11%
- `encounters_numOfPreviousHFStays_count`: KS=0.5443, W/std=0.2734, mean 51.846 -> 64.9459, missing 0% -> 0%
- `lab_results_ntProBnp_value_last`: KS=0.4938, W/std=0.1785, mean 10078.8465 -> 11295.4201, missing 19% -> 19%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0838, 10 -> 10 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0531, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0401, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0364, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0196, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps1_seed0]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=0.7531, W/std=0.7346, mean 10.5818 -> 20.1782, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.6775, W/std=0.4822, mean 43.6592 -> 63.448, missing 11% -> 11%
- `lab_results_crpNonHs_value_first`: KS=0.6207, W/std=0.2818, mean 45.8171 -> 61.0106, missing 11% -> 12%
- `encounters_numOfPreviousHFStays_count`: KS=0.5443, W/std=0.397, mean 51.846 -> 74.3373, missing 0% -> 0%
- `lab_results_ntProBnp_value_last`: KS=0.4938, W/std=0.1887, mean 10078.8465 -> 10888.653, missing 19% -> 21%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0858, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0523, 10 -> 10 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0452, 5 -> 4 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0386, 7 -> 7 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w3a_first`: TVD=0.0185, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ctgan_qt_seed0]

Worst numeric columns (by KS):
- `lab_results_ldl_value_last`: KS=0.7413, W/std=2.5405, mean 2.1208 -> 4.6898, missing 91% -> 94%
- `lab_results_cholTot_value_first`: KS=0.6763, W/std=2.1681, mean 3.9591 -> 6.4953, missing 89% -> 89%
- `lab_results_cholTot_value_last`: KS=0.6223, W/std=1.8828, mean 3.9492 -> 6.1691, missing 89% -> 83%
- `lab_results_hdl_value_first`: KS=0.4969, W/std=1.5781, mean 1.1873 -> 1.8931, missing 91% -> 85%
- `lab_results_triGly_value_first`: KS=0.4918, W/std=2.1787, mean 1.5982 -> 4.6911, missing 90% -> 87%
Worst categorical columns (by TVD):
- `med_mra`: TVD=0.254, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2534, 7 -> 7 categories, missing 0% -> 0%
- `med_ll_history`: TVD=0.2381, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.2335, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.2327, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed0]

Worst numeric columns (by KS):
- `lab_results_tropTHs_value_last`: KS=0.6847, W/std=0.3316, mean 0.4652 -> 1.0607, missing 63% -> 56%
- `echocardiographs_lvef_pET_last`: KS=0.6492, W/std=1.818, mean 40.6716 -> 76.0297, missing 83% -> 90%
- `lab_results_tropTnHs_value_last`: KS=0.6124, W/std=0.5346, mean 281.7666 -> 694.2468, missing 88% -> 92%
- `lab_results_ferritin_value_first`: KS=0.602, W/std=0.4291, mean 561.1328 -> 1092.2969, missing 80% -> 74%
- `echocardiographs_lvef_pET_first`: KS=0.5964, W/std=1.7063, mean 40.9911 -> 9.912, missing 82% -> 92%
Worst categorical columns (by TVD):
- `med_acei_history`: TVD=0.3281, 2 -> 2 categories, missing 0% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.2869, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.2443, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w1a_first`: TVD=0.2409, 2 -> 2 categories, missing 0% -> 0%
- `med_insulins`: TVD=0.2392, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed1]

Worst numeric columns (by KS):
- `lab_results_tropTHs_value_last`: KS=0.7126, W/std=0.4801, mean 0.4652 -> 1.517, missing 63% -> 81%
- `lab_results_sodium_value_first`: KS=0.6322, W/std=1.4343, mean 137.0846 -> 144.2849, missing 4% -> 7%
- `lab_results_tropTnHs_value_last`: KS=0.6255, W/std=0.4769, mean 281.7666 -> 629.3572, missing 88% -> 85%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.6235, W/std=1.8271, mean 471.831 -> 380.6963, missing 49% -> 38%
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.5724, W/std=0.5173, mean 82.6105 -> 129.1957, missing 95% -> 97%
Worst categorical columns (by TVD):
- `med_rdoad`: TVD=0.2972, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.2966, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.2719, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_plat_history`: TVD=0.2585, 2 -> 2 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.2335, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed2]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_last`: KS=0.7336, W/std=0.7409, mean 281.7666 -> 880.8121, missing 88% -> 97%
- `lab_results_tropTHs_value_first`: KS=0.6502, W/std=0.4395, mean 0.2171 -> 0.5141, missing 63% -> 41%
- `lab_results_tropTnHs_value_first`: KS=0.5756, W/std=0.586, mean 212.254 -> 475.604, missing 88% -> 97%
- `lab_results_potassium_value_last`: KS=0.5726, W/std=1.4326, mean 4.1751 -> 3.34, missing 4% -> 8%
- `lab_results_tropTHs_value_last`: KS=0.5679, W/std=0.2706, mean 0.4652 -> 0.9749, missing 63% -> 98%
Worst categorical columns (by TVD):
- `med_anti_plat`: TVD=0.3386, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.3125, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.231, 2 -> 2 categories, missing 0% -> 0%
- `med_vasodil_history`: TVD=0.229, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.2179, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[ddpm_g_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_last`: KS=0.9997, W/std=6.5374, mean 77.3464 -> 207.0, missing 15% -> 44%
- `vital_signs_systolicBp_value_first`: KS=0.9997, W/std=4.3831, mean 130.3643 -> 257.0, missing 5% -> 61%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.6299, mean 121.9202 -> 209.482, missing 3% -> 67%
- `lab_results_crpNonHs_value_last`: KS=0.9997, W/std=9.6461, mean 43.6592 -> 629.7, missing 11% -> 61%
- `lab_results_crpNonHs_value_first`: KS=0.9997, W/std=6.9095, mean 45.8171 -> 542.0, missing 11% -> 56%
Worst categorical columns (by TVD):
- `med_thrombolytic`: TVD=0.8009, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hypothyroid`: TVD=0.6821, 2 -> 2 categories, missing 0% -> 0%
- `conditions_dem`: TVD=0.6713, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_smoker_last`: TVD=0.6489, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w3mo_first`: TVD=0.6472, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=0.9997, W/std=6.5386, mean 77.4079 -> 207.0, missing 8% -> 43%
- `vital_signs_weight_value_last`: KS=0.9997, W/std=6.5374, mean 77.3464 -> 207.0, missing 15% -> 44%
- `vital_signs_systolicBp_value_last`: KS=0.9997, W/std=4.4648, mean 121.0177 -> 221.0, missing 5% -> 53%
- `lab_results_crpNonHs_value_last`: KS=0.9997, W/std=9.6461, mean 43.6592 -> 629.7, missing 11% -> 61%
- `lab_results_potassium_value_first`: KS=0.9997, W/std=6.5672, mean 4.2915 -> 8.8, missing 4% -> 57%
Worst categorical columns (by TVD):
- `med_thrombolytic`: TVD=0.8006, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_smoker_last`: TVD=0.733, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1mo_first`: TVD=0.7085, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.706, 3 -> 3 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.7045, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed1]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=0.9997, W/std=6.5386, mean 77.4079 -> 207.0, missing 8% -> 36%
- `vital_signs_weight_value_p6mo_first`: KS=0.9997, W/std=7.9548, mean 79.6287 -> 240.0, missing 8% -> 36%
- `vital_signs_weight_value_last`: KS=0.9997, W/std=6.5374, mean 77.3464 -> 207.0, missing 15% -> 55%
- `vital_signs_systolicBp_value_last`: KS=0.9997, W/std=4.4648, mean 121.0177 -> 221.0, missing 5% -> 40%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.6299, mean 121.9202 -> 209.482, missing 3% -> 46%
Worst categorical columns (by TVD):
- `cause_of_death_isCV_f5a_w7d_first`: TVD=0.7563, 3 -> 3 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.7355, 5 -> 5 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.7151, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.7023, 3 -> 3 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.654, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed2]

Worst numeric columns (by KS):
- `lab_results_potassium_value_last`: KS=0.9997, W/std=5.361, mean 4.1751 -> 7.3, missing 4% -> 47%
- `lab_results_validSerumCreatinine_value_last`: KS=0.9997, W/std=2.3007, mean 12.1376 -> 22.5748, missing 10% -> 47%
- `eGFR_2021_ckd_epi_creatinine`: KS=0.9997, W/std=4.9822, mean 64.4886 -> 193.8265, missing 10% -> 38%
- `vital_signs_bmi_value_last`: KS=0.9995, W/std=8.1102, mean 26.9378 -> 79.5802, missing 46% -> 47%
- `vital_signs_heartRate_value_first`: KS=0.9995, W/std=6.6818, mean 112.7017 -> 223.0, missing 48% -> 44%
Worst categorical columns (by TVD):
- `cause_of_death_isAllCause_f5a_w1mo_first`: TVD=0.8165, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w7d_first`: TVD=0.7756, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.7582, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w1mo_first`: TVD=0.7452, 3 -> 3 categories, missing 0% -> 0%
- `smoking_status_smoker_last`: TVD=0.7392, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps10_seed0]

Worst numeric columns (by KS):
- `lab_results_potassium_value_first`: KS=0.9997, W/std=6.567, mean 4.2915 -> 8.7999, missing 4% -> 0%
- `lab_results_sodium_value_first`: KS=0.9997, W/std=4.3653, mean 137.0846 -> 158.9994, missing 4% -> 0%
- `vital_signs_height_value_last`: KS=0.9995, W/std=3.4312, mean 171.4561 -> 206.9008, missing 44% -> 0%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.9994, W/std=5.9144, mean 471.831 -> 766.8441, missing 49% -> 0%
- `echocardiographs_lvef_pET_first`: KS=0.9984, W/std=4.4492, mean 40.9911 -> -40.2001, missing 82% -> 99%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.8835, 6 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8744, 10 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.8563, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.8278, 3 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8273, 7 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=0.9997, W/std=2.3963, mean 70.9054 -> 103.9993, missing 0% -> 0%
- `vital_signs_heartRate_value_last`: KS=0.9995, W/std=8.8535, mean 109.7305 -> 238.1404, missing 48% -> 0%
- `lab_results_albuminBS_value_last`: KS=0.9993, W/std=2.9172, mean 29.92 -> 49.9998, missing 57% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9992, W/std=25.6364, mean 0.4652 -> 64.398, missing 63% -> 0%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.9986, W/std=5.3776, mean 471.831 -> 740.0633, missing 49% -> 0%
Worst categorical columns (by TVD):
- `conditions_aidshiv`: TVD=0.9864, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_smoker_last`: TVD=0.9705, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.9523, 7 -> 4 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w1mo_first`: TVD=0.9193, 2 -> 2 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.917, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed1]

Worst numeric columns (by KS):
- `lab_results_potassium_value_first`: KS=0.9997, W/std=6.5626, mean 4.2915 -> 8.7969, missing 4% -> 0%
- `vital_signs_height_value_p1a_avg`: KS=0.9996, W/std=3.4631, mean 171.0332 -> 206.9996, missing 21% -> 0%
- `vital_signs_heartRate_value_last`: KS=0.9995, W/std=9.9337, mean 109.7305 -> 253.8132, missing 48% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=0.9994, W/std=6.4024, mean 77.4079 -> 204.3, missing 8% -> 0%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.9994, W/std=5.8748, mean 471.831 -> 764.8703, missing 49% -> 0%
Worst categorical columns (by TVD):
- `conditions_dep`: TVD=0.9747, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.9537, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.9344, 3 -> 2 categories, missing 0% -> 0%
- `conditions_tia`: TVD=0.9256, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.9037, 7 -> 6 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed2]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=0.9997, W/std=2.372, mean 70.9054 -> 103.6628, missing 0% -> 0%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.6125, mean 121.9202 -> 209.0623, missing 3% -> 0%
- `lab_results_sodium_value_first`: KS=0.9997, W/std=4.3583, mean 137.0846 -> 158.9639, missing 4% -> 0%
- `lab_results_validSerumCreatinine_value_first`: KS=0.9994, W/std=2.2922, mean 12.3144 -> 22.5746, missing 10% -> 0%
- `vital_signs_systolicBp_value_first`: KS=0.9985, W/std=4.3096, mean 130.3643 -> 254.8772, missing 5% -> 0%
Worst categorical columns (by TVD):
- `conditions_pericardial`: TVD=0.9551, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first`: TVD=0.8639, 2 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.854, 10 -> 6 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.8503, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first`: TVD=0.8483, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps1_seed0]

Worst numeric columns (by KS):
- `lab_results_validSerumCreatinine_value_last`: KS=1.0, W/std=2.4453, mean 12.1376 -> 1.0445, missing 10% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=0.9997, W/std=5.5247, mean 69.0778 -> 143.9954, missing 5% -> 0%
- `lab_results_hemoglobin_value_last`: KS=0.9997, W/std=3.8736, mean 120.0203 -> 208.528, missing 3% -> 0%
- `lab_results_sodium_value_first`: KS=0.9997, W/std=4.3649, mean 137.0846 -> 158.9974, missing 4% -> 0%
- `vital_signs_height_value_p1a_avg`: KS=0.9996, W/std=3.4374, mean 171.0332 -> 206.732, missing 21% -> 0%
Worst categorical columns (by TVD):
- `cause_of_death_isCV_f5a_w1mo_first`: TVD=0.9926, 3 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3mo_first`: TVD=0.9858, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.9486, 3 -> 3 categories, missing 0% -> 0%
- `med_diuretics`: TVD=0.9241, 2 -> 2 categories, missing 0% -> 0%
- `med_antiarrhytmic_history`: TVD=0.9006, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps20_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_last`: KS=0.9997, W/std=6.5374, mean 77.3464 -> 206.9983, missing 15% -> 0%
- `vital_signs_diastolicBp_value_first`: KS=0.9997, W/std=6.7316, mean 75.7013 -> 199.8302, missing 5% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.9997, W/std=9.6429, mean 43.6592 -> 629.5102, missing 11% -> 0%
- `lab_results_potassium_value_last`: KS=0.9997, W/std=5.3601, mean 4.1751 -> 7.2995, missing 4% -> 0%
- `lab_results_sodium_value_last`: KS=0.9997, W/std=4.8214, mean 137.8251 -> 159.9996, missing 4% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.9563, 3 -> 2 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.9514, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.9361, 7 -> 6 categories, missing 0% -> 0%
- `conditions_ld`: TVD=0.9247, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hypothyroid`: TVD=0.9227, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps5_seed0]

Worst numeric columns (by KS):
- `vital_signs_diastolicBp_value_first`: KS=0.9997, W/std=6.7813, mean 75.7013 -> 200.7468, missing 5% -> 0%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.6298, mean 121.9202 -> 209.4786, missing 3% -> 0%
- `lab_results_sodium_value_last`: KS=0.9997, W/std=4.8213, mean 137.8251 -> 159.999, missing 4% -> 0%
- `lab_results_sodium_value_first`: KS=0.9997, W/std=4.3145, mean 137.0846 -> 158.7441, missing 4% -> 0%
- `vital_signs_height_value_p1a_avg`: KS=0.9996, W/std=3.461, mean 171.0332 -> 206.9777, missing 21% -> 0%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.9446, 5 -> 4 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w1mo_first`: TVD=0.9241, 3 -> 3 categories, missing 0% -> 0%
- `med_antiarrhytmic_history`: TVD=0.9145, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9068, 10 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`: TVD=0.9068, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps8_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=0.9997, W/std=2.3958, mean 70.9054 -> 103.9917, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9997, W/std=7.4532, mean 51.846 -> 580.2233, missing 0% -> 0%
- `vital_signs_systolicBp_value_last`: KS=0.9997, W/std=4.4647, mean 121.0177 -> 220.9966, missing 5% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=0.9997, W/std=5.5248, mean 69.0778 -> 143.9971, missing 5% -> 0%
- `lab_results_sodium_value_first`: KS=0.9997, W/std=4.3654, mean 137.0846 -> 158.9996, missing 4% -> 0%
Worst categorical columns (by TVD):
- `med_thrombolytic`: TVD=0.983, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.9494, 3 -> 2 categories, missing 0% -> 0%
- `conditions_hypothyroid`: TVD=0.9014, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.875, 10 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8594, 7 -> 4 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed0]

Worst numeric columns (by KS):
- `lab_results_hdl_value_first`: KS=0.9906, W/std=2.1127, mean 1.1873 -> 0.2425, missing 91% -> 100%
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=3.5091, mean 0.0804 -> 3.0497, missing 0% -> 0%
- `lab_results_creatUS_value_first`: KS=0.9288, W/std=1.1478, mean 754.8314 -> 147.4408, missing 90% -> 100%
- `echocardiographs_lvef_pET_first`: KS=0.8986, W/std=3.5865, mean 40.9911 -> -24.4573, missing 82% -> 97%
- `lab_results_tropTHs_value_last`: KS=0.8585, W/std=1.591, mean 0.4652 -> 4.2487, missing 63% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4401, 10 -> 9 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0267, 6 -> 6 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.0241, 3 -> 3 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.0207, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.0207, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed1]

Worst numeric columns (by KS):
- `lab_results_ldl_value_first`: KS=0.9937, W/std=1.933, mean 2.1313 -> 0.1964, missing 91% -> 100%
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=3.5218, mean 0.0804 -> 3.0611, missing 0% -> 0%
- `lab_results_hdl_value_first`: KS=0.9781, W/std=1.7686, mean 1.1873 -> 0.3987, missing 91% -> 100%
- `echocardiographs_lvef_pET_first`: KS=0.9569, W/std=3.6837, mean 40.9911 -> -26.2312, missing 82% -> 97%
- `lab_results_creatUS_value_first`: KS=0.9402, W/std=1.2155, mean 754.8314 -> 111.2266, missing 90% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4435, 10 -> 9 categories, missing 0% -> 0%
- `med_ll_history`: TVD=0.0216, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0205, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0196, 6 -> 6 categories, missing 0% -> 0%
- `med_platelet_history`: TVD=0.0193, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed2]

Worst numeric columns (by KS):
- `lab_results_hdl_value_first`: KS=0.9938, W/std=2.2037, mean 1.1873 -> 0.2015, missing 91% -> 100%
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=3.4054, mean 0.0804 -> 2.9619, missing 0% -> 0%
- `lab_results_ferritin_value_last`: KS=0.9341, W/std=0.6011, mean 523.1277 -> 1295.371, missing 80% -> 100%
- `echocardiographs_lvef_pET_first`: KS=0.9092, W/std=3.5778, mean 40.9911 -> -24.2982, missing 82% -> 97%
- `lab_results_cholTot_value_first`: KS=0.8622, W/std=1.9996, mean 3.9591 -> 1.6192, missing 89% -> 99%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4287, 10 -> 9 categories, missing 0% -> 0%
- `med_mra`: TVD=0.0233, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.0224, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.0224, 3 -> 3 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.0219, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps0p5_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=2.0059, mean 0.0804 -> 1.7759, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.9592, mean 0.4652 -> 5.345, missing 63% -> 57%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=8.0161, mean 523.1277 -> 15840.852, missing 80% -> 63%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=5.7203, mean 561.1328 -> 12827.7585, missing 80% -> 79%
- `lab_results_triGly_value_last`: KS=0.9638, W/std=6.4788, mean 1.5467 -> 8.8586, missing 90% -> 82%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.1895, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.1659, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1562, 10 -> 9 categories, missing 0% -> 0%
- `conditions_devices`: TVD=0.156, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1523, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[mst_eps10_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.172, mean 0.0804 -> 1.0709, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.5974, mean 0.4652 -> 4.3963, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.3151, mean 523.1277 -> 3027.739, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.254, mean 561.1328 -> 3245.6978, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8528, mean 0.2171 -> 0.9068, missing 63% -> 63%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0105, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0099, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0082, 7 -> 7 categories, missing 0% -> 0%
- `med_potassium_binders`: TVD=0.0074, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w1a_first`: TVD=0.0071, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1783, mean 0.0804 -> 1.0774, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.5832, mean 0.4652 -> 4.3218, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2218, mean 523.1277 -> 2809.0544, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.092, mean 561.1328 -> 2874.2554, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8905, mean 0.2171 -> 0.9464, missing 63% -> 63%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0091, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0065, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.006, 10 -> 10 categories, missing 0% -> 0%
- `med_antiinfl`: TVD=0.006, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0048, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed1]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1928, mean 0.0804 -> 1.0898, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.6192, mean 0.4652 -> 4.4688, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2604, mean 523.1277 -> 2882.1347, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.0912, mean 561.1328 -> 2754.2231, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8394, mean 0.2171 -> 0.8881, missing 63% -> 63%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0065, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0062, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.006, 10 -> 10 categories, missing 0% -> 0%
- `conditions_osa`: TVD=0.0054, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.0051, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed2]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1655, mean 0.0804 -> 1.0661, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.7225, mean 0.4652 -> 4.7545, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2148, mean 523.1277 -> 2775.2319, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1327, mean 561.1328 -> 2968.7054, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8505, mean 0.2171 -> 0.8248, missing 63% -> 63%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w7d_first`: TVD=0.0071, 3 -> 3 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0057, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0057, 7 -> 7 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.0057, 2 -> 2 categories, missing 0% -> 0%
- `med_potassium_binders`: TVD=0.0054, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps1_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=2.498, mean 0.0804 -> 2.1953, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=2.6532, mean 0.4652 -> 7.0757, missing 63% -> 66%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2246, mean 523.1277 -> 2781.1457, missing 80% -> 79%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=2.3385, mean 561.1328 -> 5577.4695, missing 80% -> 78%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=1.5272, mean 0.2171 -> 1.5112, missing 63% -> 64%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1131, 10 -> 10 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.1011, 5 -> 5 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.0739, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0639, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0611, 7 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps20_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1548, mean 0.0804 -> 1.0558, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.567, mean 0.4652 -> 4.3205, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.217, mean 523.1277 -> 2716.736, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1195, mean 561.1328 -> 2656.0708, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8672, mean 0.2171 -> 0.9146, missing 63% -> 63%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0077, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0057, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w1a_first`: TVD=0.0043, 3 -> 3 categories, missing 0% -> 0%
- `conditions_hypothyroid`: TVD=0.0037, 2 -> 2 categories, missing 0% -> 0%
- `conditions_myocarditis`: TVD=0.0037, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps5_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.2837, mean 0.0804 -> 1.1664, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.622, mean 0.4652 -> 4.4762, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2448, mean 523.1277 -> 2633.5, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.3395, mean 561.1328 -> 3431.2135, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8334, mean 0.2171 -> 0.8529, missing 63% -> 63%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0273, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0244, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0216, 7 -> 7 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.0182, 2 -> 2 categories, missing 0% -> 0%
- `med_inotropes`: TVD=0.0159, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps8_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1933, mean 0.0804 -> 1.0903, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.587, mean 0.4652 -> 4.3327, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.716, mean 523.1277 -> 3796.3317, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1021, mean 561.1328 -> 2693.4316, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.839, mean 0.2171 -> 0.8568, missing 63% -> 63%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0125, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.0105, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0102, 7 -> 7 categories, missing 0% -> 0%
- `med_cortico_syst_history`: TVD=0.0102, 2 -> 2 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0097, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps15_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9804, W/std=0.3438, mean 0.0804 -> 0.3617, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.8716, W/std=6.1449, mean 0.4652 -> 15.7832, missing 63% -> 81%
- `lab_results_ferritin_value_last`: KS=0.835, W/std=4.6965, mean 523.1277 -> 9492.9744, missing 80% -> 86%
- `lab_results_tropTHs_value_first`: KS=0.8179, W/std=4.2907, mean 0.2171 -> 3.8547, missing 63% -> 83%
- `lab_results_ferritin_value_first`: KS=0.7841, W/std=3.2728, mean 561.1328 -> 7578.6567, missing 80% -> 95%
Worst categorical columns (by TVD):
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.4207, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w5a_first`: TVD=0.4205, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w5a_first`: TVD=0.4168, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w5a_first`: TVD=0.4151, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w3a_first`: TVD=0.3619, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps1_seed0]

Worst numeric columns (by KS):
- `echocardiographs_lvef_pET_first`: KS=0.9984, W/std=4.1919, mean 40.9911 -> -35.5056, missing 82% -> 94%
- `echocardiographs_lvef_pET_last`: KS=0.9983, W/std=10.0815, mean 40.6716 -> -157.2786, missing 83% -> 82%
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.2814, mean 0.0804 -> 1.0991, missing 0% -> 0%
- `lab_results_hdl_value_first`: KS=0.975, W/std=1.9919, mean 1.1873 -> 0.2963, missing 91% -> 99%
- `lab_results_hdl_value_last`: KS=0.9656, W/std=1.9862, mean 1.1854 -> 0.3143, missing 91% -> 100%
Worst categorical columns (by TVD):
- `cause_of_death_isNonRenalAndNonCV_f5a_w3a_first`: TVD=0.2264, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag_history`: TVD=0.2128, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.204, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1a_first`: TVD=0.1889, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.1849, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps5_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=0.2719, mean 0.0804 -> 0.2152, missing 0% -> 0%
- `lab_results_tropTnHs_value_last`: KS=0.8444, W/std=1.1369, mean 281.7666 -> 1189.502, missing 88% -> 100%
- `lab_results_ferritin_value_last`: KS=0.8215, W/std=2.5039, mean 523.1277 -> 5244.4992, missing 80% -> 98%
- `lab_results_ferritin_value_first`: KS=0.8002, W/std=2.1433, mean 561.1328 -> 5119.4362, missing 80% -> 98%
- `echocardiographs_lvef_pET_last`: KS=0.797, W/std=5.6815, mean 40.6716 -> -70.097, missing 83% -> 77%
Worst categorical columns (by TVD):
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first`: TVD=0.6997, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.6986, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first`: TVD=0.6923, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.6824, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.6625, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[tvae_cap256_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5476, W/std=0.1235, mean 11.0621 -> 9.7598, missing 0% -> 30%
- `lab_results_tropTHs_value_last`: KS=0.467, W/std=0.163, mean 0.4652 -> 0.2821, missing 63% -> 58%
- `lab_results_creatUS_value_last`: KS=0.4469, W/std=0.5719, mean 694.4878 -> 465.2602, missing 90% -> 97%
- `lab_results_ferritin_value_first`: KS=0.3814, W/std=0.1932, mean 561.1328 -> 456.8995, missing 80% -> 83%
- `lab_results_ferritin_value_last`: KS=0.3751, W/std=0.1378, mean 523.1277 -> 491.2811, missing 80% -> 84%
Worst categorical columns (by TVD):
- `conditions_copd`: TVD=0.1733, 2 -> 2 categories, missing 0% -> 0%
- `med_bb`: TVD=0.1702, 2 -> 2 categories, missing 0% -> 0%
- `med_arb`: TVD=0.1688, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.167, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.1594, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ep1000_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5444, W/std=0.1028, mean 11.0621 -> 9.8131, missing 0% -> 37%
- `lab_results_tropTHs_value_last`: KS=0.4685, W/std=0.1192, mean 0.4652 -> 0.5362, missing 63% -> 59%
- `lab_results_ferritin_value_first`: KS=0.4517, W/std=0.2125, mean 561.1328 -> 520.1654, missing 80% -> 83%
- `lab_results_creatUS_value_last`: KS=0.4328, W/std=0.5954, mean 694.4878 -> 461.1771, missing 90% -> 99%
- `lab_results_ferritin_value_last`: KS=0.3937, W/std=0.1786, mean 523.1277 -> 443.6254, missing 80% -> 83%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.2054, 7 -> 7 categories, missing 0% -> 0%
- `med_platelet`: TVD=0.1929, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1892, 10 -> 10 categories, missing 0% -> 0%
- `med_anti_plat`: TVD=0.1864, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1827, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[tvae_ind_seed0]

Worst numeric columns (by KS):
- `lab_results_creatUS_value_last`: KS=0.4243, W/std=0.5122, mean 694.4878 -> 457.7703, missing 90% -> 91%
- `lab_results_creatUS_value_first`: KS=0.4149, W/std=0.5097, mean 754.8314 -> 517.9095, missing 90% -> 92%
- `vital_signs_heartRate_value_first`: KS=0.3906, W/std=0.5002, mean 112.7017 -> 105.0661, missing 48% -> 50%
- `vital_signs_heartRate_value_last`: KS=0.3803, W/std=0.4709, mean 109.7305 -> 103.5375, missing 48% -> 50%
- `lab_results_ferritin_value_first`: KS=0.3787, W/std=0.1878, mean 561.1328 -> 240.1496, missing 80% -> 84%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1932, 10 -> 10 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1864, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1733, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.1656, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.1625, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed0]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_last`: KS=0.4165, W/std=0.188, mean 281.7666 -> 249.3732, missing 88% -> 92%
- `lab_results_tropTnHs_value_first`: KS=0.3874, W/std=0.2144, mean 212.254 -> 185.99, missing 88% -> 92%
- `lab_results_ferritin_value_first`: KS=0.3866, W/std=0.1853, mean 561.1328 -> 414.4173, missing 80% -> 90%
- `lab_results_ferritin_value_last`: KS=0.3558, W/std=0.1834, mean 523.1277 -> 402.2464, missing 80% -> 91%
- `lab_results_creatUS_value_first`: KS=0.3439, W/std=0.5237, mean 754.8314 -> 743.8806, missing 90% -> 94%
Worst categorical columns (by TVD):
- `med_bb`: TVD=0.2483, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.2219, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.1991, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1866, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1861, 10 -> 10 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed1]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_last`: KS=0.3829, W/std=0.233, mean 281.7666 -> 126.9935, missing 88% -> 96%
- `lab_results_ferritin_value_last`: KS=0.374, W/std=0.186, mean 523.1277 -> 373.1853, missing 80% -> 95%
- `vital_signs_heartRate_value_first`: KS=0.3624, W/std=0.5353, mean 112.7017 -> 104.4866, missing 48% -> 43%
- `lab_results_ferritin_value_first`: KS=0.3532, W/std=0.1699, mean 561.1328 -> 385.6075, missing 80% -> 95%
- `lab_results_tropTnHs_value_first`: KS=0.3491, W/std=0.2756, mean 212.254 -> 100.4851, missing 88% -> 96%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.3, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2577, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.2236, 2 -> 2 categories, missing 0% -> 0%
- `med_mra`: TVD=0.2168, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.2119, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed2]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_first`: KS=0.36, W/std=0.2704, mean 212.254 -> 110.664, missing 88% -> 92%
- `lab_results_tropTnHs_value_last`: KS=0.3492, W/std=0.2246, mean 281.7666 -> 133.3844, missing 88% -> 92%
- `lab_results_creatUS_value_first`: KS=0.3348, W/std=0.5181, mean 754.8314 -> 724.2968, missing 90% -> 96%
- `lab_results_albuminBS_value_first`: KS=0.3344, W/std=0.4924, mean 31.1435 -> 28.1925, missing 57% -> 62%
- `lab_results_albuminBS_value_last`: KS=0.3225, W/std=0.4727, mean 29.92 -> 27.4514, missing 57% -> 61%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2844, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2534, 2 -> 2 categories, missing 0% -> 0%
- `med_ll`: TVD=0.2102, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.1943, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.1926, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5485, W/std=0.1227, mean 11.0621 -> 11.356, missing 0% -> 36%
- `lab_results_creatUS_value_last`: KS=0.4674, W/std=0.6229, mean 694.4878 -> 441.2814, missing 90% -> 98%
- `lab_results_tropTHs_value_last`: KS=0.4297, W/std=0.1664, mean 0.4652 -> 0.2265, missing 63% -> 67%
- `lab_results_tropTnHs_value_last`: KS=0.3853, W/std=0.2239, mean 281.7666 -> 165.6781, missing 88% -> 91%
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.3823, W/std=0.2874, mean 82.6105 -> 44.9845, missing 95% -> 96%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.306, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.1926, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1801, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1733, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.1716, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed1]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5452, W/std=0.1271, mean 11.0621 -> 12.4017, missing 0% -> 35%
- `lab_results_creatUS_value_last`: KS=0.4689, W/std=0.6365, mean 694.4878 -> 428.2142, missing 90% -> 98%
- `lab_results_tropTHs_value_last`: KS=0.4571, W/std=0.1706, mean 0.4652 -> 0.2562, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.3818, W/std=0.1372, mean 523.1277 -> 569.0884, missing 80% -> 88%
- `lab_results_tropTnHs_value_last`: KS=0.38, W/std=0.2272, mean 281.7666 -> 147.5155, missing 88% -> 94%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.269, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2011, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1866, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1761, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1707, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed2]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5457, W/std=0.0868, mean 11.0621 -> 10.6546, missing 0% -> 35%
- `lab_results_ferritin_value_first`: KS=0.511, W/std=0.2413, mean 561.1328 -> 589.0632, missing 80% -> 90%
- `lab_results_creatUS_value_last`: KS=0.4985, W/std=0.6796, mean 694.4878 -> 393.8098, missing 90% -> 99%
- `lab_results_tropTHs_value_last`: KS=0.4625, W/std=0.1702, mean 0.4652 -> 0.2517, missing 63% -> 67%
- `lab_results_ferritin_value_last`: KS=0.4509, W/std=0.1668, mean 523.1277 -> 566.38, missing 80% -> 91%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2849, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1991, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.1835, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1776, 2 -> 2 categories, missing 0% -> 0%
