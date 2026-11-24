# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 22 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 167 | 0.0007 | 0.0 | 1.0 | 0.0063 | 0.0 | 1.0 | 0.0007 |
| train vs holdout | 234 | 0.0589 | 0.0485 | 0.8939 | 0.0941 | 0.0179 | 0.9702 | 0.0121 |
| train vs synthetic[aim40_eps1_seed0] | 40 | 0.4588 | 0.4186 | 0.0476 | 0.9345 | 0.0264 | 0.9474 | 0.0105 |
| train vs synthetic[aim40_eps5_seed0] | 40 | 0.4485 | 0.4018 | 0.0476 | 0.6807 | 0.018 | 0.9474 | 0.0044 |
| train vs synthetic[aim40_eps8_seed0] | 40 | 0.4483 | 0.3929 | 0.0476 | 0.7032 | 0.0218 | 0.8947 | 0.0049 |
| train vs synthetic[aim50_eps1_seed0] | 50 | 0.492 | 0.4138 | 0.04 | 1.3528 | 0.0283 | 0.92 | 0.0113 |
| train vs synthetic[aim50_eps5_seed0] | 50 | 0.4872 | 0.4108 | 0.04 | 0.8281 | 0.0175 | 0.96 | 0.0057 |
| train vs synthetic[aim50_eps8_seed0] | 50 | 0.4864 | 0.3988 | 0.04 | 0.8051 | 0.0174 | 1.0 | 0.0039 |
| train vs synthetic[ctgan_qt_seed0] | 234 | 0.2438 | 0.2074 | 0.1818 | 0.6338 | 0.0724 | 0.5357 | 0.0843 |
| train vs synthetic[ctgan_seed0] | 234 | 0.3267 | 0.3091 | 0.0758 | 0.6368 | 0.0809 | 0.4167 | 0.1172 |
| train vs synthetic[ctgan_seed1] | 234 | 0.3081 | 0.3107 | 0.0606 | 0.5904 | 0.079 | 0.4524 | 0.1086 |
| train vs synthetic[ctgan_seed2] | 234 | 0.2854 | 0.2576 | 0.0758 | 0.5433 | 0.0717 | 0.4881 | 0.0982 |
| train vs synthetic[ddpm_g_seed0] | 234 | 0.5452 | 0.657 | 0.0303 | 2.204 | 0.2374 | 0.2262 | 0.093 |
| train vs synthetic[ddpm_seed0] | 234 | 0.5151 | 0.6472 | 0.0303 | 2.2226 | 0.0934 | 0.3869 | 0.0849 |
| train vs synthetic[ddpm_seed1] | 234 | 0.5165 | 0.6517 | 0.0303 | 2.2373 | 0.0992 | 0.369 | 0.0845 |
| train vs synthetic[ddpm_seed2] | 234 | 0.5139 | 0.6576 | 0.0455 | 2.2276 | 0.099 | 0.381 | 0.0816 |
| train vs synthetic[dpctgan_eps10_seed0] | 234 | 0.9348 | 1.0 | 0.0 | 3.2906 | 0.2644 | 0.2381 | 0.4003 |
| train vs synthetic[dpctgan_eps15_seed0] | 234 | 0.9118 | 1.0 | 0.0 | 5.9341 | 0.2818 | 0.2381 | 0.3619 |
| train vs synthetic[dpctgan_eps15_seed1] | 234 | 0.8846 | 1.0 | 0.0 | 4.0152 | 0.2782 | 0.2381 | 0.3776 |
| train vs synthetic[dpctgan_eps15_seed2] | 234 | 0.9462 | 1.0 | 0.0 | 4.5657 | 0.2735 | 0.244 | 0.3783 |
| train vs synthetic[dpctgan_eps1_seed0] | 234 | 0.9131 | 1.0 | 0.0 | 5.8584 | 0.3155 | 0.2143 | 0.4288 |
| train vs synthetic[dpctgan_eps20_seed0] | 234 | 0.9157 | 1.0 | 0.0 | 3.5442 | 0.2871 | 0.256 | 0.4068 |
| train vs synthetic[dpctgan_eps5_seed0] | 234 | 0.9032 | 1.0 | 0.0 | 3.8988 | 0.2872 | 0.2381 | 0.3401 |
| train vs synthetic[dpctgan_eps8_seed0] | 234 | 0.9469 | 1.0 | 0.0 | 4.7295 | 0.2985 | 0.2262 | 0.3474 |
| train vs synthetic[gaussian_copula_seed0] | 234 | 0.3228 | 0.31 | 0.2121 | 0.8534 | 0.0135 | 0.994 | 0.1434 |
| train vs synthetic[gaussian_copula_seed1] | 234 | 0.3303 | 0.3056 | 0.2121 | 0.8814 | 0.0144 | 0.994 | 0.139 |
| train vs synthetic[gaussian_copula_seed2] | 234 | 0.3225 | 0.295 | 0.1818 | 0.8507 | 0.0145 | 0.994 | 0.1421 |
| train vs synthetic[mst_eps10_seed0] | 234 | 0.5033 | 0.4387 | 0.0303 | 0.8742 | 0.0107 | 0.994 | 0.0122 |
| train vs synthetic[mst_eps15_seed0] | 234 | 0.5028 | 0.4187 | 0.0303 | 0.847 | 0.009 | 1.0 | 0.0102 |
| train vs synthetic[mst_eps15_seed1] | 234 | 0.5018 | 0.4297 | 0.0303 | 0.8823 | 0.0106 | 1.0 | 0.0116 |
| train vs synthetic[mst_eps15_seed2] | 234 | 0.506 | 0.4227 | 0.0303 | 0.8545 | 0.0105 | 1.0 | 0.0107 |
| train vs synthetic[mst_eps1_seed0] | 234 | 0.5536 | 0.4929 | 0.0303 | 1.553 | 0.0383 | 0.7262 | 0.0354 |
| train vs synthetic[mst_eps20_seed0] | 234 | 0.5022 | 0.4218 | 0.0303 | 0.8352 | 0.0099 | 1.0 | 0.0122 |
| train vs synthetic[mst_eps5_seed0] | 234 | 0.5042 | 0.4289 | 0.0303 | 1.0149 | 0.0132 | 0.994 | 0.0149 |
| train vs synthetic[mst_eps8_seed0] | 234 | 0.5058 | 0.4421 | 0.0303 | 0.9057 | 0.0124 | 0.9881 | 0.0148 |
| train vs synthetic[patectgan_eps15_seed0] | 234 | 0.3977 | 0.3678 | 0.0 | 1.0709 | 0.0826 | 0.4821 | 0.1145 |
| train vs synthetic[patectgan_eps1_seed0] | 234 | 0.5083 | 0.5307 | 0.0606 | 1.479 | 0.0546 | 0.5417 | 0.1983 |
| train vs synthetic[patectgan_eps5_seed0] | 234 | 0.425 | 0.4002 | 0.0 | 1.3258 | 0.1353 | 0.5238 | 0.1544 |
| train vs synthetic[tvae_cap256_seed0] | 234 | 0.1721 | 0.1572 | 0.197 | 0.2393 | 0.0492 | 0.5952 | 0.042 |
| train vs synthetic[tvae_ep1000_seed0] | 234 | 0.164 | 0.1488 | 0.3182 | 0.2253 | 0.0482 | 0.6429 | 0.0338 |
| train vs synthetic[tvae_ind_seed0] | 234 | 0.1803 | 0.1732 | 0.1061 | 0.2433 | 0.0566 | 0.5357 | 0.03 |
| train vs synthetic[tvae_qt_seed0] | 234 | 0.1496 | 0.1549 | 0.303 | 0.2595 | 0.051 | 0.5774 | 0.0194 |
| train vs synthetic[tvae_qt_seed1] | 234 | 0.1546 | 0.1437 | 0.3333 | 0.2618 | 0.0557 | 0.5238 | 0.0249 |
| train vs synthetic[tvae_qt_seed2] | 234 | 0.1502 | 0.138 | 0.303 | 0.2527 | 0.0576 | 0.5238 | 0.0325 |
| train vs synthetic[tvae_seed0] | 234 | 0.1808 | 0.1714 | 0.1061 | 0.2576 | 0.0567 | 0.5595 | 0.0442 |
| train vs synthetic[tvae_seed1] | 234 | 0.1824 | 0.1734 | 0.1515 | 0.2571 | 0.0525 | 0.625 | 0.0465 |
| train vs synthetic[tvae_seed2] | 234 | 0.1778 | 0.1674 | 0.1667 | 0.2549 | 0.0567 | 0.5595 | 0.043 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| aim | 1 | 1 | 0.492 | 0.0283 | 0.0113 |
| aim | 5 | 1 | 0.4872 | 0.0175 | 0.0057 |
| aim | 8 | 1 | 0.4864 | 0.0174 | 0.0039 |
| aim40 | 1 | 1 | 0.4588 | 0.0264 | 0.0105 |
| aim40 | 5 | 1 | 0.4485 | 0.018 | 0.0044 |
| aim40 | 8 | 1 | 0.4483 | 0.0218 | 0.0049 |
| ctgan | - | 3 | 0.3067 ± 0.0207 | 0.0772 ± 0.0049 | 0.108 ± 0.0095 |
| ctgan_qt | - | 1 | 0.2438 | 0.0724 | 0.0843 |
| ddpm | - | 3 | 0.5152 ± 0.0013 | 0.0972 ± 0.0033 | 0.0837 ± 0.0018 |
| ddpm_g | - | 1 | 0.5452 | 0.2374 | 0.093 |
| dpctgan | 1 | 1 | 0.9131 | 0.3155 | 0.4288 |
| dpctgan | 5 | 1 | 0.9032 | 0.2872 | 0.3401 |
| dpctgan | 8 | 1 | 0.9469 | 0.2985 | 0.3474 |
| dpctgan | 10 | 1 | 0.9348 | 0.2644 | 0.4003 |
| dpctgan | 15 | 3 | 0.9142 ± 0.0309 | 0.2778 ± 0.0042 | 0.3726 ± 0.0093 |
| dpctgan | 20 | 1 | 0.9157 | 0.2871 | 0.4068 |
| gaussian_copula | - | 3 | 0.3252 ± 0.0044 | 0.0141 ± 0.0006 | 0.1415 ± 0.0023 |
| mst | 1 | 1 | 0.5536 | 0.0383 | 0.0354 |
| mst | 5 | 1 | 0.5042 | 0.0132 | 0.0149 |
| mst | 8 | 1 | 0.5058 | 0.0124 | 0.0148 |
| mst | 10 | 1 | 0.5033 | 0.0107 | 0.0122 |
| mst | 15 | 3 | 0.5035 ± 0.0022 | 0.01 ± 0.0009 | 0.0108 ± 0.0007 |
| mst | 20 | 1 | 0.5022 | 0.0099 | 0.0122 |
| patectgan | 1 | 1 | 0.5083 | 0.0546 | 0.1983 |
| patectgan | 5 | 1 | 0.425 | 0.1353 | 0.1544 |
| patectgan | 15 | 1 | 0.3977 | 0.0826 | 0.1145 |
| tvae | - | 3 | 0.1803 ± 0.0023 | 0.0553 ± 0.0024 | 0.0446 ± 0.0018 |
| tvae_cap256 | - | 1 | 0.1721 | 0.0492 | 0.042 |
| tvae_ep1000 | - | 1 | 0.164 | 0.0482 | 0.0338 |
| tvae_ind | - | 1 | 0.1803 | 0.0566 | 0.03 |
| tvae_qt | - | 3 | 0.1515 ± 0.0027 | 0.0548 ± 0.0034 | 0.0256 ± 0.0066 |

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.5021**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| aim40_eps1_seed0 | 1.0 | 0.0 | 0.1709 |
| aim40_eps5_seed0 | 1.0 | 0.0 | 0.1709 |
| aim40_eps8_seed0 | 1.0 | 0.0 | 0.1709 |
| aim50_eps1_seed0 | 1.0 | 0.0 | 0.2137 |
| aim50_eps5_seed0 | 1.0 | 0.0 | 0.2137 |
| aim50_eps8_seed0 | 1.0 | 0.0 | 0.2137 |
| ctgan_qt_seed0 | 1.0 | 0.0 | 1.0 |
| ctgan_seed0 | 1.0 | 0.0 | 1.0 |
| ctgan_seed1 | 1.0 | 0.0 | 1.0 |
| ctgan_seed2 | 1.0 | 0.0 | 1.0 |
| ddpm_g_seed0 | 1.0 | 0.0 | 1.0 |
| ddpm_seed0 | 0.9998 | 0.0002 | 1.0 |
| ddpm_seed1 | 0.9999 | 0.0001 | 1.0 |
| ddpm_seed2 | 1.0 | 0.0001 | 1.0 |
| dpctgan_eps10_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps15_seed0 | 0.9996 | 0.0007 | 1.0 |
| dpctgan_eps15_seed1 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps15_seed2 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps1_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps20_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps5_seed0 | 0.9996 | 0.0007 | 1.0 |
| dpctgan_eps8_seed0 | 0.9996 | 0.0007 | 1.0 |
| gaussian_copula_seed0 | 1.0 | 0.0 | 1.0 |
| gaussian_copula_seed1 | 1.0 | 0.0 | 1.0 |
| gaussian_copula_seed2 | 0.9999 | 0.0001 | 1.0 |
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
| tvae_cap256_seed0 | 0.9979 | 0.0011 | 1.0 |
| tvae_ep1000_seed0 | 0.9979 | 0.0023 | 1.0 |
| tvae_ind_seed0 | 0.9986 | 0.001 | 1.0 |
| tvae_qt_seed0 | 0.9958 | 0.0029 | 1.0 |
| tvae_qt_seed1 | 0.9955 | 0.004 | 1.0 |
| tvae_qt_seed2 | 0.9971 | 0.0017 | 1.0 |
| tvae_seed0 | 0.9992 | 0.0006 | 1.0 |
| tvae_seed1 | 0.9977 | 0.0015 | 1.0 |
| tvae_seed2 | 0.9989 | 0.001 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.0848 | 0.0766 | 0.1172 | 0.0868 | 0.0938 |
| aim40_eps1_seed0 | 0.4743 | 0.4745 | 0.5096 | 0.4633 | 0.5162 |
| aim40_eps5_seed0 | 0.5033 | 0.488 | 0.5971 | 0.4727 | 0.5352 |
| aim40_eps8_seed0 | 0.584 | 0.4684 | 0.5717 | 0.486 | 0.5509 |
| aim50_eps1_seed0 | 0.5252 | 0.5069 | 0.5463 | 0.4992 | 0.5521 |
| aim50_eps5_seed0 | 0.5755 | 0.4974 | 0.5931 | 0.5031 | 0.571 |
| aim50_eps8_seed0 | 0.5603 | 0.5171 | 0.526 | 0.4957 | 0.5397 |
| ctgan_qt_seed0 | 0.27 | 0.2637 | 0.2987 | 0.2569 | 0.2871 |
| ctgan_seed0 | 0.3438 | 0.346 | 0.364 | 0.3403 | 0.349 |
| ctgan_seed1 | 0.3355 | 0.3146 | 0.3431 | 0.3254 | 0.3287 |
| ctgan_seed2 | 0.3149 | 0.3005 | 0.3435 | 0.3073 | 0.3252 |
| ddpm_g_seed0 | 0.5554 | 0.5598 | 0.5748 | 0.5436 | 0.5697 |
| ddpm_seed0 | 0.5306 | 0.5234 | 0.5385 | 0.51 | 0.5239 |
| ddpm_seed1 | 0.525 | 0.5209 | 0.5361 | 0.512 | 0.5259 |
| ddpm_seed2 | 0.5198 | 0.5237 | 0.5337 | 0.5175 | 0.5154 |
| dpctgan_eps10_seed0 | 0.9364 | - | 0.9356 | - | - |
| dpctgan_eps15_seed0 | 0.9156 | - | - | - | 0.9199 |
| dpctgan_eps15_seed1 | - | 0.8854 | 0.892 | 0.8846 | 0.8951 |
| dpctgan_eps15_seed2 | - | 0.945 | 0.9454 | - | - |
| dpctgan_eps1_seed0 | - | 0.9148 | 0.9136 | 0.9137 | 0.9112 |
| dpctgan_eps20_seed0 | - | 0.9123 | 0.9049 | 0.9119 | - |
| dpctgan_eps5_seed0 | 0.9045 | - | - | - | 0.9092 |
| dpctgan_eps8_seed0 | 0.9478 | 0.9472 | - | - | 0.9491 |
| gaussian_copula_seed0 | 0.3513 | 0.3366 | 0.372 | 0.3356 | 0.3472 |
| gaussian_copula_seed1 | 0.3541 | 0.3458 | 0.3736 | 0.346 | 0.3505 |
| gaussian_copula_seed2 | 0.3499 | 0.3427 | 0.3611 | 0.3414 | 0.3537 |
| mst_eps10_seed0 | 0.5413 | 0.5255 | 0.6007 | 0.541 | 0.5757 |
| mst_eps15_seed0 | 0.5412 | 0.5278 | 0.6644 | 0.5654 | 0.6281 |
| mst_eps15_seed1 | 0.565 | 0.5268 | 0.6959 | 0.5438 | 0.6194 |
| mst_eps15_seed2 | 0.5406 | 0.5316 | 0.6847 | 0.5521 | 0.6211 |
| mst_eps1_seed0 | 0.5864 | 0.5529 | 0.5981 | 0.548 | 0.6097 |
| mst_eps20_seed0 | 0.5472 | 0.5277 | 0.6713 | 0.5504 | 0.6264 |
| mst_eps5_seed0 | 0.5321 | 0.5223 | 0.5947 | 0.5172 | 0.5707 |
| mst_eps8_seed0 | 0.5477 | 0.5251 | 0.6337 | 0.5472 | 0.6179 |
| patectgan_eps15_seed0 | 0.4097 | 0.4127 | 0.4242 | 0.4191 | 0.421 |
| patectgan_eps1_seed0 | 0.5345 | 0.5152 | 0.5614 | 0.5187 | 0.5477 |
| patectgan_eps5_seed0 | 0.4439 | 0.4359 | 0.481 | 0.4398 | 0.4501 |
| tvae_cap256_seed0 | 0.2027 | 0.1831 | 0.2241 | 0.186 | 0.1946 |
| tvae_ep1000_seed0 | 0.1958 | 0.1774 | 0.2116 | 0.1809 | 0.1921 |
| tvae_ind_seed0 | 0.2215 | 0.1961 | 0.2111 | 0.197 | 0.192 |
| tvae_qt_seed0 | 0.2044 | 0.151 | 0.2014 | 0.1598 | 0.1898 |
| tvae_qt_seed1 | 0.2088 | 0.166 | 0.2003 | 0.1662 | 0.1954 |
| tvae_qt_seed2 | 0.2032 | 0.1625 | 0.2169 | 0.1627 | 0.1945 |
| tvae_seed0 | 0.2334 | 0.2003 | 0.2244 | 0.2016 | 0.2074 |
| tvae_seed1 | 0.2158 | 0.1948 | 0.2275 | 0.1988 | 0.215 |
| tvae_seed2 | 0.2238 | 0.1892 | 0.2299 | 0.1937 | 0.2065 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4588 | 0.4674 | 0.0264 | 0.0363 |
| aim40_eps5_seed0 | 0.4485 | 0.4624 | 0.018 | 0.0269 |
| aim40_eps8_seed0 | 0.4483 | 0.4631 | 0.0218 | 0.0305 |
| aim50_eps1_seed0 | 0.492 | 0.5024 | 0.0283 | 0.0311 |
| aim50_eps5_seed0 | 0.4872 | 0.4985 | 0.0175 | 0.0233 |
| aim50_eps8_seed0 | 0.4864 | 0.4989 | 0.0174 | 0.0253 |
| ctgan_qt_seed0 | 0.2438 | 0.2426 | 0.0724 | 0.0763 |
| ctgan_seed0 | 0.3267 | 0.3202 | 0.0809 | 0.078 |
| ctgan_seed1 | 0.3081 | 0.3053 | 0.079 | 0.0792 |
| ctgan_seed2 | 0.2854 | 0.2884 | 0.0717 | 0.0751 |
| ddpm_g_seed0 | 0.5452 | 0.5445 | 0.2374 | 0.2476 |
| ddpm_seed0 | 0.5151 | 0.5151 | 0.0934 | 0.0994 |
| ddpm_seed1 | 0.5165 | 0.5121 | 0.0992 | 0.1011 |
| ddpm_seed2 | 0.5139 | 0.5114 | 0.099 | 0.1019 |
| dpctgan_eps10_seed0 | 0.9348 | 0.9312 | 0.2644 | 0.2604 |
| dpctgan_eps15_seed0 | 0.9118 | 0.9116 | 0.2818 | 0.2787 |
| dpctgan_eps15_seed1 | 0.8846 | 0.8829 | 0.2782 | 0.2763 |
| dpctgan_eps15_seed2 | 0.9462 | 0.9467 | 0.2735 | 0.2687 |
| dpctgan_eps1_seed0 | 0.9131 | 0.9134 | 0.3155 | 0.3132 |
| dpctgan_eps20_seed0 | 0.9157 | 0.913 | 0.2871 | 0.2844 |
| dpctgan_eps5_seed0 | 0.9032 | 0.9019 | 0.2872 | 0.2845 |
| dpctgan_eps8_seed0 | 0.9469 | 0.9466 | 0.2985 | 0.2957 |
| gaussian_copula_seed0 | 0.3228 | 0.3127 | 0.0135 | 0.0197 |
| gaussian_copula_seed1 | 0.3303 | 0.3258 | 0.0144 | 0.0183 |
| gaussian_copula_seed2 | 0.3225 | 0.3186 | 0.0145 | 0.0215 |
| mst_eps10_seed0 | 0.5033 | 0.5005 | 0.0107 | 0.0189 |
| mst_eps15_seed0 | 0.5028 | 0.5029 | 0.009 | 0.0181 |
| mst_eps15_seed1 | 0.5018 | 0.5003 | 0.0106 | 0.0191 |
| mst_eps15_seed2 | 0.506 | 0.5018 | 0.0105 | 0.0189 |
| mst_eps1_seed0 | 0.5536 | 0.548 | 0.0383 | 0.0414 |
| mst_eps20_seed0 | 0.5022 | 0.5002 | 0.0099 | 0.0188 |
| mst_eps5_seed0 | 0.5042 | 0.4995 | 0.0132 | 0.0215 |
| mst_eps8_seed0 | 0.5058 | 0.5052 | 0.0124 | 0.0196 |
| patectgan_eps15_seed0 | 0.3977 | 0.3979 | 0.0826 | 0.0771 |
| patectgan_eps1_seed0 | 0.5083 | 0.4969 | 0.0546 | 0.0559 |
| patectgan_eps5_seed0 | 0.425 | 0.4284 | 0.1353 | 0.1311 |
| tvae_cap256_seed0 | 0.1721 | 0.1781 | 0.0492 | 0.0531 |
| tvae_ep1000_seed0 | 0.164 | 0.1669 | 0.0482 | 0.0492 |
| tvae_ind_seed0 | 0.1803 | 0.187 | 0.0566 | 0.0575 |
| tvae_qt_seed0 | 0.1496 | 0.1569 | 0.051 | 0.0535 |
| tvae_qt_seed1 | 0.1546 | 0.1568 | 0.0557 | 0.0554 |
| tvae_qt_seed2 | 0.1502 | 0.1557 | 0.0576 | 0.0611 |
| tvae_seed0 | 0.1808 | 0.1859 | 0.0567 | 0.0573 |
| tvae_seed1 | 0.1824 | 0.1901 | 0.0525 | 0.0531 |
| tvae_seed2 | 0.1778 | 0.1822 | 0.0567 | 0.0583 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 2053 | 0.0658 | 0.0498 | 0.7901 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 13203 | 0.0363 | 0.0271 | 0.944 | 1 | - |
| *noise floor* | corr-ratio (num-cat) | 12540 | 0.0379 | 0.028 | 0.9275 | 0 | - |
| aim40_eps1_seed0 | Spearman (num-num) | 207 | 0.2071 | 0.1432 | 0.3913 | 0 | `lab_results_hdl_value_last|lab_results_hdl_value_first` (0.9984 -> -0.0906) |
| aim40_eps1_seed0 | Cramer's V (cat-cat) | 171 | 0.1948 | 0.1552 | 0.4327 | 12 | `encounter_primary_reason_CV_Disease_f5a_w1mo_first|med_sglt2i` (0.0477 -> 0.8631) |
| aim40_eps1_seed0 | corr-ratio (num-cat) | 861 | 0.0487 | 0.0 | 0.8455 | 4 | `lab_results_validSerumCreatinine_value_pET|ckd_severity_from_calculated_egfr` (0.8613 -> 0.0884) |
| aim40_eps5_seed0 | Spearman (num-num) | 190 | 0.2255 | 0.1854 | 0.3105 | 0 | `lab_results_ntProBnp_value_first|lab_results_hdl_value_last` (-0.2149 -> 0.7144) |
| aim40_eps5_seed0 | Cramer's V (cat-cat) | 171 | 0.1169 | 0.0778 | 0.5906 | 3 | `med_acei|med_oral_antidiabetic` (0.0215 -> 0.5886) |
| aim40_eps5_seed0 | corr-ratio (num-cat) | 820 | 0.0634 | 0.0 | 0.7732 | 12 | `vital_signs_height_value_p1a_avg|med_acei` (0.0834 -> 0.6364) |
| aim40_eps8_seed0 | Spearman (num-num) | 210 | 0.2202 | 0.1869 | 0.3429 | 2 | `lab_results_valideGFR_value_first|lab_results_creatBS_value_p3a_avg` (-0.8269 -> -0.0785) |
| aim40_eps8_seed0 | Cramer's V (cat-cat) | 171 | 0.154 | 0.1109 | 0.4912 | 5 | `encounter_primary_reason_CV_Disease_f5a_w1mo_first|med_diuretics_loop_history` (0.0141 -> 0.7951) |
| aim40_eps8_seed0 | corr-ratio (num-cat) | 861 | 0.0556 | 0.0 | 0.8084 | 11 | `lab_results_cholTot_value_last|encounter_primary_reason_CV_Disease_f5a_w1mo_first` (0.0305 -> 0.7244) |
| aim50_eps1_seed0 | Spearman (num-num) | 297 | 0.194 | 0.1409 | 0.367 | 0 | `electrocardiographs_ecg_qrs_duration_pET_last|electrocardiographs_ecg_qrs_duration_pET_first` (0.8809 -> -0.2105) |
| aim50_eps1_seed0 | Cramer's V (cat-cat) | 300 | 0.1427 | 0.0828 | 0.53 | 6 | `med_diuretics_loop_history|med_mra_history` (0.8153 -> 0.0251) |
| aim50_eps1_seed0 | corr-ratio (num-cat) | 1175 | 0.0521 | 0.0063 | 0.8306 | 8 | `lab_results_validSerumCreatinine_value_pET|ckd_severity_calculated_or_measured` (0.8613 -> 0.0921) |
| aim50_eps5_seed0 | Spearman (num-num) | 296 | 0.2284 | 0.1615 | 0.3581 | 1 | `lab_results_ntProBnp_value_first|lab_results_hdl_value_first` (-0.2148 -> 0.7711) |
| aim50_eps5_seed0 | Cramer's V (cat-cat) | 300 | 0.1337 | 0.0855 | 0.5633 | 9 | `patient_demographics_gender|encounters_admissionYear` (0.0948 -> 0.7087) |
| aim50_eps5_seed0 | corr-ratio (num-cat) | 1175 | 0.065 | 0.0074 | 0.7864 | 7 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.7623 -> 0.0645) |
| aim50_eps8_seed0 | Spearman (num-num) | 292 | 0.202 | 0.1597 | 0.3425 | 0 | `lab_results_triGly_value_last|lab_results_triGly_value_first` (0.9997 -> 0.1397) |
| aim50_eps8_seed0 | Cramer's V (cat-cat) | 300 | 0.1529 | 0.1163 | 0.4367 | 8 | `patient_demographics_gender|med_rasi` (0.0909 -> 0.862) |
| aim50_eps8_seed0 | corr-ratio (num-cat) | 1175 | 0.0717 | 0.0102 | 0.7498 | 19 | `vital_signs_height_value_p1a_avg|encounters_admissionYear` (0.0618 -> 0.8585) |
| ctgan_qt_seed0 | Spearman (num-num) | 1643 | 0.1438 | 0.0979 | 0.5064 | 0 | `vital_signs_diastolicBp_value_last|vital_signs_diastolicBp_value_first` (1.0 -> -0.1271) |
| ctgan_qt_seed0 | Cramer's V (cat-cat) | 13366 | 0.0866 | 0.0359 | 0.7951 | 0 | `cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first|cause_of_death_isAllCause_f5a_w1mo_first` (1.0 -> 0.0027) |
| ctgan_qt_seed0 | corr-ratio (num-cat) | 11780 | 0.0544 | 0.0325 | 0.8498 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9573 -> 0.0667) |
| ctgan_seed0 | Spearman (num-num) | 1755 | 0.1465 | 0.1042 | 0.4849 | 0 | `lab_results_hdl_value_last|lab_results_hdl_value_first` (0.9984 -> -0.2216) |
| ctgan_seed0 | Cramer's V (cat-cat) | 13695 | 0.0874 | 0.0343 | 0.794 | 0 | `cause_of_death_isCV_f5a_w1a_first|cause_of_death_isRenal_f5a_w1a_first` (1.0 -> 0.0002) |
| ctgan_seed0 | corr-ratio (num-cat) | 12350 | 0.0556 | 0.0334 | 0.8417 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9675 -> 0.0536) |
| ctgan_seed1 | Spearman (num-num) | 1613 | 0.1449 | 0.1024 | 0.4861 | 0 | `vital_signs_diastolicBp_value_last|vital_signs_diastolicBp_value_first` (1.0 -> -0.1402) |
| ctgan_seed1 | Cramer's V (cat-cat) | 13695 | 0.0883 | 0.0347 | 0.7899 | 0 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0005) |
| ctgan_seed1 | corr-ratio (num-cat) | 11780 | 0.056 | 0.033 | 0.8399 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9573 -> 0.0517) |
| ctgan_seed2 | Spearman (num-num) | 1725 | 0.1495 | 0.1069 | 0.4783 | 0 | `vital_signs_systolicBp_value_last|vital_signs_systolicBp_value_first` (1.0 -> -0.1753) |
| ctgan_seed2 | Cramer's V (cat-cat) | 13366 | 0.0894 | 0.0351 | 0.7884 | 3 | `cause_of_death_isCV_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (1.0 -> 0.0014) |
| ctgan_seed2 | corr-ratio (num-cat) | 11780 | 0.0547 | 0.0324 | 0.8446 | 2 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9675 -> 0.0674) |
| ddpm_g_seed0 | Spearman (num-num) | 1940 | 0.137 | 0.0954 | 0.5242 | 3 | `electrocardiographs_ecg_qrs_duration_pET_first|electrocardiographs_ecg_qrs_axis_pET_first` (-0.2363 -> 0.8912) |
| ddpm_g_seed0 | Cramer's V (cat-cat) | 7750 | 0.0724 | 0.0328 | 0.8454 | 0 | `encounter_primary_reason_HF_Disease_f5a_w6mo_first|encounter_primary_reason_HF_Disease_f5a_w1a_first` (0.9508 -> 0.0034) |
| ddpm_g_seed0 | corr-ratio (num-cat) | 11970 | 0.0571 | 0.036 | 0.8284 | 0 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isNonRenalAndNonCV_f5a_w1a_first` (0.8418 -> 0.0023) |
| ddpm_seed0 | Spearman (num-num) | 2053 | 0.1333 | 0.0903 | 0.5446 | 3 | `electrocardiographs_ecg_qrs_duration_pET_first|electrocardiographs_ecg_qrs_axis_pET_first` (-0.2363 -> 0.9103) |
| ddpm_seed0 | Cramer's V (cat-cat) | 14028 | 0.0689 | 0.0332 | 0.8221 | 0 | `encounter_primary_reason_CV_Disease_f5a_w7d_first|encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (1.0 -> 0.1224) |
| ddpm_seed0 | corr-ratio (num-cat) | 12350 | 0.0501 | 0.0321 | 0.873 | 0 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isRenal_f5a_w1a_first` (0.8418 -> 0.0145) |
| ddpm_seed1 | Spearman (num-num) | 2075 | 0.1303 | 0.0896 | 0.5407 | 3 | `electrocardiographs_ecg_qrs_duration_pET_last|electrocardiographs_ecg_qrs_axis_pET_first` (-0.2229 -> 0.9189) |
| ddpm_seed1 | Cramer's V (cat-cat) | 14028 | 0.0692 | 0.0329 | 0.8216 | 0 | `encounter_primary_reason_CV_Disease_f5a_w7d_first|encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (1.0 -> 0.2008) |
| ddpm_seed1 | corr-ratio (num-cat) | 12540 | 0.0499 | 0.0317 | 0.873 | 0 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isCV_f5a_w1a_first` (0.8418 -> 0.003) |
| ddpm_seed2 | Spearman (num-num) | 2073 | 0.1334 | 0.0937 | 0.5239 | 3 | `electrocardiographs_ecg_qrs_duration_pET_first|electrocardiographs_ecg_qrs_axis_pET_first` (-0.2363 -> 0.9079) |
| ddpm_seed2 | Cramer's V (cat-cat) | 14028 | 0.0692 | 0.0331 | 0.824 | 1 | `encounter_primary_reason_CV_Disease_f5a_w7d_first|encounter_primary_reason_renal_complications_f5a_w7d_first` (1.0 -> 0.1785) |
| ddpm_seed2 | corr-ratio (num-cat) | 12540 | 0.0506 | 0.0326 | 0.8671 | 0 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isCV_f5a_w1a_first` (0.8418 -> 0.0071) |
| dpctgan_eps10_seed0 | Spearman (num-num) | 374 | 0.3446 | 0.2212 | 0.2594 | 55 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7571 -> 0.9641) |
| dpctgan_eps10_seed0 | Cramer's V (cat-cat) | 3160 | 0.1231 | 0.0533 | 0.6921 | 3 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isAllCause_f5a_w1mo_first` (1.0 -> 0.0019) |
| dpctgan_eps10_seed0 | corr-ratio (num-cat) | 5320 | 0.0608 | 0.0364 | 0.813 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9675 -> 0.0) |
| dpctgan_eps15_seed0 | Spearman (num-num) | 630 | 0.3673 | 0.2957 | 0.1937 | 106 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7571 -> 0.9751) |
| dpctgan_eps15_seed0 | Cramer's V (cat-cat) | 2701 | 0.1388 | 0.0631 | 0.6338 | 0 | `cause_of_death_isCV_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (1.0 -> 0.0019) |
| dpctgan_eps15_seed0 | corr-ratio (num-cat) | 6840 | 0.0554 | 0.0332 | 0.85 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9675 -> 0.0) |
| dpctgan_eps15_seed1 | Spearman (num-num) | 405 | 0.2688 | 0.1799 | 0.316 | 21 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7571 -> 0.6456) |
| dpctgan_eps15_seed1 | Cramer's V (cat-cat) | 4278 | 0.1289 | 0.0554 | 0.6629 | 6 | `cause_of_death_isCV_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (1.0 -> 0.0019) |
| dpctgan_eps15_seed1 | corr-ratio (num-cat) | 5510 | 0.0606 | 0.0357 | 0.8196 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9573 -> 0.009) |
| dpctgan_eps15_seed2 | Spearman (num-num) | 433 | 0.4279 | 0.358 | 0.1824 | 75 | `lab_results_valideGFR_value_first|eGFR_2021_ckd_epi_creatinine` (0.7929 -> -0.9851) |
| dpctgan_eps15_seed2 | Cramer's V (cat-cat) | 4950 | 0.1185 | 0.0521 | 0.6891 | 7 | `cause_of_death_isRenal_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (1.0 -> 0.0019) |
| dpctgan_eps15_seed2 | corr-ratio (num-cat) | 5700 | 0.0636 | 0.0377 | 0.8084 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9573 -> 0.0) |
| dpctgan_eps1_seed0 | Spearman (num-num) | 700 | 0.2554 | 0.1787 | 0.3014 | 27 | `lab_results_creatBS_value_p3a_avg|lab_results_validSerumCreatinine_value_pET` (0.8425 -> -0.3383) |
| dpctgan_eps1_seed0 | Cramer's V (cat-cat) | 6441 | 0.1046 | 0.0419 | 0.7463 | 8 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first` (1.0 -> 0.0019) |
| dpctgan_eps1_seed0 | corr-ratio (num-cat) | 7220 | 0.0588 | 0.0344 | 0.8292 | 0 | `lab_results_valideGFR_value_first|ckd_severity_categorizedValue` (0.9645 -> 0.0113) |
| dpctgan_eps20_seed0 | Spearman (num-num) | 526 | 0.3459 | 0.211 | 0.2567 | 61 | `lab_results_validSerumCreatinine_value_pET|eGFR_2021_ckd_epi_creatinine` (-0.8946 -> 0.875) |
| dpctgan_eps20_seed0 | Cramer's V (cat-cat) | 3240 | 0.1197 | 0.0469 | 0.7022 | 12 | `encounter_primary_reason_non_CV_Disease_f5a_w7d_first|encounter_primary_reason_renal_complications_f5a_w7d_first` (1.0 -> 0.0019) |
| dpctgan_eps20_seed0 | corr-ratio (num-cat) | 6270 | 0.0664 | 0.0389 | 0.7871 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9675 -> 0.0) |
| dpctgan_eps5_seed0 | Spearman (num-num) | 625 | 0.3607 | 0.2916 | 0.2096 | 84 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7571 -> 0.9197) |
| dpctgan_eps5_seed0 | Cramer's V (cat-cat) | 3003 | 0.1438 | 0.0622 | 0.63 | 3 | `cause_of_death_isRenal_f5a_w5a_first|cause_of_death_isAllCause_f5a_w5a_first` (1.0 -> 0.0027) |
| dpctgan_eps5_seed0 | corr-ratio (num-cat) | 6840 | 0.0621 | 0.0368 | 0.8079 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9675 -> 0.0) |
| dpctgan_eps8_seed0 | Spearman (num-num) | 496 | 0.383 | 0.3285 | 0.1774 | 87 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7571 -> 0.9923) |
| dpctgan_eps8_seed0 | Cramer's V (cat-cat) | 3240 | 0.1389 | 0.0583 | 0.6454 | 3 | `cause_of_death_isCV_f5a_w3a_first|cause_of_death_isNonRenalAndNonCV_f5a_w3a_first` (1.0 -> 0.0019) |
| dpctgan_eps8_seed0 | corr-ratio (num-cat) | 6080 | 0.0601 | 0.0355 | 0.828 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9573 -> 0.0465) |
| gaussian_copula_seed0 | Spearman (num-num) | 1747 | 0.134 | 0.0884 | 0.5404 | 3 | `electrocardiographs_ecg_qrs_duration_pET_first|electrocardiographs_ecg_qrs_axis_pET_first` (-0.2363 -> 0.9239) |
| gaussian_copula_seed0 | Cramer's V (cat-cat) | 13041 | 0.0792 | 0.0345 | 0.7983 | 0 | `encounter_primary_reason_CV_Disease_f5a_w6mo_first|encounter_primary_reason_renal_complications_f5a_w6mo_first` (1.0 -> 0.0439) |
| gaussian_copula_seed0 | corr-ratio (num-cat) | 11590 | 0.0507 | 0.0312 | 0.868 | 4 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isCV_f5a_w1a_first` (0.8418 -> 0.0473) |
| gaussian_copula_seed1 | Spearman (num-num) | 1753 | 0.1329 | 0.0906 | 0.5351 | 9 | `electrocardiographs_ecg_qrs_duration_pET_first|electrocardiographs_ecg_qrs_axis_pET_first` (-0.2363 -> 0.9196) |
| gaussian_copula_seed1 | Cramer's V (cat-cat) | 13366 | 0.0767 | 0.0335 | 0.8106 | 1 | `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first|encounter_primary_reason_renal_complications_f5a_w6mo_first` (1.0 -> 0.0323) |
| gaussian_copula_seed1 | corr-ratio (num-cat) | 11590 | 0.0515 | 0.0316 | 0.8577 | 6 | `lab_results_valideGFR_value_first|ckd_severity_categorizedValue` (0.9645 -> 0.209) |
| gaussian_copula_seed2 | Spearman (num-num) | 1752 | 0.1343 | 0.0897 | 0.528 | 7 | `electrocardiographs_ecg_qrs_duration_pET_first|electrocardiographs_ecg_qrs_axis_pET_first` (-0.2363 -> 0.9439) |
| gaussian_copula_seed2 | Cramer's V (cat-cat) | 13041 | 0.0779 | 0.0339 | 0.8055 | 0 | `encounter_primary_reason_CV_Disease_f5a_w3mo_first|encounter_primary_reason_renal_complications_f5a_w3mo_first` (1.0 -> 0.011) |
| gaussian_copula_seed2 | corr-ratio (num-cat) | 11780 | 0.0513 | 0.0317 | 0.8615 | 6 | `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first|cause_of_death_isAllCause_f5a_w1a_first` (0.8418 -> 0.1231) |
| mst_eps10_seed0 | Spearman (num-num) | 1874 | 0.2468 | 0.209 | 0.2572 | 96 | `lab_results_crpNonHs_value_first|lab_results_albuminBS_value_last` (-0.5863 -> 0.7254) |
| mst_eps10_seed0 | Cramer's V (cat-cat) | 13861 | 0.1771 | 0.1239 | 0.4385 | 944 | `symptoms_Syncope_display_pET_any|med_inotropes` (0.0058 -> 1.0) |
| mst_eps10_seed0 | corr-ratio (num-cat) | 11970 | 0.1528 | 0.1011 | 0.4968 | 613 | `conditions_heartFailure_timeFromEarliest_first|symptoms_Chest_pain_display_pET_any` (0.0159 -> 0.9515) |
| mst_eps15_seed0 | Spearman (num-num) | 2009 | 0.2515 | 0.206 | 0.2693 | 141 | `lab_results_tropTHs_value_first|smoking_status_smoker_totalSmokingDuration_sum` (-0.3025 -> 1.0) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 13530 | 0.1774 | 0.1222 | 0.434 | 958 | `symptoms_Syncope_display_pET_any|electrocardiographs_ecg_ischemia_without_st_pET` (0.0277 -> 1.0) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 12350 | 0.1567 | 0.1014 | 0.4956 | 737 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|med_arb` (0.0395 -> 0.9091) |
| mst_eps15_seed1 | Spearman (num-num) | 2089 | 0.2696 | 0.2472 | 0.2111 | 108 | `lab_results_crpNonHs_value_first|lab_results_albuminBS_value_last` (-0.5863 -> 0.6782) |
| mst_eps15_seed1 | Cramer's V (cat-cat) | 13861 | 0.1731 | 0.1258 | 0.432 | 822 | `med_insulins|med_rdoad_history` (0.0168 -> 0.942) |
| mst_eps15_seed1 | corr-ratio (num-cat) | 12540 | 0.1641 | 0.1157 | 0.4585 | 743 | `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count|conditions_osa` (0.0132 -> 0.9234) |
| mst_eps15_seed2 | Spearman (num-num) | 2072 | 0.2486 | 0.2059 | 0.2688 | 113 | `lab_results_crpNonHs_value_first|lab_results_albuminBS_value_first` (-0.5879 -> 0.5555) |
| mst_eps15_seed2 | Cramer's V (cat-cat) | 13861 | 0.1858 | 0.1353 | 0.4011 | 998 | `symptoms_Orthopnoea_display_pET_any|symptoms_Tachypnoea_display_pET_any` (0.0011 -> 1.0) |
| mst_eps15_seed2 | corr-ratio (num-cat) | 12540 | 0.1642 | 0.1129 | 0.4675 | 716 | `smoking_status_smoker_totalSmokingDuration_sum|med_antiinfl_history` (0.0 -> 0.9311) |
| mst_eps1_seed0 | Spearman (num-num) | 1973 | 0.2018 | 0.1563 | 0.3259 | 27 | `vital_signs_weight_value_p6mo_first|vital_signs_bmi_value_pET` (0.8325 -> -0.4248) |
| mst_eps1_seed0 | Cramer's V (cat-cat) | 13695 | 0.162 | 0.1162 | 0.4533 | 479 | `symptoms_Palpitations_display_pET_any|med_inotropes_history` (0.002 -> 1.0) |
| mst_eps1_seed0 | corr-ratio (num-cat) | 12350 | 0.1122 | 0.0676 | 0.609 | 236 | `echocardiographs_lvef_pET_last|symptoms_Ascites_display_pET_any` (0.0076 -> 1.0) |
| mst_eps20_seed0 | Spearman (num-num) | 1937 | 0.2657 | 0.228 | 0.221 | 138 | `lab_results_crpNonHs_value_first|lab_results_albuminBS_value_last` (-0.5863 -> 0.5436) |
| mst_eps20_seed0 | Cramer's V (cat-cat) | 13530 | 0.1755 | 0.1257 | 0.4323 | 832 | `symptoms_Irregular_pulse_display_pET_any|encounter_primary_reason_renal_complications_f5a_w1a_first` (0.0343 -> 1.0) |
| mst_eps20_seed0 | corr-ratio (num-cat) | 12160 | 0.164 | 0.1091 | 0.4757 | 761 | `lab_results_tropTHs_value_last|symptoms_Irregular_pulse_display_pET_any` (0.0198 -> 1.0) |
| mst_eps5_seed0 | Spearman (num-num) | 1999 | 0.2237 | 0.1905 | 0.2691 | 52 | `lab_results_validSerumCreatinine_value_pET|eGFR_2021_ckd_epi_creatinine` (-0.8946 -> 0.3969) |
| mst_eps5_seed0 | Cramer's V (cat-cat) | 13366 | 0.1666 | 0.1183 | 0.4466 | 723 | `encounter_primary_reason_renal_complications_f5a_w1mo_first|med_diuretics_loop_history` (0.0002 -> 0.9511) |
| mst_eps5_seed0 | corr-ratio (num-cat) | 12350 | 0.1361 | 0.088 | 0.5321 | 330 | `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count|conditions_hyperthyroid` (0.0141 -> 0.8397) |
| mst_eps8_seed0 | Spearman (num-num) | 1914 | 0.2498 | 0.2181 | 0.2675 | 91 | `lab_results_valideGFR_value_first|maggic_total_score` (-0.5623 -> 0.4562) |
| mst_eps8_seed0 | Cramer's V (cat-cat) | 13861 | 0.1831 | 0.1347 | 0.4154 | 897 | `encounter_primary_reason_renal_complications_f5a_w1mo_first|med_mra_history` (0.0055 -> 0.9721) |
| mst_eps8_seed0 | corr-ratio (num-cat) | 12160 | 0.1478 | 0.0955 | 0.5118 | 488 | `echocardiographs_lvef_pET_last|conditions_hyperthyroid` (0.0504 -> 1.0) |
| patectgan_eps15_seed0 | Spearman (num-num) | 1848 | 0.1499 | 0.1126 | 0.4535 | 6 | `lab_results_validSerumCreatinine_value_first|lab_results_creatBS_value_p3a_avg` (0.9015 -> -0.2218) |
| patectgan_eps15_seed0 | Cramer's V (cat-cat) | 11935 | 0.0702 | 0.0402 | 0.7955 | 2 | `cause_of_death_isRenal_f5a_w1mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first` (1.0 -> 0.0068) |
| patectgan_eps15_seed0 | corr-ratio (num-cat) | 12350 | 0.0558 | 0.0346 | 0.8395 | 13 | `conditions_heartFailure_timeFromEarliest_first|conditions_dem` (0.046 -> 0.8606) |
| patectgan_eps1_seed0 | Spearman (num-num) | 1317 | 0.1413 | 0.0959 | 0.5118 | 0 | `lab_results_cholTot_value_last|lab_results_cholTot_value_first` (0.9994 -> -0.0765) |
| patectgan_eps1_seed0 | Cramer's V (cat-cat) | 14028 | 0.0859 | 0.0357 | 0.8072 | 0 | `cause_of_death_isRenal_f5a_w6mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first` (1.0 -> 0.0026) |
| patectgan_eps1_seed0 | corr-ratio (num-cat) | 10070 | 0.0519 | 0.0321 | 0.8595 | 1 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9675 -> 0.0778) |
| patectgan_eps5_seed0 | Spearman (num-num) | 1739 | 0.1698 | 0.1255 | 0.4077 | 8 | `vital_signs_systolicBp_value_first|vital_signs_systolicBpDuringEncounter_value_pET` (1.0 -> -0.0598) |
| patectgan_eps5_seed0 | Cramer's V (cat-cat) | 9870 | 0.0899 | 0.0385 | 0.7802 | 2 | `cause_of_death_isCV_f5a_w6mo_first|cause_of_death_isRenal_f5a_w6mo_first` (1.0 -> 0.0019) |
| patectgan_eps5_seed0 | corr-ratio (num-cat) | 12160 | 0.0606 | 0.0375 | 0.8174 | 2 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_HF_Disease_f5a_w1a_first` (0.8602 -> 0.0) |
| tvae_cap256_seed0 | Spearman (num-num) | 1877 | 0.0903 | 0.0711 | 0.6612 | 0 | `vital_signs_heartRate_value_last|vital_signs_heartRate_value_first` (1.0 -> 0.1571) |
| tvae_cap256_seed0 | Cramer's V (cat-cat) | 8256 | 0.0596 | 0.0425 | 0.8198 | 1 | `med_arni_history|conditions_myocarditis` (0.0269 -> 0.5762) |
| tvae_cap256_seed0 | corr-ratio (num-cat) | 11970 | 0.0451 | 0.0319 | 0.8848 | 2 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6606 -> 0.0304) |
| tvae_ep1000_seed0 | Spearman (num-num) | 1948 | 0.0842 | 0.0638 | 0.6879 | 0 | `lab_results_potassium_value_last|lab_results_potassium_value_first` (0.9229 -> 0.1571) |
| tvae_ep1000_seed0 | Cramer's V (cat-cat) | 9730 | 0.0473 | 0.0332 | 0.8869 | 14 | `med_inotropes_history|conditions_vd` (0.0231 -> 1.0) |
| tvae_ep1000_seed0 | corr-ratio (num-cat) | 12160 | 0.0415 | 0.0279 | 0.9067 | 6 | `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count|conditions_mc` (0.0051 -> 0.6716) |
| tvae_ind_seed0 | Spearman (num-num) | 1930 | 0.1012 | 0.0791 | 0.6171 | 1 | `lab_results_sodium_value_last|lab_results_sodium_value_first` (0.9672 -> 0.1371) |
| tvae_ind_seed0 | Cramer's V (cat-cat) | 7381 | 0.0634 | 0.0447 | 0.796 | 1 | `conditions_heart_failure_occurred_prior_to_18_months_any|med_antiarrhytmic_history` (0.2654 -> 0.748) |
| tvae_ind_seed0 | corr-ratio (num-cat) | 12350 | 0.0482 | 0.0322 | 0.8668 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_CV_Disease_f5a_w1a_first` (0.8608 -> 0.1461) |
| tvae_qt_seed0 | Spearman (num-num) | 1957 | 0.0921 | 0.069 | 0.676 | 0 | `vital_signs_heartRate_value_last|vital_signs_heartRate_value_first` (1.0 -> 0.1035) |
| tvae_qt_seed0 | Cramer's V (cat-cat) | 7875 | 0.0551 | 0.0376 | 0.8375 | 2 | `symptoms_Tachycardia_display_pET_any|med_arni_history` (0.0266 -> 0.7064) |
| tvae_qt_seed0 | corr-ratio (num-cat) | 12160 | 0.0445 | 0.0305 | 0.8902 | 1 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6606 -> 0.0732) |
| tvae_qt_seed1 | Spearman (num-num) | 1936 | 0.0904 | 0.0672 | 0.6694 | 0 | `vital_signs_heartRate_value_last|vital_signs_heartRate_value_first` (1.0 -> 0.1226) |
| tvae_qt_seed1 | Cramer's V (cat-cat) | 7750 | 0.0615 | 0.0448 | 0.8187 | 3 | `med_insulins_history|conditions_hypothyroid` (0.0343 -> 0.6647) |
| tvae_qt_seed1 | corr-ratio (num-cat) | 12160 | 0.0447 | 0.0298 | 0.8916 | 0 | `smoking_status_smoker_totalSmokingDuration_sum|conditions_af` (0.1328 -> 0.7452) |
| tvae_qt_seed2 | Spearman (num-num) | 1897 | 0.0975 | 0.0722 | 0.6394 | 0 | `lab_results_triGly_value_last|lab_results_triGly_value_first` (0.9997 -> 0.128) |
| tvae_qt_seed2 | Cramer's V (cat-cat) | 7503 | 0.0587 | 0.0439 | 0.8247 | 2 | `med_arni|med_insulins_history` (0.0103 -> 0.7064) |
| tvae_qt_seed2 | corr-ratio (num-cat) | 12160 | 0.0461 | 0.0313 | 0.8851 | 2 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_HF_Disease_f5a_w3a_first` (0.6337 -> 0.0504) |
| tvae_seed0 | Spearman (num-num) | 1925 | 0.0998 | 0.0795 | 0.6135 | 0 | `lab_results_potassium_value_last|lab_results_potassium_value_first` (0.9229 -> -0.0035) |
| tvae_seed0 | Cramer's V (cat-cat) | 7381 | 0.0584 | 0.0403 | 0.8231 | 2 | `med_rdoad|med_arni_history` (0.0222 -> 0.7064) |
| tvae_seed0 | corr-ratio (num-cat) | 12350 | 0.0476 | 0.0321 | 0.8768 | 1 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6606 -> 0.0) |
| tvae_seed1 | Spearman (num-num) | 1800 | 0.0913 | 0.0689 | 0.6594 | 0 | `vital_signs_heartRate_value_last|vital_signs_heartRate_value_first` (1.0 -> 0.003) |
| tvae_seed1 | Cramer's V (cat-cat) | 7503 | 0.06 | 0.0434 | 0.8194 | 5 | `med_rdoad|med_insulins_history` (0.0566 -> 0.8149) |
| tvae_seed1 | corr-ratio (num-cat) | 11780 | 0.045 | 0.0306 | 0.8841 | 1 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.6606 -> 0.0) |
| tvae_seed2 | Spearman (num-num) | 1869 | 0.0934 | 0.0683 | 0.657 | 0 | `lab_results_sodium_value_last|lab_results_sodium_value_first` (0.9672 -> 0.1024) |
| tvae_seed2 | Cramer's V (cat-cat) | 7381 | 0.0566 | 0.0397 | 0.8331 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (0.6957 -> 0.2381) |
| tvae_seed2 | corr-ratio (num-cat) | 11970 | 0.0437 | 0.03 | 0.8961 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_renal_complications_f5a_w3a_first` (0.7585 -> 0.0621) |

## original vs preprocessed

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.0155, W/std=0.0931, mean 6.1768 -> 4.9444, missing 0% -> 2%
- `lab_results_potassium_value_last`: KS=0.0075, W/std=0.0443, mean 4.1289 -> 4.16, missing 3% -> 4%
- `lab_results_potassium_value_first`: KS=0.007, W/std=0.0421, mean 4.1351 -> 4.1642, missing 3% -> 4%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.0063, W/std=0.0413, mean 5.3932 -> 4.9854, missing 0% -> 1%
- `vital_signs_bmi_value_pET`: KS=0.0024, W/std=0.0421, mean 28.6618 -> 28.0251, missing 0% -> 0%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 14 -> 14 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `lab_results_hemoglobin_value_first`: KS=0.1754, W/std=0.2909, mean 122.7778 -> 129.7097, missing 83% -> 82%
- `lab_results_hemoglobin_value_last`: KS=0.1651, W/std=0.298, mean 122.6322 -> 129.5054, missing 83% -> 82%
- `lab_results_hba1c_value_first`: KS=0.1387, W/std=0.2267, mean 50.7995 -> 47.931, missing 75% -> 72%
- `lab_results_hba1c_value_last`: KS=0.1387, W/std=0.2267, mean 50.7995 -> 47.931, missing 75% -> 72%
- `vital_signs_systolicBp_value_last`: KS=0.1325, W/std=0.2078, mean 130.1717 -> 125.5783, missing 50% -> 52%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0623, 14 -> 14 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w3mo_first`: TVD=0.0548, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w3mo_first`: TVD=0.0548, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first`: TVD=0.0548, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w3mo_first`: TVD=0.0548, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps1_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=6.632, mean 13.8027 -> 60.2067, missing 0% -> 0%
- `lab_results_triGly_value_last`: KS=0.8289, W/std=0.8329, mean 1.3399 -> 1.9766, missing 54% -> 57%
- `lab_results_triGly_value_first`: KS=0.8289, W/std=1.8069, mean 1.3407 -> 3.0813, missing 54% -> 52%
- `lab_results_crpNonHs_value_last`: KS=0.7964, W/std=1.2273, mean 40.7741 -> 118.677, missing 3% -> 0%
- `lab_results_crpNonHs_value_first`: KS=0.7951, W/std=0.8578, mean 40.8452 -> 89.8932, missing 3% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0921, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0473, 7 -> 7 categories, missing 0% -> 0%
- `med_acei`: TVD=0.0365, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0347, 5 -> 5 categories, missing 0% -> 0%
- `conditions_devices`: TVD=0.0305, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps5_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=5.2191, mean 13.8027 -> 50.2584, missing 0% -> 0%
- `lab_results_triGly_value_last`: KS=0.8289, W/std=1.0144, mean 1.3399 -> 2.2326, missing 54% -> 53%
- `lab_results_triGly_value_first`: KS=0.8289, W/std=0.97, mean 1.3407 -> 2.1311, missing 54% -> 53%
- `lab_results_crpNonHs_value_last`: KS=0.7964, W/std=0.7644, mean 40.7741 -> 82.7725, missing 3% -> 3%
- `lab_results_crpNonHs_value_first`: KS=0.7951, W/std=0.7658, mean 40.8452 -> 85.0394, missing 3% -> 3%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0636, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0404, 7 -> 7 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.0263, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first`: TVD=0.0231, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0225, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps8_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=5.8565, mean 13.8027 -> 54.7804, missing 0% -> 0%
- `lab_results_triGly_value_last`: KS=0.8289, W/std=1.1282, mean 1.3399 -> 2.3211, missing 54% -> 53%
- `lab_results_triGly_value_first`: KS=0.8289, W/std=0.997, mean 1.3407 -> 2.2276, missing 54% -> 53%
- `lab_results_crpNonHs_value_last`: KS=0.7964, W/std=0.7305, mean 40.7741 -> 80.9827, missing 3% -> 3%
- `lab_results_crpNonHs_value_first`: KS=0.7951, W/std=0.7511, mean 40.8452 -> 83.0147, missing 3% -> 2%
Worst categorical columns (by TVD):
- `med_sglt2i`: TVD=0.0677, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0524, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0367, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0348, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0311, 7 -> 7 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps1_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=12.9285, mean 13.8027 -> 104.2636, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=3.5288, mean 10.3363 -> 78.6176, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.958, W/std=2.4809, mean 6.5029 -> 21.9638, missing 0% -> 0%
- `lab_results_triGly_value_last`: KS=0.8289, W/std=1.1512, mean 1.3399 -> 2.3553, missing 54% -> 57%
- `lab_results_triGly_value_first`: KS=0.8289, W/std=2.6294, mean 1.3407 -> 3.8777, missing 54% -> 52%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.178, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0687, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0453, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0422, 5 -> 5 categories, missing 0% -> 0%
- `med_sglt2i`: TVD=0.035, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps5_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=5.3214, mean 13.8027 -> 51.0336, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=2.9487, mean 10.3363 -> 67.1835, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.958, W/std=2.4326, mean 6.5029 -> 21.602, missing 0% -> 0%
- `lab_results_triGly_value_last`: KS=0.8289, W/std=0.8982, mean 1.3399 -> 2.058, missing 54% -> 54%
- `lab_results_triGly_value_first`: KS=0.8289, W/std=1.0202, mean 1.3407 -> 2.2085, missing 54% -> 53%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0735, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0386, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0366, 7 -> 7 categories, missing 0% -> 0%
- `med_diuretics_history`: TVD=0.0279, 2 -> 2 categories, missing 0% -> 0%
- `med_diuretics_loop_history`: TVD=0.0256, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps8_seed0]

Worst numeric columns (by KS):
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=5.7646, mean 13.8027 -> 54.1344, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=2.2707, mean 10.3363 -> 53.23, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.958, W/std=2.3968, mean 6.5029 -> 21.2145, missing 0% -> 0%
- `lab_results_triGly_value_last`: KS=0.8289, W/std=0.8806, mean 1.3399 -> 2.0625, missing 54% -> 53%
- `lab_results_triGly_value_first`: KS=0.8289, W/std=0.9842, mean 1.3407 -> 2.1503, missing 54% -> 54%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.044, 14 -> 14 categories, missing 0% -> 0%
- `med_acei`: TVD=0.0385, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag_history`: TVD=0.0315, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0277, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0255, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[ctgan_qt_seed0]

Worst numeric columns (by KS):
- `vital_signs_heartRate_value_last`: KS=0.6702, W/std=2.0303, mean 83.6361 -> 128.0356, missing 57% -> 35%
- `electrocardiographs_ecg_qrs_axis_pET_first`: KS=0.6587, W/std=2.1489, mean 10.8663 -> 170.5914, missing 56% -> 51%
- `lab_results_hdl_value_first`: KS=0.6429, W/std=1.7987, mean 1.0702 -> 1.6828, missing 55% -> 61%
- `echocardiographs_lvef_pET_first`: KS=0.5198, W/std=1.2291, mean 38.3216 -> 57.409, missing 52% -> 76%
- `vital_signs_bmi_value_pET`: KS=0.4733, W/std=1.622, mean 28.05 -> 37.675, missing 0% -> 0%
Worst categorical columns (by TVD):
- `med_activeDuringEncounter_ace_inhibitors_arb_use`: TVD=0.2728, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w3a_first`: TVD=0.2645, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.2483, 3 -> 3 categories, missing 0% -> 0%
- `med_diuretics_loop_history`: TVD=0.2263, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.2145, 14 -> 14 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed0]

Worst numeric columns (by KS):
- `vital_signs_heartRate_value_first`: KS=0.6588, W/std=1.4353, mean 83.6361 -> 52.1889, missing 57% -> 40%
- `lab_results_cholTot_value_first`: KS=0.6572, W/std=1.5333, mean 4.1266 -> 2.3108, missing 55% -> 71%
- `lab_results_hdl_value_last`: KS=0.6505, W/std=1.7491, mean 1.0686 -> 0.469, missing 55% -> 37%
- `vital_signs_systolicBp_value_first`: KS=0.6398, W/std=1.5561, mean 130.1717 -> 95.435, missing 50% -> 51%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.6389, W/std=0.3339, mean 0.3398 -> 0.8731, missing 0% -> 2%
Worst categorical columns (by TVD):
- `cause_of_death_isNonRenalAndNonCV_f5a_w1a_first`: TVD=0.3238, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1a_first`: TVD=0.2916, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w6mo_first`: TVD=0.2877, 2 -> 2 categories, missing 0% -> 0%
- `med_diuretics_history`: TVD=0.2819, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.2437, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed1]

Worst numeric columns (by KS):
- `electrocardiographs_ecg_qt_duration_corrected_pET_last`: KS=0.6521, W/std=1.7729, mean 469.2839 -> 556.4573, missing 55% -> 57%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5836, W/std=0.5212, mean 820.2403 -> 1839.241, missing 77% -> 91%
- `lab_results_potassium_value_last`: KS=0.5632, W/std=1.4177, mean 4.1536 -> 3.2964, missing 4% -> 4%
- `lab_results_triGly_value_last`: KS=0.5517, W/std=1.4144, mean 1.3399 -> 2.7242, missing 54% -> 64%
- `lab_results_hdl_value_first`: KS=0.5424, W/std=1.4853, mean 1.0702 -> 1.5761, missing 55% -> 75%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.308, 14 -> 14 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1a_first`: TVD=0.2899, 3 -> 3 categories, missing 0% -> 0%
- `med_ll`: TVD=0.2736, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.2645, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w3a_first`: TVD=0.2567, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed2]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.6683, W/std=0.1108, mean 0.3398 -> 0.3877, missing 0% -> 2%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.6234, W/std=0.8455, mean 820.2403 -> 2624.8524, missing 77% -> 76%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.5468, W/std=1.3397, mean 466.2944 -> 399.45, missing 55% -> 67%
- `vital_signs_heartRate_value_first`: KS=0.5296, W/std=1.7439, mean 83.6361 -> 121.844, missing 57% -> 59%
- `lab_results_hba1c_value_first`: KS=0.5054, W/std=1.027, mean 50.7995 -> 66.2912, missing 75% -> 88%
Worst categorical columns (by TVD):
- `encounter_primary_reason_renal_complications_f5a_w6mo_first`: TVD=0.2916, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.2869, 14 -> 13 categories, missing 0% -> 0%
- `electrocardiographs_ecg_st_pET`: TVD=0.2807, 3 -> 3 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.2714, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.2606, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ddpm_g_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=1.0, W/std=None, mean 128.0879 -> None, missing 65% -> 100%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.9399, W/std=0.671, mean 125.9636 -> 158.1318, missing 68% -> 100%
- `lab_results_ntProBnp_value_last`: KS=0.7646, W/std=2.7167, mean 6807.5675 -> 29836.1655, missing 41% -> 46%
- `lab_results_hemoglobin_value_first`: KS=0.7595, W/std=2.027, mean 122.7778 -> 176.3264, missing 83% -> 65%
- `lab_results_ntProBnp_value_first`: KS=0.7594, W/std=2.7132, mean 6813.8474 -> 29798.7252, missing 41% -> 45%
Worst categorical columns (by TVD):
- `med_oral_antidiabetic_history`: TVD=0.8373, 2 -> 1 categories, missing 0% -> 0%
- `med_diuretics_loop`: TVD=0.7863, 2 -> 1 categories, missing 0% -> 0%
- `med_oral_antidiabetic`: TVD=0.6875, 2 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.6643, 3 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.6643, 3 -> 1 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed0]

Worst numeric columns (by KS):
- `lab_results_triGly_value_last`: KS=0.7584, W/std=6.5652, mean 1.3399 -> 7.7699, missing 54% -> 54%
- `lab_results_tropTHs_value_first`: KS=0.7501, W/std=3.434, mean 1.0572 -> 8.0557, missing 58% -> 53%
- `lab_results_hemoglobin_value_first`: KS=0.7486, W/std=1.9857, mean 122.7778 -> 175.237, missing 83% -> 63%
- `lab_results_triGly_value_first`: KS=0.7477, W/std=6.4423, mean 1.3407 -> 7.6445, missing 54% -> 53%
- `lab_results_tropTHs_value_last`: KS=0.746, W/std=3.413, mean 1.0632 -> 8.1238, missing 58% -> 53%
Worst categorical columns (by TVD):
- `smoking_status_smoker_last`: TVD=0.3276, 2 -> 2 categories, missing 0% -> 0%
- `med_diuretics_loop_history`: TVD=0.3193, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.3039, 2 -> 2 categories, missing 0% -> 0%
- `med_diuretics_history`: TVD=0.287, 2 -> 2 categories, missing 0% -> 0%
- `med_mra_history`: TVD=0.2786, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed1]

Worst numeric columns (by KS):
- `lab_results_albuminBS_value_first`: KS=0.7557, W/std=2.1652, mean 38.4892 -> 50.2715, missing 64% -> 60%
- `lab_results_tropTHs_value_last`: KS=0.7431, W/std=3.3877, mean 1.0632 -> 8.0715, missing 58% -> 58%
- `lab_results_tropTHs_value_first`: KS=0.7392, W/std=3.3874, mean 1.0572 -> 7.9606, missing 58% -> 58%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.737, W/std=8.292, mean 820.2403 -> 19968.2177, missing 77% -> 56%
- `electrocardiographs_ecg_qrs_duration_pET_last`: KS=0.7344, W/std=2.5183, mean 125.0331 -> 220.3366, missing 55% -> 53%
Worst categorical columns (by TVD):
- `med_diuretics_loop_history`: TVD=0.3407, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_smoker_last`: TVD=0.3367, 2 -> 2 categories, missing 0% -> 0%
- `med_mra_history`: TVD=0.3186, 2 -> 2 categories, missing 0% -> 0%
- `med_diuretics_history`: TVD=0.311, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.3058, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed2]

Worst numeric columns (by KS):
- `lab_results_ntProBnp_value_last`: KS=0.7646, W/std=2.7256, mean 6807.5675 -> 29911.685, missing 41% -> 51%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.7644, W/std=8.1999, mean 820.2403 -> 19755.479, missing 77% -> 57%
- `lab_results_ntProBnp_value_first`: KS=0.7624, W/std=2.7118, mean 6813.8474 -> 29787.0269, missing 41% -> 50%
- `electrocardiographs_ecg_qt_duration_corrected_pET_last`: KS=0.7373, W/std=2.8806, mean 469.2839 -> 610.9213, missing 55% -> 54%
- `lab_results_triGly_value_last`: KS=0.7356, W/std=6.2225, mean 1.3399 -> 7.4343, missing 54% -> 55%
Worst categorical columns (by TVD):
- `smoking_status_smoker_last`: TVD=0.3528, 2 -> 2 categories, missing 0% -> 0%
- `med_diuretics_loop_history`: TVD=0.3296, 2 -> 2 categories, missing 0% -> 0%
- `med_bb_history`: TVD=0.3058, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.289, 2 -> 2 categories, missing 0% -> 0%
- `med_diuretics_history`: TVD=0.28, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps10_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 83.9323 -> None, missing 12% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=7.957, mean 170.6066 -> 249.9916, missing 1% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=12.8388, mean 84.0306 -> 329.08, missing 16% -> 0%
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 170.6693 -> None, missing 2% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 75.5312 -> None, missing 50% -> 100%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.9658, 3 -> 3 categories, missing 0% -> 0%
- `med_vasodil`: TVD=0.9619, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8857, 14 -> 1 categories, missing 0% -> 0%
- `symptoms_Depression_display_pET_any`: TVD=0.8463, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`: TVD=0.8438, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=1.0, W/std=2.9742, mean 72.9832 -> 109.7362, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 84.2946 -> None, missing 12% -> 100%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=13.7809, mean 83.9323 -> 347.8847, missing 12% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=13.8202, mean 84.0306 -> 347.8111, missing 16% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=12.8409, mean 75.5312 -> 245.6235, missing 50% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.9612, 7 -> 2 categories, missing 0% -> 0%
- `med_insulins`: TVD=0.9477, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis_history`: TVD=0.9096, 2 -> 1 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.8916, 2 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8141, 14 -> 6 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed1]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 84.2946 -> None, missing 12% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=7.3003, mean 170.6066 -> 243.4397, missing 1% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 84.0306 -> None, missing 16% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 75.5312 -> None, missing 50% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 75.5312 -> None, missing 50% -> 100%
Worst categorical columns (by TVD):
- `conditions_dem`: TVD=0.9897, 2 -> 1 categories, missing 0% -> 0%
- `symptoms_Dizziness_display_pET_any`: TVD=0.9845, 2 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.9296, 3 -> 3 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9206, 14 -> 1 categories, missing 0% -> 0%
- `conditions_hypothyroid`: TVD=0.8844, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed2]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 84.2946 -> None, missing 12% -> 100%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 83.9323 -> None, missing 12% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=None, mean 170.6066 -> None, missing 1% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=13.837, mean 84.0306 -> 348.1319, missing 16% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 75.5312 -> None, missing 50% -> 100%
Worst categorical columns (by TVD):
- `med_potassium_binders`: TVD=0.9787, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9161, 14 -> 2 categories, missing 0% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.8851, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.8651, 5 -> 2 categories, missing 0% -> 0%
- `med_sglt2i`: TVD=0.8121, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps1_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 84.2946 -> None, missing 12% -> 100%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 83.9323 -> None, missing 12% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 75.5312 -> None, missing 50% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=12.544, mean 75.5312 -> 241.6913, missing 50% -> 0%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 83.6361 -> None, missing 57% -> 100%
Worst categorical columns (by TVD):
- `med_potassium_binders`: TVD=0.9929, 2 -> 2 categories, missing 0% -> 0%
- `med_antiinfl`: TVD=0.9903, 2 -> 2 categories, missing 0% -> 0%
- `med_arni_history`: TVD=0.9716, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mc`: TVD=0.9626, 2 -> 1 categories, missing 0% -> 0%
- `med_insulins`: TVD=0.9329, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps20_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 84.2946 -> None, missing 12% -> 100%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=13.8852, mean 83.9323 -> 349.8826, missing 12% -> 0%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=7.5653, mean 170.6066 -> 246.084, missing 1% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 84.0306 -> None, missing 16% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 75.5312 -> None, missing 50% -> 100%
Worst categorical columns (by TVD):
- `med_rdoad_syst`: TVD=0.9471, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9361, 14 -> 2 categories, missing 0% -> 0%
- `symptoms_Peripheral_edema_display_pET_any`: TVD=0.8683, 2 -> 1 categories, missing 0% -> 0%
- `med_platelet_history`: TVD=0.8625, 2 -> 1 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first`: TVD=0.856, 2 -> 1 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps5_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=1.0, W/std=2.9947, mean 72.9832 -> 109.9892, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 84.2946 -> None, missing 12% -> 100%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=12.9283, mean 83.9323 -> 331.5534, missing 12% -> 0%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=7.9459, mean 170.6066 -> 249.8802, missing 1% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 84.0306 -> None, missing 16% -> 100%
Worst categorical columns (by TVD):
- `med_insulins_history`: TVD=0.9509, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8993, 14 -> 1 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.8851, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1mo_first`: TVD=0.847, 2 -> 2 categories, missing 0% -> 0%
- `symptoms_Peripheral_edema_display_pET_any`: TVD=0.8437, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps8_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 84.2946 -> None, missing 12% -> 100%
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 83.9323 -> None, missing 12% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=5.6948, mean 170.6066 -> 227.4223, missing 1% -> 0%
- `vital_signs_height_value_last`: KS=1.0, W/std=8.08, mean 170.6693 -> 249.9934, missing 2% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 75.5312 -> None, missing 50% -> 100%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.9929, 5 -> 1 categories, missing 0% -> 0%
- `conditions_osa`: TVD=0.9871, 2 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.9432, 3 -> 3 categories, missing 0% -> 0%
- `med_insulins`: TVD=0.9387, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w7d_first`: TVD=0.9012, 2 -> 1 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=1.0, W/std=None, mean 125.9636 -> None, missing 68% -> 100%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8414, W/std=0.6325, mean 0.3398 -> 1.1454, missing 0% -> 40%
- `vital_signs_diastolicBp_value_last`: KS=0.7495, W/std=4.0032, mean 75.5312 -> 121.9868, missing 50% -> 48%
- `vital_signs_diastolicBp_value_first`: KS=0.7495, W/std=4.0035, mean 75.5312 -> 121.9882, missing 50% -> 48%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.7426, W/std=3.2962, mean 466.2944 -> 605.8072, missing 55% -> 51%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1611, 14 -> 13 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first`: TVD=0.0435, 3 -> 3 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.0424, 2 -> 2 categories, missing 0% -> 0%
- `electrocardiographs_ecg_st_pET`: TVD=0.0402, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0378, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed1]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=1.0, W/std=None, mean 125.9636 -> None, missing 68% -> 100%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8462, W/std=0.6276, mean 0.3398 -> 1.1259, missing 0% -> 39%
- `vital_signs_diastolicBp_value_last`: KS=0.8261, W/std=4.3251, mean 75.5312 -> 129.967, missing 50% -> 45%
- `vital_signs_diastolicBp_value_first`: KS=0.8261, W/std=4.3253, mean 75.5312 -> 129.9623, missing 50% -> 45%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.7844, W/std=3.4727, mean 466.2944 -> 613.2712, missing 55% -> 52%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1293, 14 -> 13 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.0458, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0445, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.0409, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w5a_first`: TVD=0.0404, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed2]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=1.0, W/std=None, mean 125.9636 -> None, missing 68% -> 100%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8296, W/std=0.6366, mean 0.3398 -> 1.1521, missing 0% -> 41%
- `vital_signs_diastolicBp_value_last`: KS=0.8162, W/std=4.2788, mean 75.5312 -> 128.9424, missing 50% -> 45%
- `vital_signs_diastolicBp_value_first`: KS=0.8143, W/std=4.275, mean 75.5312 -> 128.6521, missing 50% -> 45%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.7104, W/std=3.2525, mean 466.2944 -> 596.6123, missing 55% -> 52%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1695, 14 -> 13 categories, missing 0% -> 0%
- `conditions_devices`: TVD=0.0418, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0393, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.0392, 3 -> 3 categories, missing 0% -> 0%
- `electrocardiographs_ecg_st_pET`: TVD=0.0377, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps10_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9955, W/std=7.6306, mean 0.0562 -> 2.9587, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=6.5397, mean 13.8027 -> 59.5607, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9761, W/std=1.8826, mean 0.3398 -> 3.2825, missing 0% -> 0%
- `smoking_status_smoker_startTime_count`: KS=0.9658, W/std=2.2078, mean 0.408 -> 2.6938, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=2.7148, mean 10.3363 -> 62.4031, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0565, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0423, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0333, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0326, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0275, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9955, W/std=7.1212, mean 0.0562 -> 2.7649, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=5.8565, mean 13.8027 -> 54.7804, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9761, W/std=1.9622, mean 0.3398 -> 3.4449, missing 0% -> 0%
- `smoking_status_smoker_startTime_count`: KS=0.9658, W/std=2.2028, mean 0.408 -> 2.6938, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=2.4728, mean 10.3363 -> 57.8165, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0432, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0242, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0234, 5 -> 5 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.0229, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0226, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed1]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9955, W/std=8.0128, mean 0.0562 -> 3.104, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=6.3919, mean 13.8027 -> 58.5271, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9761, W/std=2.1279, mean 0.3398 -> 3.727, missing 0% -> 1%
- `smoking_status_smoker_startTime_count`: KS=0.9658, W/std=2.3588, mean 0.408 -> 2.865, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=2.5068, mean 10.3363 -> 58.5271, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0419, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0318, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.0287, 3 -> 3 categories, missing 0% -> 0%
- `electrocardiographs_ecg_type_of_rhythms_pET_first`: TVD=0.0274, 2 -> 2 categories, missing 0% -> 0%
- `electrocardiographs_ecg_ischemia_without_st_pET`: TVD=0.0266, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed2]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9955, W/std=7.0872, mean 0.0562 -> 2.7519, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=5.543, mean 13.8027 -> 52.584, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9761, W/std=1.9872, mean 0.3398 -> 3.465, missing 0% -> 0%
- `smoking_status_smoker_startTime_count`: KS=0.9658, W/std=2.394, mean 0.408 -> 2.8908, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=2.6789, mean 10.3363 -> 62.0155, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0469, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0345, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0345, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0253, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0231, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps1_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9955, W/std=13.8635, mean 0.0562 -> 5.3295, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=6.0227, mean 13.8027 -> 55.9432, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9761, W/std=2.3686, mean 0.3398 -> 4.1259, missing 0% -> 2%
- `smoking_status_smoker_startTime_count`: KS=0.9658, W/std=5.6605, mean 0.408 -> 6.3243, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=5.1968, mean 10.3363 -> 110.9173, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.1638, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.1471, 5 -> 5 categories, missing 0% -> 0%
- `symptoms_Syncope_display_pET_any`: TVD=0.1092, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first`: TVD=0.108, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.1021, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps20_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9955, W/std=7.0702, mean 0.0562 -> 2.7455, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=5.6996, mean 13.8027 -> 53.6822, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9761, W/std=1.8915, mean 0.3398 -> 3.3155, missing 0% -> 0%
- `smoking_status_smoker_startTime_count`: KS=0.9658, W/std=2.2067, mean 0.408 -> 2.6873, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=2.5594, mean 10.3363 -> 59.4315, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0439, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0363, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first`: TVD=0.0244, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0228, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0227, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps5_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9955, W/std=8.344, mean 0.0562 -> 3.23, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=8.6077, mean 13.8027 -> 74.031, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9761, W/std=1.9476, mean 0.3398 -> 3.4493, missing 0% -> 0%
- `smoking_status_smoker_startTime_count`: KS=0.9658, W/std=3.3551, mean 0.408 -> 3.9147, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=2.9539, mean 10.3363 -> 67.5065, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0549, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0494, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0402, 5 -> 5 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3mo_first`: TVD=0.0384, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.0357, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps8_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9955, W/std=7.3589, mean 0.0562 -> 2.8553, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9935, W/std=5.6261, mean 13.8027 -> 53.1654, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9761, W/std=1.9339, mean 0.3398 -> 3.4197, missing 0% -> 0%
- `smoking_status_smoker_startTime_count`: KS=0.9658, W/std=2.5395, mean 0.408 -> 3.0491, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9606, W/std=3.2247, mean 10.3363 -> 72.6744, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0591, 14 -> 14 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0528, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0326, 5 -> 5 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first`: TVD=0.0299, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.0298, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps15_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.938, W/std=0.7808, mean 0.0562 -> 0.3531, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8844, W/std=2.4757, mean 0.3398 -> 4.2736, missing 0% -> 73%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.7708, W/std=0.2541, mean 5.9206 -> 1.3898, missing 0% -> 0%
- `smoking_status_smoker_startTime_count`: KS=0.765, W/std=0.2521, mean 0.408 -> 0.5585, missing 0% -> 0%
- `electrocardiographs_ecg_qt_duration_corrected_pET_last`: KS=0.746, W/std=2.8091, mean 469.2839 -> 331.7262, missing 55% -> 41%
Worst categorical columns (by TVD):
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.3858, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`: TVD=0.3678, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.3633, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.3588, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.3588, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps1_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=1.0, W/std=None, mean 125.9636 -> None, missing 68% -> 100%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9651, W/std=8.6393, mean 0.0562 -> 3.3423, missing 0% -> 0%
- `lab_results_hdl_value_last`: KS=0.8856, W/std=2.3682, mean 1.0686 -> 0.2567, missing 55% -> 97%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8844, W/std=1.4769, mean 0.3398 -> 2.6068, missing 0% -> 95%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.8639, W/std=3.639, mean 466.2944 -> 282.1672, missing 55% -> 58%
Worst categorical columns (by TVD):
- `med_diuretics_loop_history`: TVD=0.2237, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag_history`: TVD=0.2049, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w1a_first`: TVD=0.1927, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w5a_first`: TVD=0.1786, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w1mo_first`: TVD=0.1565, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps5_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9651, W/std=0.3595, mean 0.0562 -> 0.169, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8844, W/std=2.8712, mean 0.3398 -> 4.9181, missing 0% -> 83%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.7792, W/std=0.3459, mean 5.9206 -> 12.4138, missing 0% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=0.7522, W/std=2.9347, mean 75.5312 -> 37.1973, missing 50% -> 77%
- `lab_results_tropTHs_value_last`: KS=0.7253, W/std=5.8871, mean 1.0632 -> 13.2422, missing 58% -> 63%
Worst categorical columns (by TVD):
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first`: TVD=0.6759, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.6701, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.6584, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.6507, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.6443, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_cap256_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5169, W/std=0.1841, mean 820.2403 -> 894.3628, missing 77% -> 91%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.4035, W/std=0.1438, mean 4.8221 -> 4.0687, missing 1% -> 21%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.3843, W/std=0.0907, mean 5.0222 -> 4.5832, missing 1% -> 24%
- `lab_results_hemoglobin_value_first`: KS=0.3466, W/std=0.4982, mean 122.7778 -> 114.7506, missing 83% -> 93%
- `lab_results_hemoglobin_value_last`: KS=0.299, W/std=0.4311, mean 122.6322 -> 116.1773, missing 83% -> 93%
Worst categorical columns (by TVD):
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.2021, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1924, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1874, 14 -> 14 categories, missing 0% -> 0%
- `med_arb`: TVD=0.1756, 2 -> 2 categories, missing 0% -> 0%
- `med_ccb`: TVD=0.1704, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ep1000_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.4813, W/std=0.1795, mean 820.2403 -> 609.8431, missing 77% -> 86%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.4012, W/std=0.141, mean 4.8221 -> 5.9253, missing 1% -> 22%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.386, W/std=0.1665, mean 5.0222 -> 6.321, missing 1% -> 20%
- `vital_signs_diastolicBp_value_first`: KS=0.2868, W/std=0.3262, mean 75.5312 -> 73.5227, missing 50% -> 44%
- `nyha_nyha`: KS=0.2782, W/std=0.5699, mean 2.3719 -> 2.1143, missing 0% -> 0%
Worst categorical columns (by TVD):
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.1969, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1712, 14 -> 14 categories, missing 0% -> 0%
- `med_ccb`: TVD=0.1666, 2 -> 2 categories, missing 0% -> 0%
- `med_mra`: TVD=0.1626, 2 -> 2 categories, missing 0% -> 0%
- `med_bb`: TVD=0.1523, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ind_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.3967, W/std=0.3057, mean 820.2403 -> 117.2313, missing 77% -> 87%
- `lab_results_ntProBnp_value_last`: KS=0.3537, W/std=0.4164, mean 6807.5675 -> 3325.4002, missing 41% -> 45%
- `lab_results_ntProBnp_value_first`: KS=0.3314, W/std=0.3772, mean 6813.8474 -> 3642.9037, missing 41% -> 44%
- `lab_results_crpNonHs_value_first`: KS=0.3276, W/std=0.3929, mean 40.8452 -> 15.5362, missing 3% -> 0%
- `lab_results_triGly_value_first`: KS=0.3109, W/std=0.3081, mean 1.3407 -> 1.0462, missing 54% -> 56%
Worst categorical columns (by TVD):
- `med_mra`: TVD=0.222, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.2129, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.2097, 2 -> 2 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.2079, 2 -> 2 categories, missing 0% -> 0%
- `med_arb`: TVD=0.1917, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed0]

Worst numeric columns (by KS):
- `nyha_nyha_pET`: KS=0.335, W/std=0.6705, mean 2.3757 -> 2.0756, missing 0% -> 0%
- `nyha_nyha`: KS=0.3202, W/std=0.65, mean 2.3719 -> 2.0723, missing 0% -> 0%
- `lab_results_hba1c_value_last`: KS=0.3027, W/std=0.377, mean 50.7995 -> 51.6181, missing 75% -> 73%
- `lab_results_hba1c_value_first`: KS=0.2835, W/std=0.3474, mean 50.7995 -> 51.3408, missing 75% -> 73%
- `lab_results_tropTHs_value_first`: KS=0.2475, W/std=0.3137, mean 1.0572 -> 0.5485, missing 58% -> 59%
Worst categorical columns (by TVD):
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.2098, 2 -> 2 categories, missing 0% -> 0%
- `med_ccb`: TVD=0.1814, 2 -> 2 categories, missing 0% -> 0%
- `med_bb`: TVD=0.1762, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1762, 2 -> 2 categories, missing 0% -> 0%
- `med_arb`: TVD=0.1717, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed1]

Worst numeric columns (by KS):
- `lab_results_hemoglobin_value_first`: KS=0.4515, W/std=0.6641, mean 122.7778 -> 135.4422, missing 83% -> 95%
- `lab_results_hemoglobin_value_last`: KS=0.4291, W/std=0.6545, mean 122.6322 -> 134.7661, missing 83% -> 95%
- `nyha_nyha`: KS=0.3357, W/std=0.6796, mean 2.3719 -> 2.0568, missing 0% -> 0%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.2577, W/std=0.1545, mean 820.2403 -> 639.1483, missing 77% -> 85%
- `lab_results_hba1c_value_last`: KS=0.2537, W/std=0.2997, mean 50.7995 -> 49.9967, missing 75% -> 72%
Worst categorical columns (by TVD):
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.2111, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.211, 2 -> 2 categories, missing 0% -> 0%
- `med_bb`: TVD=0.1949, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.182, 2 -> 2 categories, missing 0% -> 0%
- `med_arb`: TVD=0.1743, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed2]

Worst numeric columns (by KS):
- `lab_results_hemoglobin_value_last`: KS=0.3646, W/std=0.5799, mean 122.6322 -> 134.4531, missing 83% -> 93%
- `lab_results_hemoglobin_value_first`: KS=0.351, W/std=0.6086, mean 122.7778 -> 137.3522, missing 83% -> 93%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.3505, W/std=0.2407, mean 125.9636 -> 128.2644, missing 68% -> 66%
- `lab_results_tropTHs_value_last`: KS=0.2957, W/std=0.2939, mean 1.0632 -> 0.6812, missing 58% -> 69%
- `nyha_nyha`: KS=0.2801, W/std=0.5736, mean 2.3719 -> 2.1124, missing 0% -> 0%
Worst categorical columns (by TVD):
- `med_anti_coag`: TVD=0.2207, 2 -> 2 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.2137, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1975, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.1955, 2 -> 2 categories, missing 0% -> 0%
- `med_mra`: TVD=0.1916, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5156, W/std=0.243, mean 820.2403 -> 816.8397, missing 77% -> 94%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.4335, W/std=0.1527, mean 4.8221 -> 4.194, missing 1% -> 23%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.3651, W/std=0.1388, mean 5.0222 -> 5.3024, missing 1% -> 23%
- `lab_results_crpNonHs_value_last`: KS=0.2857, W/std=0.3025, mean 40.7741 -> 21.7912, missing 3% -> 11%
- `electrocardiographs_ecg_qrs_duration_pET_last`: KS=0.2848, W/std=0.3602, mean 125.0331 -> 111.6555, missing 55% -> 49%
Worst categorical columns (by TVD):
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.2137, 2 -> 2 categories, missing 0% -> 0%
- `med_mra`: TVD=0.213, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1988, 2 -> 2 categories, missing 0% -> 0%
- `med_ccb`: TVD=0.1801, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.1775, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed1]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5164, W/std=0.4063, mean 820.2403 -> 1545.0544, missing 77% -> 92%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.3957, W/std=0.1674, mean 4.8221 -> 3.827, missing 1% -> 21%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.3888, W/std=0.1208, mean 5.0222 -> 4.307, missing 1% -> 21%
- `nyha_nyha`: KS=0.3021, W/std=0.6156, mean 2.3719 -> 2.0905, missing 0% -> 0%
- `lab_results_crpNonHs_value_first`: KS=0.2954, W/std=0.265, mean 40.8452 -> 24.2191, missing 3% -> 13%
Worst categorical columns (by TVD):
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.2079, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1853, 14 -> 14 categories, missing 0% -> 0%
- `med_arb`: TVD=0.184, 2 -> 2 categories, missing 0% -> 0%
- `med_ccb`: TVD=0.1795, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1736, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed2]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5368, W/std=0.2722, mean 820.2403 -> 1210.4995, missing 77% -> 94%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.4082, W/std=0.1862, mean 5.0222 -> 5.7008, missing 1% -> 17%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.3912, W/std=0.161, mean 4.8221 -> 4.1501, missing 1% -> 22%
- `nyha_nyha`: KS=0.3337, W/std=0.6759, mean 2.3719 -> 2.0588, missing 0% -> 0%
- `nyha_nyha_pET`: KS=0.3266, W/std=0.6553, mean 2.3757 -> 2.084, missing 0% -> 0%
Worst categorical columns (by TVD):
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.2117, 2 -> 2 categories, missing 0% -> 0%
- `med_arb`: TVD=0.1956, 2 -> 1 categories, missing 0% -> 0%
- `med_diuretics_loop`: TVD=0.1853, 2 -> 2 categories, missing 0% -> 0%
- `med_ccb`: TVD=0.1821, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1769, 2 -> 2 categories, missing 0% -> 0%
