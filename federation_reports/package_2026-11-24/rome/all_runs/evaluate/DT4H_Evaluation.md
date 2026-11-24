# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 104 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 166 | 0.0001 | 0.0 | 1.0 | 0.0009 | 0.0 | 1.0 | 0.0001 |
| train vs holdout | 151 | 0.0308 | 0.0294 | 1.0 | 0.0527 | 0.0122 | 0.9884 | 0.0088 |
| train vs synthetic[aim40_eps1_seed0] | 40 | 0.5207 | 0.4296 | 0.0455 | 1.3587 | 0.018 | 0.9444 | 0.0054 |
| train vs synthetic[aim40_eps5_seed0] | 40 | 0.519 | 0.423 | 0.0455 | 1.2158 | 0.0101 | 1.0 | 0.0028 |
| train vs synthetic[aim50_eps1_seed0] | 50 | 0.5236 | 0.4424 | 0.0357 | 1.254 | 0.0146 | 1.0 | 0.0053 |
| train vs synthetic[ctgan_qt_seed0] | 151 | 0.2146 | 0.1981 | 0.2154 | 0.6526 | 0.104 | 0.2093 | 0.0951 |
| train vs synthetic[ctgan_seed0] | 151 | 0.3175 | 0.3218 | 0.0769 | 0.6558 | 0.0808 | 0.3372 | 0.0814 |
| train vs synthetic[ctgan_seed1] | 151 | 0.2727 | 0.2615 | 0.0615 | 0.5695 | 0.0894 | 0.2093 | 0.0726 |
| train vs synthetic[ctgan_seed2] | 151 | 0.2724 | 0.257 | 0.1077 | 0.5627 | 0.076 | 0.3256 | 0.0763 |
| train vs synthetic[ddpm_g_seed0] | 151 | 0.3675 | 0.2126 | 0.3077 | 1.5916 | 0.1105 | 0.5465 | 0.1071 |
| train vs synthetic[ddpm_seed0] | 151 | 0.3465 | 0.1971 | 0.3538 | 1.6242 | 0.0263 | 0.814 | 0.0973 |
| train vs synthetic[ddpm_seed1] | 151 | 0.3746 | 0.2172 | 0.3385 | 1.7021 | 0.0417 | 0.6628 | 0.0841 |
| train vs synthetic[ddpm_seed2] | 151 | 0.373 | 0.2363 | 0.3385 | 1.7074 | 0.03 | 0.7791 | 0.0969 |
| train vs synthetic[dpctgan_eps10_seed0] | 151 | 0.8581 | 0.9973 | 0.0 | 4.0597 | 0.2195 | 0.4302 | 0.2771 |
| train vs synthetic[dpctgan_eps15_seed0] | 151 | 0.7971 | 0.952 | 0.0 | 3.3035 | 0.1869 | 0.4535 | 0.2436 |
| train vs synthetic[dpctgan_eps15_seed1] | 151 | 0.8543 | 0.9882 | 0.0 | 5.1926 | 0.1821 | 0.4419 | 0.303 |
| train vs synthetic[dpctgan_eps15_seed2] | 151 | 0.841 | 0.9934 | 0.0 | 4.425 | 0.24 | 0.4535 | 0.2688 |
| train vs synthetic[dpctgan_eps1_seed0] | 151 | 0.8923 | 1.0 | 0.0 | 3.9387 | 0.1998 | 0.4302 | 0.3291 |
| train vs synthetic[dpctgan_eps20_seed0] | 151 | 0.8562 | 0.9933 | 0.0 | 10.5964 | 0.2331 | 0.4419 | 0.3105 |
| train vs synthetic[dpctgan_eps5_seed0] | 151 | 0.8243 | 0.9979 | 0.0 | 4.5133 | 0.1911 | 0.4651 | 0.325 |
| train vs synthetic[dpctgan_eps8_seed0] | 151 | 0.831 | 0.9804 | 0.0 | 4.438 | 0.2477 | 0.4302 | 0.3102 |
| train vs synthetic[gaussian_copula_seed0] | 151 | 0.3757 | 0.3109 | 0.2154 | 1.3468 | 0.0098 | 0.9884 | 0.1291 |
| train vs synthetic[gaussian_copula_seed1] | 151 | 0.3829 | 0.3478 | 0.2154 | 1.3637 | 0.0097 | 0.9884 | 0.1284 |
| train vs synthetic[gaussian_copula_seed2] | 151 | 0.3847 | 0.3176 | 0.2154 | 1.3724 | 0.009 | 0.9884 | 0.1282 |
| train vs synthetic[mst_eps0p5_seed0] | 151 | 0.5567 | 0.4881 | 0.0308 | 3.053 | 0.0307 | 0.8953 | 0.0225 |
| train vs synthetic[mst_eps10_seed0] | 151 | 0.5336 | 0.4635 | 0.0308 | 1.9153 | 0.0058 | 1.0 | 0.0054 |
| train vs synthetic[mst_eps15_seed0] | 151 | 0.5335 | 0.4608 | 0.0308 | 1.8655 | 0.0052 | 1.0 | 0.0044 |
| train vs synthetic[mst_eps15_seed1] | 151 | 0.5324 | 0.458 | 0.0308 | 1.8605 | 0.0051 | 1.0 | 0.0046 |
| train vs synthetic[mst_eps15_seed2] | 151 | 0.5331 | 0.4581 | 0.0308 | 1.8665 | 0.0056 | 1.0 | 0.0046 |
| train vs synthetic[mst_eps1_seed0] | 151 | 0.5388 | 0.4745 | 0.0308 | 2.4099 | 0.016 | 1.0 | 0.0152 |
| train vs synthetic[mst_eps20_seed0] | 151 | 0.5332 | 0.4593 | 0.0308 | 1.8478 | 0.0061 | 1.0 | 0.0043 |
| train vs synthetic[mst_eps5_seed0] | 151 | 0.5344 | 0.4503 | 0.0308 | 1.9666 | 0.0076 | 1.0 | 0.0057 |
| train vs synthetic[mst_eps8_seed0] | 151 | 0.5341 | 0.4549 | 0.0308 | 1.8973 | 0.006 | 1.0 | 0.0041 |
| train vs synthetic[patectgan_eps15_seed0] | 151 | 0.4214 | 0.3835 | 0.0154 | 2.5968 | 0.1255 | 0.6163 | 0.135 |
| train vs synthetic[patectgan_eps1_seed0] | 151 | 0.4931 | 0.426 | 0.0154 | 3.7124 | 0.1135 | 0.186 | 0.167 |
| train vs synthetic[patectgan_eps5_seed0] | 151 | 0.4427 | 0.4314 | 0.0154 | 2.244 | 0.1545 | 0.5233 | 0.1097 |
| train vs synthetic[tvae_cap256_seed0] | 151 | 0.129 | 0.1171 | 0.3385 | 0.186 | 0.0332 | 0.7558 | 0.0242 |
| train vs synthetic[tvae_ep1000_seed0] | 151 | 0.1431 | 0.1359 | 0.3231 | 0.2076 | 0.0372 | 0.6744 | 0.0226 |
| train vs synthetic[tvae_ind_seed0] | 151 | 0.1477 | 0.1328 | 0.3077 | 0.2133 | 0.0362 | 0.7209 | 0.0225 |
| train vs synthetic[tvae_qt_seed0] | 151 | 0.1467 | 0.1295 | 0.3538 | 0.2389 | 0.0349 | 0.686 | 0.0285 |
| train vs synthetic[tvae_qt_seed1] | 151 | 0.1353 | 0.113 | 0.4154 | 0.2234 | 0.0426 | 0.6279 | 0.0396 |
| train vs synthetic[tvae_qt_seed2] | 151 | 0.1496 | 0.1263 | 0.3538 | 0.2417 | 0.04 | 0.6163 | 0.0475 |
| train vs synthetic[tvae_seed0] | 151 | 0.1491 | 0.133 | 0.3692 | 0.2198 | 0.0345 | 0.7674 | 0.0297 |
| train vs synthetic[tvae_seed1] | 151 | 0.1391 | 0.1242 | 0.3385 | 0.2032 | 0.0336 | 0.7326 | 0.0247 |
| train vs synthetic[tvae_seed2] | 151 | 0.1536 | 0.1264 | 0.2154 | 0.2393 | 0.0375 | 0.7093 | 0.0256 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| aim | 1 | 1 | 0.5236 | 0.0146 | 0.0053 |
| aim40 | 1 | 1 | 0.5207 | 0.018 | 0.0054 |
| aim40 | 5 | 1 | 0.519 | 0.0101 | 0.0028 |
| ctgan | - | 3 | 0.2875 ± 0.026 | 0.0821 ± 0.0068 | 0.0768 ± 0.0044 |
| ctgan_qt | - | 1 | 0.2146 | 0.104 | 0.0951 |
| ddpm | - | 3 | 0.3647 ± 0.0158 | 0.0327 ± 0.008 | 0.0928 ± 0.0075 |
| ddpm_g | - | 1 | 0.3675 | 0.1105 | 0.1071 |
| dpctgan | 1 | 1 | 0.8923 | 0.1998 | 0.3291 |
| dpctgan | 5 | 1 | 0.8243 | 0.1911 | 0.325 |
| dpctgan | 8 | 1 | 0.831 | 0.2477 | 0.3102 |
| dpctgan | 10 | 1 | 0.8581 | 0.2195 | 0.2771 |
| dpctgan | 15 | 3 | 0.8308 ± 0.0299 | 0.203 ± 0.0321 | 0.2718 ± 0.0298 |
| dpctgan | 20 | 1 | 0.8562 | 0.2331 | 0.3105 |
| gaussian_copula | - | 3 | 0.3811 ± 0.0048 | 0.0095 ± 0.0004 | 0.1286 ± 0.0005 |
| mst | 0.5 | 1 | 0.5567 | 0.0307 | 0.0225 |
| mst | 1 | 1 | 0.5388 | 0.016 | 0.0152 |
| mst | 5 | 1 | 0.5344 | 0.0076 | 0.0057 |
| mst | 8 | 1 | 0.5341 | 0.006 | 0.0041 |
| mst | 10 | 1 | 0.5336 | 0.0058 | 0.0054 |
| mst | 15 | 3 | 0.533 ± 0.0006 | 0.0053 ± 0.0003 | 0.0045 ± 0.0001 |
| mst | 20 | 1 | 0.5332 | 0.0061 | 0.0043 |
| patectgan | 1 | 1 | 0.4931 | 0.1135 | 0.167 |
| patectgan | 5 | 1 | 0.4427 | 0.1545 | 0.1097 |
| patectgan | 15 | 1 | 0.4214 | 0.1255 | 0.135 |
| tvae | - | 3 | 0.1473 ± 0.0074 | 0.0352 ± 0.002 | 0.0267 ± 0.0027 |
| tvae_cap256 | - | 1 | 0.129 | 0.0332 | 0.0242 |
| tvae_ep1000 | - | 1 | 0.1431 | 0.0372 | 0.0226 |
| tvae_ind | - | 1 | 0.1477 | 0.0362 | 0.0225 |
| tvae_qt | - | 3 | 0.1439 ± 0.0076 | 0.0392 ± 0.0039 | 0.0385 ± 0.0095 |

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.5177**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| aim40_eps1_seed0 | 1.0 | 0.0 | 0.2649 |
| aim40_eps5_seed0 | 1.0 | 0.0 | 0.2649 |
| aim50_eps1_seed0 | 1.0 | 0.0 | 0.3311 |
| ctgan_qt_seed0 | 1.0 | 0.0 | 1.0 |
| ctgan_seed0 | 1.0 | 0.0 | 1.0 |
| ctgan_seed1 | 1.0 | 0.0 | 1.0 |
| ctgan_seed2 | 1.0 | 0.0 | 1.0 |
| ddpm_g_seed0 | 1.0 | 0.0 | 1.0 |
| ddpm_seed0 | 0.997 | 0.0014 | 1.0 |
| ddpm_seed1 | 0.9977 | 0.0012 | 1.0 |
| ddpm_seed2 | 0.9976 | 0.0009 | 1.0 |
| dpctgan_eps10_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps15_seed0 | 0.9998 | 0.0003 | 1.0 |
| dpctgan_eps15_seed1 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps15_seed2 | 0.9998 | 0.0003 | 1.0 |
| dpctgan_eps1_seed0 | 0.9998 | 0.0003 | 1.0 |
| dpctgan_eps20_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps5_seed0 | 0.9998 | 0.0003 | 1.0 |
| dpctgan_eps8_seed0 | 0.9998 | 0.0003 | 1.0 |
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
| tvae_cap256_seed0 | 0.999 | 0.001 | 1.0 |
| tvae_ep1000_seed0 | 0.9993 | 0.0003 | 1.0 |
| tvae_ind_seed0 | 0.9971 | 0.001 | 1.0 |
| tvae_qt_seed0 | 0.9965 | 0.0013 | 1.0 |
| tvae_qt_seed1 | 0.9948 | 0.0009 | 1.0 |
| tvae_qt_seed2 | 0.9965 | 0.0021 | 1.0 |
| tvae_seed0 | 0.9989 | 0.0005 | 1.0 |
| tvae_seed1 | 0.9987 | 0.0005 | 1.0 |
| tvae_seed2 | 0.9992 | 0.0009 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.055 | 0.0382 | 0.0589 | 0.0491 | 0.0503 |
| aim40_eps1_seed0 | 0.5912 | 0.5451 | 0.6571 | 0.5486 | 0.5961 |
| aim40_eps5_seed0 | 0.5681 | 0.5483 | 0.6191 | 0.5513 | 0.5866 |
| aim50_eps1_seed0 | 0.6352 | 0.5766 | 0.7326 | 0.5729 | 0.6665 |
| ctgan_qt_seed0 | 0.2394 | 0.2204 | 0.255 | 0.2264 | 0.2394 |
| ctgan_seed0 | 0.3259 | 0.3268 | 0.3392 | 0.3266 | 0.3393 |
| ctgan_seed1 | 0.2874 | 0.2754 | 0.3052 | 0.286 | 0.2718 |
| ctgan_seed2 | 0.278 | 0.2825 | 0.3112 | 0.2801 | 0.2841 |
| ddpm_g_seed0 | 0.3898 | 0.3734 | 0.3754 | 0.3768 | 0.3949 |
| ddpm_seed0 | 0.3647 | 0.353 | 0.352 | 0.3514 | 0.3729 |
| ddpm_seed1 | 0.3877 | 0.3754 | 0.3788 | 0.3877 | 0.3868 |
| ddpm_seed2 | 0.3875 | 0.3813 | 0.3959 | 0.3803 | 0.3884 |
| dpctgan_eps10_seed0 | - | 0.8581 | - | 0.8621 | 0.8659 |
| dpctgan_eps15_seed0 | 0.8054 | 0.8157 | 0.8017 | 0.795 | - |
| dpctgan_eps15_seed1 | - | 0.8522 | 0.8523 | 0.8559 | - |
| dpctgan_eps15_seed2 | - | 0.8449 | 0.8567 | 0.8423 | 0.8464 |
| dpctgan_eps1_seed0 | - | 0.8926 | 0.8954 | - | - |
| dpctgan_eps20_seed0 | - | 0.8569 | 0.8589 | - | - |
| dpctgan_eps5_seed0 | 0.8271 | - | - | 0.8203 | 0.8294 |
| dpctgan_eps8_seed0 | - | 0.8327 | 0.8449 | 0.8353 | 0.8385 |
| gaussian_copula_seed0 | 0.3976 | 0.392 | 0.4281 | 0.3975 | 0.3955 |
| gaussian_copula_seed1 | 0.397 | 0.3935 | 0.4332 | 0.3962 | 0.4002 |
| gaussian_copula_seed2 | 0.4055 | 0.3953 | 0.4364 | 0.4058 | 0.4021 |
| mst_eps0p5_seed0 | 0.5648 | 0.5687 | 0.6085 | 0.5841 | 0.5886 |
| mst_eps10_seed0 | 0.5567 | 0.5447 | 0.6354 | 0.5504 | 0.6178 |
| mst_eps15_seed0 | 0.5568 | 0.5431 | 0.6488 | 0.553 | 0.6141 |
| mst_eps15_seed1 | 0.5574 | 0.5434 | 0.6248 | 0.5551 | 0.6162 |
| mst_eps15_seed2 | 0.5585 | 0.5418 | 0.6268 | 0.5529 | 0.6145 |
| mst_eps1_seed0 | 0.5581 | 0.551 | 0.5881 | 0.5819 | 0.5923 |
| mst_eps20_seed0 | 0.558 | 0.5431 | 0.6493 | 0.5521 | 0.6194 |
| mst_eps5_seed0 | 0.5549 | 0.5435 | 0.5824 | 0.5933 | 0.6097 |
| mst_eps8_seed0 | 0.5525 | 0.5417 | 0.5859 | 0.5908 | 0.5984 |
| patectgan_eps15_seed0 | 0.4367 | 0.4296 | 0.4554 | 0.4268 | 0.4358 |
| patectgan_eps1_seed0 | 0.4981 | 0.5021 | 0.5401 | 0.5024 | 0.4952 |
| patectgan_eps5_seed0 | 0.447 | 0.4468 | 0.4898 | 0.4618 | 0.4483 |
| tvae_cap256_seed0 | 0.1525 | 0.1296 | 0.1449 | 0.1392 | 0.1495 |
| tvae_ep1000_seed0 | 0.1525 | 0.1496 | 0.1691 | 0.1489 | 0.1529 |
| tvae_ind_seed0 | 0.165 | 0.1583 | 0.168 | 0.1492 | 0.1679 |
| tvae_qt_seed0 | 0.1716 | 0.1515 | 0.1602 | 0.1479 | 0.1747 |
| tvae_qt_seed1 | 0.1649 | 0.1399 | 0.1507 | 0.1425 | 0.1551 |
| tvae_qt_seed2 | 0.1764 | 0.1679 | 0.16 | 0.1528 | 0.1843 |
| tvae_seed0 | 0.1679 | 0.1549 | 0.1761 | 0.1582 | 0.1596 |
| tvae_seed1 | 0.1656 | 0.1404 | 0.1609 | 0.1437 | 0.1587 |
| tvae_seed2 | 0.1733 | 0.1597 | 0.1841 | 0.159 | 0.1642 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5207 | 0.5202 | 0.018 | 0.0255 |
| aim40_eps5_seed0 | 0.519 | 0.5204 | 0.0101 | 0.0204 |
| aim50_eps1_seed0 | 0.5236 | 0.5233 | 0.0146 | 0.0206 |
| ctgan_qt_seed0 | 0.2146 | 0.2138 | 0.104 | 0.1143 |
| ctgan_seed0 | 0.3175 | 0.3153 | 0.0808 | 0.0896 |
| ctgan_seed1 | 0.2727 | 0.2681 | 0.0894 | 0.0997 |
| ctgan_seed2 | 0.2724 | 0.2741 | 0.076 | 0.0848 |
| ddpm_g_seed0 | 0.3675 | 0.3702 | 0.1105 | 0.1027 |
| ddpm_seed0 | 0.3465 | 0.3484 | 0.0263 | 0.0201 |
| ddpm_seed1 | 0.3746 | 0.374 | 0.0417 | 0.0352 |
| ddpm_seed2 | 0.373 | 0.3739 | 0.03 | 0.0252 |
| dpctgan_eps10_seed0 | 0.8581 | 0.859 | 0.2195 | 0.2138 |
| dpctgan_eps15_seed0 | 0.7971 | 0.7976 | 0.1869 | 0.1802 |
| dpctgan_eps15_seed1 | 0.8543 | 0.8543 | 0.1821 | 0.1756 |
| dpctgan_eps15_seed2 | 0.841 | 0.8411 | 0.24 | 0.2352 |
| dpctgan_eps1_seed0 | 0.8923 | 0.8897 | 0.1998 | 0.195 |
| dpctgan_eps20_seed0 | 0.8562 | 0.8567 | 0.2331 | 0.229 |
| dpctgan_eps5_seed0 | 0.8243 | 0.8259 | 0.1911 | 0.1839 |
| dpctgan_eps8_seed0 | 0.831 | 0.8302 | 0.2477 | 0.2455 |
| gaussian_copula_seed0 | 0.3757 | 0.3757 | 0.0098 | 0.0161 |
| gaussian_copula_seed1 | 0.3829 | 0.38 | 0.0097 | 0.0166 |
| gaussian_copula_seed2 | 0.3847 | 0.381 | 0.009 | 0.0159 |
| mst_eps0p5_seed0 | 0.5567 | 0.5559 | 0.0307 | 0.0325 |
| mst_eps10_seed0 | 0.5336 | 0.5377 | 0.0058 | 0.0121 |
| mst_eps15_seed0 | 0.5335 | 0.5368 | 0.0052 | 0.0123 |
| mst_eps15_seed1 | 0.5324 | 0.5361 | 0.0051 | 0.012 |
| mst_eps15_seed2 | 0.5331 | 0.5366 | 0.0056 | 0.012 |
| mst_eps1_seed0 | 0.5388 | 0.5405 | 0.016 | 0.0193 |
| mst_eps20_seed0 | 0.5332 | 0.5365 | 0.0061 | 0.0121 |
| mst_eps5_seed0 | 0.5344 | 0.5381 | 0.0076 | 0.0129 |
| mst_eps8_seed0 | 0.5341 | 0.5367 | 0.006 | 0.0122 |
| patectgan_eps15_seed0 | 0.4214 | 0.4242 | 0.1255 | 0.1255 |
| patectgan_eps1_seed0 | 0.4931 | 0.4867 | 0.1135 | 0.1213 |
| patectgan_eps5_seed0 | 0.4427 | 0.4393 | 0.1545 | 0.1519 |
| tvae_cap256_seed0 | 0.129 | 0.1295 | 0.0332 | 0.034 |
| tvae_ep1000_seed0 | 0.1431 | 0.14 | 0.0372 | 0.0369 |
| tvae_ind_seed0 | 0.1477 | 0.1495 | 0.0362 | 0.0355 |
| tvae_qt_seed0 | 0.1467 | 0.1435 | 0.0349 | 0.0374 |
| tvae_qt_seed1 | 0.1353 | 0.1389 | 0.0426 | 0.045 |
| tvae_qt_seed2 | 0.1496 | 0.1534 | 0.04 | 0.0408 |
| tvae_seed0 | 0.1491 | 0.1447 | 0.0345 | 0.0345 |
| tvae_seed1 | 0.1391 | 0.1406 | 0.0336 | 0.0343 |
| tvae_seed2 | 0.1536 | 0.1517 | 0.0375 | 0.0378 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 2076 | 0.0385 | 0.0319 | 0.9485 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 3486 | 0.0257 | 0.0156 | 0.9725 | 0 | - |
| *noise floor* | corr-ratio (num-cat) | 12350 | 0.0118 | 0.0 | 0.995 | 0 | - |
| aim40_eps1_seed0 | Spearman (num-num) | 230 | 0.1772 | 0.0983 | 0.5043 | 5 | `lab_results_hemoglobin_value_first|lab_results_bun_value_last` (-0.232 -> 0.7501) |
| aim40_eps1_seed0 | Cramer's V (cat-cat) | 153 | 0.149 | 0.0674 | 0.549 | 4 | `ckd_severity_from_calculated_egfr|conditions_cm` (0.0448 -> 0.64) |
| aim40_eps1_seed0 | corr-ratio (num-cat) | 2684 | 0.0203 | 0.0 | 0.9426 | 23 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.0703) |
| aim40_eps5_seed0 | Spearman (num-num) | 230 | 0.2016 | 0.1487 | 0.4 | 4 | `lab_results_creatBS_value_p3a_avg|lab_results_validSerumCreatinine_value_pET` (0.8512 -> -0.0094) |
| aim40_eps5_seed0 | Cramer's V (cat-cat) | 153 | 0.1476 | 0.0621 | 0.6275 | 10 | `patient_demographics_gender|conditions_dysl` (0.0742 -> 0.7292) |
| aim40_eps5_seed0 | corr-ratio (num-cat) | 2684 | 0.0222 | 0.0 | 0.9352 | 24 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.0448) |
| aim50_eps1_seed0 | Spearman (num-num) | 377 | 0.2385 | 0.1315 | 0.4271 | 16 | `patient_demographics_age|lab_results_hemoglobin_value_first` (-0.2703 -> 0.6982) |
| aim50_eps1_seed0 | Cramer's V (cat-cat) | 231 | 0.1128 | 0.046 | 0.6753 | 8 | `ckd_severity_categorizedValue|conditions_ihd` (0.0547 -> 0.5936) |
| aim50_eps1_seed0 | corr-ratio (num-cat) | 3528 | 0.0267 | 0.0 | 0.9289 | 29 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0047) |
| ctgan_qt_seed0 | Spearman (num-num) | 2044 | 0.1099 | 0.072 | 0.6438 | 0 | `echocardiographs_lvef_pET_first|echocardiographs_lvef_pET_last` (0.9451 -> -0.0622) |
| ctgan_qt_seed0 | Cramer's V (cat-cat) | 3655 | 0.1053 | 0.0659 | 0.6189 | 4 | `ckd_severity_from_calculated_egfr|ckd_severity_calculated_or_measured` (1.0 -> 0.0987) |
| ctgan_qt_seed0 | corr-ratio (num-cat) | 12350 | 0.03 | 0.0 | 0.9101 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.1351) |
| ctgan_seed0 | Spearman (num-num) | 2060 | 0.1085 | 0.0744 | 0.633 | 0 | `vital_signs_weight_value_p6mo_last|vital_signs_weight_value_last` (1.0 -> -0.135) |
| ctgan_seed0 | Cramer's V (cat-cat) | 3655 | 0.0879 | 0.0474 | 0.7119 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.0116) |
| ctgan_seed0 | corr-ratio (num-cat) | 12350 | 0.027 | 0.0 | 0.9249 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9054 -> 0.1182) |
| ctgan_seed1 | Spearman (num-num) | 2061 | 0.1085 | 0.0758 | 0.6298 | 0 | `vital_signs_weight_value_p6mo_last|vital_signs_weight_value_last` (1.0 -> -0.1111) |
| ctgan_seed1 | Cramer's V (cat-cat) | 3655 | 0.1001 | 0.0601 | 0.6585 | 0 | `ckd_severity_from_calculated_egfr|ckd_severity_calculated_or_measured` (1.0 -> 0.1104) |
| ctgan_seed1 | corr-ratio (num-cat) | 12350 | 0.0301 | 0.0 | 0.9065 | 2 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.1474) |
| ctgan_seed2 | Spearman (num-num) | 2041 | 0.1097 | 0.0702 | 0.6365 | 0 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.9913 -> -0.0961) |
| ctgan_seed2 | Cramer's V (cat-cat) | 3570 | 0.1139 | 0.0676 | 0.6025 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isAllCause_f5a_w7d_first` (1.0 -> 0.0048) |
| ctgan_seed2 | corr-ratio (num-cat) | 12350 | 0.0269 | 0.0 | 0.9202 | 1 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.076) |
| ddpm_g_seed0 | Spearman (num-num) | 1781 | 0.1003 | 0.06 | 0.6951 | 35 | `electrocardiographs_ecg_qrs_duration_pET_last|electrocardiographs_ecg_qrs_axis_pET_last` (-0.2732 -> 0.7592) |
| ddpm_g_seed0 | Cramer's V (cat-cat) | 780 | 0.0775 | 0.0336 | 0.7538 | 0 | `cause_of_death_isRenal_f5a_w5a_first|cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first` (0.8802 -> 0.3455) |
| ddpm_g_seed0 | corr-ratio (num-cat) | 11590 | 0.0192 | 0.0 | 0.9695 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0) |
| ddpm_seed0 | Spearman (num-num) | 2011 | 0.0951 | 0.059 | 0.72 | 34 | `electrocardiographs_ecg_qrs_duration_pET_last|electrocardiographs_ecg_qrs_axis_pET_last` (-0.2732 -> 0.7852) |
| ddpm_seed0 | Cramer's V (cat-cat) | 2926 | 0.0504 | 0.0286 | 0.8855 | 0 | `encounter_primary_reason_CV_Disease_f5a_w1mo_first|encounter_primary_reason_CV_Disease_f5a_w3mo_first` (0.6646 -> 0.2782) |
| ddpm_seed0 | corr-ratio (num-cat) | 12350 | 0.0178 | 0.0 | 0.9649 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (0.6697 -> 0.0) |
| ddpm_seed1 | Spearman (num-num) | 2009 | 0.0979 | 0.0556 | 0.7143 | 50 | `electrocardiographs_ecg_qrs_duration_pET_last|electrocardiographs_ecg_qrs_axis_pET_last` (-0.2732 -> 0.9217) |
| ddpm_seed1 | Cramer's V (cat-cat) | 2926 | 0.0376 | 0.0264 | 0.933 | 0 | `encounter_primary_reason_HF_Disease_f5a_w7d_first|encounter_primary_reason_HF_Disease_f5a_w1mo_first` (0.6352 -> 0.3772) |
| ddpm_seed1 | corr-ratio (num-cat) | 12350 | 0.0162 | 0.0 | 0.9778 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_HF_Disease_f5a_w1a_first` (0.8295 -> 0.0136) |
| ddpm_seed2 | Spearman (num-num) | 2010 | 0.1013 | 0.061 | 0.707 | 48 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2471 -> 0.8705) |
| ddpm_seed2 | Cramer's V (cat-cat) | 2926 | 0.0415 | 0.0226 | 0.8917 | 0 | `cause_of_death_isCV_f5a_w3mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first` (1.0 -> 0.6109) |
| ddpm_seed2 | corr-ratio (num-cat) | 12350 | 0.0171 | 0.0 | 0.973 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_HF_Disease_f5a_w1a_first` (0.8437 -> 0.0502) |
| dpctgan_eps10_seed0 | Spearman (num-num) | 891 | 0.2745 | 0.1882 | 0.2682 | 88 | `lab_results_validSerumCreatinine_value_first|lab_results_creatBS_value_p3a_avg` (0.9039 -> -0.422) |
| dpctgan_eps10_seed0 | Cramer's V (cat-cat) | 1326 | 0.1682 | 0.0546 | 0.7074 | 0 | `cause_of_death_isCV_f5a_w3a_first|cause_of_death_isCV_f5a_w5a_first` (1.0 -> 0.0011) |
| dpctgan_eps10_seed0 | corr-ratio (num-cat) | 8170 | 0.0239 | 0.0 | 0.954 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0) |
| dpctgan_eps15_seed0 | Spearman (num-num) | 1078 | 0.2358 | 0.1413 | 0.372 | 81 | `lab_results_validSerumCreatinine_value_first|lab_results_creatBS_value_p3a_avg` (0.9039 -> -0.5963) |
| dpctgan_eps15_seed0 | Cramer's V (cat-cat) | 990 | 0.1401 | 0.0483 | 0.7939 | 0 | `cause_of_death_isRenal_f5a_w3a_first|cause_of_death_isAllCause_f5a_w3a_first` (1.0 -> 0.0008) |
| dpctgan_eps15_seed0 | corr-ratio (num-cat) | 8930 | 0.025 | 0.0 | 0.9513 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0) |
| dpctgan_eps15_seed1 | Spearman (num-num) | 860 | 0.2241 | 0.1202 | 0.4337 | 64 | `lab_results_validSerumCreatinine_value_first|lab_results_creatBS_value_p3a_avg` (0.9039 -> -0.3102) |
| dpctgan_eps15_seed1 | Cramer's V (cat-cat) | 1035 | 0.1685 | 0.0515 | 0.7072 | 3 | `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (1.0 -> 0.0008) |
| dpctgan_eps15_seed1 | corr-ratio (num-cat) | 7980 | 0.0251 | 0.0 | 0.9497 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0) |
| dpctgan_eps15_seed2 | Spearman (num-num) | 1032 | 0.2452 | 0.1655 | 0.3459 | 70 | `encounters_lengthOfStay|encounters_numOfPreviousHFStays_count` (-0.0865 -> 0.9859) |
| dpctgan_eps15_seed2 | Cramer's V (cat-cat) | 1326 | 0.158 | 0.0495 | 0.7323 | 1 | `cause_of_death_isCV_f5a_w3a_first|cause_of_death_isRenal_f5a_w5a_first` (1.0 -> 0.0008) |
| dpctgan_eps15_seed2 | corr-ratio (num-cat) | 8740 | 0.0239 | 0.0 | 0.9525 | 0 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.0209) |
| dpctgan_eps1_seed0 | Spearman (num-num) | 814 | 0.2916 | 0.1758 | 0.3305 | 92 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7625 -> 0.9802) |
| dpctgan_eps1_seed0 | Cramer's V (cat-cat) | 1596 | 0.1496 | 0.0498 | 0.7525 | 0 | `cause_of_death_isCV_f5a_w3a_first|cause_of_death_isAllCause_f5a_w5a_first` (1.0 -> 0.0008) |
| dpctgan_eps1_seed0 | corr-ratio (num-cat) | 7790 | 0.023 | 0.0 | 0.9583 | 0 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.0087) |
| dpctgan_eps20_seed0 | Spearman (num-num) | 894 | 0.2401 | 0.1481 | 0.368 | 74 | `lab_results_validSerumCreatinine_value_first|lab_results_creatBS_value_p3a_avg` (0.9039 -> -0.7089) |
| dpctgan_eps20_seed0 | Cramer's V (cat-cat) | 1128 | 0.1496 | 0.0443 | 0.7544 | 0 | `cause_of_death_isNonRenalAndNonCV_f5a_w1a_first|cause_of_death_isAllCause_f5a_w1a_first` (1.0 -> 0.0008) |
| dpctgan_eps20_seed0 | corr-ratio (num-cat) | 8170 | 0.0246 | 0.0 | 0.9552 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0095) |
| dpctgan_eps5_seed0 | Spearman (num-num) | 945 | 0.2451 | 0.1533 | 0.3429 | 91 | `lab_results_valideGFR_value_last|eGFR_2021_ckd_epi_creatinine` (0.9089 -> -0.2345) |
| dpctgan_eps5_seed0 | Cramer's V (cat-cat) | 1326 | 0.1438 | 0.0493 | 0.7549 | 0 | `cause_of_death_isCV_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (1.0 -> 0.0008) |
| dpctgan_eps5_seed0 | corr-ratio (num-cat) | 8360 | 0.0226 | 0.0 | 0.9587 | 0 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.0041) |
| dpctgan_eps8_seed0 | Spearman (num-num) | 980 | 0.2398 | 0.1358 | 0.4194 | 89 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7625 -> 0.9756) |
| dpctgan_eps8_seed0 | Cramer's V (cat-cat) | 1275 | 0.1444 | 0.0452 | 0.771 | 0 | `ckd_severity_from_calculated_egfr|ckd_severity_calculated_or_measured` (1.0 -> 0.0008) |
| dpctgan_eps8_seed0 | corr-ratio (num-cat) | 8550 | 0.0236 | 0.0 | 0.9571 | 0 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.0324) |
| gaussian_copula_seed0 | Spearman (num-num) | 1754 | 0.1153 | 0.0672 | 0.6391 | 13 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2471 -> 0.97) |
| gaussian_copula_seed0 | Cramer's V (cat-cat) | 3570 | 0.1259 | 0.0277 | 0.7706 | 0 | `cause_of_death_isCV_f5a_w3a_first|cause_of_death_isRenal_f5a_w3a_first` (1.0 -> 0.0005) |
| gaussian_copula_seed0 | corr-ratio (num-cat) | 11400 | 0.0185 | 0.0 | 0.9643 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.8326 -> 0.0341) |
| gaussian_copula_seed1 | Spearman (num-num) | 1751 | 0.1169 | 0.0673 | 0.6374 | 10 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2471 -> 0.9696) |
| gaussian_copula_seed1 | Cramer's V (cat-cat) | 3321 | 0.1299 | 0.0279 | 0.7715 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.0016) |
| gaussian_copula_seed1 | corr-ratio (num-cat) | 11400 | 0.019 | 0.0 | 0.9643 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.8326 -> 0.0234) |
| gaussian_copula_seed2 | Spearman (num-num) | 1747 | 0.1192 | 0.0715 | 0.6262 | 11 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.2471 -> 0.9701) |
| gaussian_copula_seed2 | Cramer's V (cat-cat) | 3403 | 0.1312 | 0.0293 | 0.7626 | 0 | `cause_of_death_isCV_f5a_w5a_first|cause_of_death_isNonRenalAndNonCV_f5a_w3a_first` (1.0 -> 0.0007) |
| gaussian_copula_seed2 | corr-ratio (num-cat) | 11400 | 0.0185 | 0.0 | 0.9635 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.8326 -> 0.0571) |
| mst_eps0p5_seed0 | Spearman (num-num) | 1983 | 0.2094 | 0.1573 | 0.3404 | 69 | `vital_signs_weight_value_p6mo_last|vital_signs_weight_value_last` (1.0 -> -0.919) |
| mst_eps0p5_seed0 | Cramer's V (cat-cat) | 3321 | 0.2538 | 0.196 | 0.3068 | 334 | `cause_of_death_isCV_f5a_w3mo_first|cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first` (1.0 -> 0.0087) |
| mst_eps0p5_seed0 | corr-ratio (num-cat) | 12160 | 0.074 | 0.0 | 0.7657 | 320 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0119) |
| mst_eps10_seed0 | Spearman (num-num) | 2060 | 0.2517 | 0.1939 | 0.3029 | 160 | `lab_results_valideGFR_value_last|maggic_total_score` (-0.5085 -> 0.6719) |
| mst_eps10_seed0 | Cramer's V (cat-cat) | 3655 | 0.1776 | 0.1351 | 0.4 | 201 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.0011) |
| mst_eps10_seed0 | corr-ratio (num-cat) | 12350 | 0.0879 | 0.0 | 0.7019 | 249 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.0633) |
| mst_eps15_seed0 | Spearman (num-num) | 1920 | 0.2726 | 0.222 | 0.2854 | 201 | `lab_results_bun_value_last|lab_results_valideGFR_value_last` (-0.6921 -> 0.5314) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 3486 | 0.161 | 0.127 | 0.4033 | 67 | `cause_of_death_isCV_f5a_w1mo_first|cause_of_death_isRenal_f5a_w1mo_first` (1.0 -> 0.0044) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 11970 | 0.0893 | 0.0 | 0.7004 | 309 | `smoking_status_smoker_totalSmokingDuration_sum|cause_of_death_isCV_f5a_w7d_first` (0.002 -> 1.0) |
| mst_eps15_seed1 | Spearman (num-num) | 1938 | 0.2877 | 0.2411 | 0.2265 | 205 | `lab_results_valideGFR_value_last|maggic_total_score` (-0.5085 -> 0.6698) |
| mst_eps15_seed1 | Cramer's V (cat-cat) | 3486 | 0.1811 | 0.1258 | 0.4389 | 287 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0011) |
| mst_eps15_seed1 | corr-ratio (num-cat) | 11970 | 0.1033 | 0.0 | 0.6689 | 404 | `lab_results_ferritin_value_last|encounter_primary_reason_HF_Disease_f5a_w7d_first` (0.0216 -> 1.0) |
| mst_eps15_seed2 | Spearman (num-num) | 1876 | 0.2744 | 0.2212 | 0.2873 | 186 | `lab_results_bun_value_last|lab_results_valideGFR_value_last` (-0.6921 -> 0.5192) |
| mst_eps15_seed2 | Cramer's V (cat-cat) | 3403 | 0.1806 | 0.1334 | 0.412 | 224 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.002) |
| mst_eps15_seed2 | corr-ratio (num-cat) | 11780 | 0.0999 | 0.0 | 0.684 | 432 | `lab_results_triGly_value_first|cause_of_death_isCV_f5a_w7d_first` (0.0135 -> 1.0) |
| mst_eps1_seed0 | Spearman (num-num) | 1929 | 0.2266 | 0.1813 | 0.2939 | 68 | `lab_results_bun_value_last|lab_results_valideGFR_value_last` (-0.6921 -> 0.3818) |
| mst_eps1_seed0 | Cramer's V (cat-cat) | 3486 | 0.2201 | 0.1617 | 0.3603 | 273 | `cause_of_death_isCV_f5a_w6mo_first|cause_of_death_isAllCause_f5a_w6mo_first` (1.0 -> 0.0076) |
| mst_eps1_seed0 | corr-ratio (num-cat) | 11970 | 0.0815 | 0.0 | 0.7377 | 362 | `lab_results_ferritin_value_first|cause_of_death_isNonRenalAndNonCV_f5a_w5a_first` (0.0058 -> 0.9423) |
| mst_eps20_seed0 | Spearman (num-num) | 1822 | 0.2883 | 0.2401 | 0.2387 | 202 | `lab_results_bun_value_last|lab_results_valideGFR_value_last` (-0.6921 -> 0.539) |
| mst_eps20_seed0 | Cramer's V (cat-cat) | 3570 | 0.1818 | 0.1535 | 0.3686 | 186 | `cause_of_death_isCV_f5a_w7d_first|conditions_osa` (0.0005 -> 1.0) |
| mst_eps20_seed0 | corr-ratio (num-cat) | 11590 | 0.1029 | 0.0 | 0.6741 | 426 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0371) |
| mst_eps5_seed0 | Spearman (num-num) | 2060 | 0.2495 | 0.1913 | 0.2947 | 154 | `lab_results_ntProBnp_value_last|encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` (-0.1684 -> 0.9616) |
| mst_eps5_seed0 | Cramer's V (cat-cat) | 3486 | 0.1557 | 0.1069 | 0.4816 | 118 | `cause_of_death_isCV_f5a_w1mo_first|cause_of_death_isRenal_f5a_w1mo_first` (1.0 -> 0.003) |
| mst_eps5_seed0 | corr-ratio (num-cat) | 12350 | 0.0859 | 0.0 | 0.7198 | 373 | `smoking_status_smoker_startTime_count|smoking_status_smoker_last` (1.0 -> 0.0004) |
| mst_eps8_seed0 | Spearman (num-num) | 1881 | 0.2629 | 0.2156 | 0.2562 | 155 | `lab_results_ntProBnp_value_first|encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` (-0.142 -> 0.8851) |
| mst_eps8_seed0 | Cramer's V (cat-cat) | 3486 | 0.2089 | 0.1697 | 0.3359 | 303 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0032) |
| mst_eps8_seed0 | corr-ratio (num-cat) | 11780 | 0.1029 | 0.0 | 0.6731 | 380 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0156) |
| patectgan_eps15_seed0 | Spearman (num-num) | 1998 | 0.14 | 0.0974 | 0.5125 | 16 | `lab_results_creatBS_value_p3a_avg|lab_results_validSerumCreatinine_value_pET` (0.8512 -> -0.2965) |
| patectgan_eps15_seed0 | Cramer's V (cat-cat) | 1653 | 0.1413 | 0.0402 | 0.7743 | 0 | `cause_of_death_isCV_f5a_w3a_first|cause_of_death_isNonRenalAndNonCV_f5a_w5a_first` (1.0 -> 0.0011) |
| patectgan_eps15_seed0 | corr-ratio (num-cat) | 12160 | 0.0233 | 0.0 | 0.9512 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0015) |
| patectgan_eps1_seed0 | Spearman (num-num) | 1842 | 0.121 | 0.0807 | 0.589 | 0 | `vital_signs_weight_value_p6mo_first|vital_signs_weight_value_last` (0.962 -> -0.0835) |
| patectgan_eps1_seed0 | Cramer's V (cat-cat) | 3655 | 0.1362 | 0.0283 | 0.7715 | 0 | `cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first|cause_of_death_isAllCause_f5a_w6mo_first` (1.0 -> 0.0) |
| patectgan_eps1_seed0 | corr-ratio (num-cat) | 11780 | 0.0211 | 0.0 | 0.961 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0037) |
| patectgan_eps5_seed0 | Spearman (num-num) | 2003 | 0.1409 | 0.0964 | 0.5112 | 28 | `echocardiographs_lvef|echocardiographs_lvef_pET_first` (0.9451 -> -0.0324) |
| patectgan_eps5_seed0 | Cramer's V (cat-cat) | 1540 | 0.1407 | 0.0375 | 0.7688 | 0 | `cause_of_death_isCV_f5a_w1a_first|cause_of_death_isAllCause_f5a_w1a_first` (1.0 -> 0.0008) |
| patectgan_eps5_seed0 | corr-ratio (num-cat) | 12160 | 0.0227 | 0.0 | 0.9566 | 0 | `smoking_status_smoker_startTime_count|smoking_status_formerSmoker_last` (1.0 -> 0.0048) |
| tvae_cap256_seed0 | Spearman (num-num) | 2075 | 0.0603 | 0.0466 | 0.8193 | 0 | `electrocardiographs_ecg_qrs_axis_pET_first|electrocardiographs_ecg_qrs_axis_pET_last` (0.7924 -> 0.2324) |
| tvae_cap256_seed0 | Cramer's V (cat-cat) | 2701 | 0.0473 | 0.0293 | 0.863 | 0 | `encounter_primary_reason_renal_complications_f5a_w1a_first|encounter_primary_reason_renal_complications_f5a_w5a_first` (0.8372 -> 0.5484) |
| tvae_cap256_seed0 | corr-ratio (num-cat) | 12350 | 0.0154 | 0.0 | 0.9766 | 0 | `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count|encounter_primary_reason_renal_complications_f5a_w5a_first` (0.5413 -> 0.1768) |
| tvae_ep1000_seed0 | Spearman (num-num) | 2075 | 0.0577 | 0.0437 | 0.8386 | 0 | `lab_results_triGly_value_last|lab_results_triGly_value_first` (0.877 -> 0.331) |
| tvae_ep1000_seed0 | Cramer's V (cat-cat) | 2926 | 0.0618 | 0.0345 | 0.8134 | 0 | `conditions_diabetes|conditions_mi` (0.1835 -> 0.5787) |
| tvae_ep1000_seed0 | corr-ratio (num-cat) | 12350 | 0.0172 | 0.0 | 0.9649 | 1 | `vital_signs_oxygenSaturation_value_last|hyperkalemia_severity_categorizedValue` (0.0818 -> 0.7757) |
| tvae_ind_seed0 | Spearman (num-num) | 2077 | 0.0649 | 0.0507 | 0.7949 | 0 | `lab_results_hba1c%_value_first|lab_results_hba1c%_value_last` (0.9916 -> 0.355) |
| tvae_ind_seed0 | Cramer's V (cat-cat) | 2485 | 0.0516 | 0.0343 | 0.8515 | 0 | `hyperkalemia_severity_categorizedValue|conditions_copd` (0.0208 -> 0.4466) |
| tvae_ind_seed0 | corr-ratio (num-cat) | 12350 | 0.0176 | 0.0 | 0.9666 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5066 -> 0.0413) |
| tvae_qt_seed0 | Spearman (num-num) | 2076 | 0.0744 | 0.0571 | 0.7408 | 0 | `lab_results_triGly_value_last|lab_results_triGly_value_first` (0.877 -> 0.3157) |
| tvae_qt_seed0 | Cramer's V (cat-cat) | 2775 | 0.0653 | 0.0303 | 0.8007 | 0 | `cause_of_death_isRenal_f5a_w3a_first|cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first` (0.6151 -> 0.2174) |
| tvae_qt_seed0 | corr-ratio (num-cat) | 12350 | 0.0168 | 0.0 | 0.9711 | 0 | `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count|encounter_primary_reason_HF_Disease_f5a_w3mo_first` (0.5232 -> 0.0062) |
| tvae_qt_seed1 | Spearman (num-num) | 2071 | 0.0729 | 0.0589 | 0.7557 | 0 | `lab_results_hba1c%_value_first|lab_results_hba1c%_value_last` (0.9916 -> 0.3102) |
| tvae_qt_seed1 | Cramer's V (cat-cat) | 2415 | 0.0667 | 0.0338 | 0.8149 | 0 | `encounter_primary_reason_CV_Disease_f5a_w1mo_first|encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (0.6498 -> 0.1896) |
| tvae_qt_seed1 | corr-ratio (num-cat) | 12350 | 0.0171 | 0.0 | 0.9663 | 0 | `conditions_heartFailure_timeFromEarliest_first|conditions_heart_failure_occurred_prior_to_18_months_any` (0.7765 -> 0.2953) |
| tvae_qt_seed2 | Spearman (num-num) | 2076 | 0.0796 | 0.0631 | 0.6985 | 2 | `lab_results_triGly_value_last|lab_results_triGly_value_first` (0.877 -> 0.2756) |
| tvae_qt_seed2 | Cramer's V (cat-cat) | 2485 | 0.0649 | 0.0331 | 0.8149 | 0 | `cause_of_death_isCV_f5a_w6mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (0.6989 -> 0.2165) |
| tvae_qt_seed2 | corr-ratio (num-cat) | 12350 | 0.0179 | 0.0 | 0.9619 | 0 | `smoking_status_smoker_startTime_count|conditions_dysl` (0.1846 -> 0.6446) |
| tvae_seed0 | Spearman (num-num) | 2072 | 0.063 | 0.0486 | 0.8065 | 0 | `lab_results_hba1c%_value_first|lab_results_hba1c%_value_last` (0.9916 -> 0.2915) |
| tvae_seed0 | Cramer's V (cat-cat) | 2850 | 0.069 | 0.0331 | 0.8102 | 0 | `cause_of_death_isAllCause_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w5a_first` (0.6151 -> 0.1692) |
| tvae_seed0 | corr-ratio (num-cat) | 12350 | 0.017 | 0.0 | 0.9692 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5066 -> 0.078) |
| tvae_seed1 | Spearman (num-num) | 2076 | 0.0594 | 0.0449 | 0.8285 | 0 | `lab_results_triGly_value_last|lab_results_triGly_value_first` (0.877 -> 0.1987) |
| tvae_seed1 | Cramer's V (cat-cat) | 3003 | 0.0484 | 0.0302 | 0.8531 | 0 | `encounter_primary_reason_renal_complications_f5a_w6mo_first|encounter_primary_reason_renal_complications_f5a_w5a_first` (0.7096 -> 0.3989) |
| tvae_seed1 | corr-ratio (num-cat) | 12350 | 0.0162 | 0.0 | 0.9702 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5066 -> 0.026) |
| tvae_seed2 | Spearman (num-num) | 2044 | 0.0645 | 0.045 | 0.7867 | 0 | `lab_results_triGly_value_last|lab_results_triGly_value_first` (0.877 -> 0.0373) |
| tvae_seed2 | Cramer's V (cat-cat) | 2775 | 0.0603 | 0.0292 | 0.8231 | 0 | `conditions_af|conditions_dysl` (0.0707 -> 0.4782) |
| tvae_seed2 | corr-ratio (num-cat) | 12350 | 0.0169 | 0.0 | 0.9667 | 0 | `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count|encounter_primary_reason_renal_complications_f5a_w5a_first` (0.5413 -> 0.051) |

## original vs preprocessed

Worst numeric columns (by KS):
- `vital_signs_height_value_last`: KS=0.0027, W/std=0.026, mean 167.9084 -> 168.1919, missing 41% -> 41%
- `vital_signs_weight_value_p6mo_first`: KS=0.0019, W/std=0.0083, mean 74.5281 -> 74.661, missing 36% -> 36%
- `vital_signs_height_value_p1a_avg`: KS=0.001, W/std=0.0111, mean 167.8143 -> 167.9272, missing 23% -> 23%
- `electrocardiographs_ecg_qrs_duration_pET_first`: KS=0.0007, W/std=0.0023, mean 116.0414 -> 116.1153, missing 41% -> 41%
- `electrocardiographs_ecg_qrs_duration_pET_last`: KS=0.0007, W/std=0.0026, mean 116.5002 -> 116.5821, missing 41% -> 42%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 9 -> 9 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `lab_results_ferritin_value_first`: KS=0.066, W/std=0.0877, mean 222.1348 -> 242.604, missing 71% -> 72%
- `lab_results_ferritin_value_last`: KS=0.0653, W/std=0.0987, mean 228.3411 -> 254.0456, missing 71% -> 72%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.0653, W/std=0.049, mean 240.1188 -> 241.9773, missing 83% -> 86%
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=0.0592, W/std=0.0414, mean 265.3056 -> 272.3516, missing 82% -> 83%
- `lab_results_valideGFR_value_last`: KS=0.055, W/std=0.085, mean 64.5392 -> 66.6719, missing 18% -> 16%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0544, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.0343, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.0337, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.0319, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.0319, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps1_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=5.7773, mean 0.4516 -> 3.3267, missing 0% -> 0%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.9959, W/std=6.5976, mean 125.1725 -> 2716.6763, missing 55% -> 54%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9857, W/std=4.598, mean 12.7127 -> 56.4918, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9642, W/std=2.3991, mean 9.4357 -> 58.903, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.8471, W/std=1.2879, mean 32.9605 -> 88.1516, missing 46% -> 46%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0587, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0411, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0327, 5 -> 5 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.0205, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.017, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[aim40_eps5_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=4.2412, mean 0.4516 -> 2.5623, missing 0% -> 0%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.9959, W/std=6.6721, mean 125.1725 -> 2745.9242, missing 55% -> 55%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9857, W/std=4.3187, mean 12.7127 -> 53.6831, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9642, W/std=2.1584, mean 9.4357 -> 53.1532, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.8471, W/std=1.0441, mean 32.9605 -> 75.5226, missing 46% -> 47%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0241, 5 -> 5 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0227, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0195, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0166, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0116, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps1_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=5.2582, mean 0.4516 -> 3.0684, missing 0% -> 0%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.9959, W/std=8.5909, mean 125.1725 -> 3499.6182, missing 55% -> 54%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9857, W/std=4.4783, mean 12.7127 -> 55.3524, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.9642, W/std=2.2338, mean 9.4357 -> 54.478, missing 0% -> 0%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9057, W/std=0.95, mean 96.0798 -> 93.3251, missing 30% -> 32%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0475, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0349, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0202, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0201, 5 -> 5 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.0195, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_qt_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.5546, W/std=1.9785, mean 240.1188 -> 888.4469, missing 83% -> 89%
- `vital_signs_weight_value_last`: KS=0.4876, W/std=1.7769, mean 74.4175 -> 101.9596, missing 52% -> 44%
- `lab_results_albuminBS_value_last`: KS=0.4577, W/std=1.2153, mean 32.9604 -> 40.3144, missing 20% -> 29%
- `patient_demographics_age`: KS=0.4545, W/std=1.0012, mean 73.0641 -> 85.8096, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.4409, W/std=1.629, mean 222.1348 -> 684.7061, missing 71% -> 77%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2588, 9 -> 8 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.2262, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.206, 5 -> 5 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.1865, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.1754, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.6396, W/std=0.5555, mean 125.1725 -> 340.4445, missing 55% -> 81%
- `vital_signs_weight_value_p6mo_first`: KS=0.6065, W/std=1.7682, mean 74.8432 -> 102.4942, missing 36% -> 52%
- `electrocardiographs_ecg_qrs_duration_pET_last`: KS=0.6052, W/std=1.1434, mean 116.3844 -> 80.6246, missing 42% -> 38%
- `vital_signs_height_value_last`: KS=0.5792, W/std=1.6295, mean 168.3591 -> 153.0209, missing 41% -> 58%
- `vital_signs_systolicBp_value_first`: KS=0.5633, W/std=1.6194, mean 122.6211 -> 152.6483, missing 33% -> 59%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2756, 9 -> 8 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.2379, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.2129, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.2026, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.1835, 7 -> 7 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed1]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.7217, W/std=0.6857, mean 95.9324 -> 98.803, missing 30% -> 40%
- `lab_results_albuminBS_value_first`: KS=0.5849, W/std=1.7569, mean 34.4163 -> 23.8041, missing 20% -> 24%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5739, W/std=0.1861, mean 125.1725 -> 170.0398, missing 55% -> 87%
- `vital_signs_heartRate_value_last`: KS=0.4591, W/std=1.0319, mean 74.2156 -> 61.5294, missing 27% -> 20%
- `lab_results_triGly_value_last`: KS=0.4574, W/std=0.7926, mean 1.294 -> 0.806, missing 33% -> 60%
Worst categorical columns (by TVD):
- `conditions_ihd`: TVD=0.214, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.2004, 5 -> 5 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.1973, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.1787, 2 -> 2 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.1544, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed2]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5904, W/std=0.2322, mean 125.1725 -> 208.1581, missing 55% -> 86%
- `vital_signs_systolicBp_value_last`: KS=0.5746, W/std=1.4193, mean 118.1401 -> 95.1503, missing 33% -> 25%
- `echocardiographs_lvef`: KS=0.532, W/std=1.3434, mean 47.9039 -> 67.7378, missing 24% -> 16%
- `lab_results_triGly_value_first`: KS=0.5088, W/std=1.2006, mean 1.2734 -> 2.0086, missing 33% -> 34%
- `vital_signs_heartRate_value_first`: KS=0.507, W/std=1.1662, mean 75.9759 -> 57.6699, missing 27% -> 33%
Worst categorical columns (by TVD):
- `smoking_status_smoker_last`: TVD=0.2455, 2 -> 2 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.2344, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.1984, 2 -> 2 categories, missing 0% -> 0%
- `conditions_af`: TVD=0.1734, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.1693, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[ddpm_g_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=1.0, W/std=None, mean 240.1188 -> None, missing 83% -> 100%
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=1.0, W/std=None, mean 265.3056 -> None, missing 82% -> 100%
- `lab_results_hba1c%_value_first`: KS=0.7832, W/std=4.0119, mean 6.6366 -> 12.1808, missing 70% -> 88%
- `lab_results_hba1c%_value_last`: KS=0.7754, W/std=3.9887, mean 6.622 -> 12.0362, missing 70% -> 87%
- `lab_results_ferritin_value_last`: KS=0.7629, W/std=4.6511, mean 228.3411 -> 1565.691, missing 71% -> 86%
Worst categorical columns (by TVD):
- `smoking_status_smoker_last`: TVD=0.5484, 2 -> 1 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.5274, 2 -> 1 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.4516, 2 -> 1 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.4381, 2 -> 1 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.3419, 3 -> 1 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed0]

Worst numeric columns (by KS):
- `lab_results_hba1c%_value_last`: KS=0.7673, W/std=3.9566, mean 6.622 -> 11.9946, missing 70% -> 85%
- `lab_results_hba1c%_value_first`: KS=0.7612, W/std=3.9306, mean 6.6366 -> 12.0684, missing 70% -> 85%
- `lab_results_ferritin_value_last`: KS=0.753, W/std=4.6897, mean 228.3411 -> 1576.8057, missing 71% -> 86%
- `lab_results_ferritin_value_first`: KS=0.7498, W/std=4.7332, mean 222.1348 -> 1566.2179, missing 71% -> 86%
- `electrocardiographs_ecg_qrs_duration_pET_first`: KS=0.7287, W/std=3.823, mean 116.1958 -> 238.088, missing 42% -> 53%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.1492, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1101, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.1016, 5 -> 5 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0852, 9 -> 6 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.0718, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed1]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_first`: KS=0.7881, W/std=4.9363, mean 222.1348 -> 1623.8876, missing 71% -> 84%
- `lab_results_ferritin_value_last`: KS=0.7855, W/std=4.8688, mean 228.3411 -> 1628.2921, missing 71% -> 84%
- `electrocardiographs_ecg_qt_duration_corrected_pET_last`: KS=0.7353, W/std=3.5785, mean 430.9394 -> 578.6821, missing 42% -> 52%
- `electrocardiographs_ecg_qrs_axis_pET_first`: KS=0.7312, W/std=3.0673, mean 12.4372 -> 209.1964, missing 42% -> 52%
- `electrocardiographs_ecg_qrs_duration_pET_last`: KS=0.7308, W/std=3.3106, mean 116.3844 -> 220.0252, missing 42% -> 52%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.1657, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.1313, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.1313, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.1299, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.1283, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed2]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=0.7913, W/std=4.8186, mean 228.3411 -> 1613.8487, missing 71% -> 83%
- `lab_results_ferritin_value_first`: KS=0.7891, W/std=4.894, mean 222.1348 -> 1611.8782, missing 71% -> 83%
- `lab_results_hba1c%_value_first`: KS=0.7852, W/std=4.0945, mean 6.6366 -> 12.295, missing 70% -> 84%
- `lab_results_hba1c%_value_last`: KS=0.7822, W/std=4.0911, mean 6.622 -> 12.1772, missing 70% -> 84%
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=0.7453, W/std=2.9909, mean 265.3056 -> 1266.5067, missing 82% -> 93%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.135, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.1058, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.105, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.1042, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.1018, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps10_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 74.8432 -> None, missing 36% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=7.9699, mean 168.0439 -> 244.6538, missing 23% -> 0%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 74.4175 -> None, missing 52% -> 100%
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 168.3591 -> None, missing 41% -> 100%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=12.8167, mean 75.9759 -> 278.6307, missing 27% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.9271, 3 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.895, 3 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.863, 3 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`: TVD=0.8415, 3 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.829, 7 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed0]

Worst numeric columns (by KS):
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=None, mean 168.0439 -> None, missing 23% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=17.6132, mean 74.4175 -> 347.451, missing 52% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 68.3117 -> None, missing 27% -> 100%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 75.9759 -> None, missing 27% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 74.2156 -> None, missing 27% -> 100%
Worst categorical columns (by TVD):
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.8998, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first`: TVD=0.8991, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.8439, 5 -> 3 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.7996, 9 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.7114, 7 -> 7 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed1]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 74.8432 -> None, missing 36% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=None, mean 168.0439 -> None, missing 23% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 70.8836 -> None, missing 27% -> 100%
- `vital_signs_systolicBp_value_first`: KS=1.0, W/std=9.2972, mean 122.6211 -> 295.1361, missing 33% -> 0%
- `vital_signs_systolicBp_value_last`: KS=1.0, W/std=None, mean 118.1401 -> None, missing 33% -> 100%
Worst categorical columns (by TVD):
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`: TVD=0.9608, 3 -> 3 categories, missing 0% -> 0%
- `conditions_ld`: TVD=0.9109, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8198, 7 -> 6 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.7829, 9 -> 1 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.7525, 5 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed2]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 74.6255 -> None, missing 36% -> 100%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 74.8432 -> None, missing 36% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 74.4175 -> None, missing 52% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=17.0612, mean 70.8836 -> 248.3053, missing 27% -> 0%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 74.2156 -> None, missing 27% -> 100%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.9865, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.942, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w6mo_first`: TVD=0.9369, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first`: TVD=0.9173, 3 -> 3 categories, missing 0% -> 0%
- `conditions_ap`: TVD=0.917, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps1_seed0]

Worst numeric columns (by KS):
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=None, mean 168.0439 -> None, missing 23% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=15.8135, mean 74.4175 -> 319.5533, missing 52% -> 0%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=16.9274, mean 70.8836 -> 246.914, missing 27% -> 0%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 74.2156 -> None, missing 27% -> 100%
- `vital_signs_oxygenSaturation_value_last`: KS=1.0, W/std=None, mean 95.9324 -> None, missing 30% -> 100%
Worst categorical columns (by TVD):
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.97, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first`: TVD=0.9163, 3 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.8561, 5 -> 3 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8503, 7 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.8339, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps20_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 74.6255 -> None, missing 36% -> 100%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 74.8432 -> None, missing 36% -> 100%
- `vital_signs_weight_value_last`: KS=1.0, W/std=None, mean 74.4175 -> None, missing 52% -> 100%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 75.9759 -> None, missing 27% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=16.2786, mean 74.2156 -> 281.094, missing 27% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_renal_complications_f5a_w1mo_first`: TVD=0.9107, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.89, 3 -> 3 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.8778, 2 -> 1 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.8574, 5 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.8399, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps5_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 74.6255 -> None, missing 36% -> 100%
- `vital_signs_height_value_last`: KS=1.0, W/std=None, mean 168.3591 -> None, missing 41% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 70.8836 -> None, missing 27% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 68.3117 -> None, missing 27% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=13.4807, mean 74.2156 -> 245.5374, missing 27% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.8508, 5 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.8145, 3 -> 3 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.7795, 9 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.7729, 3 -> 3 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.7636, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps8_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=1.0, W/std=None, mean 74.6255 -> None, missing 36% -> 100%
- `vital_signs_weight_value_p6mo_first`: KS=1.0, W/std=None, mean 74.8432 -> None, missing 36% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=1.0, W/std=None, mean 168.0439 -> None, missing 23% -> 100%
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 70.8836 -> None, missing 27% -> 100%
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 68.3117 -> None, missing 27% -> 100%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.9942, 5 -> 1 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w1a_first`: TVD=0.9674, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first`: TVD=0.8696, 3 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.8683, 3 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8275, 9 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9733, W/std=0.9638, mean 95.9324 -> 99.4164, missing 30% -> 29%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9617, W/std=1.2091, mean 96.0798 -> 99.4883, missing 30% -> 29%
- `vital_signs_height_value_p1a_avg`: KS=0.8744, W/std=4.7082, mean 168.0439 -> 209.5949, missing 23% -> 20%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.8306, W/std=1.2323, mean 125.1725 -> 538.5222, missing 55% -> 100%
- `vital_signs_height_value_last`: KS=0.8006, W/std=2.67, mean 168.3591 -> 188.3327, missing 41% -> 35%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2039, 9 -> 9 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.0378, 2 -> 2 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.0267, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0261, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0247, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed1]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.9679, W/std=1.0003, mean 95.9324 -> 99.1684, missing 30% -> 28%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9571, W/std=1.2224, mean 96.0798 -> 99.3925, missing 30% -> 28%
- `vital_signs_height_value_p1a_avg`: KS=0.87, W/std=4.7079, mean 168.0439 -> 209.2773, missing 23% -> 21%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.8322, W/std=0.56, mean 125.1725 -> 167.6667, missing 55% -> 100%
- `vital_signs_systolicBp_value_last`: KS=0.805, W/std=5.3703, mean 118.1401 -> 199.5242, missing 33% -> 26%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2126, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0321, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0257, 5 -> 5 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.0218, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.021, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed2]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=1.0, W/std=None, mean 265.3056 -> None, missing 82% -> 100%
- `vital_signs_oxygenSaturation_value_last`: KS=0.972, W/std=0.9492, mean 95.9324 -> 99.467, missing 30% -> 30%
- `vital_signs_oxygenSaturation_value_first`: KS=0.9584, W/std=1.2506, mean 96.0798 -> 99.2826, missing 30% -> 30%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.8781, W/std=1.1826, mean 125.1725 -> 522.4333, missing 55% -> 100%
- `vital_signs_height_value_p1a_avg`: KS=0.8652, W/std=4.6751, mean 168.0439 -> 208.8747, missing 23% -> 21%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1935, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0249, 5 -> 5 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.0232, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0223, 7 -> 7 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.0222, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps0p5_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=21.8863, mean 228.3411 -> 6521.4411, missing 71% -> 70%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=49.5293, mean 222.1348 -> 14286.9578, missing 71% -> 71%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=7.9017, mean 0.4516 -> 4.384, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=30.415, mean 0.0204 -> 4.4886, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=17.6698, mean 0.0501 -> 4.0726, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2724, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.1469, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1446, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.1292, 5 -> 5 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.0773, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps10_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=22.5273, mean 228.3411 -> 6705.7304, missing 71% -> 72%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=21.7651, mean 222.1348 -> 6402.7816, missing 71% -> 72%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=4.3743, mean 0.4516 -> 2.6285, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=17.2754, mean 0.0204 -> 2.5583, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=11.3205, mean 0.0501 -> 2.6272, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0239, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0218, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0207, 5 -> 5 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0185, 9 -> 8 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.0129, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=20.9423, mean 228.3411 -> 6250.0, missing 71% -> 71%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=21.2271, mean 222.1348 -> 6250.0, missing 71% -> 71%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=4.3796, mean 0.4516 -> 2.6311, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=17.4467, mean 0.0204 -> 2.5835, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=11.1983, mean 0.0501 -> 2.5994, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0218, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0203, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0189, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0184, 5 -> 5 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.0118, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed1]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=21.1865, mean 228.3411 -> 6320.2247, missing 71% -> 72%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=21.2682, mean 222.1348 -> 6261.6713, missing 71% -> 72%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=4.2651, mean 0.4516 -> 2.5742, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=17.2844, mean 0.0204 -> 2.5596, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=11.0528, mean 0.0501 -> 2.5662, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0186, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0183, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0178, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0178, 5 -> 5 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.0137, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed2]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=21.3389, mean 228.3411 -> 6364.0373, missing 71% -> 72%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=21.2271, mean 222.1348 -> 6250.0, missing 71% -> 72%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=4.1427, mean 0.4516 -> 2.5132, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=17.9157, mean 0.0204 -> 2.6524, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=10.9829, mean 0.0501 -> 2.5504, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0246, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0231, 9 -> 9 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0206, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0206, 5 -> 5 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.0118, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps1_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=30.981, mean 228.3411 -> 9136.4859, missing 71% -> 71%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=31.321, mean 222.1348 -> 9116.3608, missing 71% -> 71%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=5.2635, mean 0.4516 -> 3.071, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=20.3416, mean 0.0204 -> 3.0088, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=11.9548, mean 0.0501 -> 2.7716, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0463, 9 -> 9 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.0428, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0374, 5 -> 5 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.0369, 3 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w6mo_first`: TVD=0.0363, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps20_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=20.9423, mean 228.3411 -> 6250.0, missing 71% -> 72%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=21.2271, mean 222.1348 -> 6250.0, missing 71% -> 72%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=4.1906, mean 0.4516 -> 2.5371, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=16.8786, mean 0.0204 -> 2.5, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=11.0644, mean 0.0501 -> 2.5689, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.0221, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0209, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0208, 5 -> 5 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0178, 9 -> 8 categories, missing 0% -> 0%
- `conditions_af`: TVD=0.0119, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps5_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=23.697, mean 228.3411 -> 7042.0777, missing 71% -> 72%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=21.8041, mean 222.1348 -> 6413.8516, missing 71% -> 72%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=4.3716, mean 0.4516 -> 2.6272, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=17.1852, mean 0.0204 -> 2.545, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=12.3971, mean 0.0501 -> 2.8723, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0247, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0236, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0231, 5 -> 5 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0219, 5 -> 5 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0177, 9 -> 9 categories, missing 0% -> 0%

## train vs synthetic[mst_eps8_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=1.0, W/std=20.9423, mean 228.3411 -> 6250.0, missing 71% -> 72%
- `lab_results_ferritin_value_first`: KS=1.0, W/std=21.4816, mean 222.1348 -> 6322.2543, missing 71% -> 72%
- `smoking_status_smoker_startTime_count`: KS=1.0, W/std=4.1693, mean 0.4516 -> 2.5265, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=18.6101, mean 0.0204 -> 2.7544, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=11.4136, mean 0.0501 -> 2.6484, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0225, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0206, 5 -> 5 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0197, 9 -> 8 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0182, 5 -> 5 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.0148, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps15_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=0.925, W/std=55.5863, mean 228.3411 -> 16211.3521, missing 71% -> 55%
- `lab_results_ferritin_value_first`: KS=0.8707, W/std=27.4108, mean 222.1348 -> 8005.9652, missing 71% -> 74%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.8169, W/std=0.1604, mean 0.0501 -> 0.047, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8166, W/std=0.139, mean 0.0204 -> 0.0, missing 0% -> 0%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.8034, W/std=7.4506, mean 125.1725 -> 3051.7049, missing 55% -> 87%
Worst categorical columns (by TVD):
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`: TVD=0.8192, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.8174, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.8116, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.7994, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.6753, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps1_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9804, W/std=22.0643, mean 0.0204 -> 3.2618, missing 0% -> 0%
- `vital_signs_oxygenSaturation_value_last`: KS=0.9655, W/std=7.8922, mean 95.9324 -> 62.7809, missing 30% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.952, W/std=13.925, mean 0.0501 -> 3.2201, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.9316, W/std=54.9002, mean 222.1348 -> 15812.1368, missing 71% -> 37%
- `lab_results_ferritin_value_last`: KS=0.9159, W/std=45.0069, mean 228.3411 -> 13169.4169, missing 71% -> 46%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.3351, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`: TVD=0.3245, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.3031, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.2961, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.2778, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps5_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_last`: KS=0.9021, W/std=35.4872, mean 228.3411 -> 10432.1788, missing 71% -> 65%
- `lab_results_ferritin_value_first`: KS=0.8849, W/std=24.3081, mean 222.1348 -> 7124.9219, missing 71% -> 86%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8808, W/std=0.139, mean 0.0204 -> 0.0, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.8495, W/std=0.2201, mean 0.0501 -> 0.0, missing 0% -> 0%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.8348, W/std=10.4477, mean 125.1725 -> 4228.9552, missing 55% -> 59%
Worst categorical columns (by TVD):
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.7893, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3a_first`: TVD=0.7795, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w5a_first`: TVD=0.7779, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`: TVD=0.7721, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.6502, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_cap256_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5119, W/std=0.0996, mean 125.1725 -> 144.7102, missing 55% -> 62%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.2867, W/std=0.2913, mean 240.1188 -> 316.8505, missing 83% -> 82%
- `lab_results_triGly_value_last`: KS=0.2643, W/std=0.3828, mean 1.294 -> 1.0599, missing 33% -> 41%
- `lab_results_triGly_value_first`: KS=0.2404, W/std=0.3404, mean 1.2734 -> 1.0683, missing 33% -> 40%
- `nyha_nyha_pET`: KS=0.2359, W/std=0.5381, mean 2.7265 -> 2.8924, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1649, 9 -> 8 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.141, 2 -> 2 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.122, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.0929, 3 -> 3 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.0925, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ep1000_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5243, W/std=0.2411, mean 125.1725 -> 198.7006, missing 55% -> 62%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.3477, W/std=0.3414, mean 240.1188 -> 327.2121, missing 83% -> 83%
- `vital_signs_systolicBp_value_last`: KS=0.273, W/std=0.302, mean 118.1401 -> 113.5964, missing 33% -> 34%
- `maggic_total_score`: KS=0.265, W/std=0.4127, mean 26.7355 -> 29.3361, missing 28% -> 31%
- `vital_signs_diastolicBp_value_first`: KS=0.2587, W/std=0.263, mean 70.8836 -> 69.0074, missing 27% -> 31%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2132, 9 -> 7 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1635, 2 -> 2 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.1272, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.1212, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.1212, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ind_seed0]

Worst numeric columns (by KS):
- `vital_signs_heartRate_value_first`: KS=0.3777, W/std=0.6246, mean 75.9759 -> 66.2868, missing 27% -> 30%
- `lab_results_potassium_value_last`: KS=0.3075, W/std=0.5327, mean 3.9473 -> 3.7014, missing 3% -> 2%
- `lab_results_crpNonHs_value_last`: KS=0.304, W/std=0.3831, mean 32.9605 -> 15.5425, missing 46% -> 45%
- `vital_signs_systolicBpDuringEncounter_value_pET`: KS=0.2914, W/std=0.3617, mean 118.764 -> 115.5673, missing 0% -> 0%
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.2887, W/std=0.0934, mean 125.1725 -> 109.923, missing 55% -> 50%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1889, 9 -> 7 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1776, 2 -> 2 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.131, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.1109, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1055, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_first`: KS=0.3451, W/std=0.3555, mean 96.0798 -> 95.1311, missing 30% -> 40%
- `vital_signs_heartRate_value_last`: KS=0.3344, W/std=0.5185, mean 74.2156 -> 67.8154, missing 27% -> 36%
- `lab_results_triGly_value_last`: KS=0.2989, W/std=0.4112, mean 1.294 -> 1.0493, missing 33% -> 39%
- `nyha_nyha_pET`: KS=0.2979, W/std=0.6418, mean 2.7265 -> 2.9383, missing 0% -> 0%
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=0.2854, W/std=0.3543, mean 265.3056 -> 203.4061, missing 82% -> 80%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1829, 9 -> 8 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1527, 2 -> 2 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.1445, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1031, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.0946, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed1]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_first`: KS=0.3575, W/std=0.3554, mean 96.0798 -> 95.0596, missing 30% -> 42%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.3072, W/std=0.2961, mean 240.1188 -> 217.1794, missing 83% -> 82%
- `lab_results_crpNonHs_value_last`: KS=0.2983, W/std=0.4025, mean 32.9605 -> 15.002, missing 46% -> 52%
- `nyha_nyha_pET`: KS=0.2778, W/std=0.6141, mean 2.7265 -> 2.9186, missing 0% -> 0%
- `lab_results_sodium_value_last`: KS=0.2698, W/std=0.4615, mean 139.7388 -> 138.2097, missing 3% -> 4%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2062, 9 -> 7 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.154, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1445, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.1369, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.1343, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed2]

Worst numeric columns (by KS):
- `lab_results_triGly_value_last`: KS=0.3541, W/std=0.4818, mean 1.294 -> 1.006, missing 33% -> 46%
- `vital_signs_oxygenSaturation_value_first`: KS=0.353, W/std=0.3532, mean 96.0798 -> 95.1121, missing 30% -> 43%
- `lab_results_crpNonHs_value_last`: KS=0.3167, W/std=0.4432, mean 32.9605 -> 13.1118, missing 46% -> 57%
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=0.3161, W/std=0.3172, mean 265.3056 -> 265.8852, missing 82% -> 74%
- `nyha_nyha_pET`: KS=0.3021, W/std=0.6456, mean 2.7265 -> 2.9438, missing 0% -> 0%
Worst categorical columns (by TVD):
- `conditions_hyp`: TVD=0.1731, 2 -> 2 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.1614, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1417, 9 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.1178, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.1104, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5277, W/std=0.1778, mean 125.1725 -> 70.5669, missing 55% -> 60%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.35, W/std=0.3006, mean 240.1188 -> 316.4019, missing 83% -> 82%
- `vital_signs_systolicBpDuringEncounter_value_pET`: KS=0.2987, W/std=0.3439, mean 118.764 -> 115.0672, missing 0% -> 0%
- `nyha_nyha_pET`: KS=0.2781, W/std=0.613, mean 2.7265 -> 2.9189, missing 0% -> 0%
- `lab_results_triGly_value_last`: KS=0.2746, W/std=0.4092, mean 1.294 -> 1.0492, missing 33% -> 40%
Worst categorical columns (by TVD):
- `conditions_stroke`: TVD=0.1601, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1588, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.125, 9 -> 8 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.0997, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.0959, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed1]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.5338, W/std=0.1484, mean 125.1725 -> 80.0149, missing 55% -> 66%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.3204, W/std=0.4449, mean 240.1188 -> 372.0037, missing 83% -> 82%
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=0.315, W/std=0.572, mean 265.3056 -> 444.0147, missing 82% -> 74%
- `nyha_nyha_pET`: KS=0.2844, W/std=0.621, mean 2.7265 -> 2.9266, missing 0% -> 0%
- `lab_results_triGly_value_last`: KS=0.2824, W/std=0.4211, mean 1.294 -> 1.0421, missing 33% -> 39%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.171, 9 -> 9 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.1386, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.136, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.1165, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.116, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed2]

Worst numeric columns (by KS):
- `smoking_status_smoker_totalSmokingDuration_sum`: KS=0.4689, W/std=0.1481, mean 125.1725 -> 77.8544, missing 55% -> 69%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.4118, W/std=0.542, mean 240.1188 -> 407.1964, missing 83% -> 83%
- `lab_results_hba1c%_value_first`: KS=0.3328, W/std=0.466, mean 6.6366 -> 6.0236, missing 70% -> 76%
- `lab_results_triGly_value_last`: KS=0.3227, W/std=0.4961, mean 1.294 -> 0.9885, missing 33% -> 34%
- `nyha_nyha_pET`: KS=0.3069, W/std=0.6502, mean 2.7265 -> 2.9494, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2033, 9 -> 7 categories, missing 0% -> 0%
- `conditions_stroke`: TVD=0.1725, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1532, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1079, 2 -> 2 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.0965, 2 -> 2 categories, missing 0% -> 0%
