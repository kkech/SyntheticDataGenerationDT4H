# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 3520 rows | holdout test: 1174 rows

## `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`
train 853 labelled (531 positive), holdout 264 labelled (168 positive) | baseline AUC **0.6861** (HistGB) / 0.6558 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.5146 | +0.1715 | +0.0608 | -0.0070 |
| aim50_eps5_seed0 | 0.4906 | +0.1955 | +0.2913 | -0.0049 |
| ctgan_seed0 | 0.3633 | +0.3228 | +0.2380 | -0.0192 |
| ctgan_seed1 | 0.4407 | +0.2454 | +0.1631 | -0.0328 |
| ctgan_seed2 | 0.6207 | +0.0654 | +0.1390 | +0.0051 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | 0.5114 | +0.1747 | +0.2235 | +0.0061 |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.5195 | +0.1666 | +0.0955 | +0.0039 |
| gaussian_copula_seed1 | 0.5101 | +0.1760 | +0.1790 | -0.0100 |
| gaussian_copula_seed2 | 0.5385 | +0.1476 | +0.1771 | +0.0093 |
| mst_eps10_seed0 | 0.5595 | +0.1266 | +0.0771 | +0.0054 |
| mst_eps15_seed0 | 0.5097 | +0.1764 | +0.0954 | -0.0112 |
| mst_eps15_seed1 | 0.522 | +0.1641 | +0.1818 | -0.0112 |
| mst_eps15_seed2 | 0.5321 | +0.1540 | +0.1210 | +0.0165 |
| mst_eps1_seed0 | 0.5767 | +0.1094 | +0.1216 | -0.0059 |
| mst_eps20_seed0 | 0.5172 | +0.1689 | +0.1170 | -0.0025 |
| mst_eps5_seed0 | 0.4684 | +0.2177 | +0.2185 | +0.0070 |
| mst_eps8_seed0 | 0.4669 | +0.2192 | +0.0754 | -0.0086 |
| tvae_seed0 | 0.5811 | +0.1050 | +0.0880 | +0.0117 |
| tvae_seed1 | 0.5763 | +0.1098 | +0.0744 | -0.0107 |
| tvae_seed2 | 0.5937 | +0.0924 | +0.1250 | -0.0152 |

## `encounter_primary_reason_CV_Disease_f5a_w1mo_first`
train 853 labelled (322 positive), holdout 264 labelled (96 positive) | baseline AUC **0.6861** (HistGB) / 0.6558 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.5347 | +0.1514 | +0.0360 | -0.0068 |
| aim50_eps5_seed0 | 0.4905 | +0.1956 | +0.3079 | -0.0077 |
| ctgan_seed0 | 0.427 | +0.2591 | +0.1291 | -0.0444 |
| ctgan_seed1 | 0.5111 | +0.1750 | +0.1763 | -0.0190 |
| ctgan_seed2 | 0.5861 | +0.1000 | +0.1329 | +0.0090 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.5154 | +0.1707 | +0.1497 | -0.0069 |
| gaussian_copula_seed1 | 0.5429 | +0.1432 | +0.1813 | -0.0044 |
| gaussian_copula_seed2 | 0.4956 | +0.1905 | +0.0534 | -0.0164 |
| mst_eps10_seed0 | 0.508 | +0.1781 | +0.0793 | -0.0041 |
| mst_eps15_seed0 | 0.523 | +0.1631 | +0.0731 | -0.0124 |
| mst_eps15_seed1 | 0.537 | +0.1491 | +0.1622 | +0.0092 |
| mst_eps15_seed2 | 0.4581 | +0.2280 | +0.1152 | +0.0243 |
| mst_eps1_seed0 | 0.5146 | +0.1715 | +0.0875 | -0.0159 |
| mst_eps20_seed0 | 0.5085 | +0.1776 | +0.1357 | -0.0093 |
| mst_eps5_seed0 | 0.5391 | +0.1470 | +0.1565 | -0.0040 |
| mst_eps8_seed0 | 0.4537 | +0.2324 | +0.0748 | +0.0056 |
| tvae_seed0 | 0.5823 | +0.1038 | +0.0900 | +0.0158 |
| tvae_seed1 | 0.6001 | +0.0860 | +0.0660 | -0.0166 |
| tvae_seed2 | 0.557 | +0.1291 | +0.1228 | -0.0123 |

## `encounter_primary_reason_HF_Disease_f5a_w1mo_first`
train 853 labelled (98 positive), holdout 264 labelled (18 positive) | baseline AUC **0.5729** (HistGB) / 0.5438 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.5411 | +0.0318 | -0.0219 | +0.0086 |
| aim50_eps5_seed0 | 0.5312 | +0.0417 | -0.0739 | +0.0262 |
| ctgan_seed0 | 0.5479 | +0.0250 | +0.0616 | -0.0374 |
| ctgan_seed1 | 0.5176 | +0.0553 | +0.0246 | -0.0603 |
| ctgan_seed2 | 0.568 | +0.0049 | -0.0212 | -0.0158 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | 0.6383 | -0.0654 | +0.0808 | -0.0266 |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.371 | +0.2019 | +0.0763 | -0.0790 |
| gaussian_copula_seed1 | 0.516 | +0.0569 | +0.0714 | +0.0204 |
| gaussian_copula_seed2 | 0.3835 | +0.1894 | +0.1288 | -0.0564 |
| mst_eps10_seed0 | 0.5589 | +0.0140 | +0.0707 | +0.0165 |
| mst_eps15_seed0 | 0.5567 | +0.0162 | +0.0336 | -0.0257 |
| mst_eps15_seed1 | 0.4713 | +0.1016 | +0.0266 | +0.0154 |
| mst_eps15_seed2 | 0.6716 | -0.0987 | -0.0614 | -0.0241 |
| mst_eps1_seed0 | 0.5637 | +0.0092 | +0.1145 | -0.0275 |
| mst_eps20_seed0 | 0.6493 | -0.0764 | +0.0348 | +0.0400 |
| mst_eps5_seed0 | 0.6064 | -0.0335 | -0.0118 | -0.0399 |
| mst_eps8_seed0 | 0.5906 | -0.0177 | +0.0366 | +0.0265 |
| tvae_seed0 | 0.4756 | +0.0973 | -0.0007 | -0.0729 |
| tvae_seed1 | 0.3568 | +0.2161 | +0.0761 | -0.0110 |
| tvae_seed2 | 0.4636 | +0.1093 | +0.0300 | -0.0266 |

## `cause_of_death_isAllCause_f5a_w3a_first`
train 1291 labelled (1208 positive), holdout 438 labelled (417 positive) | baseline AUC **0.5435** (HistGB) / 0.5144 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.5025 | +0.0410 | +0.0987 | -0.0836 |
| aim50_eps5_seed0 | 0.5108 | +0.0327 | -0.0076 | +0.0389 |
| ctgan_seed0 | 0.5657 | -0.0222 | -0.0183 | -0.0072 |
| ctgan_seed1 | 0.5244 | +0.0191 | +0.0240 | -0.0359 |
| ctgan_seed2 | 0.444 | +0.0995 | +0.0942 | -0.0706 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.6282 | -0.0847 | -0.0561 | -0.0590 |
| gaussian_copula_seed1 | 0.4716 | +0.0719 | +0.0124 | -0.0193 |
| gaussian_copula_seed2 | 0.5487 | -0.0052 | +0.0742 | -0.0319 |
| mst_eps10_seed0 | 0.4969 | +0.0466 | -0.0175 | -0.0228 |
| mst_eps15_seed0 | 0.4704 | +0.0731 | +0.0240 | -0.0160 |
| mst_eps15_seed1 | 0.4172 | +0.1263 | +0.0221 | +0.0256 |
| mst_eps15_seed2 | 0.3669 | +0.1766 | -0.0399 | -0.0085 |
| mst_eps1_seed0 | 0.4836 | +0.0599 | +0.0734 | +0.0763 |
| mst_eps20_seed0 | 0.5035 | +0.0400 | -0.0131 | -0.0245 |
| mst_eps5_seed0 | 0.5067 | +0.0368 | -0.0645 | +0.0227 |
| mst_eps8_seed0 | 0.4259 | +0.1176 | +0.2297 | -0.0188 |
| tvae_seed0 | 0.5567 | -0.0132 | -0.1172 | -0.0095 |
| tvae_seed1 | 0.5372 | +0.0063 | -0.0940 | -0.0447 |
| tvae_seed2 | 0.6072 | -0.0637 | +0.0061 | +0.0121 |

## `cause_of_death_isCV_f5a_w3a_first`
train 1291 labelled (83 positive), holdout 438 labelled (21 positive) | baseline AUC **0.5435** (HistGB) / 0.5144 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.4587 | +0.0848 | +0.1160 | -0.0237 |
| aim50_eps5_seed0 | 0.5761 | -0.0326 | -0.0175 | -0.0222 |
| ctgan_seed0 | 0.5369 | +0.0066 | -0.0110 | +0.0467 |
| ctgan_seed1 | 0.4502 | +0.0933 | -0.0152 | -0.0071 |
| ctgan_seed2 | 0.4056 | +0.1379 | +0.0259 | -0.1112 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.523 | +0.0205 | +0.0604 | -0.0212 |
| gaussian_copula_seed1 | 0.4837 | +0.0598 | +0.0629 | -0.0401 |
| gaussian_copula_seed2 | 0.5255 | +0.0180 | +0.0431 | -0.0625 |
| mst_eps10_seed0 | 0.4361 | +0.1074 | +0.0641 | +0.0140 |
| mst_eps15_seed0 | 0.4985 | +0.0450 | +0.0284 | -0.0414 |
| mst_eps15_seed1 | 0.6033 | -0.0598 | +0.0473 | -0.0681 |
| mst_eps15_seed2 | 0.5009 | +0.0426 | -0.0449 | -0.0088 |
| mst_eps1_seed0 | 0.4428 | +0.1007 | +0.0668 | +0.0472 |
| mst_eps20_seed0 | 0.4636 | +0.0799 | -0.0319 | -0.0071 |
| mst_eps5_seed0 | 0.5063 | +0.0372 | -0.1291 | -0.0012 |
| mst_eps8_seed0 | 0.5087 | +0.0348 | +0.1822 | +0.0274 |
| tvae_seed0 | 0.5487 | -0.0052 | -0.1239 | +0.0006 |
| tvae_seed1 | 0.543 | +0.0005 | -0.0935 | -0.0205 |
| tvae_seed2 | 0.5942 | -0.0507 | +0.0187 | +0.0545 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value).

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ |
|---|---|---|---|---|---|
| aim | 1 | 1 | +0.0961 | 0.0579 | -0.0225 |
| aim | 5 | 1 | +0.0866 | 0.1 | 0.0061 |
| ctgan | - | 3 | +0.1058 ± 0.021 | 0.0762 | -0.0267 |
| dpctgan | 5 | 1 | -0.0654 | 0.0808 | -0.0266 |
| dpctgan | 15 | 1 | +0.1747 | 0.2235 | 0.0061 |
| gaussian_copula | - | 3 | +0.1015 ± 0.0065 | 0.0873 | -0.0249 |
| mst | 1 | 1 | +0.0901 | 0.0928 | 0.0148 |
| mst | 5 | 1 | +0.0810 | 0.0339 | -0.0031 |
| mst | 8 | 1 | +0.1173 | 0.1197 | 0.0064 |
| mst | 10 | 1 | +0.0945 | 0.0547 | 0.0018 |
| mst | 15 | 3 | +0.0972 ± 0.003 | 0.0523 | -0.0091 |
| mst | 20 | 1 | +0.0780 | 0.0485 | -0.0007 |
| tvae | - | 3 | +0.0615 ± 0.0205 | 0.0179 | -0.0097 |
