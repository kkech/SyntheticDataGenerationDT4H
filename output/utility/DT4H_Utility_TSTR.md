# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 3520 rows | holdout test: 1174 rows

## `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`
train 853 labelled (531 positive), holdout 264 labelled (168 positive) | baseline AUC **0.6861** (HistGB) / 0.6558 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.5463 | +0.1398 | +0.0697 | +0.0096 |
| aim50_eps5_seed0 | 0.4595 | +0.2266 | +0.2674 | +0.0070 |
| ctgan_seed0 | 0.4407 | +0.2454 | +0.2045 | -0.0671 |
| ctgan_seed1 | 0.5016 | +0.1845 | +0.1343 | -0.0172 |
| ctgan_seed2 | 0.6064 | +0.0797 | +0.1003 | +0.0273 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | 0.5075 | +0.1786 | +0.2168 | +0.0232 |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.5281 | +0.1580 | +0.0843 | +0.0038 |
| gaussian_copula_seed1 | 0.5032 | +0.1829 | +0.1689 | -0.0347 |
| gaussian_copula_seed2 | 0.5464 | +0.1397 | +0.1698 | -0.0049 |
| mst_eps10_seed0 | 0.5409 | +0.1452 | +0.2280 | +0.0080 |
| mst_eps15_seed0 | 0.5417 | +0.1444 | +0.0876 | -0.0041 |
| mst_eps15_seed1 | 0.5217 | +0.1644 | +0.1105 | +0.0066 |
| mst_eps15_seed2 | 0.5466 | +0.1395 | +0.1356 | -0.0042 |
| mst_eps1_seed0 | 0.5596 | +0.1265 | +0.1066 | +0.0082 |
| mst_eps20_seed0 | 0.498 | +0.1881 | +0.1377 | -0.0039 |
| mst_eps5_seed0 | 0.5148 | +0.1713 | +0.0917 | -0.0103 |
| mst_eps8_seed0 | 0.4785 | +0.2076 | +0.1325 | +0.0116 |
| tvae_seed0 | 0.5972 | +0.0889 | +0.0266 | -0.0078 |
| tvae_seed1 | 0.6655 | +0.0206 | +0.0669 | +0.0071 |
| tvae_seed2 | 0.6039 | +0.0822 | +0.0822 | -0.0057 |

## `encounter_primary_reason_CV_Disease_f5a_w1mo_first`
train 853 labelled (322 positive), holdout 264 labelled (96 positive) | baseline AUC **0.6861** (HistGB) / 0.6558 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.5474 | +0.1387 | +0.0898 | +0.0063 |
| aim50_eps5_seed0 | 0.4642 | +0.2219 | +0.2799 | -0.0338 |
| ctgan_seed0 | 0.4214 | +0.2647 | +0.1674 | -0.0484 |
| ctgan_seed1 | 0.5133 | +0.1728 | +0.1627 | -0.0571 |
| ctgan_seed2 | 0.5778 | +0.1083 | +0.1221 | -0.0002 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.4787 | +0.2074 | +0.1197 | -0.0715 |
| gaussian_copula_seed1 | 0.5521 | +0.1340 | +0.1631 | +0.0269 |
| gaussian_copula_seed2 | 0.5316 | +0.1545 | +0.0500 | -0.0126 |
| mst_eps10_seed0 | 0.5257 | +0.1604 | +0.2174 | -0.0052 |
| mst_eps15_seed0 | 0.5177 | +0.1684 | +0.0997 | -0.0017 |
| mst_eps15_seed1 | 0.5133 | +0.1728 | +0.0595 | +0.0047 |
| mst_eps15_seed2 | 0.4073 | +0.2788 | +0.1586 | -0.0136 |
| mst_eps1_seed0 | 0.5187 | +0.1674 | +0.1554 | -0.0097 |
| mst_eps20_seed0 | 0.5249 | +0.1612 | +0.1614 | +0.0160 |
| mst_eps5_seed0 | 0.5275 | +0.1586 | +0.0940 | -0.0083 |
| mst_eps8_seed0 | 0.4813 | +0.2048 | +0.1308 | +0.0157 |
| tvae_seed0 | 0.5951 | +0.0910 | +0.0315 | -0.0036 |
| tvae_seed1 | 0.6696 | +0.0165 | +0.0676 | +0.0140 |
| tvae_seed2 | 0.6323 | +0.0538 | +0.0835 | -0.0014 |

## `encounter_primary_reason_HF_Disease_f5a_w1mo_first`
train 853 labelled (98 positive), holdout 264 labelled (18 positive) | baseline AUC **0.5729** (HistGB) / 0.5438 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.5506 | +0.0223 | +0.0102 | +0.0152 |
| aim50_eps5_seed0 | 0.54 | +0.0329 | +0.0090 | +0.0215 |
| ctgan_seed0 | 0.4481 | +0.1248 | -0.0179 | -0.0560 |
| ctgan_seed1 | 0.3726 | +0.2003 | +0.0036 | +0.0206 |
| ctgan_seed2 | 0.6192 | -0.0463 | -0.0422 | -0.0302 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | 0.5574 | +0.0155 | +0.0911 | -0.0230 |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.4257 | +0.1472 | +0.0515 | -0.0623 |
| gaussian_copula_seed1 | 0.5673 | +0.0056 | +0.0235 | +0.0019 |
| gaussian_copula_seed2 | 0.4736 | +0.0993 | +0.0741 | -0.0652 |
| mst_eps10_seed0 | 0.5138 | +0.0591 | +0.0919 | +0.0177 |
| mst_eps15_seed0 | 0.5928 | -0.0199 | -0.0077 | +0.0125 |
| mst_eps15_seed1 | 0.5095 | +0.0634 | +0.0009 | +0.0077 |
| mst_eps15_seed2 | 0.6579 | -0.0850 | -0.0574 | +0.0271 |
| mst_eps1_seed0 | 0.5781 | -0.0052 | +0.1027 | -0.0273 |
| mst_eps20_seed0 | 0.6879 | -0.1150 | -0.0538 | +0.0579 |
| mst_eps5_seed0 | 0.5838 | -0.0109 | -0.0678 | +0.0131 |
| mst_eps8_seed0 | 0.6107 | -0.0378 | -0.0280 | -0.0094 |
| tvae_seed0 | 0.4815 | +0.0914 | -0.0244 | -0.1113 |
| tvae_seed1 | 0.3304 | +0.2425 | +0.0842 | -0.0009 |
| tvae_seed2 | 0.4921 | +0.0808 | +0.0063 | -0.0962 |

## `cause_of_death_isAllCause_f5a_w3a_first`
train 1291 labelled (1208 positive), holdout 438 labelled (417 positive) | baseline AUC **0.5435** (HistGB) / 0.5144 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.4407 | +0.1028 | +0.1637 | -0.0873 |
| aim50_eps5_seed0 | 0.5079 | +0.0356 | +0.0558 | +0.0008 |
| ctgan_seed0 | 0.5014 | +0.0421 | +0.0371 | -0.0022 |
| ctgan_seed1 | 0.5305 | +0.0130 | +0.0975 | -0.0134 |
| ctgan_seed2 | 0.4542 | +0.0893 | +0.1244 | -0.1061 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.6404 | -0.0969 | +0.0325 | -0.0143 |
| gaussian_copula_seed1 | 0.4639 | +0.0796 | +0.0045 | +0.0294 |
| gaussian_copula_seed2 | 0.5041 | +0.0394 | +0.1079 | -0.0512 |
| mst_eps10_seed0 | 0.5713 | -0.0278 | -0.0609 | -0.0018 |
| mst_eps15_seed0 | 0.4932 | +0.0503 | -0.0621 | -0.0124 |
| mst_eps15_seed1 | 0.4033 | +0.1402 | -0.0981 | +0.0019 |
| mst_eps15_seed2 | 0.3703 | +0.1732 | -0.1176 | +0.0315 |
| mst_eps1_seed0 | 0.4714 | +0.0721 | +0.1087 | +0.0261 |
| mst_eps20_seed0 | 0.4398 | +0.1037 | +0.0035 | -0.0471 |
| mst_eps5_seed0 | 0.5469 | -0.0034 | -0.0746 | +0.0118 |
| mst_eps8_seed0 | 0.4495 | +0.0940 | -0.0872 | -0.0114 |
| tvae_seed0 | 0.5293 | +0.0142 | -0.0984 | +0.0397 |
| tvae_seed1 | 0.5136 | +0.0299 | -0.1086 | -0.0085 |
| tvae_seed2 | 0.6139 | -0.0704 | +0.0388 | -0.0382 |

## `cause_of_death_isCV_f5a_w3a_first`
train 1291 labelled (83 positive), holdout 438 labelled (21 positive) | baseline AUC **0.5435** (HistGB) / 0.5144 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.3797 | +0.1638 | +0.1585 | -0.0768 |
| aim50_eps5_seed0 | 0.5639 | -0.0204 | -0.0402 | +0.0133 |
| ctgan_seed0 | 0.5761 | -0.0326 | -0.0684 | +0.1529 |
| ctgan_seed1 | 0.4428 | +0.1007 | +0.0286 | -0.0114 |
| ctgan_seed2 | 0.4122 | +0.1313 | +0.0217 | -0.0457 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.5466 | -0.0031 | +0.0664 | -0.0157 |
| gaussian_copula_seed1 | 0.5203 | +0.0232 | +0.0019 | +0.0330 |
| gaussian_copula_seed2 | 0.544 | -0.0005 | +0.0630 | -0.0787 |
| mst_eps10_seed0 | 0.5575 | -0.0140 | -0.0101 | -0.0168 |
| mst_eps15_seed0 | 0.5231 | +0.0204 | -0.0762 | +0.0081 |
| mst_eps15_seed1 | 0.5872 | -0.0437 | -0.0892 | -0.0237 |
| mst_eps15_seed2 | 0.4577 | +0.0858 | -0.1339 | -0.0502 |
| mst_eps1_seed0 | 0.4957 | +0.0478 | +0.0350 | -0.0269 |
| mst_eps20_seed0 | 0.4605 | +0.0830 | -0.0811 | +0.0142 |
| mst_eps5_seed0 | 0.55 | -0.0065 | -0.0058 | -0.0011 |
| mst_eps8_seed0 | 0.5324 | +0.0111 | -0.0209 | +0.0486 |
| tvae_seed0 | 0.5363 | +0.0072 | -0.0980 | -0.0037 |
| tvae_seed1 | 0.5004 | +0.0431 | -0.0951 | -0.0601 |
| tvae_seed2 | 0.6047 | -0.0612 | +0.0453 | +0.0170 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value).

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ |
|---|---|---|---|---|---|
| aim | 1 | 1 | +0.1135 | 0.0984 | -0.0266 |
| aim | 5 | 1 | +0.0993 | 0.1144 | 0.0018 |
| ctgan | - | 3 | +0.1119 ± 0.0342 | 0.0717 | -0.0169 |
| dpctgan | 5 | 1 | +0.0155 | 0.0911 | -0.023 |
| dpctgan | 15 | 1 | +0.1786 | 0.2168 | 0.0232 |
| gaussian_copula | - | 3 | +0.0847 ± 0.002 | 0.0787 | -0.0211 |
| mst | 1 | 1 | +0.0817 | 0.1017 | -0.0059 |
| mst | 5 | 1 | +0.0618 | 0.0075 | 0.001 |
| mst | 8 | 1 | +0.0959 | 0.0254 | 0.011 |
| mst | 10 | 1 | +0.0646 | 0.0933 | 0.0004 |
| mst | 15 | 3 | +0.0969 ± 0.023 | 0.0007 | -0.0007 |
| mst | 20 | 1 | +0.0842 | 0.0335 | 0.0074 |
| tvae | - | 3 | +0.0487 ± 0.0281 | 0.0072 | -0.0173 |
