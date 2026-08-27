# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 3520 rows | holdout test: 1174 rows

## `encounter_primary_reason_CV_Disease_f5a_w1mo_first`
train 853 labelled (322 positive), holdout 264 labelled (96 positive) | baseline AUC **0.6861** (HistGB) / 0.6558 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5288 | +0.1573 | +0.1764 | +0.0119 |
| aim50_eps1_seed0 | 0.4849 | +0.2012 | +0.1004 | -0.0240 |
| ctgan_qt_seed0 | 0.5045 | +0.1816 | +0.1454 | -0.0209 |
| ctgan_seed0 | 0.4214 | +0.2647 | +0.1674 | -0.0484 |
| ctgan_seed1 | 0.5133 | +0.1728 | +0.1627 | -0.0571 |
| ctgan_seed2 | 0.5778 | +0.1083 | +0.1221 | -0.0002 |
| ddpm_g_seed0 | 0.5086 | +0.1775 | +0.1707 | +0.0101 |
| ddpm_seed0 | 0.5712 | +0.1149 | +0.1632 | +0.0028 |
| ddpm_seed1 | 0.5754 | +0.1107 | +0.1526 | +0.0068 |
| ddpm_seed2 | 0.5246 | +0.1615 | +0.2120 | -0.0004 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | 0.5 | +0.1861 | +0.1648 | -0.0134 |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | 0.563 | +0.1231 | +0.1483 | +0.0087 |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.4787 | +0.2074 | +0.1197 | -0.0715 |
| gaussian_copula_seed1 | 0.5521 | +0.1340 | +0.1631 | +0.0269 |
| gaussian_copula_seed2 | 0.5316 | +0.1545 | +0.0500 | -0.0126 |
| mst_eps0p5_seed0 | 0.4106 | +0.2755 | +0.1740 | -0.0060 |
| mst_eps10_seed0 | 0.4484 | +0.2377 | +0.1713 | -0.0101 |
| mst_eps15_seed0 | 0.4913 | +0.1948 | +0.1352 | -0.0063 |
| mst_eps15_seed1 | 0.5232 | +0.1629 | +0.1130 | -0.0069 |
| mst_eps15_seed2 | 0.4303 | +0.2558 | +0.1781 | -0.0081 |
| mst_eps1_seed0 | 0.4591 | +0.2270 | +0.1930 | +0.0151 |
| mst_eps20_seed0 | 0.4733 | +0.2128 | +0.1464 | -0.0131 |
| mst_eps5_seed0 | 0.5047 | +0.1814 | +0.1448 | +0.0085 |
| mst_eps8_seed0 | 0.4412 | +0.2449 | +0.2015 | -0.0308 |
| patectgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| patectgan_eps1_seed0 | 0.4769 | +0.2092 | +0.0562 | +0.0032 |
| patectgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| tvae_cap256_seed0 | 0.6407 | +0.0454 | +0.0108 | -0.0455 |
| tvae_ep1000_seed0 | 0.6293 | +0.0568 | +0.0991 | -0.0116 |
| tvae_ind_seed0 | 0.5252 | +0.1609 | +0.0995 | -0.0279 |
| tvae_qt_seed0 | 0.6212 | +0.0649 | +0.0140 | +0.0029 |
| tvae_qt_seed1 | 0.5822 | +0.1039 | +0.0281 | -0.0141 |
| tvae_qt_seed2 | 0.6069 | +0.0792 | +0.0166 | +0.0042 |
| tvae_seed0 | 0.5951 | +0.0910 | +0.0315 | -0.0036 |
| tvae_seed1 | 0.6696 | +0.0165 | +0.0676 | +0.0140 |
| tvae_seed2 | 0.6323 | +0.0538 | +0.0835 | -0.0014 |

## `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`
train 853 labelled (531 positive), holdout 264 labelled (168 positive) | baseline AUC **0.6861** (HistGB) / 0.6558 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4502 | +0.2359 | +0.1447 | -0.0201 |
| aim50_eps1_seed0 | 0.5269 | +0.1592 | +0.1832 | -0.0134 |
| ctgan_qt_seed0 | 0.5946 | +0.0915 | +0.1372 | -0.0006 |
| ctgan_seed0 | 0.4407 | +0.2454 | +0.2045 | -0.0671 |
| ctgan_seed1 | 0.5016 | +0.1845 | +0.1343 | -0.0172 |
| ctgan_seed2 | 0.6064 | +0.0797 | +0.1003 | +0.0273 |
| ddpm_g_seed0 | 0.425 | +0.2611 | +0.1677 | -0.0303 |
| ddpm_seed0 | 0.5448 | +0.1413 | +0.1284 | -0.0251 |
| ddpm_seed1 | 0.51 | +0.1761 | +0.1588 | -0.0021 |
| ddpm_seed2 | 0.496 | +0.1901 | +0.2052 | -0.0196 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | 0.5147 | +0.1714 | +0.1677 | -0.0125 |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.5281 | +0.1580 | +0.0843 | +0.0038 |
| gaussian_copula_seed1 | 0.5032 | +0.1829 | +0.1689 | -0.0347 |
| gaussian_copula_seed2 | 0.5464 | +0.1397 | +0.1698 | -0.0049 |
| mst_eps0p5_seed0 | 0.5382 | +0.1479 | +0.1517 | -0.0043 |
| mst_eps10_seed0 | 0.4629 | +0.2232 | +0.1740 | -0.0041 |
| mst_eps15_seed0 | 0.5388 | +0.1473 | +0.1217 | -0.0007 |
| mst_eps15_seed1 | 0.566 | +0.1201 | +0.0842 | -0.0120 |
| mst_eps15_seed2 | 0.4986 | +0.1875 | +0.1504 | -0.0263 |
| mst_eps1_seed0 | 0.4731 | +0.2130 | +0.1843 | -0.0006 |
| mst_eps20_seed0 | 0.5395 | +0.1466 | +0.1418 | -0.0194 |
| mst_eps5_seed0 | 0.504 | +0.1821 | +0.1812 | -0.0070 |
| mst_eps8_seed0 | 0.4743 | +0.2118 | +0.1834 | -0.0339 |
| patectgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| patectgan_eps1_seed0 | 0.5392 | +0.1469 | +0.1605 | -0.0154 |
| patectgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| tvae_cap256_seed0 | 0.628 | +0.0581 | +0.0075 | -0.0194 |
| tvae_ep1000_seed0 | 0.627 | +0.0591 | +0.1029 | -0.0281 |
| tvae_ind_seed0 | 0.5376 | +0.1485 | +0.1030 | -0.0398 |
| tvae_qt_seed0 | 0.6093 | +0.0768 | +0.0223 | -0.0015 |
| tvae_qt_seed1 | 0.5784 | +0.1077 | +0.0315 | +0.0033 |
| tvae_qt_seed2 | 0.6023 | +0.0838 | +0.0177 | -0.0190 |
| tvae_seed0 | 0.5972 | +0.0889 | +0.0266 | -0.0078 |
| tvae_seed1 | 0.6655 | +0.0206 | +0.0669 | +0.0071 |
| tvae_seed2 | 0.6039 | +0.0822 | +0.0822 | -0.0057 |

## `encounter_primary_reason_HF_Disease_f5a_w1mo_first`
train 853 labelled (98 positive), holdout 264 labelled (18 positive) | baseline AUC **0.5729** (HistGB) / 0.5438 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5689 | +0.0040 | +0.0395 | +0.0023 |
| aim50_eps1_seed0 | 0.6285 | -0.0556 | +0.0275 | +0.0822 |
| ctgan_qt_seed0 | 0.6344 | -0.0615 | +0.0363 | -0.0108 |
| ctgan_seed0 | 0.4481 | +0.1248 | -0.0179 | -0.0560 |
| ctgan_seed1 | 0.3726 | +0.2003 | +0.0036 | +0.0206 |
| ctgan_seed2 | 0.6192 | -0.0463 | -0.0422 | -0.0302 |
| ddpm_g_seed0 | 0.4523 | +0.1206 | -0.0183 | +0.0594 |
| ddpm_seed0 | 0.4076 | +0.1653 | +0.0305 | +0.0583 |
| ddpm_seed1 | 0.4627 | +0.1102 | +0.0336 | +0.0448 |
| ddpm_seed2 | 0.4456 | +0.1273 | +0.1933 | +0.0445 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.4257 | +0.1472 | +0.0515 | -0.0623 |
| gaussian_copula_seed1 | 0.5673 | +0.0056 | +0.0235 | +0.0019 |
| gaussian_copula_seed2 | 0.4736 | +0.0993 | +0.0741 | -0.0652 |
| mst_eps0p5_seed0 | - | target missing or single-class in synthetic data | - | - |
| mst_eps10_seed0 | 0.451 | +0.1219 | +0.0354 | -0.0004 |
| mst_eps15_seed0 | 0.6287 | -0.0558 | +0.0413 | -0.0119 |
| mst_eps15_seed1 | 0.5537 | +0.0192 | -0.0614 | +0.0147 |
| mst_eps15_seed2 | 0.4986 | +0.0743 | -0.0418 | -0.0013 |
| mst_eps1_seed0 | 0.4505 | +0.1224 | +0.0124 | -0.0182 |
| mst_eps20_seed0 | 0.5578 | +0.0151 | -0.0910 | +0.0448 |
| mst_eps5_seed0 | 0.4953 | +0.0776 | +0.0269 | -0.0474 |
| mst_eps8_seed0 | 0.6621 | -0.0892 | +0.0415 | -0.0182 |
| patectgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| patectgan_eps1_seed0 | 0.493 | +0.0799 | +0.0445 | -0.0546 |
| patectgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| tvae_cap256_seed0 | 0.5576 | +0.0153 | -0.0500 | +0.0138 |
| tvae_ep1000_seed0 | 0.516 | +0.0569 | -0.0002 | -0.0505 |
| tvae_ind_seed0 | 0.5111 | +0.0618 | -0.0260 | -0.0173 |
| tvae_qt_seed0 | 0.5851 | -0.0122 | +0.0831 | +0.0326 |
| tvae_qt_seed1 | 0.4526 | +0.1203 | +0.1274 | -0.0402 |
| tvae_qt_seed2 | 0.4058 | +0.1671 | +0.0813 | -0.1029 |
| tvae_seed0 | 0.4815 | +0.0914 | -0.0244 | -0.1113 |
| tvae_seed1 | 0.3304 | +0.2425 | +0.0842 | -0.0009 |
| tvae_seed2 | 0.4921 | +0.0808 | +0.0063 | -0.0962 |

## `cause_of_death_isAllCause_f5a_w3a_first`
train 1291 labelled (1208 positive), holdout 438 labelled (417 positive) | baseline AUC **0.5435** (HistGB) / 0.5144 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4153 | +0.1282 | +0.1362 | +0.0034 |
| aim50_eps1_seed0 | 0.4604 | +0.0831 | -0.0984 | -0.0776 |
| ctgan_qt_seed0 | 0.5272 | +0.0163 | -0.0203 | -0.0444 |
| ctgan_seed0 | 0.5014 | +0.0421 | +0.0371 | -0.0022 |
| ctgan_seed1 | 0.5305 | +0.0130 | +0.0975 | -0.0134 |
| ctgan_seed2 | 0.4542 | +0.0893 | +0.1244 | -0.1061 |
| ddpm_g_seed0 | 0.5976 | -0.0541 | +0.0648 | +0.0101 |
| ddpm_seed0 | 0.7039 | -0.1604 | +0.0684 | -0.0654 |
| ddpm_seed1 | 0.4878 | +0.0557 | +0.0210 | +0.1133 |
| ddpm_seed2 | 0.3966 | +0.1469 | +0.0636 | -0.0054 |
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
| mst_eps0p5_seed0 | 0.4801 | +0.0634 | +0.1257 | -0.0611 |
| mst_eps10_seed0 | 0.3995 | +0.1440 | -0.0663 | +0.0125 |
| mst_eps15_seed0 | 0.4921 | +0.0514 | -0.0430 | -0.0259 |
| mst_eps15_seed1 | 0.4629 | +0.0806 | +0.0004 | +0.0290 |
| mst_eps15_seed2 | 0.469 | +0.0745 | -0.0613 | -0.0173 |
| mst_eps1_seed0 | 0.5255 | +0.0180 | -0.0003 | -0.0519 |
| mst_eps20_seed0 | 0.4977 | +0.0458 | -0.0173 | +0.0186 |
| mst_eps5_seed0 | 0.4802 | +0.0633 | +0.0540 | +0.0322 |
| mst_eps8_seed0 | 0.6748 | -0.1313 | +0.0089 | -0.0096 |
| patectgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| patectgan_eps1_seed0 | 0.6267 | -0.0832 | -0.0706 | +0.0320 |
| patectgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| tvae_cap256_seed0 | 0.4612 | +0.0823 | +0.0256 | -0.0402 |
| tvae_ep1000_seed0 | 0.5432 | +0.0003 | -0.1114 | +0.0247 |
| tvae_ind_seed0 | 0.546 | -0.0025 | -0.0913 | -0.0421 |
| tvae_qt_seed0 | 0.5725 | -0.0290 | -0.0232 | +0.0430 |
| tvae_qt_seed1 | 0.5574 | -0.0139 | -0.0909 | -0.0066 |
| tvae_qt_seed2 | 0.4985 | +0.0450 | +0.0845 | -0.0617 |
| tvae_seed0 | 0.5293 | +0.0142 | -0.0984 | +0.0397 |
| tvae_seed1 | 0.5136 | +0.0299 | -0.1086 | -0.0085 |
| tvae_seed2 | 0.6139 | -0.0704 | +0.0388 | -0.0382 |

## `cause_of_death_isCV_f5a_w3a_first`
train 1291 labelled (83 positive), holdout 438 labelled (21 positive) | baseline AUC **0.5435** (HistGB) / 0.5144 (LogReg)

| run | TSTR AUC | gap | LogReg gap | augmentation Δ |
|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5751 | -0.0316 | +0.0228 | -0.0022 |
| aim50_eps1_seed0 | 0.5604 | -0.0169 | -0.0317 | -0.0324 |
| ctgan_qt_seed0 | 0.4303 | +0.1132 | +0.0536 | -0.0520 |
| ctgan_seed0 | 0.5761 | -0.0326 | -0.0684 | +0.1529 |
| ctgan_seed1 | 0.4428 | +0.1007 | +0.0286 | -0.0114 |
| ctgan_seed2 | 0.4122 | +0.1313 | +0.0217 | -0.0457 |
| ddpm_g_seed0 | 0.6113 | -0.0678 | +0.0648 | -0.0176 |
| ddpm_seed0 | 0.5398 | +0.0037 | -0.0010 | +0.0067 |
| ddpm_seed1 | 0.4906 | +0.0529 | +0.0379 | +0.0206 |
| ddpm_seed2 | 0.5375 | +0.0060 | +0.0454 | +0.0179 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps1_seed0 | 0.5625 | -0.0190 | -0.0070 | +0.0324 |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data | - | - |
| gaussian_copula_seed0 | 0.5466 | -0.0031 | +0.0664 | -0.0157 |
| gaussian_copula_seed1 | 0.5203 | +0.0232 | +0.0019 | +0.0330 |
| gaussian_copula_seed2 | 0.544 | -0.0005 | +0.0630 | -0.0787 |
| mst_eps0p5_seed0 | 0.5139 | +0.0296 | -0.1059 | +0.0408 |
| mst_eps10_seed0 | 0.6509 | -0.1074 | -0.2212 | -0.0397 |
| mst_eps15_seed0 | 0.6092 | -0.0657 | -0.0732 | -0.0133 |
| mst_eps15_seed1 | 0.349 | +0.1945 | -0.0581 | -0.0415 |
| mst_eps15_seed2 | 0.5012 | +0.0423 | -0.0960 | +0.0020 |
| mst_eps1_seed0 | 0.5487 | -0.0052 | +0.0309 | -0.0070 |
| mst_eps20_seed0 | 0.4536 | +0.0899 | -0.0163 | -0.0492 |
| mst_eps5_seed0 | 0.6258 | -0.0823 | +0.0702 | +0.0078 |
| mst_eps8_seed0 | 0.6355 | -0.0920 | -0.0392 | -0.0022 |
| patectgan_eps15_seed0 | - | target missing or single-class in synthetic data | - | - |
| patectgan_eps1_seed0 | 0.4235 | +0.1200 | -0.1066 | +0.0802 |
| patectgan_eps5_seed0 | - | target missing or single-class in synthetic data | - | - |
| tvae_cap256_seed0 | 0.4371 | +0.1064 | +0.0371 | -0.0476 |
| tvae_ep1000_seed0 | 0.5221 | +0.0214 | -0.1104 | +0.0419 |
| tvae_ind_seed0 | 0.5182 | +0.0253 | -0.1076 | -0.0011 |
| tvae_qt_seed0 | 0.576 | -0.0325 | -0.0176 | +0.0463 |
| tvae_qt_seed1 | 0.5397 | +0.0038 | -0.0790 | -0.0008 |
| tvae_qt_seed2 | 0.5141 | +0.0294 | +0.0858 | -0.0140 |
| tvae_seed0 | 0.5363 | +0.0072 | -0.0980 | -0.0037 |
| tvae_seed1 | 0.5004 | +0.0431 | -0.0951 | -0.0601 |
| tvae_seed2 | 0.6047 | -0.0612 | +0.0453 | +0.0170 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value).

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|
| aim | 1 | 1 | +0.0742 | 0.0362 | -0.013 | 0.084 | 0.2008 |
| aim40 | 1 | 1 | +0.0988 | 0.1039 | -0.0009 | 0.0751 | 0.2632 |
| ctgan | - | 3 | +0.1119 ± 0.0342 | 0.0717 | -0.0169 | 0.0374 | 0.2621 |
| ctgan_qt | - | 1 | +0.0682 | 0.0704 | -0.0257 | 0.0442 | 0.1926 |
| ddpm | - | 3 | +0.0935 ± 0.0373 | 0.1009 | 0.0132 | 0.1277 | 0.2261 |
| ddpm_g | - | 1 | +0.0875 | 0.0899 | 0.0063 | 0.1032 | 0.2017 |
| dpctgan | 1 | 1 | +0.0521 | 0.0706 | 0.0205 | 0.006 | 0.1716 |
| dpctgan | 15 | 1 | +0.1788 | 0.1663 | -0.0129 | 0.0217 | 0.3161 |
| gaussian_copula | - | 3 | +0.0847 ± 0.002 | 0.0787 | -0.0211 | 0.0203 | 0.2166 |
| mst | 0.5 | 1 | +0.1291 | 0.0864 | -0.0077 | 0.1196 | 0.2947 |
| mst | 1 | 1 | +0.1150 | 0.0841 | -0.0125 | 0.0799 | 0.2687 |
| mst | 5 | 1 | +0.0844 | 0.0954 | -0.0012 | 0.1324 | 0.2499 |
| mst | 8 | 1 | +0.0288 | 0.0792 | -0.0189 | 0.1032 | 0.1522 |
| mst | 10 | 1 | +0.1239 | 0.0186 | -0.0084 | 0.1705 | 0.2045 |
| mst | 15 | 3 | +0.0989 ± 0.039 | 0.026 | -0.0084 | 0.0808 | 0.1907 |
| mst | 20 | 1 | +0.1020 | 0.0327 | -0.0037 | 0.1486 | 0.2734 |
| patectgan | 1 | 1 | +0.0946 | 0.0168 | 0.0091 | 0.0264 | 0.2179 |
| tvae | - | 3 | +0.0487 ± 0.0281 | 0.0072 | -0.0173 | 0.0327 | 0.1618 |
| tvae_cap256 | - | 1 | +0.0615 | 0.0062 | -0.0278 | 0.0285 | 0.1452 |
| tvae_ep1000 | - | 1 | +0.0389 | -0.004 | -0.0047 | 0.0255 | 0.1275 |
| tvae_ind | - | 1 | +0.0788 | -0.0045 | -0.0256 | 0.0449 | 0.2696 |
| tvae_qt | - | 3 | +0.0530 ± 0.0351 | 0.0254 | -0.0086 | 0.0286 | 0.122 |
