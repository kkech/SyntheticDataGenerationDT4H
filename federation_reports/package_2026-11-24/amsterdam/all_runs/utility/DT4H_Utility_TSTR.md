# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 3520 rows | holdout test: 1174 rows

## `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`
train 853 labelled (531 positive), holdout 264 labelled (168 positive) | baseline AUC **0.6861** (95% CI 0.6193-0.7466) (HistGB) / 0.6558 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4502 | 0.3774-0.5255 | +0.2359 | +0.1447 | -0.0201 | -0.0229 |
| aim50_eps1_seed0 | 0.5269 | 0.4485-0.5949 | +0.1592 | +0.1832 | -0.0134 | -0.0317 |
| ctgan_qt_seed0 | 0.5946 | 0.5202-0.6647 | +0.0915 | +0.1372 | -0.0006 | -0.0082 |
| ctgan_seed0 | 0.4407 | 0.3754-0.5102 | +0.2454 | +0.2045 | -0.0671 | -0.0754 |
| ctgan_seed1 | 0.5016 | 0.4315-0.5743 | +0.1845 | +0.1343 | -0.0172 | -0.0193 |
| ctgan_seed2 | 0.6064 | 0.5343-0.6798 | +0.0797 | +0.1003 | +0.0273 | +0.0359 |
| ddpm_g_seed0 | 0.425 | 0.3506-0.4959 | +0.2611 | +0.1677 | -0.0303 | -0.0285 |
| ddpm_seed0 | 0.5448 | 0.4697-0.6171 | +0.1413 | +0.1284 | -0.0251 | -0.0284 |
| ddpm_seed1 | 0.51 | 0.4416-0.5831 | +0.1761 | +0.1588 | -0.0021 | +0.0180 |
| ddpm_seed2 | 0.496 | 0.4234-0.5698 | +0.1901 | +0.2052 | -0.0196 | -0.0152 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | 0.5147 | 0.4421-0.5841 | +0.1714 | +0.1677 | -0.0125 | -0.0118 |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.5281 | 0.4625-0.6005 | +0.1580 | +0.0843 | +0.0038 | +0.0212 |
| gaussian_copula_seed1 | 0.5032 | 0.43-0.5782 | +0.1829 | +0.1689 | -0.0347 | -0.0467 |
| gaussian_copula_seed2 | 0.5464 | 0.4709-0.6229 | +0.1397 | +0.1698 | -0.0049 | +0.0076 |
| mst_eps0p5_seed0 | 0.5382 | 0.4617-0.6098 | +0.1479 | +0.1517 | -0.0043 | -0.0034 |
| mst_eps10_seed0 | 0.4629 | 0.389-0.5378 | +0.2232 | +0.1740 | -0.0041 | +0.0040 |
| mst_eps15_seed0 | 0.5388 | 0.4716-0.6092 | +0.1473 | +0.1217 | -0.0007 | -0.0007 |
| mst_eps15_seed1 | 0.566 | 0.4916-0.6386 | +0.1201 | +0.0842 | -0.0120 | +0.0039 |
| mst_eps15_seed2 | 0.4986 | 0.4232-0.574 | +0.1875 | +0.1504 | -0.0263 | -0.0212 |
| mst_eps1_seed0 | 0.4731 | 0.4046-0.5413 | +0.2130 | +0.1843 | -0.0006 | -0.0038 |
| mst_eps20_seed0 | 0.5395 | 0.4673-0.6128 | +0.1466 | +0.1418 | -0.0194 | -0.0246 |
| mst_eps5_seed0 | 0.504 | 0.4282-0.5734 | +0.1821 | +0.1812 | -0.0070 | -0.0015 |
| mst_eps8_seed0 | 0.4743 | 0.4017-0.5455 | +0.2118 | +0.1834 | -0.0339 | -0.0404 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.5392 | 0.4674-0.6143 | +0.1469 | +0.1605 | -0.0154 | -0.0105 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.628 | 0.5597-0.6968 | +0.0581 | +0.0075 | -0.0194 | -0.0273 |
| tvae_ep1000_seed0 | 0.627 | 0.5536-0.6931 | +0.0591 | +0.1029 | -0.0281 | -0.0306 |
| tvae_ind_seed0 | 0.5376 | 0.4629-0.6143 | +0.1485 | +0.1030 | -0.0398 | -0.0399 |
| tvae_qt_seed0 | 0.6093 | 0.5336-0.6803 | +0.0768 | +0.0223 | -0.0015 | +0.0150 |
| tvae_qt_seed1 | 0.5784 | 0.5073-0.6497 | +0.1077 | +0.0315 | +0.0033 | -0.0055 |
| tvae_qt_seed2 | 0.6023 | 0.532-0.6674 | +0.0838 | +0.0177 | -0.0190 | -0.0175 |
| tvae_seed0 | 0.5972 | 0.5299-0.6628 | +0.0889 | +0.0266 | -0.0078 | -0.0210 |
| tvae_seed1 | 0.6655 | 0.5972-0.737 | +0.0206 | +0.0669 | +0.0071 | +0.0161 |
| tvae_seed2 | 0.6039 | 0.5324-0.667 | +0.0822 | +0.0822 | -0.0057 | -0.0113 |

## `encounter_primary_reason_CV_Disease_f5a_w1mo_first`
train 853 labelled (322 positive), holdout 264 labelled (96 positive) | baseline AUC **0.6861** (95% CI 0.6193-0.7466) (HistGB) / 0.6558 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5288 | 0.4535-0.6009 | +0.1573 | +0.1764 | +0.0119 | +0.0194 |
| aim50_eps1_seed0 | 0.4849 | 0.4085-0.5573 | +0.2012 | +0.1004 | -0.0240 | -0.0272 |
| ctgan_qt_seed0 | 0.5045 | 0.4371-0.5781 | +0.1816 | +0.1454 | -0.0209 | -0.0302 |
| ctgan_seed0 | 0.4214 | 0.3483-0.4919 | +0.2647 | +0.1674 | -0.0484 | -0.0442 |
| ctgan_seed1 | 0.5133 | 0.4365-0.5865 | +0.1728 | +0.1627 | -0.0571 | -0.0584 |
| ctgan_seed2 | 0.5778 | 0.5062-0.6473 | +0.1083 | +0.1221 | -0.0002 | +0.0197 |
| ddpm_g_seed0 | 0.5086 | 0.4384-0.5783 | +0.1775 | +0.1707 | +0.0101 | -0.0128 |
| ddpm_seed0 | 0.5712 | 0.5009-0.6395 | +0.1149 | +0.1632 | +0.0028 | +0.0007 |
| ddpm_seed1 | 0.5754 | 0.501-0.6463 | +0.1107 | +0.1526 | +0.0068 | +0.0081 |
| ddpm_seed2 | 0.5246 | 0.4549-0.5982 | +0.1615 | +0.2120 | -0.0004 | +0.0118 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | 0.5 | 0.5-0.5 | +0.1861 | +0.1648 | -0.0134 | +0.0013 |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | 0.563 | 0.489-0.6347 | +0.1231 | +0.1483 | +0.0087 | +0.0115 |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.4787 | 0.4059-0.5489 | +0.2074 | +0.1197 | -0.0715 | -0.0780 |
| gaussian_copula_seed1 | 0.5521 | 0.4831-0.6224 | +0.1340 | +0.1631 | +0.0269 | -0.0022 |
| gaussian_copula_seed2 | 0.5316 | 0.4602-0.6062 | +0.1545 | +0.0500 | -0.0126 | -0.0170 |
| mst_eps0p5_seed0 | 0.4106 | 0.3417-0.4798 | +0.2755 | +0.1740 | -0.0060 | +0.0137 |
| mst_eps10_seed0 | 0.4484 | 0.3775-0.5197 | +0.2377 | +0.1713 | -0.0101 | -0.0046 |
| mst_eps15_seed0 | 0.4913 | 0.4218-0.5611 | +0.1948 | +0.1352 | -0.0063 | -0.0109 |
| mst_eps15_seed1 | 0.5232 | 0.4483-0.596 | +0.1629 | +0.1130 | -0.0069 | +0.0012 |
| mst_eps15_seed2 | 0.4303 | 0.3576-0.5058 | +0.2558 | +0.1781 | -0.0081 | -0.0137 |
| mst_eps1_seed0 | 0.4591 | 0.3886-0.5328 | +0.2270 | +0.1930 | +0.0151 | +0.0220 |
| mst_eps20_seed0 | 0.4733 | 0.4072-0.5486 | +0.2128 | +0.1464 | -0.0131 | -0.0175 |
| mst_eps5_seed0 | 0.5047 | 0.4302-0.5807 | +0.1814 | +0.1448 | +0.0085 | +0.0102 |
| mst_eps8_seed0 | 0.4412 | 0.3699-0.5102 | +0.2449 | +0.2015 | -0.0308 | -0.0284 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.4769 | 0.4038-0.5529 | +0.2092 | +0.0562 | +0.0032 | +0.0060 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.6407 | 0.5739-0.7073 | +0.0454 | +0.0108 | -0.0455 | -0.0320 |
| tvae_ep1000_seed0 | 0.6293 | 0.5574-0.6929 | +0.0568 | +0.0991 | -0.0116 | -0.0145 |
| tvae_ind_seed0 | 0.5252 | 0.4514-0.6004 | +0.1609 | +0.0995 | -0.0279 | -0.0280 |
| tvae_qt_seed0 | 0.6212 | 0.5477-0.6918 | +0.0649 | +0.0140 | +0.0029 | +0.0009 |
| tvae_qt_seed1 | 0.5822 | 0.5148-0.6519 | +0.1039 | +0.0281 | -0.0141 | -0.0229 |
| tvae_qt_seed2 | 0.6069 | 0.5382-0.6713 | +0.0792 | +0.0166 | +0.0042 | +0.0004 |
| tvae_seed0 | 0.5951 | 0.5303-0.6584 | +0.0910 | +0.0315 | -0.0036 | -0.0001 |
| tvae_seed1 | 0.6696 | 0.5997-0.7397 | +0.0165 | +0.0676 | +0.0140 | +0.0159 |
| tvae_seed2 | 0.6323 | 0.5629-0.6987 | +0.0538 | +0.0835 | -0.0014 | +0.0025 |

## `encounter_primary_reason_HF_Disease_f5a_w1mo_first`
train 853 labelled (98 positive), holdout 264 labelled (18 positive) | baseline AUC **0.5729** (95% CI 0.4428-0.7) (HistGB) / 0.5438 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5689 | 0.3973-0.7473 | +0.0040 | +0.0395 | +0.0023 | +0.0138 |
| aim50_eps1_seed0 | 0.6285 | 0.4529-0.8002 | -0.0556 | +0.0275 | +0.0822 | +0.0492 |
| ctgan_qt_seed0 | 0.6344 | 0.4828-0.772 | -0.0615 | +0.0363 | -0.0108 | -0.0463 |
| ctgan_seed0 | 0.4481 | 0.3209-0.5661 | +0.1248 | -0.0179 | -0.0560 | -0.0915 |
| ctgan_seed1 | 0.3726 | 0.2725-0.4918 | +0.2003 | +0.0036 | +0.0206 | -0.0016 |
| ctgan_seed2 | 0.6192 | 0.487-0.7497 | -0.0463 | -0.0422 | -0.0302 | -0.0255 |
| ddpm_g_seed0 | 0.4523 | 0.3114-0.6014 | +0.1206 | -0.0183 | +0.0594 | +0.0695 |
| ddpm_seed0 | 0.4076 | 0.2672-0.551 | +0.1653 | +0.0305 | +0.0583 | +0.0334 |
| ddpm_seed1 | 0.4627 | 0.3028-0.627 | +0.1102 | +0.0336 | +0.0448 | +0.0364 |
| ddpm_seed2 | 0.4456 | 0.2888-0.6167 | +0.1273 | +0.1933 | +0.0445 | +0.0011 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.4257 | 0.2848-0.5841 | +0.1472 | +0.0515 | -0.0623 | -0.0786 |
| gaussian_copula_seed1 | 0.5673 | 0.4391-0.6922 | +0.0056 | +0.0235 | +0.0019 | -0.0108 |
| gaussian_copula_seed2 | 0.4736 | 0.311-0.6344 | +0.0993 | +0.0741 | -0.0652 | -0.0695 |
| mst_eps0p5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| mst_eps10_seed0 | 0.451 | 0.2904-0.6099 | +0.1219 | +0.0354 | -0.0004 | +0.0099 |
| mst_eps15_seed0 | 0.6287 | 0.5097-0.7581 | -0.0558 | +0.0413 | -0.0119 | +0.0023 |
| mst_eps15_seed1 | 0.5537 | 0.406-0.709 | +0.0192 | -0.0614 | +0.0147 | -0.0027 |
| mst_eps15_seed2 | 0.4986 | 0.364-0.6383 | +0.0743 | -0.0418 | -0.0013 | -0.0418 |
| mst_eps1_seed0 | 0.4505 | 0.3143-0.5933 | +0.1224 | +0.0124 | -0.0182 | -0.0092 |
| mst_eps20_seed0 | 0.5578 | 0.4126-0.7 | +0.0151 | -0.0910 | +0.0448 | +0.0147 |
| mst_eps5_seed0 | 0.4953 | 0.3641-0.6102 | +0.0776 | +0.0269 | -0.0474 | -0.0775 |
| mst_eps8_seed0 | 0.6621 | 0.582-0.7429 | -0.0892 | +0.0415 | -0.0182 | -0.0180 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.493 | 0.3699-0.6229 | +0.0799 | +0.0445 | -0.0546 | -0.0910 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.5576 | 0.4108-0.7087 | +0.0153 | -0.0500 | +0.0138 | +0.0250 |
| tvae_ep1000_seed0 | 0.516 | 0.3873-0.6342 | +0.0569 | -0.0002 | -0.0505 | -0.0666 |
| tvae_ind_seed0 | 0.5111 | 0.3394-0.6828 | +0.0618 | -0.0260 | -0.0173 | -0.0282 |
| tvae_qt_seed0 | 0.5851 | 0.436-0.7269 | -0.0122 | +0.0831 | +0.0326 | +0.0167 |
| tvae_qt_seed1 | 0.4526 | 0.3324-0.5675 | +0.1203 | +0.1274 | -0.0402 | -0.0405 |
| tvae_qt_seed2 | 0.4058 | 0.2814-0.534 | +0.1671 | +0.0813 | -0.1029 | -0.0849 |
| tvae_seed0 | 0.4815 | 0.3429-0.6059 | +0.0914 | -0.0244 | -0.1113 | -0.1204 |
| tvae_seed1 | 0.3304 | 0.2172-0.4548 | +0.2425 | +0.0842 | -0.0009 | +0.0065 |
| tvae_seed2 | 0.4921 | 0.3144-0.653 | +0.0808 | +0.0063 | -0.0962 | -0.1498 |

## `cause_of_death_isCV_f5a_w3a_first`
train 1291 labelled (83 positive), holdout 438 labelled (21 positive) | baseline AUC **0.5435** (95% CI 0.3988-0.6783) (HistGB) / 0.5144 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5751 | 0.4597-0.6976 | -0.0316 | +0.0228 | -0.0022 | -0.0141 |
| aim50_eps1_seed0 | 0.5604 | 0.4345-0.6894 | -0.0169 | -0.0317 | -0.0324 | -0.0071 |
| ctgan_qt_seed0 | 0.4303 | 0.3442-0.5127 | +0.1132 | +0.0536 | -0.0520 | -0.0623 |
| ctgan_seed0 | 0.5761 | 0.4406-0.6963 | -0.0326 | -0.0684 | +0.1529 | +0.1833 |
| ctgan_seed1 | 0.4428 | 0.3243-0.5668 | +0.1007 | +0.0286 | -0.0114 | -0.0286 |
| ctgan_seed2 | 0.4122 | 0.2858-0.5383 | +0.1313 | +0.0217 | -0.0457 | -0.0588 |
| ddpm_g_seed0 | 0.6113 | 0.4905-0.7206 | -0.0678 | +0.0648 | -0.0176 | -0.0007 |
| ddpm_seed0 | 0.5398 | 0.4287-0.6515 | +0.0037 | -0.0010 | +0.0067 | -0.0044 |
| ddpm_seed1 | 0.4906 | 0.3486-0.6455 | +0.0529 | +0.0379 | +0.0206 | +0.0307 |
| ddpm_seed2 | 0.5375 | 0.415-0.6476 | +0.0060 | +0.0454 | +0.0179 | +0.0089 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | 0.5625 | 0.4445-0.6818 | -0.0190 | -0.0070 | +0.0324 | -0.0502 |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.5466 | 0.4067-0.6843 | -0.0031 | +0.0664 | -0.0157 | +0.0118 |
| gaussian_copula_seed1 | 0.5203 | 0.3826-0.6564 | +0.0232 | +0.0019 | +0.0330 | +0.0108 |
| gaussian_copula_seed2 | 0.544 | 0.4131-0.6883 | -0.0005 | +0.0630 | -0.0787 | -0.0787 |
| mst_eps0p5_seed0 | 0.5139 | 0.3525-0.6585 | +0.0296 | -0.1059 | +0.0408 | +0.0557 |
| mst_eps10_seed0 | 0.6509 | 0.5434-0.7622 | -0.1074 | -0.2212 | -0.0397 | +0.0009 |
| mst_eps15_seed0 | 0.6092 | 0.5082-0.7078 | -0.0657 | -0.0732 | -0.0133 | -0.0121 |
| mst_eps15_seed1 | 0.349 | 0.2535-0.4514 | +0.1945 | -0.0581 | -0.0415 | -0.0251 |
| mst_eps15_seed2 | 0.5012 | 0.3879-0.6227 | +0.0423 | -0.0960 | +0.0020 | +0.0095 |
| mst_eps1_seed0 | 0.5487 | 0.4293-0.6618 | -0.0052 | +0.0309 | -0.0070 | -0.0132 |
| mst_eps20_seed0 | 0.4536 | 0.32-0.594 | +0.0899 | -0.0163 | -0.0492 | -0.0690 |
| mst_eps5_seed0 | 0.6258 | 0.522-0.7217 | -0.0823 | +0.0702 | +0.0078 | +0.0186 |
| mst_eps8_seed0 | 0.6355 | 0.509-0.753 | -0.0920 | -0.0392 | -0.0022 | -0.0175 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.4235 | 0.2776-0.5773 | +0.1200 | -0.1066 | +0.0802 | +0.1088 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.4371 | 0.316-0.5453 | +0.1064 | +0.0371 | -0.0476 | -0.0606 |
| tvae_ep1000_seed0 | 0.5221 | 0.3931-0.6456 | +0.0214 | -0.1104 | +0.0419 | +0.0585 |
| tvae_ind_seed0 | 0.5182 | 0.3906-0.645 | +0.0253 | -0.1076 | -0.0011 | -0.0011 |
| tvae_qt_seed0 | 0.576 | 0.4403-0.7084 | -0.0325 | -0.0176 | +0.0463 | +0.0341 |
| tvae_qt_seed1 | 0.5397 | 0.4142-0.6776 | +0.0038 | -0.0790 | -0.0008 | +0.0245 |
| tvae_qt_seed2 | 0.5141 | 0.396-0.6578 | +0.0294 | +0.0858 | -0.0140 | +0.0010 |
| tvae_seed0 | 0.5363 | 0.4102-0.6822 | +0.0072 | -0.0980 | -0.0037 | +0.0172 |
| tvae_seed1 | 0.5004 | 0.381-0.6189 | +0.0431 | -0.0951 | -0.0601 | -0.0758 |
| tvae_seed2 | 0.6047 | 0.4964-0.7111 | -0.0612 | +0.0453 | +0.0170 | +0.0727 |

## `cause_of_death_isAllCause_f5a_w3a_first`
train 1291 labelled (1208 positive), holdout 438 labelled (417 positive) | baseline AUC **0.5435** (95% CI 0.3988-0.6783) (HistGB) / 0.5144 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4153 | 0.2979-0.5263 | +0.1282 | +0.1362 | +0.0034 | -0.0312 |
| aim50_eps1_seed0 | 0.4604 | 0.3202-0.6029 | +0.0831 | -0.0984 | -0.0776 | -0.0707 |
| ctgan_qt_seed0 | 0.5272 | 0.4186-0.6305 | +0.0163 | -0.0203 | -0.0444 | -0.0090 |
| ctgan_seed0 | 0.5014 | 0.3762-0.6259 | +0.0421 | +0.0371 | -0.0022 | -0.0041 |
| ctgan_seed1 | 0.5305 | 0.3873-0.6717 | +0.0130 | +0.0975 | -0.0134 | -0.0341 |
| ctgan_seed2 | 0.4542 | 0.3193-0.6008 | +0.0893 | +0.1244 | -0.1061 | -0.0852 |
| ddpm_g_seed0 | 0.5976 | 0.459-0.7457 | -0.0541 | +0.0648 | +0.0101 | -0.0275 |
| ddpm_seed0 | 0.7039 | 0.5823-0.8184 | -0.1604 | +0.0684 | -0.0654 | -0.0843 |
| ddpm_seed1 | 0.4878 | 0.3549-0.6305 | +0.0557 | +0.0210 | +0.1133 | +0.0805 |
| ddpm_seed2 | 0.3966 | 0.2791-0.5269 | +0.1469 | +0.0636 | -0.0054 | +0.0086 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.6404 | 0.5254-0.7577 | -0.0969 | +0.0325 | -0.0143 | -0.0234 |
| gaussian_copula_seed1 | 0.4639 | 0.3389-0.591 | +0.0796 | +0.0045 | +0.0294 | +0.0402 |
| gaussian_copula_seed2 | 0.5041 | 0.3816-0.6281 | +0.0394 | +0.1079 | -0.0512 | -0.0255 |
| mst_eps0p5_seed0 | 0.4801 | 0.3369-0.6055 | +0.0634 | +0.1257 | -0.0611 | -0.0679 |
| mst_eps10_seed0 | 0.3995 | 0.289-0.5108 | +0.1440 | -0.0663 | +0.0125 | +0.0194 |
| mst_eps15_seed0 | 0.4921 | 0.3804-0.5901 | +0.0514 | -0.0430 | -0.0259 | -0.0357 |
| mst_eps15_seed1 | 0.4629 | 0.3326-0.5948 | +0.0806 | +0.0004 | +0.0290 | +0.0454 |
| mst_eps15_seed2 | 0.469 | 0.3223-0.6233 | +0.0745 | -0.0613 | -0.0173 | -0.0326 |
| mst_eps1_seed0 | 0.5255 | 0.3866-0.6496 | +0.0180 | -0.0003 | -0.0519 | -0.0690 |
| mst_eps20_seed0 | 0.4977 | 0.3725-0.6386 | +0.0458 | -0.0173 | +0.0186 | +0.0078 |
| mst_eps5_seed0 | 0.4802 | 0.361-0.5967 | +0.0633 | +0.0540 | +0.0322 | +0.0391 |
| mst_eps8_seed0 | 0.6748 | 0.5621-0.7871 | -0.1313 | +0.0089 | -0.0096 | -0.0294 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.6267 | 0.5107-0.7331 | -0.0832 | -0.0706 | +0.0320 | +0.0241 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.4612 | 0.3384-0.572 | +0.0823 | +0.0256 | -0.0402 | -0.0084 |
| tvae_ep1000_seed0 | 0.5432 | 0.4142-0.6594 | +0.0003 | -0.1114 | +0.0247 | -0.0053 |
| tvae_ind_seed0 | 0.546 | 0.4203-0.6788 | -0.0025 | -0.0913 | -0.0421 | +0.0080 |
| tvae_qt_seed0 | 0.5725 | 0.4383-0.7039 | -0.0290 | -0.0232 | +0.0430 | +0.0504 |
| tvae_qt_seed1 | 0.5574 | 0.4456-0.6898 | -0.0139 | -0.0909 | -0.0066 | -0.0054 |
| tvae_qt_seed2 | 0.4985 | 0.3617-0.6531 | +0.0450 | +0.0845 | -0.0617 | -0.0429 |
| tvae_seed0 | 0.5293 | 0.3946-0.6635 | +0.0142 | -0.0984 | +0.0397 | +0.0804 |
| tvae_seed1 | 0.5136 | 0.388-0.6444 | +0.0299 | -0.1086 | -0.0085 | +0.0080 |
| tvae_seed2 | 0.6139 | 0.512-0.71 | -0.0704 | +0.0388 | -0.0382 | -0.0338 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| aim | 1 | 1 | +0.0742 (sd n/a: single run) | 0.0362 | -0.013 | -0.0175 | 0.084 | 0.2008 |
| aim40 | 1 | 1 | +0.0988 (sd n/a: single run) | 0.1039 | -0.0009 | -0.007 | 0.0751 | 0.2632 |
| ctgan | - | 3 | +0.1119 ± 0.0342 | 0.0717 | -0.0169 | -0.0192 | 0.0374 | 0.2621 |
| ctgan_qt | - | 1 | +0.0682 (sd n/a: single run) | 0.0704 | -0.0257 | -0.0312 | 0.0442 | 0.1926 |
| ddpm | - | 3 | +0.0935 ± 0.0373 | 0.1009 | 0.0132 | 0.0071 | 0.1277 | 0.2261 |
| ddpm_g | - | 1 | +0.0875 (sd n/a: single run) | 0.0899 | 0.0063 | 0.0 | 0.1032 | 0.2017 |
| dpctgan | 1 | 1 | +0.0521 (sd n/a: single run) | 0.0706 | 0.0205 | -0.0193 | 0.006 | 0.1716 |
| dpctgan | 15 | 1 | +0.1788 (sd n/a: single run) | 0.1663 | -0.0129 | -0.0052 | 0.0217 | 0.3161 |
| gaussian_copula | - | 3 | +0.0847 ± 0.002 | 0.0787 | -0.0211 | -0.0226 | 0.0203 | 0.2166 |
| mst | 0.5 | 1 | +0.1291 (sd n/a: single run) | 0.0864 | -0.0076 | -0.0005 | 0.1196 | 0.2947 |
| mst | 1 | 1 | +0.1150 (sd n/a: single run) | 0.0841 | -0.0125 | -0.0146 | 0.0799 | 0.2687 |
| mst | 5 | 1 | +0.0844 (sd n/a: single run) | 0.0954 | -0.0012 | -0.0022 | 0.1324 | 0.2499 |
| mst | 8 | 1 | +0.0288 (sd n/a: single run) | 0.0792 | -0.0189 | -0.0267 | 0.1032 | 0.1522 |
| mst | 10 | 1 | +0.1239 (sd n/a: single run) | 0.0186 | -0.0084 | 0.0059 | 0.1705 | 0.2045 |
| mst | 15 | 3 | +0.0989 ± 0.039 | 0.026 | -0.0084 | -0.0089 | 0.0808 | 0.1907 |
| mst | 20 | 1 | +0.1020 (sd n/a: single run) | 0.0327 | -0.0037 | -0.0177 | 0.1486 | 0.2734 |
| patectgan | 1 | 1 | +0.0946 (sd n/a: single run) | 0.0168 | 0.0091 | 0.0075 | 0.0264 | 0.2179 |
| tvae | - | 3 | +0.0487 ± 0.0281 | 0.0072 | -0.0173 | -0.0129 | 0.0327 | 0.1618 |
| tvae_cap256 | - | 1 | +0.0615 (sd n/a: single run) | 0.0062 | -0.0278 | -0.0207 | 0.0285 | 0.1452 |
| tvae_ep1000 | - | 1 | +0.0389 (sd n/a: single run) | -0.004 | -0.0047 | -0.0117 | 0.0255 | 0.1275 |
| tvae_ind | - | 1 | +0.0788 (sd n/a: single run) | -0.0045 | -0.0256 | -0.0178 | 0.0449 | 0.2696 |
| tvae_qt | - | 3 | +0.0530 ± 0.0351 | 0.0254 | -0.0086 | -0.0051 | 0.0286 | 0.122 |
