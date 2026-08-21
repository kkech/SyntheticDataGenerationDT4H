# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on real data (baseline) and on each synthetic dataset, then both are scored on the same held-out real test split. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.


## `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`
1117 labelled real records (699 positive) | baseline AUC **0.6713**

| synthesizer | TSTR AUC | gap vs baseline |
|---|---|---|
| ctgan | 0.6624 | +0.0089 |
| dpctgan | - | target missing or single-class in synthetic data |
| gaussian_copula | 0.5747 | +0.0966 |
| mst | 0.415 | +0.2563 |
| tvae | 0.602 | +0.0693 |

## `encounter_primary_reason_CV_Disease_f5a_w1mo_first`
1117 labelled real records (418 positive) | baseline AUC **0.6422**

| synthesizer | TSTR AUC | gap vs baseline |
|---|---|---|
| ctgan | 0.5358 | +0.1064 |
| dpctgan | 0.5877 | +0.0545 |
| gaussian_copula | 0.5322 | +0.1100 |
| mst | 0.5666 | +0.0756 |
| tvae | 0.5759 | +0.0663 |

## `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`
472 labelled real records (296 positive) | baseline AUC **0.6732**

| synthesizer | TSTR AUC | gap vs baseline |
|---|---|---|
| ctgan | 0.5402 | +0.1330 |
| dpctgan | - | target missing or single-class in synthetic data |
| gaussian_copula | 0.492 | +0.1812 |
| mst | 0.4183 | +0.2549 |
| tvae | 0.5283 | +0.1449 |

## `encounter_primary_reason_CV_Disease_f5a_w7d_first`
472 labelled real records (176 positive) | baseline AUC **0.7343**

| synthesizer | TSTR AUC | gap vs baseline |
|---|---|---|
| ctgan | 0.4579 | +0.2764 |
| dpctgan | - | target missing or single-class in synthetic data |
| gaussian_copula | 0.5313 | +0.2030 |
| mst | 0.5034 | +0.2309 |
| tvae | 0.6413 | +0.0930 |

## `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`
2186 labelled real records (1403 positive) | baseline AUC **0.6397**

| synthesizer | TSTR AUC | gap vs baseline |
|---|---|---|
| ctgan | 0.5594 | +0.0803 |
| dpctgan | 0.453 | +0.1867 |
| gaussian_copula | 0.5445 | +0.0952 |
| mst | 0.4738 | +0.1659 |
| tvae | 0.6022 | +0.0375 |

## Mean AUC gap across targets (lower is better)

| synthesizer | mean gap |
|---|---|
| ctgan | +0.1210 |
| dpctgan | +0.1206 |
| gaussian_copula | +0.1372 |
| mst | +0.1967 |
| tvae | +0.0822 |
