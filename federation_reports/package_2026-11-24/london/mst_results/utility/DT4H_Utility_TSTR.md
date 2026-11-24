# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 754 rows | holdout test: 251 rows

## `encounter_primary_reason_CV_Disease_f5a_w3a_first`
train 239 labelled (42 positive), holdout 78 labelled (13 positive) | baseline AUC **0.4497** (95% CI 0.2612-0.6365) (HistGB) / 0.4959 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.4249 | 0.2779-0.596 | +0.0248 | -0.1550 | +0.0521 | +0.0379 |

## `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`
train 246 labelled (213 positive), holdout 80 labelled (68 positive) | baseline AUC **0.5772** (95% CI 0.384-0.7547) (HistGB) / 0.6152 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.6667 | 0.5093-0.8082 | -0.0895 | +0.1054 | +0.0025 | -0.0723 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| mst | 15 | 1 | -0.0323 (sd n/a: single run) | -0.0248 | 0.0273 | -0.0172 | 0.002 | 0.0916 |
