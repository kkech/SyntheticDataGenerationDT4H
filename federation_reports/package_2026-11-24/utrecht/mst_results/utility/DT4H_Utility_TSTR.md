# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 1862 rows | holdout test: 620 rows

## `encounter_primary_reason_HF_Disease_f5a_w1a_first`
train 456 labelled (223 positive), holdout 161 labelled (80 positive) | baseline AUC **0.7179** (95% CI 0.641-0.7901) (HistGB) / 0.6535 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.3923 | 0.3039-0.4773 | +0.3256 | +0.2253 | -0.0065 | +0.0082 |

## `encounter_primary_reason_CV_Disease_f5a_w1mo_first`
train 270 labelled (151 positive), holdout 98 labelled (59 positive) | baseline AUC **0.8661** (95% CI 0.776-0.9445) (HistGB) / 0.8205 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.4807 | 0.3633-0.6113 | +0.3854 | +0.4050 | +0.0022 | -0.0183 |

## `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`
train 270 labelled (119 positive), holdout 98 labelled (39 positive) | baseline AUC **0.8661** (95% CI 0.776-0.9445) (HistGB) / 0.8205 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.445 | 0.3298-0.5618 | +0.4211 | +0.4341 | +0.0044 | -0.0152 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| mst | 15 | 1 | +0.3774 (sd n/a: single run) | 0.3548 | 0.0 | -0.0084 | 0.3649 | 0.4125 |
