# Survival Fidelity

Endpoints use the five-year follow-up columns: a recorded days-to-event is an event, a null is administrative censoring at 1825 days -- the same rule for real and synthetic data. A recorded time beyond 1825 days is treated as censoring at 1825 (out of horizon), counted per frame as `times_beyond_horizon`. The train-vs-holdout log-rank p-value calibrates what pure sampling noise looks like.

Disclosure: synthetic days-to-event values below the real observed minimum were nulled by the sentinel decode upstream and are read here as censoring; those erased early events cannot be recovered from the released CSVs (see `decode_note` in the JSON).

Effect-replication covariates are standardized in EVERY frame (real train, real holdout, synthetic) by the REAL TRAIN split's mean/SD, so scale infidelity in a synthetic frame shows up as a coefficient discrepancy instead of being re-normalized away.

## all_cause_death

train events 633/1862 | holdout 223/620 | log-rank train-vs-holdout p = 0.431

| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |
|---|---|---|---|---|
| **train** | 0.78571 | 0.66058 | - | no |
| **holdout** | 0.78226 | 0.64194 |  |  |
| mst_eps15_seed0 | 0.77121 | 0.67078 | 0.023 | no |

Equivalence is a POSITIVE claim (90% CI of the survival difference within ±5pp at every horizon) -- unlike a non-significant log-rank, which is only absence of evidence.

## hf_rehospitalization

train events 266/1862 | holdout 96/620 | log-rank train-vs-holdout p = 0.4605

| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |
|---|---|---|---|---|
| **train** | 0.88077 | 0.85714 | - | yes ✅ |
| **holdout** | 0.87097 | 0.84516 |  |  |
| mst_eps15_seed0 | 0.87057 | 0.85714 | 0.178 | yes ✅ |

Equivalence is a POSITIVE claim (90% CI of the survival difference within ±5pp at every horizon) -- unlike a non-significant log-rank, which is only absence of evidence.

## Effect-estimate replication

Model: cox_ph (lifelines), per-SD coefficients standardized by the real TRAIN split's mean/SD in every frame. The coefficient error is computed only over covariates present in BOTH the real and the synthetic fit (`matched`); runs sharing fewer than 4 covariates are not comparable.

| frame | n | events | sign agreement | coef matched | mean |coef error| |
|---|---|---|---|---|---|
| real train | 1762 | 610 | - | - | - |
| real holdout | 580 | 211 | - | - | - |
| mst_eps15_seed0 | 1749 | 574 | 2/5 | 6/6 | 0.3252 |
