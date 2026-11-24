# Survival Fidelity

Endpoints use the five-year follow-up columns: a recorded days-to-event is an event, a null is administrative censoring at 1825 days -- the same rule for real and synthetic data. A recorded time beyond 1825 days is treated as censoring at 1825 (out of horizon), counted per frame as `times_beyond_horizon`. The train-vs-holdout log-rank p-value calibrates what pure sampling noise looks like.

Disclosure: synthetic days-to-event values below the real observed minimum were nulled by the sentinel decode upstream and are read here as censoring; those erased early events cannot be recovered from the released CSVs (see `decode_note` in the JSON).

Effect-replication covariates are standardized in EVERY frame (real train, real holdout, synthetic) by the REAL TRAIN split's mean/SD, so scale infidelity in a synthetic frame shows up as a coefficient discrepancy instead of being re-normalized away.

## hf_rehospitalization

train events 138/602 | holdout 57/200 | log-rank train-vs-holdout p = 0.1126

| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |
|---|---|---|---|---|
| **train** | 0.77243 | 0.77076 | - | no |
| **holdout** | 0.725 | 0.715 |  |  |
| mst_eps15_seed0 | 0.77409 | 0.7691 | 0.0028 | no |

Equivalence is a POSITIVE claim (90% CI of the survival difference within ±5pp at every horizon) -- unlike a non-significant log-rank, which is only absence of evidence.

