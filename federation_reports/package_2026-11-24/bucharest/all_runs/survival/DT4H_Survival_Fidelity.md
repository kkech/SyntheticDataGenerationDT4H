# Survival Fidelity

Endpoints use the five-year follow-up columns: a recorded days-to-event is an event, a null is administrative censoring at 1825 days -- the same rule for real and synthetic data. A recorded time beyond 1825 days is treated as censoring at 1825 (out of horizon), counted per frame as `times_beyond_horizon`. The train-vs-holdout log-rank p-value calibrates what pure sampling noise looks like.

Disclosure: synthetic days-to-event values below the real observed minimum were nulled by the sentinel decode upstream and are read here as censoring; those erased early events cannot be recovered from the released CSVs (see `decode_note` in the JSON).

Effect-replication covariates are standardized in EVERY frame (real train, real holdout, synthetic) by the REAL TRAIN split's mean/SD, so scale infidelity in a synthetic frame shows up as a coefficient discrepancy instead of being re-normalized away.

