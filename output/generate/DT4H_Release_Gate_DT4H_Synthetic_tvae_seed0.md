# Release Gate: DT4H_Synthetic_tvae_seed0

Evaluated 2026-08-23T11:57:54.029276+00:00

**PASS -- cleared for release**

| check | result | detail |
|---|---|---|
| schema | PASS | 249 columns, 0 not in released schema |
| freshness | PASS | 0 cell(s) an up-to-date decode would change |
| leakage | PASS | 0 verbatim training row(s) (compared in released/decoded space) |
| coherence | PASS | violation rate 0.02252 vs holdout baseline 0.00232 (threshold 0.0232 = max(10x baseline, baseline+0.01)) |
| distance | PASS | 0/500 sampled record(s) closer than the holdout p5 threshold (0.062785) |
