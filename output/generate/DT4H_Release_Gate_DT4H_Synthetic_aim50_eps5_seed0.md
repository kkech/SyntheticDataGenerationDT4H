# Release Gate: DT4H_Synthetic_aim50_eps5_seed0

Evaluated 2026-08-23T15:04:37.320497+00:00

**PASS -- cleared for release**

| check | result | detail |
|---|---|---|
| schema | PASS | 88 columns, 0 not in released schema |
| representation | PASS | representation audit clean |
| freshness | PASS | 0 cell(s) an up-to-date decode would change |
| leakage | PASS | 0 verbatim training row(s) (compared in released/decoded space) |
| coherence | PASS | violation rate 0.00907 vs holdout baseline 0.00232 (threshold 0.0232 = max(10x baseline, baseline+0.01)) |
| distance | PASS | 0/500 sampled record(s) (0.0%) closer than the holdout p5 threshold (0.062785); policy limit 10% = 2x the natural 5% share |
