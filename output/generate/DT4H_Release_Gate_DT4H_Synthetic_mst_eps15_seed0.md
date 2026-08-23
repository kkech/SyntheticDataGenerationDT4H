# Release Gate: DT4H_Synthetic_mst_eps15_seed0

Evaluated 2026-08-23T14:53:58.535813+00:00

**FAIL -- DO NOT RELEASE**

| check | result | detail |
|---|---|---|
| schema | PASS | 249 columns, 0 not in released schema |
| representation | PASS | representation audit clean |
| freshness | PASS | 0 cell(s) an up-to-date decode would change |
| leakage | PASS | 0 verbatim training row(s) (compared in released/decoded space) |
| coherence | FAIL | violation rate 0.03632 vs holdout baseline 0.00232 (threshold 0.0232 = max(10x baseline, baseline+0.01)) |
| distance | FAIL | 88/500 sampled record(s) (17.6%) closer than the holdout p5 threshold (0.062785); policy limit 10% = 2x the natural 5% share |
