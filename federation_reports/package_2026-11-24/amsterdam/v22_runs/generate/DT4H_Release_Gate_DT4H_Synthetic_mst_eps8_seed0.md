# Release Gate: DT4H_Synthetic_mst_eps8_seed0

Evaluated 2026-11-24T09:44:58.249978+00:00

**FAIL -- DO NOT RELEASE**

Policy: `release` -- coherence 10x the holdout baseline (or +1% absolute), distance 2x the natural 5% share. Intent: open or brokered release of a file that leaves the enclave.

| check | result | detail |
|---|---|---|
| schema | PASS | 254 columns, 0 not in released schema |
| representation | PASS | representation audit clean |
| freshness | PASS | 0 cell(s) an up-to-date decode would change |
| leakage | PASS | 0 verbatim training row(s) (compared in released/decoded space) |
| coherence | FAIL | violation rate 0.04184 vs holdout baseline 0.00037 (threshold 0.01037 = max(10x baseline, baseline+0.01)) |
| distance | FAIL | 918/3334 record(s) (27.5%, privacy step (all rows, same encoder)) closer than the holdout p5 threshold (0.070828, over 254 column(s)); policy limit 10% = 2x the natural 5% share |

Distance check computed over **254** column(s); p5 threshold 0.070828 from: committed full-width holdout p5 (privacy step).

Reported, not thresholded: **23.7%** of this file's records carry at least one rule violation, against **0.4%** of the real holdout's. The coherence check above is per applicable rule-check, which is the smaller number; a release decision should be taken on both.

## Verdict under each policy

The same measurement, re-thresholded. Absolute checks (schema, representation, freshness, leakage) are policy-independent.

| policy | verdict | coherence limit | distance limit | intent |
|---|---|---|---|---|
| release | FAIL | 0.01037 ❌ | 0.1 ❌ | open or brokered release of a file that leaves the enclave |
| controlled | FAIL | 0.03037 ❌ | 0.1 ❌ | controlled-access sharing under a data-use agreement, where a recipient is bound by contract and the file is not public |
