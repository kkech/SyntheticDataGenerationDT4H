# Release Gate: DT4H_Synthetic_mst_eps15_seed0

Evaluated 2026-11-23T11:00:29.515125+00:00

**FAIL -- DO NOT RELEASE**

Policy: `release` -- coherence 10x the holdout baseline (or +1% absolute), distance 2x the natural 5% share. Intent: open or brokered release of a file that leaves the enclave.

| check | result | detail |
|---|---|---|
| schema | PASS | 257 columns, 0 not in released schema |
| representation | PASS | representation audit clean |
| freshness | PASS | 0 cell(s) an up-to-date decode would change |
| leakage | PASS | 0 verbatim training row(s) (compared in released/decoded space) |
| coherence | FAIL | violation rate 0.02879 vs holdout baseline 0.00029 (threshold 0.01029 = max(10x baseline, baseline+0.01)) |
| distance | PASS | 0/754 record(s) (0.0%, privacy step (all rows, same encoder)) closer than the holdout p5 threshold (0.048053, over 257 column(s)); policy limit 10% = 2x the natural 5% share |

Distance check computed over **257** column(s); p5 threshold 0.048053 from: committed full-width holdout p5 (privacy step).

Reported, not thresholded: **47.5%** of this file's records carry at least one rule violation, against **1.2%** of the real holdout's. The coherence check above is per applicable rule-check, which is the smaller number; a release decision should be taken on both.

## Verdict under each policy

The same measurement, re-thresholded. Absolute checks (schema, representation, freshness, leakage) are policy-independent.

| policy | verdict | coherence limit | distance limit | intent |
|---|---|---|---|---|
| release | FAIL | 0.01029 ❌ | 0.1 ✅ | open or brokered release of a file that leaves the enclave |
| controlled | PASS | 0.03029 ✅ | 0.1 ✅ | controlled-access sharing under a data-use agreement, where a recipient is bound by contract and the file is not public |
