# Release Gate: DT4H_Synthetic_tvae_seed0

Evaluated 2026-11-23T10:16:24.715616+00:00

**PASS -- cleared for release**

Policy: `release` -- coherence 10x the holdout baseline (or +1% absolute), distance 2x the natural 5% share. Intent: open or brokered release of a file that leaves the enclave.

| check | result | detail |
|---|---|---|
| schema | PASS | 229 columns, 0 not in released schema |
| representation | PASS | representation audit clean |
| freshness | PASS | 0 cell(s) an up-to-date decode would change |
| leakage | PASS | 0 verbatim training row(s) (compared in released/decoded space) |
| coherence | PASS | violation rate 0.00481 vs holdout baseline 0.0 (threshold 0.01 = max(10x baseline, baseline+0.01)) |
| distance | PASS | 352/4706 record(s) (7.5%, privacy step (all rows, same encoder)) closer than the holdout p5 threshold (0.025555, over 229 column(s)); policy limit 10% = 2x the natural 5% share |

Distance check computed over **229** column(s); p5 threshold 0.025555 from: committed full-width holdout p5 (privacy step).

Reported, not thresholded: **1.1%** of this file's records carry at least one rule violation, against **0.0%** of the real holdout's. The coherence check above is per applicable rule-check, which is the smaller number; a release decision should be taken on both.

## Verdict under each policy

The same measurement, re-thresholded. Absolute checks (schema, representation, freshness, leakage) are policy-independent.

| policy | verdict | coherence limit | distance limit | intent |
|---|---|---|---|---|
| release | PASS | 0.01 ✅ | 0.1 ✅ | open or brokered release of a file that leaves the enclave |
| controlled | PASS | 0.03 ✅ | 0.1 ✅ | controlled-access sharing under a data-use agreement, where a recipient is bound by contract and the file is not public |
