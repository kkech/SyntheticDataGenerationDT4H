# Release Gate: DT4H_Synthetic_aim50_eps1_seed0

Evaluated 2026-08-27T12:15:14.306911+00:00

**FAIL -- DO NOT RELEASE**

Policy: `release` -- coherence 10x the holdout baseline (or +1% absolute), distance 2x the natural 5% share. Intent: open or brokered release of a file that leaves the enclave.

| check | result | detail |
|---|---|---|
| schema | PASS | 88 columns, 0 not in released schema |
| representation | PASS | representation audit clean |
| freshness | PASS | 0 cell(s) an up-to-date decode would change |
| leakage | PASS | 0 verbatim training row(s) (compared in released/decoded space) |
| coherence | FAIL | violation rate 0.05282 vs holdout baseline 0.00024 (threshold 0.01024 = max(10x baseline, baseline+0.01)) |
| distance | PASS | 2/500 sampled record(s) (0.4%) closer than the holdout p5 threshold (0.049802, over 88 column(s), SUBSET of the full schema); policy limit 10% = 2x the natural 5% share |

Distance check computed over **88** column(s); p5 threshold 0.049802 from: holdout-vs-train p5 recomputed on the 88-column subset (candidate missing 161 of 249 columns). **The candidate is narrower than the full schema: the committed full-width p5 does not apply and the baseline was recomputed on the shared column subset.**

Reported, not thresholded: **11.0%** of this file's records carry at least one rule violation, against **0.8%** of the real holdout's. The coherence check above is per applicable rule-check, which is the smaller number; a release decision should be taken on both.

## Verdict under each policy

The same measurement, re-thresholded. Absolute checks (schema, representation, freshness, leakage) are policy-independent.

| policy | verdict | coherence limit | distance limit | intent |
|---|---|---|---|---|
| release | FAIL | 0.01024 ❌ | 0.1 ✅ | open or brokered release of a file that leaves the enclave |
| controlled | FAIL | 0.03024 ❌ | 0.1 ✅ | controlled-access sharing under a data-use agreement, where a recipient is bound by contract and the file is not public |
