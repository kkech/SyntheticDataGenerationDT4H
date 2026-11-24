# Release Gate: DT4H_Synthetic_aim40_eps8_seed0

Evaluated 2026-11-24T09:42:52.094643+00:00

**PASS -- cleared for release**

Policy: `release` -- coherence 10x the holdout baseline (or +1% absolute), distance 2x the natural 5% share. Intent: open or brokered release of a file that leaves the enclave.

| check | result | detail |
|---|---|---|
| schema | PASS | 77 columns, 0 not in released schema |
| representation | PASS | representation audit clean |
| freshness | PASS | 0 cell(s) an up-to-date decode would change |
| leakage | PASS | 0 verbatim training row(s) (compared in released/decoded space) |
| coherence | PASS | violation rate 0.00868 vs holdout baseline 0.00037 (threshold 0.01037 = max(10x baseline, baseline+0.01)) |
| distance | PASS | 4/500 record(s) (0.8%, spot check (500 sampled rows)) closer than the holdout p5 threshold (0.039318, over 77 column(s), SUBSET of the full schema); policy limit 10% = 2x the natural 5% share |

Distance check computed over **77** column(s); p5 threshold 0.039318 from: holdout-vs-train p5 recomputed on the 77-column subset (candidate missing 177 of 254 columns). **The candidate is narrower than the full schema: the committed full-width p5 does not apply and the baseline was recomputed on the shared column subset.**

Reported, not thresholded: **2.2%** of this file's records carry at least one rule violation, against **0.4%** of the real holdout's. The coherence check above is per applicable rule-check, which is the smaller number; a release decision should be taken on both.

## Verdict under each policy

The same measurement, re-thresholded. Absolute checks (schema, representation, freshness, leakage) are policy-independent.

| policy | verdict | coherence limit | distance limit | intent |
|---|---|---|---|---|
| release | PASS | 0.01037 ✅ | 0.1 ✅ | open or brokered release of a file that leaves the enclave |
| controlled | PASS | 0.03037 ✅ | 0.1 ✅ | controlled-access sharing under a data-use agreement, where a recipient is bound by contract and the file is not public |
