# RETIRED DRAFT — do not use

The canonical manuscript is **`main.tex`** (edited in Overleaf).

The Markdown draft that lived here predated the corrected evaluation and
contradicted the committed results (it claimed 0.0% of synthetic records
below the holdout p5 distance threshold in every run and no
membership-inference CI excluding 0.5 — both false against the corrected
privacy reports). It was retired rather than kept out of sync; recover it
from git history if ever needed.

All numbers in `main.tex` itself must be re-based on the corrected-methods
(v3) campaign outputs before any submission — the methodology fixes change
every table: matched-size noise floors, intersection-based C2ST,
decidable-rows coherence rules, corrected attacks, per-seed TOST, and the
subset-aware release gate.
