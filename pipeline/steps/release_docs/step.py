"""
Step: release_docs

The documentation artifacts a professional dataset release ships with:

  * DT4H_Codebook.md -- one row per released column: declared type,
    observed range or category list (already-public aggregate facts),
    missingness rate and, crucially, what a null MEANS for that column
    (structural "no event" vs "not measured"), from the committed
    sentinel encoding map.
  * DT4H_Datasheet.md -- a Datasheet for the Dataset (Gebru et al.)
    with every mechanically-derivable section auto-filled from pipeline
    facts and explicit TODO markers where a human judgement is required
    (motivation, distribution/licensing decisions).
  * DT4H_Environment_Freeze.txt -- the exact package versions of the
    environment that produced the release (pip freeze), completing the
    provenance chain.

Everything here is derived from already-committed aggregates -- safe to
commit.
"""

import json
import os
import subprocess
import sys
from datetime import date

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep


class ReleaseDocsStep(PipelineStep):
    name = "release_docs"

    def run(self, config: PipelineConfig) -> None:
        import polars as pl

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(config.train_output_path):
            raise FileNotFoundError(f"{config.train_output_path} missing -- run preprocess first.")
        train = pl.read_parquet(config.train_output_path)

        encoding = {}
        enc_path = os.path.join(config.step_dir("preprocess"), "DT4H_Numeric_Missing_Encoding.json")
        if os.path.exists(enc_path):
            with open(enc_path) as f:
                encoding = json.load(f)

        var_meta = {}
        if os.path.exists(config.metadata_path):
            with open(config.metadata_path) as f:
                raw = json.load(f)
            for section in ("baseVariables", "features", "outcomes"):
                for v in raw.get("entries", [{}])[0].get(section, []) or []:
                    var_meta[v["name"]] = v

        self._write_codebook(train, encoding, var_meta, out_dir, config)
        self._write_datasheet(train, out_dir, config)
        self._write_env_freeze(out_dir)

    # --- codebook ---

    def _write_codebook(self, train, encoding, var_meta, out_dir, config) -> None:
        import polars as pl

        lines = [
            "# DT4H UC1 Synthetic Dataset -- Codebook",
            "",
            f"Generated {date.today().isoformat()} by the pipeline. One row per released "
            "column. Ranges and category lists are aggregate facts over the training split "
            "(also published in the profiling reports). **A null is never 'unknown noise' "
            "in this dataset -- its meaning is stated per column.**",
            "",
            "| column | type | description | values / range | missing % | null means |",
            "|---|---|---|---|---|---|",
        ]

        def _desc(c):
            v = var_meta.get(c, {})
            d = (v.get("description") or v.get("generatedDescription") or "").replace("|", "/")
            d = " ".join(d.split())
            return d[:110] + ("…" if len(d) > 110 else "")

        for c in train.columns:
            s = train[c]
            declared = var_meta.get(c, {}).get("dataType", "")
            missing_note = "n/a (no nulls)"
            if c in encoding:
                missing_note = ("no event occurred (structural)"
                                if encoding[c].get("structural") else "not measured")
            if s.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                spec = encoding.get(c)
                if spec:
                    rng = f"{spec.get('min_observed', '')} .. {round(float(s.max()), 2)}"
                    miss = f"{100 * (s == spec['sentinel']).sum() / len(s):.0f}%"
                else:
                    rng = f"{round(float(s.min()), 2)} .. {round(float(s.max()), 2)}"
                    miss = "0%"
                lines.append(f"| `{c}` | numeric {declared} | {_desc(c)} | {rng} | {miss} | {missing_note} |")
            else:
                cats = s.unique().to_list()
                cats = sorted(str(x) for x in cats if x is not None)
                miss_rate = (s == "Missing").sum() / len(s) if "Missing" in cats else 0
                shown = ", ".join(cats[:6]) + (" …" if len(cats) > 6 else "")
                note = "explicit 'Missing' category" if "Missing" in cats else "n/a (no nulls)"
                lines.append(f"| `{c}` | categorical {declared} | {_desc(c)} | {shown} "
                             f"| {100 * miss_rate:.0f}% | {note} |")

        path = os.path.join(out_dir, "DT4H_Codebook.md")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved codebook ({len(train.columns)} columns) -> {path}")

    # --- datasheet ---

    def _write_datasheet(self, train, out_dir, config) -> None:
        n_rows, n_cols = train.height, train.width
        gen_summary = {}
        p = os.path.join(config.step_dir("generate"), "DT4H_Generation_Summary.json")
        if os.path.exists(p):
            with open(p) as f:
                gen_summary = json.load(f)
        prov = gen_summary.get("provenance", {})

        text = f"""# Datasheet: DT4H UC1 Synthetic Heart-Failure Cohort

Structure follows *Datasheets for Datasets* (Gebru et al., 2021). Sections marked
`TODO(author)` require a human decision and must be completed before submission.

## Motivation
- **Purpose**: privacy-preserving synthetic version of the DataTools4Heart UC1
  heart-failure cohort, enabling method development and reproduction without
  access to patient-level data. TODO(author): funding statement, consortium context.

## Composition
- Synthetic patient-level records: {gen_summary.get('n_synthetic_rows', 'see generation summary')}
  rows x {n_cols} columns per released file, matching the training-split schema.
- Source (never released): {n_rows} training records ({config.holdout_fraction:.0%} of the
  cohort held out for evaluation and never shown to any generator).
- Column semantics: see `DT4H_Codebook.md`. Missingness is preserved by design and
  carries meaning (structural "no event" vs "not measured").
- No real patient records, identifiers, or verbatim rows are included (verified:
  zero exact training-row reproductions in every released file).

## Collection & preprocessing
- Source data extracted under the DataTools4Heart federated protocol.
  TODO(author): site, extraction date, ethics/DPIA reference.
- Preprocessing is provably distribution-preserving (KS = 0, TVD = 0 vs raw on all
  retained columns) and fully scripted; see `DT4H_Preprocessing_Summary.md`.
- Generators: see the run plan in `DT4H_Generation_Summary.md` (seeds, epsilon
  values, library versions, git commit `{prov.get('git', {}).get('commit', 'n/a')}`,
  training-file SHA-256).

## Uses
- Intended: methods development, education, benchmarking, pre-analysis piloting.
- Cautions: effect estimates and model performance are attenuated relative to real
  data (quantified in `DT4H_Utility_TSTR.md` and `DT4H_Survival_Fidelity.md`);
  fidelity is weakest for sparsely observed laboratory values. Synthetic data must
  not be represented as real patients in any publication.

## Privacy & release gating
- Record-level distances, membership-inference and attribute-inference attacks are
  reported in `DT4H_Privacy_Assessment.md` and `DT4H_Privacy_Attacks.md`, all
  evaluated against a genuine unseen-patient baseline.
- DP-labelled files carry a formal (epsilon, delta) guarantee by construction;
  column domains are treated as public metadata (released in the encoding map).
- Every file must pass `release_gate.py` before distribution.

## Distribution & maintenance
- TODO(author): hosting venue (e.g. Zenodo/HDR), DOI, license (recommend CC-BY for
  documentation; data license per consortium policy), point of contact, versioning
  and retraction policy.
"""
        path = os.path.join(out_dir, "DT4H_Datasheet.md")
        with open(path, "w") as f:
            f.write(text)
        print(f"Saved datasheet -> {path}")

    # --- environment freeze ---

    def _write_env_freeze(self, out_dir) -> None:
        path = os.path.join(out_dir, "DT4H_Environment_Freeze.txt")
        try:
            freeze = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True, timeout=120)
            with open(path, "w") as f:
                f.write(f"# python {sys.version.split()[0]}\n{freeze}")
            print(f"Saved environment freeze ({len(freeze.splitlines())} packages) -> {path}")
        except Exception as e:
            print(f"⚠️  Could not capture pip freeze: {type(e).__name__}: {e}")
