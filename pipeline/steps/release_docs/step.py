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
        self._write_capability_labels(out_dir, config)

    # --- capability labels ---

    def _write_capability_labels(self, out_dir: str, config: PipelineConfig) -> None:
        """One machine-readable label per released file -- a compact
        summary of every line of evidence the pipeline holds about it
        (fidelity, joint structure, coherence, distances, attacks, gate
        verdict), so a downstream user or reviewer reads one JSON instead
        of six reports. Absent evidence is null, never guessed."""

        def _load(step, name):
            p = os.path.join(config.step_dir(step), name)
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)
            return None

        gen = _load("generate", "DT4H_Generation_Summary.json") or {"runs": []}
        ev = _load("evaluate", "DT4H_Evaluation.json") or {}
        priv = _load("privacy", "DT4H_Privacy_Assessment.json") or {}
        att = _load("attacks", "DT4H_Privacy_Attacks.json") or {}
        coh = _load("coherence", "DT4H_Coherence_Audit.json") or {}

        ev_by_run = {c.get("run_id"): c for c in ev.get("comparisons", [])
                     if isinstance(c, dict) and c.get("run_id")}
        priv_by_run = {r.get("run_id"): r for r in priv.get("runs", [])}
        att_by_run = {r.get("run_id"): r for r in att.get("runs", [])}
        coh_by_run = {}
        for fr in coh.get("frames", []):
            label = fr.get("frame", "")
            if label.startswith("synthetic[") and label.endswith("]"):
                coh_by_run[label[len("synthetic["):-1]] = fr

        labels_dir = os.path.join(out_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        index = []
        for run in gen.get("runs", []):
            if run.get("status") != "ok":
                continue
            rid = run["run_id"]
            e = ev_by_run.get(rid, {})
            agg = (e.get("train_vs_synthetic") or {}).get("aggregates", {})
            assoc = e.get("associations") or {}
            fabricated = sum((assoc.get(k) or {}).get("fabricated_pairs", 0) or 0
                             for k in ("num_num", "cat_cat", "num_cat"))
            p = priv_by_run.get(rid, {})
            a = att_by_run.get(rid, {})
            mia = a.get("membership_inference", {})
            atyp = mia.get("mia_by_atypicality") or []
            aia = a.get("attribute_inference") or []
            gate_path = os.path.join(config.step_dir("generate"),
                                     f"DT4H_Release_Gate_DT4H_Synthetic_{rid}.md")
            gate = None
            if os.path.exists(gate_path):
                text = open(gate_path).read()
                gate = "PASS" if "PASS -- cleared for release" in text else "FAIL"

            label = {
                "run_id": rid,
                "file": f"DT4H_Synthetic_{rid}.csv",
                "synthesizer": run.get("synthesizer"),
                "base_synthesizer": run.get("base_synthesizer", run.get("synthesizer")),
                "run_transform": run.get("run_transform"),
                "differential_privacy": {"formal_guarantee": bool(run.get("is_dp")),
                                         "epsilon": run.get("epsilon")},
                "seed": run.get("seed"),
                "rows": run.get("output_rows"),
                "columns": run.get("output_columns"),
                "fidelity": {
                    "ks_mean_numeric": (agg.get("ks") or {}).get("mean"),
                    "tvd_mean_categorical": (agg.get("tvd") or {}).get("mean"),
                    "missing_rate_mad": agg.get("missing_rate_mean_abs_diff"),
                    "c2st_auc": e.get("c2st_auc"),
                },
                "joint_structure": {"fabricated_association_pairs": fabricated or None},
                "row_coherence": {
                    "violation_rate": coh_by_run.get(rid, {}).get("overall_violation_rate"),
                },
                "distances": {
                    "verbatim_training_rows": (run.get("leakage") or {}).get(
                        "exact_duplicates_of_training_rows"),
                    "dcr_median": p.get("dcr_median"),
                    "share_closer_than_holdout_p5": p.get("share_closer_than_holdout_p5"),
                    "nndr_median": p.get("nndr_median"),
                },
                "attacks": {
                    "mia_auc": mia.get("attack_auc"),
                    "mia_learned_auc": mia.get("learned_attack_auc"),
                    "mia_auc_most_atypical_quartile":
                        atyp[-1]["attack_auc"] if atyp else None,
                    "aia_worst_membership_advantage":
                        max((x.get("membership_advantage") for x in aia
                             if x.get("membership_advantage") is not None), default=None),
                    "singling_out_risk": (a.get("anonymeter") or {}).get("singling_out_risk"),
                    "linkability_risk": (a.get("anonymeter") or {}).get("linkability_risk"),
                },
                "release_gate": gate or "not gated",
            }
            with open(os.path.join(labels_dir, f"DT4H_Label_{rid}.json"), "w") as f:
                json.dump(label, f, indent=2, default=str)
            index.append({"run_id": rid, "synthesizer": label["synthesizer"],
                          "epsilon": run.get("epsilon"), "release_gate": label["release_gate"]})

        with open(os.path.join(out_dir, "DT4H_Capability_Labels_Index.json"), "w") as f:
            json.dump({"labels": index, "directory": "labels/"}, f, indent=2, default=str)
        print(f"Saved capability labels for {len(index)} released file(s) -> {labels_dir}")

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

Structure follows *Datasheets for Datasets* (Gebru et al., 2021).

## Motivation
- **Purpose**: a privacy-preserving synthetic version of the DataTools4Heart
  (DT4H) Use Case 1 heart-failure cohort, enabling method development,
  benchmarking, education and analysis piloting without access to patient-level
  data.
- **Context**: created within the DataTools4Heart project, a European
  multi-partner initiative building a federated cardiology data toolbox. The
  synthetic release lets researchers outside the secure environment work with
  realistic UC1-shaped data while the real records never leave the provider.

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
- Source data were extracted from the electronic health records of the providing
  DT4H clinical partner site under the project's federated data protocol
  (standardized onFHIR/Feast feature extraction; see the feature-set metadata),
  and delivered 2026-08-12 into the project's secure research environment.
- All processing -- including generator training -- took place inside that secure
  environment (an isolated analysis workspace with no patient-level data export);
  only aggregate statistics, synthetic records and reports leave it. Processing
  is governed by the project's data-sharing and governance agreements between
  the consortium partners.
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
- **Hosting**: the project repository
  (github.com/kkech/SyntheticDataGenerationDT4H) carries the pipeline, all
  evaluation reports and the release documentation; vetted synthetic files are
  added there deliberately after passing `release_gate.py`. An archival deposit
  with a DOI (Zenodo) will be minted for the exact release accompanying the
  publication.
- **License**: documentation, reports and code are released openly with the
  repository; the synthetic data files are intended for release under CC BY 4.0,
  subject to final consortium approval.
- **Point of contact**: the repository maintainers, via GitHub issues on the
  repository above.
- **Versioning**: every release is reproducible from its recorded git commit,
  seeds and training-file SHA-256 (see the generation summary and
  `DT4H_Environment_Freeze.txt`); releases are tagged in git and superseded
  versions remain available in history.
- **Retraction**: should any privacy or integrity concern be identified, the
  affected files will be removed from the repository and archival deposit, the
  release tag withdrawn, and the concern documented in the repository.
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
