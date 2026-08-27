"""
Emit the semantic mapping for the pipeline's derived med_*/conditions_*
columns.

    python make_derived_columns_map.py                       # default paths
    python make_derived_columns_map.py --metadata /path/to/metadata.json

The released datasets contain ~85 columns that are NOT in the UC1
feature-set metadata.json, because preprocessing derives them: the
medication indicator variants (admins/requests, current/history) and the
condition window variants (pre_dc / pre_adm / during_pET) are OR-combined
into one presence flag each, and the source indicators are then DROPPED
from the modelling frame. So these columns cannot be skipped without
losing all medication/condition content -- and any consumer that needs
column semantics (e.g. text rendering of a patient record) needs a map
from each derived column back to its sources and their meanings.

This script writes that map, machine-readable and human-readable
(DT4H_Derived_Columns.json / .md), from metadata.json alone -- no patient
data is read, so the outputs are releasable metadata. The grouping rules
below mirror combine_medications / combine_conditions in
pipeline/steps/preprocess/transforms.py; if those change, change this
file in the same commit.
"""

import argparse
import json
import os
import re
import sys

DEFAULT_METADATA = os.path.join("output", "profile_data", "metadata.json")
DEFAULT_OUT_DIR = os.path.join("output", "release_docs")

MED_RULE = ("True if the medication is flagged by ANY source indicator "
            "(administered or requested); null in a source is treated as "
            "not-flagged, not unknown.")
COND_RULE = ("True if the condition is flagged in ANY of the three "
             "observation windows (before discharge, before admission, "
             "during the encounter); null in a source is treated as "
             "not-flagged, not unknown.")


def _strip_any_suffix(name: str) -> str:
    return name[: -len("_any")] if name.endswith("_any") else name


def load_feature_descriptions(metadata_path: str) -> dict:
    with open(metadata_path) as f:
        meta = json.load(f)
    entries = meta.get("entries") or []
    if not entries:
        sys.exit(f"🚫 {metadata_path} has no entries[] -- is this the UC1 feature-set metadata?")
    desc = {}
    for entry in entries:
        for var in (entry.get("baseVariables") or []) + (entry.get("features") or []):
            name = var.get("name")
            if name:
                desc[name] = var.get("description", "")
    return desc


def medication_groups(columns) -> dict:
    """Mirror of combine_medications: {derived_name: [source columns]}."""
    admin_pat = re.compile(r"^med_admins_(?!history_)(.+)$")
    admin_hist_pat = re.compile(r"^med_admins_history_(.+)$")
    cols = set(columns)
    groups = {}
    for med in sorted({m.group(1) for c in cols if (m := admin_pat.match(c))}):
        sources = [c for c in (f"med_admins_{med}", f"med_requests_{med}") if c in cols]
        if sources:
            groups[f"med_{_strip_any_suffix(med)}"] = sources
    for med in sorted({m.group(1) for c in cols if (m := admin_hist_pat.match(c))}):
        sources = [c for c in (f"med_admins_history_{med}", f"med_requests_history_{med}") if c in cols]
        if sources:
            groups[f"med_{_strip_any_suffix(med)}_history"] = sources
    return groups


def condition_groups(columns) -> dict:
    """Mirror of combine_conditions: {derived_name: [source columns]}."""
    suffixes = ("_pre_dc_any", "_pre_adm_any", "_during_pET_any")
    pat = re.compile(r"^conditions_(.+?)(?:_pre_dc_any|_pre_adm_any|_during_pET_any)$")
    cols = set(columns)
    groups = {}
    for base in sorted({m.group(1) for c in cols if (m := pat.match(c))}):
        sources = [f"conditions_{base}{suf}" for suf in suffixes if f"conditions_{base}{suf}" in cols]
        if sources:
            groups[f"conditions_{base}"] = sources
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the derived-columns semantic map.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA,
                        help=f"UC1 feature-set metadata.json (default: {DEFAULT_METADATA})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        sys.exit(f"🚫 metadata not found: {args.metadata} (pass --metadata)")
    descriptions = load_feature_descriptions(args.metadata)
    declared = list(descriptions)

    med = medication_groups(declared)
    cond = condition_groups(declared)

    def entry(rule, sources):
        return {
            "derivation": "any_true_of_sources",
            "rule": rule,
            "null_handling": "source nulls treated as False (presence flags)",
            "sources": {s: descriptions.get(s, "") for s in sources},
        }

    result = {
        "note": ("Columns derived by preprocessing from the UC1 feature-set "
                 "metadata; the source indicator columns are dropped from the "
                 "modelling frame after combining, so these derived columns "
                 "CARRY the medication/condition content and must not be "
                 "skipped. Generated from metadata.json only -- no patient "
                 "data. Rules mirror pipeline/steps/preprocess/transforms.py."),
        "derived_columns": {
            **{name: entry(MED_RULE, srcs) for name, srcs in med.items()},
            **{name: entry(COND_RULE, srcs) for name, srcs in cond.items()},
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "DT4H_Derived_Columns.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    md_lines = [
        "# Derived columns (med_* / conditions_*)",
        "",
        result["note"],
        "",
        "| derived column | combined as | source columns and their meanings |",
        "|---|---|---|",
    ]
    for name, info in result["derived_columns"].items():
        srcs = "<br>".join(f"`{s}` -- {d}" for s, d in info["sources"].items())
        md_lines.append(f"| `{name}` | any true | {srcs} |")
    md_path = os.path.join(args.out_dir, "DT4H_Derived_Columns.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    n_med = len(med)
    n_cond = len(cond)
    print(f"✅ {n_med} med_* + {n_cond} conditions_* = {n_med + n_cond} derived "
          f"column(s) mapped from {len(declared)} declared variables.")
    print(f"   {json_path}\n   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
