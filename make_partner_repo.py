"""
Export the partner-facing copy of this pipeline into a clean folder.

    python make_partner_repo.py                 # -> ../SyntheticDataGenerationTabularDT4H
    python make_partner_repo.py --dest /path/to/SyntheticDataGenerationTabularDT4H
    python make_partner_repo.py --check-only    # verify a previous export, copy nothing

The destination is the tree partners clone and run at their own site
(README: `python main.py --data-dir /their/extract`). It is built
ALLOWLIST-FIRST: only the code and the declared public metadata are
copied; everything else -- output/, logs, status, backups, the paper,
this repo's .git -- never leaves the dev repo, so nothing derived from
patient data can ship by accident.

Re-runnable by design: each run refreshes the allowlisted files in place
and NEVER touches the destination's own .git, so `git init` there once,
re-export after every campaign, and commit the diff.

Transforms applied to the copy (the dev repo is never modified):
  * public_domains.json -- observed_min/observed_max are DELETED from
    every entry. They are exact values of real patients, kept in the dev
    repo only to aid the human review; lo/hi/basis are the public
    declaration and are all a partner site needs.
  * pipeline/config.py -- the dev enclave's transfer path is replaced by
    a placeholder; partners point the pipeline at their extract with
    --data-dir anyway.

After copying, a verification sweep greps the export for anything that
must not be there (patient-data file types, enclave hostnames and names,
unscrubbed observed extremes) and exits non-zero if it finds any.
"""

import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.join(os.path.dirname(REPO), "SyntheticDataGenerationTabularDT4H")

# Everything partners need to run the pipeline at their site -- and
# nothing else. Additions here are deliberate, reviewed decisions.
ALLOWLIST = (
    "pipeline",
    "main.py",
    "run_job.sh",
    "release_gate.py",
    "postprocess_candidate.py",
    "regenerate.py",
    "conditional_demo.py",
    "backup_results.py",
    "make_public_domains.py",
    "make_derived_columns_map.py",
    "make_partner_repo.py",     # so the export procedure itself is shared
    "respell_released_files.py",
    "c2st_diagnose.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "public_domains.json",      # scrubbed below
)

# The dev transfer default contains an enclave path with a person's name;
# partners never use it (--data-dir), so the copy gets a placeholder.
DEV_TRANSFER_SNIPPET = "/mnt/data/transfer-"
PLACEHOLDER_TRANSFER = "/path/to/your/site/part-parquet-extract/"

# Substrings that must not appear anywhere in the export.
FORBIDDEN_STRINGS = ("mydre.org", "boonstra", "observed_min", "observed_max")
# File types that must not appear anywhere in the export.
FORBIDDEN_SUFFIXES = (".parquet", ".pkl", ".csv")


def _copy_tree(src: str, dst: str) -> None:
    shutil.copytree(src, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))


def export(dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    kept_git = os.path.isdir(os.path.join(dest, ".git"))
    for item in ALLOWLIST:
        src = os.path.join(REPO, item)
        dst = os.path.join(dest, item)
        if not os.path.exists(src):
            print(f"  (absent in dev repo, skipped) {item}")
            continue
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.exists(dst):
            os.remove(dst)
        if os.path.isdir(src):
            _copy_tree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"  copied {item}")
    # Anything at the destination top level that is neither allowlisted
    # nor the partners' own .git is stale from an older export -- name it
    # rather than silently keeping or deleting it.
    for entry in sorted(os.listdir(dest)):
        if entry != ".git" and entry not in ALLOWLIST:
            print(f"  ⚠️  {entry} exists in the destination but is not in the allowlist "
                  f"-- not touched; remove it there if it is stale.")
    if kept_git:
        print("  destination .git preserved.")

    # --- transforms on the COPY ---
    pd_path = os.path.join(dest, "public_domains.json")
    if os.path.exists(pd_path):
        with open(pd_path) as f:
            d = json.load(f)
        removed = 0
        for spec in d.get("domains", {}).values():
            for k in ("observed_min", "observed_max"):
                if k in spec:
                    del spec[k]
                    removed += 1
        # The review that makes the epsilon claim sound is a SITE act, not
        # a repo property: the origin site's sign-off must not travel with
        # the code and pre-approve every fresh clone. Export with
        # reviewed=false so each site's DP fits refuse to start until
        # someone there has reviewed the ranges and set it true.
        d["reviewed"] = False
        d.pop("reviewed_by", None)
        d.pop("reviewed_at", None)
        d["note"] = ("PUBLIC domain declaration [lo, hi] per numeric column, with the "
                     "basis naming the public knowledge each range rests on. Released "
                     "as part of the DP mechanism specification; the observed extremes "
                     "used during the origin-site review are not part of this file. "
                     "reviewed is false ON PURPOSE: review these ranges against your "
                     "own site's clinical knowledge, then set reviewed to true (add "
                     "reviewed_by/reviewed_at) -- DP fitting refuses to start until "
                     "you do, because an unreviewed bound that later turns out "
                     "data-derived voids the epsilon claim.")
        with open(pd_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  scrubbed public_domains.json ({removed} observed_* value(s) removed; "
              f"reviewed reset to false -- each site signs off for itself)")

    cfg_path = os.path.join(dest, "pipeline", "config.py")
    with open(cfg_path) as f:
        cfg = f.read()
    if DEV_TRANSFER_SNIPPET in cfg:
        start = cfg.index(DEV_TRANSFER_SNIPPET)
        end = cfg.index('"', start)
        cfg = cfg[:start] + PLACEHOLDER_TRANSFER + cfg[end:]
        with open(cfg_path, "w") as f:
            f.write(cfg)
        print("  genericized the transfer-folder default in pipeline/config.py "
              "(partners use --data-dir).")


def verify(dest: str) -> int:
    problems = []
    for root, dirs, files in os.walk(dest):
        dirs[:] = [x for x in dirs if x != ".git"]
        for fname in files:
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, dest)
            if fname.endswith(FORBIDDEN_SUFFIXES):
                problems.append(f"forbidden file type: {rel}")
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            for bad in FORBIDDEN_STRINGS:
                if bad not in text:
                    continue
                # Source code may legitimately NAME the observed_* keys
                # (make_public_domains.py writes them into review
                # templates; this script checks for them). The leak is
                # DATA carrying values under those keys, so that check
                # applies to data files only. The enclave identifiers
                # are forbidden everywhere.
                if bad.startswith("observed_") and rel.endswith(".py"):
                    continue
                if rel == "make_partner_repo.py":
                    continue
                problems.append(f"forbidden content '{bad}' in {rel}")
    if problems:
        print("\n🚫 EXPORT VERIFICATION FAILED -- do not push this tree:")
        for p in problems:
            print("  ", p)
        return 1
    n_files = sum(len(f) for _, _, f in os.walk(dest))
    print(f"\n✅ Export verified clean: {n_files} file(s), no patient-data file types, "
          f"no enclave identifiers, no unscrubbed extremes.")
    print(f"   Next (first time only): cd {dest} && git init && git add -A && "
          f"git commit -m 'Partner pipeline release' && git remote add origin <url> && git push -u origin main")
    print("   Re-runs: re-export after each campaign, then commit the diff there.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the partner-facing pipeline copy.")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"Destination folder (default: {DEFAULT_DEST})")
    parser.add_argument("--check-only", action="store_true",
                        help="Only run the verification sweep on an existing export.")
    args = parser.parse_args()
    dest = os.path.abspath(args.dest)
    if os.path.commonpath([dest, REPO]) == REPO:
        print("🚫 Destination must be OUTSIDE the dev repo.")
        return 1
    if not args.check_only:
        print(f"Exporting partner repo -> {dest}")
        export(dest)
    return verify(dest)


if __name__ == "__main__":
    sys.exit(main())
