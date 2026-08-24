"""
Snapshot the pipeline's results before a destructive rerun.

    python backup_results.py                  # back up output/ + logs + status
    python backup_results.py --list           # show existing backups
    python backup_results.py --restore NAME --yes   # put a snapshot back

A full `--force` campaign WIPES output/, and some of what lives there is
gitignored and exists nowhere else: the released synthetic CSVs and the
fitted generator pickles (the evidence behind the current paper), plus
the train/holdout parquets. This script copies the whole tree into
backups/<UTC-timestamp>_<git-commit>/ at the repo root -- gitignored,
because the snapshot contains those same never-committed files -- with a
manifest recording what was saved and from which commit.

Run it BEFORE `./run_job.sh start ... --force`. It refuses to run while
a backup would be incomplete (insufficient disk) and never overwrites an
existing snapshot.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
BACKUP_ROOT = os.path.join(REPO, "backups")
EXTRAS = ("logs.txt", "pipeline_status.json")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _tree_size(path: str):
    total = count = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
                count += 1
            except OSError:
                pass
    return total, count


def do_backup(output_dir: str) -> int:
    if not os.path.isdir(output_dir):
        print(f"Nothing to back up: {output_dir} does not exist.")
        return 1

    size, count = _tree_size(output_dir)
    free = shutil.disk_usage(BACKUP_ROOT if os.path.isdir(BACKUP_ROOT) else REPO).free
    # copy + 10% headroom; refuse rather than die halfway
    if free < size * 1.1:
        print(f"🚫 Not enough disk for a complete backup: need ~{size/1e9:.2f} GB "
              f"(+10% headroom), {free/1e9:.2f} GB free. Free space first -- an "
              f"incomplete backup is worse than none.")
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{_git_commit()}"
    dest = os.path.join(BACKUP_ROOT, name)
    if os.path.exists(dest):
        print(f"🚫 {dest} already exists -- refusing to overwrite a snapshot.")
        return 1
    os.makedirs(BACKUP_ROOT, exist_ok=True)

    print(f"Backing up {count} file(s), {size/1e9:.2f} GB -> {dest}")
    shutil.copytree(output_dir, os.path.join(dest, "output"))
    for extra in EXTRAS:
        src = os.path.join(REPO, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, extra))

    copied_size, copied_count = _tree_size(dest)
    manifest = {
        "created_utc": stamp,
        "git_commit": _git_commit(),
        "source_output_dir": output_dir,
        "files": copied_count,
        "bytes": copied_size,
        "note": "Contains gitignored patient-derived artifacts (synthetic CSVs, "
                "fitted models, split parquets). Never commit; never leave the "
                "secure environment.",
    }
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Snapshot complete: {copied_count} file(s), {copied_size/1e9:.2f} GB.")
    print(f"   Restore later with: python backup_results.py --restore {name} --yes")
    return 0


def do_list() -> int:
    if not os.path.isdir(BACKUP_ROOT):
        print("No backups yet.")
        return 0
    rows = sorted(os.listdir(BACKUP_ROOT))
    if not rows:
        print("No backups yet.")
        return 0
    for name in rows:
        mpath = os.path.join(BACKUP_ROOT, name, "manifest.json")
        if os.path.exists(mpath):
            m = json.load(open(mpath))
            print(f"  {name}: {m['files']} files, {m['bytes']/1e9:.2f} GB, "
                  f"commit {m['git_commit']}")
        else:
            print(f"  {name}: (no manifest -- possibly incomplete)")
    return 0


def do_restore(name: str, output_dir: str, yes: bool) -> int:
    src = os.path.join(BACKUP_ROOT, name)
    if not os.path.isdir(os.path.join(src, "output")):
        print(f"🚫 No snapshot named {name!r} (see --list).")
        return 1
    if not os.path.exists(os.path.join(src, "manifest.json")):
        print(f"🚫 {name} has no manifest -- it may be an incomplete backup; "
              f"refusing to restore it.")
        return 1
    if not yes:
        print(f"Restoring would REPLACE the current {output_dir} with snapshot "
              f"{name}. Re-run with --yes to confirm.")
        return 1
    if os.path.isdir(output_dir):
        holding = output_dir.rstrip("/") + ".pre-restore"
        if os.path.exists(holding):
            shutil.rmtree(holding)
        os.rename(output_dir, holding)
        print(f"Current results moved aside -> {holding} (delete manually once satisfied).")
    shutil.copytree(os.path.join(src, "output"), output_dir)
    for extra in EXTRAS:
        s = os.path.join(src, extra)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(REPO, extra))
    print(f"✅ Restored snapshot {name} -> {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up / restore pipeline results.")
    parser.add_argument("--list", action="store_true", help="List existing backups.")
    parser.add_argument("--restore", metavar="NAME", help="Restore a snapshot by name.")
    parser.add_argument("--yes", action="store_true", help="Confirm a restore.")
    parser.add_argument("--output-dir", default=os.path.join(REPO, "output"),
                        help="Results tree to back up / restore to (default: output/).")
    args = parser.parse_args()

    if args.list:
        return do_list()
    if args.restore:
        return do_restore(args.restore, args.output_dir, args.yes)
    return do_backup(args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
