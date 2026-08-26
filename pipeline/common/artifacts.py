"""Shared helpers for writing the shareable pipeline artifacts under output/<step>/."""

import os
import shutil

import polars as pl

METADATA_CANDIDATES = ["metadata.json", "metadata", "metadata.parquet"]


def copy_metadata(source_folder: str, dest_dir: str, explicit: str = None) -> None:
    """Copies whatever metadata is in source_folder (file or directory)
    into dest_dir, verbatim -- schema info, not patient data, safe to
    commit. Checks a few likely names/extensions since this has varied
    between transfers.

    `explicit` (e.g. from main.py --metadata) points at the metadata
    JSON directly when it does not live inside the transfer folder --
    the multi-site case. It is normalized to dest_dir/metadata.json, the
    name every downstream step expects."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(
                f"--metadata points to {explicit}, which is not a file.")
        dest = os.path.join(dest_dir, "metadata.json")
        shutil.copy2(explicit, dest)
        print(f"Copied metadata file (explicit --metadata) -> {dest}")
        return
    src = next(
        (
            os.path.join(source_folder, c)
            for c in METADATA_CANDIDATES
            if os.path.exists(os.path.join(source_folder, c))
        ),
        None,
    )
    if src is None:
        print(f"⚠️  No metadata file/folder found in {source_folder} (checked: {METADATA_CANDIDATES}).")
        return

    dest = os.path.join(dest_dir, os.path.basename(src))
    if os.path.isdir(src):
        shutil.copytree(src, dest, dirs_exist_ok=True)
        print(f"Copied metadata folder ({len(os.listdir(src))} entries) -> {dest}")
    else:
        shutil.copy2(src, dest)
        print(f"Copied metadata file -> {dest}")


def write_row_sample(df: pl.DataFrame, dest_path: str, n: int, seed: int = 0) -> None:
    """Writes a random sample of n rows (fixed seed, reproducible) to dest_path."""
    n = min(n, df.height)
    df.sample(n=n, seed=seed).write_parquet(dest_path)
    print(f"Saved random sample of {n} row(s) (seed={seed}) -> {dest_path}")
