"""
Provenance capture for publication.

A synthetic dataset submitted to a journal has to be reproducible from the
paper alone: same seed, same library versions, same code revision, same
inputs. This module records exactly that, so every generated file can be
traced back to the conditions that produced it.
"""

import hashlib
import os
import platform
import random
import subprocess
import sys
from importlib import metadata


def set_global_seeds(seed: int) -> dict:
    """
    Seeds every RNG the synthesizers draw from. Returns what was set, so
    the summary records it rather than relying on the config being
    unchanged afterwards.

    Note the honest limitation: seeding makes a run reproducible on the
    same hardware and library versions, but GPU kernels are not always
    bit-deterministic across driver/hardware changes, so exact
    reproduction on different hardware is not guaranteed.
    """
    record = {"seed": seed, "python_random": True, "numpy": False, "torch": False}

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        record["numpy"] = True
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        record["torch"] = True
        record["torch_cuda_seeded"] = torch.cuda.is_available()
    except ImportError:
        pass

    return record


def library_versions() -> dict:
    """Versions of everything that materially affects the output."""
    packages = [
        "sdv", "ctgan", "smartnoise-synth", "opendp", "torch",
        "numpy", "pandas", "polars", "scikit-learn",
    ]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def git_revision() -> dict:
    """Code revision, so the exact pipeline version is recoverable."""
    def _run(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def file_checksum(path: str) -> str | None:
    """SHA-256 of an input file, so the exact training data is identifiable."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = torch.version.cuda
            free, total = torch.cuda.mem_get_info()
            env["gpu_memory_total_gb"] = round(total / 1e9, 2)
            env["gpu_memory_free_gb"] = round(free / 1e9, 2)
    except ImportError:
        env["cuda_available"] = None
    return env


def provenance(input_path: str, seed: int) -> dict:
    """Everything needed to reproduce a generation run."""
    return {
        "seed_state": set_global_seeds(seed),
        "library_versions": library_versions(),
        "git": git_revision(),
        "environment": environment(),
        "training_data": {
            "path": input_path,
            "sha256": file_checksum(input_path),
        },
    }
