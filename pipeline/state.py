"""
Persists which pipeline steps have completed, so a rerun skips work that
already succeeded -- unless explicitly forced. Backed by a plain JSON file
so it's human-readable and diffable in git if committed.
"""

import json
import os
from datetime import datetime, timezone


def plan_cascade_invalidations(step_names: list[str], queued: set[str],
                               is_completed) -> list[str]:
    """
    Which completed steps become STALE because an earlier step is about to
    rerun. Every step strictly AFTER the earliest queued step (in pipeline
    order) that is currently completed -- and not already queued itself --
    would otherwise keep outputs computed from the OLD upstream data, and a
    later analysis run would silently mix generations.

    Pure planning (no state is touched here) so the semantics are testable:
    invalidation propagates only downstream of the earliest step actually
    queued to run, which is what makes analysis-only reruns cheap -- forcing
    the analysis tail never invalidates the generation steps before it.
    """
    if not queued:
        return []
    order = {name: i for i, name in enumerate(step_names)}
    earliest = min(order[name] for name in queued if name in order)
    return [name for name in step_names
            if order[name] > earliest and name not in queued and is_completed(name)]


class PipelineState:
    def __init__(self, status_path: str):
        self.status_path = status_path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.status_path):
            with open(self.status_path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.status_path) or ".", exist_ok=True)
        # Write-to-temp then os.replace (atomic on POSIX): a crash mid-write
        # can only ever leave a stray .tmp file behind, never a truncated or
        # half-written pipeline_status.json.
        tmp_path = self.status_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp_path, self.status_path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def is_completed(self, step_name: str) -> bool:
        return self._data.get(step_name, {}).get("completed", False)

    def mark_pending(self, step_name: str, note: str | None = None) -> None:
        """The step WILL run in the current pipeline run (or must rerun
        later) but has not started yet. Replaces any stale completed/failed
        entry from a previous run, so the status never claims 'completed'
        for a step that is about to be redone. `note` records WHY a step
        was re-queued (e.g. cascade invalidation by an upstream rerun)."""
        entry = {
            "completed": False,
            "pending": True,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }
        if note:
            entry["note"] = note
        self._data[step_name] = entry
        self._save()

    def mark_running(self, step_name: str) -> None:
        self._data[step_name] = {
            "completed": False,
            "running": True,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def mark_completed(self, step_name: str) -> None:
        self._data[step_name] = {
            "completed": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def mark_failed(self, step_name: str, error: str) -> None:
        self._data[step_name] = {
            "completed": False,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        self._save()

    def reset(self, step_name: str) -> None:
        self._data.pop(step_name, None)
        self._save()

    def summary(self) -> dict:
        return dict(self._data)
