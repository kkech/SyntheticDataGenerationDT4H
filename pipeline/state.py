"""
Persists which pipeline steps have completed, so a rerun skips work that
already succeeded -- unless explicitly forced. Backed by a plain JSON file
so it's human-readable and diffable in git if committed.
"""

import json
import os
from datetime import datetime, timezone


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
        with open(self.status_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_completed(self, step_name: str) -> bool:
        return self._data.get(step_name, {}).get("completed", False)

    def mark_pending(self, step_name: str) -> None:
        """The step WILL run in the current pipeline run but has not
        started yet. Replaces any stale completed/failed entry from a
        previous run, so the status never claims 'completed' for a step
        that is about to be redone."""
        self._data[step_name] = {
            "completed": False,
            "pending": True,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }
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
