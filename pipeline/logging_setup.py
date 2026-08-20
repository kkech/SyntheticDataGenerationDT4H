"""
Tee every run's console output to a log file.

Redirecting with `python main.py >> out.txt` only captures stdout, so
warnings and tracebacks -- which go to stderr, and are exactly what you
want when sharing a failing run -- are lost. This captures both streams
plus the warnings module, while still printing to the terminal so a long
CTGAN run remains watchable.
"""

import sys
import warnings
from datetime import datetime, timezone


class _Tee:
    """Write to two streams at once, flushing eagerly so a killed or
    crashed run still leaves a complete log on disk."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        self._primary.write(data)
        self._primary.flush()
        self._secondary.write(data)
        self._secondary.flush()
        return len(data)

    def flush(self):
        self._primary.flush()
        self._secondary.flush()

    def isatty(self):
        # tqdm and similar check this; report the terminal's answer so
        # progress bars keep behaving as they would without teeing.
        return getattr(self._primary, "isatty", lambda: False)()

    def __getattr__(self, item):
        return getattr(self._primary, item)


def start_logging(path: str, argv: list[str] | None = None):
    """
    Point stdout, stderr and warnings at `path` as well as the console.

    Returns the opened file handle; the caller keeps it open for the
    lifetime of the run.
    """
    handle = open(path, "w", buffering=1, encoding="utf-8")

    header = [
        "=" * 70,
        f"Run started: {datetime.now(timezone.utc).isoformat()}",
        f"Command: {' '.join(argv or sys.argv)}",
        "=" * 70,
        "",
    ]
    handle.write("\n".join(header))

    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)

    # Warnings are written through the warnings machinery rather than a
    # plain stderr print, so route them explicitly.
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        sys.stderr.write(f"{filename}:{lineno}: {category.__name__}: {message}\n")

    warnings.showwarning = _showwarning

    return handle


def stop_logging(handle) -> None:
    sys.stdout = getattr(sys.stdout, "_primary", sys.stdout)
    sys.stderr = getattr(sys.stderr, "_primary", sys.stderr)
    if handle and not handle.closed:
        handle.write(f"\nRun finished: {datetime.now(timezone.utc).isoformat()}\n")
        handle.close()
