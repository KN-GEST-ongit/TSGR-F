"""Matplotlib backend helpers for deterministic file-only report generation."""

from __future__ import annotations


def configure_headless_backend() -> None:
    """Force the non-interactive Agg backend before importing pyplot.

    Batch reports and automated tests must not depend on a working Tcl/Tk
    installation. Interactive inspection commands keep their own backend choice.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
