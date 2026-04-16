from __future__ import annotations

import marimo as mo


def wrap_anywidget(widget):
    """Wrap an anywidget using marimo's native bridge."""
    return mo.ui.anywidget(widget)
