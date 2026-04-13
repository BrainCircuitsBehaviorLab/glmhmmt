from __future__ import annotations

from weakref import WeakKeyDictionary

import marimo._output.data.data as mo_data
from marimo._plugins.ui._core.ui_element import UIElement
from marimo._plugins.ui._impl.from_anywidget import (
    anywidget as MarimoAnyWidget,
    get_anywidget_model_id,
)
from marimo._utils.code import hash_code


class _StableAnyWidget(MarimoAnyWidget):
    """Marimo anywidget wrapper with stable model-id based caching."""

    def __init__(self, widget):
        self.widget = widget
        self._initialized = False

        js = str(getattr(widget, "_esm", "") or "")
        js_hash = hash_code(js)

        # Force comm initialization so the widget has a model id before the
        # UIElement is constructed.
        _ = widget.comm
        model_id = get_anywidget_model_id(widget)

        UIElement.__init__(
            self,
            component_name="marimo-anywidget",
            initial_value={"model_id": model_id},
            label=None,
            args={
                "js-url": mo_data.js(js).url if js else "",
                "js-hash": js_hash,
                "model-id": model_id,
            },
            on_change=None,
        )


_WIDGET_CACHE: WeakKeyDictionary[object, _StableAnyWidget] = WeakKeyDictionary()


def wrap_anywidget(widget):
    cached = _WIDGET_CACHE.get(widget)
    if cached is not None:
        return cached

    wrapped = _StableAnyWidget(widget)
    _WIDGET_CACHE[widget] = wrapped
    return wrapped
