from __future__ import annotations

import base64
from functools import lru_cache
from weakref import WeakKeyDictionary

from marimo._plugins.ui._core.ui_element import UIElement
from marimo._plugins.ui._impl.from_anywidget import (
    anywidget as MarimoAnyWidget,
    get_anywidget_model_id,
)
from marimo._utils.code import hash_code
from marimo._utils.data_uri import build_data_url


@lru_cache(maxsize=32)
def _build_js_data_url(js: str) -> str:
    encoded = base64.b64encode(js.encode("utf-8"))
    return build_data_url("text/javascript", encoded)


def _resolve_js_url(js: str) -> str:
    if not js:
        return ""
    if js.startswith(("data:", "http://", "https://")):
        return js
    return _build_js_data_url(js)


class _StableAnyWidget(MarimoAnyWidget):
    """Marimo anywidget wrapper that avoids edit-mode virtual-file churn.

    Marimo's default wrapper serves `_esm` through `./@file/...` virtual files.
    In edit mode those files live in shared memory and can disappear during a
    reload, which leaves the browser trying to import a stale module URL. By
    embedding the JS as a data URL we keep the module stable for the lifetime
    of the widget instance.
    """

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
                "js-url": _resolve_js_url(js),
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
