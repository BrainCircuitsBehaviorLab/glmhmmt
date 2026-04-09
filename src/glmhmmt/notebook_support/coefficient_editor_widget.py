from pathlib import Path

import anywidget
import traitlets


_ASSET_DIR = Path(__file__).parent


def _read_asset(name: str) -> str:
    return (_ASSET_DIR / name).read_text(encoding="utf-8")


class CoefficientEditorWidget(anywidget.AnyWidget):
    _esm = _read_asset("coefficient_editor_widget.js")
    _css = _read_asset("coefficient_editor_widget.css")

    title = traitlets.Unicode("Coefficient Editor").tag(sync=True)
    subtitle = traitlets.Unicode("").tag(sync=True)
    features = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    weights = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
    original_weights = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
    slider_min = traitlets.Float(-10.0).tag(sync=True)
    slider_max = traitlets.Float(10.0).tag(sync=True)
    slider_step = traitlets.Float(0.05).tag(sync=True)
