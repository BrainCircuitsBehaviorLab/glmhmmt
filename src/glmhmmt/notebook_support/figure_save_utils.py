from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import anywidget
import traitlets

from glmhmmt.runtime import load_app_config
from .anywidget_compat import wrap_anywidget


_ASSET_DIR = Path(__file__).parent


def _read_asset(name: str) -> str:
    return (_ASSET_DIR / name).read_text(encoding="utf-8")


def _get_save_figure_config(config_path: Path | None) -> dict[str, str]:
    cfg = load_app_config(config_path)
    section = cfg.get("widgets", {}).get("save_figure", {})
    if not isinstance(section, dict):
        return {}
    return {str(key): str(value) for key, value in section.items()}


def _save_figure_theme_tokens(save_cfg: dict[str, str]) -> dict[str, str]:
    token_keys = (
        "radius",
        "padding_y",
        "padding_x",
        "font_size",
        "light_border",
        "light_background",
        "light_text",
        "light_hover_background",
        "light_hover_border",
        "light_disabled_background",
        "light_disabled_border",
        "dark_border",
        "dark_background",
        "dark_text",
        "dark_hover_background",
        "dark_hover_border",
        "dark_disabled_background",
        "dark_disabled_border",
    )
    return {
        key.replace("_", "-"): value
        for key in token_keys
        if (value := save_cfg.get(key))
    }


def _toast_detail_html(detail: str, save_cfg: dict[str, str]) -> str:
    color = save_cfg.get("toast_detail_color", "#6b7280")
    return f"<span style='color:{color}'>{detail}</span>"


class SaveFigureAnyWidget(anywidget.AnyWidget):
    _esm = _read_asset("figure_save_widget.js")
    _css = _read_asset("figure_save_widget.css")

    clicks = traitlets.Int(0).tag(sync=True)
    label = traitlets.Unicode("Save").tag(sync=True)
    disabled = traitlets.Bool(False).tag(sync=True)
    theme_tokens = traitlets.Dict(default_value={}).tag(sync=True)


def get_plot_save_format(config_path: Path | None) -> str:
    cfg = load_app_config(config_path)
    fmt = str(cfg.get("plots", {}).get("save_format", "pdf")).lower().strip(". ")
    if fmt not in {"pdf", "svg", "png"}:
        fmt = "pdf"
    return fmt


def sanitize_stem(stem: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "figure"


def build_plot_path(results_dir: Path, task_name: str, model_id: str, stem: str, fmt: str) -> Path:
    out_dir = results_dir / "plots" / task_name / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{sanitize_stem(stem)}.{fmt}"


def _axis_for_location(fig: Any, location: tuple[int, int]):
    row, col = (int(location[0]), int(location[1]))
    for ax in getattr(fig, "axes", []):
        get_subplotspec = getattr(ax, "get_subplotspec", None)
        if get_subplotspec is None:
            continue
        spec = get_subplotspec()
        if spec is None:
            continue
        if row in range(spec.rowspan.start, spec.rowspan.stop) and col in range(spec.colspan.start, spec.colspan.stop):
            return ax
    raise ValueError(f"No subplot found at location ({row}, {col}).")


def _save_axis(fig: Any, ax: Any, out_path: Path, fmt: str) -> Path:
    if hasattr(fig, "canvas") and fig.canvas is not None:
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    save_kwargs = {"bbox_inches": bbox}
    if fmt != "svg":
        save_kwargs["dpi"] = 300
    fig.savefig(out_path, **save_kwargs)
    return out_path


def save_figure(
    fig,
    *,
    results_dir: Path,
    config_path: Path | None,
    task_name: str,
    model_id: str,
    stem: str,
    location: tuple[int, int] | None = None,
) -> Path:
    fmt = get_plot_save_format(config_path)
    out_path = build_plot_path(results_dir, task_name, model_id, stem, fmt)
    if location is not None:
        ax = _axis_for_location(fig, location)
        return _save_axis(fig, ax, out_path, fmt)
    if hasattr(fig, "canvas") and fig.canvas is not None:
        fig.canvas.draw()
    save_kwargs = {"bbox_inches": "tight"}
    if fmt != "svg":
        save_kwargs["dpi"] = 300
    fig.savefig(out_path, **save_kwargs)
    return out_path


class PlotSaver:
    def __init__(self, mo, *, results_dir: Path, config_path: Path | None, task_name: str, model_id: str):
        self.mo = mo
        self.results_dir = results_dir
        self.config_path = config_path
        self.task_name = task_name
        self.model_id = model_id
        self.fmt = get_plot_save_format(config_path)
        self.save_cfg = _get_save_figure_config(config_path)
        self.theme_tokens = _save_figure_theme_tokens(self.save_cfg)
        self._registry: dict[str, dict[str, object]] = {}
        self._save_all = SaveFigureAnyWidget(
            label=self.save_cfg.get("save_all_label", "Save all model plots"),
            disabled=True,
            theme_tokens=self.theme_tokens,
        )
        self._save_all.observe(self._handle_save_all_click, names="clicks")
        self._save_all._save_observer = self._handle_save_all_click
        self._save_all_ui = None

    def _save_one(self, fig, *, stem: str, location: tuple[int, int] | None = None) -> Path:
        return save_figure(
            fig,
            results_dir=self.results_dir,
            config_path=self.config_path,
            task_name=self.task_name,
            model_id=self.model_id,
            stem=stem,
            location=location,
        )

    def _register(self, fig, *, name: str, stem: str, location: tuple[int, int] | None = None) -> None:
        self._registry[stem] = {
            "fig": fig,
            "name": name,
            "stem": stem,
            "location": location,
        }
        self._save_all.disabled = not bool(self._registry)

    def _saved_message(self, saved_paths: list[Path]) -> str:
        if not saved_paths:
            return "No files saved."
        if len(saved_paths) == 1:
            return saved_paths[0].name
        return f"{saved_paths[0].name} + {len(saved_paths) - 1} more"

    def save_all(self) -> tuple[list[Path], list[tuple[str, Exception]]]:
        saved_paths: list[Path] = []
        errors: list[tuple[str, Exception]] = []
        for item in list(self._registry.values()):
            try:
                out_path = self._save_one(
                    item["fig"],
                    stem=str(item["stem"]),
                    location=item["location"],
                )
                saved_paths.append(out_path)
            except Exception as exc:
                errors.append((str(item["name"]), exc))
        return saved_paths, errors

    def _handle_save_all_click(self, change) -> None:
        if int(change["new"]) <= int(change["old"]):
            return
        if not self._registry:
            self.mo.status.toast(
                self.save_cfg.get("empty_title", "No plots available"),
                _toast_detail_html(
                    self.save_cfg.get("empty_detail", "Render the notebook plots first."),
                    self.save_cfg,
                ),
                kind="danger",
            )
            return

        saved_paths, errors = self.save_all()
        if errors:
            detail = self._saved_message(saved_paths)
            if saved_paths:
                detail = f"{detail}; {len(errors)} failed"
            else:
                detail = f"{len(errors)} failed"
            self.mo.status.toast(
                self.save_cfg.get(
                    "saved_error_title" if saved_paths else "failed_title",
                    "Saved with errors" if saved_paths else "Could not save plots",
                ),
                _toast_detail_html(detail, self.save_cfg),
                kind="danger",
            )
            return

        count = len(saved_paths)
        title = (
            self.save_cfg.get("saved_title", "Saved")
            if count == 1
            else self.save_cfg.get("saved_many_title", "Saved {count} plots").format(count=count)
        )
        self.mo.status.toast(
            title,
            _toast_detail_html(self._saved_message(saved_paths), self.save_cfg),
        )

    def save_all_widget(self, label: str | None = None):
        self._save_all.label = label or self.save_cfg.get("save_all_label", "Save all model plots")
        if self._save_all_ui is None:
            self._save_all_ui = wrap_anywidget(self._save_all)
        return self._save_all_ui

    def __call__(
        self,
        fig,
        name: str,
        *,
        stem: str | None = None,
        label: str | None = None,
        location: tuple[int, int] | None = None,
    ):
        if location is not None:
            row, col = (int(location[0]), int(location[1]))
            default_stem = f"{sanitize_stem(name.lower())}_r{row}_c{col}"
        else:
            default_stem = sanitize_stem(name.lower())
        resolved_stem = stem or default_stem
        button_label = label or f"{self.save_cfg.get('default_label', 'Save')} .{self.fmt}"
        self._register(fig, name=name, stem=resolved_stem, location=location)
        widget = SaveFigureAnyWidget(label=button_label, theme_tokens=self.theme_tokens)

        def _handle_click(change):
            if int(change["new"]) <= int(change["old"]):
                return
            try:
                out_path = self._save_one(fig, stem=resolved_stem, location=location)
                self.mo.status.toast(
                    self.save_cfg.get("saved_title", "Saved"),
                    _toast_detail_html(out_path.name, self.save_cfg),
                )
            except Exception as exc:
                self.mo.status.toast(
                    self.save_cfg.get("failed_title", "Could not save plots"),
                    _toast_detail_html(f"{type(exc).__name__}: {exc}", self.save_cfg),
                    kind="danger",
                )

        widget.observe(_handle_click, names="clicks")
        widget._save_observer = _handle_click
        return wrap_anywidget(widget)


def make_plot_saver(mo, *, results_dir: Path, config_path: Path | None, task_name: str, model_id: str):
    return PlotSaver(
        mo,
        results_dir=results_dir,
        config_path=config_path,
        task_name=task_name,
        model_id=model_id,
    )


def save_button(
    mo,
    fig,
    *,
    results_dir: Path,
    config_path: Path | None,
    task_name: str,
    model_id: str,
    stem: str,
    label: str | None = None,
    location: tuple[int, int] | None = None,
):
    fmt = get_plot_save_format(config_path)
    default_label = label or _get_save_figure_config(config_path).get("default_label", "Save")
    return make_plot_saver(
        mo,
        results_dir=results_dir,
        config_path=config_path,
        task_name=task_name,
        model_id=model_id,
    )(
        fig,
        name=default_label,
        stem=stem,
        label=f"{default_label} .{fmt}",
        location=location,
    )
