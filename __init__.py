"""Local development shim for the `src/` package layout.

When working from the thesis workspace root (`code/`), Python sees the
`code/glmhmmt/` project directory before the installed package. This shim
redirects local imports to `code/glmhmmt/src/glmhmmt` so `uv run` from the
workspace root behaves the same as an installed package import.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_SRC_PACKAGE_DIR = Path(__file__).resolve().parent / "src" / "glmhmmt"
_SRC_INIT = _SRC_PACKAGE_DIR / "__init__.py"

_spec = spec_from_file_location(
    __name__,
    _SRC_INIT,
    submodule_search_locations=[str(_SRC_PACKAGE_DIR)],
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load glmhmmt package from {_SRC_INIT}")

_module = module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
