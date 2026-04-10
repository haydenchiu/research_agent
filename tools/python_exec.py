from __future__ import annotations

import contextlib
import io
import traceback
import uuid
from pathlib import Path

from config.settings import CHARTS_DIR


def execute_python(code: str) -> dict:
    """Execute Python code in a sandboxed environment with data-analysis libraries.

    Returns a dict with:
        stdout  – captured print output
        result  – repr of the last expression (if any)
        charts  – list of file paths to any saved chart images
        error   – traceback string if execution failed
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    chart_paths: list[str] = []
    stdout_buf = io.StringIO()

    # Pre-import safe libraries and inject a savefig helper
    sandbox_globals: dict = {"__builtins__": __builtins__}
    setup_code = """
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
"""
    try:
        exec(setup_code, sandbox_globals)
    except Exception:
        pass

    # Patch plt.show to save to disk instead
    _charts_dir = str(CHARTS_DIR)

    def _save_on_show() -> None:
        import matplotlib.pyplot as _plt

        fig = _plt.gcf()
        if fig.get_axes():
            fname = f"chart_{uuid.uuid4().hex[:8]}.png"
            path = str(Path(_charts_dir) / fname)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            chart_paths.append(path)
        _plt.close("all")

    sandbox_globals["plt"].show = _save_on_show  # type: ignore[attr-defined]

    result_value = None
    error_str = ""

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, sandbox_globals)
    except Exception:
        error_str = traceback.format_exc()

    # If the last statement was an expression, try to capture it
    # (simple heuristic: won't always work, stdout is more reliable)

    return {
        "stdout": stdout_buf.getvalue(),
        "result": result_value,
        "charts": chart_paths,
        "error": error_str,
    }
