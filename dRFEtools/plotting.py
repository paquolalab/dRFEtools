"""Plotting utilities for dRFEtools."""

from __future__ import annotations

from typing import Dict
from pathlib import Path
from warnings import filterwarnings

import pandas as pd
from matplotlib import MatplotlibDeprecationWarning
from plotnine import aes, geom_point, geom_vline, ggplot, labs, scale_x_log10, theme_light

from .lowess.redundant import (
    _cal_lowess,
    extract_max_lowess,
    extract_peripheral_lowess,
)
from .utils import normalize_rfe_result, save_plot_variants

filterwarnings("ignore", category=MatplotlibDeprecationWarning)
filterwarnings("ignore", category=UserWarning, module="plotnine.*")
filterwarnings("ignore", category=DeprecationWarning, module="plotnine.*")

__all__ = ["plot_metric", "plot_with_lowess_vline"]

METRIC_KEYS = {
    "nmi": ("nmi_score", "r2_score"),
    "roc": ("roc_auc_score",),
    "acc": ("accuracy_score",),
    "r2": ("r2_score",),
    "mse": ("mse_score",),
    "evar": ("explain_var",),
}


def _metric_value(entry: Dict, metric_name: str):
    normalized = normalize_rfe_result(entry)
    metrics = normalized.get("metrics", {})
    for key in METRIC_KEYS[metric_name]:
        if key in metrics:
            return metrics[key]
    return None


def plot_metric(d: Dict, fold: int, output_dir: str | Path, metric_name: str, y_label: str) -> None:
    """Plot feature elimination results for a metric."""

    if metric_name not in METRIC_KEYS:
        raise ValueError(f"Unknown metric_name: {metric_name}")

    df_elim = pd.DataFrame(
        [{"n features": k, y_label: _metric_value(v, metric_name)} for k, v in d.items()]
    )

    gg = (
        ggplot(df_elim, aes(x="n features", y=y_label))
        + geom_point()
        + scale_x_log10()
        + theme_light()
        + labs(x="Number of features", y=y_label)
    )

    outfile = Path(output_dir) / f"{metric_name}_fold_{fold}"
    save_plot_variants(gg, outfile)
    print(gg)


def plot_with_lowess_vline(
    d: Dict,
    fold: int,
    output_dir: str | Path,
    frac: float = DEFAULT_FRAC,
    step_size: float = DEFAULT_STEP_SIZE,
    classify: bool = True,
    multi: bool = False,
    acc: bool = False,
) -> None:
    """Plot LOWESS smoothing with feature count annotations."""

    label = "ROC AUC" if multi else "Accuracy" if acc else "NMI" if classify else "R2"
    _, max_feat_log10 = extract_max_lowess(d, frac, multi, acc)
    x, y, _, _, _ = _cal_lowess(d, frac, multi, acc)
    df_elim = pd.DataFrame({"X": x, "Y": y})
    _, lo = extract_max_lowess(d, frac, multi, acc)
    _, l1 = extract_peripheral_lowess(d, frac, step_size, multi, acc)

    gg = (
        ggplot(df_elim, aes(x="X", y="Y"))
        + geom_point(color="blue")
        + geom_vline(xintercept=lo, color="blue", linetype="dashed")
        + geom_vline(xintercept=l1, color="orange", linetype="dashed")
        + scale_x_log10()
        + labs(x="log10(N Features)", y=label)
        + theme_light()
    )

    print(gg)
    outfile = Path(output_dir) / f"{label.replace(' ', '_')}_log10_dRFE_fold_{fold}"
    save_plot_variants(gg, outfile)
