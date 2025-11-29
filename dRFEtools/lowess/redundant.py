"""LOWESS-based utilities for dynamic RFE plots and thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy import interpolate

from ..utils import normalize_rfe_result, save_plot_variants

__author__ = "Kynon J Benjamin"

__all__ = [
    "_cal_lowess",
    "extract_max_lowess",
    "optimize_lowess_plot",
    "extract_peripheral_lowess",
]

DEFAULT_FRAC = 0.3
DEFAULT_STEP_SIZE = 0.02
LOWESS_POINTS = 5001


def _run_lowess(xnew: ArrayLike, ynew: ArrayLike, frac: float) -> np.ndarray:
    """Execute LOWESS smoothing."""

    lowess = sm.nonparametric.lowess
    return lowess(ynew, xnew, frac=frac, it=20)


def _array_to_tuple(np_array: ArrayLike):
    """Recursively convert numpy arrays into tuples for statsmodels."""

    try:
        return tuple(_array_to_tuple(_) for _ in np_array)
    except TypeError:
        return np_array


def _get_elim_df_ordered(d: Dict, multi: bool, use_accuracy: bool) -> pd.DataFrame:
    """Convert elimination dictionary into an ordered DataFrame."""

    rows = []
    for n_features, value in d.items():
        normalized = normalize_rfe_result(value)
        metrics = normalized.get("metrics", {})
        metric_key = "accuracy_score" if use_accuracy else "roc_auc_score" if multi else "nmi_score"
        y_val = metrics.get(metric_key) if metric_key in metrics else metrics.get("r2_score")
        rows.append(
            {
                "x": n_features,
                "y": y_val,
                "acc": metrics.get("accuracy_score"),
            }
        )
    df_elim = pd.DataFrame(rows).sort_values("x")
    df_elim["log10_x"] = np.log10(df_elim["x"] + 0.5)
    return df_elim


def _cal_lowess(d: Dict, frac: float, multi: bool, acc: bool) -> Tuple[np.ndarray, ...]:
    """Calculate the LOWESS curve for elimination metrics."""

    df_elim = _get_elim_df_ordered(d, multi, acc)
    x = df_elim["log10_x"].to_numpy()
    y = df_elim["acc"].to_numpy() if acc else df_elim["y"].to_numpy()
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(x.min(), x.max(), num=LOWESS_POINTS, endpoint=True)
    ynew = interpolate.splev(xnew, tck, der=0)
    z = _run_lowess(_array_to_tuple(xnew), _array_to_tuple(ynew), frac)
    return x, y, z, xnew, ynew


def _cal_lowess_rate_log10(d: Dict, frac: float = DEFAULT_FRAC, multi: bool = False, acc: bool = False) -> pd.DataFrame:
    """Compute rate of change on the log10-transformed LOWESS curve."""

    _, _, z, _, _ = _cal_lowess(d, frac, multi, acc)
    dfz = pd.DataFrame(z, columns=["Features", "LOWESS"])
    pts = dfz.drop(0).copy()
    pts["DxDy"] = np.diff(dfz.Features) / np.diff(dfz.LOWESS)
    return pts


def extract_max_lowess(d: Dict, frac: float = DEFAULT_FRAC, multi: bool = False, acc: bool = False) -> Tuple[int, float]:
    """Extract max features based on LOWESS rate of change."""

    _, _, z, xnew, ynew = _cal_lowess(d, frac, multi, acc)
    df_elim = _get_elim_df_ordered(d, multi, acc)
    df_lowess = pd.DataFrame(
        {
            "X": xnew,
            "Y": ynew,
            "xprime": pd.DataFrame(z)[0],
            "yprime": pd.DataFrame(z)[1],
        }
    )
    val = df_lowess[df_lowess["yprime"] == max(df_lowess.yprime)].X.values
    closest_val = min(df_elim["log10_x"].values, key=lambda x: abs(x - val))
    return df_elim[df_elim["log10_x"] == closest_val].x.values[0], closest_val


def extract_peripheral_lowess(
    d: Dict,
    frac: float = DEFAULT_FRAC,
    step_size: float = DEFAULT_STEP_SIZE,
    multi: bool = False,
    acc: bool = False,
) -> Tuple[int, float]:
    """Extract peripheral features based on LOWESS curvature."""

    _, _, z, xnew, ynew = _cal_lowess(d, frac, multi, acc)
    df_elim = _get_elim_df_ordered(d, multi, acc)
    df_lowess = pd.DataFrame(
        {
            "X": xnew,
            "Y": ynew,
            "xprime": pd.DataFrame(z)[0],
            "yprime": pd.DataFrame(z)[1],
        }
    )
    dxdy = _cal_lowess_rate_log10(d, frac, multi, acc)
    dxdy = dxdy[dxdy.LOWESS >= max(dxdy.LOWESS) - np.std(dxdy.LOWESS)].copy()
    local_peak = [
        (dxdy.iloc[yy, 2] - dxdy.iloc[yy - 1, 2]) > step_size and dxdy.iloc[yy, 2] < 0
        for yy in range(1, dxdy.shape[0])
    ]
    local_peak.append(False)
    peaks = dxdy[local_peak]
    val = df_lowess[df_lowess["xprime"] == max(peaks.Features)].X.values
    redunt_feat_log10 = min(df_elim["log10_x"].values, key=lambda x: abs(x - val))
    peripheral_feat = df_elim[df_elim["log10_x"] == redunt_feat_log10].x.values[0]
    return peripheral_feat, redunt_feat_log10


def optimize_lowess_plot(
    d: Dict,
    fold: int,
    output_dir: str | Path,
    frac: float = DEFAULT_FRAC,
    step_size: float = DEFAULT_STEP_SIZE,
    classify: bool = True,
    save_plot: bool = False,
    multi: bool = False,
    acc: bool = False,
    print_out: bool = True,
) -> None:
    """Plot the LOWESS smoothing curve with selection annotations."""

    label = "ROC AUC" if (classify and multi) else "Accuracy" if acc else "NMI" if classify else "R2"
    title = f"Fraction: {frac:.2f}, Step Size: {step_size:.2f}"

    x, y, z, _, _ = _cal_lowess(d, frac, multi, acc)
    df_elim = pd.DataFrame({"X": 10 ** x - 0.5, "Y": y})
    lowess_df = pd.DataFrame(z, columns=["X0", "Y0"])
    lowess_df["X0"] = 10 ** lowess_df["X0"] - 0.5
    lo, _ = extract_max_lowess(d, frac, multi, acc)
    l1, _ = extract_peripheral_lowess(d, frac, step_size, multi, acc)

    fig, ax = plt.subplots()
    ax.plot(df_elim["X"], df_elim["Y"], "o", label="dRFE")
    ax.plot(lowess_df["X0"], lowess_df["Y0"], "-", label="Lowess")
    ax.vlines(lo, ymin=np.min(y), ymax=np.max(y), colors="b", linestyles="--", label="Max Features")
    ax.vlines(
        l1,
        ymin=np.min(y),
        ymax=np.max(y),
        colors="orange",
        linestyles="--",
        label="Peripheral Features",
    )
    ax.set_xscale("log")
    ax.set_xlabel("log(N Features)")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend(loc="best")

    if save_plot:
        output_dir = Path(output_dir)
        base = output_dir / f"optimize_lowess_{fold}_frac{frac:.2f}_step_{step_size:.2f}_{label.replace(' ', '_')}"
        save_plot_variants(fig, base)

    if print_out:
        plt.show()
