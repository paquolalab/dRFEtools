"""Utility helpers for :mod:`dRFEtools`.

The helpers centralize common behaviors such as result normalization,
feature importance extraction, and plot saving to improve maintainability
across the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

StandardizedRFEResult = Dict[str, Any]


def normalize_rfe_result(result: Any) -> StandardizedRFEResult:
    """Coerce RFE outputs into the standardized dictionary format.

    Parameters
    ----------
    result
        RFE output, either a mapping with expected keys or a legacy tuple.

    Returns
    -------
    dict
        Normalized RFE results with ``n_features``, ``metrics``, ``indices``,
        and ``selected`` keys.
    """

    if isinstance(result, dict):
        return result

    warn(
        "Tuple-based RFE outputs are deprecated; use the dictionary format instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(result, (list, tuple)):
        raise TypeError("RFE results must be a mapping or tuple-like sequence")

    n_features = result[0]
    payload = list(result[1:])

    indices = payload[3] if len(payload) >= 4 else None
    if isinstance(indices, np.ndarray):
        indices = indices.copy()

    metrics: Dict[str, float] = {}
    if len(payload) >= 1:
        metrics["nmi_score"] = payload[0]
        metrics["r2_score"] = payload[0]
    if len(payload) >= 2:
        metrics["accuracy_score"] = payload[1]
        metrics["mse_score"] = payload[1]
    if len(payload) >= 3:
        metrics["roc_auc_score"] = payload[2]
        metrics["explain_var"] = payload[2]

    return {
        "n_features": n_features,
        "metrics": metrics,
        "indices": indices,
        "selected": None,
    }


def get_feature_importances(estimator: Any) -> np.ndarray:
    """Return absolute feature importance values from an estimator.

    Parameters
    ----------
    estimator
        A fitted scikit-learn estimator that exposes ``coef_`` or
        ``feature_importances_``.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of feature importance values.

    Raises
    ------
    ValueError
        If the estimator does not expose a supported attribute.
    """

    if hasattr(estimator, "coef_"):
        importances: ArrayLike = estimator.coef_
        if getattr(importances, "ndim", 1) > 1:
            return np.linalg.norm(importances, axis=0, ord=1).flatten()
        return np.abs(np.asarray(importances)).flatten()
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_).flatten()
    raise ValueError(
        f"The estimator {estimator.__class__.__name__} must expose `coef_` or `feature_importances_`."
    )


def save_plot_variants(plot_obj: Any, output_path: Path, *, width: int = 7, height: int = 7) -> None:
    """Persist a plot to standard image formats.

    Both plotnine objects and Matplotlib figures are supported.
    """

    output_path = Path(output_path)
    for ext in (".svg", ".png", ".pdf"):
        destination = output_path.with_suffix(ext)
        if hasattr(plot_obj, "savefig"):
            plot_obj.savefig(destination, bbox_inches="tight")
        else:
            plot_obj.save(str(destination), width=width, height=height)


def ensure_path(path_like: str | Path) -> Path:
    """Normalize user-provided file system paths.

    Parameters
    ----------
    path_like
        Input path as string or :class:`~pathlib.Path`.

    Returns
    -------
    pathlib.Path
        Normalized ``Path`` object.
    """

    return Path(path_like).expanduser().resolve()
