"""
Utility helpers for :mod:`dRFEtools`.

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
    """
    Coerce RFE outputs into the standardized dictionary format.

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

    if isinstance(result, tuple):
        warn(
            "Tuple-based RFE outputs are deprecated; use the dictionary format instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if len(result) != 4:
            raise TypeError("Tuple RFE results must have 4 elements.")

        n_features, metric1, metric2, indices = result

        return {
            "n_features": n_features,
            "metrics": {
                "nmi_score": metric1,
                "r2_score": metric1,
                "accuracy_score": metric2,
                "mse_score": metric2,
                "roc_auc_score": metric1,
                "explain_var": metric2,
            },
            "indices": list(indices),
        }

    raise TypeError(
        f"normalize_rfe_result expected dict or 4-tuple, but got {type(result).__name__}"
    )


def get_feature_importances(estimator: Any) -> np.ndarray:
    """
    Return absolute feature importance values from an estimator.

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
    """
    Persist a plot to standard image formats.

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
    """
    Normalize user-provided file system paths.

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
