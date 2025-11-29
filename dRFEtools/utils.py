"""Utility helpers for :mod:`dRFEtools`."""

from __future__ import annotations

from typing import Any, Dict
from warnings import warn

import numpy as np

StandardizedRFEResult = Dict[str, Any]


def normalize_rfe_result(result: Any) -> StandardizedRFEResult:
    """Coerce RFE outputs into the standardized dictionary format.

    The expected keys are ``n_features``, ``metrics``, ``indices``, and
    ``selected``. Tuple-based inputs are still supported for backward
    compatibility but emit a :class:`DeprecationWarning`.
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

    metrics = {}
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
