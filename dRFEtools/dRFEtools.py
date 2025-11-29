#!/usr/bin/env python
"""Dynamic recursive feature elimination interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple
from warnings import filterwarnings

from matplotlib import MatplotlibDeprecationWarning
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier

from .scoring import _regr_fe, _rf_fe
from .utils import normalize_rfe_result

filterwarnings("ignore", category=MatplotlibDeprecationWarning)
filterwarnings("ignore", category=UserWarning, module="plotnine.*")
filterwarnings("ignore", category=DeprecationWarning, module="plotnine.*")

__all__ = ["rf_rfe", "dev_rfe", "plot_metric", "plot_with_lowess_vline"]


def _n_features_iter(nf: int, keep_rate: float) -> Iterable[int]:
    """Yield the number of features to keep at each elimination step."""

    while nf != 1:
        nf = max(1, int(nf * keep_rate))
        yield nf


def _normalize_metrics(estimator, normalized: Dict) -> Dict:
    """Select task-appropriate metrics from a normalized payload."""

    metrics = normalized.get("metrics", {})
    if not isinstance(normalized, dict):
        return metrics

    if isinstance(estimator, RandomForestClassifier):
        return {
            key: metrics[key]
            for key in ("nmi_score", "accuracy_score", "roc_auc_score")
            if key in metrics
        }
    if is_classifier(estimator):
        return {
            key: metrics[key]
            for key in ("nmi_score", "accuracy_score", "roc_auc_score")
            if key in metrics
        }
    return {
        key: metrics[key]
        for key in ("r2_score", "mse_score", "explain_var")
        if key in metrics
    }


def rf_rfe(
    estimator,
    X,
    Y,
    features,
    fold: int,
    out_dir: str | Path = ".",
    elimination_rate: float = 0.2,
    RANK: bool = True,
) -> Tuple[Dict[int, Dict], Dict]:
    """Run random-forest feature elimination over an iterator process."""

    if not 0 < elimination_rate < 1:
        raise ValueError("elimination_rate must be between 0 and 1")

    results: Dict[int, Dict] = {}
    first_pass: Dict | None = None
    keep_rate = 1 - elimination_rate
    out_dir = Path(out_dir)

    for payload in _rf_fe(
        estimator,
        X,
        Y,
        _n_features_iter(X.shape[1], keep_rate),
        features,
        fold,
        str(out_dir),
        RANK,
    ):
        normalized = normalize_rfe_result(payload)
        normalized["metrics"] = _normalize_metrics(estimator, normalized)

        if first_pass is None:
            first_pass = normalized
        results[normalized["n_features"]] = normalized

    return results, first_pass


def dev_rfe(
    estimator,
    X,
    Y,
    features,
    fold: int,
    out_dir: str | Path = ".",
    elimination_rate: float = 0.2,
    dev_size: float = 0.2,
    RANK: bool = True,
    SEED: bool = False,
    random_state: int | None = None,
) -> Tuple[Dict[int, Dict], Dict]:
    """Recursive feature elimination for estimators using a dev split."""

    if not 0 < elimination_rate < 1 or not 0 < dev_size < 1:
        raise ValueError("elimination_rate and dev_size must be between 0 and 1")

    results: Dict[int, Dict] = {}
    first_pass: Dict | None = None
    keep_rate = 1 - elimination_rate
    out_dir = Path(out_dir)

    resolved_random_state = 13 if SEED else random_state

    for payload in _regr_fe(
        estimator,
        X,
        Y,
        _n_features_iter(X.shape[1], keep_rate),
        features,
        fold,
        str(out_dir),
        dev_size,
        resolved_random_state,
        RANK,
    ):
        normalized = normalize_rfe_result(payload)
        normalized["metrics"] = _normalize_metrics(estimator, normalized)

        if first_pass is None:
            first_pass = normalized
        results[normalized["n_features"]] = normalized

    return results, first_pass
