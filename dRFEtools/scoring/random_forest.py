"""Random-forest specific scoring and feature elimination helpers.

The routines here wrap out-of-bag (OOB) metrics and recursive feature
elimination for both classification and regression random forest models.
"""

from __future__ import annotations

from itertools import chain
from typing import Dict, Iterable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
)

from ..metrics.ranking import features_rank_fnc

__author__ = "Kynon Jade Benjamin"

__all__ = [
    "_rf_fe",
    "oob_score_r2",
    "oob_score_nmi",
    "oob_score_roc",
    "oob_score_mse",
    "oob_score_evar",
    "oob_score_accuracy",
]


def _oob_predictions(estimator: RandomForestClassifier | RandomForestRegressor) -> np.ndarray:
    """Return OOB predictions for supported random forest estimators."""

    if isinstance(estimator, RandomForestClassifier):
        return estimator.classes_[
            (estimator.oob_decision_function_[:, 1] > 0.5).astype(int)
        ]
    if isinstance(estimator, RandomForestRegressor):
        return estimator.oob_prediction_
    raise ValueError(
        "Estimator must be either RandomForestClassifier or RandomForestRegressor"
    )


def oob_score_roc(estimator: RandomForestClassifier, Y: ArrayLike) -> float:
    """Area under the ROC curve for OOB predictions."""

    if len(np.unique(Y)) > 2:
        labels_pred = estimator.oob_decision_function_
        kwargs: Dict[str, str] = {"multi_class": "ovr", "average": "weighted"}
    else:
        labels_pred = _oob_predictions(estimator)
        kwargs = {"average": "weighted"}
    return roc_auc_score(Y, labels_pred, **kwargs)


def oob_score_nmi(estimator: RandomForestClassifier, Y: ArrayLike) -> float:
    """Normalized mutual information for OOB predictions."""

    labels_pred = _oob_predictions(estimator)
    return normalized_mutual_info_score(Y, labels_pred, average_method="arithmetic")


def oob_score_accuracy(estimator: RandomForestClassifier, Y: ArrayLike) -> float:
    """Accuracy for OOB predictions."""

    labels_pred = _oob_predictions(estimator)
    return accuracy_score(Y, labels_pred)


def oob_score_r2(estimator: RandomForestRegressor, Y: ArrayLike) -> float:
    """Coefficient of determination for OOB predictions."""

    labels_pred = _oob_predictions(estimator)
    return r2_score(Y, labels_pred)


def oob_score_mse(estimator: RandomForestRegressor, Y: ArrayLike) -> float:
    """Mean squared error for OOB predictions."""

    labels_pred = _oob_predictions(estimator)
    return mean_squared_error(Y, labels_pred)


def oob_score_evar(estimator: RandomForestRegressor, Y: ArrayLike) -> float:
    """Explained variance for OOB predictions."""

    labels_pred = _oob_predictions(estimator)
    return explained_variance_score(Y, labels_pred, multioutput="uniform_average")


def _rf_fe_step(
    estimator: RandomForestClassifier | RandomForestRegressor,
    X: ArrayLike,
    Y: ArrayLike,
    n_features_to_keep: int,
    features: ArrayLike,
    fold: int,
    out_dir: str,
    rank_features: bool,
) -> Dict[str, ArrayLike | Dict[str, float]]:
    """Eliminate features step-by-step using OOB metrics."""

    if n_features_to_keep > X.shape[1]:
        raise ValueError(
            "n_features_to_keep cannot be greater than the number of features in X"
        )

    estimator.fit(X, Y)
    feature_importances = estimator.feature_importances_
    rank = np.argsort(feature_importances)[::-1]
    selected = rank[:n_features_to_keep]

    features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, rank_features)
    metrics: Dict[str, float] = {}

    if isinstance(estimator, RandomForestClassifier):
        metrics.update(
            {
                "nmi_score": oob_score_nmi(estimator, Y),
                "accuracy_score": oob_score_accuracy(estimator, Y),
                "roc_auc_score": oob_score_roc(estimator, Y),
            }
        )
    else:
        metrics.update(
            {
                "r2_score": oob_score_r2(estimator, Y),
                "mse_score": oob_score_mse(estimator, Y),
                "explain_var": oob_score_evar(estimator, Y),
            }
        )

    return {
        "n_features": X.shape[1],
        "selected": selected,
        "metrics": metrics,
    }


def _rf_fe(
    estimator: RandomForestClassifier | RandomForestRegressor,
    X: ArrayLike,
    Y: ArrayLike,
    n_features_iter: Iterable[int],
    features: ArrayLike,
    fold: int,
    out_dir: str,
    rank_features: bool,
):
    """Iterate over features to be eliminated step-by-step."""

    if X.shape[1] != len(features):
        raise ValueError("Number of columns in X must match the length of features")

    indices = np.arange(X.shape[1])

    for nf in chain(n_features_iter, [1]):
        payload = _rf_fe_step(estimator, X, Y, nf, features, fold, out_dir, rank_features)

        yield {
            "n_features": payload["n_features"],
            "metrics": payload["metrics"],
            "indices": indices.copy(),
            "selected": payload["selected"],
        }
        indices = indices[payload["selected"]]
        features = features[payload["selected"]]
        X = (
            X[:, payload["selected"]]
            if isinstance(X, np.ndarray)
            else X.iloc[:, payload["selected"]]
        )
