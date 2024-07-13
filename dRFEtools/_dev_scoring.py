"""
This script contains the feature elimination step using a
validation set (developmental) to train the dynamic feature
elimination.

Developed by Kynon Jade Benjamin.
"""

__author__ = 'Kynon J Benjamin'

import numpy as np
from itertools import chain
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    explained_variance_score,
    normalized_mutual_info_score
)
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split

from ._rank_function import features_rank_fnc

__all__ = [
    "_regr_fe",
    "dev_score_r2",
    "dev_score_roc",
    "dev_score_mse",
    "dev_score_nmi",
    "dev_score_evar",
    "dev_score_accuracy",
]

def _dev_cal_feature_imp(estimator):
    """
    Adds feature importance to a scikit-learn estimator.

    Args:
        estimator: A scikit-learn estimator

    Returns:
        estimator: The estimator with added feature_importances_ attribute

    Raises:
        AttributeError: If the estimator is not supported

    Notes:
        This function also checks dimensions to handle multi-class inputs.
    """
    if hasattr(estimator, "feature_log_prob_"):
        estimator.feature_importances_ = np.exp(np.amax(estimator.feature_log_prob_, axis=0)).flatten()
    elif hasattr(estimator, "feature_importances_"):
        pass
    elif hasattr(estimator, "coef_"):
        estimator.feature_importances_ = np.abs(estimator.coef_).flatten() if estimator.coef_.ndim == 1 else np.amax(np.abs(estimator.coef_), axis=0).flatten()
    elif hasattr(estimator, "dual_coef_"):
        estimator.feature_importances_ = np.amax(np.abs(estimator.dual_coef_),
                                                 axis=0).flatten()
    else:
        raise AttributeError("model not supported")
    return estimator


def _get_feature_importances(estimator):
    """
    Retrieve and aggregate (ndim > 1) feature importance
    from scikit-learn estimator.

    This function also checks dimensions to handle multi-class inputs.

    Args:
        estimator : A scikit-learn estimator

    Returns:
        importances (np.ndarray) : The features importances

    Raises:
        ValueError: If the estimator doesn't have coef_ or feature_importances_ attribute
    """
    if hasattr(estimator, "coef_"):
        importances = estimator.coef_
        return np.linalg.norm(importances, axis=0, ord=1).flatten() if importances.ndim > 1 else np.abs(importances).flatten()
    elif hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    else:
        raise ValueError(f"The estimator {estimator.__class__.__name__} should have `coef_` or `feature_importances_` attribute.")


def _dev_predictions(estimator, X):
    """
    Extracts predictions using a development fold.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset

    Returns:
        np.ndarray: Predicted labels or values
    """
    return estimator.predict(X)


def dev_score_roc(estimator, X, Y):
    """
    Calculates the area under the ROC curve score
    for the DEV predictions.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: AUC ROC score
    """
    if len(np.unique(Y)) > 2:
        labels_pred = estimator.predict_proba(X)
        kwargs = {'multi_class': 'ovr', "average": "weighted"}
    else:
        labels_pred = _dev_predictions(estimator, X)
        kwargs = {"average": "weighted"}
    return roc_auc_score(Y, labels_pred, **kwargs)


def dev_score_nmi(estimator, X, Y):
    """
    Calculates the normalized mutual information score
    from the DEV predictions.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: normalized mutual information score
    """
    labels_pred = _dev_predictions(estimator, X)
    return normalized_mutual_info_score(Y, labels_pred,
                                        average_method='arithmetic')


def dev_score_accuracy(estimator, X, Y):
    """
    Calculates the accuracy score from the DEV predictions.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: accuracy score
    """
    labels_pred = _dev_predictions(estimator, X)
    return accuracy_score(Y, labels_pred)


def dev_score_r2(estimator, X, Y):
    """
    Calculates the R2 score from the DEV predictions.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: R2 score
    """
    labels_pred = _dev_predictions(estimator, X)
    return r2_score(Y, labels_pred)


def dev_score_mse(estimator, X, Y):
    """
    Calculates the mean square error from the DEV predictions.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: mean square error
    """
    labels_pred = _dev_predictions(estimator, X)
    return mean_squared_error(Y, labels_pred)


def dev_score_evar(estimator, X, Y):
    """
    Calculates the explained variance score from the DEV predictions.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: explained variance score
    """
    labels_pred = _dev_predictions(estimator, X)
    return explained_variance_score(Y, labels_pred,
                                    multioutput='uniform_average')


def _regr_fe_step(estimator, X, Y, n_features_to_keep, features,
                  fold, out_dir, dev_size, SEED, RANK):
    """
    Performs a single step of feature elimination using a development dataset.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset
        Y: np.ndarray of sample labels from training data set
        n_features_to_keep: Number of features to keep
        features: np.ndarray of feature names
        fold (int): Current fold
        out_dir (str): Output directory
        dev_size (flaot): Size of development set
        SEED (bool): Whether to use a fixed random state
        RANK (bool): Whether to perform feature ranking

    Returns:
        dict: A dictionary containing feature elimination results

    Raises:
        ValueError: If n_features_to_keep is greater than the number of features in X
    """
    if n_features_to_keep > X1.shape[1]:
        raise ValueError("n_features_to_keep cannot be greater than the number of features in X")

    kwargs = {'random_state': 13,
              'test_size': dev_size} if SEED else {'test_size': dev_size}
    X1, X2, Y1, Y2 = train_test_split(X, Y, **kwargs)

    estimator.fit(X1, Y1)
    estimator.feature_importances_ = _get_feature_importances(estimator)

    rank = np.argsort(estimator.feature_importances_)[::-1] # reverse sort
    selected = rank[:n_features_to_keep]

    features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK)
    result = {
        'n_features': X1.shape[1],
        'selected': selected
    }
    if is_classifier(estimator):
        result.update({
            "nmi_score": dev_score_nmi(estimator, X2, Y2),
            "accuracy_score": dev_score_accuracy(estimator, X2, Y2),
            "roc_auc_score": dev_score_roc(estimator, X2, Y2)
        })
    else:
        result.update({
            'r2_score': dev_score_r2(estimator, X2, Y2),
            'mse_score': dev_score_mse(estimator, X2, Y2),
            'explain_var': dev_score_evar(estimator, X2, Y2)
        })
    return result


def _regr_fe(estimator, X, Y, n_features_iter, features, fold, out_dir,
             dev_size, SEED, RANK):
    """
    Iterate over features to by eliminated by step.

    Args:
        estimator: A scikit-learn estimator
        X: DataFrame or np.ndarray of developmental dataset
        Y: np.ndarray of sample labels from training data set
        n_features_iter: Iterator for number of features to keep loop
        features: np.ndarray of feature names
        fold (int): Current fold
        out_dir (str): Output directory
        dev_size (flaot): Size of development set
        SEED (bool): Whether to use a fixed random state
        RANK (bool): Whether to perform feature ranking

    Returns:
        tuple: Feature elimination results for each iteration

    Raises:
        ValueError: If X and features have different number of columns
    """
    if X.shape[1] != len(features):
        raise ValueError("Number of columns in X must match the length of features")
    indices = np.arange(X.shape[1])

    for nf in chain(n_features_iter, [1]):
        p = _regr_fe_step(estimator, X, Y, nf, features, fold,
                          out_dir, dev_size, SEED, RANK)

        if is_classifier(estimator):
            yield p["n_features"], p["nmi_score"], p["accuracy_score"], p["roc_auc_score"], indices
        else:
            yield p['n_features'], p['r2_score'], p['mse_score'], p['explain_var'], indices
        indices = indices[p['selected']]
        features = features[p['selected']]
        X = X[:, p['selected']] if isinstance(X, np.ndarray) else X.iloc[:, p['selected']]
