"""
This package provides several functions to run feature elimination for
random forest models, assuming OOB (out-of-bag) is set to True. Three
measurements are calculated for feature selection:

1. Normalized mutual information
2. Accuracy
3. Area under the curve (AUC) ROC curve

Original author: Apuã Paquola.
Edits and maintenance: Kynon Jade Benjamin.
Feature ranking modified from Tarun Katipalli's ranking function.

Additional modifications by Kynon Jade Benjamin to replace class definitions of
functions.
"""

__author__ = 'Kynon Jade Benjamin'

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from ._rank_function import features_rank_fnc

__all__ = [
    "_rf_fe",
    "oob_score_r2",
    "oob_score_nmi",
    "oob_score_roc",
    "oob_score_mse",
    "oob_score_evar",
    "oob_score_accuracy",
]

def _oob_predictions(estimator):
    """
    Extracts out-of-bag (OOB) predictions from random forest
    classifier classes.

    Args:
        estimator: Random forest classifier or regressor object

    Returns:
        np.ndarray: OOB predicted labels

    Raises:
       ValueError: If the estimator is not a RandomForestClassifier or
                   RandomForestRegressor
    """
    if isinstance(estimator, RandomForestClassifier):
        return estimator.classes_[(estimator.oob_decision_function_[:, 1]
                                   > 0.5).astype(int)]
    elif isinstance(estimator, RandomForestRegressor):
        return estimator.oob_prediction_
    else:
        raise ValueError("Estimator must be either RandomForestClassifier or RandomForestRegressor")


def oob_score_roc(estimator, Y):
    """
    Calculates the area under the ROC curve score
    for the OOB predictions.

    Args:
        estimator: Random forest classifier object
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: AUC ROC score
    """
    if len(np.unique(Y)) > 2:
        labels_pred = estimator.oob_decision_function_
        kwargs = {'multi_class': 'ovr', "average": "weighted"}
    else:
        labels_pred = _oob_predictions(estimator)
        kwargs = {"average": "weighted"}
    return roc_auc_score(Y, labels_pred, **kwargs)


def oob_score_nmi(estimator, Y):
    """
    Calculates the normalized mutual information score
    from the OOB predictions.

    Args:
        estimator: Random forest classifier object
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: normalized mutual information score
    """
    labels_pred = _oob_predictions(estimator)
    return normalized_mutual_info_score(Y, labels_pred,
                                        average_method='arithmetic')


def oob_score_accuracy(estimator, Y):
    """
    Calculates the accuracy score from the OOB predictions.

    Args:
        estimator: Random forest classifier object
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: accuracy score
    """
    labels_pred = _oob_predictions(estimator)
    return accuracy_score(Y, labels_pred)


def oob_score_r2(estimator, Y):
    """
    Calculates the R2 score from the OOB predictions.

    Args:
        estimator: Random forest regressor object
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: R2 score
    """
    labels_pred = _oob_predictions(estimator)
    return r2_score(Y, labels_pred)


def oob_score_mse(estimator, Y):
    """
    Calculates the mean square error from the OOB predictions.

    Args:
        estimator: Random forest regressor object
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: mean square error
    """
    labels_pred = _oob_predictions(estimator)
    return mean_squared_error(Y, labels_pred)


def oob_score_evar(estimator, Y):
    """
    Calculates the explained variance score from the OOB predictions.

    Args:
        estimator: Random forest regressor object
        Y: np.ndarray of sample labels from training data set

    Returns:
        float: explained variance score
    """
    labels_pred = _oob_predictions(estimator)
    return explained_variance_score(Y, labels_pred,
                                    multioutput='uniform_average')


def _rf_fe_step(estimator, X, Y, n_features_to_keep, features, fold, out_dir,
               RANK):
    """
    Eliminates features step-by-step.

    Args:
        estimator: Random forest classifier object
        X: a data frame of training data
        Y: np.ndarray of sample labels from training data set
        n_features_to_keep: number of features to keep
        features: np.ndarray of feature names
        fold: current fold
        out_dir: output directory. default '.'
        RANK: Boolean (True/False) to return ranks

    Returns:
        dict: a dictionary containing feature elimination results

    Raises:
        ValueError: If n_features_to_keep is greater than the number of features in X
    """
    if n_features_to_keep > X.shape[1]:
        raise ValueError("n_features_to_keep cannot be greater than the number of features in X")

    estimator.fit(X, Y)
    feature_importances = estimator.feature_importances_
    rank = np.argsort(feature_importances)[::-1] # reverse sort
    selected = rank[:n_features_to_keep]

    features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK)
    result = {
        'n_features': X.shape[1],
        'selected': selected
    }

    if isinstance(estimator, RandomForestClassifier):
        result.update({
            'nmi_score': oob_score_nmi(estimator, Y),
            'accuracy_score': oob_score_accuracy(estimator, Y),
            'roc_auc_score': oob_score_roc(estimator, Y)
        })
    else:
        result.update({
            'r2_score': oob_score_r2(estimator, Y),
            'mse_score': oob_score_mse(estimator, Y),
            'explain_var': oob_score_evar(estimator, Y)
        })
    return result


def _rf_fe(estimator, X, Y, n_features_iter, features, fold, out_dir, RANK):
    """
    Iterates over features to be eliminated step-by-step.

    Args:
        estimator: Random forest classifier or regressor object
        X: DataFrame or np.ndarray of training data
        Y: np.ndarray of sample labels from training data set
        n_features_iter: Iterator for number of features to keep loop
        features: np.ndarray of feature names
        fold (int): Current fold
        out_dir (str): Output directory.
        RANK (bool): Whether to return ranks

    Returns:
        tuple: Feature elimination results for each iteration

    Raises:
        ValueError: If X and features have different number of columns
    """
    if X.shape[1] != len(features):
        raise ValueError("Number of columns in X must match the length of features")

    indices = np.arange(X.shape[1])

    for nf in chain(n_features_iter, [1]):
        p = _rf_fe_step(estimator, X, Y, nf, features, fold, out_dir, RANK)

        if isinstance(estimator, RandomForestClassifier):
            yield p['n_features'], p['nmi_score'], p['accuracy_score'], p['roc_auc_score'], indices
        else:
            yield p['n_features'], p['r2_score'], p['mse_score'], p['explain_var'], indices
        indices = indices[p['selected']]
        features = features[p['selected']]
        X = X[:, p['selected']] if isinstance(X, np.ndarray) else X.iloc[:, p['selected']]
