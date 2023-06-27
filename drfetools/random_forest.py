"""
This package has several function to run feature elimination for random forest
classifier. Specifically, Out-of-Bag (OOB) must be set to True. Three
measurements are calculated for feature selection.

1. Normalized mutual information
2. Accuracy
3. Area under the curve (AUC) ROC curve

Original author Apua Paquola.
Edits: Kynon Jade Benjamin
Feature ranking modified from Tarun Katipalli ranking function.

Additional modification by KJB to replace class definitions of functions.
"""

__author__ = 'Kynon Jade Benjamin'

import numpy as np
from itertools import chain
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from .rank_function import features_rank_fnc
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import normalized_mutual_info_score


def oob_predictions(estimator):
    """
    Extracts out-of-bag (OOB) predictions from random forest
    classifier classes.

    Args:
    estimator: Random forest classifier or regressor object

    Yields:
    vector: OOB predicted labels
    """
    if isinstance(estimator, RandomForestClassifier):
        return estimator.classes_[(estimator.oob_decision_function_[:, 1]
                                   > 0.5).astype(int)]
    else:
        return estimator.oob_prediction_


def oob_score_roc(estimator, Y):
    """
    Calculates the area under the ROC curve score
    for the OOB predictions.

    Args:
    estimator: Random forest classifier object
    Y: a vector of sample labels from training data set

    Yields:
    float: AUC ROC score
    """
    if len(np.unique(Y)) > 2:
        labels_pred = estimator.oob_decision_function_
        kwargs = {'multi_class': 'ovr', "average": "weighted"}
    else:
        labels_pred = oob_predictions(estimator)
        kwargs = {"average": "weighted"}
    return roc_auc_score(Y, labels_pred, **kwargs)


def oob_score_nmi(estimator, Y):
    """
    Calculates the normalized mutual information score
    from the OOB predictions.

    Args:
    estimator: Random forest classifier object
    Y: a vector of sample labels from training data set

    Yields:
    float: normalized mutual information score
    """
    labels_pred = oob_predictions(estimator)
    return normalized_mutual_info_score(Y, labels_pred,
                                        average_method='arithmetic')


def oob_score_accuracy(estimator, Y):
    """
    Calculates the accuracy score from the OOB predictions.

    Args:
    estimator: Random forest classifier object
    Y: a vector of sample labels from training data set

    Yields:
    float: accuracy score
    """
    labels_pred = oob_predictions(estimator)
    return accuracy_score(Y, labels_pred)


def oob_score_r2(estimator, Y):
    """
    Calculates the R2 score from the OOB predictions.

    Args:
    estimator: Random forest regressor object
    Y: a vector of sample labels from training data set

    Yields:
    float: R2 score
    """
    labels_pred = oob_predictions(estimator)
    return r2_score(Y, labels_pred)


def oob_score_mse(estimator, Y):
    """
    Calculates the mean square error from the OOB predictions.

    Args:
    estimator: Random forest regressor object
    Y: a vector of sample labels from training data set

    Yields:
    float: mean square error
    """
    labels_pred = oob_predictions(estimator)
    return mean_squared_error(Y, labels_pred)


def oob_score_evar(estimator, Y):
    """
    Calculates the explained variance score from the OOB predictions.

    Args:
    estimator: Random forest regressor object
    Y: a vector of sample labels from training data set

    Yields:
    float: explained variance score
    """
    labels_pred = oob_predictions(estimator)
    return explained_variance_score(Y, labels_pred,
                                    multioutput='uniform_average')


def rf_fe_step(estimator, X, Y, n_features_to_keep, features, fold, out_dir,
               RANK):
    """
    Apply random forest to training data, rank features, conduct feature
    elimination.

    Args:
    estimator: Random forest classifier object
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    n_features_to_keep: number of features to keep
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    dict: a dictionary with number of features, normalized mutual
          information score, accuracy score, auc roc score and selected features
    """
    # kwargs = {'random_state': 13, 'test_size': 0.20}
    # X1, X2, Y1, Y2 = train_test_split(X, Y, **kwargs)
    assert n_features_to_keep <= X.shape[1]
    estimator.fit(X, Y)
    test_indices = np.array(range(X.shape[1]))
    #res = permutation_importance(estimator, X2, Y2, n_jobs=-1, random_state=13)
    #rank = test_indices[res.importances_mean.argsort()]
    rank = test_indices[np.argsort(estimator.feature_importances_)]
    rank = rank[::-1] # reverse sort
    selected = rank[0:n_features_to_keep]
    features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK)
    if isinstance(estimator, RandomForestClassifier):
        return {'n_features': X.shape[1],
                'nmi_score': oob_score_nmi(estimator, Y),
                'accuracy_score': oob_score_accuracy(estimator, Y),
                'roc_auc_score': oob_score_roc(estimator, Y),
                'selected': selected}
    else:
        return {'n_features': X.shape[1],
                'r2_score': oob_score_r2(estimator, Y),
                'mse_score': oob_score_mse(estimator, Y),
                'explain_var': oob_score_evar(estimator, Y),
                'selected': selected}


def rf_fe(estimator, X, Y, n_features_iter, features, fold, out_dir, RANK):
    """
    Iterate over features to by eliminated by step.

    Args:
    estimator: Random forest classifier or regressor object
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    n_features_iter: iterator for number of features to keep loop
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    list: a list with number of features, normalized mutual
          information score, accuracy score, auc roc curve and array of the
          indices for features to keep
    """
    indices = np.array(range(X.shape[1]))
    for nf in chain(n_features_iter, [1]):
        p = rf_fe_step(estimator, X, Y, nf, features, fold, out_dir, RANK)
        if isinstance(estimator, RandomForestClassifier):
            yield p['n_features'], p['nmi_score'], p['accuracy_score'], p['roc_auc_score'], indices
        else:
            yield p['n_features'], p['r2_score'], p['mse_score'], p['explain_var'], indices
        indices = indices[p['selected']]
        features = features[p['selected']]
        if type(X) == np.ndarray:
            X = X[:, p['selected']]
        else:
            X = X.iloc[:, p['selected']]
