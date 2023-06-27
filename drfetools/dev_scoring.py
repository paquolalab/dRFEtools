"""
This script contains the feature elimination step using a
validation set (developmental) to train the dynamic feature
elimination.

Developed by Kynon Jade Benjamin.
"""

__author__ = 'Kynon J Benjamin'

import numpy as np
from itertools import chain
from sklearn.metrics import r2_score
from sklearn.base import is_classifier
from .rank_function import features_rank_fnc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score, accuracy_score


def cal_feature_imp(estimator):
    """
    Adds feature importance to a scikit-learn estimator. This is similar
    to random forest output feature importances output.

    This function also checks dimensions to handle multi-class inputs.
    """
    if estimator.coef_.ndim == 1:
        estimator.feature_importances_ = np.abs(estimator.coef_).flatten()
    else:
        estimator.feature_importances_ = np.amax(np.abs(estimator.coef_),
                                                 axis=0).flatten()
    return estimator


def dev_predictions(estimator, X):
    """
    Extracts predictions using a development fold for linear model
    regression.

    Args:
    estimator: linear model regression object
    X: a data frame of normalized values from developmental dataset

    Yields:
    vector: Development set predicted labels
    """
    return estimator.predict(X)


def dev_score_roc(estimator, X, Y):
    """
    Calculates the area under the ROC curve score
    for the DEV predictions.

    Args:
    estimator: linear model classifier object
    X: a data frame of normalized values from developmental dataset
    Y: a vector of sample labels from training data set

    Yields:
    float: AUC ROC score
    """
    if len(np.unique(Y)) > 2:
        labels_pred = estimator.predict_proba(X)
        kwargs = {'multi_class': 'ovr', "average": "weighted"}
    else:
        labels_pred = dev_predictions(estimator, X)
        kwargs = {"average": "weighted"}
    return roc_auc_score(Y, labels_pred, **kwargs)


def dev_score_nmi(estimator, X, Y):
    """
    Calculates the normalized mutual information score
    from the DEV predictions.

    Args:
    estimator: linear model classifier object
    X: a data frame of normalized values from developmental dataset
    Y: a vector of sample labels from training data set

    Yields:
    float: normalized mutual information score
    """
    labels_pred = dev_predictions(estimator, X)
    return normalized_mutual_info_score(Y, labels_pred,
                                        average_method='arithmetic')


def dev_score_accuracy(estimator, X, Y):
    """
    Calculates the accuracy score from the DEV predictions.

    Args:
    estimator: linear model classifier object
    X: a data frame of normalized values from developmental dataset
    Y: a vector of sample labels from training data set

    Yields:
    float: accuracy score
    """
    labels_pred = dev_predictions(estimator, X)
    return accuracy_score(Y, labels_pred)


def dev_score_r2(estimator, X, Y):
    """
    Calculates the R2 score from the DEV predictions.

    Args:
    estimator: linear model regressor object
    X: a data frame of normalized values from developmental dataset
    Y: a vector of sample labels from training data set

    Yields:
    float: R2 score
    """
    labels_pred = dev_predictions(estimator, X)
    return r2_score(Y, labels_pred)


def dev_score_mse(estimator, X, Y):
    """
    Calculates the mean square error from the DEV predictions.

    Args:
    estimator: linear model regressor object
    X: a data frame of normalized values from developmental dataset
    Y: a vector of sample labels from training data set

    Yields:
    float: mean square error
    """
    labels_pred = dev_predictions(estimator, X)
    return mean_squared_error(Y, labels_pred)


def dev_score_evar(estimator, X, Y):
    """
    Calculates the explained variance score from the DEV predictions.

    Args:
    estimator: linear model regressor object
    X: a data frame of normalized values from developmental dataset
    Y: a vector of sample labels from training data set

    Yields:
    float: explained variance score
    """
    labels_pred = dev_predictions(estimator, X)
    return explained_variance_score(Y, labels_pred,
                                    multioutput='uniform_average')


def regr_fe_step(estimator, X, Y, n_features_to_keep, features,
                 fold, out_dir, dev_size, SEED, RANK):
    """
    Split training data into developmental dataset and apply estimator
    to developmental dataset, rank features, and conduct feature
    elimination, single steps.

    Args:
    estimator: regression linear model object
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    n_features_to_keep: number of features to keep
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'
    dev_size: developmental size. default '0.20'
    SEED: random state. default 'True'
    RANK: run feature ranking. default 'True'

    Yields:
    dict: a dictionary with number of features, r2 score, mean square error,
          expalined variance, and selected features
    """
    kwargs = {'random_state': 13,
              'test_size': dev_size} if SEED else {'test_size': dev_size}
    X1, X2, Y1, Y2 = train_test_split(X, Y, **kwargs)
    # print(X.shape[1], n_features_to_keep)
    assert n_features_to_keep <= X1.shape[1]
    estimator.fit(X1, Y1)
    test_indices = np.array(range(X1.shape[1]))
    #res = permutation_importance(estimator, X2, Y2, n_jobs=-1, random_state=13)
    #rank = test_indices[res.importances_mean.argsort()]
    estimator = cal_feature_imp(estimator)
    rank = test_indices[np.argsort(estimator.feature_importances_)]
    rank = rank[::-1] # reverse sort
    selected = rank[0:n_features_to_keep]
    features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK)
    if is_classifier(estimator):
        return {"n_features": X1.shape[1],
                "nmi_score": dev_score_nmi(estimator, X2, Y2),
                "accuracy_score": dev_score_accuracy(estimator, X2, Y2),
                "roc_auc_score": dev_score_roc(estimator, X2, Y2),
                "selected": selected}
    else:
        return {'n_features': X1.shape[1],
                'r2_score': dev_score_r2(estimator, X2, Y2),
                'mse_score': dev_score_mse(estimator, X2, Y2),
                'explain_var': dev_score_evar(estimator, X2, Y2),
                'selected': selected}


def regr_fe(estimator, X, Y, n_features_iter, features, fold, out_dir,
            dev_size, SEED, RANK):
    """
    Iterate over features to by eliminated by step.

    Args:
    estimator: Non random forest classifier or regressor object
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    n_features_iter: iterator for number of features to keep loop
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'
    dev_size: developmental size. default '0.20'
    SEED: random state. default 'True'
    RANK: run feature ranking. default 'True'

    Yields:
    list: a list with number of features, r2 score, mean square error,
          expalined variance, and array of the indices for features to keep
    """
    indices = np.array(range(X.shape[1]))
    for nf in chain(n_features_iter, [1]):
        p = regr_fe_step(estimator, X, Y, nf, features, fold,
                         out_dir, dev_size, SEED, RANK)
        if is_classifier(estimator):
            yield p["n_features"], p["nmi_score"], p["accuracy_score"], p["roc_auc_score"], indices
        else:
            yield p['n_features'], p['r2_score'], p['mse_score'], p['explain_var'], indices
        indices = indices[p['selected']]
        features = features[p['selected']]
        if type(X) == np.ndarray:
            X = X[:, p['selected']]
        else:
            X = X.iloc[:, p['selected']]
