#!/usr/bin/env python
"""
This script contains the linear model modification of the original
random forest feature elimination package. Instead of Out-of-Bag, it creates
a developmental test set from the training data.

Developed by Kynon Jade Benjamin.
"""

__author__ = 'Kynon J Benjamin'

import numpy as np
import pandas as pd
from itertools import chain
from sklearn.metrics import r2_score
from .rank_function import features_rank_fnc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


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


def dev_score_r2(estimator, X, Y):
    """
    Calculates the R2 score from the DEV predictions.

    Args:
    estimator: linear model regressor object
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
    rank = test_indices[np.argsort(estimator.feature_importances_)]
    rank = rank[::-1] # reverse sort
    selected = rank[0:n_features_to_keep]
    features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK)
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
    estimator: regression linear model object
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
        yield p['n_features'], p['r2_score'], p['mse_score'], p['explain_var'], indices
        indices = indices[p['selected']]
        X = X[:, p['selected']]
        features = features[p['selected']]
