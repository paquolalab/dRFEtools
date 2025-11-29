#!/usr/bin/env python
"""
This package has several function to run dynamic recursive feature elimination
(dRFE) for random forest and linear model classifier and regression models. For
random forest, it assumes Out-of-Bag (OOB) is set to True. For linear models,
it generates a developmental set. For both classification and regression, three
measurements are calculated for feature selection:

Classification:
1. Normalized mutual information
2. Accuracy
3. Area under the curve (AUC) ROC curve

Regression:
1. R2 (this can be negative if model is arbitrarily worse)
2. Explained variance
3. Mean squared error

The package has been split in to four additional scripts for:
1. Out-of-bag dynamic RFE metrics (AP)
2. Validation set dynamic RFE metrics (KJB)
3. Rank features function (TK)
4. Lowess core + peripheral selection (KJB)

Original author Apuã Paquola (AP).
Edits and package management by Kynon Jade Benjamin (KJB)
Feature ranking modified from Tarun Katipalli (TK) ranking function.
"""
__author__ = 'Apuã Paquola'

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    ggplot,
    geom_point,
    geom_vline,
    labs,
    scale_x_log10,
    theme_light,
)
from warnings import filterwarnings
from matplotlib import MatplotlibDeprecationWarning

from .lowess import _cal_lowess, extract_max_lowess, extract_peripheral_lowess, optimize_lowess_plot
from .scoring import _regr_fe, _rf_fe

filterwarnings("ignore", category=MatplotlibDeprecationWarning)
filterwarnings('ignore', category=UserWarning, module='plotnine.*')
filterwarnings('ignore', category=DeprecationWarning, module='plotnine.*')

__all__ = [
    "rf_rfe",
    "dev_rfe",
    "plot_metric",
    "plot_with_lowess_vline",
]

def _n_features_iter(nf: int, keep_rate: float) -> int:
    """
    Determines the features to keep.

    Args:
        nf (int): Current number of features
        keep_rate (float): Percentage of features to keep

    Returns:
        int: Number of features to keep
    """
    while nf != 1:
        nf = max(1, int(nf * keep_rate))
        yield nf


def rf_rfe(estimator, X, Y, features, fold, out_dir='.', elimination_rate=0.2,
           RANK=True):
    """
    Runs random forest feature elimination step over iterator process.

    Args:
        estimator: Random forest classifier object
        X (DataFrame): Training data
        Y (array-like): Sample labels from training data set
        features (array-like): Feature names
        fold (int): Current fold
        out_dir (str): Output directory. Default '.'
        elimination_rate (float): Percent rate to reduce feature list. Default 0.2
        RANK (bool): Whether to perform feature ranking. Default True

    Returns:
        tuple: Dictionary with elimination results, and first elimination step results
    """
    if not 0 < elimination_rate < 1:
        raise ValueError("elimination_rate must be between 0 and 1")

    d = {}
    pfirst = None
    keep_rate = 1 - elimination_rate

    for p in _rf_fe(estimator, X, Y, _n_features_iter(X.shape[1], keep_rate),
                    features, fold, out_dir, RANK):
        if pfirst is None:
            pfirst = p
        d[p[0]] = p

    return d, pfirst


def dev_rfe(estimator, X, Y, features, fold, out_dir='.', elimination_rate=0.2,
            dev_size=0.2, RANK=True, SEED=False):
    """
    Runs recursive feature elimination for linear model step over iterator
    process assuming developmental set is needed.

    Args:
        estimator: Classifier or regression linear model object
        X (DataFrame): Training data
        Y (array-like): Sample labels from training data set
        features (array-like): Feature names
        fold (int): Current fold
        out_dir (str): Output directory. Default '.'
        elimination_rate (float): Percent rate to reduce feature list. Default 0.2
        dev_size (float): Developmental set size. Default 0.2
        RANK (bool): Run feature ranking. Default True
        SEED (bool): Use fixed random state. Default False

    Returns:
        tuple: Dictionary with elimination results, and first elimination step results
    """
    if not 0 < elimination_rate < 1 or not 0 < dev_size < 1:
        raise ValueError("elimination_rate and dev_size must be between 0 and 1")

    d = {}
    pfirst = None
    keep_rate = 1 - elimination_rate

    for p in _regr_fe(estimator, X, Y, _n_features_iter(X.shape[1], keep_rate),
                      features, fold, out_dir, dev_size, SEED, RANK):
        if pfirst is None:
            pfirst = p
        d[p[0]] = p

    return d, pfirst
