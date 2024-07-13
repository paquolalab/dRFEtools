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
from plotnine import *
from warnings import filterwarnings
from matplotlib import MatplotlibDeprecationWarning

from ._dev_scoring import _regr_fe
from ._random_forest import _rf_fe
from ._lowess_redundant import (
    _cal_lowess,
    extract_max_lowess,
    optimize_lowess_plot,
    extract_peripheral_lowess
)

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


def _save_plot(p, fn, width=7, height=7):
    '''
    Save plot as svg, png, and pdf with specific label and dimension.

    Args:
        p: Plot object
        fn (str): File name (without extension)
        width (int): Plot width. Default 7
        height (int): Plot height. Default 7
    '''
    for ext in ['.svg', '.png', '.pdf']:
        p.save(fn + ext, width=width, height=height)


def plot_metric(d, fold, output_dir, metric_name, y_label):
    """
    Plot feature elimination results for normalized mutual information.

    Args:
        d (dict): Feature elimination class dictionary
        fold (int): Current fold
        output_dir (str): Output directory
        metric_name (str): Name of the metric (used for file naming)
        y_label (str): Label for y-axis

    Returns:
        None: Saves plot files and prints the plot
    """
    if metric_name in ["nmi", "r2"]:
        key_num = 1
    elif metric_name in ["roc", "mse"]:
        key_num = 2
    elif metric_name in ["acc", "evar"]:
        key_num = 3
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")
    df_elim = pd.DataFrame([{'n features': k,
                             y_label: d[k][key_num]} for k in d.keys()])

    gg = (ggplot(df_elim, aes(x='n features', y=y_label))
          + geom_point()
          + scale_x_log10()
          + theme_light()
          + labs(x="Number of features", y=y_label))

    outfile = f"{output_dir}/{metric_name}_fold_{fold}"
    _save_plot(gg, outfile)
    print(gg)


def plot_with_lowess_vline(d, fold, output_dir, frac=3/10, step_size=0.05,
                           classify=True, multi=False, acc=False):
    """
    Plot the LOWESS smoothing plot for RFE with lines annotating set selection.

    Args:
        d (dict): Feature elimination class dictionary
        fold (int): Current fold
        output_dir (str): Output directory
        frac (float): Fraction for LOWESS smoothing. Default 3/10
        step_size (float): Step size for peripheral feature extraction. Default 0.05
        classify (bool): Whether it's a classification task. Default True
        multi (bool): Whether it's a multi-class classification. Default False
        acc (bool): Whether to use accuracy for optimization. Default False

    Returns:
        None: Saves plot files and prints the plot
    """
    if classify:
        label = 'ROC AUC' if multi else 'Accuracy' if acc else 'NMI'
    else:
        label = 'R2'

    _, max_feat_log10 = extract_max_lowess(d, frac, multi, acc)
    x, y, z, _, _ = _cal_lowess(d, frac, multi, acc)
    df_elim = pd.DataFrame({'X': x, 'Y': y})
    _, lo = extract_max_lowess(d, frac, multi, acc)
    _, l1 = extract_peripheral_lowess(d, frac, step_size, multi, acc)

    gg = (ggplot(df_elim, aes(x='X', y='Y'))
          + geom_point(color='blue')
          + geom_vline(xintercept=lo, color='blue', linetype='dashed')
          + geom_vline(xintercept=l1, color='orange', linetype='dashed')
          + scale_x_log10()
          + labs(x='log10(N Features)', y=label)
          + theme_light())

    print(gg)
    outfile = f"{output_dir}/{label.replace(' ', '_')}_log10_dRFE_fold_{fold}"
    _save_plot(gg, outfile)
