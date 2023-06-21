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
from .dev_scoring import *
from .random_forest import *
from .lowess_redundant import *
from warnings import filterwarnings
from matplotlib.cbook import mplDeprecation

filterwarnings("ignore", category=mplDeprecation)
filterwarnings('ignore', category=UserWarning, module='plotnine.*')
filterwarnings('ignore', category=DeprecationWarning, module='plotnine.*')


def n_features_iter(nf, keep_rate):
    """
    Determines the features to keep.

    Args:
    nf: current number of features
    keep_rate: percentage of features to keep

    Yields:
    int: number of features to keep
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
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'
    elimination_rate: percent rate to reduce feature list. default .2

    Yields:
    dict: a dictionary with number of features, normalized mutual
          information score, accuracy score, auc roc curve and array of the
          indexes for features to keep
    """
    d = dict()
    pfirst = None
    keep_rate = 1-elimination_rate
    for p in rf_fe(estimator, X, Y, n_features_iter(X.shape[1], keep_rate),
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
    estimator: classifier or regression linear model object
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'
    elimination_rate: percent rate to reduce feature list. default .2
    dev_size: developmental set size. default '0.20'
    RANK: run feature ranking, default 'True'
    SEED: random state. default 'True'

    Yields:
    dict: a dictionary with number of features, r2 score, mean square error,
          expalined variance, and array of the indices for features to keep
    """
    d = dict()
    pfirst = None
    keep_rate = 1-elimination_rate
    for p in regr_fe(estimator, X, Y, n_features_iter(X.shape[1], keep_rate),
                     features, fold, out_dir, dev_size, SEED, RANK):
        if pfirst is None:
            pfirst = p
        d[p[0]] = p
    return d, pfirst


def save_plot(p, fn, width=7, height=7):
    '''Save plot as svg, png, and pdf with specific label and dimension.'''
    for ext in ['.svg', '.png', '.pdf']:
        p.save(fn+ext, width=width, height=height)


def plot_nmi(d, fold, output_dir):
    """
    Plot feature elimination results for normalized mutual information.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by NMI, automatically saves files as pdf, png, and
           svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'normalized mutual information':d[k][1]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='normalized mutual information'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/nmi_fold_%d" % (fold))
    print(gg)


def plot_roc(d, fold, output_dir):
    """
    Plot feature elimination results for AUC ROC curve.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by AUC, automatically saves files as pdf, png, and
           svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'ROC AUC':d[k][3]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='ROC AUC'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/roc_fold_%d" % (fold))
    print(gg)


def plot_acc(d, fold, output_dir):
    """
    Plot feature elimination results for accuracy.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by accuracy, automatically saves files as pdf, png,
           and svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'Accuracy':d[k][2]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='Accuracy'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/acc_fold_%d" % (fold))
    print(gg)


def plot_r2(d, fold, output_dir):
    """
    Plot feature elimination results for R2.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by R2, automatically saves files as pdf, png, and svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'R2':d[k][1]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='R2'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/r2_fold_%d" % (fold))
    print(gg)


def plot_mse(d, fold, output_dir):
    """
    Plot feature elimination results for mean square error curve.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by mean square error, automatically saves files as
           pdf, png, and svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'Mean Square Error':d[k][2]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='Mean Square Error'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/mse_fold_%d" % (fold))
    print(gg)


def plot_evar(d, fold, output_dir):
    """
    Plot feature elimination results for explained variance.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by explained variance, automatically saves files as
           png, pdf, and svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'Explained Variance':d[k][3]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='Explained Variance'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/evar_fold_%d" % (fold))
    print(gg)


def plot_with_lowess_vline(d, fold, output_dir, frac=3/10, step_size=0.05,
                           classify=True, multi=False, acc=False):
    if classify:
        if multi:
            label = "ROC AUC"
        elif acc:
            label = "Accuracy"
        else:
            label = 'NMI'
    else:
        label = 'R2'
    _, max_feat_log10 = extract_max_lowess(d, frac, multi, acc)
    x,y,z,_,_ = cal_lowess(d, frac, multi, acc)
    df_elim = pd.DataFrame({'X': x, 'Y': y})
    _,lo = extract_max_lowess(d, frac, multi, acc)
    _,l1 = extract_peripheral_lowess(d, frac, step_size, multi, acc)
    gg = ggplot(df_elim, aes(x='X', y='Y')) + geom_point(color='blue') + \
        geom_vline(xintercept=lo, color='blue', linetype='dashed') + \
        geom_vline(xintercept=l1, color='orange', linetype='dashed') +\
        scale_x_log10() + labs(x='log10(N Features)', y=label) +\
        theme_light()
    print(gg)
    save_plot(gg, "%s/%s_log10_dRFE_fold_%d" %
              (output_dir,label.replace(" ", "_"),fold))

# def plot_scores(d, alpha, output_dir):
#     df_nmi = pd.DataFrame([{'n features':k, 'Score':d[k][1]} for k in d.keys()])
#     df_nmi['Type'] = 'NMI'
#     df_acc = pd.DataFrame([{'n features':k, 'Score':d[k][2]} for k in d.keys()])
#     df_acc['Type'] = 'Acc'
#     df_roc = pd.DataFrame([{'n features':k, 'Score':d[k][3]} for k in d.keys()])
#     df_roc['Type'] = 'ROC'
#     df_elim = pd.concat([df_nmi, df_acc, df_roc], axis=0)
#     gg = ggplot(df_elim, aes(x='n features', y='Score', color='Type'))\
#         + geom_point() + scale_x_log10() + theme_light()
#     gg.save(output_dir+"/scores_wgt_%.2f.png" % (alpha))
#     gg.save(output_dir+"/scores_wgt_%.2f.svg" % (alpha))
#     print(gg)
