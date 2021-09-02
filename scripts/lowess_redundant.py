#!/usr/bin/env python
"""
This script contains functions to calculate redundant features based
on a lowess fitted curve. Uses a log10 transformation of feature inputs.
Also contains the optimization plot function for manual parameter
optimization for lowess.

Developed by Kynon Jade Benjamin.
"""

__author__ = 'Kynon J Benjamin'

import functools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import interpolate
import matplotlib.pyplot as plt


@functools.lru_cache()
def run_lowess(xnew, ynew, frac):
    lowess = sm.nonparametric.lowess
    z = lowess(ynew, xnew, frac=frac, it=10)
    return z


def array_to_tuple(np_array):
    try:
        return tuple(array_to_tuple(_) for _ in np_array)
    except TypeError:
        return np_array


def get_elim_df_ordered(d, multi):
    if multi:
        col = 3
    else:
        col = 1
    df_elim = pd.DataFrame([{'x':k, 'y':d[k][col]} for k in d.keys()])\
                .sort_values('x')
    df_elim['log10_x'] = np.log(df_elim['x']+0.5)
    return df_elim


def cal_lowess(d, frac, multi):
    """
    Calculated lowess curve.

    Args:
    d: Dictionary from dRFE
    frac: Fraction for lowess smoothing. Default 3/10.

    Yields:
    variables: n_features (x); model validation values (y; accuracy for
    classification and r2 for regression); lowess curve (z)
    """
    df_elim = get_elim_df_ordered(d, multi)
    x = df_elim['log10_x'].values
    y = df_elim['y'].values
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(x.min(), x.max(), num=5001, endpoint=True)
    ynew = interpolate.splev(xnew, tck, der=0)
    z = run_lowess(array_to_tuple(xnew), array_to_tuple(ynew), frac)
    return x,y,z,xnew,ynew


def cal_lowess_rate_log10(d, frac=3/10, multi=False):
    """
    Calculate rate of change on the lowess fitted curve with log10
    transformation.

    Args:
    d: Dictionary from dRFE
    frac: Fraction for lowess smoothing. Default 3/10.

    Yields:
    data frame: Data frame with n_featuers, lowess value, and
    rate of change (DxDy)
    """
    _,_,z,_,_ = cal_lowess(d, frac, multi)
    dfz = pd.DataFrame(z)
    list_values = []
    for xx in range(z.shape[0]-1):
        dx = dfz.iloc[xx, 0] - dfz.iloc[xx+1, 0]
        dy = dfz.iloc[xx, 1] - dfz.iloc[xx+1, 1]
        list_values.append(dx/dy)
    pts = dfz.rename(columns={0:'Features', 1:'LOWESS'})
    pts = pts.drop(pts.index[0])
    pts['DxDy'] = list_values
    return pts


def extract_max_lowess(d, frac=3/10, multi=False):
    """
    Extract max features based on rate of change of log10
    transformed lowess fit curve.

    Args:
    d: Dictionary from dRFE
    frac: Fraction for lowess smoothing. Default 3/10.

    Yields:
    int: number of redundant features
    """
    _,_,z,xnew,ynew = cal_lowess(d, frac, multi)
    df_elim = get_elim_df_ordered(d, multi)
    df_lowess = pd.DataFrame({'X': xnew, 'Y': ynew,
                              'xprime': pd.DataFrame(z)[0],
                              'yprime': pd.DataFrame(z)[1]})
    val = df_lowess[(df_lowess['yprime'] == max(df_lowess.yprime))].X.values
    closest_val = min(df_elim['log10_x'].values, key=lambda x: abs(x - val))
    return df_elim[(df_elim['log10_x'] == closest_val)].x.values[0], closest_val


def extract_redundant_lowess(d, frac=3/10, step_size=0.05, multi=False):
    """
    Extract redundant features based on rate of change of log10
    transformed lowess fit curve.

    Args:
    d: Dictionary from dRFE
    frac: Fraction for lowess smoothing. Default 3/10.
    step_size: Rate of change step size to analyze for extraction
    (default: 0.05)

    Yields:
    int: number of redundant features
    """
    _,_,z,xnew,ynew = cal_lowess(d, frac, multi)
    df_elim = get_elim_df_ordered(d, multi)
    df_lowess = pd.DataFrame({'X': xnew, 'Y': ynew,
                              'xprime': pd.DataFrame(z)[0],
                              'yprime': pd.DataFrame(z)[1]})
    dxdy = cal_lowess_rate_log10(d, frac, multi)
    dxdy = dxdy[(dxdy.LOWESS >= max(dxdy.LOWESS) - np.std(dxdy.LOWESS))].copy()
    local_peak = []
    for yy in range(1, dxdy.shape[0]):
        lo = (dxdy.iloc[yy, 2] - dxdy.iloc[yy-1, 2]) > step_size
        l1 = dxdy.iloc[yy, 2] < 0
        local_peak.append(lo & l1)
    local_peak.append(False)
    peaks = dxdy[local_peak]
    val = df_lowess[(df_lowess['xprime'] == max(peaks.Features))].X.values
    redunt_feat_log10 = min(df_elim['log10_x'].values, key=lambda x: abs(x - val))
    redundant_feat = df_elim[(df_elim['log10_x'] == redunt_feat_log10)].x.values[0]
    return redundant_feat, redunt_feat_log10


def optimize_lowess_plot(d, fold, output_dir, frac=3/10, step_size=0.05,
                         classify=True, save_plot=False, multi=False):
    if classify:
        if multi:
            label = 'ROC AUC'
        else:
            label = 'NMI'
    else:
        label = 'R2'
    title = 'Fraction: %.2f, Step Size: %.2f' % (frac, step_size)
    _, max_feat_log10 = extract_max_lowess(d, frac, multi)
    x,y,z,_,_ = cal_lowess(d, frac, multi)
    df_elim = pd.DataFrame({'X': x, 'Y': y})
    _,lo = extract_max_lowess(d, frac, multi)
    _,l1 = extract_redundant_lowess(d, frac, step_size, multi)
    plt.plot(x, y, 'o', label="dRFE")
    plt.plot(pd.DataFrame(z)[0], pd.DataFrame(z)[1], '-', label="Lowess")
    plt.vlines(lo, ymin=np.min(y), ymax=np.max(y), colors='b',
               linestyles='--', label='Max Features')
    plt.vlines(l1, ymin=np.min(y), ymax=np.max(y),
               colors='orange', linestyles='--',
               label='Redundant Features')
    plt.xscale('log')
    plt.xlabel('log10(N Features)')
    plt.ylabel(label)
    plt.legend(loc='best')
    plt.title(title)
    if save_plot:
        plt.savefig("%s/optimize_lowess_%s_frac%.2f_step_%.2f_%s.png" %
                    (output_dir, fold, frac, step_size, label.replace(" ", "_")))
        plt.savefig("%s/optimize_lowess_%s_frac%.2f_step_%.2f_%s.pdf" %
                    (output_dir, fold, frac, step_size, label.replace(" ", "_")))
        plt.savefig("%s/optimize_lowess_%s_frac%.2f_step_%.2f_%s.svg" %
                    (output_dir, fold, frac, step_size, label.replace(" ", "_")))
    plt.show()
