"""
This script contains functions to calculate core + peripheral features
based on a lowess fitted curve. Uses a log10 transformation of feature
inputs. Also contains the optimization plot function for manual
parameter optimization for lowess.

Developed by Kynon Jade Benjamin.
"""

__author__ = 'Kynon J Benjamin'

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import interpolate
import matplotlib.pyplot as plt


def run_lowess(xnew, ynew, frac):
    """
    Internal function to run LOWESS.
    """
    lowess = sm.nonparametric.lowess
    z = lowess(ynew, xnew, frac=frac, it=20)
    return z


def array_to_tuple(np_array):
    """
    Internal function to convert array to tuple.
    """
    try:
        return tuple(array_to_tuple(_) for _ in np_array)
    except TypeError:
        return np_array


def get_elim_df_ordered(d, multi):
    """
    Internal function to extract elimination information from dictionary
    and convert to data frame. Also performs log10 normalization on
    features.
    """
    if multi:
        col = 3
    else:
        col = 1
    df_elim = pd.DataFrame([{'x':k, 'y':d[k][col],
                             'acc':d[k][2]} for k in d.keys()]).sort_values('x')
    df_elim['log10_x'] = np.log(df_elim['x']+0.5)
    return df_elim


def cal_lowess(d, frac, multi, acc):
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
    if acc:
        y = df_elim['acc'].values
    else:
        y = df_elim['y'].values
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(x.min(), x.max(), num=5001, endpoint=True)
    ynew = interpolate.splev(xnew, tck, der=0)
    z = run_lowess(array_to_tuple(xnew), array_to_tuple(ynew), frac)
    return x,y,z,xnew,ynew


def cal_lowess_rate_log10(d, frac=3/10, multi=False, acc=False):
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
    _,_,z,_,_ = cal_lowess(d, frac, multi, acc)
    dfz = pd.DataFrame(z, columns=["Features", "LOWESS"])
    pts = dfz.drop(0)
    pts['DxDy'] = np.diff(dfz.Features) / np.diff(dfz.LOWESS)
    return pts


def extract_max_lowess(d, frac=3/10, multi=False, acc=False):
    """
    Extract max features based on rate of change of log10
    transformed lowess fit curve.

    Args:
    d: Dictionary from dRFE
    frac: Fraction for lowess smoothing. Default 3/10.

    Yields:
    int: number of peripheral features
    """
    _,_,z,xnew,ynew = cal_lowess(d, frac, multi, acc)
    df_elim = get_elim_df_ordered(d, multi)
    df_lowess = pd.DataFrame({'X': xnew, 'Y': ynew,
                              'xprime': pd.DataFrame(z)[0],
                              'yprime': pd.DataFrame(z)[1]})
    val = df_lowess[(df_lowess['yprime'] == max(df_lowess.yprime))].X.values
    closest_val = min(df_elim['log10_x'].values, key=lambda x: abs(x - val))
    return df_elim[(df_elim['log10_x'] == closest_val)].x.values[0], closest_val


def extract_peripheral_lowess(d, frac=3/10, step_size=0.02, multi=False,
                             acc=False):
    """
    Extract peripheral features based on rate of change of log10
    transformed lowess fit curve.

    Args:
    d: Dictionary from dRFE
    frac: Fraction for lowess smoothing. Default 3/10.
    step_size: Rate of change step size to analyze for extraction
    (default: 0.02)
    multi: Is the target multi-class (boolean). Default False.
    classify: Is the target classification (boolean). Default True.
    acc: Use accuracy metric to optimize data (boolean). Default False.

    Yields:
    int: number of peripheral features
    """
    _,_,z,xnew,ynew = cal_lowess(d, frac, multi, acc)
    df_elim = get_elim_df_ordered(d, multi)
    df_lowess = pd.DataFrame({'X': xnew, 'Y': ynew,
                              'xprime': pd.DataFrame(z)[0],
                              'yprime': pd.DataFrame(z)[1]})
    dxdy = cal_lowess_rate_log10(d, frac, multi, acc)
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
    peripheral_feat = df_elim[(df_elim['log10_x'] == redunt_feat_log10)].x.values[0]
    return peripheral_feat, redunt_feat_log10


def optimize_lowess_plot(d, fold, output_dir, frac=3/10, step_size=0.02,
                         classify=True, save_plot=False, multi=False, acc=False,
                         print_out=True):
    """
    Plot the LOWESS smoothing plot for RFE with lines annotating set selection.

    Args:
    d: Dictionary from dRFE
    frac: Fraction for lowess smoothing. Default 3/10.
    step_size: Rate of change step size to analyze for extraction
    (default: 0.02)
    multi: Is the target multi-class (boolean). Default False.
    classify: Is the target classification (boolean). Default True.
    acc: Use accuracy metric to optimize data (boolean). Default False.
    save_plot: Save the optmization plot (boolean). Default False.
    print_out: Print to screen (boolean). Default True.

    Yields:
    Plot of the data with annotation and LOWESS smoothing.
    Saves plot if specified.
    """
    if classify:
        if multi:
            label = 'ROC AUC'
        elif acc:
            label = "Accuracy"
        else:
            label = 'NMI'
    else:
        label = 'R2'
    title = 'Fraction: %.2f, Step Size: %.2f' % (frac, step_size)
    # transform to linear scale
    x,y,z,_,_ = cal_lowess(d, frac, multi, acc)
    df_elim = pd.DataFrame({'X': np.exp(x) - 0.5, 'Y': y})
    lowess_df = pd.DataFrame(z, columns=["X0", "Y0"])
    lowess_df.loc[:,"X0"] = np.exp(lowess_df["X0"]) - 0.5
    lo,_ = extract_max_lowess(d, frac, multi, acc)
    l1,_ = extract_peripheral_lowess(d, frac, step_size, multi, acc)
    # Plot
    plt.clf()
    f1 = plt.figure()
    plt.plot(df_elim["X"], df_elim["Y"], 'o', label="dRFE")
    plt.plot(lowess_df["X0"], lowess_df["Y0"], '-', label="Lowess")
    plt.vlines(lo, ymin=np.min(y), ymax=np.max(y), colors='b',
               linestyles='--', label='Max Features')
    plt.vlines(l1, ymin=np.min(y), ymax=np.max(y),
               colors='orange', linestyles='--',
               label='Peripheral Features')
    plt.xscale('log'); plt.xlabel('log(N Features)')
    plt.ylabel(label); plt.legend(loc='best'); plt.title(title)
    if save_plot:
        plt.savefig("%s/optimize_lowess_%s_frac%.2f_step_%.2f_%s.png" %
                    (output_dir, fold, frac, step_size, label.replace(" ", "_")))
        plt.savefig("%s/optimize_lowess_%s_frac%.2f_step_%.2f_%s.pdf" %
                    (output_dir, fold, frac, step_size, label.replace(" ", "_")))
        plt.savefig("%s/optimize_lowess_%s_frac%.2f_step_%.2f_%s.svg" %
                    (output_dir, fold, frac, step_size, label.replace(" ", "_")))
    if print_out:
        plt.show()
