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

__all__ = [
    "_cal_lowess",
    "extract_max_lowess",
    "optimize_lowess_plot",
    "extract_peripheral_lowess"
]

def _run_lowess(xnew, ynew, frac):
    """
    Internal function to run LOWESS.

    Args:
        xnew (array-like): Independent variable values.
        ynew (array-like): Dependent variable values.
        frac (float): The fraction of the data used when estimating each y-value.

    Returns:
        numpy.ndarray: Smoothed values from LOWESS.
    """
    lowess = sm.nonparametric.lowess
    return lowess(ynew, xnew, frac=frac, it=20)


def _array_to_tuple(np_array):
    """
    Internal function to convert array to tuple.

    Args:
        np_array (numpy.ndarray): Numpy array to be converted.

    Returns:
        tuple: Converted tuple.
    """
    try:
        return tuple(array_to_tuple(_) for _ in np_array)
    except TypeError:
        return np_array


def _get_elim_df_ordered(d, multi):
    """
    Internal function to extract elimination information from dictionary
    and convert to data frame. Also performs log10 normalization on features.

    Args:
        d (dict): Dictionary containing elimination information.
        multi (bool): Whether the target is multi-class.

    Returns:
        pandas.DataFrame: DataFrame with elimination information.
    """
    col = 3 if multi else 1
    df_elim = pd.DataFrame([{'x': k, 'y': d[k][col],
                             'acc': d[k][2]} for k in d.keys()]).sort_values('x')
    df_elim['log10_x'] = np.log10(df_elim['x'] + 0.5)
    return df_elim


def _cal_lowess(d, frac, multi, acc):
    """
    Internal function to calculate the lowess curve.

    Args:
        d (dict): Dictionary from dRFE.
        frac (float): Fraction for lowess smoothing.
        multi (bool): Whether the target is multi-class.
        acc (bool): Use accuracy metric to optimize data.

    Returns:
        tuple: n_features (x), model validation values (y), lowess curve (z), xnew, ynew.
    """
    df_elim = _get_elim_df_ordered(d, multi)
    x = df_elim['log10_x'].values
    y = df_elim['acc'].values if acc else df_elim['y'].values
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(x.min(), x.max(), num=5001, endpoint=True)
    ynew = interpolate.splev(xnew, tck, der=0)
    z = _run_lowess(_array_to_tuple(xnew), array_to_tuple(ynew), frac)
    return x, y, z, xnew, ynew


def _cal_lowess_rate_log10(d, frac=3/10, multi=False, acc=False):
    """
    Calculate rate of change on the lowess fitted curve with log10
    transformation.

    Args:
        d (dict): Dictionary from dRFE.
        frac (float): Fraction for lowess smoothing.
        multi (bool): Whether the target is multi-class.
        acc (bool): Use accuracy metric to optimize data.

    Returns:
        pandas.DataFrame: DataFrame with n_features, lowess value, and rate of change (DxDy).
    """
    _, _, z, _, _ = _cal_lowess(d, frac, multi, acc)
    dfz = pd.DataFrame(z, columns=["Features", "LOWESS"])
    pts = dfz.drop(0)
    pts['DxDy'] = np.diff(dfz.Features) / np.diff(dfz.LOWESS)
    return pts


def extract_max_lowess(d, frac=3/10, multi=False, acc=False):
    """
    Extract max features based on rate of change of log10
    transformed lowess fit curve.

    Args:
        d (dict): Dictionary from dRFE.
        frac (float): Fraction for lowess smoothing.
        multi (bool): Whether the target is multi-class.
        acc (bool): Use accuracy metric to optimize data.

    Returns:
        tuple: Number of peripheral features and closest value.
    """
    _, _, z, xnew, ynew = _cal_lowess(d, frac, multi, acc)
    df_elim = _get_elim_df_ordered(d, multi)
    df_lowess = pd.DataFrame({'X': xnew, 'Y': ynew,
                              'xprime': pd.DataFrame(z)[0],
                              'yprime': pd.DataFrame(z)[1]})
    val = df_lowess[df_lowess['yprime'] == max(df_lowess.yprime)].X.values
    closest_val = min(df_elim['log10_x'].values, key=lambda x: abs(x - val))
    return df_elim[df_elim['log10_x'] == closest_val].x.values[0], closest_val


def extract_peripheral_lowess(d, frac=3/10, step_size=0.02, multi=False,
                              acc=False):
    """
    Extract peripheral features based on rate of change of log10
    transformed lowess fit curve.

    Args:
        d (dict): Dictionary from dRFE.
        frac (float): Fraction for lowess smoothing.
        step_size (float): Rate of change step size to analyze for extraction.
        multi (bool): Whether the target is multi-class.
        acc (bool): Use accuracy metric to optimize data.

    Returns:
        tuple: Number of peripheral features and redundant feature log10 value.
    """
    _, _, z, xnew, ynew = _cal_lowess(d, frac, multi, acc)
    df_elim = _get_elim_df_ordered(d, multi)
    df_lowess = pd.DataFrame({'X': xnew, 'Y': ynew,
                              'xprime': pd.DataFrame(z)[0],
                              'yprime': pd.DataFrame(z)[1]})
    dxdy = _cal_lowess_rate_log10(d, frac, multi, acc)
    dxdy = dxdy[dxdy.LOWESS >= max(dxdy.LOWESS) - np.std(dxdy.LOWESS)].copy()
    local_peak = [(dxdy.iloc[yy, 2] - dxdy.iloc[yy-1, 2]) > step_size and dxdy.iloc[yy, 2] < 0 for yy in range(1, dxdy.shape[0])]
    local_peak.append(False)
    peaks = dxdy[local_peak]
    val = df_lowess[df_lowess['xprime'] == max(peaks.Features)].X.values
    redunt_feat_log10 = min(df_elim['log10_x'].values, key=lambda x: abs(x - val))
    peripheral_feat = df_elim[df_elim['log10_x'] == redunt_feat_log10].x.values[0]
    return peripheral_feat, redunt_feat_log10


def optimize_lowess_plot(d, fold, output_dir, frac=3/10, step_size=0.02,
                         classify=True, save_plot=False, multi=False, acc=False,
                         print_out=True):
    """
    Plot the LOWESS smoothing plot for RFE with lines annotating set selection.

    Args:
        d (dict): Dictionary from dRFE.
        fold (int): Current fold.
        output_dir (str): Output directory.
        frac (float): Fraction for lowess smoothing.
        step_size (float): Rate of change step size to analyze for extraction.
        classify (bool): Whether the target is classification.
        save_plot (bool): Save the optimization plot.
        multi (bool): Whether the target is multi-class.
        acc (bool): Use accuracy metric to optimize data.
        print_out (bool): Print to screen.

    Returns:
        None

    Notes:
       Will generate a plot with LOWESS smoothing
    """
    if classify:
        label = 'ROC AUC' if multi else 'Accuracy' if acc else 'NMI'
    else:
        label = 'R2'
    title = f'Fraction: {frac:.2f}, Step Size: {step_size:.2f}'

    x, y, z, _, _ = _cal_lowess(d, frac, multi, acc)
    df_elim = pd.DataFrame({'X': np.exp(x) - 0.5, 'Y': y})
    lowess_df = pd.DataFrame(z, columns=["X0", "Y0"])
    lowess_df["X0"] = np.exp(lowess_df["X0"]) - 0.5
    lo, _ = extract_max_lowess(d, frac, multi, acc)
    l1, _ = extract_peripheral_lowess(d, frac, step_size, multi, acc)

    plt.clf()
    plt.figure()
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
        for ext in ["png", "pdf", "svg"]:
            plt.savefig(f"{output_dir}/optimize_lowess_{fold}_frac{frac:.2f}_step_{step_size:.2f}_{label.replace(' ', '_')}.{ext}")

    if print_out:
        plt.show()
