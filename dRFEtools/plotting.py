"""
Plotting utilities for dRFEtools.
"""
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

from ._lowess_redundant import (
    _cal_lowess,
    extract_max_lowess,
    optimize_lowess_plot,
    extract_peripheral_lowess,
)

filterwarnings("ignore", category=MatplotlibDeprecationWarning)
filterwarnings('ignore', category=UserWarning, module='plotnine.*')
filterwarnings('ignore', category=DeprecationWarning, module='plotnine.*')

__all__ = ["plot_metric", "plot_with_lowess_vline"]


def _save_plot(p, fn, width=7, height=7):
    """
    Save plot as svg, png, and pdf with specific label and dimension.

    Args:
        p: Plot object
        fn (str): File name (without extension)
        width (int): Plot width. Default 7
        height (int): Plot height. Default 7
    """
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
