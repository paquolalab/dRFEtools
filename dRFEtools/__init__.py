from ._dev_scoring import (
    dev_score_r2, dev_score_nmi,
    dev_score_roc, dev_score_mse,
    dev_score_evar, dev_score_accuracy
)
from ._random_forest import (
    oob_score_r2, oob_score_nmi,
    oob_score_roc, oob_score_mse,
    oob_score_evar, oob_score_accuracy
)
from ._lowess_redundant import (
    extract_max_lowess,
    optimize_lowess_plot,
    extract_peripheral_lowess
)

from .dRFEtools import (
    rf_rfe, dev_rfe,
    plot_metric, plot_with_lowess_vline
)

__version__ = "0.3.5"

__all__ = [
    "rf_rfe",
    "dev_rfe",
    "__version__",
    "plot_metric",
    "features_rank_fnc",
    "extract_max_lowess",
    "optimize_lowess_plot",
    "plot_with_lowess_vline",
    "extract_peripheral_lowess",
    "oob_score_r2", "dev_score_r2",
    "oob_score_roc", "dev_score_roc",
    "oob_score_nmi", "dev_score_nmi",
    "oob_score_mse", "dev_score_mse",
    "oob_score_evar", "dev_score_evar",
    "oob_score_accuracy", "dev_score_accuracy",
]
