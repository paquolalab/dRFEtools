***
API
***

.. currentmodule:: dRFEtools

Core elimination interfaces
--------------------------

.. autosummary::
   :toctree: functions

   rf_rfe
   dev_rfe


LOWESS helpers
--------------

.. autosummary::
   :toctree: functions

   extract_max_lowess
   extract_peripheral_lowess
   optimize_lowess_plot


Plotting functions
------------------

Plotting helpers live in :mod:`dRFEtools.plotting` and are re-exported from the
top-level package for convenience.

.. autosummary::
   :toctree: functions

   plot_metric
   plot_with_lowess_vline


Ranking utilities
-----------------

.. autosummary::
   :toctree: functions

   features_rank_fnc


Developmental set metrics functions
-----------------------------------

.. autosummary::
   :toctree: functions

   dev_score_accuracy
   dev_score_evar
   dev_score_mse
   dev_score_nmi
   dev_score_r2
   dev_score_roc


Out-of-Bag (OOB) metrics functions
----------------------------------

.. autosummary::
   :toctree: functions

   oob_score_accuracy
   oob_score_evar
   oob_score_mse
   oob_score_nmi
   oob_score_r2
   oob_score_roc


Utility helpers
---------------

.. autosummary::
   :toctree: functions

   utils.normalize_rfe_result
   utils.get_feature_importances
   utils.save_plot_variants


Command-line entry points
-------------------------

.. autosummary::
   :toctree: functions

   cli.build_parser
   cli.run_rf_rfe
   cli.run_dev_rfe
