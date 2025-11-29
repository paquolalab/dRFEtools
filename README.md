# dRFEtools - dynamic Recursive Feature Elimination

`dRFEtools` is a package for dynamic recursive feature elimination with
scikit-learn.

Authors: Apuã Paquola, Kynon Jade Benjamin, and Tarun Katipalli

Package developed in Python 3.10+.

In addition to scikit-learn, `dRFEtools` is also built with NumPy, SciPy,
Pandas, matplotlib, plotnine, and statsmodels. Currently, dynamic RFE supports
models with `coef_` or `feature_importances_` attribute.

This package provides several functions to run dynamic recursive feature
elimination (dRFE) for random forest and linear model classifier and regression
models. For random forest workflows, dRFEtools assumes Out-of-Bag (OOB) scoring
is enabled. Linear-model workflows build a developmental split internally. For
both classification and regression, three measurements are calculated for
feature selection:

Classification:

1.  Normalized mutual information
2.  Accuracy
3.  Area under the curve (AUC) ROC curve

Regression:

1.  R2 (this can be negative if model is arbitrarily worse)
2.  Explained variance
3.  Mean squared error

Package structure
-----------------

The repository is organized into focused modules to match the runtime
architecture:

* ``dRFEtools.py`` – core interfaces for random-forest and developmental-set
  elimination workflows.
* ``scoring/`` – metric implementations for developmental splits and
  random-forest OOB scoring.
* ``lowess/`` – helpers for smoothing elimination curves and extracting optimal
  feature counts.
* ``metrics/`` – feature ranking utilities used during elimination.
* ``plotting.py`` – visualization helpers re-exported from the top-level
  package.
* ``cli.py`` – command-line entry points for running full dRFE pipelines.
* ``utils.py`` – shared helpers for normalizing results and persisting plots.

# Table of Contents

1.  [Citation](#citation)
2.  [Installation](#installation)
3.  [Tutorials](#tutorials)
4.  [Reference Manual](#reference-manual)
    1.  [Core elimination functions](#core-elimination-functions)
    2.  [Ranking and scoring utilities](#ranking-and-scoring-utilities)
    3.  [LOWESS helpers](#lowess-helpers)
    4.  [Plotting functions](#plotting-functions)
    5.  [Utilities and CLI](#utilities-and-cli)

## Citation

If using please cite the following:

Kynon J M Benjamin, Tarun Katipalli, Apuã C M Paquola, 
dRFEtools: dynamic recursive feature elimination for omics, 
Bioinformatics, Volume 39, Issue 8, August 2023, btad513, 
https://doi.org/10.1093/bioinformatics/btad513

PMID: 37632789

DOI: [10.1093/bioinformatics/btad513](10.1093/bioinformatics/btad513).


## Installation

`pip install --user dRFEtools`

## Tutorials

We have two tutorials for [optimization](./examples/optimization.md)
(version 0.2) and [classification](./examples/classification.md) (version 0.3+).

In addition to this, we have example code used in the manuscript for
scikit-learn simulation, biological simulation, and BrainSEQ Phase 1
at the link below.

[https://github.com/LieberInstitute/dRFEtools_manuscript](https://github.com/LieberInstitute/dRFEtools_manuscript/tree/main)

## Reference Manual

### Core elimination functions

* **`rf_rfe`** – Runs random-forest feature elimination and returns a tuple of
  `(results_by_feature_count, first_step)`. Each dictionary entry contains the
  number of retained features (`n_features`), the task-appropriate metrics, and
  the indices of surviving features.
* **`dev_rfe`** – Performs the same elimination loop for estimators that rely on
  a developmental split, yielding the same standardized result structure as
  `rf_rfe`.

### Ranking and scoring utilities

* **`features_rank_fnc`** – Ranks features during elimination and optionally
  persists the ranking table for each fold.
* Developmental-set metrics (``dev_score_*``) live under
  ``dRFEtools.scoring.dev``.
* Random-forest OOB metrics (``oob_score_*``) live under
  ``dRFEtools.scoring.random_forest``.

### LOWESS helpers

* **`extract_max_lowess`** – Identifies the optimal feature count from the
  LOWESS-smoothed elimination curve.
* **`extract_peripheral_lowess`** – Detects the inflection point associated with
  peripheral features.
* **`optimize_lowess_plot`** – Visualizes the LOWESS curve with annotations
  about the selected feature counts.

### Plotting functions

Plotting helpers are defined in ``dRFEtools.plotting`` and re-exported from the
top-level package:

* **`plot_metric`** – Render elimination trajectories for individual metrics.
* **`plot_with_lowess_vline`** – Overlay LOWESS-derived selection cutoffs on the
  metric trajectory plot.

### Utilities and CLI

* **`normalize_rfe_result`**, **`get_feature_importances`**, and
  **`save_plot_variants`** are available under ``dRFEtools.utils`` and support
  the standardized dictionary-based API.
* The command-line interface in ``dRFEtools.cli`` wraps the same workflows for
  CSV inputs: run ``python -m dRFEtools.cli --help`` to explore available
  commands.
