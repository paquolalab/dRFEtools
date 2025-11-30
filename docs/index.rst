.. dRFEtools documentation master file, created by
   sphinx-quickstart on Fri Jul 12 22:21:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dRFEtools's documentation!
=====================================

.. currentmodule:: dRFEtools

``dRFEtools`` is a package for dynamic recursive feature elimination with
scikit-learn.

Authors
-------

* Apuã Paquola
* Kynon Jade Benjamin
* Tarun Katipalli

Package Information
-------------------

dRFEtools is developed in Python 3.10+.

In addition to scikit-learn, ``dRFEtools`` is also built with:

* NumPy
* SciPy
* Pandas
* matplotlib
* plotnine
* statsmodels

Currently, dynamic RFE supports models with ``coef_`` or ``feature_importances_`` attribute.

Features
--------

This package provides several functions to run dynamic recursive feature
elimination (dRFE) for random forest and linear model classifier and regression
models.

**Random Forest Models:**
    Assume Out-of-Bag (OOB) scoring is enabled.

**Linear Models:**
    Build an internal developmental split.

Measurements for Feature Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Classification:**

1. Normalized mutual information
2. Accuracy
3. Area under the curve (AUC) ROC curve

**Regression:**

1. R2 (can be negative if model is arbitrarily worse)
2. Explained variance
3. Mean squared error

Package Structure
-----------------

The codebase is organized into focused modules:

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

API Overview
------------

``rf_rfe`` and ``dev_rfe`` return standardized dictionaries for each iteration
that include the number of retained features, a task-specific ``metrics``
mapping, and the indices of surviving features. The same format is consumed by
the plotting, LOWESS, and scoring helpers documented in the reference manual
below, and is reflected throughout the tutorials for the 0.4.x release.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   citation
   install
   optimization
   classification
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
