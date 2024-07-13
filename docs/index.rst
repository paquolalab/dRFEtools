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

This package provides several functions to run dynamic recursive feature elimination (dRFE) for random forest and linear model classifier and regression models.

**Random Forest Models:**
    Assumes Out-of-Bag (OOB) is set to True.

**Linear Models:**
    Generates a developmental set.

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

The package is divided into four additional scripts:

1. Out-of-bag dynamic RFE metrics (AP/KJB)
2. Validation (developmental) set dynamic RFE metrics (KJB)
3. Rank features function (TK)
4. Lowess core + peripheral selection (KJB)

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
