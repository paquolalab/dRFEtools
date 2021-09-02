from setuptools import setup, find_packages

LONG_DESCRIPTION="""
dRFEtools - A package for preforming dynamic recursive feature elimination with sklearn
=======================================================================

``dRFEtools`` is a package for dynamic recursive feature elimination supporting
random forest and several linear models for classification and regression.

Authors: Apuã Paquola, Kynon Jade Benjamin, and Tarun Katipalli

If using please cite: XXX.

Table of Contents
=================

-  `Installation <#installation>`__
-  `Reference Manual <#reference-manual>`__

Installation
============

``pip install --user dRFEtools``

Reference Manual
================

============================================================ ==================================================================================================
Function                                                     Description
============================================================ ==================================================================================================
`feature_elimination <#feature-elimination-main>`__          Runs random forest classification feature elimination
`features_rank_fnc <#feature-rank-function>`__               Rank features
`n_features_iter <#n-feature-iterator>`__                    Determines the features to keep
`oob_predictions <#oob-prediction>`__                        Extracts out-of-bag (OOB) predictions from random forest classifier classes
`oob_score_accuracy <#oob-accuracy-score>`__                 Calculates the accuracy score for the OOB predictions
`oob_score_nmi <#oob-normalized-mutual-information-score>`__ Calculates the normalized mutual information score for the OOB predictions
`oob_score_roc <#oob-area-under-roc-curve-score>`__          Calculates the area under the ROC curve (AUC) for the OOB predictions
`plot_acc <#plot-feature-elimination-by-accuracy>`__         Plot feature elimination with accuracy as measurement
`plot_nmi <#plot-feature-elimination-by-nmi>`__              Plot feature elimination with NMI as measurement
`plot_roc <#plot-feature-elimination-by-auc>`__              Plot feature elimination with AUC ROC curve as measurement
`rf_fe <#feature-elimination-subfunction>`__                 Iterate over features to be eliminated
`rf_fe_step <#feature-elimination-step>`__                   Apply random forest to training data, rank features, and conduct feature elimination (single step)
============================================================ ==================================================================================================

Feature Elimination Main
------------------------

``feature_elimination``

Runs random forest feature elimination step over iterator process.

**Args:**

-  estimator: Random forest classifier object
-  X: a data frame of training data
-  Y: a vector of sample labels from training data set
-  features: a vector of feature names
-  fold: current fold
-  out_dir: output directory. default '.'
-  elimination_rate: percent rate to reduce feature list.
   default .2
-  RANK: Output feature ranking. default=True (Boolean)

**Yields:**

-  dict: a dictionary with number of features, normalized mutual
   information score, accuracy score, and AUC ROC score, array of the
   indexes for features to keep

Feature Rank Function
---------------------

``feature_rank_fnc``

Ranks features.

**Args:**

-  features: A vector of feature names
-  rank: A vector with feature ranks based on absolute value of feature
   importance
-  n_features_to_keep: Number of features to keep. (Int)
-  fold: Fold to analyzed. (Int)
-  out_dir: Output directory for text file. Default '.'
-  RANK: Boolean (True or False)

**Yields:**

-  Text file: Ranked features by fold tab-delimited text file, only if
   RANK=True

N Feature Iterator
------------------

``n_features_iter``

Determines the features to keep.

**Args:**

-  nf: current number of features
-  keep_rate: percentage of features to keep

**Yields:**

-  int: number of features to keep

OOB Prediction
--------------

``oob_predictions``

Extracts out-of-bag (OOB) predictions from random forest classifier
classes.

**Args:**

-  estimator: Random forest classifier object

**Yields:**

-  vector: OOB predicted labels

OOB Accuracy Score
------------------

``oob_score_accuracy``

Calculates the accuracy score from the OOB predictions.

**Args:**

-  estimator: Random forest classifier object
-  Y: a vector of sample labels from training data set

**Yields:**

-  float: accuracy score

OOB Normalized Mutual Information Score
---------------------------------------

``oob_score_nmi``

Calculates the normalized mutual information score from the OOB
predictions.

**Args:**

-  estimator: Random forest classifier object
-  Y: a vector of sample labels from training data set

**Yields:**

-  float: normalized mutual information score

OOB Area Under ROC Curve Score
------------------------------

``oob_score_roc``

Calculates the area under the ROC curve score for the OOB predictions.

**Args:**

-  estimator: Random forest classifier object
-  Y: a vector of sample labels from training data set

**Yields:**

-  float: AUC ROC score

Plot Feature Elimination by Accuracy
------------------------------------

``plot_acc``

Plot feature elimination results for accuracy.

**Args:**

-  d: feature elimination class dictionary
-  fold: current fold
-  out_dir: output directory. default '.'

**Yields:**

-  graph: plot of feature by accuracy, automatically saves files as png
   and svg

Plot Feature Elimination by NMI
-------------------------------

``plot_nmi``

Plot feature elimination results for normalized mutual information.

**Args:**

-  d: feature elimination class dictionary
-  fold: current fold
-  out_dir: output directory. default '.'

**Yields:**

-  graph: plot of feature by NMI, automatically saves files as png and
   svg

Plot Feature Elimination by AUC
-------------------------------

``plot_roc``

Plot feature elimination results for AUC ROC curve.

**Args:**

-  d: feature elimination class dictionary
-  fold: current fold
-  out_dir: output directory. default '.'

**Yields:**

-  graph: plot of feature by AUC, automatically saves files as png and
   svg

Feature Elimination Subfunction
-------------------------------

``rf_fe``

Iterate over features to by eliminated by step.

**Args:**

-  estimator: Random forest classifier object
-  X: a data frame of training data
-  Y: a vector of sample labels from training data set
-  n_features_iter`: iterator for number of features to keep loop
-  features: a vector of feature names
-  fold: current fold
-  out_dir: output directory. default '.'
-  RANK: Boolean (True or False)

**Yields:**

-  list: a list with number of features, normalized mutual information
   score, accuracy score, and AUC ROC score, array of the indices for
   features to keep

Feature Elimination Step
------------------------

``rf_fe_step``

Apply random forest to training data, rank features, conduct feature
elimination.

**Args:**

-  estimator: Random forest classifier object
-  X: a data frame of training data
-  Y: a vector of sample labels from training data set
-  n_features_to_keep`: number of features to keep
-  features: a vector of feature names
-  fold: current fold
-  out_dir: output directory. default '.'
-  RANK: Boolean (True or False)

**Yields:**

-  dict: a dictionary with number of features, normalized mutual
   information score, accuracy score, AUC ROC score, and selected
   features

"""

setup(name='dRFEtools',
      version='0.0.1',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.18.1',
          'pandas>=0.24.2',
          'matplotlib>=3.1.1',
          'plotnine>=0.6.0',
          'scikit-learn>=0.21.1',
          'scipy>=1.3.3',
          'rpy2>=3.2.2',
          'numexpr>=2.7.1',
          'matplotlib>=3.1.1',
          'gtfparse>=1.2.0',
      ],
      author="Kynon JM Benjamin",
      author_email="kj.benjamin90@gmail.com",
      decription="A package for preforming dynamic recursive feature elimination with sklearn",
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/x-rst',
      package_data={
          '': ['*md'],
      },
      url="https://github.com/paquolalab/dRFEtools.git",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
      ],
      keywords='random-forest recursive-feature-elimination sklearn linear-models feature-ranking',
      zip_safe=False)
