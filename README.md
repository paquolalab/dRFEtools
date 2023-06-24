# dRFEtools - dynamic Recursive Feature Elimination

`dRFEtools` is a package for dynamic recursive feature elimination with
sklearn. Currently supporting random forest classification and regression,
and linear models (linear, lasso, ridge, and elastic net).

Authors: Apuã Paquola, Kynon Jade Benjamin, and Tarun Katipalli

Package developed in Python 3.7+.

In addition to scikit-learn, `dRFEtools` is also built with NumPy, SciPy,
Pandas, matplotlib, plotnine, and statsmodels.

This package has several function to run dynamic recursive feature elimination
(dRFE) for random forest and linear model classifier and regression models. For
random forest, it assumes Out-of-Bag (OOB) is set to True. For linear models,
it generates a developmental set. For both classification and regression, three
measurements are calculated for feature selection:

Classification:

1.  Normalized mutual information
2.  Accuracy
3.  Area under the curve (AUC) ROC curve

Regression:

1.  R2 (this can be negative if model is arbitrarily worse)
2.  Explained variance
3.  Mean squared error

The package has been split in to four additional scripts for:

1.  Out-of-bag dynamic RFE metrics (AP)
2.  Validation set dynamic RFE metrics (KJB)
3.  Rank features function (TK)
4.  Lowess core + peripheral selection (KJB)

# Table of Contents

1.  [Citation](#org7b64d47)
2.  [Installation](#org04443e4)
3.  [Tutorials](#org07777f88)
4.  [Reference Manual](#org5afd041)
    1.  [dRFEtools main functions](#org6171433)
    2.  [Peripheral features functions](#org3cfdf65)
    3.  [Plotting functions](#org8ecca01)
    4.  [Metric functions](#org377b1aa)
    5.  [Random forest helper functions](#orga29d49b)
    6.  [Linear model helper functions](#orgbda21bf)

<a id="org7b64d47"></a>

## Citation

If using please cite the following:
Pre-print DOI: https://doi.org/10.1101/2022.07.27.501227
[![DOI](https://zenodo.org/badge/402494754.svg)](https://zenodo.org/badge/latestdoi/402494754).


<a id="org04443e4"></a>

## Installation

`pip install --user dRFEtools`

<a id="org07777f88"></a>
## Tutorials

Follow [this](https://github.com/LieberInstitute/dRFEtools_manuscript/blob/main/optimization/_m/optimization.ipynb) jupyter notebook for an example on optimization.

The GitHub below has example code for sklearn simulation, biological simulation, and using BrainSEQ Phase 1.

[https://github.com/LieberInstitute/dRFEtools_manuscript](https://github.com/LieberInstitute/dRFEtools_manuscript/tree/main)

<a id="org5afd041"></a>

## Reference Manual

<a id="org6171433"></a>

### dRFEtools main functions

1.  dRFE - Random Forest

    `rf_rfe`

    Runs random forest feature elimination step over iterator process.

    **Args:**

    -   estimator: Random forest classifier object
    -   X: a data frame of training data
    -   Y: a vector of sample labels from training data set
    -   features: a vector of feature names
    -   fold: current fold
    -   out_dir: output directory. default '.'
    -   elimination_rate: percent rate to reduce feature list. default .2
    -   RANK: Output feature ranking. default=True (Boolean)

    **Yields:**

    -   dict: a dictionary with number of features, normalized mutual information score, accuracy score, and array of the indexes for features to keep

2.  dRFE - Linear Models

    `dev_rfe`

    Runs recursive feature elimination for linear model step over iterator
    process assuming developmental set is needed.

    **Args:**

    -   estimator: regressor or classifier linear model object
    -   X: a data frame of training data
    -   Y: a vector of sample labels from training data set
    -   features: a vector of feature names
    -   fold: current fold
    -   out_dir: output directory. default '.'
    -   elimination_rate: percent rate to reduce feature list. default .2
    -   dev_size: developmental set size. default '0.20'
    -   RANK: run feature ranking, default 'True'
    -   SEED: random state. default 'True'

    **Yields:**

    -   dict: a dictionary with number of features, r2 score, mean square error,
        expalined variance, and array of the indices for features to keep

3.  Feature Rank Function

    `feature_rank_fnc`

    This function ranks features within the feature elimination loop.

    **Args:**

    -   features: A vector of feature names
    -   rank: A vector with feature ranks based on absolute value of feature importance
    -   n_features_to_keep: Number of features to keep. (Int)
    -   fold: Fold to analyzed. (Int)
    -   out_dir: Output directory for text file. Default '.'
    -   RANK: Boolean (True or False)

    **Yields:**

    -   Text file: Ranked features by fold tab-delimited text file, only if RANK=True

4.  N Feature Iterator

    `n_features_iter`

    Determines the features to keep.

    **Args:**

    -   nf: current number of features
    -   keep_rate: percentage of features to keep

    **Yields:**

    -   int: number of features to keep


5.  Calculate feature importance

    `cal_feature_imp`

    Generates feature importance from absolute value of feature weights.
	
	**Args:**
	
	-  estimator: the estimator to generate feature importance for
	
	**Yields:**
	
	-  estimator: returns the estimator with feature importance


<a id="org3cfdf65"></a>

### Peripheral features functions

1.  Run lowess

    `run_lowess`

    This function runs the lowess function and caches it to memory.

    **Args:**

    -   x: the x-values of the observed points
    -   y: the y-values of the observed points
    -   frac: the fraction of the data used when estimating each y-value. default 3/10

    **Yields:**

    -   z: 2D array of results

2.  Convert array to tuple

    `array_to_tuple`

    This function attempts to convert a numpy array to a tuple.

    **Args:**

    -   np_array: numpy array

    **Yields:**

    -   tuple

3.  Extract dRFE as a dataframe

    `get_elim_df_ordered`

    This function converts the dRFE dictionary to a pandas dataframe.

    **Args:**

    -   d: dRFE dictionary
    -   multi: is this for multiple classes. (True or False)

    **Yields:**

    -   df_elim: dRFE as a dataframe with log10 transformed features

4.  Calculate lowess curve

    `cal_lowess`

    This function calculates the lowess curve.

    **Args:**

    -   d: dRFE dictionary
    -   frac: the fraction of the data used when estimating each y-value
    -   multi: is this for multiple classes. (True or False)

    **Yields:**

    -   x: dRFE log10 transformed features
    -   y: dRFE metrics
    -   z: 2D numpy array with lowess curve
    -   xnew: increased intervals
    -   ynew: interpolated metrics for xnew

5.  Calculate lowess curve for log10

    `cal_lowess`

    This function calculates the rate of change on the lowess fitted curve with
    log10 transformated input.

    **Args:**

    -   d: dRFE dictionary
    -   frac: the fraction of the data used when estimating each y-value
    -   multi: is this for multiple classes. default False

    **Yields:**

    -   data frame: dataframe with n_features, lowess value, and rate of change (DxDy)

6.  Extract max lowess

    `extract_max_lowess`

    This function extracts the max features based on rate of change of log10
    transformed lowess fit curve.

    **Args:**

    -   d: dRFE dictionary
    -   frac: the fraction of the data used when estimating each y-value. default 3/10
    -   multi: is this for multiple classes. default False

    **Yields:**

    -   int: number of max features (smallest subset)

7.  Extract peripheral lowess

    `extract_peripheral_lowess`

    This function extracts the peripheral features based on rate of change of log10
    transformed lowess fit curve.

    **Args:**

    -   d: dRFE dictionary
    -   frac: the fraction of the data used when estimating each y-value. default 3/10
    -   step_size: rate of change step size to analyze for extraction. default 0.05
    -   multi: is this for multiple classes. default False

    **Yields:**

    -   int: number of peripheral features

8.  Optimize lowess plot

    `plot_with_lowess_vline`

    Peripheral set selection optimization plot. This will be ROC AUC for multiple
    classification (3+), NMI for binary classification, or R2 for regression. The
    plot returned has fraction and step size as well as lowess smoothed curve and
    indication of predicted peripheral set.

    **Args:**

    -   d: feature elimination class dictionary
    -   fold: current fold
    -   out_dir: output directory. default '.'
    -   frac: the fraction of the data used when estimating each y-value. default 3/10
    -   step_size: rate of change step size to analyze for extraction. default 0.05
    -   classify: is this a classification algorithm. default True
    -   multi: does this have multiple (3+) classes. default True

    **Yields:**

    -   graph: plot of dRFE with estimated peripheral set indicated as well as fraction and set size used. It automatically saves files as pdf, png, and svg

9.  Plot lowess vline

    `plot_with_lowess_vline`

    Plot feature elimination results with the peripheral set indicated. This will be
    ROC AUC for multiple classification (3+), NMI for binary classification, or R2
    for regression.

    **Args:**

    -   d: feature elimination class dictionary
    -   fold: current fold
    -   out_dir: output directory. default '.'
    -   frac: the fraction of the data used when estimating each y-value. default 3/10
    -   step_size: rate of change step size to analyze for extraction. default 0.05
    -   classify: is this a classification algorithm. default True
    -   multi: does this have multiple (3+) classes. default True

    **Yields:**

    -   graph: plot of dRFE with estimated peripheral set indicated, automatically saves files as pdf, png, and svg


<a id="org8ecca01"></a>

### Plotting functions

1.  Save plots

    `save_plots`

    This function save plot as svg, png, and pdf with specific label and dimension.

    **Args:**

    -   p: plotnine object
    -   fn: file name without extensions
    -   w: width, default 7
    -   h: height, default 7

    **Yields:** SVG, PNG, and PDF of plotnine object

2.  Plot dRFE Accuracy

    `plot_acc`

    Plot feature elimination results for accuracy.

    **Args:**

    -   d: feature elimination class dictionary
    -   fold: current fold
    -   out_dir: output directory. default '.'

    **Yields:**

    -   graph: plot of feature by accuracy, automatically saves files as pdf, png, and svg

3.  Plot dRFE NMI

    `plot_nmi`

    Plot feature elimination results for normalized mutual information.

    **Args:**

    -   d: feature elimination class dictionary
    -   fold: current fold
    -   out_dir: output directory. default '.'

    **Yields:**

    -   graph: plot of feature by NMI, automatically saves files as pdf, png, and svg

4.  Plot dRFE ROC AUC

    `plot_roc`

    Plot feature elimination results for AUC ROC curve.

    **Args:**

    -   d: feature elimination class dictionary
    -   fold: current fold
    -   out_dir: output directory. default '.'

    **Yields:**

    -   graph: plot of feature by AUC, automatically saves files as pdf, png, and svg

5.  Plot dRFE R2

    `plot_r2`

    Plot feature elimination results for R2 score. Note that this can be negative
    if model is arbitarily worse.

    **Args:**

    -   d: feature elimination class dictionary
    -   fold: current fold
    -   out_dir: output directory. default '.'

    **Yields:**

    -   graph: plot of feature by R2, automatically saves files as pdf, png, and svg

6.  Plot dRFE MSE

    `plot_mse`

    Plot feature elimination results for mean squared error score.

    **Args:**

    -   d: feature elimination class dictionary
    -   fold: current fold
    -   out_dir: output directory. default '.'

    **Yields:**

    -   graph: plot of feature by mean squared error, automatically saves files as pdf, png, and svg

7.  Plot dRFE Explained Variance

    `plot_evar`

    Plot feature elimination results for explained variance score.

    **Args:**

    -   d: feature elimination class dictionary
    -   fold: current fold
    -   out_dir: output directory. default '.'

    **Yields:**

    -   graph: plot of feature by explained variance, automatically saves files as pdf, png, and svg


<a id="org377b1aa"></a>

### Metric functions

1.  OOB Prediction

    `oob_predictions`

    Extracts out-of-bag (OOB) predictions from random forest classifier classes.

    **Args:**

    -   estimator: Random forest classifier object

    **Yields:**

    -   vector: OOB predicted labels

2.  OOB Accuracy Score

    `oob_score_accuracy`

    Calculates the accuracy score from the OOB predictions.

    **Args:**

    -   estimator: Random forest classifier object
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: accuracy score

3.  OOB Normalized Mutual Information Score

    `oob_score_nmi`

    Calculates the normalized mutual information score from the OOB predictions.

    **Args:**

    -   estimator: Random forest classifier object
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: normalized mutual information score

4.  OOB Area Under ROC Curve Score

    `oob_score_roc`

    Calculates the area under the ROC curve score for the OOB predictions.

    **Args:**

    -   estimator: Random forest classifier object
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: AUC ROC score

5.  OOB R2 Score

    `oob_score_r2`

    Calculates the r2 score from the OOB predictions.

    **Args:**

    -   estimator: Random forest regressor object
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: r2 score

6.  OOB Mean Squared Error Score

    `oob_score_mse`

    Calculates the mean squared error score from the OOB predictions.

    **Args:**

    -   estimator: Random forest regressor object
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: mean squared error score

7.  OOB Explained Variance Score

    `oob_score_evar`

    Calculates the explained variance score for the OOB predictions.

    **Args:**

    -   estimator: Random forest regressor object
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: explained variance score

8.  Developmental Test Set Predictions

    `dev_predictions`

    Extracts predictions using a development fold for linear
    regressor.

    **Args:**

    -   estimator: Linear model regression classifier object
    -   X: a data frame of normalized values from developmental dataset

    **Yields:**

    -   vector: Development set predicted labels

9.  Developmental Test Set R2 Score

    `dev_score_r2`

    Calculates the r2 score from the developmental dataset
    predictions.

    **Args:**

    -   estimator: Linear model regressor object
    -   X: a data frame of normalized values from developmental dataset
    -   Y: a vector of sample labels from developmental dataset

    **Yields:**

    -   float: r2 score

10. Developmental Test Set Mean Squared Error Score

    `dev_score_mse`

    Calculates the mean squared error score from the developmental dataset
    predictions.

    **Args:**

    -   estimator: Linear model regressor object
    -   X: a data frame of normalized values from developmental dataset
    -   Y: a vector of sample labels from developmental dataset

    **Yields:**

    -   float: mean squared error score

11. Developmental Test Set Explained Variance Score

    `dev_score_evar`

    Calculates the explained variance score for the develomental dataset predictions.

    **Args:**

    -   estimator: Linear model regressor object
    -   X: a data frame of normalized values from developmental dataset
    -   Y: a vector of sample labels from developmental data set

    **Yields:**

    -   float: explained variance score

12.  DEV Accuracy Score

    `dev_score_accuracy`

    Calculates the accuracy score from the DEV predictions.

    **Args:**

    -   estimator: Linear model classifier object
    -   X: a data frame of normalized values from developmental dataset
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: accuracy score

13.  DEV Normalized Mutual Information Score

    `dev_score_nmi`

    Calculates the normalized mutual information score from the DEV predictions.

    **Args:**

    -   estimator: Linear model classifier object
    -   X: a data frame of normalized values from developmental dataset
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: normalized mutual information score

14.  DEV Area Under ROC Curve Score

    `dev_score_roc`

    Calculates the area under the ROC curve score for the DEV predictions.

    **Args:**

    -   estimator: Linear model classifier object
    -   X: a data frame of normalized values from developmental dataset
    -   Y: a vector of sample labels from training data set

    **Yields:**

    -   float: AUC ROC score


<a id="orgbda21bf"></a>

### Linear model helper functions

1.  dRFE Subfunction

    `regr_fe`

    Iterate over features to by eliminated by step.

    **Args:**

    -   estimator: regressor or classifier linear model object
    -   X: a data frame of training data
    -   Y: a vector of sample labels from training data set
    -   n_features_iter: iterator for number of features to keep loop
    -   features: a vector of feature names
    -   fold: current fold
    -   out_dir: output directory. default '.'
    -   dev_size: developmental test set propotion of training
    -   SEED: random state
    -   RANK: Boolean (True or False)

    **Yields:**

    -   list: a list with number of features, r2 score, mean square error, expalined variance, and array of the indices for features to keep

2.  dRFE Step function

    `regr_fe_step`

    Split training data into developmental dataset and apply estimator
    to developmental dataset, rank features, and conduct feature
    elimination, single steps.

    **Args:**

    -   estimator: regressor or classifier linear model object
    -   X: a data frame of training data
    -   Y: a vector of sample labels from training data set
    -   n_features_to_keep: number of features to keep
    -   features: a vector of feature names
    -   fold: current fold
    -   out_dir: output directory. default '.'
    -   dev_size: developmental test set propotion of training
    -   SEED: random state
    -   RANK: Boolean (True or False)

    **Yields:**

    -   dict: a dictionary with number of features, r2 score, mean square error, expalined variance, and selected features
