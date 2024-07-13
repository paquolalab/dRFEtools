Classification Example
======================

**General application of dRFEtools, version 0.3.0+**

.. code:: python

   import numpy as np
   import pandas as pd
   import os,errno,dRFEtools
   from sklearn.model_selection import KFold
   from sklearn.datasets import make_regression
   from sklearn.datasets import make_classification
   from sklearn.model_selection import StratifiedKFold
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, roc_auc_score
   from sklearn.metrics import normalized_mutual_info_score as nmi

.. code:: python

   dRFEtools.__version__ 

::

   '0.3.4'

Define functions to analyze cross-validation
--------------------------------------------

Function
~~~~~~~~

.. code:: python

   def dynamicRFE(estimator, x_train, x_test, y_train, y_test, fold, outdir, RF):
       # Extract feature names / generate unique features names for ranking
       features = ["feature_%d" % x for x in range(x_train.shape[1])]
       if RF:
           # Run dynamic RFE for random forest classification
           d, pfirst = dRFEtools.rf_rfe(estimator, x_train, y_train, np.array(features),
                                        fold, outdir, RANK=False) ## Using default values
       else:
           # Run dynamic RFE for all other models
           d, pfirst = dRFEtools.dev_rfe(estimator, x_train, y_train, np.array(features),
                                        fold, outdir, RANK=False) ## Do not rank
       df_elim = pd.DataFrame([{'fold':fold, 'elimination': 0.2, 'n features':k,
                                'NMI score':d[k][1], 'Accuracy score':d[k][2],
                                'ROC AUC score':d[k][3]} for k in d.keys()])
       n_features_max = max(d, key=lambda x: d[x][1])
       try:
           ## Max features from lowess curve
           n_features, _ = dRFEtools.extract_max_lowess(d) ## Using default value: 0.3
           n_redundant, _ = dRFEtools.extract_peripheral_lowess(d) ## Using default values
       except ValueError:
           ## For errors in lowess estimate
           n_features = n_features_max
           n_redundant = n_features
       ## Fit model
       estimator.fit(x_train, y_train)
       all_fts = estimator.predict(x_test)
       estimator.fit(x_train[:, d[n_redundant][4]], y_train)
       labels_pred_redundant = estimator.predict(x_test[:, d[n_redundant][4]])
       estimator.fit(x_train[:,d[n_features][4]], y_train)
       labels_pred = estimator.predict(x_test[:, d[n_features][4]])
       ## Output test predictions
       kwargs = {"average": "weighted"}
       pd.DataFrame({'fold': fold, "elimination": 0.2, 'real': y_test, 
                     'predict_all': all_fts, 'predict_max': labels_pred,
                     'predict_redundant': labels_pred_redundant})\
         .to_csv("%s/test_predictions.txt" % outdir, sep='\t', mode='a',
                 index=True, header=True if fold == 0 else False)
       ## Save results into dictionary for easy manipulation
       output = dict()
       output['fold'] = fold
       output['elimination'] = 0.2
       output['n_features'] = n_features
       output['n_redundant'] = n_redundant
       output['n_max'] = n_features_max
       if RF:
           output['train_nmi'] = dRFEtools.oob_score_nmi(estimator, y_train)
           output['train_acc'] = dRFEtools.oob_score_accuracy(estimator, y_train)
           output['train_roc'] = dRFEtools.oob_score_roc(estimator, y_train)
       else:
           output['train_nmi'] = dRFEtools.dev_score_nmi(estimator,
                                                         x_train[:,d[n_features][4]],y_train)
           output['train_acc'] = dRFEtools.dev_score_accuracy(estimator,
                                                              x_train[:,d[n_features][4]],
                                                              y_train)
           output['train_roc'] = dRFEtools.dev_score_roc(estimator,
                                                         x_train[:,d[n_features][4]],y_train)
       output['test_nmi'] = nmi(y_test, labels_pred, average_method="arithmetic")
       output['test_acc'] = accuracy_score(y_test, labels_pred)
       output['test_roc'] = roc_auc_score(y_test, labels_pred, **kwargs)
       metrics_df = pd.DataFrame.from_records(output, index=[0])\
                                .reset_index().drop('index', axis=1)
       return df_elim, metrics_df

Details on the main functions used
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   help(dRFEtools.rf_rfe)

::

   Help on function rf_rfe in module dRFEtools.dRFEtools:

   rf_rfe(estimator, X, Y, features, fold, out_dir='.', elimination_rate=0.2, RANK=True)
       Runs random forest feature elimination step over iterator process.
       
       Args:
       estimator: Random forest classifier object
       X: a data frame of training data
       Y: a vector of sample labels from training data set
       features: a vector of feature names
       fold: current fold
       out_dir: output directory. default '.'
       elimination_rate: percent rate to reduce feature list. default .2
       
       Yields:
       dict: a dictionary with number of features, normalized mutual
             information score, accuracy score, auc roc curve and array of the
             indexes for features to keep

.. code:: python

   help(dRFEtools.extract_max_lowess)

::

   Help on function extract_max_lowess in module dRFEtools.lowess_redundant:

   extract_max_lowess(d, frac=0.3, multi=False, acc=False)
       Extract max features based on rate of change of log10
       transformed lowess fit curve.
       
       Args:
       d: Dictionary from dRFE
       frac: Fraction for lowess smoothing. Default 3/10.
       
       Yields:
       int: number of peripheral features

.. code:: python

   help(dRFEtools.extract_peripheral_lowess)

::

   Help on function extract_peripheral_lowess in module dRFEtools.lowess_redundant:

   extract_peripheral_lowess(d, frac=0.3, step_size=0.02, multi=False, acc=False)
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

This function has been updated from the previous name
**extract_redundant_lowess**!

Generate classification simulation data
---------------------------------------

We will first generate binary classification data on the same class as
large-scale omics data.

1. We will assume a sample size of 500, which would be a large number of
   samples for most human tissues.
2. We will use a N of 20k for features. This is approximately the number
   of genes in a given region after removing low expression features.
3. Finally, we will do roughly 400 total informative (1:3 informative to
   redundant). This is assuming 2% of genes are significant for the
   phenotype of interest.

.. code:: python

   # Create a dataset with only 10 informative features
   X, y = make_classification(
       n_samples=500, n_features=20000, n_informative=100, n_redundant=300,
       n_repeated=0, n_classes=2, n_clusters_per_class=1, random_state=13,
       shuffle=False,
   )

Initialize stratified 5-fold cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

Running analysis
----------------

Define functions
~~~~~~~~~~~~~~~~

.. code:: python

   def mkdir_p(directory):
       try:
           os.makedirs(directory)
       except OSError as e:
           if e.errno != errno.EEXIST:
               raise

.. code:: python

   def dRFE_run(estimator, X, y, cv, outdir, RF=True):
       mkdir_p(outdir); fold = 0
       df_dict = pd.DataFrame(); output = pd.DataFrame()
       for train_index, test_index in cv.split(X, y):
           X_train, X_test = X[train_index, :], X[test_index, :]
           y_train, y_test = y[train_index], y[test_index]
           df_elim, metrics_df = dynamicRFE(estimator, X_train, X_test, 
                                            y_train, y_test, fold, outdir, RF)
           df_dict = pd.concat([df_dict, df_elim], axis=0)
           output = pd.concat([output, metrics_df], axis=0)
           fold += 1
       df_dict.to_csv(f"{outdir}/dRFE_simulation.tsv", sep='\t', 
                      index=False, header=True)
       output.to_csv(f"{outdir}/dRFE_simulation_metrics.tsv", 
                     sep='\t', index=False, header=True)

Logistic regression
~~~~~~~~~~~~~~~~~~~

.. code:: python

   from sklearn.linear_model import LogisticRegression

   outdir = "lr"
   clf = LogisticRegression(max_iter=1000, n_jobs=-1)
   dRFE_run(clf, X, y, cv, outdir, False)

::

   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
   STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

   Increase the number of iterations (max_iter) or scale the data as shown in:
       https://scikit-learn.org/stable/modules/preprocessing.html
   Please also refer to the documentation for alternative solver options:
       https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
     n_iter_i = _check_optimize_result(

.. code:: python

   pd.read_csv(f"{outdir}/dRFE_simulation_metrics.tsv", sep="\t")

.. container::

   .. raw:: html

      <style scoped>
          .dataframe tbody tr th:only-of-type {
              vertical-align: middle;
          }

          .dataframe tbody tr th {
              vertical-align: top;
          }

          .dataframe thead th {
              text-align: right;
          }
      </style>

   .. raw:: html

      <table border="1" class="dataframe">

   .. raw:: html

      <thead>

   .. raw:: html

      <tr style="text-align: right;">

   .. raw:: html

      <th>

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   elimination

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   fold

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_features

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_max

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_redundant

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_acc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_nmi

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_roc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_acc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_nmi

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_roc

   .. raw:: html

      </th>

   .. raw:: html

      </tr>

   .. raw:: html

      </thead>

   .. raw:: html

      <tbody>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   0

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   145

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   58

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   6553

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.88

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.472161

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.879552

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   1

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   20000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   73

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1372

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.86

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.415761

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.860000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   2

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   228

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   182

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   5242

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.82

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.320912

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.820000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   3

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   3

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   182

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   73

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   5242

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.91

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.564205

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.910000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   4

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   4

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1097

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   73

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   5242

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.84

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.365690

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.840000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      </tbody>

   .. raw:: html

      </table>

SGD (Stochastic Gradient Descent) Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from sklearn.linear_model import SGDClassifier

   outdir = "sgd_class"
   clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
   dRFE_run(clf, X, y, cv, outdir, False)

::

   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/linear_model/_stochastic_gradient.py:702: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.

.. code:: python

   pd.read_csv(f"{outdir}/dRFE_simulation_metrics.tsv", sep="\t")

.. container::

   .. raw:: html

      <style scoped>
          .dataframe tbody tr th:only-of-type {
              vertical-align: middle;
          }

          .dataframe tbody tr th {
              vertical-align: top;
          }

          .dataframe thead th {
              text-align: right;
          }
      </style>

   .. raw:: html

      <table border="1" class="dataframe">

   .. raw:: html

      <thead>

   .. raw:: html

      <tr style="text-align: right;">

   .. raw:: html

      <th>

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   elimination

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   fold

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_features

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_max

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_redundant

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_acc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_nmi

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_roc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_acc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_nmi

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_roc

   .. raw:: html

      </th>

   .. raw:: html

      </tr>

   .. raw:: html

      </thead>

   .. raw:: html

      <tbody>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   0

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   20000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1097

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   2683

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.81

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.300823

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.810524

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.9500

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.714946

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.950000

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   1

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   701

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   877

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   20000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.85

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.390494

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.850000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.9400

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.672832

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.940024

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   2

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   701

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   116

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   5242

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.88

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.478239

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.880000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.9250

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.630066

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.925198

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   3

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   3

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   92

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   286

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   4193

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.92

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.610964

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.920000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.9400

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.673663

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.940049

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   4

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   4

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   116

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   2683

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   6553

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.87

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.446329

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.870000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.9325

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.667614

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.932736

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      </tbody>

   .. raw:: html

      </table>

SVC linear kernel
~~~~~~~~~~~~~~~~~

.. code:: python

   from sklearn.svm import LinearSVC

   outdir = "svc"
   clf = LinearSVC(random_state=13)
   dRFE_run(clf, X, y, cv, outdir, False)

::

   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
   /home/kynon/.local/lib/python3.11/site-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.

.. code:: python

   pd.read_csv(f"{outdir}/dRFE_simulation_metrics.tsv", sep="\t")

.. container::

   .. raw:: html

      <style scoped>
          .dataframe tbody tr th:only-of-type {
              vertical-align: middle;
          }

          .dataframe tbody tr th {
              vertical-align: top;
          }

          .dataframe thead th {
              text-align: right;
          }
      </style>

   .. raw:: html

      <table border="1" class="dataframe">

   .. raw:: html

      <thead>

   .. raw:: html

      <tr style="text-align: right;">

   .. raw:: html

      <th>

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   elimination

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   fold

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_features

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_max

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_redundant

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_acc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_nmi

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_roc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_acc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_nmi

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_roc

   .. raw:: html

      </th>

   .. raw:: html

      </tr>

   .. raw:: html

      </thead>

   .. raw:: html

      <tbody>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   0

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   560

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   92

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   5242

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.90

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.533486

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.90036

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   1

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   20000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   46

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1372

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.85

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.390494

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.85000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   2

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   228

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1097

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   5242

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.79

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.260181

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.79000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   3

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   3

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   560

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1097

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   5242

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.91

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.569739

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.91000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   4

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   4

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   3354

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   8192

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   5242

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.85

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.390494

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.85000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1.0

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      </tbody>

   .. raw:: html

      </table>

Random forest classifier
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from sklearn.ensemble import RandomForestClassifier

   outdir = "rf_class"
   clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, random_state=13)
   dRFE_run(clf, X, y, cv, outdir, True)

.. code:: python

   pd.read_csv(f"{outdir}/dRFE_simulation_metrics.tsv", sep="\t")

.. container::

   .. raw:: html

      <style scoped>
          .dataframe tbody tr th:only-of-type {
              vertical-align: middle;
          }

          .dataframe tbody tr th {
              vertical-align: top;
          }

          .dataframe thead th {
              text-align: right;
          }
      </style>

   .. raw:: html

      <table border="1" class="dataframe">

   .. raw:: html

      <thead>

   .. raw:: html

      <tr style="text-align: right;">

   .. raw:: html

      <th>

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   elimination

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   fold

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_features

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_max

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   n_redundant

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_acc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_nmi

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   test_roc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_acc

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_nmi

   .. raw:: html

      </th>

   .. raw:: html

      <th>

   train_roc

   .. raw:: html

      </th>

   .. raw:: html

      </tr>

   .. raw:: html

      </thead>

   .. raw:: html

      <tbody>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   0

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   92

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   58

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   286

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.82

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.324322

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.820728

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.8225

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.325818

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.822500

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   1

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   1

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   92

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   145

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   358

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.80

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.278884

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.800000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.8350

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.354173

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.835046

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   2

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   116

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   58

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   560

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.86

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.421817

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.860000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.8400

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.369307

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.839821

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   3

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   3

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   116

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   145

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   228

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.83

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.344766

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.830000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.8600

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.416094

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.859946

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      <tr>

   .. raw:: html

      <th>

   4

   .. raw:: html

      </th>

   .. raw:: html

      <td>

   0.2

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   4

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   116

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   116

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   286

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.81

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.298752

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.810000

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.8800

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.471719

   .. raw:: html

      </td>

   .. raw:: html

      <td>

   0.880072

   .. raw:: html

      </td>

   .. raw:: html

      </tr>

   .. raw:: html

      </tbody>

   .. raw:: html

      </table>

Session information
-------------------

.. code:: python

   import session_info
   session_info.show()

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click to view session information

.. raw:: html

   </summary>

.. raw:: html

   <pre>
   -----
   dRFEtools           0.3.4
   numpy               1.24.3
   pandas              2.0.2
   session_info        1.0.0
   sklearn             1.2.2
   -----
   </pre>

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click to view modules imported as dependencies

.. raw:: html

   </summary>

.. raw:: html

   <pre>
   PIL                 9.5.0
   anyio               NA
   arrow               1.2.3
   asttokens           NA
   attr                22.2.0
   babel               2.12.1
   backcall            0.2.0
   cairo               1.23.0
   cffi                1.15.1
   chardet             5.1.0
   colorama            0.4.6
   comm                0.1.3
   cycler              0.10.0
   cython_runtime      NA
   dateutil            2.8.2
   debugpy             1.6.7
   decorator           5.1.1
   defusedxml          0.7.1
   executing           1.2.0
   fastjsonschema      NA
   fqdn                NA
   gi                  3.44.1
   gio                 NA
   glib                NA
   gobject             NA
   gtk                 NA
   idna                3.4
   ipykernel           6.23.1
   isoduration         NA
   jaraco              NA
   jedi                0.18.2
   jinja2              3.1.2
   joblib              1.2.0
   json5               NA
   jsonpointer         2.3
   jsonschema          4.17.3
   jupyter_events      0.6.3
   jupyter_server      2.6.0
   jupyterlab_server   2.22.1
   kiwisolver          1.4.4
   markupsafe          2.1.3
   matplotlib          3.7.1
   mizani              0.9.2
   more_itertools      9.1.0
   mpl_toolkits        NA
   nbformat            5.9.0
   ordered_set         4.1.0
   overrides           NA
   packaging           23.1
   parso               0.8.3
   patsy               0.5.3
   pexpect             4.8.0
   pickleshare         0.7.5
   pkg_resources       NA
   platformdirs        3.5.1
   plotnine            0.12.1
   prometheus_client   NA
   prompt_toolkit      3.0.38
   psutil              5.9.5
   ptyprocess          0.7.0
   pure_eval           0.2.2
   pydev_ipython       NA
   pydevconsole        NA
   pydevd              2.9.5
   pydevd_file_utils   NA
   pydevd_plugins      NA
   pydevd_tracing      NA
   pygments            2.15.1
   pyparsing           3.0.9
   pyrsistent          NA
   pythonjsonlogger    NA
   pytz                2023.3
   requests            2.28.2
   rfc3339_validator   0.1.4
   rfc3986_validator   0.1.1
   scipy               1.11.0
   send2trash          NA
   setuptools          68.0.0
   six                 1.16.0
   sniffio             1.3.0
   stack_data          0.6.2
   statsmodels         0.14.0
   threadpoolctl       3.1.0
   tornado             6.3.2
   traitlets           5.9.0
   typing_extensions   NA
   uri_template        NA
   urllib3             1.26.15
   wcwidth             0.2.6
   webcolors           1.13
   websocket           1.5.2
   yaml                6.0
   zmq                 25.1.0
   </pre>

.. raw:: html

   </details>

.. raw:: html

   <pre>
   -----
   IPython             8.14.0
   jupyter_client      8.2.0
   jupyter_core        5.3.0
   jupyterlab          4.0.1
   -----
   Python 3.11.3 (main, Jun  5 2023, 09:32:32) [GCC 13.1.1 20230429]
   Linux-6.3.8-arch1-1-x86_64-with-glibc2.37
   -----
   Session information updated at 2023-06-28 11:40
   </pre>

.. raw:: html

   </details>
