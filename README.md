# dRFEtools

`dRFEtools` is a package for dynamic recursive feature elimination supports random
forest classification and regresssion, and linear regression models.

Authors: Apuã Paquola, Kynon Jade Benjamin, and Tarun Katipalli

## Citation
If using please cite: XXX.

## Table of Contents

1.  [Installation](#orgc61a26f)
2.  [Reference Manual](#orged7f79c)
    1.  [Feature Elimination Main](#orgdcb05e0)
    2.  [Feature Rank Function](#org6d007c2)
    3.  [N Feature Iterator](#orgf676484)
    4.  [OOB Prediction](#org6ef4a54)
    5.  [OOB Accuracy Score](#orgd8c0565)
    6.  [OOB Normalized Mutual Information Score](#org2bede4d)
    7.  [OOB Area Under ROC Curve Score](#org1a39bf3)
    8.  [Plot Feature Elimination by Accuracy](#org828dfec)
    9.  [Plot Feature Elimination by NMI](#org075f47f)
    10. [Plot Feature Elimination by AUC](#orgd34b506)
    11. [Feature Elimination Subfunction](#org31603e5)
    12. [Feature Elimination Step](#org45393f3)
    13. [Logistic Regression Class](#orgb71b427)
    14. [Developmental Test Set Predictions](#org8e341e8)
    15. [Developmental Test Set Accuracy Score](#org4347e30)
    16. [Developmental Test Set NMI Score](#org2f60132)
    17. [Developmental Test Set ROC Score](#orgc8af4ad)
    18. [Logistic Regression Feature Elimination Subfunction](#orgc36d680)
    19. [Logistic Regression Feature Elimination Step](#org82e515d)

<a id="orgc61a26f"></a>

## Installation

`pip install --user dRFEtools`


<a id="orged7f79c"></a>

## Reference Manual

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Function</td>
<td class="org-left">Description</td>
</tr>


<tr>
<td class="org-left">feature<sub>elimination</sub></td>
<td class="org-left">Runs random forest classification feature elimination</td>
</tr>


<tr>
<td class="org-left">features<sub>rank</sub><sub>fnc</sub></td>
<td class="org-left">Rank features</td>
</tr>


<tr>
<td class="org-left">n<sub>features</sub><sub>iter</sub></td>
<td class="org-left">Determines the features to keep</td>
</tr>


<tr>
<td class="org-left">oob<sub>predictions</sub></td>
<td class="org-left">Extracts out-of-bag (OOB) predictions from random forest classifier classes</td>
</tr>


<tr>
<td class="org-left">oob<sub>score</sub><sub>accuracy</sub></td>
<td class="org-left">Calculates the accuracy score for the OOB predictions</td>
</tr>


<tr>
<td class="org-left">oob<sub>score</sub><sub>nmi</sub></td>
<td class="org-left">Calculates the normalized mutual information score for the OOB predictions</td>
</tr>


<tr>
<td class="org-left">oob<sub>score</sub><sub>roc</sub></td>
<td class="org-left">Calculates the area under the ROC curve (AUC) for the OOB predictions</td>
</tr>


<tr>
<td class="org-left">plot<sub>acc</sub></td>
<td class="org-left">Plot feature elimination with accuracy as measurement</td>
</tr>


<tr>
<td class="org-left">plot<sub>nmi</sub></td>
<td class="org-left">Plot feature elimination with NMI as measurement</td>
</tr>


<tr>
<td class="org-left">plot<sub>roc</sub></td>
<td class="org-left">Plot feature elimination with AUC ROC curve as measurement</td>
</tr>


<tr>
<td class="org-left">rf<sub>fe</sub></td>
<td class="org-left">Iterate over features to be eliminated</td>
</tr>


<tr>
<td class="org-left">rf<sub>fe</sub><sub>step</sub></td>
<td class="org-left">Apply random forest to training data, rank features, and conduct feature elimination (single step)</td>
</tr>


<tr>
<td class="org-left">dev<sub>predictions</sub></td>
<td class="org-left">Uses development testing set to predict logistic regression classifier classes</td>
</tr>


<tr>
<td class="org-left">dev<sub>score</sub><sub>accuracy</sub></td>
<td class="org-left">Calculates the accuracy score for the developmental test set predictions</td>
</tr>


<tr>
<td class="org-left">dev<sub>score</sub><sub>nmi</sub></td>
<td class="org-left">Calculates the normalized mutual information score for the developmental test set predictions</td>
</tr>


<tr>
<td class="org-left">dev<sub>score</sub><sub>roc</sub></td>
<td class="org-left">Calculates the area under the ROC curve (AUC) for the developmental test set predictions</td>
</tr>


<tr>
<td class="org-left">lr<sub>fe</sub></td>
<td class="org-left">Iterate over features to be eliminated, logistic regression</td>
</tr>


<tr>
<td class="org-left">lr<sub>fe</sub><sub>step</sub></td>
<td class="org-left">Apply logistic regression to training data, split to developmental test set, rank features, and conduct feature elimination (single step)</td>
</tr>
</tbody>
</table>


<a id="orgdcb05e0"></a>

### Feature Elimination Main

`feature_elimination`

Runs random forest feature elimination step over iterator process.

**Args:**

-   estimator: Random forest classifier object
-   X: a data frame of training data
-   Y: a vector of sample labels from training data set
-   features: a vector of feature names
-   fold: current fold
-   out<sub>dir</sub>: output directory. default '.'
-   elimination<sub>rate</sub>: percent rate to reduce feature list. default .2
-   RANK: Output feature ranking. default=True (Boolean)

**Yields:**

-   dict: a dictionary with number of features, normalized mutual information score, accuracy score, and array of the indexes for features to keep


<a id="org6d007c2"></a>

### Feature Rank Function

`feature_rank_fnc`

Ranks features.

**Args:**

-   features: A vector of feature names
-   rank: A vector with feature ranks based on absolute value of feature importance
-   n<sub>features</sub><sub>to</sub><sub>keep</sub>: Number of features to keep. (Int)
-   fold: Fold to analyzed. (Int)
-   out<sub>dir</sub>: Output directory for text file. Default '.'
-   RANK: Boolean (True or False)

**Yields:**

-   Text file: Ranked features by fold tab-delimited text file, only if RANK=True


<a id="orgf676484"></a>

### N Feature Iterator

`n_features_iter`

Determines the features to keep.

**Args:**

-   nf: current number of features
-   keep<sub>rate</sub>: percentage of features to keep

**Yields:**

-   int: number of features to keep


<a id="org6ef4a54"></a>

### OOB Prediction

`oob_predictions`

Extracts out-of-bag (OOB) predictions from random forest classifier classes.

**Args:**

-   estimator: Random forest classifier object

**Yields:**

-   vector: OOB predicted labels


<a id="orgd8c0565"></a>

### OOB Accuracy Score

`oob_score_accuracy`

Calculates the accuracy score from the OOB predictions.

**Args:**

-   estimator: Random forest classifier object
-   Y: a vector of sample labels from training data set

**Yields:**

-   float: accuracy score


<a id="org2bede4d"></a>

### OOB Normalized Mutual Information Score

`oob_score_nmi`

Calculates the normalized mutual information score from the OOB predictions.

**Args:**

-   estimator: Random forest classifier object
-   Y: a vector of sample labels from training data set

**Yields:**

-   float: normalized mutual information score


<a id="org1a39bf3"></a>

### OOB Area Under ROC Curve Score

`oob_score_roc`

Calculates the area under the ROC curve score for the OOB predictions.

**Args:**

-   estimator: Random forest classifier object
-   Y: a vector of sample labels from training data set

**Yields:**

-   float: AUC ROC score


<a id="org828dfec"></a>

### Plot Feature Elimination by Accuracy

`plot_acc`

Plot feature elimination results for accuracy.

**Args:**

-   d: feature elimination class dictionary
-   fold: current fold
-   out<sub>dir</sub>: output directory. default '.'

**Yields:**

-   graph: plot of feature by accuracy, automatically saves files as png and svg


<a id="org075f47f"></a>

### Plot Feature Elimination by NMI

`plot_nmi`

Plot feature elimination results for normalized mutual information.

**Args:**

-   d: feature elimination class dictionary
-   fold: current fold
-   out<sub>dir</sub>: output directory. default '.'

**Yields:**

-   graph: plot of feature by NMI, automatically saves files as png and svg


<a id="orgd34b506"></a>

### Plot Feature Elimination by AUC

`plot_roc`

Plot feature elimination results for AUC ROC curve.

**Args:**

-   d: feature elimination class dictionary
-   fold: current fold
-   out<sub>dir</sub>: output directory. default '.'

**Yields:**

-   graph: plot of feature by AUC, automatically saves files as png and svg


<a id="org31603e5"></a>

### Feature Elimination Subfunction

`rf_fe`

Iterate over features to by eliminated by step.

**Args:**

-   estimator: Random forest classifier object
-   X: a data frame of training data
-   Y: a vector of sample labels from training data set
-   n<sub>features</sub><sub>iter</sub>: iterator for number of features to keep loop
-   features: a vector of feature names
-   fold: current fold
-   out<sub>dir</sub>: output directory. default '.'
-   RANK: Boolean (True or False)

**Yields:**

-   list: a list with number of features, normalized mutual information score, accuracy score, and array of the indices for features to keep


<a id="org45393f3"></a>

### Feature Elimination Step

`rf_fe_step`

Apply random forest to training data, rank features, conduct feature elimination.

**Args:**

-   estimator: Random forest classifier object
-   X: a data frame of training data
-   Y: a vector of sample labels from training data set
-   n<sub>features</sub><sub>to</sub><sub>keep</sub>: number of features to keep
-   features: a vector of feature names
-   fold: current fold
-   out<sub>dir</sub>: output directory. default '.'
-   RANK: Boolean (True or False)

**Yields:**

-   dict: a dictionary with number of features, normalized mutual information score, accuracy score, and selected features


<a id="orgb71b427"></a>

### Logistic Regression Class

`LogisticRegression_FI`

Add feature importance to Logistic Regression class similar to
random forest output.


<a id="org8e341e8"></a>

### Developmental Test Set Predictions

`dev_predictions`

Extracts predictions using a development fold for logistic
regression classifier classes.

**Args:**

-   estimator: Logistic regression classifier object
-   X: a data frame of normalized values from developmental dataset

**Yields:**

-   vector: Development set predicted labels


<a id="org4347e30"></a>

### Developmental Test Set Accuracy Score

`dev_score_acc`

Calculates the accuracy score from the developmental dataset
predictions.

**Args:**

-   estimator: Random forest classifier object
-   X: a data frame of normalized values from developmental dataset
-   Y: a vector of sample labels from developmental dataset

**Yields:**

-   float: accuracy score


<a id="org2f60132"></a>

### Developmental Test Set NMI Score

`dev_score_nmi`

Calculates the normalized mutual information score
from the developmental dataset predictions.

**Args:**

-   estimator: Random forest classifier object
-   X: a data frame of normalized values from developmental dataset
-   Y: a vector of sample labels from developmental dataset

**Yields:**

-   float: normalized mutual information score


<a id="orgc8af4ad"></a>

### Developmental Test Set ROC Score

`dev_score_roc`

Calculates the area under the ROC curve score
for the develomental dataset predictions.

**Args:**

-   estimator: Logistic regression classifier object
-   X: a data frame of normalized values from developmental dataset
-   Y: a vector of sample labels from developmental data set

**Yields:**

-   float: AUC ROC score


<a id="orgc36d680"></a>

### Logistic Regression Feature Elimination Subfunction

`lr_fe`

Iterate over features to by eliminated by step.

**Args:**

-   estimator: Logistic regression classifier object
-   X: a data frame of training data
-   Y: a vector of sample labels from training data set
-   n<sub>features</sub><sub>iter</sub>: iterator for number of features to keep loop
-   features: a vector of feature names
-   fold: current fold
-   out<sub>dir</sub>: output directory. default '.'
-   dev<sub>size</sub>: developmental test set propotion of training. default '0.20'
-   SEED: random state. default 'True'
-   RANK: run feature ranking. default 'True'

**Yields:**

-   list: a list with number of features, normalized mutual information score, accuracy score, auc roc curve and array of the indices for features to keep


<a id="org82e515d"></a>

### Logistic Regression Feature Elimination Step

`lr_fe_step`

Split training data into developmental dataset and apply
logistic regression to developmental dataset, rank features,
and conduct feature elimination, single steps.

**Args:**

-   estimator: Logistic regression classifier object
-   X: a data frame of training data
-   Y: a vector of sample labels from training data set
-   n<sub>features</sub><sub>to</sub><sub>keep</sub>: number of features to keep
-   features: a vector of feature names
-   fold: current fold
-   out<sub>dir</sub>: output directory. default '.'
-   dev<sub>size</sub>: developmental size. default '0.20'
-   SEED: random state. default 'True'
-   RANK: run feature ranking. default 'True'

**Yields:**

-   dict: a dictionary with number of features, normalized mutual information score, accuracy score, auc roc score and selected features
