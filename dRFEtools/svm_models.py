"""
This script contains the classes for support vector machines (SVMs) and
Stochastic Gradient Descent (SGD) models to use for dynamic recursive feature
elmination (dRFE). SGD are includes as they can optimize the same cost function
as linear SVMs by adjusting the penalty and loss parameters. Additionally, SGDs
require less memoiry, allows incremental learning, and implements various loss
functions and regularization. This script specifically calculates feature
importance for the linear from model coefficients.

Developed by Kynon Jade Benjamin.
"""

__author__ = 'Kynon J Benjamin'

import numpy as np
from sklearn.svm import LinearSVC,LinearSVR
from sklearn.linear_model import SGDClassifier,SGDRegressor

class LinearSVC(LinearSVC):
    """
    Add feature importance to linear SVC class similar to
    random forest output.
    """
    def fit(self, *args, **kwargs):
        super(LinearSVC, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class SGDClassifier(SGDClassifier):
    """
    Add feature importance to stochastic gradient descent classification
    class similar to random forest output.
    """
    def fit(self, *args, **kwargs):
        super(SGDClassifier, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class LinearSVR(LinearSVR):
    """
    Add feature importance to linear SVR class similar to
    random forest output.
    """
    def fit(self, *args, **kwargs):
        super(LinearSVR, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class SGDRegressor(SGDRegressor):
    """
    Add feature importance to stochastic gradient descent regression
    class similar to random forest output.
    """
    def fit(self, *args, **kwargs):
        super(SGDRegressor, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()
