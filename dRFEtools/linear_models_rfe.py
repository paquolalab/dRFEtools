"""
This script contains the classes for linear models to use for dynamic recursive
feature elmination (dRFE). This calculates feature importance from model
coefficients. Current linear models supported: Logistic and linear regression,
Lasso, and Elastic net.

Developed by Kynon Jade Benjamin.
"""

__author__ = 'Kynon J Benjamin'

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class Lasso(Lasso):
    """
    Add feature importance to Lasso class similar to
    random forest output.
    """
    def fit(self, *args, **kwargs):
        super(Lasso, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class LassoCV(LassoCV):
    """
    Add feature importance to Lasso class similar to
    random forest output. This has been updated to use
    CV for alpha tuning.
    """
    def fit(self, *args, **kwargs):
        super(LassoCV, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class Ridge(Ridge):
    """
    Add feature importance to Ridge class similar to
    random forest output.
    """
    def fit(self, *args, **kwargs):
        super(Ridge, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class RidgeCV(RidgeCV):
    """
    Add feature importance to Ridge class similar to
    random forest output. This has been updated to use
    CV for alpha tuning.
    """
    def fit(self, *args, **kwargs):
        super(RidgeCV, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class ElasticNet(ElasticNet):
    """
    Add feature importance to ElasticNet class similar to
    random forest output.
    """
    def fit(self, *args, **kwargs):
        super(ElasticNet, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class ElasticNetCV(ElasticNetCV):
    """
    Add feature importance to ElasticNet class similar to
    random forest output. This uses cross-validation to chose alpha.
    """
    def fit(self, *args, **kwargs):
        super(ElasticNetCV, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class LogisticRegression(LogisticRegression):
    """
    Add feature importance to Logistic Regression class similar to
    random forest output.
    """
    def fit(self, *args, **kwargs):
        super(LogisticRegression, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


class LinearRegression(LinearRegression):
    """
    Add feature importance to Linear Regression class similar to
    random forest output.
    """
    def fit(self, *args, **kwargs):
        super(LinearRegression, self).fit(*args, **kwargs)
        self.feature_importances_ = np.abs(self.coef_).flatten()


def calculate_alpha(X, Y):
    """
    Using cross-validation with developmental set to learn
    alpha. This is for elastic net optimization. Note that
    ElasticNet preforms this automatically.

    Args:
    X: a data frame of normalized values from training dataset
    Y: a vector of sample labels from training data set

    Yields:
    alpha: alpha value maximized for elastic net
    """
    kwargs = {'test_size': 0.3}
    X1, X2, Y1, Y2 = train_test_split(X, Y, **kwargs)
    regr = ElasticNetCV(cv=5, random_state=13).fit(X2, Y2)
    return regr.alpha_
