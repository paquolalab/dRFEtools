import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

from dRFEtools.utils import get_feature_importances, normalize_rfe_result


def test_normalize_rfe_result_tuple_conversion():
    result = normalize_rfe_result((5, 0.5, 0.6, np.array([0, 1, 2])))
    assert result["n_features"] == 5
    assert set(result["metrics"].keys()) == {
        "nmi_score",
        "r2_score",
        "accuracy_score",
        "mse_score",
        "roc_auc_score",
        "explain_var",
    }
    assert isinstance(result["indices"], np.ndarray) is False


def test_get_feature_importances_linear_regression():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = LinearRegression().fit(X, y)
    importances = get_feature_importances(model)
    assert importances.shape == (2,)
    assert np.all(importances >= 0)
