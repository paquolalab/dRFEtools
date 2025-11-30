import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

from pathlib import Path

import matplotlib.pyplot as plt

from dRFEtools.utils import (
    ensure_path,
    get_feature_importances,
    normalize_rfe_result,
    save_plot_variants,
)


def test_normalize_rfe_result_tuple_conversion():
    with pytest.warns(DeprecationWarning):
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


def test_normalize_rfe_result_dict_passthrough():
    payload = {"n_features": 3, "metrics": {"nmi_score": 0.1}}
    assert normalize_rfe_result(payload) == payload


def test_normalize_rfe_result_invalid_type():
    with pytest.raises(TypeError):
        normalize_rfe_result("not-a-result")


def test_get_feature_importances_linear_regression():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = LinearRegression().fit(X, y)
    importances = get_feature_importances(model)
    assert importances.shape == (2,)
    assert np.all(importances >= 0)


def test_get_feature_importances_multidimensional_coef():
    class DummyEstimator:
        def __init__(self):
            self.coef_ = np.array([[1.0, -1.0, 0.5], [2.0, 0.0, 0.0]])

    norms = get_feature_importances(DummyEstimator())
    assert np.allclose(norms, np.array([3.0, 1.0, 0.5]))


def test_get_feature_importances_feature_importances_attribute():
    class FeatureImportanceEstimator:
        def __init__(self):
            self.feature_importances_ = np.array([0.1, 0.2, 0.3])

    importances = get_feature_importances(FeatureImportanceEstimator())
    assert importances.tolist() == [0.1, 0.2, 0.3]


def test_save_plot_variants_matplotlib(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    output_base = tmp_path / "plot"
    save_plot_variants(fig, output_base)

    for ext in (".svg", ".png", ".pdf"):
        assert (output_base.with_suffix(ext)).exists()


def test_ensure_path_resolves_user_and_relative(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    relative = home / "sub" / ".." / "file.txt"
    resolved = ensure_path(str(relative))
    assert resolved == (home / "file.txt").resolve()
    assert isinstance(resolved, Path)
