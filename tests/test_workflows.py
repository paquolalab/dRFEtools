"""Exercise scientific workflows and image output with the patched dependencies."""

import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from dRFEtools import dev_rfe, plot_metric, rf_rfe


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("workflow", ["rf", "dev"])
def test_rfe_workflow_and_plot(task, workflow, tmp_path):
    if task == "classification":
        X, y = make_classification(n_samples=100, n_features=6, random_state=42)
        estimator = (
            RandomForestClassifier(n_estimators=64, oob_score=True, random_state=42)
            if workflow == "rf" else LogisticRegression(random_state=42)
        )
        metrics = {"nmi_score", "accuracy_score", "roc_auc_score"}
        plot_name = "acc"
    else:
        X, y = make_regression(n_samples=100, n_features=6, random_state=42)
        estimator = (
            RandomForestRegressor(n_estimators=64, oob_score=True, random_state=42)
            if workflow == "rf" else LinearRegression()
        )
        metrics = {"r2_score", "mse_score", "explain_var"}
        plot_name = "r2"
    run = rf_rfe if workflow == "rf" else dev_rfe
    kwargs = {"random_state": 42} if workflow == "dev" else {}
    history, first = run(
        estimator, X, y, np.array([f"feature_{i}" for i in range(6)]),
        fold=0, out_dir=tmp_path, elimination_rate=0.5, RANK=False, **kwargs,
    )
    assert list(history) == [6, 3, 1]
    assert first is history[6]
    for count, result in history.items():
        assert result["n_features"] == count
        assert len(result["indices"]) == count
        assert set(result["metrics"]) == metrics
        assert np.isfinite(list(result["metrics"].values())).all()
    plot_metric(history, 0, tmp_path, plot_name, "Score")
    for extension in ("svg", "png", "pdf"):
        assert (tmp_path / f"{plot_name}_fold_0.{extension}").stat().st_size > 0
    with Image.open(tmp_path / f"{plot_name}_fold_0.png") as image:
        image.verify()
