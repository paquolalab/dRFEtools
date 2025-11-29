from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from dRFEtools.dRFEtools import _normalize_metrics, _n_features_iter


def test_n_features_iter_respects_keep_rate():
    sequence = list(_n_features_iter(10, 0.5))
    assert sequence[-1] == 1
    assert sequence[0] == 5
    assert all(n > 0 for n in sequence)


def test_normalize_metrics_classifier_subset():
    metrics = {"nmi_score": 0.1, "accuracy_score": 0.2, "roc_auc_score": 0.3, "r2_score": 0.4}
    normalized = {"metrics": metrics}

    clf_metrics = _normalize_metrics(SVC(), normalized)
    assert clf_metrics == {"nmi_score": 0.1, "accuracy_score": 0.2, "roc_auc_score": 0.3}


def test_normalize_metrics_random_forest_classifier_uses_same_subset():
    metrics = {"nmi_score": 0.5, "accuracy_score": 0.6, "roc_auc_score": 0.7, "mse_score": 0.8}
    normalized = {"metrics": metrics}

    rf_metrics = _normalize_metrics(RandomForestClassifier(), normalized)
    assert rf_metrics == {"nmi_score": 0.5, "accuracy_score": 0.6, "roc_auc_score": 0.7}


def test_normalize_metrics_regressor_subset():
    metrics = {"r2_score": 0.9, "mse_score": 1.0, "explain_var": 1.1, "accuracy_score": 0.2}
    normalized = {"metrics": metrics}

    reg_metrics = _normalize_metrics(LinearRegression(), normalized)
    assert reg_metrics == {"r2_score": 0.9, "mse_score": 1.0, "explain_var": 1.1}
