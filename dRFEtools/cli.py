"""Command-line interface for :mod:`dRFEtools`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from . import __version__, dev_rfe, rf_rfe

CLASSIFICATION_METRICS = {
    "nmi": 1,
    "accuracy": 2,
    "roc_auc": 3,
}
REGRESSION_METRICS = {
    "r2": 1,
    "mse": 2,
    "explained_variance": 3,
}
MINIMIZE = {"mse"}


def _load_dataset(
    data_path: Path, target: str
) -> Tuple[pd.DataFrame, pd.Series, Iterable[str]]:
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {data_path}")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y, X.columns


def _resolve_metric(task: str, metric: str | None) -> str:
    if task == "classification":
        allowed = CLASSIFICATION_METRICS
        default = "nmi"
    else:
        allowed = REGRESSION_METRICS
        default = "r2"

    if metric is None:
        return default
    if metric not in allowed:
        valid = ", ".join(sorted(allowed))
        raise ValueError(
            f"Metric '{metric}' is not valid for task '{task}'. Choose from: {valid}"
        )
    return metric


def _metric_index(task: str, metric: str) -> int:
    return (CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS)[
        metric
    ]


def _summarize_results(
    results: Dict[int, Tuple], task: str, metric: str
) -> Tuple[int, float]:
    idx = _metric_index(task, metric)
    comparator = min if metric in MINIMIZE else max
    best = comparator(results.values(), key=lambda record: record[idx])
    return best[0], best[idx]


def _results_frame(results: Dict[int, Tuple], task: str) -> pd.DataFrame:
    if task == "classification":
        columns = ["n_features", "nmi", "accuracy", "roc_auc", "indices"]
    else:
        columns = ["n_features", "r2", "mse", "explained_variance", "indices"]
    ordered = sorted(results.values(), key=lambda record: record[0])
    return pd.DataFrame(ordered, columns=columns)


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_rf_rfe(args: argparse.Namespace) -> None:
    X, y, features = _load_dataset(Path(args.data), args.target)
    estimator: RandomForestClassifier | RandomForestRegressor
    if args.task == "classification":
        estimator = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            oob_score=True,
            n_jobs=args.n_jobs,
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            oob_score=True,
            n_jobs=args.n_jobs,
        )

    results, first_step = rf_rfe(
        estimator,
        X,
        y,
        features,
        args.fold,
        out_dir=str(_ensure_output_dir(Path(args.output_dir))),
        elimination_rate=args.elimination_rate,
        RANK=args.rank,
    )

    metric = _resolve_metric(args.task, args.metric)
    best_n, best_score = _summarize_results(results, args.task, metric)

    print(f"First elimination step retained {first_step[0]} features.")
    direction = "lowest" if metric in MINIMIZE else "highest"
    print(
        f"Best {direction} {metric} achieved with {best_n} features: {best_score:.4f}"
    )

    if args.save_summary:
        summary_path = Path(args.save_summary)
        _ensure_output_dir(summary_path.parent)
        _results_frame(results, args.task).to_csv(summary_path, index=False)
        print(f"Saved summary metrics to {summary_path}")


def run_dev_rfe(args: argparse.Namespace) -> None:
    X, y, features = _load_dataset(Path(args.data), args.target)
    estimator = (
        LogisticRegression(max_iter=1000)
        if args.task == "classification"
        else LinearRegression()
    )

    results, first_step = dev_rfe(
        estimator,
        X,
        y,
        features,
        args.fold,
        out_dir=str(_ensure_output_dir(Path(args.output_dir))),
        elimination_rate=args.elimination_rate,
        dev_size=args.dev_size,
        RANK=args.rank,
        SEED=args.seed,
    )

    metric = _resolve_metric(args.task, args.metric)
    best_n, best_score = _summarize_results(results, args.task, metric)

    print(f"First elimination step retained {first_step[0]} features.")
    direction = "lowest" if metric in MINIMIZE else "highest"
    print(
        f"Best {direction} {metric} achieved with {best_n} features: {best_score:.4f}"
    )

    if args.save_summary:
        summary_path = Path(args.save_summary)
        _ensure_output_dir(summary_path.parent)
        _results_frame(results, args.task).to_csv(summary_path, index=False)
        print(f"Saved summary metrics to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dynamic recursive feature elimination workflows."
    )
    parser.add_argument(
        "--version", action="version", version=f"dRFEtools {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = {
        "data": dict(help="Path to a CSV file containing features and target column."),
        "target": dict(help="Name of the target column to predict."),
        "task": dict(
            choices=["classification", "regression"], default="classification"
        ),
        "output_dir": dict(default=".", help="Directory to write ranking artifacts."),
        "elimination_rate": dict(
            type=float, default=0.2, help="Fraction of features removed per iteration."
        ),
        "metric": dict(default=None, help="Metric used to pick the best iteration."),
        "fold": dict(
            type=int, default=1, help="Fold identifier used in saved outputs."
        ),
        "rank": dict(
            action="store_true",
            help="Persist feature ranking files during elimination.",
        ),
        "save_summary": dict(
            default=None, help="Optional path to write a CSV of iteration metrics."
        ),
    }

    rf_parser = subparsers.add_parser("rf-rfe", help="Run random-forest-based dRFE.")
    rf_parser.set_defaults(func=run_rf_rfe)
    rf_parser.add_argument("--data", required=True, **common["data"])
    rf_parser.add_argument("--target", required=True, **common["target"])
    rf_parser.add_argument("--task", **common["task"])
    rf_parser.add_argument("--output-dir", **common["output_dir"])
    rf_parser.add_argument("--elimination-rate", **common["elimination_rate"])
    rf_parser.add_argument("--metric", **common["metric"])
    rf_parser.add_argument("--fold", **common["fold"])
    rf_parser.add_argument("--rank", **common["rank"])
    rf_parser.add_argument("--save-summary", **common["save_summary"])
    rf_parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the random forest.",
    )
    rf_parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of jobs used by the estimator."
    )
    rf_parser.add_argument(
        "--random-state", type=int, default=13, help="Random state for reproducibility."
    )

    dev_parser = subparsers.add_parser(
        "dev-rfe", help="Run development-set-based dRFE."
    )
    dev_parser.set_defaults(func=run_dev_rfe)
    dev_parser.add_argument("--data", required=True, **common["data"])
    dev_parser.add_argument("--target", required=True, **common["target"])
    dev_parser.add_argument("--task", **common["task"])
    dev_parser.add_argument("--output-dir", **common["output_dir"])
    dev_parser.add_argument("--elimination-rate", **common["elimination_rate"])
    dev_parser.add_argument("--metric", **common["metric"])
    dev_parser.add_argument("--fold", **common["fold"])
    dev_parser.add_argument("--rank", **common["rank"])
    dev_parser.add_argument("--save-summary", **common["save_summary"])
    dev_parser.add_argument(
        "--dev-size",
        type=float,
        default=0.2,
        help="Fraction reserved for the development split.",
    )
    dev_parser.add_argument(
        "--seed", action="store_true", help="Use a deterministic train/dev split."
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
