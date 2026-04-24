#!/usr/bin/env python

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from ads.opctl.operator.lowcode.regression.__main__ import operate
from ads.opctl.operator.lowcode.regression.deployment.deployment_manager import (
    ModelDeploymentManager,
)
from ads.opctl.operator.lowcode.regression.deployment.score import predict as deployment_predict
from ads.opctl.operator.lowcode.regression.model.auto import (
    AutoRegressionOperatorModel,
)
from ads.opctl.operator.lowcode.regression.model.knn import (
    KNNRegressionOperatorModel,
)
from ads.opctl.operator.lowcode.regression.model.linear_regression import (
    LinearRegressionOperatorModel,
)
from ads.opctl.operator.lowcode.regression.model.random_forest import (
    RandomForestRegressionOperatorModel,
)
from ads.opctl.operator.lowcode.regression.model.xgboost import (
    XGBoostRegressionOperatorModel,
)
from ads.opctl.operator.lowcode.regression.model.regression_dataset import (
    RegressionDatasets,
)
from ads.opctl.operator.lowcode.regression.model.factory import (
    RegressionOperatorModelFactory,
)
from ads.opctl.operator.lowcode.regression.operator_config import RegressionOperatorConfig


class _DummyRegressionModel:
    def __init__(self, feature_columns):
        self.preprocessor = SimpleNamespace(feature_columns_=feature_columns)

    def predict(self, X):
        return np.zeros(len(X))


def test_random_forest_uses_robust_defaults_for_mae_metric():
    rows = 1200
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            f"x{i}": rng.normal(loc=i, scale=1.0, size=rows)
            for i in range(8)
        }
    )
    df["target"] = df.sum(axis=1) + rng.normal(0, 0.5, rows)

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "random_forest",
                "metric": "mae",
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = RandomForestRegressionOperatorModel(config, datasets)

    estimator = model._build_estimator()

    assert estimator.n_estimators == 400
    assert estimator.criterion == "absolute_error"
    assert estimator.bootstrap is True
    assert estimator.oob_score is True
    assert estimator.max_samples == 0.85
    assert estimator.max_features == "sqrt"
    assert estimator.min_samples_leaf == 3
    assert estimator.min_samples_split == 6
    assert estimator.random_state == 42


def test_random_forest_passes_native_model_kwargs_directly_to_estimator():
    rows = 300
    rng = np.random.default_rng(22)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
            "x4": rng.normal(size=rows),
            "x5": rng.normal(size=rows),
            "target": rng.normal(size=rows),
        }
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "random_forest",
                "model_kwargs": {
                    "n_estimators": 150,
                    "max_depth": 9,
                    "max_features": 0.6,
                    "max_samples": 0.75,
                    "min_samples_leaf": 4,
                    "min_samples_split": 10,
                    "random_state": 7,
                    "oob_score": True,
                },
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = RandomForestRegressionOperatorModel(config, datasets)

    estimator = model._build_estimator()

    assert estimator.n_estimators == 150
    assert estimator.max_depth == 9
    assert estimator.max_features == 0.6
    assert estimator.max_samples == 0.75
    assert estimator.min_samples_leaf == 4
    assert estimator.min_samples_split == 10
    assert estimator.random_state == 7
    assert estimator.oob_score is True


def test_knn_uses_dataset_aware_defaults_and_direct_model_kwargs():
    rows = 600
    rng = np.random.default_rng(55)
    df = pd.DataFrame(
        {
            f"x{i}": rng.normal(size=rows)
            for i in range(6)
        }
    )
    df["target"] = df.sum(axis=1) + rng.normal(0, 0.2, rows)

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "knn",
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = KNNRegressionOperatorModel(config, datasets)

    estimator = model._build_estimator()

    assert estimator.n_neighbors == 9
    assert estimator.weights == "uniform"
    assert estimator.p == 2
    assert estimator.n_jobs == -1

    config.spec.model_kwargs = {
        "n_neighbors": 13,
        "weights": "uniform",
        "p": 1,
    }
    estimator = model._build_estimator()
    assert estimator.n_neighbors == 13
    assert estimator.weights == "uniform"
    assert estimator.p == 1


def test_xgboost_uses_metric_aware_defaults_and_direct_model_kwargs():
    rows = 1200
    rng = np.random.default_rng(66)
    df = pd.DataFrame(
        {
            f"x{i}": rng.normal(size=rows)
            for i in range(12)
        }
    )
    df["target"] = df.sum(axis=1) + rng.normal(0, 0.3, rows)

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "xgboost",
                "metric": "mae",
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = XGBoostRegressionOperatorModel(config, datasets)

    estimator = model._build_estimator()

    assert estimator.get_params()["n_estimators"] == 400
    assert estimator.get_params()["learning_rate"] == 0.05
    assert estimator.get_params()["max_depth"] == 5
    assert estimator.get_params()["min_child_weight"] == 3
    assert estimator.get_params()["subsample"] == 0.85
    assert estimator.get_params()["colsample_bytree"] == 0.9
    assert estimator.get_params()["eval_metric"] == "mae"
    assert estimator.get_params()["tree_method"] == "hist"

    config.spec.model_kwargs = {
        "n_estimators": 250,
        "learning_rate": 0.08,
        "max_depth": 7,
        "subsample": 0.75,
    }
    estimator = model._build_estimator()

    assert estimator.get_params()["n_estimators"] == 250
    assert estimator.get_params()["learning_rate"] == 0.08
    assert estimator.get_params()["max_depth"] == 7
    assert estimator.get_params()["subsample"] == 0.75


def test_auto_regression_selects_best_model_via_cv_and_populates_predictions():
    rng = np.random.default_rng(77)
    train_rows = 120
    test_rows = 40

    train_df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, train_rows),
            "x2": rng.normal(5, 2, train_rows),
            "x3": rng.normal(-1, 0.5, train_rows),
        }
    )
    train_df["target"] = (
        10
        + 4.0 * train_df["x1"]
        - 1.5 * train_df["x2"]
        + 2.2 * train_df["x3"]
    )

    test_df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, test_rows),
            "x2": rng.normal(5, 2, test_rows),
            "x3": rng.normal(-1, 0.5, test_rows),
        }
    )
    test_df["target"] = (
        10
        + 4.0 * test_df["x1"]
        - 1.5 * test_df["x2"]
        + 2.2 * test_df["x3"]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train.csv")
        test_path = os.path.join(tmp_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        config = RegressionOperatorConfig.from_dict(
            {
                "kind": "operator",
                "type": "regression",
                "version": "v1",
                "spec": {
                    "training_data": {"url": train_path},
                    "test_data": {"url": test_path},
                    "target_column": "target",
                    "model": "auto",
                    "metric": "rmse",
                    "generate_report": False,
                    "generate_explanations": False,
                },
            }
        )
        datasets = RegressionDatasets(config)
        model = RegressionOperatorModelFactory.get_model(config, datasets)

        assert isinstance(model, AutoRegressionOperatorModel)

        model._train_and_predict(
            datasets.training_data[datasets.feature_columns],
            datasets.training_data[config.spec.target_column],
        )

        assert model.selected_model_name == "linear_regression"
        assert config.spec.model == "linear_regression"
        assert model.train_predictions is not None
        assert model.test_predictions is not None
        assert model.test_metrics is not None
        assert not model.cv_results_df.empty


def test_linear_regression_tunes_hyperparameters_with_cv():
    rng = np.random.default_rng(88)
    rows = 100
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, rows),
            "x2": rng.normal(2, 1.5, rows),
            "x3": rng.normal(-1, 0.8, rows),
        }
    )
    df["target"] = 3.5 * df["x1"] - 1.2 * df["x2"] + 0.8 * df["x3"] + rng.normal(
        0, 0.1, rows
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "linear_regression",
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = LinearRegressionOperatorModel(config, datasets)

    model._train_and_predict(
        datasets.training_data[datasets.feature_columns],
        datasets.training_data[config.spec.target_column],
    )

    assert model.best_tuned_params
    assert not model.tuning_results_df.empty
    assert set(model.tuning_results_df["metric"]) == {config.spec.metric}
    assert len(model.tuning_results_df) == 20


def test_random_forest_tunes_hyperparameters_with_cv():
    rng = np.random.default_rng(89)
    rows = 120
    df = pd.DataFrame(
        {
            f"x{i}": rng.normal(size=rows)
            for i in range(6)
        }
    )
    df["target"] = (
        df["x0"] * df["x1"]
        + 0.5 * df["x2"]
        - 0.8 * df["x3"]
        + rng.normal(0, 0.2, rows)
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "random_forest",
                "model_kwargs": {"n_estimators": 30, "max_depth": 8},
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = RandomForestRegressionOperatorModel(config, datasets)

    model._train_and_predict(
        datasets.training_data[datasets.feature_columns],
        datasets.training_data[config.spec.target_column],
    )

    assert model.best_tuned_params
    assert not model.tuning_results_df.empty
    assert "params" in model.tuning_results_df.columns


def test_knn_tunes_hyperparameters_with_cv():
    rng = np.random.default_rng(90)
    rows = 100
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
        }
    )
    df["target"] = df["x1"] ** 2 + 0.5 * df["x2"] - df["x3"] + rng.normal(
        0, 0.05, rows
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "knn",
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = KNNRegressionOperatorModel(config, datasets)

    model._train_and_predict(
        datasets.training_data[datasets.feature_columns],
        datasets.training_data[config.spec.target_column],
    )

    assert model.best_tuned_params
    assert not model.tuning_results_df.empty
    assert model.regressor is not None


def test_xgboost_tunes_hyperparameters_with_cv():
    rng = np.random.default_rng(91)
    rows = 120
    df = pd.DataFrame(
        {
            f"x{i}": rng.normal(size=rows)
            for i in range(5)
        }
    )
    df["target"] = (
        1.5 * df["x0"]
        - 2.0 * df["x1"]
        + df["x2"] * df["x3"]
        + rng.normal(0, 0.1, rows)
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "xgboost",
                "model_kwargs": {"n_estimators": 30, "max_depth": 4},
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = XGBoostRegressionOperatorModel(config, datasets)

    model._train_and_predict(
        datasets.training_data[datasets.feature_columns],
        datasets.training_data[config.spec.target_column],
    )

    assert model.best_tuned_params
    assert not model.tuning_results_df.empty
    assert model.regressor is not None


def test_model_kwargs_can_override_tuning_trial_count():
    rng = np.random.default_rng(92)
    rows = 80
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
            "target": rng.normal(size=rows),
        }
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "knn",
                "model_kwargs": {
                    "tuning_n_trials": 3,
                },
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = KNNRegressionOperatorModel(config, datasets)

    model._train_and_predict(
        datasets.training_data[datasets.feature_columns],
        datasets.training_data[config.spec.target_column],
    )

    assert len(model.tuning_results_df) == 3
    assert "tuning_n_trials" not in model.best_tuned_params

def test_regression_operator_smoke_random_forest_with_missing_values():
    rng = np.random.default_rng(123)
    rows = 90
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, rows),
            "x2": rng.normal(5, 2, rows),
            "city": rng.choice(["A", "B", "C", None], size=rows),
            "event_date": pd.date_range("2025-01-01", periods=rows, freq="D").astype(str),
        }
    )
    df.loc[df.index[::7], "x1"] = np.nan
    df.loc[df.index[::11], "x2"] = np.nan
    df["target"] = (
        8
        + df["x1"].fillna(0) * 1.8
        + df["x2"].fillna(5) * 0.6
        + (df["city"] == "B").astype(int) * 1.2
        + rng.normal(0, 0.3, rows)
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train.csv")
        out_path = os.path.join(tmp_dir, "out")
        df.to_csv(train_path, index=False)

        cfg = {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"url": train_path},
                "target_column": "target",
                "model": "random_forest",
                "output_directory": {"url": out_path},
                "generate_report": False,
                "generate_explanations": False,
            },
        }

        operate(RegressionOperatorConfig.from_dict(cfg))

        assert os.path.exists(os.path.join(out_path, "training_predictions.csv"))
        assert os.path.exists(os.path.join(out_path, "training_metrics.csv"))
        assert os.path.exists(os.path.join(out_path, "global_explanations.csv"))
        preds = pd.read_csv(os.path.join(out_path, "training_predictions.csv"))
        assert preds["predicted_value"].notna().all()


def test_regression_operator_smoke_linear_regression():
    rng = np.random.default_rng(42)
    rows = 80
    x1 = rng.normal(0, 1, rows)
    x2 = rng.normal(5, 2, rows)
    city = rng.choice(["A", "B", "C"], size=rows)
    y = 5 + (2.5 * x1) + (0.7 * x2) + (city == "B").astype(int) * 1.1 + rng.normal(0, 0.2, rows)

    df = pd.DataFrame({"x1": x1, "x2": x2, "city": city, "target": y})

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train.csv")
        out_path = os.path.join(tmp_dir, "out")
        df.to_csv(train_path, index=False)

        cfg = {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"url": train_path},
                "target_column": "target",
                "model": "linear_regression",
                "output_directory": {"url": out_path},
                "generate_report": True,
                "generate_explanations": False,
            },
        }

        operate(RegressionOperatorConfig.from_dict(cfg))

        assert os.path.exists(os.path.join(out_path, "training_predictions.csv"))
        assert not os.path.exists(os.path.join(out_path, "test_predictions.csv"))
        assert os.path.exists(os.path.join(out_path, "training_metrics.csv"))
        assert os.path.exists(os.path.join(out_path, "global_explanations.csv"))
        assert os.path.exists(os.path.join(out_path, "model.pkl"))
        # assert os.path.exists(os.path.join(out_path, "models.pickle"))
        assert not os.path.exists(os.path.join(out_path, "model_registration_info.json"))
        assert not os.path.exists(os.path.join(out_path, "feature_importance.csv"))
        assert not os.path.exists(os.path.join(out_path, "local_explanations.csv"))
        report_path = os.path.join(out_path, "report.html")
        assert os.path.exists(report_path)

        training_predictions_df = pd.read_csv(
            os.path.join(out_path, "training_predictions.csv")
        )
        assert list(training_predictions_df.columns) == [
            "input_value",
            "predicted_value",
            "residual",
        ]

        with open(report_path) as f:
            report_html = f.read()
        assert "First 5 Rows of Data" in report_html
        assert "Data Summary Statistics" in report_html
        assert "Training Data Metrics" in report_html
        assert "Reference: YAML File" in report_html
        assert "The following tables summarize the training dataset used for this regression analysis" in report_html
        assert "Training Actual vs Predicted" in report_html
        assert "Training Actual vs Predicted with Ideal Fit Reference" in report_html
        assert "Training Actual and Predicted Values by Row" not in report_html
        assert "Explainability" in report_html
        assert "Estimator Hyperparameters" in report_html
        assert "Model Configuration and Hyperparameters" not in report_html
        assert "Effective Model Configuration" not in report_html
        assert "fit_intercept" in report_html
        assert "copy_X" in report_html
        assert "#636EFA" in report_html


def test_regression_predictions_csv_includes_test_rows():
    rng = np.random.default_rng(7)
    train_rows = 60
    test_rows = 20

    train_df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, train_rows),
            "x2": rng.normal(5, 2, train_rows),
            "city": rng.choice(["A", "B", "C"], size=train_rows),
        }
    )
    train_df["target"] = (
        3
        + (1.7 * train_df["x1"])
        + (0.5 * train_df["x2"])
        + (train_df["city"] == "B").astype(int) * 0.8
        + rng.normal(0, 0.1, train_rows)
    )

    test_df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, test_rows),
            "x2": rng.normal(5, 2, test_rows),
            "city": rng.choice(["A", "B", "C"], size=test_rows),
        }
    )
    test_df["target"] = (
        3
        + (1.7 * test_df["x1"])
        + (0.5 * test_df["x2"])
        + (test_df["city"] == "B").astype(int) * 0.8
        + rng.normal(0, 0.1, test_rows)
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train.csv")
        test_path = os.path.join(tmp_dir, "test.csv")
        out_path = os.path.join(tmp_dir, "out")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        cfg = {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"url": train_path},
                "test_data": {"url": test_path},
                "target_column": "target",
                "model": "linear_regression",
                "output_directory": {"url": out_path},
                "generate_report": False,
                "generate_explanations": False,
            },
        }

        operate(RegressionOperatorConfig.from_dict(cfg))

        training_predictions_df = pd.read_csv(
            os.path.join(out_path, "training_predictions.csv")
        )
        test_predictions_df = pd.read_csv(
            os.path.join(out_path, "test_predictions.csv")
        )

        assert list(training_predictions_df.columns) == [
            "input_value",
            "predicted_value",
            "residual",
        ]
        assert list(test_predictions_df.columns) == [
            "input_value",
            "predicted_value",
            "residual",
        ]
        assert len(training_predictions_df) == train_rows
        assert len(test_predictions_df) == test_rows
        assert training_predictions_df["predicted_value"].notna().sum() == train_rows
        assert test_predictions_df["predicted_value"].notna().sum() == test_rows
        assert training_predictions_df["residual"].notna().sum() == train_rows
        assert test_predictions_df["residual"].notna().sum() == test_rows
        assert os.path.exists(os.path.join(out_path, "training_metrics.csv"))
        assert os.path.exists(os.path.join(out_path, "test_metrics.csv"))
        assert os.path.exists(os.path.join(out_path, "global_explanations.csv"))


def test_regression_infers_numeric_categorical_and_date_columns():
    df = pd.DataFrame(
        {
            "numeric_text": ["1.0", "2.5", "3.2", "4.8", "5.1", "6.0"],
            "category_id": [1, 2, 1, 2, 3, 3],
            "customer_id": [101, 102, 103, 104, 105, 106],
            "event_date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
            ],
            "target": [10.0, 12.0, 11.5, 13.2, 14.1, 15.0],
        }
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": df},
                "target_column": "target",
                "model": "linear_regression",
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = LinearRegressionOperatorModel(config, datasets)

    numeric_cols, categorical_cols, date_cols = model._infer_column_types(
        df[datasets.feature_columns]
    )

    assert "numeric_text" in numeric_cols
    assert "event_date" in date_cols
    assert "category_id" in categorical_cols
    assert "customer_id" in categorical_cols


def test_regression_date_features_are_generated_from_date_columns():
    rows = 12
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2025-01-01", periods=rows, freq="D").astype(str),
            "numeric_text": [str(v) for v in np.linspace(10, 20, rows)],
            "city": ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
        }
    )
    df["target"] = np.linspace(100, 130, rows) + np.arange(rows) * 0.5

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train.csv")
        out_path = os.path.join(tmp_dir, "out")
        df.to_csv(train_path, index=False)

        cfg = {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"url": train_path},
                "target_column": "target",
                "model": "linear_regression",
                "output_directory": {"url": out_path},
                "generate_report": False,
                "generate_explanations": False,
            },
        }

        operate(RegressionOperatorConfig.from_dict(cfg))

        explanations_df = pd.read_csv(
            os.path.join(out_path, "global_explanations.csv")
        )

        assert "event_date_year" in explanations_df["feature"].values
        assert "event_date_month" in explanations_df["feature"].values
        assert "event_date_day" in explanations_df["feature"].values
        assert "event_date_dayofweek" in explanations_df["feature"].values
        assert "event_date_dayofyear" in explanations_df["feature"].values
        assert "numeric_text" in explanations_df["feature"].values


def test_regression_deployment_sanity_test_uses_training_data_subset():
    training_df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 20.0, 30.0, 40.0],
            "target": [5.0, 6.0, 7.0, 8.0],
        }
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": training_df},
                "target_column": "target",
                "model": "linear_regression",
                "generate_report": False,
                "generate_explanations": False,
                "save_and_deploy_to_md": {
                    "project_id": "ocid1.project.oc1..exampleuniqueID",
                    "compartment_id": "ocid1.compartment.oc1..exampleuniqueID",
                    "model_deployment": {
                        "display_name": "regression-md",
                        "initial_shape": "VM.Standard.E3.Flex",
                    },
                },
            },
        }
    )

    manager = ModelDeploymentManager(
        spec=config.spec,
        model_name="linear_regression",
    )

    manager._copy_score_file()
    model_bundle = {
        "spec": config.spec.to_dict(),
        "models": _DummyRegressionModel(["x1", "x2"]),
    }
    with open(os.path.join(manager.path_to_artifact, "models.pickle"), "wb") as f:
        import cloudpickle

        cloudpickle.dump(model_bundle, f)

    with patch(
        "ads.opctl.operator.lowcode.regression.deployment.deployment_manager.logger.info"
    ) as mock_info:
        manager._sanity_test()

    assert any(
        "Regression deployment sanity test completed" in str(call.args[0])
        for call in mock_info.call_args_list
    )


def test_regression_invalid_dates_do_not_fail_operator():
    rows = 12
    df = pd.DataFrame(
        {
            "event_date": [
                "2025-01-01",
                "bad-date",
                "",
                None,
                "01/05/2025",
                "2025-01-06T10:15:00Z",
                "not-a-date",
                "2025/01/08",
                "20250109",
                "  ",
                "Jan 11 2025",
                "2025-13-99",
            ],
            "numeric_text": [str(v) for v in np.linspace(10, 20, rows)],
            "city": ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
        }
    )
    df["target"] = np.linspace(100, 130, rows) + np.arange(rows) * 0.5

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train.csv")
        out_path = os.path.join(tmp_dir, "out")
        df.to_csv(train_path, index=False)

        cfg = {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"url": train_path},
                "target_column": "target",
                "model": "linear_regression",
                "output_directory": {"url": out_path},
                "generate_report": False,
                "generate_explanations": False,
            },
        }

        operate(RegressionOperatorConfig.from_dict(cfg))

        explanations_df = pd.read_csv(
            os.path.join(out_path, "global_explanations.csv")
        )
        training_predictions_df = pd.read_csv(
            os.path.join(out_path, "training_predictions.csv")
        )

        assert os.path.exists(os.path.join(out_path, "training_metrics.csv"))
        assert not explanations_df.empty
        assert len(training_predictions_df) == rows
        assert training_predictions_df["predicted_value"].notna().all()


def test_regression_deployment_score_predict_uses_artifact_bundle():
    class MockModel:
        def __init__(self):
            self.preprocessor = SimpleNamespace(feature_columns_=["x1", "x2"])

        def predict(self, x_df):
            return x_df["x1"].fillna(0).astype(float) + x_df["x2"].fillna(0).astype(float)

    bundle = {
        "spec": {"target_column": "target"},
        "models": MockModel(),
    }

    result = deployment_predict(
        {
            "data": {
                "x1": [1.5, None],
                "x2": [None, 2.5],
                "target": [10, None],
            }
        },
        model=bundle,
    )

    assert json.loads(result["prediction"]) == [1.5, 2.5]


def test_regression_deployment_score_predict_preprocesses_training_style_payload():
    training_df = pd.DataFrame(
        {
            "numeric_text": ["1.0", "2.0", "3.0", "4.0"],
            "city": ["A", "B", "A", "B"],
            "event_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "target": [10.0, 12.0, 14.0, 16.0],
        }
    )

    config = RegressionOperatorConfig.from_dict(
        {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"data": training_df},
                "target_column": "target",
                "model": "linear_regression",
                "generate_report": False,
                "generate_explanations": False,
            },
        }
    )
    datasets = RegressionDatasets(config)
    model = LinearRegressionOperatorModel(config, datasets)
    x_train = datasets.training_data[datasets.feature_columns]
    y_train = datasets.training_data[config.spec.target_column]
    model._train_and_predict(x_train, y_train)

    bundle = {
        "spec": config.spec.to_dict(),
        "models": model.model_obj,
    }

    result = deployment_predict(
        {
            "data": {
                "numeric_text": ["1.5", "2.5"],
                "city": ["A", "B"],
                "event_date": ["2025-01-05", "2025-01-06"],
                "target": [999.0, 888.0],
            }
        },
        model=bundle,
    )

    predictions = json.loads(result["prediction"])
    assert len(predictions) == 2
    assert all(isinstance(prediction, float) for prediction in predictions)


def test_regression_save_and_deploy_to_md_writes_registration_info():
    rng = np.random.default_rng(12)
    rows = 20
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, rows),
            "x2": rng.normal(2, 1, rows),
        }
    )
    df["target"] = 2 + df["x1"] - 0.4 * df["x2"] + rng.normal(0, 0.1, rows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train.csv")
        out_path = os.path.join(tmp_dir, "out")
        df.to_csv(train_path, index=False)

        cfg = {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"url": train_path},
                "target_column": "target",
                "model": "linear_regression",
                "output_directory": {"url": out_path},
                "generate_report": False,
                "generate_explanations": False,
                "save_and_deploy_to_md": {
                    "model_catalog_display_name": "regression-linear-model",
                    "model_deployment": {
                        "display_name": "regression-linear-md",
                        "initial_shape": "VM.Standard.E4.Flex",
                        "description": "deployment description",
                    },
                },
            },
        }

        registration_info = {
            "model_ocid": "ocid1.datasciencemodel.oc1..example",
            "saved_to_model_catalog": True,
            "deployed_to_model_deployment": False,
            "model_name": "linear_regression",
        }
        with patch(
            "ads.opctl.operator.lowcode.regression.model.base_model.RegressionOperatorBaseModel._publish_to_oci",
            return_value=registration_info,
        ) as mock_publish:
            operate(RegressionOperatorConfig.from_dict(cfg))

        mock_publish.assert_called_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["deploy_config"].model_catalog_display_name == "regression-linear-model"

        with open(os.path.join(out_path, "model_registration_info.json")) as f:
            saved_info = json.load(f)
        assert saved_info["model_ocid"] == registration_info["model_ocid"]
        assert saved_info["saved_to_model_catalog"] is True
        assert saved_info["deployed_to_model_deployment"] is False


def test_regression_save_and_deploy_to_md_passes_deployment_config():
    rng = np.random.default_rng(17)
    rows = 20
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, rows),
            "x2": rng.normal(2, 1, rows),
        }
    )
    df["target"] = 2 + df["x1"] - 0.4 * df["x2"] + rng.normal(0, 0.1, rows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train.csv")
        out_path = os.path.join(tmp_dir, "out")
        df.to_csv(train_path, index=False)

        cfg = {
            "kind": "operator",
            "type": "regression",
            "version": "v1",
            "spec": {
                "training_data": {"url": train_path},
                "target_column": "target",
                "model": "linear_regression",
                "output_directory": {"url": out_path},
                "generate_report": False,
                "generate_explanations": False,
                "save_and_deploy_to_md": {
                    "model_catalog_display_name": "regression-linear-model",
                    "model_deployment": {
                        "display_name": "regression-linear-md",
                        "initial_shape": "VM.Standard.E4.Flex",
                        "description": "deployment description",
                    },
                },
            },
        }

        registration_info = {
            "model_ocid": "ocid1.datasciencemodel.oc1..example",
            "model_deployment_ocid": "ocid1.datasciencemodeldeployment.oc1..example",
            "saved_to_model_catalog": True,
            "deployed_to_model_deployment": True,
            "model_name": "linear_regression",
        }
        with patch(
            "ads.opctl.operator.lowcode.regression.model.base_model.RegressionOperatorBaseModel._publish_to_oci",
            return_value=registration_info,
        ) as mock_publish:
            operate(RegressionOperatorConfig.from_dict(cfg))

        mock_publish.assert_called_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["deploy_config"].model_catalog_display_name == "regression-linear-model"
        assert (
            kwargs["deploy_config"].model_deployment.display_name
            == "regression-linear-md"
        )

        with open(os.path.join(out_path, "model_registration_info.json")) as f:
            saved_info = json.load(f)
        assert saved_info["model_ocid"] == registration_info["model_ocid"]
        assert (
            saved_info["model_deployment_ocid"]
            == registration_info["model_deployment_ocid"]
        )
        assert saved_info["deployed_to_model_deployment"] is True
