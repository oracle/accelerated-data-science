#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import report_creator as rc
from plotly import graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ads.common.decorator.runtime_dependency import OptionalDependency, runtime_dependency
from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import (
    default_signer,
    human_time_friendly,
    write_data,
    write_file,
    write_pkl,
    write_simple_json,
)
from ads.opctl.operator.lowcode.regression.const import SupportedMetrics
from ads.opctl.operator.lowcode.regression.model.regression_dataset import RegressionDatasets
from ads.opctl.operator.lowcode.regression.operator_config import (
    RegressionOperatorConfig,
    RegressionOperatorSpec,
    RegressionDeploymentConfig,
)
from ads.opctl.operator.lowcode.regression.deployment import ModelDeploymentManager
from ads.opctl.operator.lowcode.regression.model.inference_model import (
    RegressionInferenceModel,
)
from ads.opctl.operator.lowcode.regression.model.transformers import (
    ColumnTypeResolver,
    RegressionFeaturePreprocessor,
)

logging.getLogger("report_creator").setLevel(logging.WARNING)

PLOTLY_COLORWAY = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

PREDICTION_SERIES_COLOR = "#2563EB"
GLOBAL_EXPLANATIONS_COLOR = "#7C3AED"
REFERENCE_LINE_COLOR = "#6B7280"


class RegressionOperatorBaseModel(ABC):
    """Base class for all regression operator models."""

    def __init__(self, config: RegressionOperatorConfig, datasets: RegressionDatasets):
        self.config = config
        self.spec: RegressionOperatorSpec = config.spec
        self.datasets = datasets

        self.feature_columns = datasets.feature_columns
        self.target_column = self.spec.target_column

        self.preprocessor = None
        self.regressor = None
        self.model_obj = None
        self.feature_names_out = []
        self.train_predictions = None
        self.test_predictions = None
        self.train_metrics = None
        self.test_metrics = None
        self.global_explanations_df = None

    @abstractmethod
    def _build_estimator(self):
        """Returns model estimator instance."""

    @abstractmethod
    def _train_and_predict(self, x_train, y_train):
        """Fits the model pipeline and populates train/test predictions and metrics."""

    @classmethod
    @abstractmethod
    def get_model_display_name(cls):
        """Returns the human-readable display name for the concrete model."""

    @classmethod
    @abstractmethod
    def get_model_description(cls):
        """Returns the report description for the concrete model."""

    def _report_model_display_name(self):
        return self.get_model_display_name()

    def _report_model_description(self):
        return self.get_model_description()

    @property
    def model_name(self):
        return self.spec.model

    def _create_inference_model(self):
        self.model_obj = RegressionInferenceModel(
            preprocessor=self.preprocessor,
            regressor=self.regressor,
        )
        return self.model_obj

    def _metric_columns(self):
        return [
            SupportedMetrics.RMSE,
            SupportedMetrics.MAE,
            SupportedMetrics.MSE,
            SupportedMetrics.R2,
            SupportedMetrics.MAPE,
        ]

    def _compute_metrics(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
        return {
            SupportedMetrics.RMSE: float(np.sqrt(mean_squared_error(y_true, y_pred))),
            SupportedMetrics.MAE: float(mean_absolute_error(y_true, y_pred)),
            SupportedMetrics.MSE: float(mean_squared_error(y_true, y_pred)),
            SupportedMetrics.R2: float(r2_score(y_true, y_pred)),
            SupportedMetrics.MAPE: float(mape),
        }

    def _infer_column_types(self, x_df: pd.DataFrame):
        return ColumnTypeResolver.infer_column_types(
            x_df=x_df,
            feature_columns=self.feature_columns,
            configured_column_types=self.spec.column_types or {},
        )

    def _is_numeric_like(self, series: pd.Series) -> bool:
        return ColumnTypeResolver.is_numeric_like(series)

    def _is_categorical_like(self, series: pd.Series, column_name: str) -> bool:
        return ColumnTypeResolver.is_categorical_like(series, column_name)

    def _is_datetime_like(self, series: pd.Series) -> bool:
        return ColumnTypeResolver.is_datetime_like(series)

    @staticmethod
    def _normalize_string_series(series: pd.Series) -> pd.Series:
        return ColumnTypeResolver.normalize_string_series(series)

    @staticmethod
    def _has_low_numeric_cardinality(series: pd.Series) -> bool:
        return ColumnTypeResolver.has_low_numeric_cardinality(series)

    @staticmethod
    def _looks_like_identifier(column_name: str) -> bool:
        return ColumnTypeResolver.looks_like_identifier(column_name)

    def _build_preprocessor(self, x_df: pd.DataFrame):
        preprocessing_enabled = (
            self.spec.preprocessing.enabled
            if self.spec.preprocessing is not None
            else True
        )
        impute_missing = (
            self.spec.preprocessing.steps.missing_value_imputation
            if self.spec.preprocessing is not None
            else True
        )
        encode_cat = (
            self.spec.preprocessing.steps.categorical_encoding
            if self.spec.preprocessing is not None
            else True
        )
        return RegressionFeaturePreprocessor(
            feature_columns=self.feature_columns,
            column_types=self.spec.column_types or {},
            preprocessing_enabled=preprocessing_enabled,
            missing_value_imputation=impute_missing,
            categorical_encoding=encode_cat,
        )

    def _derive_feature_names(self):
        preprocessor = self.preprocessor
        try:
            names = preprocessor.get_feature_names_out()
            self.feature_names_out = [n.split("__", 1)[-1] for n in names]
        except Exception:
            self.feature_names_out = self.feature_columns

    def _compute_global_explanations(self, x_train: pd.DataFrame, y_train: pd.Series):
        model = self.regressor
        x_proc = self.preprocessor.preprocess_for_prediction(x_train)
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            importances = np.abs(coef.reshape(-1))

        if importances is not None:
            feature_names = self.feature_names_out or self.feature_columns
            if len(feature_names) != len(importances):
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            self.global_explanations_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)

        if self.global_explanations_df is not None:
            self.global_explanations_df = self.global_explanations_df.reset_index(drop=True)
        return x_proc

    @runtime_dependency(
        module="shap",
        install_from=OptionalDependency.OPCTL,
        err_msg=(
            "Please run `python3 -m pip install shap` to install SHAP dependencies for model explanation."
        ),
    )
    def _generate_shap_explanations(self, x_train: pd.DataFrame, x_proc):
        if not self.spec.generate_explanations:
            return

        import shap

        model = self.regressor
        sample_size = min(200, len(x_train))
        if sample_size <= 0:
            return

        x_sample_raw = x_train.sample(n=sample_size, random_state=42)
        x_sample = self.preprocessor.preprocess_for_prediction(x_sample_raw)

        feature_names = self.feature_names_out or self.feature_columns

        try:
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_sample)
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
            else:
                explainer = shap.Explainer(model.predict, x_sample)
                shap_obj = explainer(x_sample)
                values = shap_obj.values

            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, 0]

            global_vals = np.mean(np.abs(values), axis=0)
            self.global_explanations_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": global_vals,
                }
            ).sort_values("importance", ascending=False)

        except Exception as e:
            logger.warning(f"Unable to generate SHAP explanations. Error: {e}")

    def _apply_default_plot_layout(self, fig: go.Figure, title: str, **layout_kwargs):
        fig.update_layout(
            template="plotly_white",
            colorway=PLOTLY_COLORWAY,
            title=title,
            **layout_kwargs,
        )
        return fig

    def _build_bar_plot(
        self,
        plot_df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        color: str,
    ):
        fig = go.Figure(
            data=[
                go.Bar(
                    x=plot_df[x],
                    y=plot_df[y],
                    marker=dict(color=color),
                )
            ]
        )
        self._apply_default_plot_layout(fig, title=title, xaxis_title=x, yaxis_title=y)
        return rc.Widget(fig, label=title)

    def _build_actual_vs_predicted_scatter_plot(
        self, predictions_df: pd.DataFrame, title: str
    ):
        actual = predictions_df["actual"]
        prediction = predictions_df["prediction"]
        lower_bound = float(min(actual.min(), prediction.min()))
        upper_bound = float(max(actual.max(), prediction.max()))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=actual,
                y=prediction,
                mode="markers",
                name="Predictions",
                marker=dict(
                    color=PREDICTION_SERIES_COLOR,
                    size=8,
                    opacity=0.8,
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[lower_bound, upper_bound],
                y=[lower_bound, upper_bound],
                mode="lines",
                name="Ideal Fit (y = x)",
                line=dict(color=REFERENCE_LINE_COLOR, width=2, dash="dash"),
            )
        )
        self._apply_default_plot_layout(
            fig,
            title=title,
            xaxis_title="Actual",
            yaxis_title="Predicted",
        )
        return rc.Widget(fig, label=title)

    def _candidate_models_text(self):
        from .factory import RegressionOperatorModelFactory

        model_names = [
            model_cls.get_model_display_name()
            for model_cls in RegressionOperatorModelFactory._MAP.values()
        ]
        model_names.append("Auto")
        return ", ".join(model_names)

    def _data_summary_section(self):
        training_data = self.datasets.training_data[
            self.feature_columns + [self.target_column]
        ]
        try:
            summary_df = training_data.describe(include="all", datetime_is_numeric=True)
        except TypeError:
            summary_df = training_data.describe(include="all")

        return rc.Block(
            rc.Text(
                "The following tables summarize the training dataset used for this "
                "regression analysis, including a preview of the records and "
                "descriptive statistics across the selected columns."
            ),
            rc.Block(
                rc.Heading("First 5 Rows of Data", level=3),
                rc.DataTable(training_data.head(5), index=False),
            ),
            rc.Block(
                rc.Heading("Last 5 Rows of Data", level=3),
                rc.DataTable(training_data.tail(5), index=False),
            ),
            rc.Block(
                rc.Heading("Data Summary Statistics", level=3),
                rc.DataTable(summary_df, index=True),
            ),
        )

    def _report_config_dict(self):
        config_dict = self.config.to_dict()
        for dataset_key in ("training_data", "test_data"):
            dataset_config = config_dict.get("spec", {}).get(dataset_key)
            if isinstance(dataset_config, dict):
                dataset_config.pop("data", None)
        return config_dict

    @staticmethod
    def _sanitize_report_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [
                RegressionOperatorBaseModel._sanitize_report_value(item)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                str(key): RegressionOperatorBaseModel._sanitize_report_value(item)
                for key, item in value.items()
            }
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _build_hyperparameters_report_df(self):
        if self.regressor is None or not hasattr(self.regressor, "get_params"):
            return pd.DataFrame(columns=["parameter", "value"])

        params = self.regressor.get_params(deep=False)
        rows = []
        for key in sorted(params):
            value = self._sanitize_report_value(params[key])
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)
            rows.append({"parameter": key, "value": value})
        return pd.DataFrame(rows)

    def _build_predictions_output_df(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        if predictions_df is None or predictions_df.empty:
            return pd.DataFrame(columns=["input_value", "predicted_value", "residual"])

        return pd.DataFrame(
            {
                "input_value": predictions_df["actual"],
                "predicted_value": predictions_df["prediction"],
                "residual": predictions_df["residual"],
            }
        )

    def _write_outputs(self, output_dir: str, storage_options):
        if not ObjectStorageDetails.is_oci_path(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        train_predictions_path = os.path.join(
            output_dir, self.spec.training_predictions_filename
        )
        test_predictions_path = os.path.join(
            output_dir, self.spec.test_predictions_filename
        )
        metrics_path = os.path.join(output_dir, self.spec.training_metrics_filename)
        test_metrics_path = os.path.join(output_dir, self.spec.test_metrics_filename)
        global_expl_path = os.path.join(output_dir, self.spec.global_explanation_filename)

        write_data(
            data=self._build_predictions_output_df(self.train_predictions),
            filename=train_predictions_path,
            format="csv",
            storage_options=storage_options,
            index=False,
        )

        if self.test_predictions is not None and not self.test_predictions.empty:
            write_data(
                data=self._build_predictions_output_df(self.test_predictions),
                filename=test_predictions_path,
                format="csv",
                storage_options=storage_options,
                index=False,
            )

        write_data(
            data=self.train_metrics,
            filename=metrics_path,
            format="csv",
            storage_options=storage_options,
            index=False,
        )

        if self.test_metrics is not None and not self.test_metrics.empty:
            write_data(
                data=self.test_metrics,
                filename=test_metrics_path,
                format="csv",
                storage_options=storage_options,
                index=False,
            )

        if self.global_explanations_df is not None and not self.global_explanations_df.empty:
            write_data(
                data=self.global_explanations_df,
                filename=global_expl_path,
                format="csv",
                storage_options=storage_options,
                index=False,
            )

        write_pkl(
            obj=self.model_obj,
            filename="model.pkl",
            output_dir=output_dir,
            storage_options=storage_options,
        )
        # write_pkl(
        #     obj={"spec": self.spec.to_dict(), "models": self.model_obj},
        #     filename="models.pickle",
        #     output_dir=output_dir,
        #     storage_options=storage_options,
        # )

    def _publish_to_oci(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        deploy_config: RegressionDeploymentConfig = None,
    ):
        manager = ModelDeploymentManager(
            spec=self.spec,
            model_name=self.model_name,
        )
        manager.save_to_catalog()
        manager.create_deployment()
        manager.save_deployment_info()
        return manager.deployment_info

    def _generate_report(self, output_dir: str, elapsed_time: float, storage_options):
        if not self.spec.generate_report:
            return

        training_rows = len(self.datasets.training_data)
        test_rows = len(self.datasets.test_data) if self.datasets.test_data is not None else 0

        sections = [
            rc.Block(
                rc.Heading(self.spec.report_title, level=1),
                rc.Text(
                    f"You selected the {self._report_model_display_name()} model. "
                    f"Based on your dataset, you could have also selected any of the models: "
                    f"{self._candidate_models_text()}."
                ),
                rc.Text(self._report_model_description()),
                rc.Group(
                    rc.Metric(
                        heading="Analysis was completed in",
                        value=human_time_friendly(elapsed_time),
                    ),
                    rc.Metric(heading="Training rows", value=training_rows),
                    rc.Metric(heading="Test rows", value=test_rows),
                    rc.Metric(heading="Features", value=len(self.feature_columns)),
                ),
            ),
            self._data_summary_section(),
            rc.Heading("Estimator Hyperparameters", level=3),
            rc.DataTable(self._build_hyperparameters_report_df(), index=False),
            rc.Heading("Training Data Metrics", level=2),
            rc.Text(
                "These metrics summarize how closely the model fits the training "
                "data across the configured regression objectives."
            ),
            rc.DataTable(self.train_metrics, index=False),
            rc.Heading("Training Actual vs Predicted", level=2),
            rc.Text(
                "The following chart compares actual and predicted target values on "
                "the training dataset and shows overall agreement between observed "
                "and predicted outcomes."
            ),
            self._build_actual_vs_predicted_scatter_plot(
                self.train_predictions,
                "Training Actual vs Predicted with Ideal Fit Reference",
            ),
            rc.Heading("Training Predictions (Top Rows)", level=3),
            rc.DataTable(self._build_predictions_output_df(self.train_predictions).head(25), index=False),
            rc.Heading("Global Explainability", level=2),
            rc.Text(
                "The following table and chart summarize which features had the "
                "largest influence on the fitted model."
            ),
        ]

        if self.global_explanations_df is not None and not self.global_explanations_df.empty:
            sections.extend(
                [
                    self._build_bar_plot(
                        self.global_explanations_df.head(20),
                        x="feature",
                        y="importance",
                        title="Global Feature Importance",
                        color=GLOBAL_EXPLANATIONS_COLOR,
                    ),
                    rc.DataTable(self.global_explanations_df.head(25), index=False),
                ]
            )
        else:
            sections.append(
                rc.Text("Global explainability is unavailable for this run.")
            )

        if self.test_metrics is not None and not self.test_metrics.empty:
            sections.extend(
                [
                    rc.Heading("Test Data Evaluation Metrics", level=2),
                    rc.Text(
                        "These metrics evaluate the model on held-out test data to "
                        "show how well the learned relationships generalize beyond the "
                        "training sample."
                    ),
                    rc.DataTable(self.test_metrics, index=False),
                ]
            )
            if self.test_predictions is not None and not self.test_predictions.empty:
                sections.extend(
                    [
                        rc.Heading("Test Actual vs Predicted", level=2),
                        self._build_actual_vs_predicted_scatter_plot(
                            self.test_predictions,
                            "Test Actual vs Predicted with Ideal Fit Reference",
                        ),
                        rc.Heading("Test Predictions (Top Rows)", level=3),
                        rc.DataTable(
                            self._build_predictions_output_df(self.test_predictions).head(25),
                            index=False,
                        ),
                    ]
                )

        if self.global_explanations_df is None or self.global_explanations_df.empty:
            if self.spec.generate_explanations:
                sections.extend(
                    [
                        rc.Text(
                            "Explainability was requested but outputs are unavailable. "
                            "Check logs for SHAP dependency or runtime errors."
                        ),
                    ]
                )
        elif self.spec.generate_explanations:
            sections.extend(
                [
                    rc.Text(
                        "Global explainability uses model-derived feature importance when "
                        "available and falls back to SHAP-based importance otherwise."
                    ),
                ]
            )
        else:
            sections.extend(
                [
                    rc.Heading("Explainability", level=2),
                    rc.Text(
                        "SHAP-based explainability is disabled for this run. When the "
                        "model exposes built-in feature importance, the global explanation "
                        "shown above still uses that model-derived importance."
                    ),
                ]
            )

        sections.extend(
            [
                rc.Heading("Reference: YAML File", level=2),
                rc.Yaml(self._report_config_dict()),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            local_report = os.path.join(temp_dir, "report.html")
            with rc.ReportCreator(self.spec.report_title) as report:
                report.save(rc.Block(*sections), local_report)
            write_file(
                local_filename=local_report,
                remote_filename=os.path.join(output_dir, self.spec.report_filename),
                storage_options=storage_options,
            )

    def generate_report(self):
        """Trains model and generates all requested artifacts."""
        start_time = time.time()

        x_train = self.datasets.training_data[self.feature_columns]
        y_train = self.datasets.training_data[self.target_column]
        x_proc = self._train_and_predict(x_train, y_train)

        if self.spec.generate_explanations and self.global_explanations_df is None:
            try:
                self._generate_shap_explanations(x_train, x_proc)
            except Exception as e:
                logger.warning(f"Skipping explainability generation. Error: {e}")

        output_dir = self.spec.output_directory.url
        storage_options = default_signer() if ObjectStorageDetails.is_oci_path(output_dir) else {}

        self._write_outputs(output_dir, storage_options)
        elapsed_time = time.time() - start_time
        self._generate_report(output_dir, elapsed_time, storage_options)

        model_registration_info = None
        if self.spec.save_and_deploy_to_md is not None:
            model_registration_info = self._publish_to_oci(
                x_train=x_train,
                y_train=y_train,
                deploy_config=self.spec.save_and_deploy_to_md,
            )
            write_simple_json(
                model_registration_info,
                os.path.join(output_dir, "model_registration_info.json"),
            )

        logger.info(f"Regression artifacts generated at: {output_dir}")
        print(f"Regression artifacts generated at: {output_dir}")

        return {
            "output_directory": output_dir,
            "model": self.model_name,
            "metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "model_registration": model_registration_info,
        }
