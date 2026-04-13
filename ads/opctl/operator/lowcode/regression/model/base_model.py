#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import report_creator as rc
from plotly import graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ads.common.decorator.runtime_dependency import OptionalDependency, runtime_dependency
from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import (
    default_signer,
    human_time_friendly,
    write_data,
    write_file,
    write_pkl,
)
from ads.opctl.operator.lowcode.regression.const import (
    ColumnType,
    SupportedMetrics,
)
from ads.opctl.operator.lowcode.regression.model.regression_dataset import RegressionDatasets
from ads.opctl.operator.lowcode.regression.operator_config import (
    RegressionOperatorConfig,
    RegressionOperatorSpec,
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

ACTUAL_SERIES_COLOR = "#1F2937"
PREDICTION_SERIES_COLOR = "#2563EB"
FEATURE_IMPORTANCE_COLOR = "#0F766E"
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

        self.pipeline = None
        self.feature_names_out = []
        self.train_predictions = None
        self.test_predictions = None
        self.train_metrics = None
        self.test_metrics = None
        self.feature_importance_df = None
        self.global_explanations_df = None
        self.local_explanations_df = None

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

    @property
    def model_name(self):
        return self.spec.model.name

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
        numeric_cols = []
        categorical_cols = []
        configured = self.spec.column_types or {}

        for col in self.feature_columns:
            if col in configured:
                if str(configured[col]).lower() == ColumnType.CATEGORICAL:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
                continue

            if pd.api.types.is_numeric_dtype(x_df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        return numeric_cols, categorical_cols

    def _build_preprocessor(self, x_df: pd.DataFrame):
        numeric_cols, categorical_cols = self._infer_column_types(x_df)

        if not self.spec.preprocessing or not self.spec.preprocessing.enabled:
            return ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_cols),
                    ("cat", "passthrough", categorical_cols),
                ],
                remainder="drop",
            )

        impute_missing = self.spec.preprocessing.steps.missing_value_imputation
        encode_cat = self.spec.preprocessing.steps.categorical_encoding

        num_steps = []
        if impute_missing:
            num_steps.append(("imputer", SimpleImputer(strategy="median")))

        cat_steps = []
        if impute_missing:
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        if encode_cat:
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            cat_steps.append(
                ("encoder", encoder)
            )

        num_pipeline = Pipeline(steps=num_steps) if num_steps else "passthrough"
        cat_pipeline = Pipeline(steps=cat_steps) if cat_steps else "passthrough"

        return ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numeric_cols),
                ("cat", cat_pipeline, categorical_cols),
            ],
            remainder="drop",
        )

    def _derive_feature_names(self):
        preprocessor = self.pipeline.named_steps["preprocessor"]
        try:
            names = preprocessor.get_feature_names_out()
            self.feature_names_out = [n.split("__", 1)[-1] for n in names]
        except Exception:
            self.feature_names_out = self.feature_columns

    def _compute_feature_importance(self, x_train: pd.DataFrame, y_train: pd.Series):
        model = self.pipeline.named_steps["regressor"]
        x_proc = self.pipeline.named_steps["preprocessor"].transform(x_train)

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            importances = np.abs(coef.reshape(-1))
        else:
            perm = permutation_importance(
                self.pipeline,
                x_train,
                y_train,
                n_repeats=5,
                random_state=42,
                scoring="neg_root_mean_squared_error",
            )
            importances = perm.importances_mean

        feature_names = self.feature_names_out or self.feature_columns
        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        self.feature_importance_df = df.reset_index(drop=True)
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

        model = self.pipeline.named_steps["regressor"]
        sample_size = min(200, len(x_train))
        if sample_size <= 0:
            return

        x_sample_raw = x_train.sample(n=sample_size, random_state=42)
        x_sample = self.pipeline.named_steps["preprocessor"].transform(x_sample_raw)

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

            local_df = pd.DataFrame(values, columns=feature_names)
            local_df.insert(0, "row_id", x_sample_raw.index.to_numpy())
            self.local_explanations_df = local_df
        except Exception as e:
            logger.warning(f"Unable to generate SHAP explanations. Error: {e}")

    def _actual_vs_prediction_plot_data(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        plot_df = predictions_df[["actual", "prediction"]].reset_index(drop=True).copy()
        plot_df.insert(0, "row_id", np.arange(1, len(plot_df) + 1))
        return plot_df

    def _apply_default_plot_layout(self, fig: go.Figure, title: str, **layout_kwargs):
        fig.update_layout(
            template="plotly_white",
            colorway=PLOTLY_COLORWAY,
            title=title,
            **layout_kwargs,
        )
        return fig

    def _build_actual_vs_predicted_line_plot(
        self, predictions_df: pd.DataFrame, title: str
    ):
        plot_df = self._actual_vs_prediction_plot_data(predictions_df)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df["row_id"],
                y=plot_df["actual"],
                mode="lines",
                name="Actual",
                line=dict(color=ACTUAL_SERIES_COLOR, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["row_id"],
                y=plot_df["prediction"],
                mode="lines",
                name="Prediction",
                line=dict(color=PREDICTION_SERIES_COLOR, width=2),
            )
        )
        self._apply_default_plot_layout(
            fig,
            title=title,
            xaxis_title="Row",
            yaxis_title=self.target_column,
        )
        return rc.Widget(fig, label=title)

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
        for dataset_key in ("training_data", "validation_data", "test_data"):
            dataset_config = config_dict.get("spec", {}).get(dataset_key)
            if isinstance(dataset_config, dict):
                dataset_config.pop("data", None)
        return config_dict

    def _build_predictions_output_df(self) -> pd.DataFrame:
        prediction_frames = []

        if self.train_predictions is not None and not self.train_predictions.empty:
            train_df = self.train_predictions.copy()
            train_row_id = (
                train_df["row_id"]
                if "row_id" in train_df.columns
                else pd.Series(train_df.index.to_numpy(), index=train_df.index)
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_id": train_row_id,
                        "input_value": train_df["actual"],
                        "fitted_value": train_df["prediction"],
                        "predicted_value": np.nan,
                        "residual": train_df["residual"],
                    }
                )
            )

        if self.test_predictions is not None and not self.test_predictions.empty:
            test_df = self.test_predictions.copy()
            test_row_id = (
                test_df["row_id"]
                if "row_id" in test_df.columns
                else pd.Series(test_df.index.to_numpy(), index=test_df.index)
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_id": test_row_id,
                        "input_value": test_df["actual"],
                        "fitted_value": np.nan,
                        "predicted_value": test_df["prediction"],
                        "residual": test_df["residual"],
                    }
                )
            )

        if not prediction_frames:
            return pd.DataFrame(
                columns=[
                    "row_id",
                    "input_value",
                    "fitted_value",
                    "predicted_value",
                    "residual",
                ]
            )

        return pd.concat(prediction_frames, ignore_index=True)

    def _write_outputs(self, output_dir: str, storage_options):
        if not ObjectStorageDetails.is_oci_path(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        predictions_path = os.path.join(output_dir, self.spec.predictions_filename)
        metrics_path = os.path.join(output_dir, self.spec.metrics_filename)
        test_metrics_path = os.path.join(output_dir, self.spec.test_metrics_filename)
        fi_path = os.path.join(output_dir, self.spec.feature_importance_filename)
        global_expl_path = os.path.join(output_dir, self.spec.global_explanation_filename)
        local_expl_path = os.path.join(output_dir, self.spec.local_explanation_filename)

        write_data(
            data=self._build_predictions_output_df(),
            filename=predictions_path,
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

        if self.feature_importance_df is not None and not self.feature_importance_df.empty:
            write_data(
                data=self.feature_importance_df,
                filename=fi_path,
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

        if self.local_explanations_df is not None and not self.local_explanations_df.empty:
            write_data(
                data=self.local_explanations_df,
                filename=local_expl_path,
                format="csv",
                storage_options=storage_options,
                index=False,
            )

        if self.spec.generate_model_pickle:
            write_pkl(
                obj=self.pipeline,
                filename="model.pkl",
                output_dir=output_dir,
                storage_options=storage_options,
            )

    def _generate_report(self, output_dir: str, elapsed_time: float, storage_options):
        if not self.spec.generate_report:
            return

        training_rows = len(self.datasets.training_data)
        test_rows = len(self.datasets.test_data) if self.datasets.test_data is not None else 0

        sections = [
            rc.Block(
                rc.Heading(self.spec.report_title, level=1),
                rc.Text(
                    f"You selected the {self.get_model_display_name()} model. "
                    f"Based on your dataset, you could have also selected any of the models: "
                    f"{self._candidate_models_text()}."
                ),
                rc.Text(self.get_model_description()),
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
            rc.Heading("Training Data Metrics", level=2),
            rc.Text(
                "These metrics summarize how closely the model fits the training "
                "data across the configured regression objectives."
            ),
            rc.DataTable(self.train_metrics, index=False),
            rc.Heading("Training Actual vs Predicted", level=2),
            rc.Text(
                "The following charts compare actual and predicted target values on "
                "the training dataset. The line chart highlights row-by-row tracking, "
                "while the scatter plot shows overall agreement between observed and "
                "predicted outcomes."
            ),
            self._build_actual_vs_predicted_line_plot(
                self.train_predictions,
                "Training Actual and Predicted Values by Row",
            ),
            self._build_actual_vs_predicted_scatter_plot(
                self.train_predictions,
                "Training Actual vs Predicted with Ideal Fit Reference",
            ),
            rc.Heading("Training Predictions (Top Rows)", level=3),
            rc.DataTable(self.train_predictions.head(25), index=False),
            rc.Heading("Feature Importance", level=2),
            rc.Text(
                "The following table and chart summarize which features had the "
                "largest influence on the fitted model."
            ),
        ]

        if self.feature_importance_df is not None and not self.feature_importance_df.empty:
            sections.extend(
                [
                    self._build_bar_plot(
                        self.feature_importance_df.head(20),
                        x="feature",
                        y="importance",
                        title="Feature Importance",
                        color=FEATURE_IMPORTANCE_COLOR,
                    ),
                    rc.DataTable(self.feature_importance_df.head(25), index=False),
                ]
            )
        else:
            sections.append(
                rc.Text("Feature importance is unavailable for this run.")
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
                        self._build_actual_vs_predicted_line_plot(
                            self.test_predictions,
                            "Test Actual and Predicted Values by Row",
                        ),
                        self._build_actual_vs_predicted_scatter_plot(
                            self.test_predictions,
                            "Test Actual vs Predicted with Ideal Fit Reference",
                        ),
                        rc.Heading("Test Predictions (Top Rows)", level=3),
                        rc.DataTable(self.test_predictions.head(25), index=False),
                    ]
                )

        if self.global_explanations_df is not None and not self.global_explanations_df.empty:
            sections.extend(
                [
                    rc.Heading("Global Explainability", level=2),
                    rc.Text(
                        "The following tables provide the feature attribution for the "
                        "global explainability."
                    ),
                    self._build_bar_plot(
                        self.global_explanations_df.head(20),
                        x="feature",
                        y="importance",
                        title="Global SHAP Importance",
                        color=GLOBAL_EXPLANATIONS_COLOR,
                    ),
                    rc.DataTable(self.global_explanations_df.head(25), index=False),
                ]
            )
        elif self.spec.generate_explanations:
            sections.extend(
                [
                    rc.Heading("Global Explainability", level=2),
                    rc.Text(
                        "Explainability was requested but outputs are unavailable. "
                        "Check logs for SHAP dependency or runtime errors."
                    ),
                ]
            )
        else:
            sections.extend(
                [
                    rc.Heading("Explainability", level=2),
                    rc.Text(
                        "Explainability is disabled for this run. Set "
                        "`generate_explanations: true` to include SHAP outputs."
                    ),
                ]
            )

        if self.local_explanations_df is not None and not self.local_explanations_df.empty:
            sections.extend(
                [
                    rc.Heading("Local Explanation of Models", level=2),
                    rc.Text(
                        "The following sample shows local feature attributions for "
                        "individual rows, which helps explain how feature values "
                        "influenced specific predictions."
                    ),
                    rc.DataTable(self.local_explanations_df.head(20), index=False),
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

        if self.spec.generate_explanations:
            try:
                self._generate_shap_explanations(x_train, x_proc)
            except Exception as e:
                logger.warning(f"Skipping explainability generation. Error: {e}")

        output_dir = self.spec.output_directory.url
        storage_options = default_signer() if ObjectStorageDetails.is_oci_path(output_dir) else {}

        self._write_outputs(output_dir, storage_options)
        elapsed_time = time.time() - start_time
        self._generate_report(output_dir, elapsed_time, storage_options)

        if self.spec.deploy_to_md:
            logger.warning(
                "`deploy_to_md` is enabled, but automatic Model Deployment is not yet wired for regression operator. "
                "Use generated `model.pkl` with `whatifserve/score.py` for deployment packaging."
            )

        logger.info(f"Regression artifacts generated at: {output_dir}")
        print(f"Regression artifacts generated at: {output_dir}")

        return {
            "output_directory": output_dir,
            "model": self.model_name,
            "metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
        }
