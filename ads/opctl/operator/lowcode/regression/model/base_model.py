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
from ads.opctl.operator.lowcode.regression.const import ColumnType, SupportedMetrics
from ads.opctl.operator.lowcode.regression.model.regression_dataset import RegressionDatasets
from ads.opctl.operator.lowcode.regression.operator_config import (
    RegressionOperatorConfig,
    RegressionOperatorSpec,
)

logging.getLogger("report_creator").setLevel(logging.WARNING)


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
            data=self.train_predictions,
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

        sections = [
            rc.Heading(self.spec.report_title, level=1),
            rc.Group(
                rc.Metric(heading="Model", value=self.model_name),
                rc.Metric(heading="Target", value=self.target_column),
                rc.Metric(heading="Features", value=len(self.feature_columns)),
                rc.Metric(
                    heading="Analysis completed in",
                    value=human_time_friendly(elapsed_time),
                ),
                label="Summary",
            ),
            rc.Heading("Training Metrics", level=2),
            rc.DataTable(self.train_metrics, index=False),
            rc.Heading("Training Actual vs Predicted", level=2),
            rc.Line(
                self._actual_vs_prediction_plot_data(self.train_predictions),
                x="row_id",
                y=["actual", "prediction"],
                label="Training Series",
            ),
            rc.Scatter(
                self.train_predictions,
                x="actual",
                y="prediction",
                label="Training: Actual vs Predicted",
            ),
            rc.Heading("Training Predictions (Top Rows)", level=2),
            rc.DataTable(self.train_predictions.head(25), index=False),
            rc.Heading("Data Preview", level=2),
            rc.DataTable(self.datasets.training_data[self.feature_columns + [self.target_column]].head(10), index=False),
            rc.Heading("Feature Importance", level=2),
            rc.DataTable(self.feature_importance_df.head(25), index=False),
        ]

        if self.test_metrics is not None and not self.test_metrics.empty:
            sections.extend(
                [
                    rc.Heading("Test Metrics", level=2),
                    rc.DataTable(self.test_metrics, index=False),
                ]
            )
            if self.test_predictions is not None and not self.test_predictions.empty:
                sections.extend(
                    [
                        rc.Heading("Test Actual vs Predicted", level=2),
                        rc.Line(
                            self._actual_vs_prediction_plot_data(self.test_predictions),
                            x="row_id",
                            y=["actual", "prediction"],
                            label="Test Series",
                        ),
                        rc.Scatter(
                            self.test_predictions,
                            x="actual",
                            y="prediction",
                            label="Test: Actual vs Predicted",
                        ),
                    ]
                )

        if self.global_explanations_df is not None and not self.global_explanations_df.empty:
            sections.extend(
                [
                    rc.Heading("Explainability", level=2),
                    rc.Text("SHAP-based global and local explainability outputs."),
                    rc.Heading("Global Explainability", level=3),
                    rc.Bar(
                        self.global_explanations_df.head(20),
                        x="feature",
                        y="importance",
                        label="Global SHAP Importance",
                    ),
                    rc.DataTable(self.global_explanations_df.head(25), index=False),
                ]
            )
        elif self.spec.generate_explanations:
            sections.extend(
                [
                    rc.Heading("Explainability", level=2),
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
                    rc.Heading("Local Explainability (Sample)", level=3),
                    rc.DataTable(self.local_explanations_df.head(20), index=False),
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
