#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd

from ads.common.decorator.runtime_dependency import runtime_dependency

from .base_model import AnomalyOperatorBaseModel
from .anomaly_dataset import AnomalyOutput


class AutoTSOperatorModel(AnomalyOperatorBaseModel):
    """Class representing TODS Anomaly Detection operator model."""

    @runtime_dependency(
        module="autots",
        err_msg=(
            "Please run `pip3 install autots` to "
            "install the required dependencies for TODS."
        ),
    )
    def _build_model(self) -> pd.DataFrame:
        from autots.evaluator.anomaly_detector import AnomalyDetector

        date_column = self.spec.datetime_column.name

        model = AnomalyDetector(
            output=self.spec.model_kwargs.get("output", "univariate"),
            method=self.spec.model_kwargs.get("method", "zscore"),
            transform_dict=self.spec.model_kwargs.get("tranform_dict", None),
            forecast_params=self.spec.model_kwargs.get("forecast_params", None),
            method_params=self.spec.model_kwargs.get("method_params", {}),
            eval_period=self.spec.model_kwargs.get("eval_period", None),
            n_jobs=self.spec.model_kwargs.get("n_jobs", 1),
        )

        data = self.datasets.data
        data.set_index(data_column)

        (anomaly, scores) = model.detect(data)

        inliers = pd.DataFrame()
        outliers = pd.DataFrame()

        if len(anomaly.columns) == 1:
            outlier_indices = anomaly.index[anomaly[anomaly.columns.values[0]] == -1]
            inlier_indices = anomaly.index[anomaly[anomaly.columns.values[0]] == 1]
            outliers = data[outlier_indices]
            inliers = data[inlier_indices]

        else:
            "TBD"

        self.anomaly_output = AnomalyOutput(
            inliers=inliers, ouliers=outliers, scores=scores
        )

        return self.anomaly_output

    def _generate_report(self):
        import datapane as dp

        """The method that needs to be implemented on the particular model level."""
        selected_models_text = dp.Text(
            f"## Selected Models Overview \n "
            "The following tables provide information regarding the chosen model."
        )
        all_sections = [selected_models_text]

        model_description = dp.Text(
            "The automlx model automatically pre-processes, selects and engineers "
            "high-quality features in your dataset, which then given to an automatically "
            "chosen and optimized machine learning model.."
        )
        other_sections = all_sections

        return (
            model_description,
            other_sections,
        )
