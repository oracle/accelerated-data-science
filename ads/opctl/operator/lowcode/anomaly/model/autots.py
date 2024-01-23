#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd

from ads.common.decorator.runtime_dependency import runtime_dependency

from .base_model import AnomalyOperatorBaseModel
from .anomaly_dataset import AnomalyOutput
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns


class AutoTSOperatorModel(AnomalyOperatorBaseModel):
    """Class representing TODS Anomaly Detection operator model."""

    @runtime_dependency(
        module="autots",
        err_msg=(
            "Please run `pip3 install autots` to "
            "install the required dependencies for TODS."
        ),
    )
    def _build_model(self) -> AnomalyOutput:
        from autots.evaluator.anomaly_detector import AnomalyDetector

        method = self.spec.model_kwargs.get("method")

        if method == "random" or method == "deep" or method == "fast":
            new_params = AnomalyDetector.get_new_params(method=method)
            new_params.pop("transform_dict")

            for key, value in new_params.items():
                self.spec.model_kwargs[key] = value

        if self.spec.model_kwargs.get("output") is None:
            self.spec.model_kwargs["output"] = "univariate"

        if "transform_dict" not in self.spec.model_kwargs:
            self.spec.model_kwargs["transform_dict"] = {}

        model = AnomalyDetector(**self.spec.model_kwargs)

        date_column = self.spec.datetime_column.name
        dataset = self.datasets

        full_data_dict = dataset.full_data_dict

        anomaly_output = AnomalyOutput(date_column=date_column)

        # Iterate over the full_data_dict items
        for target, df in full_data_dict.items():
            data = df.set_index(date_column)

            if self.spec.target_category_columns is not None:
                data = data.drop(self.spec.target_category_columns[0], axis=1)

            (anomaly, score) = model.detect(data)

            if len(anomaly.columns) == 1:
                score.rename(
                    columns={score.columns.values[0]: OutputColumns.SCORE_COL},
                    inplace=True,
                )
                score = 1-score
                score = score.reset_index(drop=False)

                col = anomaly.columns.values[0]
                anomaly[col] = anomaly[col].replace({1: 0, -1: 1})
                anomaly.rename(columns={col: OutputColumns.ANOMALY_COL}, inplace=True)
                anomaly = anomaly.reset_index(drop=False)

                anomaly_output.add_output(target, anomaly, score)

            else:
                raise NotImplementedError(
                    "Multi-Output Anomaly Detection is not yet supported in autots"
                )

        return anomaly_output

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