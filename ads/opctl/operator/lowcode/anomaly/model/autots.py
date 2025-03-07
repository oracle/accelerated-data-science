#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd

from ads.common.decorator.runtime_dependency import runtime_dependency

from .base_model import AnomalyOperatorBaseModel
from .anomaly_dataset import AnomalyOutput
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns


class AutoTSOperatorModel(AnomalyOperatorBaseModel):
    """Class representing AutoTS Anomaly Detection operator model."""

    @runtime_dependency(
        module="autots",
        err_msg=(
            "Please run `pip3 install autots` to "
            "install the required dependencies for AutoTS."
        ),
    )
    def _build_model(self) -> AnomalyOutput:
        from autots.evaluator.anomaly_detector import AnomalyDetector

        method = self.spec.model_kwargs.get("method")
        transform_dict = self.spec.model_kwargs.get("transform_dict", {})

        if method == "random" or method == "deep" or method == "fast":
            new_params = AnomalyDetector.get_new_params(method=method)
            transform_dict = new_params.pop("transform_dict")

            for key, value in new_params.items():
                self.spec.model_kwargs[key] = value

        if self.spec.model_kwargs.get("output") is None:
            self.spec.model_kwargs["output"] = "univariate"

        if "transform_dict" not in self.spec.model_kwargs:
            self.spec.model_kwargs["transform_dict"] = transform_dict

        if self.spec.contamination != 0.1:  # TODO: remove hard-coding
            self.spec.model_kwargs.get("method_params", {})[
                "contamination"
            ] = self.spec.contamination

        model = AnomalyDetector(**self.spec.model_kwargs)

        date_column = self.spec.datetime_column.name

        anomaly_output = AnomalyOutput(date_column=date_column)

        for target, df in self.datasets.full_data_dict.items():
            data = df.set_index(date_column)

            (anomaly, score) = model.detect(data)

            if len(anomaly.columns) == 1:
                score.rename(
                    columns={score.columns.values[0]: OutputColumns.SCORE_COL},
                    inplace=True,
                )
                score = 1 - score
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
        import report_creator as rc

        """The method that needs to be implemented on the particular model level."""
        other_sections = [
            rc.Heading("Selected Models Overview", level=2),
            rc.Text(
                "The following tables provide information regarding the chosen model."
            ),
        ]
        model_description = rc.Text(
            "The autots model automatically pre-processes, selects and engineers "
            "high-quality features in your dataset, which then given to an automatically "
            "chosen and optimized machine learning model.."
        )

        return (
            model_description,
            other_sections,
        )
