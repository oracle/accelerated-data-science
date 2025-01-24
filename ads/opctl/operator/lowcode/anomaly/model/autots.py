#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

import report_creator as rc

from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl import logger
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns

from ..const import SupportedModels
from .anomaly_dataset import AnomalyOutput
from .base_model import AnomalyOperatorBaseModel

logging.getLogger("report_creator").setLevel(logging.WARNING)


class AutoTSOperatorModel(AnomalyOperatorBaseModel):
    """Class representing AutoTS Anomaly Detection operator model."""

    model_mapping = {
        "isolationforest": "IsolationForest",
        "lof": "LOF",
        "ee": "EE",
        "zscore": "zscore",
        "rolling_zscore": "rolling_zscore",
        "mad": "mad",
        "minmax": "minmax",
        "iqr": "IQR",
    }

    @runtime_dependency(
        module="autots",
        err_msg=(
            "Please run `pip3 install autots` to "
            "install the required dependencies for AutoTS."
        ),
    )
    def _build_model(self) -> AnomalyOutput:
        from autots.evaluator.anomaly_detector import AnomalyDetector

        method = (
            SupportedModels.ISOLATIONFOREST
            if self.spec.model == SupportedModels.AutoTS
            else self.spec.model
        )
        model_params = {
            "method": self.model_mapping[method],
            "transform_dict": self.spec.model_kwargs.get("transform_dict", {}),
            "output": self.spec.model_kwargs.get("output", "univariate"),
            "method_params": {},
        }
        # Supported methods with contamination param
        if method in [
            SupportedModels.ISOLATIONFOREST,
            SupportedModels.LOF,
            SupportedModels.EE,
        ]:
            model_params["method_params"]["contamination"] = (
                self.spec.contamination if self.spec.contamination else 0.01
            )
        elif self.spec.contamination:
            raise ValueError(
                f'The contamination parameter is not supported for the selected model "{method}"'
            )
        logger.info(f"model params: {model_params}")

        model = AnomalyDetector(**model_params)

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
