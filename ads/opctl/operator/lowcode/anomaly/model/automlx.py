#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd

from ads.common.decorator.runtime_dependency import runtime_dependency
from .anomaly_dataset import AnomalyOutput

from .base_model import AnomalyOperatorBaseModel
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns

class AutoMLXOperatorModel(AnomalyOperatorBaseModel):
    """Class representing AutoMLX operator model."""

    @runtime_dependency(
        module="automl",
        err_msg=(
            "Please run `pip3 install oracle-automlx==23.2.3` to "
            "install the required dependencies for automlx."
        ),
    )
    def _build_model(self) -> pd.DataFrame:


        date_column = self.spec.datetime_column.name
        dataset = self.datasets

        full_data_dict = dataset.full_data_dict

        anomaly_output = AnomalyOutput(date_column=date_column)

        # Iterate over the full_data_dict items
        for target, df in full_data_dict.items():
            est = automl.Pipeline(task='anomaly_detection')
            est.fit(df, y=None)
            y_pred = est.predict(df)
            scores = est.predict_proba(df)

            anomaly = pd.DataFrame({
                date_column: df[date_column],
                OutputColumns.ANOMALY_COL: y_pred
            }).reset_index(drop=True)
            score = pd.DataFrame({
                date_column: df[date_column],
                OutputColumns.SCORE_COL: [item[1] for item in scores]
            }).reset_index(drop=True)
            anomaly_output.add_output(target, anomaly, score)

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
