#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import pandas as pd

from ads.common.decorator.runtime_dependency import runtime_dependency

from .base_model import AnomalyOperatorBaseModel
from .anomaly_dataset import AnomalyOutput
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns


class IsolationForestOperatorModel(AnomalyOperatorBaseModel):
    """Class representing OneClassSVM Anomaly Detection operator model."""

    @runtime_dependency(
        module="sklearn",
        err_msg=(
            "Please run `pip3 install scikit-learn` to "
            "install the required dependencies for OneClassSVM."
        ),
    )
    def _build_model(self) -> AnomalyOutput:
        from sklearn.ensemble import IsolationForest

        model_kwargs = self.spec.model_kwargs
        # map the output as per anomaly dataset class, 1: outlier, 0: inlier
        self.outlier_map = {1: 0, -1: 1}

        anomaly_output = AnomalyOutput(date_column="index")

        for target, df in self.datasets.full_data_dict.items():
            model = IsolationForest(**model_kwargs)
            model.fit(df)
            y_pred = np.vectorize(self.outlier_map.get)(
                model.predict(df)
            )

            scores = model.score_samples(
                df
            )

            index_col = df.columns[0]

            anomaly = pd.DataFrame(
                {index_col: df[index_col], OutputColumns.ANOMALY_COL: y_pred}
            ).reset_index(drop=True)
            score = pd.DataFrame(
                {"index": df[index_col], OutputColumns.SCORE_COL: scores}
            ).reset_index(drop=True)

            anomaly_output.add_output(target, anomaly, score)

        return anomaly_output

    def _generate_report(self):
        """Generates the report."""
        import report_creator as rc

        other_sections = [
            rc.Heading("Selected Models Overview", level=2),
            rc.Text(
                "The following tables provide information regarding the chosen model."
            ),
        ]

        model_description = rc.Text(
            "The Isolation Forest is an ensemble of “Isolation Trees” that “isolate” observations by recursive random partitioning"
            " which can be represented by a tree structure. The number of splittings required to isolate a sample is lower for outliers and higher for inliers."
        )

        return (
            model_description,
            other_sections,
        )
