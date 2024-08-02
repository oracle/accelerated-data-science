#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import pandas as pd

from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns

from .anomaly_dataset import AnomalyOutput
from .base_model import AnomalyOperatorBaseModel


class RandomCutForestOperatorModel(AnomalyOperatorBaseModel):
    """
    Class representing Random Cut Forest Anomaly Detection operator model.
    """

    @runtime_dependency(
        module="rrcf",
        err_msg=(
            "Please run `pip install rrcf` to "
            "install the required dependencies for RandomCutForest."
        ),
    )
    def _build_model(self) -> AnomalyOutput:
        from rrcf import RCTree

        model_kwargs = self.spec.model_kwargs
        # map the output as per anomaly dataset class, 1: outlier, 0: inlier
        self.outlier_map = {1: 0, -1: 1}

        anomaly_output = AnomalyOutput(date_column="index")
        #TODO: PDB
        import pdb

        pdb.set_trace()

        for target, df in self.datasets.full_data_dict.items():
            model = RCTree(**model_kwargs)
            model.fit(df)
            y_pred = model.predict(df)
            y_pred = np.vectorize(self.outlier_map.get)(y_pred)

            scores = model.score_samples(df)

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
            "The Random Cut Forest (RCF) is an unsupervised machine learning algorithm that is used for anomaly detection."
            " It works by building an ensemble of binary trees (random cut trees) and using them to compute anomaly scores for data points."
        )

        return (
            model_description,
            other_sections,
        )
