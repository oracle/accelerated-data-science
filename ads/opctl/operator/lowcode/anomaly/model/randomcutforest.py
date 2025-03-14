#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

import numpy as np
import pandas as pd
import report_creator as rc

from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl import logger
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns

from .anomaly_dataset import AnomalyOutput
from .base_model import AnomalyOperatorBaseModel

logging.getLogger("report_creator").setLevel(logging.WARNING)


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
        import rrcf

        model_kwargs = self.spec.model_kwargs

        anomaly_output = AnomalyOutput(date_column="index")

        # Set tree parameters
        num_trees = model_kwargs.get("num_trees", 200)
        shingle_size = model_kwargs.get("shingle_size", None)
        anomaly_threshold = model_kwargs.get("anomaly_threshold", 95)

        for target, df in self.datasets.full_data_dict.items():
            try:
                if df.shape[0] == 1:
                    raise ValueError("Dataset size must be greater than 1")
                df_values = df[self.spec.target_column].astype(float).values

                cal_shingle_size = (
                    shingle_size
                    if shingle_size
                    else int(2 ** np.floor(np.log2(df.shape[0])) / 2)
                )
                points = np.vstack(list(rrcf.shingle(df_values, size=cal_shingle_size)))

                sample_size_range = (1, points.shape[0])
                n = points.shape[0]
                avg_codisp = pd.Series(0.0, index=np.arange(n))
                index = np.zeros(n)

                forest = []
                while len(forest) < num_trees:
                    ixs = np.random.choice(n, size=sample_size_range, replace=False)
                    trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
                    forest.extend(trees)

                for tree in forest:
                    codisp = pd.Series(
                        {leaf: tree.codisp(leaf) for leaf in tree.leaves}
                    )
                    avg_codisp[codisp.index] += codisp
                    np.add.at(index, codisp.index.values, 1)

                avg_codisp /= index
                avg_codisp.index = df.iloc[(cal_shingle_size - 1) :].index
                avg_codisp = (avg_codisp - avg_codisp.min()) / (
                    avg_codisp.max() - avg_codisp.min()
                )

                y_pred = (
                    avg_codisp > np.percentile(avg_codisp, anomaly_threshold)
                ).astype(int)

                index_col = df.columns[0]

                anomaly = pd.DataFrame(
                    {index_col: y_pred.index, OutputColumns.ANOMALY_COL: y_pred}
                ).reset_index(drop=True)
                score = pd.DataFrame(
                    {"index": avg_codisp.index, OutputColumns.SCORE_COL: avg_codisp}
                ).reset_index(drop=True)

                anomaly_output.add_output(target, anomaly, score)
            except Exception as e:
                logger.warning(f"Encountered Error: {e}. Skipping series {target}.")

        return anomaly_output

    def _generate_report(self):
        """Generates the report."""

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
