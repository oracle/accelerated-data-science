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
        # self.outlier_map = {1: 0, -1: 1}

        anomaly_output = AnomalyOutput(date_column="index")

        # Set tree parameters
        num_trees = model_kwargs.get("num_trees", 200)
        shingle_size = model_kwargs.get("shingle_size", 1)
        tree_size = model_kwargs.get("tree_size", 1000)

        for target, df in self.datasets.full_data_dict.items():
            df_values = df[self.spec.target_column].astype(float).values

            # TODO: Update size to log logic
            points = np.vstack(list(rrcf.shingle(df_values, size=4)))

            # TODO: remove hardcode
            sample_size_range = (1, 6)
            n = points.shape[0]
            avg_codisp = pd.Series(0.0, index=np.arange(n))
            index = np.zeros(n)

            forest = []
            while len(forest) < num_trees:
                ixs = np.random.choice(n, size=sample_size_range, replace=False)
                trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
                forest.extend(trees)
                print(len(forest))

            for tree in forest:
                codisp = pd.Series({leaf: tree.codisp(leaf) for leaf in tree.leaves})
                avg_codisp[codisp.index] += codisp
                np.add.at(index, codisp.index.values, 1)

            avg_codisp /= index
            # TODO: remove hardcode
            avg_codisp.index = df.iloc[(4 - 1) :].index
            avg_codisp = (avg_codisp - avg_codisp.min()) / (
                avg_codisp.max() - avg_codisp.min()
            )

            # TODO: use model kwargs for percentile threshold
            y_pred = (avg_codisp > np.percentile(avg_codisp, 95)).astype(int)

            # TODO: rem pdb
            # import pdb

            # pdb.set_trace()
            print("Done")

            # scores = model.score_samples(df)

            # index_col = df.columns[0]

            # anomaly = pd.DataFrame(
            #     {index_col: df[index_col], OutputColumns.ANOMALY_COL: y_pred}
            # ).reset_index(drop=True)
            # score = pd.DataFrame(
            #     {"index": df[index_col], OutputColumns.SCORE_COL: scores}
            # ).reset_index(drop=True)

            # anomaly_output.add_output(target, anomaly, score)

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
