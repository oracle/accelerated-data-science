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
    def _build_model(self) -> pd.DataFrame:
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

        target_category_column = (
            self.spec.target_category_columns[0]
            if self.spec.target_category_columns is not None
            else None
        )

        inliers = pd.DataFrame(columns=dataset.data.columns.values)
        outliers = pd.DataFrame(columns=dataset.data.columns.values)
        scores = pd.DataFrame(columns=[date_column, target_category_column, "score"])

        # Iterate over the full_data_dict items
        for target, df in full_data_dict.items():
            data = df.set_index(date_column)

            if self.spec.target_category_columns is not None:
                data = data.drop(self.spec.target_category_columns[0], axis=1)

            (anomaly, score) = model.detect(data)

            if len(anomaly.columns) == 1:
                anomaly = anomaly.reset_index(drop=True)

                score.rename(columns={score.columns.values[0]: "score"}, inplace=True)
                score = score.reset_index(drop=False)

                col = anomaly.columns.values[0]
                anomaly[col] = anomaly[col].replace({1: 0, -1: 1})

                outlier_indices = anomaly.index[anomaly[col] == 1]
                inlier_indices = anomaly.index[anomaly[col] == 0]

                if target_category_column is not None:
                    outliers = pd.concat(
                        [outliers, df.loc[outlier_indices]], axis=0, ignore_index=True
                    )
                    inliers = pd.concat(
                        [inliers, df.loc[inlier_indices]], axis=0, ignore_index=True
                    )

                    score[self.spec.target_category_columns[0]] = target
                    scores = pd.concat([scores, score], axis=0, ignore_index=True)

                else:
                    outliers = df.loc[outlier_indices]
                    inliers = df.loc[inlier_indices]
                    scores = score

                full_data_dict[target][OutputColumns.ANOMALY_COL] = anomaly[col].values

            else:
                "TBD"

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
