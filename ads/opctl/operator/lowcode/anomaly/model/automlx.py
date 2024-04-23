#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd

from ads.common.decorator.runtime_dependency import runtime_dependency
from .anomaly_dataset import AnomalyOutput

from .base_model import AnomalyOperatorBaseModel
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns


class AutoMLXOperatorModel(AnomalyOperatorBaseModel):
    """Class representing AutoMLX operator model."""

    @runtime_dependency(
        module="automlx",
        err_msg=(
            "Please run `pip3 install oracle-automlx>=23.4.1` and "
            "`pip3 install oracle-automlx[classic]>=23.4.1` "
            "to install the required dependencies for automlx."
        ),
    )
    def _build_model(self) -> pd.DataFrame:
        from automlx import init
        import logging

        try:
            init(
                engine="ray",
                engine_opts={"ray_setup": {"_temp_dir": "/tmp/ray-temp"}},
                loglevel=logging.CRITICAL,
            )
        except Exception as e:
            logger.info("Ray already initialized")
        date_column = self.spec.datetime_column.name
        anomaly_output = AnomalyOutput(date_column=date_column)
        time_budget = self.spec.model_kwargs.pop("time_budget", -1)

        # Iterate over the full_data_dict items
        for target, df in self.datasets.full_data_dict.items():
            est = automlx.Pipeline(task="anomaly_detection", **self.spec.model_kwargs)
            est.fit(
                X=df,
                X_valid=self.X_valid_dict[target]
                if self.X_valid_dict is not None
                else None,
                y_valid=self.y_valid_dict[target]
                if self.y_valid_dict is not None
                else None,
                contamination=self.spec.contamination
                if self.y_valid_dict is not None
                else None,
                time_budget=time_budget,
            )
            y_pred = est.predict(df)
            scores = est.predict_proba(df)

            anomaly = pd.DataFrame(
                {date_column: df[date_column], OutputColumns.ANOMALY_COL: y_pred}
            ).reset_index(drop=True)
            score = pd.DataFrame(
                {
                    date_column: df[date_column],
                    OutputColumns.SCORE_COL: [item[1] for item in scores],
                }
            ).reset_index(drop=True)
            anomaly_output.add_output(target, anomaly, score)

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
            "The automlx model automatically pre-processes, selects and engineers "
            "high-quality features in your dataset, which then given to an automatically "
            "chosen and optimized machine learning model.."
        )

        return (
            model_description,
            other_sections,
        )
