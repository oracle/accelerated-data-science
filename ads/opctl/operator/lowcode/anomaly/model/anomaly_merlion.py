#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
from collections import defaultdict

import numpy as np
import pandas as pd
from merlion.utils import TimeSeries

from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl.operator.lowcode.anomaly.const import (
    MERLION_DEFAULT_MODEL,
    MERLIONAD_IMPORT_MODEL_MAP,
    MERLIONAD_MODEL_MAP,
    OutputColumns,
)

from .anomaly_dataset import AnomalyOutput
from .base_model import AnomalyOperatorBaseModel


class AnomalyMerlionOperatorModel(AnomalyOperatorBaseModel):
    """Class representing Merlion Anomaly Detection operator model."""

    @runtime_dependency(
        module="merlion",
        err_msg=(
            "Please run `pip3 install salesforce-merlion[all]` to "
            "install the required packages."
        ),
    )
    def _get_config_model(self, model_list):
        """
        Returns a dictionary with model names as keys and a list of model config and model object as values.

        Parameters
        ----------
        model_list : list
            A list of model names.

        Returns
        -------
        dict
            A dictionary with model names as keys and a list of model config and model object as values.
        """
        model_config_map = {}
        for model_name in model_list:
            model_module = importlib.import_module(
                name=MERLIONAD_IMPORT_MODEL_MAP.get(model_name),
                package="merlion.models.anomaly",
            )
            model_config = getattr(
                model_module, MERLIONAD_MODEL_MAP.get(model_name) + "Config"
            )
            model = getattr(model_module, MERLIONAD_MODEL_MAP.get(model_name))
            model_config_map[model_name] = [model_config, model]
        return model_config_map

    def _build_model(self) -> AnomalyOutput:
        """
        Builds a Merlion anomaly detection model and trains it using the given data.

        Parameters
        ----------
        None

        Returns
        -------
        AnomalyOutput
            An AnomalyOutput object containing the anomaly detection results.
        """
        model_kwargs = self.spec.model_kwargs
        anomaly_output = AnomalyOutput(date_column="index")
        anomaly_threshold = model_kwargs.get("anomaly_threshold", 95)
        model_config_map = {}
        if model_kwargs.get("sub_model", None):
            model_config_map = self._get_config_model(model_kwargs.get("sub_model"))
        else:
            from merlion.models.anomaly.forecast_based.prophet import (  # noqa: I001
                ProphetDetector,
                ProphetDetectorConfig,
            )

            model_config_map[MERLION_DEFAULT_MODEL] = [
                ProphetDetectorConfig,
                ProphetDetector,
            ]

            date_column = self.spec.datetime_column.name

            anomaly_output = AnomalyOutput(date_column=date_column)
            # model_objects = defaultdict(list)
            for target, df in self.datasets.full_data_dict.items():
                data = df.set_index(date_column)
                data = TimeSeries.from_pd(data)
                for model_name, (model_config, model) in model_config_map.items():
                    model_config = model_config(**self.spec.model_kwargs)
                    if hasattr(model_config, "target_seq_index"):
                        model_config.target_seq_index = df.columns.get_loc(self.spec.target_column)
                    model = model(model_config)


                    scores = model.train(train_data=data, anomaly_labels=None)

                    try:
                        y_pred = model.get_anomaly_label(data)
                        y_pred =(y_pred.to_pd().reset_index()["anom_score"] > 0).astype(int)
                    except Exception as e:
                        y_pred = (
                            scores.to_pd().reset_index()["anom_score"]
                            > np.percentile(
                                scores.to_pd().reset_index()["anom_score"], anomaly_threshold
                            )
                        ).astype(int)

                    index_col = df.columns[0]

                    anomaly = pd.DataFrame(
                        {index_col: df[index_col], OutputColumns.ANOMALY_COL: y_pred}
                    ).reset_index(drop=True)
                    score = pd.DataFrame(
                        {
                            index_col: df[index_col],
                            OutputColumns.SCORE_COL: scores.to_pd().reset_index()[
                                "anom_score"
                            ],
                        }
                    ).reset_index(drop=True)
                    # model_objects[model_name].append(model)

                    anomaly_output.add_output(target, anomaly, score)
        return anomaly_output

    def _generate_report(self):
        """Genreates a report for the model."""
        import report_creator as rc

        other_sections = [
            rc.Heading("Selected Models Overview", level=2),
            rc.Text(
                "The following tables provide information regarding the chosen model."
            ),
        ]

        model_description = rc.Text(
            "The Merlion anomaly detection model is a full-stack automated machine learning system for anomaly detection."
        )

        return (
            model_description,
            other_sections,
        )
