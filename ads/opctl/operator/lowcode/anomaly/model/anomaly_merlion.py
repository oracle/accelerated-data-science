#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import logging

import numpy as np
import pandas as pd
import report_creator as rc
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils import TimeSeries

from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl.operator.lowcode.anomaly.const import (
    MERLIONAD_IMPORT_MODEL_MAP,
    MERLIONAD_MODEL_MAP,
    OutputColumns,
    SupportedModels,
)

from .anomaly_dataset import AnomalyOutput
from .base_model import AnomalyOperatorBaseModel

logging.getLogger("report_creator").setLevel(logging.WARNING)


def prepare_model_kwargs(model_name, model_kwargs):
    model_name = MERLIONAD_MODEL_MAP.get(model_name)

    # individual handling by model
    if model_name == SupportedModels.BOCPD:
        if (
            model_kwargs.get("threshold", None) is None
            and model_kwargs.get("alm_threshold") is not None
        ):
            model_kwargs["threshold"] = AggregateAlarms(
                alm_threshold=model_kwargs.get("alm_threshold")
            )
    elif (
        model_name == SupportedModels.MSES
        and model_kwargs.get("max_forecast_steps", None) is None
    ):
        model_kwargs["max_forecast_steps"] = 1
    return model_kwargs


def init_merlion_model(model_name, model_kwargs):
    from merlion.models.factory import ModelFactory

    model_name = MERLIONAD_MODEL_MAP.get(model_name)

    if model_name == "DeepPointAnomalyDetector":
        from merlion.models.anomaly.deep_point_anomaly_detector import (
            DeepPointAnomalyDetector,
        )

        model = DeepPointAnomalyDetector(
            DeepPointAnomalyDetector.config_class(**model_kwargs)
        )
        unused_model_kwargs = model_kwargs
    else:
        model, unused_model_kwargs = ModelFactory.create(
            model_name, return_unused_kwargs=True, **model_kwargs
        )

    return model, unused_model_kwargs


class AnomalyMerlionOperatorModel(AnomalyOperatorBaseModel):
    """Class representing Merlion Anomaly Detection operator model."""

    @runtime_dependency(
        module="merlion",
        err_msg=(
            "Please run `pip3 install salesforce-merlion[all]` to "
            "install the required packages."
        ),
    )
    def _get_config_model(self, model_name):
        """
        Returns a dictionary with model names as keys and a list of model config and model object as values.

        Parameters
        ----------
        model_name : str
            model name from the Merlion model list.

        Returns
        -------
        dict
            A dictionary with model names as keys and a list of model config and model object as values.
        """
        model_config_map = {}
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

    def _preprocess_data(self, df, date_column):
        df[date_column] = pd.to_datetime(df[date_column])
        if df[date_column].dt.tz is not None:
            df[date_column] = df[date_column].dt.tz_convert(None)
        data = df.set_index(date_column)
        return data

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

        def _inject_train_data(
            v_data: pd.DataFrame, train_data: pd.DataFrame
        ) -> pd.DataFrame:
            # Step 1: Get index from train data not already present in validation data
            v_index_set = set(v_data.index)
            filtered_train = train_data[~train_data.index.isin(v_index_set)]

            combined_data = pd.concat([filtered_train, v_data]).sort_index()
            return combined_data

        model_kwargs = prepare_model_kwargs(
            self.spec.model, self.spec.model_kwargs.copy()
        )
        anomaly_output = AnomalyOutput(date_column="index")
        anomaly_threshold = model_kwargs.get("anomaly_threshold", 95)
        date_column = self.spec.datetime_column.name

        anomaly_output = AnomalyOutput(date_column=date_column)

        for s_id, df in self.datasets.full_data_dict.items():
            df_clean = self._preprocess_data(df, date_column)
            data = TimeSeries.from_pd(df_clean)
            target_seq_index = df_clean.columns.get_loc(self.spec.target_column)
            model, unused_model_kwargs = init_merlion_model(
                self.spec.model, target_seq_index, model_kwargs
            )
            scores = None

            if (
                hasattr(self.datasets, "valid_data")
                and self.datasets.valid_data.get_data_for_series(s_id) is not None
            ):
                # try:
                v_df = self.datasets.valid_data.get_data_for_series(s_id)
                v_data = self._preprocess_data(v_df, date_column)

                v_labels = TimeSeries.from_pd(v_data["anomaly"])
                v_data = v_data.drop("anomaly", axis=1)
                v_data = _inject_train_data(v_data, df_clean)
                scores_v = model.train(
                    train_data=TimeSeries.from_pd(v_data), anomaly_labels=v_labels
                )
                scores = TimeSeries.from_pd(scores_v.to_pd().loc[df_clean.index])
                # except Exception as e:
                #     logging.debug(f"Failed to use validation data with error: {e}")
            if scores is None:
                scores = model.train(train_data=data)

            # Normalize scores out of 100
            scores = scores.to_pd().reset_index()
            scores["anom_score"] = (
                scores["anom_score"] - scores["anom_score"].min()
            ) / (scores["anom_score"].max() - scores["anom_score"].min())

            try:
                y_pred = model.get_anomaly_label(data)
                y_pred = (y_pred.to_pd().reset_index()["anom_score"] > 0).astype(int)
            except Exception:
                y_pred = (
                    scores["anom_score"]
                    > np.percentile(
                        scores["anom_score"],
                        anomaly_threshold,
                    )
                ).astype(int)

            index_col = df.columns[0]

            anomaly = pd.DataFrame(
                {index_col: df[index_col], OutputColumns.ANOMALY_COL: y_pred}
            ).reset_index(drop=True)
            score = pd.DataFrame(
                {
                    index_col: df[index_col],
                    OutputColumns.SCORE_COL: scores["anom_score"],
                }
            ).reset_index(drop=True)

            anomaly_output.add_output(s_id, anomaly, score)
        return anomaly_output

    def _generate_report(self):
        """Genreates a report for the model."""
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
