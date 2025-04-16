#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
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

        model_kwargs = self.spec.model_kwargs
        anomaly_output = AnomalyOutput(date_column="index")
        anomaly_threshold = model_kwargs.get("anomaly_threshold", 95)
        model_config_map = {}
        model_config_map = self._get_config_model(self.spec.model)

        date_column = self.spec.datetime_column.name

        anomaly_output = AnomalyOutput(date_column=date_column)
        # model_objects = defaultdict(list)
        for s_id, df in self.datasets.full_data_dict.items():
            df_clean = self._preprocess_data(df, date_column)
            data = TimeSeries.from_pd(df_clean)
            for _, (model_config, model) in model_config_map.items():
                if self.spec.model == SupportedModels.BOCPD:
                    model_config = model_config(**self.spec.model_kwargs)
                else:
                    model_config = model_config(
                        **{
                            **self.spec.model_kwargs,
                            "threshold": AggregateAlarms(
                                alm_threshold=model_kwargs.get("alm_threshold")
                                if model_kwargs.get("alm_threshold")
                                else None
                            ),
                        }
                    )
                if hasattr(model_config, "target_seq_index"):
                    model_config.target_seq_index = df.columns.get_loc(
                        self.spec.target_column
                    )
                model = model(model_config)
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
                    y_pred = (y_pred.to_pd().reset_index()["anom_score"] > 0).astype(
                        int
                    )
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
