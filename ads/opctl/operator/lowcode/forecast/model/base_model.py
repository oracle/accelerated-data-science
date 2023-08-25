#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Tuple

import datapane as dp
import fsspec
import numpy as np
import pandas as pd

from ads.common.auth import default_signer
from ads.opctl import logger

from .. import utils
from ..const import SupportedModels
from ..operator_config import ForecastOperatorConfig, ForecastOperatorSpec
from .transformations import Transformations

class ForecastOperatorBaseModel(ABC):
    """The base class for the forecast operator models."""

    def __init__(self, config: ForecastOperatorConfig):
        """Instantiates the ForecastOperatorBaseModel instance.

        Properties
        ----------
        config: ForecastOperatorConfig
            The forecast operator configuration.
        """

        self.config: ForecastOperatorConfig = config
        self.spec: ForecastOperatorSpec = config.spec

        # these fields are populated in the _load_data() method
        self.original_user_data = None
        self.original_total_data = None
        self.original_additional_data = None
        self.full_data_dict = None
        self.target_columns = None
        self.categories = None
        self.original_target_column = self.spec.target_column

        # these fields are populated in the _build_model() method
        self.data = None
        self.models = None
        self.outputs = None

        self.target_columns = (
            None  # This will become [target__category1__category2 ...]
        )

        self.perform_tuning = self.spec.tuning != None

    def generate_report(self):
        """Generates the forecasting report."""

        # load data and build models
        start_time = time.time()
        self._load_data()
        result_df = self._build_model()
        elapsed_time = time.time() - start_time

        # build the report
        (
            model_description,
            other_sections,
            forecast_col_name,
            train_metrics,
            ds_column_series,
            ds_forecast_col,
            ci_col_names,
        ) = self._generate_report()

        report_sections = []
        title_text = dp.Text("# Forecast Report")

        md_columns = " * ".join([f"{x} \n" for x in self.target_columns])
        summary = dp.Blocks(
            dp.Select(
                blocks=[
                    dp.Group(
                        dp.Text(f"You selected the **`{self.spec.model}`** model."),
                        model_description,
                        dp.Text(
                            "Based on your dataset, you could have also selected "
                            f"any of the models: `{'`, `'.join(SupportedModels.keys())}`."
                        ),
                        dp.Group(
                            dp.BigNumber(
                                heading="Analysis was completed in ",
                                value=utils.human_time_friendly(elapsed_time),
                            ),
                            dp.BigNumber(
                                heading="Starting time index",
                                value=ds_column_series.min().strftime(
                                    "%B %d, %Y"
                                ),  # "%r" # TODO: Figure out a smarter way to format
                            ),
                            dp.BigNumber(
                                heading="Ending time index",
                                value=ds_column_series.max().strftime(
                                    "%B %d, %Y"
                                ),  # "%r" # TODO: Figure out a smarter way to format
                            ),
                            dp.BigNumber(
                                heading="Num series", value=len(self.target_columns)
                            ),
                            columns=4,
                        ),
                        dp.Text("### First 10 Rows of Data"),
                        dp.Select(
                            blocks=[
                                dp.DataTable(
                                    df.head(10).rename(
                                        {col: self.spec.target_column}, axis=1
                                    ),
                                    caption="Start",
                                    label=col,
                                )
                                for col, df in self.full_data_dict.items()
                            ]
                        ),
                        dp.Text("----"),
                        dp.Text("### Last 10 Rows of Data"),
                        dp.Select(
                            blocks=[
                                dp.DataTable(
                                    df.tail(10).rename(
                                        {col: self.spec.target_column}, axis=1
                                    ),
                                    caption="End",
                                    label=col,
                                )
                                for col, df in self.full_data_dict.items()
                            ]
                        ),
                        dp.Text("### Data Summary Statistics"),
                        dp.Select(
                            blocks=[
                                dp.DataTable(
                                    df.rename(
                                        {col: self.spec.target_column}, axis=1
                                    ).describe(),
                                    caption="Summary Statistics",
                                    label=col,
                                )
                                for col, df in self.full_data_dict.items()
                            ]
                        ),
                        label="Summary",
                    ),
                    dp.Text(
                        "The following report compares a variety of metrics and plots "
                        f"for your target columns: \n {md_columns}.\n",
                        label="Target Columns",
                    ),
                ]
            ),
        )

        train_metric_sections = []
        if train_metrics:
            self.eval_metrics = utils.evaluate_metrics(
                self.target_columns,
                self.data,
                self.outputs,
                target_col=forecast_col_name,
            )
            sec6_text = dp.Text(f"## Historical Data Evaluation Metrics")
            sec6 = dp.DataTable(self.eval_metrics)
            train_metric_sections = [sec6_text, sec6]

        test_eval_metrics = []
        test_data = None
        if self.spec.test_data:
            (
                self.test_eval_metrics,
                summary_metrics,
                test_data,
            ) = self._test_evaluate_metrics(
                target_columns=self.target_columns,
                test_filename=self.spec.test_data.url,
                outputs=self.outputs,
                target_col=forecast_col_name,
                elapsed_time=elapsed_time,
            )
            sec7_text = dp.Text(f"## Holdout Data Evaluation Metrics")
            sec7 = dp.DataTable(self.test_eval_metrics)

            sec8_text = dp.Text(f"## Holdout Data Summary Metrics")
            sec8 = dp.DataTable(summary_metrics)

            test_eval_metrics = [sec7_text, sec7, sec8_text, sec8]

        forecast_text = dp.Text(f"## Forecasted Data Overlaying Historical")
        forecast_sec = utils.get_forecast_plots(
            self.data,
            self.outputs,
            self.target_columns,
            test_data=test_data,
            forecast_col_name=forecast_col_name,
            ds_col=ds_column_series,
            ds_forecast_col=ds_forecast_col,
            ci_col_names=ci_col_names,
            ci_interval_width=self.spec.confidence_interval_width,
        )
        forecast_plots = [forecast_text, forecast_sec]

        yaml_appendix_title = dp.Text(f"## Reference: YAML File")
        yaml_appendix = dp.Code(code=self.config.to_yaml(), language="yaml")
        report_sections = (
            [title_text, summary]
            + forecast_plots
            + other_sections
            + test_eval_metrics
            + train_metric_sections
            + [yaml_appendix_title, yaml_appendix]
        )

        # save the report and result CSV
        self._save_report(report_sections=report_sections, result_df=result_df)

    def _load_data(self):
        """Loads forecasting input data."""

        raw_data = utils._load_data(
            filename=self.spec.historical_data.url,
            format=self.spec.historical_data.format,
            storage_options=default_signer(),
            columns=self.spec.historical_data.columns,
        )
        self.original_user_data = raw_data.copy()
        data = Transformations(raw_data, self.spec).run()
        self.original_total_data = data
        additional_data = None
        if self.spec.additional_data is not None:
            additional_data = utils._load_data(
                filename=self.spec.additional_data.url,
                format=self.spec.additional_data.format,
                storage_options=default_signer(),
                columns=self.spec.additional_data.columns,
            )

            self.original_additional_data = additional_data.copy()
            self.original_total_data = pd.concat([data, additional_data], axis=1)
        (
            self.full_data_dict,
            self.target_columns,
            self.categories,
        ) = utils._build_indexed_datasets(
            data=data,
            target_column=self.spec.target_column,
            datetime_column=self.spec.datetime_column.name,
            horizon=self.spec.horizon.periods,
            target_category_columns=self.spec.target_category_columns,
            additional_data=additional_data,
        )

    def _test_evaluate_metrics(
        self, target_columns, test_filename, outputs, target_col="yhat", elapsed_time=0
    ):
        total_metrics = pd.DataFrame()

        data = utils._load_data(
            filename=test_filename,
            format=self.spec.test_data.format,
            storage_options=default_signer(),
            columns=self.spec.test_data.columns,
        )

        data = self._preprocess(
            data, self.spec.datetime_column.name, self.spec.datetime_column.format
        )
        data, confirm_targ_columns = utils._clean_data(
            data=data,
            target_column=self.original_target_column,
            target_category_columns=self.spec.target_category_columns,
            datetime_column="ds",
        )

        for idx, col in enumerate(target_columns):
            y_true = np.asarray(data[col])
            y_pred = np.asarray(outputs[idx][target_col][-len(y_true) :])

            metrics_df = utils._build_metrics_df(
                y_true=y_true, y_pred=y_pred, column_name=col
            )
            total_metrics = pd.concat([total_metrics, metrics_df], axis=1)

        summary_metrics = pd.DataFrame(
            {
                "Mean sMAPE": np.mean(total_metrics.loc["sMAPE"]),
                "Median sMAPE": np.median(total_metrics.loc["sMAPE"]),
                "Mean MAPE": np.mean(total_metrics.loc["MAPE"]),
                "Median MAPE": np.median(total_metrics.loc["MAPE"]),
                "Mean RMSE": np.mean(total_metrics.loc["RMSE"]),
                "Median RMSE": np.median(total_metrics.loc["RMSE"]),
                "Mean r2": np.mean(total_metrics.loc["r2"]),
                "Median r2": np.median(total_metrics.loc["r2"]),
                "Mean Explained Variance": np.mean(
                    total_metrics.loc["Explained Variance"]
                ),
                "Median Explained Variance": np.median(
                    total_metrics.loc["Explained Variance"]
                ),
                "Elapsed Time": elapsed_time,
            },
            index=["All Targets"],
        )
        return total_metrics, summary_metrics, data

    def _save_report(self, report_sections: Tuple, result_df: pd.DataFrame):
        """Saves resulting reports to the given folder."""
        if self.spec.output_directory:
            output_dir = self.spec.output_directory.url
        else:
            output_dir = "tmp_fc_operator_result"
            logger.warn(
                "Since the output directory was not specified, the output will be saved to {} directory.".format(
                    output_dir))
        # datapane html report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_local_path = os.path.join(temp_dir, "___report.html")
            dp.save_report(report_sections, report_local_path)
            with open(report_local_path) as f1:
                with fsspec.open(
                        os.path.join(output_dir, self.spec.report_file_name),
                        "w",
                        **default_signer(),
                ) as f2:
                    f2.write(f1.read())

        # metrics csv report
        utils._write_data(
            data=result_df,
            filename=os.path.join(output_dir, self.spec.report_metrics_name),
            format="csv",
            storage_options=default_signer(),
        )

        logger.warn(
            f"The report has been successfully "
            f"generated and placed to the: {output_dir}."
        )

    def _preprocess(self, data, ds_column, datetime_format):
        """The method that needs to be implemented on the particular model level."""
        data["ds"] = pd.to_datetime(data[ds_column], format=datetime_format)
        if ds_column != "ds":
            data.drop([ds_column], axis=1, inplace=True)
        return data

    @abstractmethod
    def _generate_report(self):
        """
        Generates the report for the particular model.
        The method that needs to be implemented on the particular model level.
        """

    @abstractmethod
    def _build_model(self) -> pd.DataFrame:
        """
        Build the model.
        The method that needs to be implemented on the particular model level.
        """
