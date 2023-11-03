#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Tuple

import fsspec
import numpy as np
import pandas as pd

from ads.common.auth import default_signer
from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl import logger

from .. import utils
from ..const import SUMMARY_METRICS_HORIZON_LIMIT, SupportedMetrics, SupportedModels
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
        self.test_eval_metrics = None
        self.original_target_column = self.spec.target_column

        # these fields are populated in the _build_model() method
        self.data = None
        self.models = None
        self.outputs = None

        self.target_columns = (
            None  # This will become [target__category1__category2 ...]
        )
        self.train_metrics = False
        self.forecast_col_name = "yhat"
        self.perform_tuning = self.spec.tuning != None

    def generate_report(self):
        """Generates the forecasting report."""
        import datapane as dp

        # load data and build models
        start_time = time.time()
        self._load_data()
        result_df = self._build_model()
        elapsed_time = time.time() - start_time

        # Generate metrics
        summary_metrics = None
        test_data = None
        self.eval_metrics = None

        if self.spec.generate_report or self.spec.generate_metrics:
            if self.train_metrics:
                self.eval_metrics = utils.evaluate_metrics(
                    self.target_columns,
                    self.data,
                    self.outputs,
                    target_col=self.forecast_col_name,
                )

            if self.spec.test_data:
                (
                    self.test_eval_metrics,
                    summary_metrics,
                    test_data,
                ) = self._test_evaluate_metrics(
                    target_columns=self.target_columns,
                    test_filename=self.spec.test_data.url,
                    outputs=self.outputs,
                    target_col=self.forecast_col_name,
                    elapsed_time=elapsed_time,
                )
        report_sections = []

        if self.spec.generate_report:
            # build the report
            (
                model_description,
                other_sections,
                ds_column_series,
                ds_forecast_col,
                ci_col_names,
            ) = self._generate_report()

            title_text = dp.Text("# Forecast Report")

            md_columns = " * ".join([f"{x} \n" for x in self.target_columns])
            first_10_rows_blocks = [
                dp.DataTable(
                    df.head(10).rename({col: self.spec.target_column}, axis=1),
                    caption="Start",
                    label=col,
                )
                for col, df in self.full_data_dict.items()
            ]

            last_10_rows_blocks = [
                dp.DataTable(
                    df.tail(10).rename({col: self.spec.target_column}, axis=1),
                    caption="End",
                    label=col,
                )
                for col, df in self.full_data_dict.items()
            ]

            data_summary_blocks = [
                dp.DataTable(
                    df.rename({col: self.spec.target_column}, axis=1).describe(),
                    caption="Summary Statistics",
                    label=col,
                )
                for col, df in self.full_data_dict.items()
            ]
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
                            dp.Select(blocks=first_10_rows_blocks)
                            if len(first_10_rows_blocks) > 1
                            else first_10_rows_blocks[0],
                            dp.Text("----"),
                            dp.Text("### Last 10 Rows of Data"),
                            dp.Select(blocks=last_10_rows_blocks)
                            if len(last_10_rows_blocks) > 1
                            else last_10_rows_blocks[0],
                            dp.Text("### Data Summary Statistics"),
                            dp.Select(blocks=data_summary_blocks)
                            if len(data_summary_blocks) > 1
                            else data_summary_blocks[0],
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

            test_metrics_sections = []
            if self.test_eval_metrics is not None and not self.test_eval_metrics.empty:
                sec7_text = dp.Text(f"## Test Data Evaluation Metrics")
                sec7 = dp.DataTable(self.test_eval_metrics)
                test_metrics_sections = test_metrics_sections + [sec7_text, sec7]

            if summary_metrics is not None and not summary_metrics.empty:
                sec8_text = dp.Text(f"## Test Data Summary Metrics")
                sec8 = dp.DataTable(summary_metrics)
                test_metrics_sections = test_metrics_sections + [sec8_text, sec8]

            train_metrics_sections = []
            if self.eval_metrics is not None and not self.eval_metrics.empty:
                sec9_text = dp.Text(f"## Training Data Metrics")
                sec9 = dp.DataTable(self.eval_metrics)
                train_metrics_sections = [sec9_text, sec9]

            forecast_text = dp.Text(f"## Forecasted Data Overlaying Historical")
            forecast_sec = utils.get_forecast_plots(
                self.data,
                self.outputs,
                self.target_columns,
                test_data=test_data,
                forecast_col_name=self.forecast_col_name,
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
                + test_metrics_sections
                + train_metrics_sections
                + [yaml_appendix_title, yaml_appendix]
            )

        # save the report and result CSV
        self._save_report(
            report_sections=report_sections,
            result_df=result_df,
            metrics_df=self.eval_metrics,
            test_metrics_df=self.test_eval_metrics,
        )

    def _load_data(self):
        """Loads forecasting input data."""
        raw_data = utils._load_data(
            filename=self.spec.historical_data.url,
            format=self.spec.historical_data.format,
            columns=self.spec.historical_data.columns,
        )
        self.original_user_data = raw_data.copy()
        date_column = self.spec.datetime_column.name
        try:
            self.spec.freq = utils.get_frequency_of_datetime(raw_data, self.spec)
        except TypeError as e:
            logger.warn(
                f"Error determining frequency: {e.args}. Setting Frequency to None"
            )
            logger.debug(f"Full traceback: {e}")
            self.spec.freq = None
        data = Transformations(raw_data, self.spec).run()
        self.original_total_data = data

        additional_data = None
        if self.spec.additional_data is not None:
            additional_data = utils._load_data(
                filename=self.spec.additional_data.url,
                format=self.spec.additional_data.format,
                columns=self.spec.additional_data.columns,
            )
            additional_data = Transformations(
                raw_data, self.spec
            )._sort_by_datetime_col(additional_data)
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
            horizon=self.spec.horizon,
            target_category_columns=self.spec.target_category_columns,
            additional_data=additional_data,
        )

    def _test_evaluate_metrics(
        self, target_columns, test_filename, outputs, target_col="yhat", elapsed_time=0
    ):
        total_metrics = pd.DataFrame()
        summary_metrics = pd.DataFrame()
        data = None
        try:
            storage_options = (
                default_signer()
                if ObjectStorageDetails.is_oci_path(test_filename)
                else {}
            )
            data = utils._load_data(
                filename=test_filename,
                format=self.spec.test_data.format,
                storage_options=storage_options,
                columns=self.spec.test_data.columns,
            )
        except pd.errors.EmptyDataError:
            logger.warn("Empty testdata file")
            return total_metrics, summary_metrics, None

        if data.empty:
            return total_metrics, summary_metrics, None

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
            # Only columns present in test file will be used to generate test error
            if col in data:
                # Assuming that predictions have all forecast values
                dates = outputs[idx]["ds"]
                # Filling zeros for any date missing in test data to maintain consistency in metric calculation as in all other missing values cases it comes as 0
                y_true = [
                    data.loc[data["ds"] == date, col].values[0]
                    if date in data["ds"].values
                    else 0
                    for date in dates
                ]
                y_pred = np.asarray(outputs[idx][target_col][-len(y_true) :])

                metrics_df = utils._build_metrics_df(
                    y_true=y_true, y_pred=y_pred, column_name=col
                )
                total_metrics = pd.concat([total_metrics, metrics_df], axis=1)
            else:
                logger.warn(f"{col} is not there in test file")

        if not total_metrics.empty:
            summary_metrics = pd.DataFrame(
                {
                    SupportedMetrics.MEAN_SMAPE: np.mean(
                        total_metrics.loc[SupportedMetrics.SMAPE]
                    ),
                    SupportedMetrics.MEDIAN_SMAPE: np.median(
                        total_metrics.loc[SupportedMetrics.SMAPE]
                    ),
                    SupportedMetrics.MEAN_MAPE: np.mean(
                        total_metrics.loc[SupportedMetrics.MAPE]
                    ),
                    SupportedMetrics.MEDIAN_MAPE: np.median(
                        total_metrics.loc[SupportedMetrics.MAPE]
                    ),
                    SupportedMetrics.MEAN_RMSE: np.mean(
                        total_metrics.loc[SupportedMetrics.RMSE]
                    ),
                    SupportedMetrics.MEDIAN_RMSE: np.median(
                        total_metrics.loc[SupportedMetrics.RMSE]
                    ),
                    SupportedMetrics.MEAN_R2: np.mean(
                        total_metrics.loc[SupportedMetrics.R2]
                    ),
                    SupportedMetrics.MEDIAN_R2: np.median(
                        total_metrics.loc[SupportedMetrics.R2]
                    ),
                    SupportedMetrics.MEAN_EXPLAINED_VARIANCE: np.mean(
                        total_metrics.loc[SupportedMetrics.EXPLAINED_VARIANCE]
                    ),
                    SupportedMetrics.MEDIAN_EXPLAINED_VARIANCE: np.median(
                        total_metrics.loc[SupportedMetrics.EXPLAINED_VARIANCE]
                    ),
                    SupportedMetrics.ELAPSED_TIME: elapsed_time,
                },
                index=["All Targets"],
            )

            """Calculates Mean sMAPE, Median sMAPE, Mean MAPE, Median MAPE, Mean wMAPE, Median wMAPE values for each horizon
            if horizon <= 10."""
            target_columns_in_output = set(target_columns).intersection(data.columns)
            if self.spec.horizon <= SUMMARY_METRICS_HORIZON_LIMIT and len(
                outputs
            ) == len(target_columns_in_output):
                metrics_per_horizon = utils._build_metrics_per_horizon(
                    data=data,
                    outputs=outputs,
                    target_columns=target_columns,
                    target_col=target_col,
                    horizon_periods=self.spec.horizon,
                )

                summary_metrics = summary_metrics.append(metrics_per_horizon)

                new_column_order = [
                    SupportedMetrics.MEAN_SMAPE,
                    SupportedMetrics.MEDIAN_SMAPE,
                    SupportedMetrics.MEAN_MAPE,
                    SupportedMetrics.MEDIAN_MAPE,
                    SupportedMetrics.MEAN_WMAPE,
                    SupportedMetrics.MEDIAN_WMAPE,
                    SupportedMetrics.MEAN_RMSE,
                    SupportedMetrics.MEDIAN_RMSE,
                    SupportedMetrics.MEAN_R2,
                    SupportedMetrics.MEDIAN_R2,
                    SupportedMetrics.MEAN_EXPLAINED_VARIANCE,
                    SupportedMetrics.MEDIAN_EXPLAINED_VARIANCE,
                    SupportedMetrics.ELAPSED_TIME,
                ]
                summary_metrics = summary_metrics[new_column_order]

        return total_metrics, summary_metrics, data

    def _save_report(
        self,
        report_sections: Tuple,
        result_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        test_metrics_df: pd.DataFrame,
    ):
        """Saves resulting reports to the given folder."""
        import datapane as dp

        if self.spec.output_directory:
            output_dir = self.spec.output_directory.url
        else:
            output_dir = "tmp_fc_operator_result"
            logger.warn(
                "Since the output directory was not specified, the output will be saved to {} directory.".format(
                    output_dir
                )
            )

        if ObjectStorageDetails.is_oci_path(output_dir):
            storage_options = default_signer()
        else:
            storage_options = dict()

        # datapane html report
        if self.spec.generate_report:
            # datapane html report
            with tempfile.TemporaryDirectory() as temp_dir:
                report_local_path = os.path.join(temp_dir, "___report.html")
                dp.save_report(report_sections, report_local_path)

                report_path = os.path.join(output_dir, self.spec.report_filename)
                with open(report_local_path) as f1:
                    with fsspec.open(
                        report_path,
                        "w",
                        **storage_options,
                    ) as f2:
                        f2.write(f1.read())

        # forecast csv report
        utils._write_data(
            data=result_df,
            filename=os.path.join(output_dir, self.spec.forecast_filename),
            format="csv",
            storage_options=storage_options,
        )

        # metrics csv report
        if self.spec.generate_metrics:
            if metrics_df is not None:
                utils._write_data(
                    data=metrics_df.rename_axis("metrics").reset_index(),
                    filename=os.path.join(output_dir, self.spec.metrics_filename),
                    format="csv",
                    storage_options=storage_options,
                    index=False,
                )
            else:
                logger.warn(
                    f"Attempted to generated {self.spec.metrics_filename} file with the training metrics, however the training metrics could not be properly generated."
                )

            # test_metrics csv report
            if self.spec.generate_metrics and test_metrics_df is not None:
                utils._write_data(
                    data=test_metrics_df.rename_axis("metrics").reset_index(),
                    filename=os.path.join(output_dir, self.spec.test_metrics_filename),
                    format="csv",
                    storage_options=storage_options,
                    index=False,
                )
            else:
                logger.warn(
                    f"Attempted to generated {self.spec.test_metrics_filename} file with the test metrics, however the test metrics could not be properly generated."
                )

        logger.warn(
            f"The outputs have been successfully "
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

    @abstractmethod
    def explain_model(self) -> dict:
        """
        explain model using global & local explanations
        """
        raise NotImplementedError()
