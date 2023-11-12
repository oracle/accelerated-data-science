#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import numpy as np
import yaml

from ads.opctl import logger
from ads.opctl.operator.lowcode.forecast import utils

from .. import utils
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
from ads.common.decorator.runtime_dependency import runtime_dependency
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..const import ForecastOutputColumns


AUTOTS_MAX_GENERATION = 10
AUTOTS_MODELS_TO_VALIDATE = 0.15


class AutoTSOperatorModel(ForecastOperatorBaseModel):
    """Class representing AutoTS operator model."""

    @runtime_dependency(
        module="autots",
        err_msg="Please run `pip3 install autots` to install the required dependencies for autots.",
    )
    def _build_model(self) -> pd.DataFrame:
        """Builds the AutoTS model and generates forecasts.

        Returns:
            pd.DataFrame: AutoTS model forecast dataframe
        """

        # Import necessary libraries
        from autots import AutoTS

        models = dict()
        outputs = dict()
        outputs_legacy = []
        # Get the name of the datetime column
        date_column = self.spec.datetime_column.name
        self.datasets.datetime_col = date_column
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width
        )

        # Initialize the AutoTS model with specified parameters
        model = AutoTS(
            forecast_length=self.spec.horizon,
            frequency=self.spec.model_kwargs.get("frequency", "infer"),
            prediction_interval=self.spec.confidence_interval_width,
            max_generations=self.spec.model_kwargs.get(
                "max_generations", AUTOTS_MAX_GENERATION
            ),
            no_negatives=self.spec.model_kwargs.get("no_negatives", False),
            constraint=self.spec.model_kwargs.get("constraint", None),
            ensemble=self.spec.model_kwargs.get("ensemble", "auto"),
            initial_template=self.spec.model_kwargs.get(
                "initial_template", "General+Random"
            ),
            random_seed=self.spec.model_kwargs.get("random_seed", 2022),
            holiday_country=self.spec.model_kwargs.get("holiday_country", "US"),
            subset=self.spec.model_kwargs.get("subset", None),
            aggfunc=self.spec.model_kwargs.get("aggfunc", "first"),
            na_tolerance=self.spec.model_kwargs.get("na_tolerance", 1),
            drop_most_recent=self.spec.model_kwargs.get("drop_most_recent", 0),
            drop_data_older_than_periods=self.spec.model_kwargs.get(
                "drop_data_older_than_periods", None
            ),
            model_list=self.spec.model_kwargs.get("model_list", "fast_parallel"),
            transformer_list=self.spec.model_kwargs.get("transformer_list", "auto"),
            transformer_max_depth=self.spec.model_kwargs.get(
                "transformer_max_depth", 6
            ),
            models_mode=self.spec.model_kwargs.get("models_mode", "random"),
            num_validations=self.spec.model_kwargs.get("num_validations", "auto"),
            models_to_validate=self.spec.model_kwargs.get(
                "models_to_validate", AUTOTS_MODELS_TO_VALIDATE
            ),
            max_per_model_class=self.spec.model_kwargs.get("max_per_model_class", None),
            validation_method=self.spec.model_kwargs.get(
                "validation_method", "backwards"
            ),
            min_allowed_train_percent=self.spec.model_kwargs.get(
                "min_allowed_train_percent", 0.5
            ),
            remove_leading_zeroes=self.spec.model_kwargs.get(
                "remove_leading_zeroes", False
            ),
            prefill_na=self.spec.model_kwargs.get("prefill_na", None),
            introduce_na=self.spec.model_kwargs.get("introduce_na", None),
            preclean=self.spec.model_kwargs.get("preclean", None),
            model_interrupt=self.spec.model_kwargs.get("model_interrupt", True),
            generation_timeout=self.spec.model_kwargs.get("generation_timeout", None),
            current_model_file=self.spec.model_kwargs.get("current_model_file", None),
            verbose=self.spec.model_kwargs.get("verbose", 1),
            n_jobs=self.spec.model_kwargs.get("n_jobs", -1),
        )

        # Prepare the data for model training
        full_data_dict = self.datasets.full_data_dict
        temp_list = [full_data_dict[i] for i in full_data_dict.keys()]
        melt_temp = [
            temp_list[i].melt(
                temp_list[i].columns.difference(self.target_columns),
                var_name="series_id",
                value_name=self.original_target_column,
            )
            for i in range(len(self.target_columns))
        ]
        full_data_long = pd.concat(melt_temp)
        full_data_long[self.spec.datetime_column.name] = pd.to_datetime(
            full_data_long[self.spec.datetime_column.name],
            format=self.spec.datetime_column.format,
        )

        # Fit the model to the training data
        model = model.fit(
            full_data_long,
            date_col=self.spec.datetime_column.name,
            value_col=self.original_target_column,
            id_col="series_id",
        )

        # Store the trained model and generate forecasts
        self.models = model
        logger.debug("===========Forecast Generated===========")
        self.prediction = model.predict()
        outputs = dict()

        output_col = pd.DataFrame()
        yhat_upper_name = ForecastOutputColumns.UPPER_BOUND
        yhat_lower_name = ForecastOutputColumns.LOWER_BOUND

        for cat in self.categories:
            output_i = pd.DataFrame()
            cat_target = f"{self.original_target_column}_{cat}"
            input_data_i = full_data_dict[cat_target]

            output_i["Date"] = pd.to_datetime(
                input_data_i[self.spec.datetime_column.name],
                format=self.spec.datetime_column.format,
            )
            output_i["Series"] = cat
            output_i["input_value"] = input_data_i[cat_target]
            output_i["fitted_value"] = float("nan")
            output_i = output_i.set_index("Date")

            output_i["forecast_value"] = self.prediction.forecast[[cat_target]]
            output_i[yhat_upper_name] = self.prediction.upper_forecast[[cat_target]]
            output_i[yhat_lower_name] = self.prediction.lower_forecast[[cat_target]]

            output_i = output_i.reset_index()
            output_col = pd.concat([output_col, output_i])
            self.forecast_output.add_category(
                category=cat, target_category_column=cat_target, forecast=output_i
            )

        output_col = output_col.reset_index(drop=True)

        logger.debug("===========Done===========")

        return output_col

    def _generate_report(self) -> tuple:
        """
        Generates the report for the given function.

        Returns:
            tuple: A tuple containing the following elements:
            - model_description (dp.Text): A text object containing the description of the AutoTS model.
            - other_sections (list): A list of sections to be included in the report.
            - forecast_col_name (str): The name of the forecast column.
            - train_metrics (bool): A boolean indicating whether to include train metrics.
            - ds_column_series (pd.Series): A pandas Series containing the datetime column values.
            - ds_forecast_col (pd.Index): A pandas Index containing the forecast column values.
            - ci_col_names (list): A list of column names for confidence intervals.
        """
        import datapane as dp

        # Section 1: Forecast Overview
        sec1_text = dp.Text(
            "## Forecast Overview \n"
            "These plots show your forecast in the context of historical data."
        )
        sec_1 = utils._select_plot_list(
            lambda idx, *args: self.prediction.plot(
                self.models.df_wide_numeric,
                series=self.models.df_wide_numeric.columns[idx],
                start_date=self.models.df_wide_numeric.reset_index()[
                    self.spec.datetime_column.name
                ].min(),
            ),
            target_columns=self.target_columns,
        )

        # Section 2: AutoTS Model Parameters
        sec2_text = dp.Text(f"## AutoTS Model Parameters")
        try:
            sec2 = dp.Code(
                code=yaml.dump(list(self.models.best_model.T.to_dict().values())[0]),
                language="yaml",
            )

        except KeyError as ke:
            logger.warn(f"Issue generating Model Parameters Table Section. Skipping")
            sec2 = dp.Text(f"Error generating model parameters.")
        all_sections = [sec1_text, sec_1, sec2_text, sec2]

        # Model Description
        model_description = dp.Text(
            "AutoTS is a time series package for Python designed for rapidly deploying high-accuracy forecasts at scale. "
            "In 2023, AutoTS has won in the M6 forecasting competition, "
            "delivering the highest performance investment decisions across 12 months of stock market forecasting."
        )

        other_sections = all_sections

        return (
            model_description,
            other_sections,
        )

    def explain_model(self) -> dict:
        """
        explain model using global & local explanations
        """
        raise NotImplementedError()

    def _generate_train_metrics(self) -> pd.DataFrame:
        """
        Generate Training Metrics when fitted data is not available.
        The method that needs to be implemented on the particular model level.

        metrics	Sales_Store 1
        sMAPE	26.19
        MAPE	2.96E+18
        RMSE	2014.192531
        r2	-4.60E-06
        Explained Variance	0.002177087
        """
        mapes = pd.DataFrame(self.models.best_model_per_series_mape()).T
        scores = pd.DataFrame(
            self.models.best_model_per_series_score(), columns=["AutoTS Score"]
        ).T
        return pd.concat([mapes, scores])
