#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import traceback
import pandas as pd
import numpy as np
import yaml

from ads.opctl import logger
from ads.opctl.operator.lowcode.forecast.utils import convert_target
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
from ads.common.decorator.runtime_dependency import runtime_dependency
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..const import ForecastOutputColumns, SupportedModels
from ads.opctl.operator.lowcode.forecast.utils import _select_plot_list

AUTOTS_MAX_GENERATION = 10
AUTOTS_MODELS_TO_VALIDATE = 0.15


class AutoTSOperatorModel(ForecastOperatorBaseModel):
    """Class representing AutoTS operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config, datasets)
        self.global_explanation = {}
        self.local_explanation = {}

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
        from autots import AutoTS, create_regressor

        self.outputs = None
        models = dict()
        # Get the name of the datetime column
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.spec.datetime_column.name,
        )
        model = self.loaded_models if self.loaded_models is not None else None
        if model is None:
            # Initialize the AutoTS model with specified parameters
            model = AutoTS(
                forecast_length=self.spec.horizon,
                frequency=self.spec.model_kwargs.get(
                    "frequency", "infer"
                ),  # TODO: Use datasets.get_datetime_frequency ?
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
                max_per_model_class=self.spec.model_kwargs.get(
                    "max_per_model_class", None
                ),
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
                generation_timeout=self.spec.model_kwargs.get(
                    "generation_timeout", None
                ),
                current_model_file=self.spec.model_kwargs.get(
                    "current_model_file", None
                ),
                verbose=-1 if logger.level > 40 else 1,
                n_jobs=self.spec.model_kwargs.get("n_jobs", -1),
            )

        self.full_data_long = self.datasets.get_all_data_long()

        if self.spec.additional_data:
            df_temp = (
                self.full_data_long.set_index([self.spec.target_column])
                .reset_index(drop=True)
                .copy()
            )
            df_temp[self.spec.datetime_column.name] = pd.to_datetime(
                df_temp[self.spec.datetime_column.name]
            )
            r_tr, _ = create_regressor(
                df_temp.pivot(
                    [self.spec.datetime_column.name],
                    columns=ForecastOutputColumns.SERIES,
                    values=self.datasets.get_additional_data_column_names(),
                ),
                forecast_length=self.spec.horizon,
            )

            self.future_regressor_train = r_tr.copy()

        if self.loaded_models is None:
            # Fit the model to the training data
            model = model.fit(
                self.full_data_long.groupby(ForecastOutputColumns.SERIES)
                .head(-self.spec.horizon)
                .reset_index(drop=True),
                date_col=self.spec.datetime_column.name,
                value_col=self.original_target_column,
                future_regressor=r_tr.head(-self.spec.horizon)
                if self.spec.additional_data
                else None,
                id_col=ForecastOutputColumns.SERIES,
            )

            # Store the trained model and generate forecasts
            self.models = copy.deepcopy(model)
        else:
            self.models = self.loaded_models
        logger.debug("===========Forecast Generated===========")
        self.outputs = model.predict(
            future_regressor=r_tr.tail(self.spec.horizon)
            if self.spec.additional_data
            else None
        )

        hist_df = model.back_forecast().forecast

        params = vars(model).copy()
        for param in [
            "ens_copy",
            "df_wide_numeric",
            "future_regressor_train",
            "initial_results",
            "score_per_series",
            "validation_results",
            "validation_train_indexes",
            "validation_test_indexes",
            "validation_indexes",
            "best_model",
        ]:
            if param in params:
                params.pop(param)

        for s_id in self.datasets.list_series_ids():
            self.forecast_output.init_series_output(
                series_id=s_id, data_at_series=self.datasets.get_data_at_series(s_id)
            )

            self.forecast_output.populate_series_output(
                series_id=s_id,
                fit_val=hist_df[s_id].values,
                forecast_val=self.outputs.forecast[s_id].values,
                upper_bound=self.outputs.upper_forecast[s_id].values,
                lower_bound=self.outputs.lower_forecast[s_id].values,
            )

            self.model_parameters[s_id] = {
                "framework": SupportedModels.AutoTS,
                **params,
            }

        logger.debug("===========Done===========")

        return self.forecast_output.get_forecast_long()

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
        sec_1 = _select_plot_list(
            lambda s_id: self.outputs.plot(self.models.df_wide_numeric, series=s_id),
            self.datasets.list_series_ids(),
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

        if self.spec.generate_explanations:
            # If the key is present, call the "explain_model" method
            try:
                self.explain_model()

                # Create a markdown text block for the global explanation section
                global_explanation_text = dp.Text(
                    f"## Global Explanation of Models \n "
                    "The following tables provide the feature attribution for the global explainability."
                )

                # Convert the global explanation data to a DataFrame
                global_explanation_df = pd.DataFrame(self.global_explanation).drop(
                    index=[ForecastOutputColumns.SERIES, self.spec.target_column]
                )

                self.formatted_global_explanation = (
                    global_explanation_df / global_explanation_df.sum(axis=0) * 100
                )

                # Create a markdown section for the global explainability
                global_explanation_section = dp.Blocks(
                    "### Global Explainability ",
                    dp.DataTable(self.formatted_global_explanation),
                )

                aggregate_local_explanations = pd.DataFrame()
                for s_id, local_ex_df in self.local_explanation.items():
                    local_ex_df_copy = local_ex_df.copy()
                    local_ex_df_copy["Series"] = s_id
                    aggregate_local_explanations = pd.concat(
                        [aggregate_local_explanations, local_ex_df_copy], axis=0
                    )
                self.formatted_local_explanation = aggregate_local_explanations

                local_explanation_text = dp.Text(f"## Local Explanation of Models \n ")
                blocks = [
                    dp.DataTable(
                        local_ex_df.div(local_ex_df.abs().sum(axis=1), axis=0) * 100,
                        label=s_id,
                    )
                    for s_id, local_ex_df in self.local_explanation.items()
                ]
                local_explanation_section = (
                    dp.Select(blocks=blocks) if len(blocks) > 1 else blocks[0]
                )

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_text,
                    global_explanation_section,
                    local_explanation_text,
                    local_explanation_section,
                ]
            except Exception as e:
                logger.warn(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")
                raise e

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

    def _custom_predict_autots(self, data):
        raise NotImplementedError("Autots does not yet support explanations.")

    def get_explain_predict_fn(self, series_id):
        selected_model = self.models

        def _custom_predict_fn(
            data,
            model=selected_model,
            dt_column_name=self.datasets._datetime_column_name,
        ):
            """
            data: ForecastDatasets.get_data_at_series(s_id)
            """
            data[dt_column_name] = pd.to_datetime(data[dt_column_name], unit="s")
            return model.predict(data)["yhat"]

        return _custom_predict_fn

    def generate_train_metrics(self) -> pd.DataFrame:
        """
        Generate Training Metrics when fitted data is not available.
        The method that needs to be implemented on the particular model level.

        metrics	Store 1
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
        df = pd.concat([mapes, scores])
        new_column_names = {
            old_name: convert_target(old_name, self.original_target_column)
            for old_name in df.columns
        }
        return df.rename(columns=new_column_names)

    def local_explainer(self, kernel_explainer, series_id, datetime_col_name) -> None:
        super().local_explainer(kernel_explainer, series_id, datetime_col_name)

        # set the index of the DataFrame to the datetime column
        # local_kernel_explnr_df.index = data_horizon.index
        local_kernel_explnr_df = self.local_explanation[series_id]
        local_kernel_explnr_df.drop(
            ["series_id", self.spec.target_column], axis=1, inplace=True
        )
        self.local_explanation[series_id] = local_kernel_explnr_df
