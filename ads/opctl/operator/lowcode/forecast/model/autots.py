import pandas as pd
import numpy as np

from ads.opctl import logger
from ads.opctl.operator.lowcode.forecast import utils

from .. import utils
from .base_model import ForecastOperatorBaseModel
from ads.common.decorator.runtime_dependency import runtime_dependency


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

        # Initialize the AutoTS model with specified parameters
        model = AutoTS(
            forecast_length=self.spec.horizon.periods,
            frequency="infer",
            prediction_interval=self.spec.confidence_interval_width,
            max_generations=10,
            no_negatives=False,
            constraint=None,
            ensemble=self.spec.model_kwargs.get("ensemble", "auto"),
            initial_template="General+Random",
            random_seed=2022,
            holiday_country="US",
            subset=None,
            aggfunc="first",
            na_tolerance=1,
            drop_most_recent=0,
            drop_data_older_than_periods=None,
            model_list="multivariate",
            transformer_list="auto",
            transformer_max_depth=6,
            models_mode="random",
            num_validations="auto",
            models_to_validate=0.15,
            max_per_model_class=None,
            validation_method="backwards",
            min_allowed_train_percent=0.5,
            remove_leading_zeroes=False,
            prefill_na=None,
            introduce_na=None,
            preclean=None,
            model_interrupt=True,
            generation_timeout=None,
            current_model_file=None,
            verbose=1,
            n_jobs=-1,
        )

        # Prepare the data for model training
        temp_list = [self.full_data_dict[i] for i in self.full_data_dict.keys()]
        melt_temp = [
            temp_list[i].melt(
                temp_list[i].columns.difference(self.target_columns),
                var_name="series_id",
                value_name=self.original_target_column,
            )
            for i in range(len(self.target_columns))
        ]
        full_data_long = pd.concat(melt_temp)

        # Fit the model to the training data
        model = model.fit(
            full_data_long,
            date_col=self.spec.datetime_column.name,
            value_col=self.original_target_column,
            id_col="series_id",
        )

        # Store the trained model and generate forecasts
        self.models = model
        logger.info("===========Forecast Generated===========")
        self.prediction = model.predict()
        outputs

        # Process the forecasts for each target series
        for series_idx, series in enumerate(self.target_columns):
            # Create a dictionary to store the forecasts for each series
            outputs[series] = (
                pd.concat(
                    [
                        self.prediction.forecast.reset_index()[
                            ["index", self.target_columns[series_idx]]
                        ].rename(
                            columns={
                                "index": self.spec.datetime_column.name,
                                self.target_columns[series_idx]: "yhat",
                            }
                        ),
                        self.prediction.lower_forecast.reset_index()[
                            ["index", self.target_columns[series_idx]]
                        ].rename(
                            columns={
                                "index": self.spec.datetime_column.name,
                                self.target_columns[series_idx]: "yhat_lower",
                            }
                        ),
                        self.prediction.upper_forecast.reset_index()[
                            ["index", self.target_columns[series_idx]]
                        ].rename(
                            columns={
                                "index": self.spec.datetime_column.name,
                                self.target_columns[series_idx]: "yhat_upper",
                            }
                        ),
                    ],
                    axis=1,
                )
                .T.drop_duplicates()
                .T
            )

        # Store the processed forecasts in a list
        self.outputs = [fc for fc in outputs.values()]

        # Re-merge historical datas for processing
        data_merged = pd.concat(
            [
                v[v[k].notna()].set_index(date_column)
                for k, v in self.full_data_dict.items()
            ],
            axis=1,
        ).reset_index()
        self.data = data_merged

        outputs_merged = pd.DataFrame()

        col = self.original_target_column
        output_col = pd.DataFrame()
        yhat_lower_percentage = (100 - self.spec.confidence_interval_width * 100) // 2
        yhat_upper_name = "p" + str(int(100 - yhat_lower_percentage))
        yhat_lower_name = "p" + str(int(yhat_lower_percentage))

        for cat in self.categories:
            output_i = pd.DataFrame()

            output_i["Date"] = outputs[f"{col}_{cat}"][self.spec.datetime_column.name]
            output_i["Series"] = cat
            output_i[f"forecast_value"] = outputs[f"{col}_{cat}"]["yhat"]
            output_i[yhat_upper_name] = outputs[f"{col}_{cat}"]["yhat_upper"]
            output_i[yhat_lower_name] = outputs[f"{col}_{cat}"]["yhat_lower"]
            output_col = pd.concat([output_col, output_i])
        output_col = output_col.reset_index(drop=True)
        outputs_merged = pd.concat([outputs_merged, output_col], axis=1)

        logger.info("===========Done===========")

        return outputs_merged

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
        # TODO: Format the parameters better for display in report.
        sec2 = dp.Select(
            blocks=[
                dp.HTML(
                    pd.DataFrame(
                        [self.models.best_model_params["models"][x]["ModelParameters"]]
                    ).to_html(),
                    label=self.original_target_column + "_model_" +str(i),
                )
                for i, x in enumerate(
                    list(self.models.best_model_params["models"].keys())
                )
            ]
        )
        all_sections = [sec1_text, sec_1, sec2_text, sec2]

        # Model Description
        model_description = dp.Text(
            "AutoTS is a time series package for Python designed for rapidly deploying high-accuracy forecasts at scale."
            "In 2023, AutoTS has won in the M6 forecasting competition,"
            "delivering the highest performance investment decisions across 12 months of stock market forecasting."
        )

        other_sections = all_sections
        forecast_col_name = "yhat"
        train_metrics = False

        ds_column_series = pd.to_datetime(self.data[self.spec.datetime_column.name])
        ds_forecast_col = self.outputs[0].index
        ci_col_names = ["yhat_lower", "yhat_upper"]

        return (
            model_description,
            other_sections,
            forecast_col_name,
            train_metrics,
            ds_column_series,
            ds_forecast_col,
            ci_col_names,
        )
