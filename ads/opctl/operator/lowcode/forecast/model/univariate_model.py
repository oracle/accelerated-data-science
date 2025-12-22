from abc import ABC

from ads.opctl.operator.lowcode.forecast.model.base_model import ForecastOperatorBaseModel
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig
from .forecast_datasets import ForecastDatasets
import pandas as pd
import numpy as np


class UnivariateForecasterOperatorModel(ForecastOperatorBaseModel, ABC):

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)

    def explain_model(self):
        """
        Explanation logic for univariate model which do not depend on exogenous variables.:
        - Target gets full weight
        - All exogenous features get zero weight
        """

        self.local_explanation = {}
        global_expl = []
        self.explanations_info = {}

        for series_id, sm in self.models.items():
            model = sm["model"]
            if hasattr(model, "_y"):
                idx = model._y.index
            else:
                idx = self.full_data_dict[series_id].index

            df_orig = self.full_data_dict[series_id]
            dt_col = self.spec.datetime_column.name
            target_col = self.original_target_column

            exog_cols = [c for c in df_orig.columns if c not in {dt_col, target_col}]

            expl_df = pd.DataFrame(index=idx)
            expl_df[target_col] = 1.0
            for col in exog_cols:
                expl_df[col] = 0.0

            self.explanations_info[series_id] = expl_df

            local_df = self.get_horizon(expl_df)
            local_df["Series"] = series_id
            local_df.index.rename(self.dt_column_name, inplace=True)
            self.local_explanation[series_id] = local_df

            g_expl = self.drop_horizon(expl_df).mean()
            g_expl.name = series_id
            global_expl.append(np.abs(g_expl))

        self.global_explanation = pd.concat(global_expl, axis=1)
        self.formatted_global_explanation = (
                self.global_explanation
                / self.global_explanation.sum(axis=0)
                * 100
        )

        self.formatted_local_explanation = pd.concat(
            self.local_explanation.values()
        )
