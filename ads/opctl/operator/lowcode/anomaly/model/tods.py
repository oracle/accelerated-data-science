# #!/usr/bin/env python
# # -*- coding: utf-8 -*--

# # Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# # Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

# import importlib

# import numpy as np
# import pandas as pd

# from ads.common.decorator.runtime_dependency import runtime_dependency
# from .anomaly_dataset import AnomalyOutput

# from ..const import (
#     TODS_IMPORT_MODEL_MAP,
#     TODS_MODEL_MAP,
#     OutputColumns,
#     TODS_DEFAULT_MODEL,
# )
# from .base_model import AnomalyOperatorBaseModel


# class TODSOperatorModel(AnomalyOperatorBaseModel):
#     """Class representing TODS Anomaly Detection operator model."""

#     @runtime_dependency(
#         module="tods",
#         err_msg=(
#             "Please run `pip3 install tods` to "
#             "install the required dependencies for TODS."
#         ),
#     )
#     def _build_model(self) -> pd.DataFrame:
#         """
#         Build the TODS model.

#         Returns
#         -------
#             Tuple: model, predictions_train, and prediction_score_test
#         """
#         # Import the TODS module
#         tods_module = importlib.import_module(
#             name=TODS_IMPORT_MODEL_MAP.get(
#                 self.spec.model_kwargs.get("sub_model", TODS_DEFAULT_MODEL)
#             ),
#             package="tods.sk_interface.detection_algorithm",
#         )

#         # Get the model kwargs
#         model_kwargs = self.spec.model_kwargs
#         sub_model = self.spec.model_kwargs.get("sub_model", TODS_DEFAULT_MODEL)
#         model_kwargs.pop("sub_model", None)

#         # Initialize variables
#         models = {}
#         predictions_train = {}
#         prediction_score_train = {}
#         predictions_test = {}
#         prediction_score_test = {}
#         date_column = self.spec.datetime_column.name
#         anomaly_output = AnomalyOutput(date_column=date_column)

#         # Iterate over the full_data_dict items
#         for target, df in self.datasets.full_data_dict.items():
#             # Instantiate the model
#             model = getattr(tods_module, TODS_MODEL_MAP.get(sub_model))(**model_kwargs)

#             # Fit the model
#             model.fit(np.array(df[self.spec.target_column]).reshape(-1, 1))

#             # Make predictions
#             predictions_train[target] = model.predict(
#                 np.array(df[self.spec.target_column]).reshape(-1, 1)
#             )
#             prediction_score_train[target] = model.predict_score(
#                 np.array(df[self.spec.target_column]).reshape(-1, 1)
#             )

#             # Store the model and predictions in dictionaries
#             models[target] = model

#             anomaly = pd.DataFrame(
#                 {
#                     date_column: df[date_column],
#                     OutputColumns.ANOMALY_COL: predictions_train[target],
#                 }
#             )
#             score = pd.DataFrame(
#                 {
#                     date_column: df[date_column],
#                     OutputColumns.SCORE_COL: prediction_score_train[target],
#                 }
#             )
#             anomaly_output.add_output(target, anomaly, score)

#         return anomaly_output

#     def _generate_report(self):
#         import report_creator as rc

#         """The method that needs to be implemented on the particular model level."""
#         selected_models_text = rc.Text(
#             f"## Selected Models Overview \n "
#             "The following tables provide information regarding the chosen model."
#         )
#         all_sections = [selected_models_text]

#         model_description = rc.Text(
#             "The tods model is a full-stack automated machine learning system for outlier detection "
#             "on univariate / multivariate time-series data. It provides exhaustive modules for building "
#             "machine learning-based outlier detection systems and wide range of algorithms."
#         )
#         other_sections = all_sections

#         return (
#             model_description,
#             other_sections,
#         )
