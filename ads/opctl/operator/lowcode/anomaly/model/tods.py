#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib

import numpy as np
import pandas as pd

from ads.common.decorator.runtime_dependency import runtime_dependency

from ..const import TODS_IMPORT_MODEL_MAP, TODS_MODEL_MAP, TODSSubModels
from .base_model import AnomalyOperatorBaseModel


class TODSOperatorModel(AnomalyOperatorBaseModel):
    """Class representing TODS Anomaly Detection operator model."""

    @runtime_dependency(
        module="tods",
        err_msg=(
            "Please run `pip3 install tods` to "
            "install the required dependencies for TODS."
        ),
    )
    def _build_model(self) -> pd.DataFrame:
        tods_module = importlib.import_module(
            name=TODS_IMPORT_MODEL_MAP.get(
                self.spec.model_kwargs.get("sub_model", "ocsvm")
            ),
            package="tods.sk_interface.detection_algorithm",
        )

        model_kwargs = self.spec.model_kwargs
        model_kwargs.pop("sub_model", None)
        sub_model = self.spec.model_kwargs.get("sub_model", "ocsvm")

        self.datasets.full_data_dict = dict(
            tuple(self.datasets.data.groupby(self.spec.target_column))
        )

        models = {}
        predictions_train = {}
        prediction_score_train = {}
        predictions_test = {}
        prediction_score_test = {}
        for target, df in self.datasets.full_data_dict.items():
            model = getattr(tods_module, TODS_MODEL_MAP.get(sub_model))(**model_kwargs)

            model.fit(np.array(df[self.spec.target_category_columns]).reshape(-1, 1))
            predictions_train[target] = model.predict(
                np.array(df[self.spec.target_category_columns]).reshape(-1, 1)
            )
            prediction_score_train[target] = model.predict_score(
                np.array(df[self.spec.target_category_columns]).reshape(-1, 1)
            )
            model[target] = model

        # result = pd.DataFrame()
        return model, predictions_train, prediction_score_test

    def _generate_report(self):
        import datapane as dp

        """The method that needs to be implemented on the particular model level."""
        selected_models_text = dp.Text(
            f"## Selected Models Overview \n "
            "The following tables provide information regarding the chosen model."
        )
        all_sections = [selected_models_text]

        model_description = dp.Text(
            "The automlx model automatically pre-processes, selects and engineers "
            "high-quality features in your dataset, which then given to an automatically "
            "chosen and optimized machine learning model.."
        )
        other_sections = all_sections

        return (
            model_description,
            other_sections,
        )
