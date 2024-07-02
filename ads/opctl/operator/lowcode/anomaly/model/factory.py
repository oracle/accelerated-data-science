#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.anomaly.utils import select_auto_model

from ..const import NonTimeADSupportedModels, SupportedModels
from ..operator_config import AnomalyOperatorConfig
from .anomaly_dataset import AnomalyDatasets
from .automlx import AutoMLXOperatorModel
from .autots import AutoTSOperatorModel

# from .tods import TODSOperatorModel
from .base_model import AnomalyOperatorBaseModel
from .isolationforest import IsolationForestOperatorModel
from .oneclasssvm import OneClassSVMOperatorModel


class UnSupportedModelError(Exception):
    """Exception raised when the model is not supported.

    Attributes:
        operator_config (AnomalyOperatorConfig): The operator configuration.
        model_type (str): The type of the unsupported model.
    """

    def __init__(self, operator_config: AnomalyOperatorConfig, model_type: str):
        supported_models = (
            SupportedModels.values
            if operator_config.spec.datetime_column
            else NonTimeADSupportedModels.values
        )
        message = (
            f"Model: `{model_type}` is not supported. "
            f"Supported models: {supported_models}"
        )
        super().__init__(message)


class AnomalyOperatorModelFactory:
    """
    The factory class helps to instantiate proper model operator based on the model type.
    """

    _MAP = {
        SupportedModels.AutoMLX: AutoMLXOperatorModel,
        # SupportedModels.TODS: TODSOperatorModel,
        SupportedModels.AutoTS: AutoTSOperatorModel,
    }

    _NonTime_MAP = {
        NonTimeADSupportedModels.OneClassSVM: OneClassSVMOperatorModel,
        NonTimeADSupportedModels.IsolationForest: IsolationForestOperatorModel,
        # TODO: Add DBScan model for non time based anomaly
        # NonTimeADSupportedModels.DBScan: DBScanOperatorModel,
    }

    @classmethod
    def get_model(
        cls, operator_config: AnomalyOperatorConfig, datasets: AnomalyDatasets
    ) -> AnomalyOperatorBaseModel:
        """
        Gets the operator model based on the model type.

        Parameters
        ----------
        operator_config: AnomalyOperatorConfig
            The anomaly detection operator config.

        datasets: AnomalyDatasets
            Datasets for finding anomaly

        Returns
        -------
        AnomalyOperatorBaseModel
            The anomaly detection operator model.

        Raises
        ------
        UnSupportedModelError
            In case of not supported model.
        """
        model_type = operator_config.spec.model
        if model_type == "auto":
            model_type = select_auto_model(operator_config)

        model_map = (
            cls._MAP if operator_config.spec.datetime_column else cls._NonTime_MAP
        )

        if model_type not in model_map:
            raise UnSupportedModelError(operator_config, model_type)

        return model_map[model_type](config=operator_config, datasets=datasets)
