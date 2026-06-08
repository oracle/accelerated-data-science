#!/usr/bin/env python

# Copyright (c) 2021, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib

_LAZY_ATTRS = {
    "DataScienceModel": ("ads.model.datascience_model", "DataScienceModel"),
    "ModelDeployer": ("ads.model.deployment.model_deployer", "ModelDeployer"),
    "ModelDeployment": ("ads.model.deployment.model_deployment", "ModelDeployment"),
    "ModelDeploymentProperties": (
        "ads.model.deployment.model_deployment_properties",
        "ModelDeploymentProperties",
    ),
    "AutoMLModel": ("ads.model.framework.automl_model", "AutoMLModel"),
    "EmbeddingONNXModel": (
        "ads.model.framework.embedding_onnx_model",
        "EmbeddingONNXModel",
    ),
    "HuggingFacePipelineModel": (
        "ads.model.framework.huggingface_model",
        "HuggingFacePipelineModel",
    ),
    "LightGBMModel": ("ads.model.framework.lightgbm_model", "LightGBMModel"),
    "PyTorchModel": ("ads.model.framework.pytorch_model", "PyTorchModel"),
    "SklearnModel": ("ads.model.framework.sklearn_model", "SklearnModel"),
    "SparkPipelineModel": ("ads.model.framework.spark_model", "SparkPipelineModel"),
    "TensorFlowModel": ("ads.model.framework.tensorflow_model", "TensorFlowModel"),
    "XGBoostModel": ("ads.model.framework.xgboost_model", "XGBoostModel"),
    "GenericModel": ("ads.model.generic_model", "GenericModel"),
    "ModelState": ("ads.model.generic_model", "ModelState"),
    "ModelProperties": ("ads.model.model_properties", "ModelProperties"),
    "ModelVersionSet": ("ads.model.model_version_set", "ModelVersionSet"),
    "experiment": ("ads.model.model_version_set", "experiment"),
    "SERDE": ("ads.model.serde.common", "SERDE"),
    "ModelInputSerializer": ("ads.model.serde.model_input", "ModelInputSerializer"),
    "ModelVersionSetNotExists": (
        "ads.model.service.oci_datascience_model_version_set",
        "ModelVersionSetNotExists",
    ),
    "ModelVersionSetNotSaved": (
        "ads.model.service.oci_datascience_model_version_set",
        "ModelVersionSetNotSaved",
    ),
}

__all__ = [
    "GenericModel",
    "ModelState",
    "DataScienceModel",
    "ModelProperties",
    "AutoMLModel",
    "LightGBMModel",
    "PyTorchModel",
    "SklearnModel",
    "TensorFlowModel",
    "XGBoostModel",
    "SparkPipelineModel",
    "HuggingFacePipelineModel",
    "EmbeddingONNXModel",
    "ModelDeployer",
    "ModelDeployment",
    "ModelDeploymentProperties",
    "SERDE",
    "ModelInputSerializer",
    "ModelVersionSet",
    "experiment",
    "ModelVersionSetNotExists",
    "ModelVersionSetNotSaved",
]


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
