#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib

_LAZY_ATTRS = {
    "ChatTemplates": ("ads.llm.chat_template", "ChatTemplates"),
    "ChatOCIModelDeployment": (
        "ads.llm.langchain.plugins.chat_models.oci_data_science",
        "ChatOCIModelDeployment",
    ),
    "ChatOCIModelDeploymentTGI": (
        "ads.llm.langchain.plugins.chat_models.oci_data_science",
        "ChatOCIModelDeploymentTGI",
    ),
    "ChatOCIModelDeploymentVLLM": (
        "ads.llm.langchain.plugins.chat_models.oci_data_science",
        "ChatOCIModelDeploymentVLLM",
    ),
    "OCIDataScienceEmbedding": (
        "ads.llm.langchain.plugins.embeddings.oci_data_science_model_deployment_endpoint",
        "OCIDataScienceEmbedding",
    ),
    "OCIModelDeploymentLLM": (
        "ads.llm.langchain.plugins.llms.oci_data_science_model_deployment_endpoint",
        "OCIModelDeploymentLLM",
    ),
    "OCIModelDeploymentTGI": (
        "ads.llm.langchain.plugins.llms.oci_data_science_model_deployment_endpoint",
        "OCIModelDeploymentTGI",
    ),
    "OCIModelDeploymentVLLM": (
        "ads.llm.langchain.plugins.llms.oci_data_science_model_deployment_endpoint",
        "OCIModelDeploymentVLLM",
    ),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        try:
            value = getattr(importlib.import_module(module_name), attr_name)
        except ImportError as ex:
            if ex.name and ex.name.startswith("langchain"):
                raise ImportError(
                    "ADS LLM integrations require LangChain dependencies. "
                    "Install/update them with `pip install langchain -U`."
                ) from ex
            raise
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
