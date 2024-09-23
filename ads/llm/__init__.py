#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

try:
    import langchain
    from ads.llm.langchain.plugins.llms.oci_data_science_model_deployment_endpoint import (
        OCIModelDeploymentVLLM,
        OCIModelDeploymentTGI,
    )
    from ads.llm.langchain.plugins.chat_models.oci_data_science import (
        ChatOCIModelDeployment,
        ChatOCIModelDeploymentVLLM,
        ChatOCIModelDeploymentTGI,
    )
    from ads.llm.chat_template import ChatTemplates
except ImportError as ex:
    if ex.name == "langchain":
        raise ImportError(
            f"{ex.msg}\nPlease install/update langchain with `pip install langchain -U`"
        ) from ex
    raise ex
