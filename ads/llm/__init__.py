#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

try:
    import langchain

    from ads.llm.chat_template import ChatTemplates
    from ads.llm.langchain.plugins.chat_models.oci_data_science import (
        ChatOCIModelDeployment,
        ChatOCIModelDeploymentTGI,
        ChatOCIModelDeploymentVLLM,
    )
    from ads.llm.langchain.plugins.embeddings.oci_data_science_model_deployment_endpoint import (
        OCIDataScienceEmbedding,
    )
    from ads.llm.langchain.plugins.llms.oci_data_science_model_deployment_endpoint import (
        OCIModelDeploymentLLM,
        OCIModelDeploymentTGI,
        OCIModelDeploymentVLLM,
    )
except ImportError as ex:
    if ex.name == "langchain":
        raise ImportError(
            f"{ex.msg}\nPlease install/update langchain with `pip install langchain -U`"
        ) from ex
    raise ex
