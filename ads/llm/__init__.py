#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

try:
    import langchain
    from ads.llm.langchain.plugins.llm_gen_ai import GenerativeAI
    from ads.llm.langchain.plugins.llm_md import ModelDeploymentTGI
    from ads.llm.langchain.plugins.llm_md import ModelDeploymentVLLM
    from ads.llm.langchain.plugins.embeddings import GenerativeAIEmbeddings
except ImportError as ex:
    if ex.name == "langchain":
        raise ImportError(
            f"{ex.msg}\nPlease install/update langchain with `pip install langchain -U`"
        ) from ex
    raise ex
