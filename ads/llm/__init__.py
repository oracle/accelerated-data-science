#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from ads.llm.langchain.plugins.llm_gen_ai import GenerativeAI
from ads.llm.langchain.plugins.llm_md import ModelDeploymentTGI
from ads.llm.langchain.plugins.llm_md import ModelDeploymentVLLM
from ads.llm.langchain.plugins.embeddings import GenerativeAIEmbeddings
from ads.llm.load import load
