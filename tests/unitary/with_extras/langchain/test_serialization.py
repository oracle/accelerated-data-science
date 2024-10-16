#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os

from unittest import SkipTest, TestCase, mock, skipIf

import pytest

pytest.skip(allow_module_level=True)
# TODO: Tests need to be updated

import langchain_core
from langchain.chains import LLMChain
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

from ads.llm import (
    OCIModelDeploymentTGI,
    OCIModelDeploymentVLLM,
)
from ads.llm.serialize import dump, load


def version_tuple(version):
    return tuple(map(int, version.split(".")))


class ChainSerializationTest(TestCase):
    """Contains tests for chain serialization."""

    # LangChain is updating frequently on the module organization,
    # mainly affecting the id field of the serialization.
    # In the test, we will not check the id field of some components.
    # We expect users to use the same LangChain version for serialize and de-serialize

    def setUp(self) -> None:
        # self.maxDiff = None
        return super().setUp()

    PROMPT_TEMPLATE = "Tell me a joke about {subject}"
    COMPARTMENT_ID = "<ocid>"
    GEN_AI_KWARGS = {"service_endpoint": "https://endpoint.oraclecloud.com"}
    ENDPOINT = "https://modeldeployment.customer-oci.com/ocid/predict"

    EXPECTED_GEN_AI_EMBEDDINGS = {
        "lc": 1,
        "type": "constructor",
        "id": ["ads", "llm", "GenerativeAIEmbeddings"],
        "kwargs": {
            "compartment_id": "<ocid>",
            "client_kwargs": {"service_endpoint": "https://endpoint.oraclecloud.com"},
        },
    }

    EXPECTED_RUNNABLE_SEQUENCE = {
        "lc": 1,
        "type": "constructor",
        "kwargs": {
            "first": {
                "lc": 1,
                "type": "constructor",
                "kwargs": {
                    "steps": {
                        "text": {
                            "lc": 1,
                            "type": "constructor",
                            "id": [
                                "langchain_core",
                                "runnables",
                                "RunnablePassthrough",
                            ],
                            "kwargs": {"func": None, "afunc": None, "input_type": None},
                        }
                    }
                },
                "_type": "RunnableParallel",
            },
            "middle": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "kwargs": {
                        "input_variables": ["subject"],
                        "template": "Tell me a joke about {subject}",
                        "template_format": "f-string",
                        "partial_variables": {},
                    },
                }
            ],
            "last": {
                "lc": 1,
                "type": "constructor",
                "id": ["ads", "llm", "ModelDeploymentTGI"],
                "kwargs": {
                    "endpoint": "https://modeldeployment.customer-oci.com/ocid/predict"
                },
            },
        },
    }

    @mock.patch.dict(os.environ, {"COHERE_API_KEY": "api_key"})
    def test_llm_chain_serialization_with_cohere(self):
        """Tests serialization of LLMChain with Cohere."""
        llm = Cohere()
        template = PromptTemplate.from_template(self.PROMPT_TEMPLATE)
        llm_chain = LLMChain(prompt=template, llm=llm, verbose=True)
        serialized = dump(llm_chain)

        # Check the serialized chain
        self.assertTrue(serialized.get("verbose"))
        self.assertEqual(serialized.get("_type"), "llm_chain")

        # Check the serialized prompt template
        serialized_prompt = serialized.get("prompt")
        self.assertIsInstance(serialized_prompt, dict)
        self.assertEqual(serialized_prompt.get("_type"), "prompt")
        self.assertEqual(set(serialized_prompt.get("input_variables")), {"subject"})
        self.assertEqual(serialized_prompt.get("template"), self.PROMPT_TEMPLATE)

        # Check the serialized LLM
        serialized_llm = serialized.get("llm")
        self.assertIsInstance(serialized_llm, dict)
        self.assertEqual(serialized_llm.get("_type"), "cohere")

        llm_chain = load(serialized)
        self.assertIsInstance(llm_chain, LLMChain)
        self.assertIsInstance(llm_chain.prompt, PromptTemplate)
        self.assertEqual(llm_chain.prompt.template, self.PROMPT_TEMPLATE)
        self.assertIsInstance(llm_chain.llm, Cohere)
        self.assertEqual(llm_chain.input_keys, ["subject"])

    def test_runnable_sequence_serialization(self):
        """Tests serialization of runnable sequence."""
        map_input = RunnableParallel(text=RunnablePassthrough())
        template = PromptTemplate.from_template(self.PROMPT_TEMPLATE)
        llm = OCIModelDeploymentTGI(endpoint=self.ENDPOINT)

        chain = map_input | template | llm
        serialized = dump(chain)

        self.assertEqual(serialized.get("type"), "constructor")
        self.assertNotIn("_type", serialized)

        kwargs = serialized.get("kwargs")
        self.assertIsInstance(kwargs, dict)

        element_1 = kwargs.get("first")
        self.assertEqual(element_1.get("_type"), "RunnableParallel")
        step = element_1.get("kwargs", dict()).get("steps", dict()).get("text", dict())
        self.assertEqual(
            step.get("id", ["RunnablePassthrough"])[-1], "RunnablePassthrough"
        )

        element_2 = kwargs.get("middle")[0]
        self.assertNotIn("_type", element_2)
        self.assertEqual(element_2.get("kwargs").get("template"), self.PROMPT_TEMPLATE)
        self.assertEqual(element_2.get("kwargs").get("input_variables"), ["subject"])

        element_3 = kwargs.get("last")
        self.assertNotIn("_type", element_3)
        self.assertEqual(element_3.get("id"), ["ads", "llm", "ModelDeploymentTGI"])

        if version_tuple(langchain_core.__version__) > (0, 1, 50):
            self.assertEqual(
                element_3.get("kwargs"),
                {
                    "max_tokens": 256,
                    "temperature": 0.2,
                    "p": 0.75,
                    "endpoint": "https://modeldeployment.customer-oci.com/ocid/predict",
                    "best_of": 1,
                    "do_sample": True,
                    "watermark": True,
                },
            )
        else:
            self.assertEqual(
                element_3.get("kwargs"),
                {
                    "endpoint": "https://modeldeployment.customer-oci.com/ocid/predict",
                },
            )

        chain = load(serialized)
        self.assertEqual(len(chain.steps), 3)
        self.assertIsInstance(chain.steps[0], RunnableParallel)
        self.assertEqual(
            list(chain.steps[0].dict().get("steps", dict()).keys()),
            [],
        )
        self.assertIsInstance(chain.steps[1], PromptTemplate)
        self.assertIsInstance(chain.steps[2], OCIModelDeploymentTGI)
        self.assertEqual(chain.steps[2].endpoint, self.ENDPOINT)
