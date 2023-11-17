#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from typing import Any, List, Mapping, Optional
from unittest import TestCase
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from ads.llm.guardrails import HuggingFaceEvaluation
from ads.llm.guardrails.base import BlockedByGuardrail, GuardrailIO
from ads.llm.chain import GuardrailSequence
from ads.llm.serialize import load, dump


class FakeLLM(LLM):
    """Fake LLM for testing purpose."""

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """This class is LangChain serializable."""
        return True


class GuardrailTestsBase(TestCase):
    """Base class for guardrail tests."""

    TOXIC_CONTENT = "Women is not capable of this job."
    LOAD_ARGS = {"cache_dir": os.path.expanduser("~/.cache/huggingface/evaluate")}
    FAKE_LLM = FakeLLM()


class ToxicityGuardrailTests(GuardrailTestsBase):
    """Contains tests for the toxicity guardrail."""

    def test_toxicity_without_threshold(self):
        """When using guardrail alone with is no threshold, it does not do anything."""
        toxicity = HuggingFaceEvaluation(path="toxicity", load_args=self.LOAD_ARGS)
        chain = self.FAKE_LLM | toxicity
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertEqual(output, self.TOXIC_CONTENT)
        serialized = dump(chain)
        chain = load(serialized, valid_namespaces=["tests"])
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertEqual(output, self.TOXIC_CONTENT)

    def test_toxicity_with_threshold(self):
        """Once we set a threshold, an exception will be raise for toxic output."""
        toxicity = HuggingFaceEvaluation(
            path="toxicity", threshold=0.2, load_args=self.LOAD_ARGS
        )
        chain = self.FAKE_LLM | toxicity
        with self.assertRaises(BlockedByGuardrail):
            chain.invoke(self.TOXIC_CONTENT)
        serialized = dump(chain)
        chain = load(serialized, valid_namespaces=["tests"])
        with self.assertRaises(BlockedByGuardrail):
            chain.invoke(self.TOXIC_CONTENT)

    def test_toxicity_without_exception(self):
        """Guardrail can return the custom message instead of raising an exception."""
        toxicity = HuggingFaceEvaluation(
            path="toxicity",
            threshold=0.2,
            raise_exception=False,
            custom_msg="Sorry, but let's discuss something else.",
            load_args=self.LOAD_ARGS,
        )
        chain = self.FAKE_LLM | toxicity
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertEqual(output, toxicity.custom_msg)
        serialized = dump(chain)
        chain = load(serialized, valid_namespaces=["tests"])
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertEqual(output, toxicity.custom_msg)

    def test_toxicity_return_metrics(self):
        """Return the toxicity metrics"""
        toxicity = HuggingFaceEvaluation(
            path="toxicity", return_metrics=True, load_args=self.LOAD_ARGS
        )
        chain = self.FAKE_LLM | toxicity
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertIsInstance(output, dict)
        self.assertEqual(output["output"], self.TOXIC_CONTENT)
        self.assertGreater(output["metrics"]["toxicity"][0], 0.2)
        serialized = dump(chain)
        chain = load(serialized, valid_namespaces=["tests"])
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertIsInstance(output, dict)
        self.assertEqual(output["output"], self.TOXIC_CONTENT)
        self.assertGreater(output["metrics"]["toxicity"][0], 0.2)


class GuardrailSequenceTests(GuardrailTestsBase):
    """Contains tests for GuardrailSequence."""

    def test_guardrail_sequence_with_template_and_toxicity(self):
        template = PromptTemplate.from_template("Tell me a joke about {subject}")
        map_input = RunnableMap(subject=RunnablePassthrough())
        toxicity = HuggingFaceEvaluation(
            path="toxicity", load_args=self.LOAD_ARGS, select=min
        )
        chain = GuardrailSequence.from_sequence(
            map_input | template | self.FAKE_LLM | toxicity
        )
        output = chain.run("cats", num_generations=5)
        self.assertIsInstance(output, GuardrailIO)
