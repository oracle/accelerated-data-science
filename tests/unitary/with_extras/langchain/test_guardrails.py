#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Any, List, Mapping, Optional
from unittest import TestCase
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from ads.llm.guardrails import HuggingFaceEvaluation
from ads.llm.guardrails.base import BlockedByGuardrail


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


class ToxicityGuardrailTests(TestCase):
    """Contains tests for the toxicity guardrail."""

    TOXIC_CONTENT = "Women is not capable of this job."
    FAKE_LLM = FakeLLM()

    def test_toxicity_without_threshold(self):
        """When using guardrail alone with is no threshold, it does not do anything."""
        toxicity = HuggingFaceEvaluation(path="toxicity")
        chain = self.FAKE_LLM | toxicity
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertEqual(output, self.TOXIC_CONTENT)

    def test_toxicity_with_threshold(self):
        """Once we set a threshold, an exception will be raise for toxic output."""
        toxicity = HuggingFaceEvaluation(path="toxicity", threshold=0.2)
        chain = self.FAKE_LLM | toxicity
        with self.assertRaises(BlockedByGuardrail):
            chain.invoke(self.TOXIC_CONTENT)

    def test_toxicity_without_exception(self):
        """Guardrail can return the custom message instead of raising an exception."""
        toxicity = HuggingFaceEvaluation(
            path="toxicity",
            threshold=0.2,
            raise_exception=False,
            custom_msg="Sorry, but let's discuss something else.",
        )
        chain = self.FAKE_LLM | toxicity
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertEqual(output, toxicity.custom_msg)

    def test_toxicity_return_metrics(self):
        """Return the toxicity metrics"""
        toxicity = HuggingFaceEvaluation(path="toxicity", return_metrics=True)
        chain = self.FAKE_LLM | toxicity
        output = chain.invoke(self.TOXIC_CONTENT)
        self.assertIsInstance(output, dict)
        self.assertEqual(output["output"], self.TOXIC_CONTENT)
        self.assertGreater(output["metrics"]["toxicity"][0], 0.2)
