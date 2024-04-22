#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import tempfile
from typing import Any, List, Dict, Mapping, Optional
from unittest import TestCase
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from ads.llm.guardrails.huggingface import HuggingFaceEvaluation
from ads.llm.guardrails.base import BlockedByGuardrail, GuardrailIO
from ads.llm.chain import GuardrailSequence
from ads.llm.serialize import load, dump


class FakeLLM(LLM):
    """Fake LLM for testing purpose."""

    mapping: Dict[str, str] = None
    """Mapping prompts to responses.
    If prompt is found in the mapping, the corresponding response will be returned.
    Otherwise, the prompt will be returned as is.
    """

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
        if self.mapping:
            return self.mapping.get(prompt, prompt)
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

    def assert_before_and_after_serialization(self, test_fn, chain):
        """Runs test function with chain, serialize and deserialize it, then run the test function again."""
        test_fn(chain)
        serialized = dump(chain)
        chain = load(serialized, valid_namespaces=["tests"])
        test_fn(chain)


class ToxicityGuardrailTests(GuardrailTestsBase):
    """Contains tests for the toxicity guardrail."""

    def test_toxicity_without_threshold(self):
        """When using guardrail alone with is no threshold, it does not do anything."""
        toxicity = HuggingFaceEvaluation(path="toxicity", load_args=self.LOAD_ARGS)
        chain = self.FAKE_LLM | toxicity

        def test_fn(chain):
            output = chain.invoke(self.TOXIC_CONTENT)
            self.assertEqual(output, self.TOXIC_CONTENT)

        self.assert_before_and_after_serialization(test_fn, chain)

    def test_toxicity_with_threshold(self):
        """Once we set a threshold, an exception will be raise for toxic output."""
        toxicity = HuggingFaceEvaluation(
            path="toxicity", threshold=0.2, load_args=self.LOAD_ARGS
        )
        chain = self.FAKE_LLM | toxicity

        def test_fn(chain):
            with self.assertRaises(BlockedByGuardrail):
                chain.invoke(self.TOXIC_CONTENT)

        self.assert_before_and_after_serialization(test_fn, chain)

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

        def test_fn(chain):
            output = chain.invoke(self.TOXIC_CONTENT)
            self.assertEqual(output, toxicity.custom_msg)

        self.assert_before_and_after_serialization(test_fn, chain)

    def test_toxicity_return_metrics(self):
        """Return the toxicity metrics"""
        toxicity = HuggingFaceEvaluation(
            path="toxicity", return_metrics=True, load_args=self.LOAD_ARGS
        )
        chain = self.FAKE_LLM | toxicity

        def test_fn(chain):
            output = chain.invoke(self.TOXIC_CONTENT)
            self.assertIsInstance(output, dict)
            self.assertEqual(output["output"], self.TOXIC_CONTENT)
            self.assertGreater(output["metrics"]["toxicity"][0], 0.2)

        self.assert_before_and_after_serialization(test_fn, chain)


class GuardrailSequenceTests(GuardrailTestsBase):
    """Contains tests for GuardrailSequence."""

    def test_guardrail_sequence_with_template_and_toxicity(self):
        """Tests a guardrail sequence with template and toxicity evaluation."""
        template = PromptTemplate.from_template("Tell me a joke about {subject}")
        map_input = RunnableMap(subject=RunnablePassthrough())
        toxicity = HuggingFaceEvaluation(
            path="toxicity", load_args=self.LOAD_ARGS, select="min"
        )
        chain = GuardrailSequence.from_sequence(
            map_input | template | self.FAKE_LLM | toxicity
        )

        def test_fn(chain: GuardrailSequence):
            output = chain.run("cats", num_generations=5)
            self.assertIsInstance(output, GuardrailIO)
            self.assertIsInstance(output.data, str)
            self.assertEqual(output.data, "Tell me a joke about cats")
            self.assertIsInstance(output.info, list)
            self.assertEqual(len(output.info), len(chain.steps))

        self.assert_before_and_after_serialization(test_fn, chain)

    def test_guardrail_sequence_with_filtering(self):
        message = "Let's talk something else."
        toxicity = HuggingFaceEvaluation(
            path="toxicity",
            load_args=self.LOAD_ARGS,
            threshold=0.5,
            custom_msg=message,
        )
        chain = GuardrailSequence.from_sequence(self.FAKE_LLM | toxicity)

        def test_fn(chain: GuardrailSequence):
            output = chain.run(self.TOXIC_CONTENT)
            self.assertIsInstance(output, GuardrailIO)
            self.assertIsInstance(output.data, str)
            self.assertEqual(output.data, message)
            self.assertIsInstance(output.info, list)
            self.assertEqual(len(output.info), len(chain.steps))

        self.assert_before_and_after_serialization(test_fn, chain)


    def test_save_to_file(self):
        """Tests saving to file."""
        message = "Let's talk something else."
        toxicity = HuggingFaceEvaluation(
            path="toxicity",
            load_args=self.LOAD_ARGS,
            threshold=0.5,
            custom_msg=message,
        )
        chain = GuardrailSequence.from_sequence(self.FAKE_LLM | toxicity)
        try:
            temp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            temp.close()
            with self.assertRaises(FileExistsError):
                serialized = chain.save(temp.name)
            with self.assertRaises(ValueError):
                chain.save("abc.html")
            serialized = chain.save(temp.name, overwrite=True)
            with open(temp.name, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), serialized)
        finally:
            os.unlink(temp.name)
