#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List
from langchain.llms.base import LLM
from langchain.pydantic_v1 import root_validator
from ads.llm.guardrails.base import Guardrail


class ClassificationBase(Guardrail):
    """Base class for classifying texts."""

    allowlist: List[str] = []
    denylist: List[str] = []

    @root_validator
    def check_lists(cls, values):
        """Checks the allowlist and denylist."""
        allowlist = values["allowlist"]
        denylist = values["denylist"]
        if allowlist and denylist:
            raise ValueError("Please specify only allowlist or denylist.")
        return values


class TextClassificationWithLLM(ClassificationBase):
    """Text classification guardrail using LLM."""

    llm: LLM

    template = (
        "Answer yes or no. Is the following text related to {label}?\ntext: {text}\n"
    )

    def _check_with_llm(self, text, label):
        prompt = self.template.format(label=label, text=text)
        return self.llm.invoke(prompt).strip().lower()

    def _check_allowlist(self, text) -> bool:
        for label in self.allowlist:
            response = self._check_with_llm(text, label)
            if response == "yes":
                return True
        return False

    def _check_denylist(self, text) -> bool:
        for label in self.denylist:
            response = self._check_with_llm(text, label)
            if response == "yes":
                return False
        return True

    def compute(self, data=None, **kwargs) -> dict:
        if not data:
            data = []

        passed = []
        for text in data:
            if self.allowlist:
                passed.append(self._check_allowlist(text))
            elif self.denylist:
                passed.append(self._check_denylist(text))
            else:
                passed.append(True)

        return {
            "texts": data,
            "passed": passed,
        }

    def moderate(self, metrics: dict, data=None, **kwargs) -> List[str]:
        texts = metrics["texts"]
        passed = metrics["passed"]
        return [text for i, text in enumerate(texts) if passed[i]]


class TextClassificationWithSpaCy(ClassificationBase):
    """Checks the text with a SpaCy model"""

    model_name: str
    _model = None

    def load_model(self) -> None:
        """Loads the SpaCy model."""
        import spacy

        self._model = spacy.load(self.model_name)

    def classify(self, text):
        """Classifies the text."""
        # Load the model lazily
        if not self._model:
            self.load_model()
        return self._model(text).cats

    def compute(self, data=None, **kwargs) -> dict:
        metrics = []
        if not data:
            data = []
        for text in data:
            scores = self.classify(text)
            label = max(scores, key=scores.get)
            label_score = scores[label]
            metrics.append([text, label, label_score, scores])
        return {
            "texts": data,
            "labels": [metric[1] for metric in metrics],
            # "label_scores": [metric[2] for metric in metrics],
            "scores": [metric[3] for metric in metrics],
        }

    def moderate(self, metrics: dict, data=None, **kwargs) -> List[str]:
        """Moderate the texts based on the metrics.

        Parameters
        ----------
        metrics : dict
            The metrics dictionary should contain two keys "texts", "labels", "scores".
            Each value should be a list corresponding to the input texts.
        data : list, optional
            The input texts as list of strings

        Returns
        -------
        list
            A list of texts that passed the moderation.
        """
        if not data:
            data = []
        outputs = []
        if self.allowlist:
            for i, text in enumerate(data):
                if metrics["labels"][i] in self.allowlist:
                    outputs.append(text)
        elif self.denylist:
            for i, text in enumerate(data):
                if metrics["labels"][i] not in self.denylist:
                    outputs.append(text)
        return outputs
