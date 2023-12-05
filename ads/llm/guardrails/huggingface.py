#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import evaluate
from langchain.pydantic_v1 import root_validator
from .base import Guardrail


class HuggingFaceEvaluation(Guardrail):
    path: str = ""
    load_args: dict = {}
    compute_args: dict = {}
    _evaluator: evaluate.EvaluationModule = ""

    @root_validator(skip_on_failure=True)
    def load_model(cls, values):
        """Loads the model from Huggingface."""
        if values.get("path"):
            path = values["path"]
        else:
            path = values["load_args"].get("path")
            values["path"] = path
        if not path:
            raise NotImplementedError("Please provide path in load_args.")

        if not values.get("name"):
            values["name"] = path

        return values

    def compute(self, data=None, **kwargs):
        if not self._evaluator:
            load_args = {"path": self.path}
            load_args.update(self.load_args)
            self._evaluator = evaluate.load(**load_args)
        return self._evaluator.compute(predictions=data, **self.compute_args, **kwargs)

    @property
    def metric_key(self):
        return self.path
