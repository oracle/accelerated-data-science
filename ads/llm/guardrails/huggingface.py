#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import evaluate
from .base import Guardrail


class HuggingFaceEvaluation(Guardrail):

    path: str = ""
    load_args: dict = {}
    _evaluator: evaluate.EvaluationModule = ""

    def __init__(self, **kwargs):
        if "name" not in kwargs and "path" in kwargs:
            kwargs["name"] = kwargs["path"]
        super().__init__(**kwargs)
        self._evaluator = None
        # Load evaluator only if user did not specified one in constructor.
        if not self._evaluator:
            self.load()

    def load(self) -> None:
        if not self.path and "path" not in self.load_args:
            raise NotImplementedError("Please provide path in load_args.")
        load_args = {"path": self.path}
        load_args.update(self.load_args)
        self._evaluator = evaluate.load(**load_args)

    def compute(self, data=None, **kwargs):
        return self._evaluator.compute(predictions=data, **kwargs)

    @property
    def metric_key(self):
        return self.path
