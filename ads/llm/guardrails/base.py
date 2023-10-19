#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import datetime
import operator
from abc import ABC
from typing import Any, List
from langchain.load.serializable import Serializable
from langchain.schema.prompt import PromptValue
from langchain.schema.runnable import (
    Runnable,
    RunnableConfig,
)
from pydantic import BaseModel


class RunInfo(BaseModel):
    """Represents the information about data going through a guardrail."""

    name: str = None
    """The name of the guardrail."""
    input: Any
    """The inputs to the guardrail."""
    output: Any = None
    """The outputs from the guardrail."""
    parameters: dict = {}
    """The spec for the guardrail."""
    metrics: dict = {}
    """The metrics produced by the guardrail."""
    time: datetime.datetime = None
    """The time when the guardrail was invoked."""
    duration: float = None
    """The duration in seconds taken to run the guardrail."""

    def __enter__(self):
        self.time = datetime.datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.duration = (datetime.datetime.now() - self.time).total_seconds()


class GuardrailIO(BaseModel):
    """The data structure for the guardrail inputs and outputs.

    This is designed to be the standard data object passing through the guardrails.

    The data property is designed to hold user facing data, like prompt and completion/response.
    When a guardrail is chained (|) with a LangChain (non-guardrail) ``Runnable`` component,
    the data property is used as the ``Input`` of the LangChain component.
    The ``Output`` of the LangChain component will be saved into the data property once it is invoked.
    When processed by a guardrail, the data property is passed to the ``compute()`` method of the guardrail.
    See the ``Guardrail`` base class for more details.

    As the ``GuardrailIO`` object is processed by a guardrail, a new ``RunInfo`` object will be attached,
    so that the metric and parameters can be tracked.
    If the ``GuardrailIO`` object is processed by a LangChain (non-guardrail) component,
    the ``RunInfo`` attached could be empty or contains only parameters,
    as LangChain components do not return structured metrics.
    """

    data: Any
    """User facing data, like prompt and completion/response."""
    info: List[RunInfo] = []
    """A list of RunInfo attached by guardrails that processed the data."""


class Guardrail(Serializable, Runnable):
    """Base class for guardrails.

    Each Guardrail should be compatible with the LangChain Serializable and Runnable interface.
    A new ``RunnableSerializable`` class was added in LangChain v0.0.307.
    https://github.com/langchain-ai/langchain/blob/v0.0.307/libs/langchain/langchain/schema/runnable/base.py#L863
    The Guardrail class may inherit from ``RunnableSerializable`` in the future.

    To implement a new guardrail:
    1. Add the guardrail config/spec as class properties (similar to the ``name`` property).
    2. Override the ``compute()`` and ``moderate()`` methods as needed.

    When guardrail is initialized, the YAML spec will be loaded as keyword arguments,
    and save into the class attributes, this is handled by ``pydantic``.

    The ``Input`` to the guardrail could be any data types you would like to support,
    the data may be wrapped by a ``GuardrailIO`` object when processed by another guardrail upstream.
    The ``GuardrailIO`` object is used by the guardrail internally and track the metrics.
    Normally your guardrail implementation do not need to handle the ``GuardrailIO`` object.
    If the ``Input`` is not a ``GuardrailIO`` object,
    the ``preprocess()`` method will do the following to wrap it as a ``GuardrailIO`` object:
    1. For a single string, it will be converted to a list with a single string.
    2. For a LangChain `PromptValue`, it will be converted to a string (to_string) then put it into a list.
    3. Otherwise, the ``Input`` will be saved into the ``data`` property of the ``GuardrailIO`` object.
    You may want to override the ``preprocess()`` method if you need additional handling.

    After preprocessing, the data will be passed into the ``compute()`` and ``moderate()`` methods
    in the following ways:
    1. If the data is a dict, it will be passed as ``**kwargs``. You may have ``data`` as keys in the dict.
    2. Otherwise, the data will be passed as the ``data`` argument.

    The ``compute()`` method should compute the metrics and return them as a dictionary.
    The metrics will be appended to the ``info`` property of the ``GuardrailIO`` output.
    The ``moderate()`` method should return the moderated outputs based on the inputs and metrics.
    By default, the ``moderate()`` method returns the inputs as is.
    Note that the ``metrics`` are passed into the ``moderate()`` method as reference (dict).
    Any changes on the ``metrics`` in the ``moderate()`` method will be reflected in the ``GuardrailIO`` output.
    If you are not able to separate the logic of ``compute()`` and ``moderate()``,
    you may do all the work in ``compute()`` and save the moderated data in the metrics,
    then in ``moderate()`` simply return (or pop) the moderated data from the metrics.

    In LangChain, the ``Input`` and ``Output`` types of a ``Runnable`` are usually well defined.
    In Guardrails, to enable additional data going through the guardrails,
    the Input and Output are wrapped as ``GuardrailIO`` objects.
    Although not required, you may restrict them by adding the types like
    ``class YourGuardrail(Guardrail[Union[GuardrailIO, YourInputType], GuardrailIO):``

    """

    name: str = ""
    custom_msg: str = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.name:
            self.name = self.__class__.__name__

    def __or__(self, other):
        from ..chain import GuardrailSequence

        return GuardrailSequence.from_sequence(super().__or__(other))

    def __ror__(self, other):
        from ..chain import GuardrailSequence

        return GuardrailSequence.from_sequence(super().__ror__(other))

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    def save(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "path": self.__module__,
            "spec": self.dict(),
        }

    def preprocess(self, input: Any) -> GuardrailIO:
        if isinstance(input, GuardrailIO):
            return input
        if isinstance(input, str):
            input = [input]
        if isinstance(input, PromptValue):
            input = [input.to_string()]
        return GuardrailIO(data=input)

    def invoke(self, input: Any, config: RunnableConfig = None) -> Any:
        obj = self.preprocess(input)
        with RunInfo(name=self.name, input=obj.data) as info:
            # Runnable has the to_json() method to return a dictionary
            # containing the ``kwargs`` used to initialize the object.
            # The ``kwargs`` does not contain the defaults.
            # Here the ``dict()`` method is used to return a dictionary containing the defaults.
            info.parameters = self.save()
            # Here the data in GuardrailIO is extracted and passed into compute.
            # You may need to override invoke() method
            # and extra the data for your customized guardrail.
            if isinstance(obj.data, dict):
                data = None
                kwargs = data
            else:
                data = obj.data
                kwargs = {}
            info.metrics = self.compute(data, **kwargs)
            info.output = self.moderate(info.metrics, data, **kwargs)
        obj.info.append(info)
        obj.data = info.output
        return obj

    def load(self) -> None:
        """Loads the models and configs needed for the guardrail."""

    def compute(self, data=None, **kwargs) -> dict:
        """Computes the metrics and returns a dictionary."""
        return {}

    def moderate(self, metrics: dict, data=None, **kwargs) -> List[str]:
        """Checks the metrics and see if the data can pass the guardrail.
        Returns the moderated data as needed.
        """
        return data


class SingleMetric(BaseModel, ABC):
    """Interface for guardrail processing a list of data(texts)
    and produces a single score for each element(text) in the list.

    The metrics produced by the ``compute()`` still needs to be a dictionary.
    The metrics may contain multiple keys.
    The key holding the list of scores should be returned by the ``metric_key`` property.
    """

    _SUPPORTED_FUNC = {"min": min, "max": max}
    _SUPPORTED_OPERATOR = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }

    select: str = None
    threshold: float = None
    direction: str = "<"

    @property
    def metric_key(self):
        """Returns the key holding the list of scores corresponding to the input data.
        By default, this will return the class name.
        """
        return self.__class__.__name__

    def pick(self, metrics: dict, data: list):
        if self.select not in self._SUPPORTED_FUNC:
            raise ValueError(f"select='{self.select}' is not supported.")
        func = self._SUPPORTED_FUNC[self.select]
        values = metrics[self.metric_key]
        idx = values.index(func(values))
        metrics["selected"] = idx
        data = [data[idx]]
        return data

    def apply_filter(self, metrics: dict, data: list):
        if self.direction not in self._SUPPORTED_OPERATOR:
            raise ValueError(f"direction='{self.direction}' is not supported.")
        operation = self._SUPPORTED_OPERATOR[self.direction]
        values = metrics[self.metric_key]
        passed = [operation(val, self.threshold) for val in values]
        metrics["passed"] = passed
        return [data[i] for i in range(len(passed)) if passed[i]]

    def filter_and_pick(self, metrics: dict, data: list):
        filtered_data = self.apply_filter(metrics, data)
        passed_idx = [i for i in range(len(metrics["passed"])) if metrics["passed"]]
        filtered_metrics = {
            self.metric_key: [metrics[self.metric_key][i] for i in passed_idx]
        }
        return self.pick(filtered_data, filtered_metrics)

    def moderate(self, metrics: dict, data: list, **kwargs) -> List[str]:
        if self.select and self.threshold is not None:
            return self.filter_and_pick(metrics, data)
        elif self.select:
            return self.pick(metrics, data)
        elif self.threshold is not None:
            return self.apply_filter(metrics, data)
        return data
