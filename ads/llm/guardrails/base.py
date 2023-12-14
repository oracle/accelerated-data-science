#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import datetime
import functools
import operator
import importlib.util
import sys

from typing import Any, List, Dict, Tuple
from langchain.schema.prompt import PromptValue
from langchain.tools.base import BaseTool, ToolException
from langchain.pydantic_v1 import BaseModel, root_validator


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

    This is designed to be the standard data object passing through the GuardrailSequence.

    The data property is designed to hold user facing data, like prompt and completion/response.
    In a GuardrailSequence, when a guardrail is chained (|) with a LangChain (non-guardrail)
    ``Runnable`` component, the data property is used as the ``Input`` of the LangChain component.
    The ``Output`` of the LangChain component will be saved into the data property once it is invoked.
    When processed by a guardrail, the data property is passed to the ``compute()`` method of the guardrail.
    See the ``Guardrail`` base class for more details.

    As the ``GuardrailIO`` object is processed by a guardrail in a GuardrailSequence,
    a new ``RunInfo`` object will be attached, so that the metric and parameters can be tracked.
    If the ``GuardrailIO`` object is processed by a LangChain (non-guardrail) component,
    the ``RunInfo`` attached could be empty or contains only parameters,
    as LangChain components do not return structured metrics.
    """

    data: Any
    """User facing data, like prompt and completion/response."""
    info: List[RunInfo] = []
    """A list of RunInfo attached by guardrails that processed the data."""

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        steps = []
        run_info = None
        for run_info in self.info:
            steps.append(str(run_info.input))
            steps.append(f"{run_info.name} - {run_info.metrics}")
        if run_info:
            steps.append(str(run_info.output))
        return "\n".join(steps) + "\n\n" + str(self)


class BlockedByGuardrail(ToolException):
    """Exception when the content is blocked by a guardrail."""

    def __init__(self, message: str = None, info: RunInfo = None) -> None:
        self.message = message
        self.info = info


class SingleMetric:
    """Class containing decorator for checking if the metrics is compatible
    with methods designed to work with single metric.
    """

    @staticmethod
    def check(func):
        """Checks if the metrics argument in the method call is compatible
        with methods designed to work with single metric.
        """

        @functools.wraps(func)
        def wrapper(self: "Guardrail", metrics: dict, data: list, *args, **kwargs):
            if self.metric_key not in metrics:
                raise KeyError(
                    f"Method requires the metrics contains {self.metric_key}."
                )
            if not isinstance(metrics[self.metric_key], list):
                raise ValueError(
                    f"Method requires the value of {self.metric_key} in metrics."
                )
            if len(metrics[self.metric_key]) != len(data):
                raise ValueError(
                    f"Method requires the value of {self.metric_key} in metrics "
                    "to have the same size as data."
                )
            return func(self, metrics, data, *args, **kwargs)

        return wrapper


class Guardrail(BaseTool):
    """Base class for guardrails.

    Each Guardrail is designed to be a LangChain "tool".

    To implement a new guardrail:
    1. Add the guardrail config/spec as class properties (similar to the ``name`` property).
    2. Override the ``compute()`` and ``moderate()`` methods as needed.

    When guardrail is initialized, the YAML spec will be loaded as keyword arguments,
    and save into the class attributes, this is handled by ``pydantic``.

    The ``Input`` to the guardrail could be any data types you would like to support,
    the data may be wrapped by a ``GuardrailIO`` object when processed by GuardrailSequence.
    The ``GuardrailIO`` object is used by the GuardrailSequence internally and track the metrics.
    Normally your guardrail implementation do not need to handle the ``GuardrailIO`` object.

    After preprocessing, the input will be passed into the ``compute()`` and ``moderate()`` methods
    in the following ways:
    1. If the input is a dict, it will be passed as ``**kwargs``. You may have ``data`` as keys in the dict.
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

    """

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    name: str = ""
    description: str = "Guardrail"

    custom_msg: str = "Content blocked by guardrail."
    raise_exception: bool = True
    return_metrics: bool = False

    _SELECT_OPERATOR = {"min": min, "max": max}
    _FILTER_OPERATOR = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }

    select: str = None
    """The method to select the best candidate. Should be one of the keys in ``_SELECT_OPERATOR``
    This is used by the ``apply_select()`` method.
    """

    direction: str = "<"
    """The operator for filtering the candidates. Should be one of the keys in the ``_FILTER_OPERATOR``
    This is used by the ``apply_filter()`` method.
    """

    threshold: float = None
    """The threshold for filtering the candidates.
    This is used by the ``apply_filter()`` method.
    """

    @root_validator
    def default_name(cls, values):
        """Sets the default name of the guardrail."""
        if not values.get("name"):
            values["name"] = cls.__name__
        return values

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """This class is LangChain serializable."""
        return True

    def _preprocess(self, input: Any) -> str:
        if isinstance(input, PromptValue):
            return input.to_string()
        return str(input)

    def _to_args_and_kwargs(self, tool_input: Any) -> Tuple[Tuple, Dict]:
        if isinstance(tool_input, dict):
            return (), tool_input
        else:
            return (tool_input,), {}

    def _run(self, query: Any, run_manager=None) -> Any:
        """Runs the guardrail.

        The parameters and metrics of running the guardrail are stored in a RunInfo object.
        If the ``query`` (input) is a GuardrailIO object, the RunInfo object is saved into it.
        Otherwise the RunInfo object is discarded.

        """
        if isinstance(query, GuardrailIO):
            guardrail_io = query
            query = guardrail_io.data
        else:
            guardrail_io = None
        # In this default implementation, we convert all input to list.
        # You may want to override the logic here for your customized guardrail.

        if isinstance(query, list):
            data = [self._preprocess(q) for q in query]
            kwargs = {}
        elif isinstance(query, dict):
            data = None
            kwargs = query
        else:
            data = [self._preprocess(query)]
            kwargs = {}

        with RunInfo(name=self.name, input=query) as info:
            # Runnable has the to_json() method to return a dictionary
            # containing the ``kwargs`` used to initialize the object.
            # The ``kwargs`` does not contain the defaults.
            # Here the ``dict()`` method is used to return a dictionary containing the defaults.
            info.parameters = {
                "class": self.__class__.__name__,
                "path": self.__module__,
                "spec": self.dict(),
            }
            info.metrics = self.compute(data, **kwargs)
            info.output = self.moderate(info.metrics, data, **kwargs)

        # Raise exception if there is no output after moderation (the content is blocked by guardrail).
        if not info.output:
            if self.raise_exception:
                raise BlockedByGuardrail(self.custom_msg, info=info)
            else:
                info.output = [self.custom_msg]

        # Return the element instead of a list if the input is not list/dict
        if (
            not isinstance(query, list)
            and not isinstance(query, dict)
            and isinstance(info.output, list)
            and len(info.output) == 1
        ):
            output = info.output[0]
        else:
            output = info.output

        # Return GuardrailIO with RunInfo appended if the input is GuardrailIO.
        if guardrail_io is not None:
            guardrail_io.data = output
            guardrail_io.info.append(info)
            return guardrail_io

        if self.return_metrics:
            return {"output": output, "metrics": info.metrics}
        return output

    def compute(self, data=None, **kwargs) -> dict:
        """Computes the metrics and returns a dictionary."""
        return {}

    def moderate(self, metrics: dict, data=None, **kwargs) -> List[str]:
        """Checks the metrics and see if the data can pass the guardrail.
        Returns the moderated data as needed.
        """
        if self.metric_key in metrics:
            return self.single_metric_moderate(metrics, data, **kwargs)
        return data

    @property
    def metric_key(self):
        """Returns the key in the metrics dictionary returned by ``compute()``.
        The default methods for selecting and filtering candidates can be only be applied based on a single metric.
        The value corresponding to the key should be a list of numerical scores corresponding to the input data.
        The scores will be used for selecting and filtering the candidates.
        By default, this property will return the class name.
        The implementation of the guardrail should override this property as needed.
        If the key is not found in the metrics,
        the default apply_select() and apply_filter method will raise an error when called.
        """
        return self.__class__.__name__

    @SingleMetric.check
    def apply_select(self, metrics: dict, data: list):
        """Selects a candidate from the data using the method specified by the ``select`` property.

        Parameters
        ----------
        metrics : dict
            The metrics returned by ``compute()``.
        data : list
            A list of candidates.

        Returns
        -------
        list
            The selected candidate in a list.

        Raises
        ------
        ValueError
            If the method specified by the ``select`` property is not supported.
        """
        if self.select not in self._SELECT_OPERATOR:
            raise ValueError(f"select='{self.select}' is not supported.")
        if not data:
            return data
        func = self._SELECT_OPERATOR[self.select]
        values = metrics[self.metric_key]
        idx = values.index(func(values))
        metrics["selected"] = idx
        data = [data[idx]]
        return data

    @SingleMetric.check
    def apply_filter(self, metrics: dict, data: list):
        """Filters the data by certain threshold.

        Parameters
        ----------
        metrics : dict
            The metrics returned by ``compute()``.
        data : list
            A list of candidates.

        Returns
        -------
        list
            The filtered data.

        Raises
        ------
        ValueError
            If the operator specified by the ``direction`` property is not supported.
        """
        if self.direction not in self._FILTER_OPERATOR:
            raise ValueError(f"direction='{self.direction}' is not supported.")
        operation = self._FILTER_OPERATOR[self.direction]
        values = metrics[self.metric_key]
        passed = [operation(val, self.threshold) for val in values]
        metrics["passed"] = passed
        return [data[i] for i in range(len(passed)) if passed[i]]

    def filter_and_select(self, metrics: dict, data: list):
        """Filters the data and select a candidate.

        Parameters
        ----------
        metrics : dict
            The metrics returned by ``compute()``.
        data : list
            A list of candidates.

        Returns
        -------
        list
            The selected candidate in a list.
        """
        filtered_data = self.apply_filter(metrics, data)
        passed_idx = [i for i in range(len(metrics["passed"])) if metrics["passed"][i]]
        filtered_metrics = {
            self.metric_key: [metrics[self.metric_key][i] for i in passed_idx]
        }
        return self.apply_select(filtered_metrics, filtered_data)

    def single_metric_moderate(self, metrics: dict, data=None, **kwargs) -> List[str]:
        """Applies moderation (filter and/or select) using the metrics."""
        if self.select and self.threshold is not None:
            return self.filter_and_select(metrics, data)
        elif self.select:
            return self.apply_select(metrics, data)
        elif self.threshold is not None:
            return self.apply_filter(metrics, data)
        return data


class CustomGuardrailBase(Guardrail):
    """Base class for custom guardrail."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """This class is not LangChain serializable."""
        return False

    @staticmethod
    def load_class_from_file(uri: str, class_name: str):
        """Loads a Python class from a file."""
        # TODO: Support loading from OCI object storage
        module_name = uri
        module_spec = importlib.util.spec_from_file_location(module_name, uri)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return getattr(module, class_name)

    @staticmethod
    def type() -> str:
        """A unique string as identifier to the type of the object for serialization."""
        return "ads_custom_guardrail"

    @staticmethod
    def load(config, **kwargs):
        """Loads the object from serialized config."""
        guardrail_class = CustomGuardrailBase.load_class_from_file(
            config["module"], config["class"]
        )
        return guardrail_class(**config["spec"])

    def save(self) -> dict:
        """Serialize the object into a dictionary."""
        return {
            "_type": self.type(),
            "module": self.__module__,
            "class": self.__class__.__name__,
            "spec": self.dict(),
        }
