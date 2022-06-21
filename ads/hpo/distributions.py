#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)

from typing import Any, Dict, Sequence, Union

CategoricalChoiceType = Union[None, bool, int, float, str]


class Distribution:
    """Defines the abstract base class for hyperparameter search distributions"""

    def __init__(self, dist):
        self._dist = dist

    def get_distribution(self):
        """Returns the distribution"""
        return self._dist

    def __repr__(self) -> str:

        kwargs = ", ".join(
            "{}={}".format(k, v)
            for k, v in sorted(self.__dict__.items(), reverse=True)
            if k != "_dist"
        )

        return "{}({})".format(self.__class__.__name__, kwargs)


class DiscreteUniformDistribution(Distribution):
    """
    A discretized uniform distribution in the linear domain.

    .. note::
        If the range :math:`[\\mathsf{low}, \\mathsf{high}]` is not divisible by :math:`q`,
        :math:`\\mathsf{high}` will be replaced with the maximum of :math:`k q + \\mathsf{low}
        \\lt \\mathsf{high}`, where :math:`k` is an integer.

    Parameters
    ----------
        low: float
            Lower endpoint of the range of the distribution. `low` is included in the range.
        high: float
            Upper endpoint of the range of the distribution. `high` is included in the range.
        step: float
            A discretization step.
    """

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def __init__(self, low: float, high: float, step: float):
        self.low = low
        self.high = high
        self.q = step
        dist = optuna.distributions.DiscreteUniformDistribution(
            low=low, high=high, q=step
        )
        super().__init__(dist)

    def __repr__(self) -> str:

        key_value_pairs = []
        for key in ["low", "high", "q"]:
            if key == "q":
                key_value_pairs.append("{}={}".format("step", self.__dict__[key]))
            else:
                key_value_pairs.append("{}={}".format(key, self.__dict__[key]))

        kwargs = ", ".join(key_value_pairs)

        return "{}({})".format(self.__class__.__name__, kwargs)


class CategoricalDistribution(Distribution):
    """
    A categorical distribution.

    Parameters
    ----------
    choices:
        Parameter value candidates. It is recommended to restrict the types of the choices
        to the following: :obj:`None`, :class:`bool`, :class:`int`, :class:`float`
        and :class:`str`.
    """

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def __init__(self, choices: Sequence[CategoricalChoiceType]):
        self.choices = choices
        dist = optuna.distributions.CategoricalDistribution(choices=choices)
        super().__init__(dist)


class IntLogUniformDistribution(Distribution):
    """A uniform distribution on integers in the log domain.

    Parameters
    ----------
        low:
            Lower endpoint of the range of the distribution. `low` is included in the range.
        high:
            Upper endpoint of the range of the distribution. `high` is included in the range.
        step:
            A step for spacing between values.
    """

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def __init__(self, low: float, high: float, step: float = 1):
        self.low = low
        self.high = high
        self.step = step
        dist = optuna.distributions.IntLogUniformDistribution(
            low=low, high=high, step=step
        )
        super().__init__(dist)

    def __repr__(self) -> str:
        key_value_pairs = []
        for key in ["low", "high", "step"]:
            key_value_pairs.append("{}={}".format(key, self.__dict__[key]))
        kwargs = ", ".join(key_value_pairs)

        return "{}({})".format(self.__class__.__name__, kwargs)


class IntUniformDistribution(Distribution):
    """
    A uniform distribution on integers.

    .. note::
        If the range :math:`[\\mathsf{low}, \\mathsf{high}]` is not divisible by
        :math:`\\mathsf{step}`, :math:`\\mathsf{high}` will be replaced with the maximum of
        :math:`k \\times \\mathsf{step} + \\mathsf{low} \\lt \\mathsf{high}`, where :math:`k` is
        an integer.

    Parameters
    ----------
        low:
            Lower endpoint of the range of the distribution. `low` is included in the range.
        high:
            Upper endpoint of the range of the distribution. `high` is included in the range.
        step:
            A step for spacing between values.
    """

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def __init__(self, low: float, high: float, step: float = 1):
        self.low = low
        self.high = high
        self.step = step
        dist = optuna.distributions.IntUniformDistribution(
            low=low, high=high, step=step
        )
        super().__init__(dist)

    def __repr__(self) -> str:
        key_value_pairs = []
        for key in ["low", "high", "step"]:
            key_value_pairs.append("{}={}".format(key, self.__dict__[key]))
        kwargs = ", ".join(key_value_pairs)

        return "{}({})".format(self.__class__.__name__, kwargs)


class LogUniformDistribution(Distribution):
    """
    A uniform distribution in the log domain.

    Parameters
    ----------
        low:
            Lower endpoint of the range of the distribution. `low` is included in the range.
        high:
            Upper endpoint of the range of the distribution. `high` is excluded from the range.
    """

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

        dist = optuna.distributions.LogUniformDistribution(low=low, high=high)
        super().__init__(dist)


class UniformDistribution(Distribution):
    """
    A uniform distribution in the linear domain.

    Parameters
    ----------
        low:
            Lower endpoint of the range of the distribution. `low` is included in the range.
        high:
            Upper endpoint of the range of the distribution. `high` is excluded from the range.
    """

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

        dist = optuna.distributions.UniformDistribution(low=low, high=high)
        super().__init__(dist)


class DistributionEncode(json.JSONEncoder):
    def default(self, dist: Distribution) -> Dict[str, Any]:
        if isinstance(dist, DiscreteUniformDistribution):
            return {
                "ClassName": DiscreteUniformDistribution.__name__,
                "low": dist.low,
                "high": dist.high,
                "q": dist.q,
            }
        if isinstance(dist, CategoricalDistribution):
            return {
                "ClassName": CategoricalDistribution.__name__,
                "choices": dist.choices,
            }
        if isinstance(dist, IntLogUniformDistribution):
            return {
                "ClassName": IntLogUniformDistribution.__name__,
                "low": dist.low,
                "high": dist.high,
                "step": dist.step,
            }
        if isinstance(dist, IntUniformDistribution):
            return {
                "ClassName": IntUniformDistribution.__name__,
                "low": dist.low,
                "high": dist.high,
                "step": dist.step,
            }
        if isinstance(dist, LogUniformDistribution):
            return {
                "ClassName": LogUniformDistribution.__name__,
                "low": dist.low,
                "high": dist.high,
            }
        if isinstance(dist, UniformDistribution):
            return {
                "ClassName": UniformDistribution.__name__,
                "low": dist.low,
                "high": dist.high,
            }
        return json.JSONEncoder.default(self, dist)

    @staticmethod
    def from_json(json_object: Dict[Any, Any]):
        if "ClassName" in json_object.keys():
            if json_object["ClassName"] == DiscreteUniformDistribution.__name__:
                return DiscreteUniformDistribution(
                    low=json_object["low"],
                    high=json_object["high"],
                    step=json_object["step"],
                )

            if json_object["ClassName"] == CategoricalDistribution.__name__:
                return CategoricalDistribution(choices=json_object["choices"])

            if json_object["ClassName"] == IntLogUniformDistribution.__name__:
                return IntLogUniformDistribution(
                    low=json_object["low"],
                    high=json_object["high"],
                    step=json_object["step"],
                )

            if json_object["ClassName"] == IntUniformDistribution.__name__:
                return IntUniformDistribution(
                    low=json_object["low"],
                    high=json_object["high"],
                    step=json_object["step"],
                )

            if json_object["ClassName"] == LogUniformDistribution.__name__:
                return LogUniformDistribution(
                    low=json_object["low"], high=json_object["high"]
                )

            if json_object["ClassName"] == UniformDistribution.__name__:
                return UniformDistribution(
                    low=json_object["low"], high=json_object["high"]
                )
        else:
            return json_object


def encode(o: Distribution) -> str:
    """Encodes a distribution to a string

    Parameters
    ----------
        o: :class:`Distribution`
            The distribution to encode

    Returns
    -------
        str (:class:`DistributionEncode`)
            The distribution encoded as a string
    """
    return json.dumps(o, cls=DistributionEncode)


def decode(s: str):
    """Decodes a string to an object

    Parameters
    ----------
        s: str
            The string being decoded to a distribution object

    Returns
    -------
        :class:`Distribution` or :class:`Dict`
            Decoded string
    """
    return json.loads(s, object_hook=DistributionEncode.from_json)
