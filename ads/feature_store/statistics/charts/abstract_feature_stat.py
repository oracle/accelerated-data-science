#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from abc import abstractmethod
from typing import Union

from ads.common.decorator.runtime_dependency import OptionalDependency

try:
    from plotly.graph_objs import Figure
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `plotly` module was not found. Please run `pip install "
        f"{OptionalDependency.FEATURE_STORE}`."
    )


class AbsFeatureStat:
    class ValidationFailedException(Exception):
        def __init__(self):
            pass

    def __init__(self):
        self.__validate__()

    @abstractmethod
    def __validate__(self):
        pass

    @abstractmethod
    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        pass

    @classmethod
    @abstractmethod
    def __from_json__(cls, json_dict: dict):
        pass

    @staticmethod
    def get_x_y_str_axes(xaxis: int, yaxis: int) -> ():
        return (
            ("xaxis" + str(xaxis + 1)),
            ("yaxis" + str(yaxis + 1)),
            ("x" + str(xaxis + 1)),
            ("y" + str(yaxis + 1)),
        )

    @classmethod
    def from_json(
        cls, json_dict: dict, ignore_errors: bool = False
    ) -> Union["AbsFeatureStat", None]:
        try:
            return cls.__from_json__(json_dict=json_dict)
        except Exception as e:
            if ignore_errors:
                return None
            else:
                raise e
