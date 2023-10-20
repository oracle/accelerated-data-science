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


class AbsFeatureValue:
    CONST_METRIC_DATA = "metric_data"

    def __init__(self):
        self.__validate__()

    @abstractmethod
    def __validate__(self):
        pass

    @classmethod
    @abstractmethod
    def __from_json__(cls, json_dict: dict, version: int):
        pass

    @classmethod
    @abstractmethod
    def __from_json_v2__(cls, json_dict: dict):
        pass

    @classmethod
    def from_json(
        cls, json_dict: dict, version: int, ignore_errors: bool = False
    ) -> Union["AbsFeatureValue", None]:
        try:
            return cls.__from_json__(json_dict=json_dict, version=version)
        except Exception as e:
            if ignore_errors:
                return None
            else:
                raise e
