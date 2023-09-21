#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from typing import List
from ads.common.decorator.runtime_dependency import OptionalDependency

from ads.feature_store.statistics.charts.abstract_feature_stat import AbsFeatureStat

try:
    from plotly.graph_objs import Figure
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `plotly` module was not found. Please run `pip install "
        f"{OptionalDependency.FEATURE_STORE}`."
    )


class TopKFrequentElements(AbsFeatureStat):
    def __validate__(self):
        if not (type(self.elements) == list and len(self.elements) > 0):
            raise self.ValidationFailedException

    CONST_VALUE = "value"
    CONST_TOP_K_FREQUENT_TITLE = "Top K Frequent Elements"

    class TopKFrequentElement:
        CONST_VALUE = "value"
        CONST_ESTIMATE = "estimate"
        CONST_LOWER_BOUND = "lower_bound"
        CONST_UPPER_BOUND = "upper_bound"

        def __init__(
            self, value: str, estimate: int, lower_bound: int, upper_bound: int
        ):
            self.value = value
            self.estimate = estimate
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        @classmethod
        def from_json(
            cls, json_dict: dict
        ) -> "TopKFrequentElements.TopKFrequentElement":
            return cls(
                value=json_dict.get(cls.CONST_VALUE),
                estimate=json_dict.get(cls.CONST_ESTIMATE),
                lower_bound=json_dict.get(cls.CONST_LOWER_BOUND),
                upper_bound=json_dict.get(cls.CONST_UPPER_BOUND),
            )

    def __init__(self, elements: List[TopKFrequentElement]):
        self.elements = elements
        super().__init__()

    @classmethod
    def __from_json__(cls, json_dict: dict) -> "TopKFrequentElements":
        elements = json_dict.get(cls.CONST_VALUE)
        return cls([cls.TopKFrequentElement.from_json(element) for element in elements])

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        if type(self.elements) == list and len(self.elements) > 0:
            y_axis = [element.value for element in self.elements]
            x_axis = [element.estimate for element in self.elements]
            fig.add_bar(
                x=x_axis, y=y_axis, xaxis=x_str, yaxis=y_str, name="", orientation="h"
            )
        fig.layout.annotations[xaxis].text = self.CONST_TOP_K_FREQUENT_TITLE
        fig.layout[yaxis_str]["title"] = "Element"
        fig.layout[xaxis_str]["title"] = "Count"
