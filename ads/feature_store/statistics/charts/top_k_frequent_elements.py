#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from typing import List

from ads.feature_store.statistics.abs_feature_value import AbsFeatureValue

from ads.common.decorator.runtime_dependency import OptionalDependency

from ads.feature_store.statistics.charts.abstract_feature_plot import AbsFeaturePlot

try:
    from plotly.graph_objs import Figure
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `plotly` module was not found. Please run `pip install "
        f"{OptionalDependency.FEATURE_STORE}`."
    )


class TopKFrequentElements(AbsFeaturePlot):
    CONST_VALUE = "value"
    CONST_TOP_K_FREQUENT_TITLE = "Top K Frequent Elements"

    class TopKFrequentElement(AbsFeatureValue):
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
            super().__init__()

        def __validate__(self):
            assert type(self.value) == str and len(self.value) > 0
            assert type(self.estimate) == int and self.estimate >= 0

        @classmethod
        def __from_json__(
            cls, json_dict: dict, version: int = 1
        ) -> "TopKFrequentElements.TopKFrequentElement":
            return cls(
                value=json_dict.get(cls.CONST_VALUE),
                estimate=json_dict.get(cls.CONST_ESTIMATE),
                lower_bound=json_dict.get(cls.CONST_LOWER_BOUND),
                upper_bound=json_dict.get(cls.CONST_UPPER_BOUND),
            )

    def __init__(self, values: List, estimates: List):
        self.values = values
        self.estimates = estimates
        super().__init__()

    def __validate__(self):
        assert type(self.values) == list
        assert type(self.estimates) == list
        assert 0 < len(self.values) == len(self.estimates) > 0

    @classmethod
    def __from_json__(cls, json_dict: dict, version: int = 1) -> "TopKFrequentElements":
        if version == 2:
            return cls.__from_json_v2__(json_dict)
        elements = json_dict.get(cls.CONST_VALUE)
        top_k_frequent_elements = [
            cls.TopKFrequentElement.__from_json__(element) for element in elements
        ]
        values = [element.value for element in top_k_frequent_elements]
        estimates = [element.estimate for element in top_k_frequent_elements]
        return cls(values, estimates)

    @classmethod
    def __from_json_v2__(cls, json_dict: dict) -> "TopKFrequentElements":
        metric_data = json_dict.get(AbsFeatureValue.CONST_METRIC_DATA)
        return cls(values=metric_data[0], estimates=metric_data[1])

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        if (
            type(self.values) == list
            and len(self.values) > 0
            and type(self.estimates) == list
            and len(self.estimates) > 0
        ):
            y_axis = [value for value in self.values]
            x_axis = [estimate for estimate in self.estimates]
            fig.add_bar(
                x=x_axis, y=y_axis, xaxis=x_str, yaxis=y_str, name="", orientation="h"
            )
        fig.layout.annotations[xaxis].text = self.CONST_TOP_K_FREQUENT_TITLE
        fig.layout[yaxis_str]["title"] = "Element"
        fig.layout[xaxis_str]["title"] = "Count"
