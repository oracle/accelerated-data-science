#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.decorator.runtime_dependency import OptionalDependency
from typing import List
from ads.feature_store.statistics.charts.abstract_feature_plot import AbsFeaturePlot
from ads.feature_store.statistics.charts.box_plot import BoxPlot
from ads.feature_store.statistics.charts.frequency_distribution import (
    FrequencyDistribution,
)
from ads.feature_store.statistics.charts.probability_distribution import (
    ProbabilityDistribution,
)
from ads.feature_store.statistics.charts.top_k_frequent_elements import (
    TopKFrequentElements,
)

try:
    import plotly
    from plotly.graph_objs import Figure
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `plotly` module was not found. Please run `pip install "
        f"{OptionalDependency.FEATURE_STORE}`."
    )


class FeatureStatistics:
    CONST_FREQUENCY_DISTRIBUTION = "FrequencyDistribution"
    CONST_TITLE_FORMAT = "<b>{}</b>"
    CONST_PLOT_FORMAT = "{}_plot"
    CONST_PROBABILITY_DISTRIBUTION = "ProbabilityDistribution"
    CONST_TOP_K_FREQUENT = "TopKFrequentElements"

    def __init__(
        self,
        feature_name: str,
        top_k_frequent_elements: TopKFrequentElements = None,
        frequency_distribution: FrequencyDistribution = None,
        probability_distribution: ProbabilityDistribution = None,
        box_plot: BoxPlot = None,
    ):
        self.feature_name: str = feature_name
        self.top_k_frequent_elements = top_k_frequent_elements
        self.frequency_distribution = frequency_distribution
        self.probability_distribution = probability_distribution
        self.box_plot = box_plot

    @classmethod
    def from_json(
        cls, feature_name: str, json_dict: dict, version: int
    ) -> "FeatureStatistics":
        if json_dict is not None:
            frequency_distribution = FrequencyDistribution.from_json(
                json_dict.get(cls.CONST_FREQUENCY_DISTRIBUTION),
                version,
                ignore_errors=True,
            )

            # inject box points for boxplot creation
            json_dict[
                BoxPlot.CONST_BOX_POINTS
            ] = BoxPlot.get_box_points_from_frequency_distribution(
                frequency_distribution
            )
            return cls(
                feature_name,
                TopKFrequentElements.from_json(
                    json_dict.get(cls.CONST_TOP_K_FREQUENT), version, ignore_errors=True
                ),
                frequency_distribution,
                ProbabilityDistribution.from_json(
                    json_dict.get(cls.CONST_PROBABILITY_DISTRIBUTION),
                    version,
                    ignore_errors=True,
                ),
                BoxPlot.from_json(json_dict, version, ignore_errors=True),
            )
        else:
            return cls(feature_name)

    @property
    def __feature_stat_objects__(self) -> List[AbsFeaturePlot]:
        return [
            stat
            for stat in [
                self.box_plot,
                self.top_k_frequent_elements,
                self.frequency_distribution,
                self.probability_distribution,
            ]
            if stat is not None
        ]

    def to_viz(self):
        # TODO: make it generic
        def next_graph_position_generator():
            yield 1
            yield 0
            yield 2

        if len(self.__feature_stat_objects__) > 0:
            fig = make_subplots(cols=3, column_titles=[" "] * 3)
            for idx, stat in zip(
                next_graph_position_generator(), self.__feature_stat_objects__
            ):
                stat.add_to_figure(fig, idx, idx)

            fig.layout.title = self.CONST_TITLE_FORMAT.format(self.feature_name)
            fig.update_layout(title_font_size=20)
            # Center align the title
            fig.update_layout(title_x=0.5)
            # Disable legend for unrelated plots
            fig.update_layout(showlegend=False)
            plotly.offline.iplot(
                fig,
                filename=self.CONST_PLOT_FORMAT.format(self.feature_name),
            )
        else:
            print(
                f"No statistical information for feature {self.feature_name} can be visualised"
            )
