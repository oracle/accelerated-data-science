from abc import abstractmethod

from ads.common.decorator.runtime_dependency import OptionalDependency

from typing import List

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


class FeatureStat:
    @abstractmethod
    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, json_dict: dict):
        pass

    @staticmethod
    def get_x_y_str_axes(xaxis: int, yaxis: int) -> ():
        return (
            ("xaxis" + str(xaxis + 1)),
            ("yaxis" + str(yaxis + 1)),
            ("x" + str(xaxis + 1)),
            ("y" + str(yaxis + 1)),
        )


class FrequencyDistribution(FeatureStat):
    CONST_FREQUENCY = "frequency"
    CONST_BINS = "bins"
    CONST_FREQUENCY_DISTRIBUTION_TITLE = "Frequency Distribution"

    def __init__(self, frequency: List, bins: List):
        self.frequency = frequency
        self.bins = bins

    @classmethod
    def from_json(cls, json_dict: dict) -> "FrequencyDistribution":
        if json_dict is not None:
            return FrequencyDistribution(
                frequency=json_dict.get(FrequencyDistribution.CONST_FREQUENCY),
                bins=json_dict.get(FrequencyDistribution.CONST_BINS),
            )
        else:
            return None

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        if (
            type(self.frequency) == list
            and type(self.bins) == list
            and 0 < len(self.frequency) == len(self.bins) > 0
        ):
            fig.add_bar(
                x=self.bins, y=self.frequency, xaxis=x_str, yaxis=y_str, name=""
            )
            fig.layout.annotations[xaxis].text = self.CONST_FREQUENCY_DISTRIBUTION_TITLE
            fig.layout[xaxis_str]["title"] = "Bins"
            fig.layout[yaxis_str]["title"] = "Frequency"


class ProbabilityDistribution(FeatureStat):
    CONST_DENSITY = "density"
    CONST_BINS = "bins"
    CONST_PROBABILITY_DISTRIBUTION_TITLE = "Probability Distribution"

    def __init__(self, density: List, bins: List):
        self.density = density
        self.bins = bins

    @classmethod
    def from_json(cls, json_dict: dict):
        if json_dict is not None:
            return cls(
                density=json_dict.get(ProbabilityDistribution.CONST_DENSITY),
                bins=json_dict.get(ProbabilityDistribution.CONST_BINS),
            )
        else:
            return None

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        if (
            type(self.density) == list
            and type(self.bins) == list
            and 0 < len(self.density) == len(self.bins) > 0
        ):
            fig.add_bar(
                x=self.bins,
                y=self.density,
                xaxis=x_str,
                yaxis=y_str,
                name="",
            )
        fig.layout.annotations[xaxis].text = self.CONST_PROBABILITY_DISTRIBUTION_TITLE
        fig.layout[xaxis_str]["title"] = "Bins"
        fig.layout[yaxis_str]["title"] = "Density"

        return go.Bar(x=self.bins, y=self.density)


class TopKFrequentElements(FeatureStat):
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
        def from_json(cls, json_dict: dict):
            if json_dict is not None:
                return cls(
                    value=json_dict.get(cls.CONST_VALUE),
                    estimate=json_dict.get(cls.CONST_ESTIMATE),
                    lower_bound=json_dict.get(cls.CONST_LOWER_BOUND),
                    upper_bound=json_dict.get(cls.CONST_UPPER_BOUND),
                )

            else:
                return None

    def __init__(self, elements: List[TopKFrequentElement]):
        self.elements = elements

    @classmethod
    def from_json(cls, json_dict: dict):
        if json_dict is not None and json_dict.get(cls.CONST_VALUE) is not None:
            elements = json_dict.get(cls.CONST_VALUE)
            return cls(
                [cls.TopKFrequentElement.from_json(element) for element in elements]
            )
        else:
            return None

    def add_to_figure(self, fig: Figure, xaxis: int, yaxis: int):
        xaxis_str, yaxis_str, x_str, y_str = self.get_x_y_str_axes(xaxis, yaxis)
        if type(self.elements) == list and len(self.elements) > 0:
            x_axis = [element.value for element in self.elements]
            y_axis = [element.estimate for element in self.elements]
            fig.add_bar(x=x_axis, y=y_axis, xaxis=x_str, yaxis=y_str, name="")
        fig.layout.annotations[xaxis].text = self.CONST_TOP_K_FREQUENT_TITLE
        fig.layout[yaxis_str]["title"] = "Count"
        fig.layout[xaxis_str]["title"] = "Element"


class FeatureStatistics:
    CONST_FREQUENCY_DISTRIBUTION = "FrequencyDistribution"
    CONST_TITLE_FORMAT = "<b>{}</b>"
    CONST_PLOT_FORMAT = "{}_plot"
    CONST_PROBABILITY_DISTRIBUTION = "ProbabilityDistribution"
    CONST_TOP_K_FREQUENT = "TopKFrequentElements"

    def __init__(
        self,
        feature_name: str,
        top_k_frequent_elements: TopKFrequentElements,
        frequency_distribution: FrequencyDistribution,
        probability_distribution: ProbabilityDistribution,
    ):
        self.feature_name: str = feature_name
        self.top_k_frequent_elements = top_k_frequent_elements
        self.frequency_distribution = frequency_distribution
        self.probability_distribution = probability_distribution

    @classmethod
    def from_json(cls, feature_name: str, json_dict: dict) -> "FeatureStatistics":
        if json_dict is not None:
            return cls(
                feature_name,
                TopKFrequentElements.from_json(json_dict.get(cls.CONST_TOP_K_FREQUENT)),
                FrequencyDistribution.from_json(
                    json_dict.get(cls.CONST_FREQUENCY_DISTRIBUTION)
                ),
                ProbabilityDistribution.from_json(
                    json_dict.get(cls.CONST_PROBABILITY_DISTRIBUTION)
                ),
            )
        else:
            return None

    @property
    def __stat_count__(self):
        graph_count = 0
        if self.top_k_frequent_elements is not None:
            graph_count += 1
        if self.probability_distribution is not None:
            graph_count += 1
        if self.frequency_distribution is not None:
            graph_count += 1
        return graph_count

    @property
    def __feature_stat_objects__(self) -> List[FeatureStat]:
        return [
            stat
            for stat in [
                self.top_k_frequent_elements,
                self.frequency_distribution,
                self.probability_distribution,
            ]
            if stat is not None
        ]

    def to_viz(self):
        graph_count = len(self.__feature_stat_objects__)
        if graph_count > 0:
            fig = make_subplots(cols=graph_count, column_titles=["title"] * graph_count)
            index = 0
            for stat in [
                stat for stat in self.__feature_stat_objects__ if stat is not None
            ]:
                stat.add_to_figure(fig, index, index)
                index += 1
            fig.layout.title = self.CONST_TITLE_FORMAT.format(self.feature_name)
            fig.update_layout(title_font_size=20)
            fig.update_layout(title_x=0.5)
            fig.update_layout(showlegend=False)
            plotly.offline.iplot(
                fig,
                filename=self.CONST_PLOT_FORMAT.format(self.feature_name),
            )
