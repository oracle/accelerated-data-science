import matplotlib.figure
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

from ads.feature_store.response.response_builder import ResponseBuilder

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json


def add_plots_for_stat(fig: Figure, feature: str, stat: dict):
    freq_dist = stat.get(Statistics.CONST_FREQUENCY_DISTRIBUTION)
    top_k = stat.get(Statistics.CONST_TOP_K_FREQUENT)
    probability = stat.get(Statistics.CONST_PROBABILITY_DISTRIBUTION)

    def subplot_generator():
        plot_count = 0
        if stat.get(Statistics.CONST_FREQUENCY_DISTRIBUTION) is not None:
            plot_count += 1
        # if stat.get(Statistics.CONST_TOP_K_FREQUENT) is not None:
        #     plot_count += 1
        if stat.get(Statistics.CONST_PROBABILITY_DISTRIBUTION) is not None:
            plot_count += 1

        for i in range(0, plot_count):
            yield fig.add_subplot(1, plot_count, i + 1)

    subplots = subplot_generator()
    if freq_dist is not None:
        axs = next(subplots)
        fig.suptitle(feature, fontproperties=fm.FontProperties(weight="bold"))
        axs.hist(
            x=freq_dist.get("bins"),
            weights=freq_dist.get("frequency"),
            cumulative=False,
            color="teal",
            mouseover=True,
            animated=True,
        )

        axs.set_xlabel(
            "Bins", fontdict={"fontproperties": fm.FontProperties(size="xx-small")}
        )
        axs.set_ylabel(
            "Frequency", fontdict={"fontproperties": fm.FontProperties(size="xx-small")}
        )
        axs.set_title(
            "Frequency Distribution",
            fontdict={"fontproperties": fm.FontProperties(size="small")},
        )
        axs.set_xticks(freq_dist.get("bins"))
    if probability is not None:
        axs = next(subplots)
        fig.suptitle(feature, fontproperties=fm.FontProperties(weight="bold"))
        axs.bar(
            probability.get("bins"),
            probability.get("density"),
            color="teal",
            mouseover=True,
            animated=True,
        )
        axs.set_xlabel(
            "Bins", fontdict={"fontproperties": fm.FontProperties(size="xx-small")}
        )
        axs.set_ylabel(
            "Density", fontdict={"fontproperties": fm.FontProperties(size="xx-small")}
        )
        axs.set_title(
            "Probability Distribution",
            fontdict={"fontproperties": fm.FontProperties(size="small")},
        )
        axs.set_xticks(probability.get("bins"))


def subfigure_generator(count: int, fig: Figure):
    rows = count
    subfigs = fig.subfigures(rows, 1)
    for i in range(0, rows):
        yield subfigs[i]


class Statistics(ResponseBuilder):
    """
    Represents statistical information.
    """

    CONST_FREQUENCY_DISTRIBUTION = "FrequencyDistribution"
    CONST_PROBABILITY_DISTRIBUTION = "ProbabilityDistribution"
    CONST_TOP_K_FREQUENT = "TopKFrequentElements"

    @property
    def kind(self) -> str:
        """
        Gets the kind of the statistics object.

        Returns
        -------
        str
            The kind of the statistics object, which is always "statistics".
        """
        return "statistics"

    def to_viz(self):
        if self.content is not None:
            stats: dict = json.loads(self.content)
            fig: Figure = plt.figure(figsize=(20, 20), dpi=150)
            plt.subplots_adjust(hspace=3)

            stats = {
                feature: stat
                for feature, stat in stats.items()
                if Statistics.__graph_exists__(stat)
            }
            subfigures = subfigure_generator(len(stats), fig)
            for feature, stat in stats.items():
                sub_figure = next(subfigures)
                add_plots_for_stat(sub_figure, feature, stat)

    @staticmethod
    def __graph_exists__(stat: dict):
        return (
            stat.get(Statistics.CONST_FREQUENCY_DISTRIBUTION) != None
            or stat.get(Statistics.CONST_PROBABILITY_DISTRIBUTION) != None
        )
