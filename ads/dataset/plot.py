#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import random
from collections import defaultdict
from math import pi
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from ads.dataset.helper import _log_yscale_not_set

from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.utils import _log_plot_high_cardinality_warning, MAX_DISPLAY_VALUES
from ads.type_discovery.latlon_detector import LatLonDetector
from ads.type_discovery.typed_feature import (
    ContinuousTypedFeature,
    DateTimeTypedFeature,
    ConstantTypedFeature,
    DiscreteTypedFeature,
    CreditCardTypedFeature,
    ZipcodeTypedFeature,
    OrdinalTypedFeature,
    CategoricalTypedFeature,
    GISTypedFeature,
)

from ads.dataset import logger


class Plotting:
    def __init__(self, df, feature_types, x, y=None, plot_type="infer", yscale=None):
        self.df = df
        self.feature_types = feature_types
        self.x = x
        self.y = y
        self.x_type = self.feature_types[self.x]
        self.y_type = self.feature_types[self.y] if self.y is not None else None
        self.plot_type = plot_type
        self.yscale = yscale

    def __repr__(self):
        choices = self._get_plot_method()  # add (plot_type='{0}') in plot method"
        if len(choices) > 1:
            logger.info(f"Recommended plot type is {choices[0][1].__name__}.")

            logger.info(
                "Available plot types are ",
                ", ".join([x[1].__name__ for x in choices]),
                ".",
            )
        self.show_in_notebook()
        return ""

    def select_best_plot(self):
        """
        Returns the best plot for a given dataset
        """

        #
        # auto logic
        #

        choices = self._get_plot_method()

        if len(choices) > 1:
            logger.info(
                "select_best_plot (%s, %s) called, possible plot types are %s"
                % (
                    self.x_type.meta_data["type"],
                    self.y_type.meta_data["type"] if self.y_type is not None else "",
                    ", ".join([x[0].__name__ for x in choices]),
                )
            )

        if self.plot_type != "infer":
            for choice in choices:
                if choice[1].__name__.lower().startswith(self.plot_type.lower()):
                    return choice
            logger.info("invalid plot_type: {}".format(self.plot_type))
            raise ValueError(
                "plot_type: '%s' invalid, use one of: %s"
                % (self.plot_type, ", ".join([x[0].__name__ for x in choices]))
            )

        return choices[0]

    def show_in_notebook(self, **kwargs):

        """
        Visualizes the dataset by plotting the distribution of a feature or relationship between two features.

        Parameters
        ----------
        figsize: tuple
                defines the size of the fig
        -------
        """
        plotlib_type, plot_method, plot_kwargs = self.select_best_plot()
        plotlib_type(plot_method, **plot_kwargs, **kwargs)

    @staticmethod
    def _add_identity(ax, *line_args, **line_kwargs):
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="-.3", transform=ax.transAxes)

    @staticmethod
    def _build_plot_key(x_type, y_type=None):
        if y_type is None:
            return x_type.__name__
        return x_type.__name__ + "," + y_type.__name__

    @staticmethod
    @runtime_dependency(module="scipy", install_from=OptionalDependency.VIZ)
    def _gaussian_heatmap(x, y, data, s=10, edgecolor="white", cmap=plt.cm.jet):
        """
        Generate a scatter plot and assign a color to each data point based on the local density (gaussian kernel) of
        points.

        Parameters
        ----------
        x: str
            name of the feature
        y: str
            name of the feature
        data: object
            The dataframe that contains x and y
        s: int
            area of each marker
        edgecolor: str
            edge color of each point. string value, e.g. 'blue'
        cmap: object
            color map for the heatmap

        Returns
        -------
        ax.scatter() object: a scatter plot with colored density

        Raises
        ------
        ValueError
            When the columns are identical or the columns are highly correlated.
        """
        try:
            _x = np.array(data[x])
            _y = np.array(data[y])
            xy = np.vstack([_x, _y])
            z = scipy.stats.gaussian_kde(xy)(xy)
            sc = plt.scatter(_x, _y, c=z, s=s, edgecolor=edgecolor, cmap=cmap)
            plt.xlabel(x)
            plt.ylabel(y)
            return plt.colorbar(sc)
        except:
            return plt.scatter(_x, _y)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _categorical_vs_continuous_violin_plot(x, y, data):
        # when x is categorical, we set the order based on counts of each category
        vc = data[x].value_counts()
        if len(vc.keys()) > 10:
            _log_plot_high_cardinality_warning(x, len(vc.keys()))
            idxes = vc[:10].index
        else:
            idxes = vc.index
        seaborn.violinplot(x=x, y=y, data=data[data[x].isin(idxes)], order=idxes)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _ordinal_vs_continuous_violin_plot(x, y, data):
        # when x is ordinal, we want to get a natural order to x values
        vals = np.array(data[x].values)
        sorted_x = list(np.sort(vals))
        # get the frequency of each distinct element in the list using a dictionary.
        freq = {}
        for items in sorted_x:
            freq[items] = sorted_x.count(items)
        if len(freq) > MAX_DISPLAY_VALUES:
            _log_plot_high_cardinality_warning(x, len(freq))
            idxes = list(freq.keys())[:10]
        else:
            idxes = freq.keys()
        seaborn.violinplot(x=x, y=y, data=data[data[x].isin(idxes)], order=idxes)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _categorical_vs_continuous_horizontal_violin_plot(x, y, data):
        vc = data[x].value_counts()
        if len(vc.keys()) > 10:
            _log_plot_high_cardinality_warning(x, len(vc.keys()))
            idxes = vc[:10].index
        else:
            idxes = vc.index
        seaborn.violinplot(
            x=y, y=x, data=data[data[x].isin(idxes)], order=idxes, orient="h"
        )

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _ordinal_vs_continuous_horizontal_violin_plot(x, y, data):
        # when x is ordinal, we want to get a natural order to x values
        vals = np.array(data[x].values)
        sorted_x = list(np.sort(vals))
        # get the frequency of each distinct element in the list using a dictionary.
        freq = {}
        for items in sorted_x:
            freq[items] = sorted_x.count(items)
        if len(freq) > MAX_DISPLAY_VALUES:
            _log_plot_high_cardinality_warning(x, len(freq))
            idxes = list(freq.keys())[:10]
        else:
            idxes = freq.keys()
        seaborn.violinplot(
            x=y, y=x, data=data[data[x].isin(idxes)], order=idxes, orient="h"
        )

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _categorical_vs_continuous_box_plot(x, y, data):
        vc = data[x].value_counts()
        if len(vc.keys()) > 10:
            _log_plot_high_cardinality_warning(x, len(vc.keys()))
            idxes = vc[:10].index
        else:
            idxes = vc.index
        seaborn.boxplot(x=x, y=y, data=data[data[x].isin(idxes)], order=idxes)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _ordinal_vs_continuous_box_plot(x, y, data):
        # when x is ordinal, we want to get a natural order to x values
        vals = np.array(data[x].values)
        sorted_x = list(np.sort(vals))
        # get the frequency of each distinct element in the list using a dictionary.
        freq = {}
        for items in sorted_x:
            freq[items] = sorted_x.count(items)
        if len(freq) > MAX_DISPLAY_VALUES:
            _log_plot_high_cardinality_warning(x, len(freq))
            idxes = list(freq.keys())[:10]
        else:
            idxes = freq.keys()
        seaborn.boxplot(x=x, y=y, data=data[data[x].isin(idxes)], order=idxes)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _ordinal_vs_continuous_horizontal_box_plot(x, y, data):
        # when x is ordinal, we want to get a natural order to x values
        vals = np.array(data[x].values)
        sorted_x = list(np.sort(vals))
        # get the frequency of each distinct element in the list using a dictionary.
        freq = {}
        for items in sorted_x:
            freq[items] = sorted_x.count(items)
        if len(freq) > MAX_DISPLAY_VALUES:
            _log_plot_high_cardinality_warning(x, len(freq))
            idxes = list(freq.keys())[:10]
        else:
            idxes = freq.keys()
        seaborn.boxplot(
            x=y, y=x, data=data[data[x].isin(idxes)], order=idxes, orient="h"
        )

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _count_plot(x, hue, data, yscale=None):
        if not yscale:
            _log_yscale_not_set()
        # get the copy of data and convert categorical data type to object
        data = data.copy()
        data[hue] = data[hue].astype("object")
        # get cardinality of categorical values
        cat_cardi = data[hue].value_counts()
        if len(cat_cardi) > 5:
            top_categoricals = cat_cardi[:5].index
            # modify the data to replace non top 5 categorical values to be the same value
            data[hue] = np.where(
                data[hue].isin(top_categoricals), data[hue], "all_other_categories"
            )
            cat_index = data[hue].value_counts().index
        else:
            cat_index = cat_cardi.index
        # get cardinality of ordinal values
        ordi_cardi = data[x].value_counts()
        if len(ordi_cardi) > 10:
            # bin the values and sort from small to large
            data[x] = pd.cut(data[x], 10, precision=0)
            data[x] = data[x].apply(lambda k: pd.Interval(int(k.left), int(k.right)))
        g = seaborn.countplot(x=x, hue=hue, data=data, hue_order=cat_index)
        if yscale:
            g.set_yscale(yscale)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _ordinal_count_plot(x, data, yscale=None):
        if not yscale:
            _log_yscale_not_set()
        data = data.copy()
        ordi_cardi = data[x].value_counts()
        if len(ordi_cardi) > 20:
            intervals = pd.cut(data[x], 20, precision=0)
            intervals = intervals.apply(
                lambda k: pd.Interval(int(k.left), int(k.right))
            )
        else:
            intervals = data[x]
        g = seaborn.countplot(x=intervals, color="#1f77b4")
        if yscale:
            g.set_yscale(yscale)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _ordinal_vs_constant_count_plot(x, hue, data, yscale=None):
        if not yscale:
            _log_yscale_not_set()
        data = data.copy()
        intervals = pd.cut(data[x], 20, precision=0)
        intervals = intervals.apply(lambda k: pd.Interval(int(k.left), int(k.right)))
        g = seaborn.countplot(x=intervals, hue=hue, data=data)
        if yscale:
            g.set_yscale(yscale)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _single_column_count_plot(x, data, yscale=None):
        if not yscale:
            _log_yscale_not_set()
        order = data[x].value_counts().iloc[:24].index
        g = seaborn.countplot(x=x, data=data, order=order)
        if yscale:
            g.set_yscale(yscale)

    @staticmethod
    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    @runtime_dependency(module="folium", install_from=OptionalDependency.VIZ)
    def _folium_map(x, data):
        import folium.plugins
        df = LatLonDetector.extract_x_y(data[x])
        lat_min, lat_max, long_min, long_max = (
            min(df.Y),
            max(df.Y),
            min(df.X),
            max(df.X),
        )
        m = folium.Map(tiles="Stamen Terrain", zoom_control=False)

        folium.plugins.HeatMap(df[["Y", "X"]]).add_to(m)
        m.fit_bounds([[lat_min, long_min], [lat_max, long_max]])

        from IPython.core.display import display

        display(m)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _single_pdf(x, y, data):
        seaborn.kdeplot(data[x], shade=True, shade_lowest=False)

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _multiple_pdf(x, y, data):
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        hues = [
            colors[x]
            for x in colors.keys()
            if isinstance(colors[x], str) and colors[x].startswith("#")
        ]

        for i, cat in enumerate(list(data[y].unique())):
            s = data.loc[data[y] == cat][x]
            color = random.choice(hues)
            seaborn.kdeplot(s, color=color, shade=True, shade_lowest=False, label=cat)
        plt.xlabel(x)
        plt.ylabel(y)

    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _matplot(self, plot_method, figsize=(4, 3), **kwargs):

        plt.style.use("seaborn-white")

        plt.rc("xtick", labelsize="x-small")
        plt.rc("ytick", labelsize="x-small")
        plt.rc("font", size=8)
        plt.rc("figure", dpi=144)

        fig = plt.figure(figsize=figsize)

        #
        # generate a title for the plot
        #
        text = '{}, "{}" ({})'.format(
            plot_method.__name__.upper(), self.x, self.feature_types[self.x].type
        )
        if self.y:
            text = '{} vs "{}" ({})'.format(
                text, self.y, self.feature_types[self.y].type
            )

        plt.title(text, y=1.08)
        plt.grid(linestyle="dotted")

        # draw a 45 degree dotted or dashed line indicating equality when plot method is scatter plots and the span of x equals the span of y
        if (
            plot_method is plt.scatter or plot_method is Plotting._gaussian_heatmap
        ) and (self.df[kwargs["x"]].values.ptp() == self.df[kwargs["y"]].values.ptp()):
            Plotting._add_identity(fig.axes[0], color="grey", ls="--")

        plot_method(**kwargs, data=self.df)

        # set x and y axis label
        if self.y:
            plt.ylabel(self.y)
        if self.x:
            plt.xlabel(self.x)

        # rename the y-axis label and x-axis label when "count" is the y-axis label
        if self.y == "count":
            plt.xlabel("Column: {} values ".format(self.x))
            plt.ylabel("instance count")

        # add y-axis label as "count" when plot type is hist
        if plot_method is plt.hist:
            plt.ylabel("instance count")
            # add tickmark on x-axis to see labeled values on x-axis in historgram. It has 30 intervals because that's the most tickmarks on graph can fit.
            plt.xticks(
                np.arange(
                    min(self.df[kwargs["x"]].values),
                    max(self.df[kwargs["x"]].values) + 1,
                    (
                        max(self.df[kwargs["x"]].values)
                        - min(self.df[kwargs["x"]].values)
                    )
                    / 30,
                )
            )
        # override y-axis label as "count" when plot type is _count_plot or countplot
        if plot_method is Plotting._count_plot or plot_method is seaborn.countplot:
            plt.ylabel("count")

        plt.xticks(rotation=90)

    def _generic_plot(self, plot_method, **kwargs):
        plot_method(**kwargs, data=self.df)

    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def _get_plot_method(self):

        #
        # combos contains a dictionary with the key being a composite of the x and y types, the value will
        # always be a list, possibly and empty list, indicating no match for combination
        #
        #

        combos = defaultdict(list)

        combos[
            Plotting._build_plot_key(CategoricalTypedFeature, ContinuousTypedFeature)
        ] = [
            (
                self._matplot,
                Plotting._categorical_vs_continuous_violin_plot,
                {"x": self.x, "y": self.y},
            ),
            (
                self._matplot,
                Plotting._categorical_vs_continuous_box_plot,
                {"x": self.x, "y": self.y},
            ),
        ]
        combos[
            Plotting._build_plot_key(OrdinalTypedFeature, ContinuousTypedFeature)
        ] = [
            (
                self._matplot,
                Plotting._ordinal_vs_continuous_violin_plot,
                {"x": self.x, "y": self.y},
            ),
            (
                self._matplot,
                Plotting._ordinal_vs_continuous_box_plot,
                {"x": self.x, "y": self.y},
            ),
        ]
        combos[
            Plotting._build_plot_key(ContinuousTypedFeature, OrdinalTypedFeature)
        ] = [
            (
                self._matplot,
                Plotting._ordinal_vs_continuous_horizontal_violin_plot,
                {"x": self.y, "y": self.x},
            ),
            (
                self._matplot,
                Plotting._ordinal_vs_continuous_horizontal_box_plot,
                {"x": self.y, "y": self.x},
            ),
        ]
        combos[
            Plotting._build_plot_key(ContinuousTypedFeature, CategoricalTypedFeature)
        ] = [(self._matplot, Plotting._multiple_pdf, {"x": self.x, "y": self.y})]
        combos[
            Plotting._build_plot_key(ConstantTypedFeature, ContinuousTypedFeature)
        ] = [(self._matplot, seaborn.barplot, {"x": self.x, "y": self.y})]
        combos[
            Plotting._build_plot_key(ContinuousTypedFeature, ConstantTypedFeature)
        ] = [(self._matplot, Plotting._single_pdf, {"x": self.x, "y": self.y})]
        combos[Plotting._build_plot_key(ConstantTypedFeature, DiscreteTypedFeature)] = [
            (self._matplot, seaborn.barplot, {"x": self.x, "y": self.y})
        ]
        combos[Plotting._build_plot_key(DiscreteTypedFeature, ConstantTypedFeature)] = [
            (
                self._matplot,
                Plotting._ordinal_vs_constant_count_plot,
                {"x": self.x, "hue": self.y, "yscale": self.yscale},
            )
        ]
        combos[
            Plotting._build_plot_key(DateTimeTypedFeature, ContinuousTypedFeature)
        ] = [
            (
                self._matplot,
                plt.scatter,
                {
                    "x": self.x,
                    "y": self.y,
                    "s": pi / 10 * (matplotlib.rcParams["lines.markersize"] ** 2),
                    "edgecolor": "white",
                    "linewidths": "0.1",
                },
            )
        ]
        combos[Plotting._build_plot_key(DateTimeTypedFeature, OrdinalTypedFeature)] = [
            (
                self._matplot,
                plt.scatter,
                {"x": self.x, "y": self.y, "edgecolor": "white", "linewidths": "0.1"},
            )
        ]
        combos[
            Plotting._build_plot_key(ContinuousTypedFeature, ContinuousTypedFeature)
        ] = [
            (
                self._matplot,
                Plotting._gaussian_heatmap,
                {
                    "x": self.x,
                    "y": self.y,
                    "s": pi / 10 * (matplotlib.rcParams["lines.markersize"] ** 2),
                },
            ),
            (
                self._matplot,
                plt.scatter,
                {"x": self.x, "y": self.y, "edgecolor": "white", "linewidths": "0.1"},
            ),
        ]
        combos[Plotting._build_plot_key(OrdinalTypedFeature, OrdinalTypedFeature)] = [
            (
                self._matplot,
                seaborn.scatterplot,
                {
                    "x": self.x,
                    "y": self.y,
                    "s": pi / 10 * (matplotlib.rcParams["lines.markersize"] ** 2),
                    "edgecolor": "white",
                    "linewidths": "0.1",
                },
            )
        ]
        combos[Plotting._build_plot_key(OrdinalTypedFeature, DiscreteTypedFeature)] = [
            (self._matplot, seaborn.countplot, {"x": self.x, "hue": self.y})
        ]
        combos[
            Plotting._build_plot_key(OrdinalTypedFeature, CategoricalTypedFeature)
        ] = [
            (
                self._matplot,
                Plotting._count_plot,
                {"x": self.x, "hue": self.y, "yscale": self.yscale},
            )
        ]
        combos[
            Plotting._build_plot_key(CategoricalTypedFeature, OrdinalTypedFeature)
        ] = [
            (
                self._matplot,
                Plotting._count_plot,
                {"x": self.y, "hue": self.x, "yscale": self.yscale},
            )
        ]
        combos[Plotting._build_plot_key(DiscreteTypedFeature, OrdinalTypedFeature)] = [
            (
                self._matplot,
                seaborn.countplot,
                {
                    "x": self.x,
                    "hue": self.y,
                    "order": self.df[self.x]
                    .value_counts(ascending=True)
                    .iloc[:10]
                    .index,
                },
            )
        ]
        combos[
            Plotting._build_plot_key(DiscreteTypedFeature, CategoricalTypedFeature)
        ] = [
            (
                self._matplot,
                seaborn.countplot,
                {
                    "x": self.x,
                    "hue": self.y,
                    "order": self.df[self.x].value_counts().iloc[:10].index,
                },
            )
        ]
        combos[Plotting._build_plot_key(DateTimeTypedFeature, DateTimeTypedFeature)] = [
            (
                self._matplot,
                plt.scatter,
                {
                    "x": self.x,
                    "y": self.y,
                    "s": pi / 10 * (matplotlib.rcParams["lines.markersize"] ** 2),
                    "edgecolor": "white",
                    "linewidths": "0.1",
                },
            )
        ]

        combos[Plotting._build_plot_key(ContinuousTypedFeature, None)] = [
            (self._matplot, plt.hist, {"x": self.x})
        ]
        combos[Plotting._build_plot_key(CategoricalTypedFeature, None)] = [
            (
                self._matplot,
                Plotting._single_column_count_plot,
                {"x": self.x, "yscale": self.yscale},
            )
        ]
        combos[Plotting._build_plot_key(OrdinalTypedFeature, None)] = [
            (
                self._matplot,
                Plotting._ordinal_count_plot,
                {"x": self.x, "yscale": self.yscale},
            )
        ]
        combos[Plotting._build_plot_key(ConstantTypedFeature, None)] = [
            (self._matplot, seaborn.countplot, {"x": self.x})
        ]
        combos[Plotting._build_plot_key(DateTimeTypedFeature, None)] = [
            (self._matplot, plt.hist, {"x": self.x, "bins": 10, "color": "#1f77b4"})
        ]
        combos[Plotting._build_plot_key(GISTypedFeature, None)] = [
            (self._generic_plot, Plotting._folium_map, {"x": self.x})
        ]

        y_type_name = None if self.y_type is None else self.y_type.__class__
        keys_to_check = list(
            [Plotting._build_plot_key(self.x_type.__class__, y_type_name)]
        )

        new_x_type = Plotting._change_type(self.x_type)
        new_y_type = Plotting._change_type(self.y_type)
        keys_to_check.append(Plotting._build_plot_key(new_x_type, y_type_name))
        keys_to_check.append(
            Plotting._build_plot_key(self.x_type.__class__, new_y_type)
        )
        keys_to_check.append(Plotting._build_plot_key(new_x_type, new_y_type))
        for key in keys_to_check:
            if key in combos and combos[key]:
                assert isinstance(combos[key][0], tuple)
                return combos[key]

        if y_type_name is not None:
            raise NotImplementedError(
                "Plotting for the feature combination ({0} vs {1}) is not yet supported.".format(
                    self.x_type.meta_data["type"], self.y_type.meta_data["type"]
                )
            )
        else:
            raise NotImplementedError(
                "Plotting for feature type {0} is not supported".format(
                    self.x_type.meta_data["type"]
                )
            )

    def _change_type(feature_type):
        if feature_type is None:
            return None

        return (
            DiscreteTypedFeature
            if isinstance(feature_type, DiscreteTypedFeature)
            or isinstance(feature_type, CreditCardTypedFeature)
            or isinstance(feature_type, ZipcodeTypedFeature)
            or isinstance(feature_type, OrdinalTypedFeature)
            else feature_type.__class__
        )
