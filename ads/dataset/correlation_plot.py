#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from ads.common import utils as au
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.dataset.correlation import _validate_correlation_methods
from ads.dataset.helper import deprecate_default_value


class BokehHeatMap(object):
    """
    Generate a HeatMap or horizontal bar plot to compare features.
    """

    @runtime_dependency(module="bokeh", install_from=OptionalDependency.VIZ)
    def __init__(self, ds):

        from bokeh.io import output_notebook
        from bokeh.palettes import BuPu

        output_notebook()

        self.ds = ds
        self.colormap = cm.get_cmap("BuPu")
        self.bokehpalette = [
            mpl.colors.rgb2hex(m) for m in self.colormap(np.arange(self.colormap.N))
        ]

    def debug(self):
        """
        Return True if in debug mode, otherwise False.
        """
        return au.is_debug_mode()

    def flatten_corr_matrix(self, corr_matrix):
        """
        Flatten a correlation matrix into a pandas Dataframe.

        Parameters
        ----------
        corr_matrix: Pandas Dataframe
          The correlation matrix to be flattened.

        Returns
        -------
        corr_flatten: Pandas DataFrame
            The flattened correlation matrix.

        """
        rows = corr_matrix.index.values.tolist()
        columns = corr_matrix.columns

        corr_flatten = pd.DataFrame(
            [(r, c, corr_matrix[r][c]) for c in columns for r in rows],
            columns=["x", "y", "corr"],
        )
        return corr_flatten

    @runtime_dependency(module="bokeh", install_from=OptionalDependency.VIZ)
    def plot_heat_map(
        self,
        matrix,
        xrange: list,
        yrange: list,
        low: float = 1,
        high=1,
        title: str = None,
        tool_tips: list = None,
    ):
        """
        Plots a matrix as a heatmap.


        Parameters
        ----------
        matrix: Pandas Dataframe
          The dataframe to be plotted.
        xrange: List of floats
          The range of x values to plot.
        yrange: List of floats
          The range of y values to plot.
        low: float, Defaults to 1
          The color mapping value for "low" points.
        high: float, Defaults to 1
          The color mapping value for "high" points.
        title: str, Defaults to None
          The optional title of the heat map.
        tool_tips: list of str, Defaults to None
          An optional list of tool tips to include with the plot.

        Returns
        -------
        fig: matplotlib Figure
            A matplotlib heatmap figure object.
        """
        if self.debug():
            print(matrix)
        mapper = bokeh.models.LinearColorMapper(
            palette=self.bokehpalette, low=low, high=high
        )
        source = bokeh.models.ColumnDataSource(matrix)

        from bokeh.plotting import figure

        p = figure(
            title=title,
            x_range=xrange,
            y_range=yrange,
            toolbar_location="below",
            toolbar_sticky=False,
            plot_width=600,
            plot_height=600,
        )

        p.rect(
            x="x",
            y="y",
            width=1,
            height=1,
            source=source,
            fill_color={"field": "corr", "transform": mapper},
            line_color=None,
        )
        p.xaxis.major_label_orientation = "vertical"

        if tool_tips:
            p.add_tools(bokeh.models.HoverTool(tooltips=tool_tips))

        color_bar = bokeh.models.ColorBar(
            color_mapper=mapper,
            major_label_text_font_size="5pt",
            ticker=bokeh.models.BasicTicker(desired_num_ticks=8),
            formatter=bokeh.models.PrintfTickFormatter(format="%0.2f"),
            label_standoff=6,
            border_line_color=None,
            location=(0, 0),
        )
        p.add_layout(color_bar, "right")
        return p

    @runtime_dependency(module="bokeh", install_from=OptionalDependency.VIZ)
    def plot_hbar(
        self,
        matrix,
        low: float = 1,
        high=1,
        title: str = None,
        tool_tips: list = None,
        column_name: str = None,
    ):
        """
        Plots a histogram bar-graph.

        Parameters
        ----------
        matrix: Pandas Dataframe
          The dataframe to be plotted.
        low: float, Defaults to 1
          The color mapping value for "low" points.
        high: float, Defaults to 1
          The color mapping value for "high" points.
        title: str, Defaults to None
          The optional title of the heat map.
        tool_tips: list of str, Defaults to None
          An optional list of tool tips to include with the plot.
        column_name: str, Defaults to None
          The name of the column which is being plotted.

        Returns
        -------
        fig: matplotlib Figure
            A matplotlib heatmap figure object.
        """
        mapper = bokeh.models.LinearColorMapper(
            palette=self.bokehpalette, low=low, high=high
        )
        source = bokeh.models.ColumnDataSource(matrix)

        from bokeh.plotting import figure

        p = figure(
            title=f"{title} ({column_name})",
            x_range=(low, high),
            y_range=(0, len(matrix["Y"]) + 1),
            toolbar_location="below",
            toolbar_sticky=False,
            plot_width=600,
            plot_height=600,
        )

        p.hbar(
            y="Y",
            height=0.5,
            left=0,
            source=source,
            right="corr",
            fill_color={"field": "corr", "transform": mapper},
            line_color=None,
        )

        p.add_tools(bokeh.models.HoverTool(tooltips=tool_tips))

        color_bar = bokeh.models.ColorBar(
            color_mapper=mapper,
            major_label_text_font_size="5pt",
            ticker=bokeh.models.BasicTicker(desired_num_ticks=8),
            formatter=bokeh.models.PrintfTickFormatter(format="%0.2f"),
            label_standoff=6,
            border_line_color=None,
            location=(0, 0),
        )

        labels = bokeh.models.LabelSet(
            x="label",
            y="Y",
            text="X",
            level="glyph",
            y_offset=-5,
            source=source,
            render_mode="canvas",
        )

        p.add_layout(color_bar, "right")
        p.add_layout(labels)
        return p

    @runtime_dependency(module="bokeh", install_from=OptionalDependency.VIZ)
    def generate_heatmap(
        self, corr_matrix, title: str, msg: str, correlation_threshold: float
    ):
        """
        Generate a heatmap from a correlation matrix.


        Parameters
        ----------
        corr_matrix: Pandas Dataframe
          The dataframe to be used for heatmap generation.
        title: str
          title of the heatmap.
        msg: str
          An additional msg to include in the plot.
        correlation_threshold: float
          A float between 0 and 1 which is used for excluding correlations which are not intense enough from the plot.

        Returns
        -------
        tab: matplotlib Panel
            A matplotlib Panel object which includes a plotted heatmap

        """
        from bokeh.plotting import figure

        if len(corr_matrix) == 0:
            tab = bokeh.models.Panel(
                child=figure(title=msg + ", nothing to display"),
                title=title,
            )
            return tab
        corr_matrix = _corr_filter(correlation_threshold, corr_matrix)
        corr_flatten = self.flatten_corr_matrix(corr_matrix)
        low = -1 if corr_matrix.min().min() < 0 else 0
        high = 1
        if self.debug():
            print(f"{title} : Min is {low}, Max is: {high}")

        p = self.plot_heat_map(
            corr_flatten,
            corr_matrix.index.values.tolist(),
            corr_matrix.columns.values.tolist(),
            low,
            high,
            title,
            tool_tips=[("X", "@x"), ("Y", "@y"), ("Corr", "@corr")],
        )

        tab = bokeh.models.Panel(child=p, title=title)
        return tab

    @runtime_dependency(module="bokeh", install_from=OptionalDependency.VIZ)
    def generate_target_heatmap(
        self,
        corr_matrix,
        title: str,
        correlation_target: str,
        msg: str,
        correlation_threshold: float,
    ):
        """
        Generate a heatmap from a correlation matrix and its targets.


        Parameters
        ----------
        corr_matrix: Pandas Dataframe
          The dataframe to be used for heatmap generation.
        title: str
          title of the heatmap.
        correlation_target: str
          The target column name for computing correlations against.
        msg: str
          An additional msg to include in the plot.
        correlation_threshold: float
          A float between 0 and 1 which is used for excluding correlations which are not intense enough from the plot.

        Returns
        -------
        tab: matplotlib Panel
            A matplotlib Panel object which includes a plotted heatmap.
        """
        from bokeh.plotting import figure

        if len(corr_matrix) == 0:
            tab = bokeh.models.Panel(
                child=figure(title=msg + ", nothing to display"),
                title=title,
            )
            return tab

        corr_matrix = _corr_filter(correlation_threshold, corr_matrix)

        assert correlation_target, "Correlation target is required for this plot"
        if correlation_target not in corr_matrix.columns:
            tab = bokeh.models.Panel(
                child=figure(title="No Data to display"), title=title
            )
            return tab

        corr_flatten = {}

        corr_clean = corr_matrix[correlation_target].dropna()

        if self.debug():
            print(corr_matrix[correlation_target].values.tolist())
            print(len(corr_matrix[correlation_target].values.tolist()))
            print("*" * 30)

        corr_flatten["X"] = corr_clean.index.values.tolist()

        if self.debug():
            print(corr_clean.index.values.tolist())
            print(len(corr_clean.index.values.tolist()))
            print("*" * 30)

        corr_flatten["Y"] = [
            i for i in range(1, len(corr_clean.index.values.tolist()) + 1)
        ]

        corr_flatten["corr"] = corr_clean.values.tolist()
        corr_flatten["label"] = [
            x + 0.01 if x > 0 else 0.01 for x in corr_clean.values.tolist()
        ]

        corr_flatten = pd.DataFrame.from_dict(data=corr_flatten)

        if self.debug():
            print(corr_flatten)

        low = -1 if corr_clean.min() < 0 else 0
        high = 1
        if self.debug():
            print(f"{title} : Min is {low}, Max is: {high}")

        p = self.plot_hbar(
            corr_flatten,
            low,
            high,
            title,
            tool_tips=[("Feature", "@X"), ("Corr", "@corr")],
            column_name=correlation_target,
        )

        tab = bokeh.models.Panel(child=p, title=title)
        return tab

    @runtime_dependency(module="bokeh", install_from=OptionalDependency.VIZ)
    def plot_correlation_heatmap(
        self,
        ds,
        plot_type: str = "heatmap",
        correlation_target: str = None,
        correlation_threshold=-1,
        correlation_methods: str = "pearson",
        **kwargs,
    ):
        """
        Plots a correlation heatmap.

        Parameters
        ----------
        ds: Pandas Slice
          A data slice or file
        plot_type: str Defaults to "heatmap"
          The type of plot - "bar" is another option.
        correlation_target: str, Defaults to None
          the target column for correlation calculations.
        correlation_threshold: float, Defaults to -1
          the threshold for computing correlation heatmap elements.
        correlation_methods: str, Defaults to "pearson"
          the way to compute correlations, other options are "cramers v" and "correlation ratio"
        """
        assert self.ds or ds, "Expecting input for ds or file"

        plot_ds = None

        if ds:
            plot_ds = ds
        else:
            plot_ds = self.ds

        frac = kwargs.get("frac", 1)
        frac = deprecate_default_value(
            frac,
            None,
            1,
            f"<code>frac=None</code> is deprecated. Use <code>frac=1.0</code> instead.",
            FutureWarning,
        )

        force_recompute = kwargs.get("force_recompute", False)
        nan_threshold = kwargs.get("nan_threshold", 0.8)

        cts_cts = pd.DataFrame()
        cat_cat = pd.DataFrame()
        cat_cts = pd.DataFrame()
        correlation_list = plot_ds.corr(
            frac=frac,
            nan_threshold=nan_threshold,
            force_recompute=force_recompute,
            correlation_methods=correlation_methods,
        )

        correlation_methods = _validate_correlation_methods(correlation_methods)
        for method in correlation_methods:
            if method == "pearson":
                cts_cts = plot_ds._pearson
            elif method == "cramers v":
                cat_cat = plot_ds._cramers_v
            elif method == "correlation ratio":
                cat_cts = plot_ds._correlation_ratio
            else:
                raise ValueError(f"This {method} is not supported.")

        # generate the msg
        cts_cts_msg = _generate_msg(
            "pearson", correlation_methods, correlation_target, "continuous"
        )
        cat_cat_msg = _generate_msg(
            "cramers v", correlation_methods, correlation_target, "categorical"
        )
        cat_cts_msg = _generate_msg(
            "correlation ratio",
            correlation_methods,
            correlation_target,
            "continuous and categorical",
        )

        panel_items = [
            {"data": cts_cts, "title": "Continuous vs Continuous", "msg": cts_cts_msg},
            {"data": cat_cat, "title": "Category vs Category", "msg": cat_cat_msg},
            {"data": cat_cts, "title": "Category vs Continuous", "msg": cat_cts_msg},
        ]

        if self.debug():
            print("-" * 50, "Continuous vs Continuous", "-" * 50)
            print(cts_cts.head())

            print("-" * 50, "Category vs Category", "-" * 50)
            print(cat_cat.head())

            print("-" * 50, "Category vs Continuous", "-" * 50)
            print(cat_cts.head())

        if plot_type == "heatmap":
            tabs = [
                self.generate_heatmap(
                    item["data"], item["title"], item["msg"], correlation_threshold
                )
                for item in panel_items
            ]
        elif plot_type == "bar":
            tabs = [
                self.generate_target_heatmap(
                    item["data"],
                    item["title"],
                    correlation_target,
                    item["msg"],
                    correlation_threshold,
                )
                for item in panel_items
            ]
        else:
            raise ValueError("Only supported plot types are heatmap and bar")

        bokeh_tabs = bokeh.models.Tabs(tabs=tabs)

        from bokeh.io import show

        show(bokeh_tabs)


def _generate_msg(
    method: str, correlation_methods: list, correlation_target: str, feature_type: str
):
    if method in correlation_methods or "all" in correlation_methods:
        if correlation_target is not None:
            msg = (
                correlation_target
                + f"is not of the {feature_type} type or not enough of the {feature_type} type(s)"
            )
        else:
            msg = f"Not enough of the {feature_type} type(s)"
    else:
        msg = f'Pass in "{method}" to show the plot'

    return msg


def _has_categorical_features(ds):
    df_type = pd.DataFrame.from_dict(ds.feature_types).T
    return (
        "categorical" in df_type["type"].values or "zipcode" in df_type["type"].values
    )


def plot_correlation_heatmap(ds=None, **kwargs) -> None:
    """
    Plots a correlation heatmap.

    Parameters
    ----------
    ds: Pandas Slice
      A data slice or file
    """
    corr_heatmap_helper = BokehHeatMap(ds)
    corr_heatmap_helper.plot_correlation_heatmap(ds, **kwargs)


def _corr_filter(correlation_threshold, corr):
    if correlation_threshold >= 1 or correlation_threshold < -1:
        raise ValueError(
            "The correlation_threshold value should within the range from -1 to 1."
        )
    new_corr = corr.copy(deep=True)
    np.fill_diagonal(new_corr.values, np.nan)
    new_corr = (
        new_corr[new_corr >= correlation_threshold]
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
    )
    # fill the diagonal with 1 to effectively shrink the correlation matrix
    np.fill_diagonal(new_corr.values, 1)
    return new_corr
