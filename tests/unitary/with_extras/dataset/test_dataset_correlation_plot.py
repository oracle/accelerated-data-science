#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

import pandas as pd
import pytest
import unittest

from ads.dataset.correlation_plot import BokehHeatMap
from ads.dataset.dataset import ADSDataset
from ads.dataset.exception import ValidationError


class TestBokehHeatMap(unittest.TestCase):
    dataset = pd.DataFrame(
        columns=[
            "one",
            "two",
            "three",
            "four",
            "five",
        ],
        data=[
            [
                3.4,
                999,
                1,
                None,
                95354,
            ],
            [
                3,
                999,
                5,
                None,
                90421,
            ],
            [
                2,
                999,
                1,
                15,
                89352,
            ],
            [
                3,
                999,
                6,
                11,
                89427,
            ],
            [
                2,
                999,
                1,
                46,
                94342,
            ],
        ],
    )
    bokeh_heatmap = BokehHeatMap(dataset)
    corr_matrix = dataset.corr()

    def test_plot_heat_map(self):
        p = self.bokeh_heatmap.plot_heat_map(
            self.corr_matrix,
            xrange=self.corr_matrix.index.values.tolist(),
            yrange=self.corr_matrix.columns.values.tolist(),
            title="heat_map",
        )

        self.assertEqual(str(type(p)), "<class 'bokeh.plotting._figure.figure'>")
        assert p.width == 600
        assert p.height == 600
        assert p.xaxis.major_label_orientation == "vertical"
        assert len(p.yaxis) == 1
        assert p.title.text == "heat_map"

    def test_plot_hbar(self):
        rows = self.corr_matrix.index.values.tolist()
        columns = self.corr_matrix.columns
        corr_flatten = pd.DataFrame(
            [(r, c, self.corr_matrix[r][c]) for c in columns for r in rows],
            columns=["X", "Y", "corr"],
        )
        p = self.bokeh_heatmap.plot_hbar(
            corr_flatten, title="plot_hbar", column_name="name in title"
        )

        self.assertEqual(str(type(p)), "<class 'bokeh.plotting._figure.figure'>")
        assert p.width == 600
        assert p.height == 600
        assert p.toolbar_location == "below"
        assert p.title.text == "plot_hbar (name in title)"

    def test_generate_heatmap(self):
        tabs = self.bokeh_heatmap.generate_heatmap(
            self.corr_matrix, title="heatmap", msg="", correlation_threshold=-1
        )

        self.assertEqual(str(type(tabs)), "<class 'bokeh.models.layouts.TabPanel'>")
        self.assertEqual(
            str(type(tabs.child)), "<class 'bokeh.plotting._figure.figure'>"
        )
        assert tabs.child.width == 600
        assert tabs.child.height == 600
        assert len(tabs.child.yaxis) == 1
        assert tabs.title == "heatmap"

    def test_generate_target_heatmap(self):
        tabs = self.bokeh_heatmap.generate_target_heatmap(
            self.corr_matrix,
            title="target_heatmap",
            correlation_target="one",
            msg="",
            correlation_threshold=-1,
        )

        self.assertEqual(str(type(tabs)), "<class 'bokeh.models.layouts.TabPanel'>")
        self.assertEqual(
            str(type(tabs.child)), "<class 'bokeh.plotting._figure.figure'>"
        )
        assert tabs.child.width == 600
        assert tabs.child.height == 600
        assert tabs.child.toolbar_location == "below"
        assert tabs.title == "target_heatmap"

    def test_plot_correlation_heatmap(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, "data", "orcl_attrition.csv")
        df = pd.read_csv(data_file)
        ds = ADSDataset.from_dataframe(df)
        bokeh_heatmap = BokehHeatMap(ds)

        with pytest.raises(ValidationError):
            bokeh_heatmap.plot_correlation_heatmap(
                ds=ds, correlation_methods="wrong_correlation_methods"
            )

        bokeh_heatmap.plot_correlation_heatmap(ds=ds, correlation_methods="pearson")
        bokeh_heatmap.plot_correlation_heatmap(ds=ds, correlation_methods="cramers v")
        bokeh_heatmap.plot_correlation_heatmap(
            ds=ds, correlation_methods="correlation ratio"
        )

        with pytest.raises(ValueError):
            bokeh_heatmap.plot_correlation_heatmap(ds=ds, plot_type="wrong_plot_type")
