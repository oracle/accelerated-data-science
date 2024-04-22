#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from matplotlib.patches import BoxStyle

from ads.dataset.label_encoder import DataFrameLabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

import matplotlib.font_manager
from ads.common import utils, logger
from ads.dataset.helper import (
    fix_column_names,
    convert_columns,
    get_feature_type,
    convert_to_html,
)
from ads.dataset.plot import Plotting
from ads.dataset.progress import DummyProgressBar
from ads.dataset.timeseries import Timeseries
from ads.type_discovery.type_discovery_driver import TypeDiscoveryDriver
from ads.type_discovery.typed_feature import (
    DateTimeTypedFeature,
    ContinuousTypedFeature,
    GISTypedFeature,
    ConstantTypedFeature,
    CreditCardTypedFeature,
    ZipcodeTypedFeature,
    PhoneNumberTypedFeature,
    OrdinalTypedFeature,
    CategoricalTypedFeature,
    DocumentTypedFeature,
    AddressTypedFeature,
)
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)

NATURAL_EARTH_DATASET = "naturalearth_lowres"


class PandasDataset(object):
    """
    This class provides APIs that can work on a sampled dataset.
    """

    def __init__(
        self,
        sampled_df,
        type_discovery=True,
        types={},
        metadata=None,
        progress=DummyProgressBar(),
    ):
        self.client = None
        self.sampled_df = fix_column_names(sampled_df)
        self.correlation = None
        self.feature_dist_html_dict = {}
        self.feature_types = metadata if metadata is not None else {}
        self.world = None

        self.numeric_columns = self.sampled_df.select_dtypes(
            utils.numeric_pandas_dtypes()
        ).columns.values

        # run type discovery
        if len(self.feature_types) == 0:
            if len(types) != 0:
                # update feature types as it is for the types given by user
                self._update_feature_types(types.keys())
            if type_discovery:
                try:
                    #
                    # perform type-discovery
                    #
                    progress.update("Running data type discovery")
                    tdd = TypeDiscoveryDriver()
                    self.feature_types = {
                        col_name: tdd.discover(col_name, col_vals)
                        for col_name, col_vals in self.sampled_df.items()
                        if col_name not in types
                    }
                except Exception as e:
                    print(
                        f"An error occured while performing typed discovery on this dataset. Try running again with "
                        f"`type_discovery=False`"
                    )
                    raise e

        # convert dataframe columns to the data types discovered.
        self.sampled_df = convert_columns(self.sampled_df, self.feature_types)

        # update feature types for rest of the columns - no type discovery, does not include user overrides
        self._update_feature_types(
            set(self.sampled_df.columns.values) - set(self.feature_types.keys())
        )
        self.sampled_df = self.sampled_df.reset_index(drop=True)

    def _find_feature_subset(self, df, target_name, include_n_features=32):
        if len(df.columns) <= include_n_features:
            return self.sampled_df
        else:
            try:
                y = df[target_name]
                X = df.drop(columns=[target_name])
                X_columns = X.columns

                X = X.fillna(X.mode().iloc[0])
                X = DataFrameLabelEncoder().fit_transform(X)
                X = MinMaxScaler().fit_transform(X)

                from sklearn.impute import SimpleImputer

                imp_most_freq = SimpleImputer(strategy="most_frequent")
                X = imp_most_freq.fit_transform(X)

                est = SelectKBest(score_func=chi2, k=include_n_features)
                est.fit_transform(X, y)

                subset_features = [self.target.name] + list(
                    itertools.compress(X_columns, est.get_support())
                )

                return self.sampled_df.filter(subset_features, axis=1)

            except Exception as e:
                print("_find_feature_subset failed: ", str(e))
                return pd.DataFrame()

    def _update_multiple_outputs(self, out, msg):
        if isinstance(out, (list, tuple)):
            for o in out:
                o.value = msg
        else:
            self._update_multiple_outputs([out], msg)

    def _calculate_dataset_statistics(self, is_wide_dataset, out):
        #
        # first the missing values for non-wide datasets
        #
        df_missing = pd.DataFrame()
        df_skew = pd.DataFrame()

        if is_wide_dataset:
            df_missing = pd.DataFrame()  # empty dataframe when not calculating
            df_skew = pd.DataFrame()  # empty dataframe when not calculating
        else:
            #
            # count missing values
            #
            d = {column_name: np.nan for column_name in self.df.columns}  # default
            for column_name in self.df.columns:
                self._update_multiple_outputs(
                    out, f"calculating missing values (<code>{column_name}</code>)..."
                )
                d[column_name] = self.df[column_name].isna().sum()

            df_missing = pd.DataFrame.from_dict(d, orient="index", columns=["missing"])

            #
            # calculate skew
            #
            d = {column_name: np.nan for column_name in self.df.columns}  # default
            for column_name in self.numeric_columns:
                self._update_multiple_outputs(
                    out, f"calculating skew (<code>{column_name}</code>)..."
                )
                if len(self.df[column_name].dropna()) > 0:
                    d[column_name] = self.df[column_name].dropna().skew()
                else:
                    d[column_name] = np.nan
            df_skew = pd.DataFrame.from_dict(d, orient="index", columns=["skew"])
            self._update_multiple_outputs(out, "calculating dataset statistics...")
            for col in self.df.columns:
                if df_missing["missing"][col] == len(self.df[col]):
                    self.df[col] = self.df[col].astype("object")

        #
        # compute descriptive dataset statistics
        df_stats = self.df.describe(include="all").round(2)

        self._update_multiple_outputs(out, "transpose dataset statistics...")
        df_stats = df_stats.transpose()

        self._update_multiple_outputs(out, "finalizing dataset statistics...")
        df_stats = df_stats.fillna("")

        #
        # we join the stats with missing df if we computed that
        #
        self._update_multiple_outputs(
            out, "Assembling statistics into single result..."
        )
        if not df_missing.empty:
            df_stats = df_stats.join(df_missing).fillna("")
        if not df_skew.empty:
            df_stats = df_stats.join(df_skew).fillna("")

        return df_stats

    def _generate_features_html(
        self, is_wide_dataset, n_features, df_stats, visualizations_follow
    ):
        html = utils.get_bootstrap_styles()

        if is_wide_dataset:
            html += """<p>The dataset has too many columns ({:,}) to
                efficiently show feature visualizations, instead only showing table of
                statistics</p>""".format(
                n_features
            )

        html += "<p><b>&#x2022; Note</b> these are computed on the entire dataset.</p>"
        html += "<hr>"
        html += (
            df_stats.replace(np.nan, "")
            .style.set_table_styles(utils.get_dataframe_styles(max_width=125))
            .set_table_attributes("class=table")
            .format(
                lambda x: "{:.8g}".format(x)
                if ("float" in str(type(x))) or ("int" in str(type(x)))
                else x
            )
            .to_html()
        )

        if visualizations_follow:
            html += "<br><hr><h2>Feature Visualizations...</h2>"

        return html

    def _generate_warnings_html(
        self, is_wide_dataset, n_rows, n_features, df_stats, out, accordion
    ):
        #
        # create the "Warnings" accordion section:
        #  - show high cardinal categoricals
        #  - show high missing values
        #  - large number of zeros (not for wide datasets - slow to compute)
        #
        #

        accum = []

        ignored_feature_types = ["document"]

        # more than 5% missing is a warning
        #

        if "missing" in df_stats.columns:
            df_missing = df_stats[["missing"]][df_stats["missing"] != ""]
            if not df_missing.empty:
                # ignore document types
                for column_name, missing in df_missing.iterrows():
                    if (
                        self.feature_types[column_name]["type"]
                        not in ignored_feature_types
                    ):
                        missing_values = float(missing[0])
                        missing_pct = 100 * missing_values / n_rows
                        if missing_pct >= 5:
                            accum.append(
                                {
                                    "label": "missing",
                                    "message": f"<code>{column_name}</code> has {missing_values} ({missing_pct:.1f}%) missing values. Consider remove the column or replace null values.",
                                }
                            )
                        if missing_pct == 100:
                            accum.append(
                                {
                                    "label": "missing",
                                    "message": f"<code>{column_name}</code> is excluded from correlation computation due to {missing_values} ({missing_pct:.1f}%) missing values.",
                                }
                            )

        # abs skew > 20 skewness warning issues
        #

        if "skew" in df_stats.columns:
            df_skew = df_stats[["skew"]][df_stats["skew"] != ""]
            if not df_skew.empty:
                # ignore document types
                for column_name, skew in df_skew.iterrows():
                    if (
                        self.feature_types[column_name]["type"]
                        not in ignored_feature_types
                    ):
                        skewness = float(skew[0])
                        if abs(skewness) >= 20:
                            accum.append(
                                {
                                    "label": "skew",
                                    "message": f"<code>{column_name}</code> has skew of {skewness:.3f}",
                                }
                            )

        # high cardinality (> 15 unique values) is a warning
        #

        if "unique" in df_stats.columns:
            for column_name, count in df_stats[["unique"]][
                df_stats["unique"] != ""
            ].iterrows():
                # ignore document types
                if self.feature_types[column_name]["type"] not in ignored_feature_types:
                    out.value = (
                        f"Analyzing cadinalities (<code>{column_name}</code>)..."
                    )
                    unique = self.df[column_name].unique().shape[0]
                    if unique == n_rows:
                        accum.append(
                            {
                                "label": "high-cardinality",
                                "message": f"<code>{column_name}</code> has a high cardinality: every value is distinct",
                            }
                        )
                    elif unique > 15:
                        accum.append(
                            {
                                "label": "high-cardinality",
                                "message": f"<code>{column_name}</code> has a high cardinality: {unique} distinct values",
                            }
                        )

        if not is_wide_dataset:
            # more than 10% zeros is a warning
            if "min" in df_stats.columns:
                for column_name, count in df_stats[["min"]][
                    df_stats["min"] != ""
                ].iterrows():
                    if (
                        self.feature_types[column_name]["type"]
                        not in ignored_feature_types
                    ):
                        out.value = f"Analyzing zeros (<code>{column_name}</code>)..."
                        # we've filtered on only columns that have a min value of 0 for speed
                        zeros = self.df[self.df[column_name] == 0].shape[0]
                        zeros_pct = 100 * zeros / n_rows
                        if zeros_pct >= 10:
                            accum.append(
                                {
                                    "label": "zeros",
                                    "message": f"<code>{column_name}</code> has {zeros} ({zeros_pct:.2f}%) zeros)",
                                }
                            )

        #
        # collect the warnings into an HTML presentation
        #

        out.value = "Assembling results..."

        if accum:
            html = utils.get_bootstrap_styles()

            html += utils.highlight_text(f"{len(accum)} WARNING(S) found")

            accordion.set_title(3, f"Warnings ({len(accum)})")

            html += "<hr>"

            tr_rows = [
                f"""
                <tr style="border-top:0">
                       <td>
                        {feature_warning['message']}
                       <td>
                        <span class="label label-{feature_warning['label']}">{feature_warning['label']}</span>
                       </td>
                </tr>
            """.strip()
                for feature_warning in accum
            ]

            html += """
                <table style="width: 100%; max-width: 100%;">
                <tbody>
                {}
                </tbody>
                </table>""".format(
                "\n".join(tr_rows)
            )

            return html

        else:
            return "<h3>No Feature warnings found</h3>"

    def summary(self, feature_name=None):
        """
        Display list of features & their datatypes.
        Shows the column name and the feature's meta_data if given a specific feature name.

        Parameters
        ----------
        date_col: str
            The name of the feature

        Returns
        -------
        dict
            a dictionary that contains requested information
        """

        feature_n_datatype = {}
        list_of_dfs = []

        if feature_name is None:
            feature_n_datatype = {
                col: self.feature_types[col].type
                + "/"
                + self.feature_types[col].low_level_type
                for col in self.sampled_df.columns
            }
            df = pd.DataFrame(
                feature_n_datatype.items(), columns=["Feature", "Datatype"]
            )
        else:
            if isinstance(feature_name, (list, tuple, pd.core.indexes.base.Index)):
                feature_names = list(feature_name)
            else:
                feature_names = [feature_name]

            for col in feature_names:
                if col in self.sampled_df.columns:
                    feature_n_datatype[col] = {
                        k: v
                        for k, v in self.feature_types[col].meta_data.items()
                        if k not in ["internal", "feature_name"]
                    }
                    new_dict = utils.flatten(feature_n_datatype[col])
                    tmp_df = pd.DataFrame.from_dict(
                        new_dict,
                        orient="index",
                        columns=[feature_names[feature_names.index(str(col))]],
                    )
                    list_of_dfs.append(tmp_df)
                else:
                    feature_n_datatype[col] = None

            df = pd.concat(list_of_dfs, axis=1).transpose().fillna("-")

            # get all the unique types from df
            new_list_dfs = []
            for t in df.type.unique():
                new_list_dfs.append(df[df["type"] == t])
            df = pd.concat(new_list_dfs)

            # reorder columns in df

            # get a list of columns
            cols = list(df)
            # move the column to head of list using index, pop and insert
            cols.insert(0, cols.pop(cols.index("low_level_type")))
            cols.insert(0, cols.pop(cols.index("type")))
            df = df.loc[:, cols]

        return df

    def timeseries(self, date_col):
        """
        Supports any plotting operations where x=datetime.

        Parameters
        ----------
        date_col: str
            The name of the feature to plot

        Returns
        -------
        func
            a plotting object that contains a date column and dataframe
        """

        if date_col in self.feature_types and isinstance(
            self.feature_types[date_col], DateTimeTypedFeature
        ):
            return Timeseries(date_col, self.sampled_df)
        else:
            raise ValueError("Not a date time column.")

    def plot(
        self, x, y=None, plot_type="infer", yscale=None, verbose=True, sample_size=0
    ):
        """
        Supports plotting feature distribution, and relationship between features.

        Parameters
        ----------
        x: str
            The name of the feature to plot
        y: str, optional
            Name of the feature to plot against x
        plot_type: str, default: infer
            Override the inferred plot type for certain combinations of the data types of x and y.
            By default, the best plot type is inferred based on x and y data types.
            Valid values:

            - box_plot - discrete feature vs continuous feature. Draw a box plot to show
              distributions with respect to categories,
            - scatter - continuous feature vs continuous feature. Draw a scatter plot
              with possibility of several semantic groupings.

        yscale : str, optional
            One of {"linear", "log", "symlog", "logit"}.
            The y axis scale type to apply. Can be used when either x or y is an ordinal feature.
        verbose: bool, default True
            Displays Note/Tips if True
        """
        sample_size = int(sample_size)
        min_sample_size = 10000
        if sample_size == 0:
            sub_samp_size = len(self.sampled_df)
            sub_samp_df = self.sampled_df
        else:
            sub_samp_size = max(min(sample_size, len(self.sampled_df)), min_sample_size)
            sub_samp_df = self.sampled_df.sample(n=sub_samp_size)
        plot = Plotting(
            sub_samp_df, self.feature_types, x, y=y, plot_type=plot_type, yscale=yscale
        )
        if verbose:
            if len(self.df) != sub_samp_size:
                logger.info(f"Downsampling from dataset for graphing.")
        return plot

    @runtime_dependency(module="geopandas", install_from=OptionalDependency.GEO)
    def plot_gis_scatter(self, lon="longitude", lat="latitude", ax=None):
        """
        Supports plotting Choropleth maps

        Parameters
        ----------
        df: pandas dataframe
            The dataframe to plot
        x: str
            The name of the feature to plot, usually the longitude
        y: str
            THe name of the feature to plot, usually the latitude
        """
        if lon in self.sampled_df.columns and lat in self.sampled_df.columns:
            if ax is None:
                fig, ax = plt.subplots(1, figsize=(10, 10))
            gdf = geopandas.GeoDataFrame(
                self.sampled_df,
                geometry=geopandas.points_from_xy(
                    self.sampled_df[lon], self.sampled_df[lat]
                ),
            )
            world = geopandas.read_file(
                geopandas.datasets.get_path(NATURAL_EARTH_DATASET)
            )
            ax1 = world.plot(ax=ax, color="lightgrey", linewidth=0.5, edgecolor="white")
            gdf.plot(ax=ax1, color="blue", markersize=10)

        else:
            if len(self.sampled_df.columns) > 0:
                logger.info(
                    "The available latitude and longitude columns are: "
                    + ", ".join(self.sampled_df.columns)
                    + "."
                )
            else:
                logger.info("There are no latitude and longitude columns available.")

    """
    Internal methods
    """

    def _update_feature_types(self, columns):
        # Build feature types for columns which are not type discovered, by using the inferred type as it is
        for column in columns:
            self.feature_types[column] = get_feature_type(
                column, self.sampled_df[column]
            )

    @runtime_dependency(module="geopandas", install_from=OptionalDependency.GEO)
    def _visualize_feature_distribution(self, html_widget):
        """
        This function is called once per dataset to generate html for feature distribution plots.
        """
        if len(self.feature_dist_html_dict) > 0:
            return self.feature_dist_html_dict

        feature_dist_html = ""
        figsize = (6.5, 2)

        props = {
            "boxstyle": BoxStyle("Round", pad=0),
            "facecolor": "white",
            "linestyle": "solid",
            "linewidth": 0,
            "edgecolor": "white",
        }

        font = {"size": 10}

        matplotlib.rc("font", **font)

        red_square = dict(markerfacecolor="r", marker="s")

        blues = [
            "#AED6F1",
            "#85C1E9",
            "#5DADE2",
            "#3498DB",
            "#2E86C1",
            "#2874A6",
            "#1B4F72",
        ]
        bright_colors = ["red", "green", "blue", "yellow", "green"]

        for col_index, col in enumerate(self.sampled_df.columns):
            feature_metadata = self.feature_types[col].meta_data
            text = "\n".join(
                [
                    f"{col}\n",
                    "  - type: {} ({})".format(
                        self.feature_types[col].type,
                        self.feature_types[col].low_level_type,
                    ),
                    "  - missing_percentage: {:.1f}%".format(
                        self.feature_types[col].missing_percentage
                    ),
                ]
            )

            fig, ax = PandasDataset._init_fig_ax(figsize)

            if isinstance(self.feature_types[col], ContinuousTypedFeature):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type, feature_metadata["stats"]
                )
                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )
                self.sampled_df[col].plot(
                    kind="box",
                    vert=False,
                    flierprops=red_square,
                    ax=ax,
                    figsize=figsize,
                )

            elif isinstance(self.feature_types[col], DateTimeTypedFeature):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type, feature_metadata["stats"]
                )
                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )
                from matplotlib.dates import AutoDateFormatter, AutoDateLocator

                loc = AutoDateLocator(interval_multiples=False)
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(AutoDateFormatter(loc))
                self.sampled_df[col].hist(
                    bins=50,
                    grid=False,
                    xrot=45,
                    ax=ax,
                    rwidth=0.95,
                    color=blues[-1],
                    figsize=figsize,
                )

            elif isinstance(self.feature_types[col], GISTypedFeature):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type, feature_metadata["stats"]
                )
                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )
                df = pd.DataFrame(
                    feature_metadata["internal"]["sample"], columns=["lat", "lon"]
                )
                gdf = geopandas.GeoDataFrame(
                    df, geometry=geopandas.points_from_xy(df["lon"], df["lat"])
                )

                if not self.world:
                    self.world = geopandas.read_file(
                        geopandas.datasets.get_path(NATURAL_EARTH_DATASET)
                    )

                self.world.plot(
                    ax=ax, color="lightgrey", linewidth=0.5, edgecolor="white"
                )
                gdf.plot(ax=ax, color="blue", markersize=10)
                ax.set_aspect("auto")

            elif (
                isinstance(self.feature_types[col], ConstantTypedFeature)
                and feature_metadata["missing_percentage"] < 100
            ):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type, feature_metadata["stats"]
                )
                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )
                pd.Series(
                    feature_metadata["internal"]["counts"],
                    name=col,
                    index=feature_metadata["internal"]["counts"].keys(),
                ).plot(kind="barh", ax=ax, width=0.95, figsize=figsize, color=["black"])

            elif isinstance(self.feature_types[col], CreditCardTypedFeature):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type, feature_metadata["stats"]
                )
                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )
                sorted_by_value = sorted(
                    feature_metadata["internal"]["counts"],
                    key=feature_metadata["internal"]["counts"].get,
                    reverse=True,
                )
                pd.Series(
                    feature_metadata["internal"]["counts"],
                    name=col,
                    index=sorted_by_value,
                ).plot(kind="bar", ax=ax, width=0.95, figsize=figsize, color=blues)

            elif isinstance(self.feature_types[col], ZipcodeTypedFeature):
                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )
                pd.Series(
                    feature_metadata["internal"]["histogram"],
                    name=col,
                    index=feature_metadata["internal"]["histogram"].keys(),
                ).plot(kind="bar", ax=ax, figsize=figsize, color=blues)

            elif isinstance(self.feature_types[col], PhoneNumberTypedFeature):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type, feature_metadata["stats"]
                )
                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )
                pd.Series(
                    feature_metadata["internal"]["counts"],
                    name=col,
                    index=feature_metadata["internal"]["counts"].keys(),
                ).plot(kind="bar", ax=ax, figsize=figsize, color=blues)

            elif isinstance(self.feature_types[col], OrdinalTypedFeature):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type, feature_metadata["stats"]
                )
                high_cardinality = feature_metadata["internal"]["high_cardinality"]
                very_high_cardinality = feature_metadata["internal"][
                    "very_high_cardinality"
                ]

                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )

                if very_high_cardinality:
                    addrtext = "Samples:\n\n"
                    addrtext += ", ".join(
                        feature_metadata["internal"]["counts"]
                        .keys()
                        .astype(str)
                        .to_list()[:6]
                    )

                    ax.text(
                        0.05,
                        0.95,
                        addrtext,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=props,
                    )
                    ax.axis("off")

                else:
                    if high_cardinality:
                        text += (
                            "\n\n  NOTE: plot has been capped\n  from %d to show only most\n  common top %d "
                            "categories"
                            % (
                                feature_metadata["internal"]["unique"],
                                len(feature_metadata["internal"]["counts"].keys()),
                            )
                        )

                    if feature_metadata["internal"]["unique"] < 24:
                        pd.Series(
                            feature_metadata["internal"]["counts"],
                            name=col,
                            index=feature_metadata["internal"]["counts"].keys(),
                        ).plot(
                            kind="bar",
                            ax=ax,
                            width=0.90,
                            color=blues[-1],
                            figsize=figsize,
                        )
                    else:
                        self.sampled_df[col].plot(
                            kind="hist",
                            grid=False,
                            rwidth=0.95,
                            ax=ax,
                            color=blues[-1],
                            figsize=figsize,
                        )

            elif isinstance(self.feature_types[col], CategoricalTypedFeature):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type, feature_metadata["stats"]
                )

                high_cardinality = feature_metadata["internal"]["high_cardinality"]
                very_high_cardinality = feature_metadata["internal"][
                    "very_high_cardinality"
                ]

                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )

                if very_high_cardinality:
                    # grab the first few examples as strings
                    addrtext = "Samples:\n\n"
                    addrtext += "\n".join(
                        [
                            utils.ellipsis_strings(x, 65)
                            for x in feature_metadata["internal"]["counts"]
                            .keys()
                            .astype(str)
                            .to_list()[:3]
                        ]
                    )
                    ax.text(
                        0.05,
                        0.95,
                        addrtext,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=props,
                    )
                    ax.axis("off")

                else:
                    text += PandasDataset._format_stats(
                        self.feature_types[col].type, feature_metadata["stats"]
                    )
                    if high_cardinality:
                        text += (
                            "\n\n  NOTE: plot has been capped\n  to show only most\n  common top %d categories"
                            % (len(feature_metadata["internal"]["counts"].keys()))
                        )

                    if feature_metadata["internal"]["unique"] == 2:
                        #
                        # binary
                        #
                        count_series = pd.Series(
                            feature_metadata["internal"]["counts"],
                            name=col,
                            index=feature_metadata["internal"]["counts"].keys(),
                        ).astype(float)
                        ax1 = count_series.plot(
                            kind="barh",
                            ax=ax,
                            width=0.95,
                            figsize=figsize,
                            color=[blues[0], blues[-1]],
                        )
                        # x_labels = utils.ellipsis_strings(feature_metadata['internal']['counts'].keys().astype(str))
                        # ax1.set_xticklabels(x_labels)
                    else:
                        #
                        # multiclass, potentially high cardinality
                        #

                        ax1 = pd.Series(
                            feature_metadata["internal"]["counts"],
                            name=col,
                            index=feature_metadata["internal"]["counts"].keys(),
                        ).plot(
                            kind="bar", ax=ax, width=0.95, color=blues, figsize=figsize
                        )

                        x_labels = utils.ellipsis_strings(
                            feature_metadata["internal"]["counts"].keys()
                        )
                        ax1.set_xticklabels(x_labels)

            elif isinstance(
                self.feature_types[col], DocumentTypedFeature
            ) or isinstance(self.feature_types[col], AddressTypedFeature):
                text += PandasDataset._format_stats(
                    self.feature_types[col].type,
                    {k: v for k, v in feature_metadata["stats"].items()},
                )

                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                )
                if "word_frequencies" in feature_metadata["internal"]:
                    word_freqs = feature_metadata["internal"]["word_frequencies"]
                    stats = "\n".join(["  - word count: %d" % (len(word_freqs.keys()))])
                    text = text + "\n"
                    text += stats
                    try:
                        from wordcloud import WordCloud, STOPWORDS

                        wordcloud = WordCloud(
                            width=1000,
                            height=int(1000 * (figsize[1] / figsize[0])),
                            background_color="white",
                            stopwords=set(STOPWORDS),
                            max_words=50,
                            max_font_size=75,
                        ).fit_words(word_freqs)

                        plt.imshow(wordcloud, interpolation="bilinear")
                        plt.axis("off")
                    except ModuleNotFoundError as e:
                        utils._log_missing_module("wordcloud", OptionalDependency.TEXT)
                        logger.info(
                            "The text word cloud is not plotted due to missing dependency wordcloud."
                        )

            else:
                ax.text(
                    -1.1,
                    1,
                    text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=props,
                    weight="bold",
                )
                if feature_metadata["missing_percentage"] == 100:
                    addrtext = "NOTE: plot has been disabled as all values in this column are missing."
                else:
                    addrtext = "NOTE: plot has been disabled,\nfor features of unknown type\nno visualization is available"
                ax.text(
                    0.05,
                    0.95,
                    addrtext,
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=props,
                    weight="bold",
                )
                ax.axis("off")

            self.feature_dist_html_dict[col] = convert_to_html(plt)
            plt.close()

            html_widget.value += self.feature_dist_html_dict[col]

    @staticmethod
    def _init_fig_ax(figsize, dpi=288):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.set(facecolor="white")
        return fig, ax

    @staticmethod
    def _format_stats(feature_type_name, stats):
        text = "\n  - %s statistics:" % (feature_type_name)
        for k in list(stats.keys()):
            if "percentage" in k:
                text += "\n    - {}: {:.3f}%".format(k, stats[k])
            elif isinstance(stats[k], (int, np.int64)) or (
                isinstance(stats[k], float)
                and not np.isnan(stats[k])
                and stats[k] == int(stats[k])
            ):
                text += "\n    - {}: {:,}".format(k, int(stats[k]))
            elif isinstance(stats[k], bool):
                text += "\n    - {}: {s}".format(k, "yes" if stats[k] else "no")
            elif isinstance(stats[k], (float, np.float64)):
                text += "\n    - {}: {:.3f}".format(k, stats[k])
            else:
                text += "\n    - {}: {}".format(k, stats[k])

        return text
