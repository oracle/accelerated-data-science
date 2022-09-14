#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents utility functions.

Functions:
    is_boolean(value: Any) -> bool
        Checks if value type is boolean.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from ads.common.card_identifier import card_identify
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.feature_engineering.dataset.zip_code_data import zip_code_dict
from functools import lru_cache
from typing import Any


class SchemeNeutral(str):
    BACKGROUND_LIGHT = "#F5F4F2"
    BACKGROUND_DARK = "#E4E1DD"
    AREA_LIGHT = "#BCB6B1"
    AREA_DARK = "#9E9892"
    LINE_LIGHT = "#665F5B"
    LINE_DARK = "#47423E"


class SchemeTeal(str):
    BACKGROUND_LIGHT = "#F0f6f5"
    BACKGROUND_DARK = "#D6E5E5"
    AREA_LIGHT = "#9ABFBF"
    AREA_DARK = "#76A2A0"
    LINE_LIGHT = "#3E686C"
    LINE_DARK = "#2B484B"


def _tag_to_snake(name: str) -> str:
    """Conversts string to snake representation.
    1. Converts the string to the lower case.
    2. Converts all spaces to underscore.

    Parameters
    ----------
    name: string
        The name to convert.

    Returns
    -------
    str: The name converted to the snake representation.
    """
    _name = name.strip().lower()
    if len(_name.split()) > 1:
        _name = "_".join(_name.split())
    return _name


def _add_missing(x, df):
    """
    Adds count of missing values.
    """
    n_missing = pd.isnull(x.replace(r"", np.NaN)).sum()
    if n_missing > 0:
        df.loc["missing"] = n_missing
    return df


def _count_unique_missing(x):
    """
    Returns the total count, unique count and count of missing values of a series.
    """
    df_stat = pd.Series(
        {"count": len(x), "unique": len(x.replace(r"", np.NaN).dropna().unique())},
        name=x.name,
    ).to_frame()
    return _add_missing(x, df_stat)


def is_boolean(value: Any) -> bool:
    """Checks if value type is boolean.

    Parameters
    ----------
    value: Any
        The value to check.

    Returns
    -------
    bool: True if value is boolean, False otherwise.
    """
    bool_values = ("yes", "y", "true", "t", "1", "no", "n", "false", "f", "0", "")
    return isinstance(value, bool) or str(value).lower() in bool_values


def assign_issuer(cardnumber):
    if pd.isnull(cardnumber):
        return "missing"
    else:
        return card_identify().identify_issue_network(cardnumber)


def random_color_func(
    z,
    word=None,
    font_size=None,
    position=None,
    orientation=None,
    font_path=None,
    random_state=None,
):
    """
    Returns random color function use for color_func in creating WordCloud
    """
    h = 179
    s = 23
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


def _is_float(s: str):
    """
    Checks if given string can convert to float
    """
    return re.match(r"^-?\d+(?:\.\d+)$", s) != None


def _str_lat_long_to_point(s):
    """
    Converts input data into formated geometry point
    Return formated geometry point string or np.NaN if input string is not valid
    """
    if isinstance(s, str):
        coords = s.split(",")
        if len(coords) == 2:
            lat, long = coords[0].lstrip(), coords[1].lstrip()
            if lat.startswith("("):
                lat = lat[1:]
            if long.endswith(")"):
                long = long[:-1]
            if _is_float(lat) and _is_float(long):
                return "POINT(" + long + " " + lat + ")"
    return np.NaN


@runtime_dependency(module="geopandas", install_from=OptionalDependency.GEO)
def _plot_gis_scatter(df: pd.DataFrame, lon: str, lat: str):
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe contains gis data
    lon: str
        column name mapping to longitude
    lat: str
        column name mapping to latitude
    Returns
    -------
        matplotlib.axes._subplots.AxesSubplot present location of given data on world map
    """
    if len(df.index):
        if lon in df.columns and lat in df.columns:
            fig, ax = plt.subplots(facecolor=SchemeNeutral.BACKGROUND_LIGHT)
            gdf = geopandas.GeoDataFrame(
                df, geometry=geopandas.points_from_xy(df[lon], df[lat])
            )
            world = geopandas.read_file(
                geopandas.datasets.get_path("naturalearth_lowres")
            )
            world = world[world["name"] == "United States of America"]
            ax1 = world.plot(
                ax=ax, color=SchemeNeutral.AREA_LIGHT, linewidth=0.5, edgecolor="white"
            )
            gdf.plot(ax=ax1, color=SchemeTeal.LINE_LIGHT, markersize=10)
            plt.axis("off")
            return ax1


def _to_lat_long(x: pd.Series, zipcode: pd.DataFrame):
    """
    Parameters
    ----------
    x: pd.Series
        pandas series contains zip code data
    zipcode : pd.DataFrame
        dataframe contains gis data of zip code
    Returns
    -------
    pd.DataFrame
        dataframe with 2 columes including latitude and longitude data
    """
    lats = []
    longs = []
    for s in x:
        if s in zipcode.index:
            lats.append(np.float64(zipcode.loc[s]["latitude"]))
            longs.append(np.float64(zipcode.loc[s]["longitude"]))
    df = pd.DataFrame(list(zip(lats, longs)), columns=["latitude", "longitude"])
    return df


@runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
def _set_seaborn_theme():
    """
    Sets seaborn figure & axes facecolor
    """
    plt.figure(facecolor=SchemeNeutral.BACKGROUND_LIGHT)
    seaborn.set(
        rc={
            "axes.facecolor": SchemeNeutral.BACKGROUND_LIGHT,
            "figure.facecolor": SchemeNeutral.BACKGROUND_LIGHT,
        }
    )


def _format_stat(stat: pd.Series):
    """
    Formats statistics row index.
    """
    stat.rename(
        {
            "std": "standard deviation",
            "min": "sample minimum",
            "25%": "lower quartile",
            "50%": "median",
            "75%": "upper quartile",
            "max": "sample maximum",
        },
        inplace=True,
    )


@lru_cache(maxsize=1)
def _zip_code():
    """Returns dataframe contain zip code, latitude and longitude data."""
    return pd.DataFrame.from_dict(
        zip_code_dict, orient="index", columns=["longitude", "latitude"]
    )
