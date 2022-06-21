#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a GIS feature type.

Classes:
    GIS
        The GIS feature type.
"""
import matplotlib.pyplot as plt
import pandas as pd
import re
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.utils import (
    _count_unique_missing,
    _str_lat_long_to_point,
    SchemeNeutral,
    SchemeTeal,
)
from ads.feature_engineering import schema
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)

PATTERN = re.compile(r"^[(]?(\-?\d+\.\d+?),\s*(\-?\d+\.\d+?)[)]?$", re.VERBOSE)


def default_handler(data: pd.Series, *args, **kwargs) -> pd.Series:
    """Processes given data and indicates if the data matches requirements.

    Parameters
    ----------
    data: :class:`pandas.Series`
        The data to process.

    Returns
    -------
    :class:`pandas.Series`
        The logical list indicating if the data matches requirements.
    """
    return data.apply(
        lambda x: True
        if not pd.isnull(x) and PATTERN.match(str(x)) is not None
        else False
    )


class GIS(FeatureType):
    """
    Type representing geographic information.

    Attributes
    ----------
    description: str
        The feature type description.
    name: str
        The feature type name.
    warning: FeatureWarning
        Provides functionality to register warnings and invoke them.
    validator
        Provides functionality to register validators and invoke them.

    Methods
    --------
    feature_stat(x: pd.Series) -> pd.DataFrame
        Generates feature statistics.
    feature_plot(x: pd.Series) -> plt.Axes
        Shows the location of given address on map base on longitude and latitute.

    Example
    -------
    >>> from ads.feature_engineering.feature_type.gis import GIS
    >>> import pandas as pd
    >>> s = pd.Series(["-18.2193965, -93.587285",
                        "-21.0255305, -122.478584",
                        "85.103913, 19.405744",
                        "82.913736, 178.225672",
                        "62.9795085,-66.989705",
                        "54.5604395,95.235090",
                        "24.2811855,-162.380403",
                        "-1.818319,-80.681214",
                        None,
                        "(51.816119, 175.979008)",
                        "(54.3392995,-11.801615)"],
                        name='gis')
    >>> s.ads.feature_type = ['gis']
    >>> GIS.validator.is_gis(s)
    0      True
    1      True
    2      True
    3      True
    4      True
    5      True
    6      True
    7      True
    8     False
    9      True
    10     True
    Name: gis, dtype: bool
    """

    description = "Type representing geographic information."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count) and missing(count).

        Examples
        --------
        >>> gis = pd.Series([
            "69.196241,-125.017615",
            "5.2272595,-143.465712",
            "-33.9855425,-153.445155",
            "43.340610,86.460554",
            "24.2811855,-162.380403",
            "2.7849025,-7.328156",
            "45.033805,157.490179",
            "-1.818319,-80.681214",
            "-44.510428,-169.269477",
            "-56.3344375,-166.407038",
            "",
            np.NaN,
            None
            ],
            name='gis'
        )
        >>> gis.ads.feature_type = ['gis']
        >>> gis.ads.feature_stat()
                gis
        count	13
        unique	10
        missing	3

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series provided.
        """
        return _count_unique_missing(x)

    @staticmethod
    @runtime_dependency(module="geopandas", install_from=OptionalDependency.GEO)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows the location of given address on map base on longitude and latitute.

        Examples
        --------
        >>> gis = pd.Series([
            "69.196241,-125.017615",
            "5.2272595,-143.465712",
            "-33.9855425,-153.445155",
            "43.340610,86.460554",
            "24.2811855,-162.380403",
            "2.7849025,-7.328156",
            "45.033805,157.490179",
            "-1.818319,-80.681214",
            "-44.510428,-169.269477",
            "-56.3344375,-166.407038",
            "",
            np.NaN,
            None
            ],
            name='gis'
        )
        >>> gis.ads.feature_type = ['gis']
        >>> gis.ads.feature_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the GIS feature type.
        """
        col_name = x.name if x.name else "gis"
        df = x.to_frame(name=col_name)
        df["validation"] = default_handler(x)
        df = df[df["validation"] == True]
        df = (
            df[col_name]
            .apply(lambda x: _str_lat_long_to_point(x))
            .to_frame(name="Coordinates")
            .dropna()
        )
        if len(df.index):
            df["Coordinates"] = geopandas.GeoSeries.from_wkt(df["Coordinates"])
            gdf = geopandas.GeoDataFrame(df, geometry="Coordinates")
            fig, ax = plt.subplots(facecolor=SchemeNeutral.BACKGROUND_LIGHT)
            world = geopandas.read_file(
                geopandas.datasets.get_path("naturalearth_lowres")
            )
            world.plot(
                ax=ax, color=SchemeNeutral.AREA_LIGHT, linewidth=0.5, edgecolor="white"
            )
            gdf.plot(ax=ax, color=SchemeTeal.LINE_LIGHT, markersize=10)
            plt.axis("off")
            return ax

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> gis = pd.Series([
            "69.196241,-125.017615",
            "5.2272595,-143.465712",
            "-33.9855425,-153.445155",
            "43.340610,86.460554",
            "24.2811855,-162.380403",
            "2.7849025,-7.328156",
            "45.033805,157.490179",
            "-1.818319,-80.681214",
            "-44.510428,-169.269477",
            "-56.3344375,-166.407038",
            "",
            np.NaN,
            None
            ],
            name='gis'
        )
        >>> gis.ads.feature_type = ['gis']
        >>> gis.ads.feature_domain()
        constraints: []
        stats:
            count: 13
            missing: 3
            unique: 10
        values: GIS

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the GIS feature type.
        """

        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )


GIS.validator.register("is_gis", default_handler)
