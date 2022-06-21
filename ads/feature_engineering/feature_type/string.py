#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a String feature type.

Classes:
    String
        The feature type that represents string values.
"""
import matplotlib.pyplot as plt
import pandas as pd
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.utils import (
    _count_unique_missing,
    random_color_func,
    SchemeNeutral,
)
from ads.feature_engineering import schema
from ads.common import utils, logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


def default_handler(data: pd.Series, *args, **kwargs) -> pd.Series:
    """Processes given data and indicates if the data matches requirements.

    Parameters
    ----------
    data: pd.Series
        The data to process.

    Returns
    -------
    pd.Series: The logical list indicating if the data matches requirements.
    """
    return data.apply(lambda x: isinstance(x, str))


class String(FeatureType):
    """
    Type representing string values.

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
        Shows distributions of datasets using wordcloud.


    Example
    -------
    >>> from ads.feature_engineering.feature_type.string import String
    >>> import pandas as pd
    >>> s = pd.Series(["Hello", "world", None], name='string')
    >>> String.validator.is_string(s)
    0     True
    1     True
    2    False
    Name: string, dtype: bool
    """

    description = "Type representing string values."

    @staticmethod
    def feature_stat(x: pd.Series) -> pd.DataFrame:
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count)
        and missing(count) if there is any.

        Examples
        --------
        >>> string = pd.Series(['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S',
                'S', 'S', 'S', 'Q', 'S', 'S', '', np.NaN, None], name='string')
        >>> string.ads.feature_type = ['string']
        >>> string.ads.feature_stat()
            Metric  Value
        0	count	22
        1	unique	3
        2	missing	3

        Returns
        -------
        Pandas Dataframe
            Summary statistics of the Series or Dataframe provided.
        """
        return _count_unique_missing(x)

    @staticmethod
    @runtime_dependency(module="wordcloud", install_from=OptionalDependency.TEXT)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows distributions of datasets using wordcloud.

        Examples
        --------
        >>> string = pd.Series(['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S',
                'S', 'S', 'S', 'Q', 'S', 'S', '', np.NaN, None], name='string')
        >>> string.ads.feature_type = ['string']
        >>> string.ads.feature_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the String feature type.
        """
        col_name = x.name if x.name else "text"
        df = x.to_frame(col_name)
        df["validation"] = default_handler(x)
        df = df[df["validation"] == True]
        words = " ".join(df[col_name].dropna().to_list())
        if not words:
            return

        from wordcloud import WordCloud

        wc = WordCloud(
            background_color=SchemeNeutral.BACKGROUND_LIGHT,
            color_func=random_color_func,
        ).generate(words)
        _, ax = plt.subplots(facecolor=SchemeNeutral.BACKGROUND_LIGHT)
        ax.imshow(wc)
        plt.axis("off")
        return ax

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> string = pd.Series(['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S',
                'S', 'S', 'S', 'Q', 'S', 'S', '', np.NaN, None], name='string')
        >>> string.ads.feature_type = ['string']
        >>> string.ads.feature_domain()
        constraints: []
        stats:
            count: 22
            missing: 3
            unique: 3
        values: String

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the String feature type.
        """

        return schema.Domain(cls.__name__, cls.feature_stat(x).to_dict()[x.name], [])


String.validator.register("is_string", default_handler)
