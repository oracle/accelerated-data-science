#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a Text feature type.

Classes:
    Text
        The Text feature type.
"""
import matplotlib.pyplot as plt
import pandas as pd
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.utils import random_color_func, SchemeNeutral

from ads.common import utils, logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class Text(String):
    """
    Type representing text values.

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
    feature_plot(x: pd.Series) -> plt.Axes
        Shows distributions of datasets using wordcloud.
    """

    @staticmethod
    @runtime_dependency(module="wordcloud", install_from=OptionalDependency.TEXT)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows distributions of datasets using wordcloud.

        Examples
        --------
        >>> text = pd.Series(['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S',
                'S', 'S', 'S', 'Q', 'S', 'S', '', np.NaN, None], name='text')
        >>> text.ads.feature_type = ['text']
        >>> text.ads.feature_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the Text feature type.
        """
        col_name = x.name if x.name else "text"
        df = x.to_frame(col_name)
        words = df[col_name].dropna().to_list()
        words = " ".join([s for s in words if isinstance(s, str)])
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

    description = "Type representing text values."

    @classmethod
    def feature_domain(cls):
        """
        Returns
        -------
        None
            Nothing.
        """
        return None
